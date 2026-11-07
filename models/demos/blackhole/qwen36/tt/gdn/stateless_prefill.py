# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip stateless GDN prefill: right-padded rows from a zero state, no state kept.

Rows are right-padded and the conv and the chunk op are causal, so a padded position never reaches a real one
and the final state is dropped: no valid_len masks are needed. The conv is the fused KDA op (conv + SiLU + q/k/v
split in one program, reading the TILE projection in place); q/k/v reach the chunk op flat, and o comes back
head-major (the same sequence as the tensor-parallel layer in tp.py).
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter
from models.demos.blackhole.qwen36.tt.prefill_matmul import prefill_linear

_PINNED_FUSED_CONFIG = None


def _pinned_fused_config():
    # The cost model picks phased at 16 heads x 128 tokens; pinned fused is ~25% faster, bit-exact.
    global _PINNED_FUSED_CONFIG
    if _PINNED_FUSED_CONFIG is None:
        _PINNED_FUSED_CONFIG = ttnn.ChunkGdnFusedProgramConfig(num_producers=3, num_receivers=2, producer_pool=False)
    return _PINNED_FUSED_CONFIG


_ZERO_STATES = {}


def prebuild_zero_states(device, nv, dk, dv, bps=(1, 2)):
    """Persistent fp32 zero initial states [Bp, Nv, Dk, Dv] for the chunk op (it only reads them): build before any
    trace capture and never free, so the op does not run a zeros fill per layer."""
    for bp in bps:
        _zero_state(device, bp, nv, dk, dv)


def _zero_state(device, bp, nv, dk, dv):
    key = (device.id(), bp)
    z = _ZERO_STATES.get(key)
    if z is None:  # lazy (eager path) for a Bp that was not prebuilt
        z = _ZERO_STATES[key] = ttnn.zeros(
            [bp, nv, dk, dv],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return z


def stateless_prefill_supported(gdn, mode, stateless):
    """True when recurrent_forward may take this path: stateless chunk prefill on one device, fused chunk op,
    and no conv bias (the KDA conv has none)."""
    w = gdn.weights
    return (
        stateless
        and mode == "chunk"
        and gdn.device.get_num_devices() == 1
        and gdn.fused_prefill_fn is not None
        and w.mega_fused_weight is not None
        and w.fused_conv_weight_taps is not None
        and w.fused_conv_bias_dev is None
    )


# Fused gate activations: ttnn.sigmoid's accurate mode (vector mode RC), ttnn.softplus's beta = 1, threshold = 20.
_SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)
_SOFTPLUS = ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)
_A_COL, _B_COL, _Z_COL = 6144, 6176, 6208  # a, b, z columns of the stateless projection (a, b: 16 + 16 zero cols)


def _projection_weight(gdn):
    """[dim, qkv 6144 | a 16 + 16 zero | b 16 + 16 zero | z 2048] copy of the mega weight, built once per layer.

    Whole-tile column moves keep the bfp8 values exact, and the new columns are exact zeros. Built by the warm
    eager forward, before any trace exists."""
    pw = getattr(gdn, "_stateless_projection_weight", None)
    if pw is None:
        w = gdn.weights
        qkv, a_dim, b_dim = w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim
        assert (qkv, a_dim, b_dim) == (_A_COL, 16, 16), (qkv, a_dim, b_dim)
        t = ttnn.to_torch(w.mega_fused_weight)
        t = t.reshape(t.shape[-2], t.shape[-1])
        pad = torch.zeros(t.shape[0], 32 - a_dim, dtype=t.dtype)
        fused = torch.cat(
            [
                t[:, :qkv],
                t[:, qkv : qkv + a_dim],
                pad,
                t[:, qkv + a_dim : qkv + a_dim + b_dim],
                pad,
                t[:, qkv + a_dim + b_dim :],
            ],
            dim=-1,
        ).contiguous()
        pw = gdn._stateless_projection_weight = ttnn.from_torch(
            fused,
            dtype=w.mega_fused_weight.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=gdn.device,
            memory_config=w.mega_fused_weight.memory_config(),
        )
    return pw


def _consts(gdn):
    """uint32 [1] zero (the op's actual_start) and the zero conv history [1, K-1, C] ROW_MAJOR; built once per layer."""
    c = getattr(gdn, "_stateless_consts", None)
    if c is None:
        C = gdn.cfg.q_dim + gdn.cfg.k_dim + gdn.cfg.v_dim
        start = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=gdn.device,
        )
        hist = ttnn.from_torch(
            torch.zeros(1, gdn.conv_kernel_size - 1, C, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=gdn.device,
        )
        c = gdn._stateless_consts = (start, hist)
    return c


_HEADS_PER_PROGRAM = 32  # heads one chunk-op program runs at once (B * Nv above this runs in sequential waves)
_RESET_GATE = 128.0  # added (negated) to g at a row start: exp() underflows to exactly 0 in fp32, so the reset is exact
_CONV_CHANNEL_CHUNK = 32  # one conv work item per 32 channels x 32 tokens; smallest is fastest, bit-exact


def _fold_rows(B, nv):
    """Largest divisor of B with B' * nv <= _HEADS_PER_PROGRAM (at least 1): the rows folded into one sequence."""
    return max(d for d in range(1, B + 1) if B % d == 0 and (d * nv <= _HEADS_PER_PROGRAM or d == 1))


def _reset_mask(gdn, S, nv):
    """fp32 TILE DRAM [1, S, nv]: -_RESET_GATE at s = 0, 0 elsewhere; added to g so a folded row starts from a zero
    state. Cached per (S, nv) on the layer."""
    cache = gdn.__dict__.setdefault("_stateless_reset_masks", {})
    m = cache.get((S, nv))
    if m is None:
        t = torch.zeros(1, S, nv, dtype=torch.float32)
        t[:, 0, :] = -_RESET_GATE
        m = cache[(S, nv)] = ttnn.from_torch(
            t,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=gdn.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return m


def stateless_prefill_forward(gdn, x):
    """x [B, S, dim] bf16 TILE DRAM, S a multiple of 32 -> [B, S, dim]. Weights and compute config as the
    mega-fused branch of gated_deltanet_forward_ttnn."""
    w = gdn.weights
    B, S = x.shape[0], x.shape[1]
    Nk, Dk, Nv, Dv = gdn.num_heads, gdn.head_k_dim, gdn.num_v_heads, gdn.head_v_dim
    q_dim, k_dim, v_dim = gdn.cfg.q_dim, gdn.cfg.k_dim, gdn.cfg.v_dim
    C = q_dim + k_dim + v_dim
    assert S % 32 == 0, f"stateless GDN prefill: S={S} is not a multiple of 32"
    _dram = ttnn.DRAM_MEMORY_CONFIG

    # In-projection: qkv | a | b | z, with a and b each in a tile of their own: all cuts start on a tile edge.
    mega = prefill_linear(x, _projection_weight(gdn), gdn.compute_kernel_config)
    a = mega[:, :, _A_COL : _A_COL + Nv]
    b = mega[:, :, _B_COL : _B_COL + Nv]

    # Conv + SiLU + q/k/v split from a zero history, reading qkv (columns [0, C) of mega) in place. The op takes one
    # [1, T, W] row: the B rows fold into time, and segment_length=S restarts the history at every row start.
    start, hist = _consts(gdn)
    cfg = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_CONV_CHANNEL_CHUNK)
    q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        ttnn.reshape(mega, (1, B * S, mega.shape[-1])),
        hist,
        *w.fused_conv_weight_taps,
        q_dim,
        k_dim,
        v_dim,
        program_config=cfg,
        actual_start=start,
        predecessor_carry=hist,
        memory_config=_dram,
        input_column_offset=0,
        segment_length=S,
    )

    # Gates (fp32, the chunk op's contract): beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias). a and b
    # are cast to fp32 first so no intermediate rounds to bf16 (error ~2e-3 vs ~3e-2 on g); the activations run
    # inside the binary ops.
    a = ttnn.typecast(a, ttnn.float32)
    b = ttnn.typecast(b, ttnn.float32)
    beta = ttnn.multiply(b, 1.0, input_tensor_a_activations=[_SIGMOID], dtype=ttnn.float32, memory_config=_dram)
    ttnn.deallocate(b)
    g = ttnn.multiply(
        ttnn.add(a, w.dt_bias, activations=[_SOFTPLUS], memory_config=_dram),
        w.A_neg,
        dtype=ttnn.float32,
        memory_config=_dram,
    )
    ttnn.deallocate(a)
    beta = ttnn.reshape(beta, (B, S, Nv))
    g = ttnn.reshape(g, (B, S, Nv))

    # Fold F rows into one sequence so the op fills one program of B' * Nv heads; the reset gate at each original
    # row start zeroes the carried state, so the rows stay independent.
    Bp = _fold_rows(B, Nv)
    F = B // Bp
    if F > 1:
        g_old = g
        g = ttnn.add(g_old, _reset_mask(gdn, S, Nv))
        ttnn.deallocate(g_old)
        beta = ttnn.reshape(beta, (Bp, F * S, Nv))
        g = ttnn.reshape(g, (Bp, F * S, Nv))

    q = ttnn.reshape(q, (Bp, F * S, q_dim))
    k = ttnn.reshape(k, (Bp, F * S, k_dim))
    v = ttnn.reshape(v, (Bp, F * S, v_dim))

    # Chunk op: flat q/k/v in, head-major o out; the final state is dropped.
    program_config = _pinned_fused_config() if (Bp * Nv == 16 and F * S == 128) else None
    o, final_state = chunk_gated_delta_rule_fused_adapter(
        q,
        k,
        v,
        beta,
        g,
        scale=Dk**-0.5,
        initial_state=_zero_state(gdn.device, Bp, Nv, Dk, Dv),
        device=gdn.device,
        qkv_head_dims=(Nk, Dk, Nv, Dv),
        return_o_bh=True,
        const_tiles=gdn.fused_prefill_fn.keywords["const_tiles"],
        program_config=program_config,
    )
    ttnn.deallocate(final_state)
    for t in (q, k, v, beta, g):
        ttnn.deallocate(t)

    # Per-head norm, head -> token relayout, SiLU gate on z read in place from mega (bf16 out), out-projection.
    gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        o,
        ttnn.reshape(mega, (Bp, F * S, mega.shape[-1])),
        w.o_norm_weight,
        Nv,
        epsilon=gdn.norm_eps,
        output_dtype=ttnn.bfloat16,
        gate_column_offset=_Z_COL,
        silu_gate=True,
    )
    ttnn.deallocate(o)
    ttnn.deallocate(mega)
    gated = ttnn.reshape(gated, (B, S, Nv * Dv))
    out = prefill_linear(gated, w.o_proj_weight, gdn.compute_kernel_config)
    ttnn.deallocate(gated)
    return out
