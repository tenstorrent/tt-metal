# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel Gated DeltaNet for Qwen3.5.

Recurrence is per value-head (no cross-device comms inside); all-reduce after row-parallel out.
Reuses `recurrent_gated_delta_rule_decode_ttnn`; weights interleaved. GDN norm uses raw weight
(no +1) + SiLU(z) gate — distinct from QK/layer norms.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce


def gdn_fused_decode_enabled():
    """QWEN36_GDN_FUSED_DECODE (default "1"): single-user (max batch 1) GDN decode through the fused kernels
    (qkv_causal_conv1d_silu + chunk_gated_delta_rule + sigmoid_gated_rms_norm). "0" keeps the original
    shift-register / recurrent_gated_delta_rule_decode_ttnn path byte-for-byte."""
    return os.environ.get("QWEN36_GDN_FUSED_DECODE", "1") != "0"


_fused_decode_logged = False


def gdn_decode_step_op_enabled():
    """QWEN36_GDN_DECODE_STEP_OP (default on): decode through ttnn.experimental.kda.gdn_decode_step (conv + gating +
    recurrence + gated norm in one op, packed conv history). 0 reverts to the fused-batched / original paths."""
    return os.environ.get("QWEN36_GDN_DECODE_STEP_OP", "1") == "1"


def pack_head_tiles(rows, Nv, Nk, Dk, Dv, parity=0, both_parities=False):
    """rows: list of 4 torch [C] vectors ([q | k | v] channels of one device) -> [Nv, 4, 32, 32] bf16 packed head
    tiles for gdn_decode_step: tile (h, s) holds rows[s]'s channels [q_hk | k_hk | v_h] (hk = h // rf) as 32-wide
    chunks, chunk c in row 2c + parity (user b's history uses parity b & 1; taps are stored on both parities)."""
    rf = Nv // Nk
    kd = Nk * Dk
    out = torch.zeros(Nv, 4, 32, 32, dtype=torch.bfloat16)
    for h in range(Nv):
        hk = h // rf
        for j, r in enumerate(rows):
            r = r.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    r[hk * Dk : (hk + 1) * Dk],
                    r[kd + hk * Dk : kd + (hk + 1) * Dk],
                    r[2 * kd + h * Dv : 2 * kd + (h + 1) * Dv],
                ]
            ).reshape(-1, 32)
            n = chunks.shape[0]
            for par in (0, 1) if both_parities else (parity,):
                out[h, j, par : 2 * n + par : 2, :] = chunks
    return out


def _log_fused_decode_once(msg):
    """One info line per process (the layer is built 48x)."""
    global _fused_decode_logged
    if not _fused_decode_logged:
        _fused_decode_logged = True
        logger.info(msg)


def _shard_small_fp32(torch_tensor, mesh, cache_path, dim=-1):
    """tpc.shard_small without the bf16 round-trip: per-head fp32 tensor -> fp32 TILE sharded on `dim`."""
    t = torch_tensor.float()
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 2:
        t = t.unsqueeze(0)
    return ttnn.as_tensor(
        t,
        dtype=ttnn.float32,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config, dtype=None):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs).
    dtype: optional output dtype (bf16 for the column-parallel prefill out-proj; default = x's)."""
    s = ttnn.silu(z, memory_config=memory_config)
    if dtype is None:
        return ttnn.multiply(x, s, memory_config=memory_config)
    return ttnn.multiply(x, s, memory_config=memory_config, dtype=dtype)


def kda_channel_chunk_size(channels, cap=512):
    """channel_chunk_size for qkv_causal_conv1d_silu: the largest tile-aligned divisor of `channels` not above
    `cap`. 512 at the 27B TP-4 width (2560 -> 5 blocks x 64 tile rows = 320 work items over the grid), the
    configuration the op was measured at."""
    for c in range(min(cap, channels) - min(cap, channels) % 32, 0, -32):
        if channels % c == 0:
            return c
    raise ValueError(f"no tile-aligned channel chunk divides {channels}")


_EXACT_SELECT_CFG = None


def exact_select_cfg():
    """HiFi4 + fp32 accumulation compute config for the one-hot row-select matmuls (bf16 values x 0/1 -> the
    selected row is reproduced bit-exactly; lower fidelities may drop mantissa bits of the value operand)."""
    global _EXACT_SELECT_CFG
    if _EXACT_SELECT_CFG is None:
        _EXACT_SELECT_CFG = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return _EXACT_SELECT_CFG


def kda_conv_prefill(
    qkv,
    T,
    history,
    taps,
    widths,
    actual_start,
    xin_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    conv_sel=None,
    emit_state=True,
):
    """Depthwise causal conv (K=4) + SiLU + q/k/v split in ONE program (ttnn.experimental.kda.qkv_causal_conv1d_silu).

    qkv:          [1, T, C] bf16 TILE, the projection's conv columns (q | k | v).
    history:      [1, 3, C] bf16, the three rows preceding this chunk (zeros from scratch). TILE or ROW_MAJOR;
                  the op reads ROW_MAJOR, so a TILE carry is converted here (three rows).
    taps:         four [1, 1, C] bf16 TILE tensors in kernel-position order, tap j multiplying row t-3+j
                  (tw["conv_taps"], exactly the op's tap0..tap3 contract).
    widths:       (q_width, k_width, v_width), tile-aligned, summing to C.
    actual_start: uint32 [1] device tensor holding 0, allocated once before any trace capture.
    conv_sel:     optional [1, 3, T] bf16 TILE one-hot row selector (traced masked bucket: valid_len is a persistent
                  device input, so the "last three real rows" cannot be a static slice). new_state is then
                  matmul(conv_sel, qkv) = rows valid_len-3..valid_len-1 of the conv input (all-zero rows where
                  valid_len-3+j < 0, exact for a from-scratch history of zeros). None: the static last-3-rows slice.
    Returns q [1,T,Q], k [1,T,K], v [1,T,V] bf16 TILE DRAM, and new_state [1, 3, C] bf16 TILE DRAM: the chunk's
    last three INPUT rows, i.e. the next chunk's history (the op does not emit it).
    emit_state=False skips new_state (returned as None), for callers that keep their own carry (the spec verify).
    """
    C = qkv.shape[-1]
    n_hist = history.shape[1]  # K - 1 = 3 rows, the op's fixed history depth
    assert qkv.shape[1] == T, f"kda_conv_prefill: T={T} but qkv has {qkv.shape[1]} rows"
    _dram = ttnn.DRAM_MEMORY_CONFIG
    # The op's input contract is row-major: one untilize of the conv columns.
    xin = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=xin_memory_config)
    hist = history if history.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(history, ttnn.ROW_MAJOR_LAYOUT)
    q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        xin,
        hist,
        *taps,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C)),
        actual_start=actual_start,
        # No sequence parallelism on the TP mesh (sequence_parallel_axis=0 on a 1xN mesh): the op validates
        # predecessor_carry against history and never reads it, so alias history as its docstring says.
        predecessor_carry=hist,
        memory_config=_dram,
    )
    if hist is not history:
        ttnn.deallocate(hist)
    if not emit_state:
        ttnn.deallocate(xin)
        return q, k, v, None
    # Next chunk's history: the last three input rows, taken from the row-major copy (page reads, no
    # untilize), then re-tiled because the layer-wide carry (conv_carry, decode seeding, per-user assembly)
    # is TILE.
    if conv_sel is not None:
        assert conv_sel.shape[1] == n_hist and conv_sel.shape[-1] == T, (conv_sel.shape, n_hist, T)
        ttnn.deallocate(xin)
        # bf16 one-hot x bf16 values from the TILE projection output (same values the RM copy holds), HiFi4 + fp32
        # accumulate: exactly one nonzero term per output element -> bit-exact row selection.
        # DRAM operand: an L1-resident qkv could clash with the matmul's static CBs (the op sees a 3-row LHS).
        qkv_d = qkv if qkv.memory_config().buffer_type == ttnn.BufferType.DRAM else ttnn.to_memory_config(qkv, _dram)
        new_state = ttnn.matmul(
            conv_sel, qkv_d, memory_config=_dram, compute_kernel_config=exact_select_cfg(), dtype=ttnn.bfloat16
        )
        if qkv_d is not qkv:
            ttnn.deallocate(qkv_d)
        return q, k, v, new_state
    new_state = ttnn.slice(xin, (0, T - n_hist, 0), (1, T, C), memory_config=_dram)
    ttnn.deallocate(xin)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
    return q, k, v, new_state


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's linear_attn.* weights across the mesh."""
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp
    # In-proj weight dtype (QWEN36_BFP4_GDN_IN, opt-in): BFP4. Out-proj stays BFP8.
    _in_dtype = ttnn.bfloat4_b if tpc.bfp4_gdn_in_enabled() else ttnn.bfloat8_b

    if cache_dir is not None:
        import os

        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    # State-dict keys vary by loader: optional linear_attn. prefix; conv1d may be fused or q/k/v split.
    P = "linear_attn." if any(k.startswith("linear_attn.") for k in sd) else ""

    def first_key(*names):
        for n in names:
            if (P + n) in sd:
                return sd[P + n]
        raise KeyError(f"none of {[P + n for n in names]} found in GDN state dict")

    # Fused QKV+Z (column-parallel)
    qkv_w = first_key("in_proj_qkv.weight", "qkv_proj.weight")
    if (P + "conv1d.weight") in sd:
        conv1d_w = sd[P + "conv1d.weight"]
    else:  # bf16 remap: reassemble fused conv1d from q/k/v streams
        conv1d_w = torch.cat([sd[P + "q_conv.weight"], sd[P + "k_conv.weight"], sd[P + "v_conv.weight"]], dim=0)
    qkv_re = tpc.prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp)
    z_w = sd[P + "in_proj_z.weight"]
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    tw = {}
    # Column-parallel qkvz (DRAM-sharded decode matmul when enabled); distinct .dramshard cache
    qkvz_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
    # Fold a/b into qkvz → one matmul outputs [qkv|z|a|b] (default when DRAM-sharded)
    fuse_ab = qkvz_sharded
    if fuse_ab:
        fused = torch.cat(
            [
                torch.cat(
                    [
                        qkv_re[d * qkv_per : (d + 1) * qkv_per],
                        z_w[d * z_per : (d + 1) * z_per],
                        a_w[d * nv_per : (d + 1) * nv_per],
                        b_w[d * nv_per : (d + 1) * nv_per],
                    ],
                    dim=0,
                )
                for d in range(tp)
            ],
            dim=0,
        )
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.gdn_qkvzab_weight_memcfg,
            cache_path=c("qkvzab" + (".il" if _proj1d else ".dramshard")),
            dtype=_in_dtype,
        )
    else:
        fused = torch.cat(
            [
                torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        qkvz_mc = args.gdn_qkvz_weight_memcfg if qkvz_sharded else ttnn.DRAM_MEMORY_CONFIG
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=qkvz_mc,
            cache_path=c("qkvz" + (".dramshard" if qkvz_sharded else "")),
            dtype=_in_dtype,
        )
        # Separate A+B projection (column-parallel fallback)
        ab = torch.cat(
            [
                torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        tw["ab"] = tpc.shard_w(
            ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=_in_dtype
        )
    # Row-parallel out projection: DRAM-width-sharded (like the in-proj) — decode tput win.
    _out_sharded = getattr(args, "gdn_out_weight_memcfg", None) is not None
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.gdn_out_weight_memcfg if _out_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out.dramshard" if _out_sharded else "out"),
        dtype=ttnn.bfloat8_b,
    )
    if getattr(args, "num_devices", 1) > 1:
        # COLUMN-parallel copy of the out-proj for prefill.
        # Decode keeps the row-sharded tw["out"] (matmul + all-reduce).
        tw["out_colpar"] = tpc.shard_w(
            sd[P + "out_proj.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("out.colpar"),
            dtype=ttnn.bfloat8_b,
        )
    # Per-head params
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # Conv taps (4), sharded per Q/K/V head grouping
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    if gdn_fused_decode_enabled() and args.gdn_conv_kernel_size == 4:
        # Fused single-user decode (QWEN36_GDN_FUSED_DECODE): the same constants the validated fused decoder
        # uses. dt_bias / -exp(A_log) stay fp32 (tw["dt_bias"] / tw["neg_exp_A"] are bf16), and the
        # sigmoid_gated_rms_norm op wants the norm weight as a rank-1 [Dv] bf16 TILE tensor (tw["norm_w"] is [1,1,Dv]).
        tw["dt_bias_fp32"] = _shard_small_fp32(sd[P + "dt_bias"], mesh, c("dt_bias_fp32"))
        tw["neg_exp_A_fp32"] = _shard_small_fp32(-torch.exp(sd[P + "A_log"].float()), mesh, c("neg_exp_A_fp32"))
        tw["norm_w_1d"] = ttnn.from_torch(
            sd[P + "norm.weight"].reshape(-1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if gdn_decode_step_op_enabled() and hasattr(ttnn.experimental, "kda"):
            # gdn_decode_step conv taps: per device the 4 taps packed into [Nv_tp, 4, 32, 32] head tiles (both
            # parities), sharded on dim 0 -> each device holds its own [Nv_tp, 4, 32, 32].
            nv_tp, nk_tp = nv // tp, nk // tp
            C_tp = 2 * nk_tp * dk + nv_tp * dv
            per_dev = torch.cat(
                [
                    pack_head_tiles(
                        [taps[j][d * C_tp : (d + 1) * C_tp] for j in range(4)], nv_tp, nk_tp, dk, dv, both_parities=True
                    )
                    for d in range(tp)
                ],
                dim=0,
            )
            tw["conv_taps_packed"] = ttnn.from_torch(
                per_dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    return tw


class TPGatedDeltaNet:
    """Standalone TP GDN decode (per-device value-head recurrence + all-reduce)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        # DRAM-shard the row-parallel out projection (decode tput win; matches loader gate).
        self._out_sharded = getattr(self.args, "gdn_out_weight_memcfg", None) is not None
        self.B = args.max_batch_size
        self.Nk = args.gdn_nk_tp
        self.Nv = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        # Flat q/k/v into adapter (skips prefill head-split reshapes)
        self._gdn_flat_qkv = True
        # Fuse adapter output relayout with rms_norm + head-flatten
        self._gdn_fuse_out = True
        self.gdn_program_config = getattr(args, "gdn_program_config", None)
        # WY-inverse arithmetic (ttnn.ChunkGdnWyInverse); None = the op's AUTO (the forward-substitution solve on Blackhole).
        self.gdn_wy_inverse = getattr(args, "gdn_wy_inverse", None)
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_HIFI2
        # In-proj decode matmul only (prefill keeps self.cfg = HiFi2)
        self.cfg_in = tpc.COMPUTE_LOFI if tpc.bfp4_gdn_in_enabled() else self.cfg
        # Must match load_gdn_weights_tp gates
        self._dram_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
        self._fuse_ab = self._dram_sharded
        # Fuse prefill norm-allgather + qkvzab in-proj into all_gather_minimal_matmul_async.
        # Requires the folded qkvzab weight; norm's post-AG is disabled in layer.py (GDN, prefill).
        self._fuse_agmm = self._fuse_ab
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1
        # PREFILL out-proj as column-parallel AG+matmul (takes precedence over the MMRS arm when the
        # col-sharded weight was loaded).
        self._out_colpar_prefill = "out_colpar" in tw
        # Pre-build chunk masks once (trace-safe; avoids from_torch inside captured trace)
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        # Prefill fused-op constant tiles, owned by this layer (avoids process-lifetime C++ cache vs device lifetime).
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

        self._fused_const_tiles = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)
        self.conv_states = None
        self.rec_state = None
        # Spec-decode hybrid verify slot capture (set by SpeculativeDecoder): per-token recurrent-state
        # snapshots buffered during a captured verify so commit is a slot-select, not a re-run.
        self._capture_slots = False
        self._verify_slots = None
        self._verify_states = None  # per-token rec states from the last verify (token-major)
        self._verify_states_buf = None  # same tensor; kept so traced replays can re-arm the handle
        # Pre-allocated persistent slot buffers (rec_state + conv_states shaped). Verify copies state
        # INTO these fixed addresses (trace-safe + no per-call alloc) instead of fresh ttnn.clone.
        self._slot_bufs = None
        # Fully-batched hybrid verify (_verify_fullbatch): no per-token loop at all. This is the
        # production spec-decode verify — 2.3 ms/candidate vs ~15 ms for the per-token path — and is
        # validated lossless + deterministic (test_spec_lossless.py, test_spec_determinism.py). The
        # per-token loop below survives as this flag's False branch: an A/B reference for numerics
        # work, flipped from python (no env), never from the demo.
        self.use_fullbatch_verify = True
        # Batched-conv verify only: the [1, K-1+T, qkv_dim_tp] conv window stashed with ONE copy, from
        # which commit_verify_slot slices the accepted slot's shift-register. None => per-token slots.
        self._verify_win_buf = None
        # Batched-conv verify only: the DURABLE shift register as one [1, K, qkv_dim_tp] tensor,
        # mirroring conv_states[0..K-1]. The traced verify reads its carry from a constant-offset
        # slice of this, and commit_verify_slot writes it with ONE slice+copy instead of touching K
        # separate conv_states taps — see commit_verify_slot's COST NOTE.
        self._conv_win_buf = None
        # Which of the two mirrors is authoritative. The traced fullbatch verify advances ONLY
        # _conv_win_buf (refilling the K conv_states taps inside the trace cost ~10 ms/iteration over
        # 48 layers and nothing in the spec loop reads them), so after a verify the taps are BEHIND:
        # _conv_taps_stale. Conversely everything that writes the taps from outside (prefill
        # capture_state, reset, slot edits, decode's own shift register) leaves the window behind:
        # _conv_win_stale. Exactly one can be set at a time — each setter clears the other. The
        # rebuilds are sync_conv_taps() (window -> taps, at every tap CONSUMER) and sync_conv_win()
        # (taps -> window, lazily in _ensure_conv_win, which only ever runs eagerly).
        self._conv_taps_stale = False
        self._conv_win_stale = False
        self._win_captured = False  # did THIS verify populate the window? (buffer is always allocated)
        self._conv_taps_T = None  # conv taps expanded to T rows for the batched-conv path
        # Spec decode only (set by SpeculativeDecoder): run the ONE fused recurrent device op in
        # forward_decode instead of the composite, so decode and hybrid verify share GDN math.
        # Full-batch (B == self.B) only — it has no bucketed B<Bmax state slice/writeback.
        self.use_fused_recurrent_decode = False
        self.conv_carry = None  # cross-chunk prefill conv carry [1, K-1, qkv_dim_tp]
        # Which causal conv runs the single-sequence prefill when valid_len is None (masked buckets always
        # keep the MAC FIR, whose one-hot new_state selection they need):
        #   QWEN_GDN_CONV=kda  (default) ttnn.experimental.kda.qkv_causal_conv1d_silu: conv + SiLU + q/k/v
        #                      split + tilize in ONE program (kda_conv_prefill);
        #   QWEN_GDN_CONV=fir  the shifted multiply-accumulate FIR everywhere.
        self._conv_impl = os.environ.get("QWEN_GDN_CONV", "kda")
        if self._conv_impl not in {"kda", "fir"}:
            raise ValueError(f"QWEN_GDN_CONV must be 'kda' or 'fir', got {self._conv_impl!r}")
        # The KDA op is fixed at four taps.
        self._gdn_kda_conv = self._conv_impl == "kda" and self.K == 4
        # KDA conv constants, allocated once by _ensure_kda_consts (host writes, so before any trace capture):
        # the op's actual_start scalar and the row-major zero history of a from-scratch chunk.
        self._kda_actual_start = None
        self._kda_zero_history = None
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        # Persistent init buffers the SHORT prefill trace copies into rec_state/conv_carry (zeros, or a carried/restored
        # state for a resumed prefill; see model._prepare_short_trace_gdn_init). Allocated before any trace capture.
        self._prefill_init_rec = None
        self._prefill_init_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill
        # Fused decode (QWEN36_GDN_FUSED_DECODE, default on). Decided here, once, so the state format is fixed for
        # the model's lifetime. max_batch_size == 1: the single-user path (_forward_decode_fused, active width 1);
        # max_batch_size > 1: the batched path (_fused_batched, every width 1..Bmax); otherwise the original path.
        #   _conv_hist_rm: the fused decode's conv history, [Bmax, K-1, qkv_dim_tp] ROW_MAJOR bf16 (Bmax == 1: the
        #                  single-user [1, K-1, C]; row u = user u in batched mode), the three raw
        #                  conv inputs preceding the next token, oldest first (the layout/order the kda conv op
        #                  reads). Allocated once (reset_state / first write) and only ever updated IN PLACE
        #                  (ttnn.copy), so the decode and prefill traces keep a fixed address. Decode maintains
        #                  it INSTEAD of conv_states; every code path that writes conv_states for decode
        #                  (prefill capture_state, assemble_batched_state, write_slot, forward_prefill_batched)
        #                  also refreshes it. It is deliberately NOT part of the model's per-binding state swaps
        #                  (model.py rebinds rec_state/conv_states/conv_carry on prefill scratch): with
        #                  max_batch_size == 1 there is one user, so the latest prefill IS the decode state.
        self._conv_hist_rm = None
        # Batched fused decode (max_batch_size 2..N): the same kernels over per-user 32-row blocks, see
        # _forward_decode_fused_batched. self.B is swapped by the model's scratch bindings (B=1 prefill scratch,
        # B=bg group scratch), so the decode capacity is remembered here: _conv_hist_rm is [_decode_B, K-1, C]
        # and only the decode binding (self.B == _decode_B) owns it. _scan_max_b: the widest decode width that
        # runs the fused chunk_gated_delta_rule (one scan core per value head per user); wider widths use the
        # kda conv + the original recurrent kernel on the same rec_state.
        self._decode_B = args.max_batch_size
        self._scan_max_b = self._fused_scan_max_width()
        self._nea_mask = {}  # per-width [1, 32*B, Nv] fp32 -exp(A_log) at each user's live row, 0 elsewhere
        self._fused_decode = self._fused_decode_supported(args, tw)
        self._fused_batched = self._fused_decode and self._decode_B > 1
        # gdn_decode_step decode op (QWEN36_GDN_DECODE_STEP_OP): conv + gating + recurrence + gated norm in ONE op over
        # a packed conv history [Bmax, Nv, 4, 32, 32] (_conv_hist_packed, updated in place by the op). The RM
        # history stays the canonical format of every non-decode writer; conversions are on-device embedding gathers.
        self._conv_hist_packed = None
        self._decode_op = self._decode_step_op_supported(args, tw)
        # Batched mode keeps TWO conv-state formats, one per decode width class (see prepare_decode_width):
        #   hist   _conv_hist_rm [Bmax, K-1, C] RM: read/written by the fused path (width <= SCAN_MAX)
        #   states conv_states[1..K-1] [1, Bmax, C] TILE: read/written by the original path (width > SCAN_MAX)
        # (conv_states[j] row u == hist[u, j-1]: after a shift-register step st[j]<-st[j+1], st[K-1]<-new, so
        # st[1..K-1] are the last K-1 raw inputs, oldest first; st[0] is overwritten by the shift before it is read.)
        # _conv_fmt names which formats are currently valid for the DECODE binding: "both" / "hist" / "states".
        # Whole-batch writers set "both"; partial writers (write_slot, remap_slots) write both formats for their rows
        # and leave it unchanged; prepare_decode_width(B) syncs if needed and sets it to the class it is about to run.
        # "packed" (decode_op only): _conv_hist_packed is the authoritative history (the op advances it); RM hist and
        # conv_states are stale. Every non-decode reader/writer first converts packed -> RM (_unpack_conv_hist).
        self._conv_fmt = "both"

    def _decode_step_op_supported(self, args, tw):
        if not gdn_decode_step_op_enabled() or not self._fused_batched:
            return False
        reasons = []
        if not (1 < self._decode_B <= 32):
            reasons.append(f"max_batch_size {self._decode_B} not in [2, 32]")
        if self.K != 4 or self.Dk != self.Dv or self.Nv % self.Nk or 2 * self.Nv > 32 or self.Nv > 16:
            reasons.append("unsupported GDN head geometry")
        if self.qkvz_dim_tp != self.qkv_dim_tp + self.Nv * self.Dv:
            reasons.append("qkvz_dim != qkv_dim + Nv*Dv")
        if not hasattr(ttnn.experimental.kda, "gdn_decode_step"):
            reasons.append("ttnn.experimental.kda.gdn_decode_step unavailable")
        if not (self._fuse_ab and getattr(args, "proj_1d_decode", False)):
            reasons.append("fused [qkv|z|a|b] 1D decode projection unavailable")
        if "conv_taps_packed" not in tw:
            reasons.append("conv_taps_packed missing from tw")
        if reasons:
            logger.info("[GDN] gdn_decode_step op NOT used: " + "; ".join(reasons))
            return False
        logger.info("[GDN] decode path: gdn_decode_step op (QWEN36_GDN_DECODE_STEP_OP=0 reverts)")
        return True

    def _fused_scan_max_width(self):
        """Widest decode width (users) taking the fused chunk_gated_delta_rule: grid cores // Nv_tp (9 on a 11x10
        grid at Nv_tp=12), capped at the 32-row placement limit. QWEN36_GDN_FUSED_SCAN_MAXB overrides."""
        env = os.environ.get("QWEN36_GDN_FUSED_SCAN_MAXB")
        if env is not None:
            return max(0, min(32, int(env)))
        try:
            grid = self.mesh.compute_with_storage_grid_size()
            return max(1, min(32, (grid.x * grid.y) // max(1, self.Nv)))
        except Exception:  # noqa: BLE001 - mesh without a grid query (unit-test doubles)
            return 9

    def _fused_decode_supported(self, args, tw):
        """Whether this layer takes the fused single-user decode path (QWEN36_GDN_FUSED_DECODE). Logs one info line
        per process stating the active path."""
        reasons = []
        if not gdn_fused_decode_enabled():
            reasons.append("QWEN36_GDN_FUSED_DECODE=0")
        else:
            if self.K != 4:
                reasons.append(f"conv kernel size {self.K} != 4 (the kda conv op is fixed at four taps)")
            if self.Dk != self.Dv or self.Dv % tpc.TILE_SIZE or self.key_dim_tp % tpc.TILE_SIZE:
                reasons.append(f"head dims Dk={self.Dk}/Dv={self.Dv} unsupported by the flat chunk_gated_delta_rule")
            if os.environ.get("QWEN35_GDN_STATE_BF16") == "1" or os.environ.get("QWEN35_GDN_DECODE_BF16") == "1":
                reasons.append("QWEN35_GDN_STATE_BF16/QWEN35_GDN_DECODE_BF16 request the bf16 recurrence")
            if any(k not in tw for k in ("dt_bias_fp32", "neg_exp_A_fp32", "norm_w_1d")):
                reasons.append("fused-decode weights missing from tw (built only when the flag is on at load time)")
            if not hasattr(ttnn.experimental, "kda") or not hasattr(ttnn.experimental.kda, "sigmoid_gated_rms_norm"):
                reasons.append("ttnn.experimental.kda.sigmoid_gated_rms_norm unavailable in this ttnn build")
        if reasons:
            _log_fused_decode_once(
                "[GDN] decode path: ORIGINAL (shift-register conv + recurrent kernel): " + "; ".join(reasons)
            )
            return False
        if args.max_batch_size == 1:
            _log_fused_decode_once(
                "[GDN] decode path: FUSED for max_batch_size == 1 (qkv_causal_conv1d_silu + chunk_gated_delta_rule + "
                "sigmoid_gated_rms_norm; QWEN36_GDN_FUSED_DECODE=0 reverts). conv_states is not maintained during "
                "decode; out-proj input is bf16."
            )
        else:
            _log_fused_decode_once(
                f"[GDN] decode path: FUSED-BATCHED for max_batch_size={args.max_batch_size} (widths <= "
                f"SCAN_MAX={self._scan_max_b}: qkv_causal_conv1d_silu over per-user 32-row blocks + "
                "chunk_gated_delta_rule + sigmoid_gated_rms_norm on the RM conv history; wider widths run the ORIGINAL "
                "decode on conv_states; prepare_gdn_decode_width(B) syncs the two conv formats on a width-class "
                "change; QWEN36_GDN_FUSED_DECODE=0 reverts, QWEN36_GDN_FUSED_SCAN_MAXB overrides SCAN_MAX)."
            )
        return True

    def reset_state(self):
        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 recurrent state by default (QWEN35_GDN_STATE_BF16=1 reverts)
        if os.environ.get("QWEN35_GDN_STATE_BF16") != "1":
            self.rec_state = ttnn.from_torch(
                torch.zeros(self.B, self.Nv, self.Dk, self.Dv, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        else:
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))
        # Cross-chunk conv carry + persistent zero sources (created before any trace)
        self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        self._zero_conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_rec = z((self.B, self.Nv, self.Dk, self.Dv))
        if self._gdn_kda_conv or self._fused_decode:
            self._ensure_kda_consts()
        if self._fused_decode and self._owns_conv_hist():
            # Fused-decode conv history: allocated once, zeroed IN PLACE on every later reset (it may already be
            # baked into a decode/prefill trace, and reset_state also runs for the B=1 prefill scratch). In
            # batched mode only the decode binding (self.B == _decode_B) owns it: the B=1 / group scratch bindings
            # must not touch the decode users' history.
            _had_hist = self._conv_hist_rm is not None
            self._ensure_conv_hist()
            if _had_hist:
                ttnn.copy(self._hist_zero_source(), self._conv_hist_rm)
            self._conv_fmt = "both"  # zeros in both formats
            if self._decode_op:
                self._ensure_conv_hist_packed()  # allocated once here (host write), before any trace capture
                self._pack_tables()
            if self._fused_batched and not self._decode_op:
                # Persistent width constants (placement / selection matrices, zero blocks, masked -exp(A_log)),
                # allocated here (host writes) so no decode width allocates inside a trace capture.
                self._prepare_batched_widths()
        # Chunk-outer batched-prefill conv left-context (allocated lazily by forward_prefill_batched).
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
        self._batched_conv_carry = None
        # rec_state/conv_states got fresh addresses here, so any verify slot buffers cloned from the
        # old ones are stale — drop them (re-allocated lazily on the next captured verify).
        if self._slot_bufs is not None:
            for rec, convs in self._slot_bufs:
                ttnn.deallocate(rec)
                for c in convs:
                    ttnn.deallocate(c)
            self._slot_bufs = None
        if self._conv_win_buf is not None:  # mirrors the now-stale conv_states; re-seeded at capture
            ttnn.deallocate(self._conv_win_buf)
            self._conv_win_buf = None
        # Fresh (zero) taps and no window: neither mirror is behind.
        self._conv_taps_stale = False
        self._conv_win_stale = False

    def reset_state_inplace(self):
        """Zero conv + recurrent state in place (preserves trace buffer addresses).

        Copies from preallocated _zero_* buffers only — never allocates during an active trace.
        """
        # Drop any chunk-outer batched-prefill conv left-context so the next sequence starts clean.
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
            self._batched_conv_carry = None
        if self.conv_states is None:
            self.reset_state()
            return
        # Zero sources must exist (reset_state runs first; no lazy alloc during trace)
        assert (
            self._zero_conv0 is not None and self._zero_conv_carry is not None and self._zero_rec is not None
        ), "zero sources missing; reset_state must run before reset_state_inplace"
        for cs in self.conv_states:
            ttnn.copy(self._zero_conv0, cs)
        ttnn.copy(self._zero_rec, self.rec_state)
        # Zero cross-chunk conv carry for new sequence
        ttnn.copy(self._zero_conv_carry, self.conv_carry)
        if self._fused_decode and self._conv_hist_rm is not None and self._owns_conv_hist():
            ttnn.copy(self._hist_zero_source(), self._conv_hist_rm)  # preallocated RM zeros (no allocation here)
            self._conv_fmt = "both"  # conv_states zeroed above too
        # Taps are now the truth (zeros); the window mirror still holds the previous sequence's.
        self._conv_taps_stale = False
        self._conv_win_stale = True

    def _col_proj(self, x, weight, decode_progcfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_cfg=None):
        """Column-parallel qkvz projection; DRAM-sharded decode matmul when enabled.
        out_memory_config: decode result placement (default DRAM; L1 keeps it resident).
        compute_cfg: optional compute config (default self.cfg)."""
        cfg = self.cfg if compute_cfg is None else compute_cfg
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=cfg, memory_config=out_memory_config)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
            decode_out_memory_config=out_memory_config,
        )

    def _ensure_kda_consts(self):
        """Allocate the KDA conv path's constant tensors once. Host writes: reset_state calls this (the model
        runs it before capturing its traces); the lazy call in _kda_conv_prefill covers eager callers."""
        rep_map = ttnn.ReplicateTensorToMesh(self.mesh)
        if self._kda_actual_start is None:
            self._kda_actual_start = ttnn.from_torch(
                torch.tensor([0], dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep_map,
            )
        if self._kda_zero_history is None:
            self._kda_zero_history = ttnn.from_torch(
                torch.zeros(1, self.K - 1, self.qkv_dim_tp, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep_map,
            )

    def _kda_conv_prefill(self, qkv, T, conv_state, conv_sel=None):
        """kda_conv_prefill on this layer's taps and widths. conv_state: the previous chunk's carry
        [1, K-1, C] TILE, or None from scratch. conv_sel: optional persistent one-hot [1, K-1, T] selector of the last
        real rows (traced masked bucket, from-scratch only). Returns (q, k, v, new_state), see kda_conv_prefill."""
        self._ensure_kda_consts()
        history = conv_state if conv_state is not None else self._kda_zero_history
        kd, vd = self.key_dim_tp, self.value_dim_tp
        return kda_conv_prefill(
            qkv, T, history, self.tw["conv_taps"], (kd, kd, vd), self._kda_actual_start, conv_sel=conv_sel
        )

    # ------------------------------------------------------------------ #
    # Fused single-user decode: conv history (_conv_hist_rm) + its handoff from prefill.
    # ------------------------------------------------------------------ #
    def _owns_conv_hist(self):
        """Whether the CURRENT binding is the one that owns _conv_hist_rm: the decode binding. A max_batch_size == 1
        model has a single binding (B == 1 always); in batched mode the B=1 prefill scratch and the B=bg group
        scratch (model.py swaps self.B) must not allocate, zero or write the [_decode_B, K-1, C] history."""
        return self.B == self._decode_B

    def _hist_zero_source(self):
        """Persistent zero source for an in-place history reset (preallocated: no allocation under trace)."""
        if self._decode_B == 1:
            return self._kda_zero_history
        return self._shared_const(("zero_hist", self._decode_B, self.qkv_dim_tp), self._make_zero_hist)

    def _make_zero_hist(self):
        return self._rm_zeros((self._decode_B, self.K - 1, self.qkv_dim_tp))

    def _ensure_conv_hist(self):
        """The persistent [_decode_B, K-1, qkv_dim_tp] ROW_MAJOR bf16 conv history (row u = user u's K-1 previous raw
        conv inputs, oldest first; _decode_B == 1 is the single-user [1, K-1, C] buffer), allocated (zeros) on first
        use. Host write: reset_state calls it before any trace capture; the lazy callers run eagerly."""
        self._ensure_kda_consts()
        if self._conv_hist_rm is None:
            self._conv_hist_rm = self._rm_zeros((self._decode_B, self.K - 1, self.qkv_dim_tp))
        return self._conv_hist_rm

    def _rm_zeros(self, shape):
        """Replicated ROW_MAJOR bf16 DRAM zeros (host write)."""
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _hist_hook_active(self):
        """Writers of the decode conv state refresh the fused history only on the DECODE binding: B == 1 for a
        max_batch_size == 1 model (as before), B == _decode_B for a batched model, so the B=1 prefill scratch
        bindings never write a [1, K-1, C] row into the [_decode_B, K-1, C] history (write_slot does the per-slot
        write instead)."""
        return self._fused_decode and (
            (self._decode_B == 1 and self.B == 1) or (self._decode_B > 1 and self.B == self._decode_B)
        )

    def _write_conv_hist(self, last_inputs):
        """Handoff prefill -> fused decode (max_batch_size == 1): copy the last K-1 conv inputs [1, K-1, C] TILE
        (oldest first, the same tensor that seeds conv_states[1..K-1]) into the RM history, in place.
        Device-only (trace-safe once the history exists)."""
        hist = self._ensure_conv_hist()
        rm = ttnn.to_layout(last_inputs, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(rm, hist)
        ttnn.deallocate(rm)

    def _rows_to_rm(self, tiles):
        """list of [1, 1, C] TILE conv rows -> one [1, len, C] RM bf16 tensor (consumes nothing)."""
        rows = []
        for t in tiles:
            if t.dtype != ttnn.bfloat16:
                t = ttnn.typecast(t, ttnn.bfloat16)
            rows.append(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        stacked = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for r in rows:
            if r is not stacked:
                ttnn.deallocate(r)
        return stacked

    def _sync_conv_hist_from_states(self):
        """Rebuild the RM history from conv_states[1..K-1].

        max_batch_size == 1: each [1, 1, C] TILE -> the [1, K-1, C] history. Batched: each [1, B, C] TILE (row u =
        user u) -> the full [B, K-1, C] history, a FULL replace (every row), for the writers that produce
        conv_states for the whole batch (batched prefill, group assembly)."""
        hist = self._ensure_conv_hist()
        if self._decode_B == 1:
            stacked = self._rows_to_rm([self.conv_states[m] for m in range(1, self.K)])  # [1, K-1, C]
        else:
            B, C = self.B, self.qkv_dim_tp
            per_tap = []
            for m in range(1, self.K):  # [1, B, C] TILE -> [B, 1, C] RM
                t = self.conv_states[m]
                if t.dtype != ttnn.bfloat16:
                    t = ttnn.typecast(t, ttnn.bfloat16)
                rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                per_tap.append(ttnn.reshape(rm, (B, 1, C)))
            stacked = ttnn.concat(per_tap, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, K-1, C]
            for r in per_tap:
                ttnn.deallocate(r)
        ttnn.copy(stacked, hist)
        ttnn.deallocate(stacked)

    def sync_fused_conv_hist_from_states(self):
        """Public full-replace sync of the fused conv history from conv_states (for model.py writers that write
        dn.conv_states directly, e.g. _assemble_groups_gdn_dev). No-op unless this is the decode binding. Both
        formats are valid afterwards."""
        if self._hist_hook_active():
            self._sync_conv_hist_from_states()
            self._conv_fmt = "both"

    def _sync_conv_states_from_hist(self):
        """hist -> states: conv_states[j] (j = 1..K-1, FULL Bmax rows) <- hist[:, j-1, :] as [1, Bmax, C] TILE, in
        place into the persistent tensors (conv_states[0] is overwritten by the shift before it is ever read).
        Eager; temporaries are freed immediately."""
        hist = self._ensure_conv_hist()
        B, C = self._decode_B, self.qkv_dim_tp
        for j in range(1, self.K):
            tap = self._slice_along(hist, 1, j - 1, j)  # [B, 1, C] RM
            row = ttnn.reshape(tap, (1, B, C))
            tile = ttnn.to_layout(row, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            src = tile
            if tile.dtype != self.conv_states[j].dtype:
                src = ttnn.typecast(tile, self.conv_states[j].dtype)
            ttnn.copy(src, self.conv_states[j])
            if src is not tile:
                ttnn.deallocate(src)
            ttnn.deallocate(tile)
            ttnn.deallocate(row)  # view of tap: frees tap's buffer

    # ------------------------------------------------------------------ #
    # gdn_decode_step op: packed conv history + on-device conversions to / from the RM history.
    # ------------------------------------------------------------------ #
    def _ensure_conv_hist_packed(self):
        """The persistent [_decode_B, Nv, 4, 32, 32] bf16 TILE packed history the op updates in place (zeros on first
        use; host write -> allocate before any trace capture)."""
        if self._conv_hist_packed is None:
            self._conv_hist_packed = ttnn.from_torch(
                torch.zeros(self._decode_B, self.Nv, 4, 32, 32, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._conv_hist_packed

    def _pack_tables(self):
        """Per-model uint32 index tables (+ a zero row) for the RM <-> packed embedding gathers. Both are slot-position
        aware: packed row (b, h, s, r) with s = 1..3 and r = 2c + (b & 1) holds chunk c of [q_hk | k_hk | v_h] of RM
        history row (b, s - 1); s = 0 / the other parity / r >= 2 * n_chunks are zero. (packed row 2c + (b&1) of tile
        (b, h, s) is the only place the op reads history from.)"""
        key = ("gdn_pack_tables", self._decode_B, self.Nv, self.Nk, self.Dk, self.Dv, self.K)

        def make():
            Bm, Nv, Nk, Dk, Dv, K = self._decode_B, self.Nv, self.Nk, self.Dk, self.Dv, self.K
            rf = Nv // Nk
            qn, vn = Dk // 32, Dv // 32  # 32-channel chunks per q/k head, per v head
            cpr = (2 * Nk * Dk + Nv * Dv) // 32  # RM chunk rows per (b, history row)
            nck = 2 * qn + vn  # chunks per head tile
            n_rm = Bm * (K - 1) * cpr
            zero = n_rm  # index of the appended zero row
            kd32 = Nk * Dk // 32

            def rm_chunk(h, c):
                hk = h // rf
                if c < qn:
                    return hk * qn + c
                if c < 2 * qn:
                    return kd32 + hk * qn + (c - qn)
                return 2 * kd32 + h * vn + (c - 2 * qn)

            to_packed = torch.full((Bm, Nv, 4, 32), zero, dtype=torch.int64)
            for b in range(Bm):
                for h in range(Nv):
                    for sl in range(1, 4):
                        for c in range(nck):
                            to_packed[b, h, sl, 2 * c + (b & 1)] = (b * (K - 1) + sl - 1) * cpr + rm_chunk(h, c)
            to_rm = torch.zeros((Bm, K - 1, cpr), dtype=torch.int64)
            for b in range(Bm):
                for j in range(K - 1):
                    for ch in range(cpr):
                        if ch < kd32:  # q
                            h, c = (ch // qn) * rf, ch % qn
                        elif ch < 2 * kd32:  # k
                            h, c = ((ch - kd32) // qn) * rf, qn + (ch - kd32) % qn
                        else:  # v
                            h, c = (ch - 2 * kd32) // vn, 2 * qn + (ch - 2 * kd32) % vn
                        to_rm[b, j, ch] = ((b * Nv + h) * 4 + j + 1) * 32 + 2 * c + (b & 1)

            def dev(t, dtype, layout):
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=layout,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            # Per-parity single-slot gather tables: packed rows (h, s, r) of ONE slot <- rows of a [(K-1)*cpr + 1, 32]
            # RM view of that user's [1, K-1, C] history (+ the appended zero row). Same entries as to_packed[b].
            slot_tabs = []
            for par in (0, 1):
                t = torch.full((Nv, 4, 32), (K - 1) * cpr, dtype=torch.int64)
                for h in range(Nv):
                    for sl in range(1, 4):
                        for c in range(nck):
                            t[h, sl, 2 * c + par] = (sl - 1) * cpr + rm_chunk(h, c)
                slot_tabs.append(dev(t.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))

            return {
                "slot_to_packed": slot_tabs,
                "to_packed": dev(to_packed.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "to_rm": dev(to_rm.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "zero_row": dev(torch.zeros(1, 32, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                "cpr": cpr,
            }

        return self._shared_const(key, make)

    def _pack_conv_hist(self):
        """RM history -> packed history (in place into _conv_hist_packed): one embedding gather. Eager only (it must
        not be captured into a decode trace: the op owns the packed history while the trace replays)."""
        tab = self._pack_tables()
        Bm, K, cpr = self._decode_B, self.K, tab["cpr"]
        packed = self._ensure_conv_hist_packed()
        w = ttnn.reshape(self._ensure_conv_hist(), (Bm * (K - 1) * cpr, 32))
        w = ttnn.concat([w, tab["zero_row"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(tab["to_packed"], w, layout=ttnn.TILE_LAYOUT)  # [1, Bm*Nv*4*32, 32] -> tiles in order
        ttnn.deallocate(w)
        out = ttnn.reshape(out, (Bm, self.Nv, 4, 32, 32))
        ttnn.copy(out, packed)
        ttnn.deallocate(out)

    def _unpack_conv_hist(self):
        """packed history -> RM history (in place into _conv_hist_rm): one embedding gather. Eager only."""
        tab = self._pack_tables()
        Bm, K, C = self._decode_B, self.K, self.qkv_dim_tp
        hist = self._ensure_conv_hist()
        rm = ttnn.to_layout(self._conv_hist_packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.reshape(rm, (Bm * self.Nv * 4 * 32, 32))
        out = ttnn.embedding(tab["to_rm"], w, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, Bm*(K-1)*C/32, 32]
        out = ttnn.reshape(out, (Bm, K - 1, C))
        ttnn.copy(out, hist)
        ttnn.deallocate(out)
        ttnn.deallocate(rm)

    def _write_conv_hist_slot_packed(self, slot, convs):
        """write_slot under the "packed" format: write ONLY slot `slot`'s [1, Nv, 4, 32, 32] tiles of the packed history
        straight from the user's conv taps (convs[1..K-1], [1, 1, C] TILE), bit-identical to what _pack_conv_hist
        produces for that slot (rows 2c + (slot & 1) of tiles s = 1..3; zeros elsewhere). One embedding gather with
        the per-parity table + one in-place slice_write; the other slots' tiles are untouched. Eager only."""
        tab = self._pack_tables()
        packed = self._ensure_conv_hist_packed()
        rows = self._rows_to_rm([convs[m] for m in range(1, self.K)])  # [1, K-1, C] RM bf16
        w = ttnn.reshape(rows, ((self.K - 1) * tab["cpr"], 32))
        w = ttnn.concat([w, tab["zero_row"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(tab["slot_to_packed"][slot & 1], w, layout=ttnn.TILE_LAYOUT)  # [1, Nv*4*32, 32]
        ttnn.deallocate(w)
        out = ttnn.reshape(out, (1, self.Nv, 4, 32, 32))
        ttnn.experimental.slice_write(out, packed, [slot, 0, 0, 0, 0], [slot + 1, self.Nv, 4, 32, 32], [1, 1, 1, 1, 1])
        ttnn.deallocate(out)

    def _remap_packed(self, idx):
        """remap_slots under "packed": ONE gather per layer over the packed tensor viewed as [Bm*Nv*4*32, 32] RM rows.
        Dest row (bd, h, s, r) <- src row (idx[bd], h, s, r') with r' = r - (bd & 1) + (idx[bd] & 1) for the valid
        history rows (s = 1..3, r & 1 == bd & 1, r < 2 * n_chunks), zero for everything else: exactly what
        unpack -> RM gather -> repack yields. Eager only (host index upload once per remap, shared by all layers)."""
        tab = self._pack_tables()
        Bm, Nv = self._decode_B, self.Nv
        key = tuple(idx)
        holder = self.tt_ccl if self.tt_ccl is not None else self
        cache = holder.__dict__.setdefault("_qwen36_gdn_remap_cache", {})
        # ONE persistent device table (allocated at warmup, before any trace capture), refreshed IN PLACE per distinct
        # remap: a table allocated lazily after the traces were captured can land on memory a trace replay overwrites.
        if cache.get("key") != key:
            nck = 2 * (self.Dk // 32) + self.Dv // 32
            zero = Bm * Nv * 4 * 32
            r = torch.arange(32)
            bd = torch.arange(Bm)
            bs = torch.tensor(idx, dtype=torch.int64)
            par_d, par_s = bd & 1, bs & 1
            ok = (r[None, :] < 2 * nck) & ((r[None, :] & 1) == par_d[:, None])  # [Bm, 32]
            src_r = r[None, :] - par_d[:, None] + par_s[:, None]  # [Bm, 32]
            h = torch.arange(Nv)
            sl = torch.arange(4)
            src = ((bs[:, None, None, None] * Nv + h[None, :, None, None]) * 4 + sl[None, None, :, None]) * 32 + src_r[
                :, None, None, :
            ]
            m = ok[:, None, None, :] & (sl[None, None, :, None] >= 1)
            tbl = torch.where(m, src, torch.full_like(src, zero))
            host = ttnn.from_torch(
                tbl.reshape(1, -1).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            if "dev" not in cache:
                cache["dev"] = ttnn.from_torch(
                    tbl.reshape(1, -1).to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                ttnn.copy_host_to_device_tensor(host, cache["dev"])
            cache["key"] = key
        packed = self._ensure_conv_hist_packed()
        rm = ttnn.to_layout(packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.reshape(rm, (Bm * Nv * 4 * 32, 32))
        w = ttnn.concat([w, tab["zero_row"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(cache["dev"], w, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(w)
        out = ttnn.reshape(out, (Bm, Nv, 4, 32, 32))
        ttnn.copy(out, packed)
        ttnn.deallocate(out)
        ttnn.deallocate(rm)

    def warmup_enter_packed(self):
        """Warmup only (before any trace capture): make the packed history the authoritative format so the packed slot
        write / packed remap programs (embedding -> TILE, 5D slice_write, ...) get compiled and their device-side
        program/kernel buffers allocated NOW instead of lazily after the traces were captured. Returns the previous
        conv format tag for warmup_exit_packed, or None when this layer has no packed path."""
        if not (self._hist_hook_active() and self._decode_B > 1 and self._needed_conv_fmt(self._decode_B) == "packed"):
            return None
        prev = self._conv_fmt
        self._pack_conv_hist()
        self._conv_fmt = "packed"
        return prev

    def warmup_exit_packed(self, prev):
        if prev is None:
            return
        self._unpack_conv_hist()
        self._conv_fmt = "both" if prev == "both" else "hist"

    def _ensure_rm_hist_valid(self):
        """Before any non-decode reader/writer of the RM history / conv_states: when the packed history is the
        authoritative one, convert it back to RM (conv_states stays stale) and tag the format "hist"."""
        if self._conv_fmt == "packed":
            self._unpack_conv_hist()
            self._conv_fmt = "hist"

    def _needed_conv_fmt(self, B):
        if self._decode_op and not self.use_fused_recurrent_decode and B <= 32:
            return "packed"
        return "hist" if B <= self._scan_max_b else "states"

    def _require_conv_fmt(self, need):
        """forward_decode guard (Python only, runs at eager steps / trace capture, not at replay): the format this
        width class reads must be valid. "both" (a writer ran, no prepare_decode_width yet) narrows to `need`,
        since this step makes the other format stale; a valid-other-only state needs the sync hook."""
        if self._conv_fmt == need:
            return
        if need == "packed":
            if self._conv_fmt == "both":  # eager first step after a whole-batch writer: RM -> packed here
                self._pack_conv_hist()
                self._conv_fmt = "packed"
                return
        else:
            self._ensure_rm_hist_valid()  # leaving the packed format: RM becomes the valid one ("hist")
            if self._conv_fmt == need:
                return
        if self._conv_fmt == "both":
            self._conv_fmt = need
            return
        raise RuntimeError(
            f"GDN conv state format is '{self._conv_fmt}' but this decode width needs '{need}': call "
            "prepare_gdn_decode_width(B) / prepare_decode_width(B) before decoding at a new width class"
        )

    def prepare_decode_width(self, B):
        """Make the conv state of the width class of `B` valid before decoding at width B (call once per width
        change, eagerly, outside any trace capture/replay). Returns True when a sync ran. No-op unless the batched
        fused mode is on. hist -> states and states -> hist are in-place full syncs of persistent buffers."""
        if not self._fused_batched or self.B != self._decode_B:
            return False
        need = self._needed_conv_fmt(B)
        synced = False
        if need == "packed":
            if self._conv_fmt != "packed":
                if self._conv_fmt == "states":
                    self._sync_conv_hist_from_states()
                self._pack_conv_hist()  # RM ("both" / "hist" / just synced from states) -> packed
                synced = True
            self._conv_fmt = "packed"
            return synced
        if self._conv_fmt == "packed":
            self._unpack_conv_hist()
            self._conv_fmt = "hist"
            synced = True
        if self._conv_fmt not in (need, "both"):
            if need == "hist":
                self._sync_conv_hist_from_states()
            else:
                self._sync_conv_states_from_hist()
            synced = True
        # The coming decode steps update only `need`; the other format goes stale.
        self._conv_fmt = need
        return synced

    def _write_conv_hist_users(self, per_user_last):
        """assemble_batched_state, batched mode: per_user_last[u] = [1, K-1, C] TILE last conv inputs of user u ->
        FULL replace of the [B, K-1, C] history (row u = user u), in place."""
        hist = self._ensure_conv_hist()
        rows = [
            ttnn.to_layout(
                c if c.dtype == ttnn.bfloat16 else ttnn.typecast(c, ttnn.bfloat16),
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for c in per_user_last
        ]
        stacked = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(stacked, hist)
        for r in rows:
            if r is not stacked:
                ttnn.deallocate(r)
        ttnn.deallocate(stacked)

    def _write_conv_hist_slot(self, slot, convs):
        """write_slot, batched mode: write ONLY row `slot` of the history from the user's conv taps (convs[m] =
        [1, 1, C] TILE, m = 1..K-1 are the K-1 previous inputs oldest first). Must run BEFORE the caller consumes
        convs. In-place slice_write of the [1, K-1, C] row; the other (live) users' rows are untouched."""
        hist = self._ensure_conv_hist()
        row = self._rows_to_rm([convs[m] for m in range(1, self.K)])  # [1, K-1, C] RM
        ttnn.experimental.slice_write(row, hist, [slot, 0, 0], [slot + 1, self.K - 1, self.qkv_dim_tp], [1, 1, 1])
        ttnn.deallocate(row)

    # ------------------------------------------------------------------ #
    # Batched fused decode: per-width persistent constants.
    # ------------------------------------------------------------------ #
    def _shared_const(self, key, make):
        """Per-MODEL (not per-layer) constant cache: 48 GDN layers share one set of placement / selection matrices
        and zero blocks. Lives on the shared tt_ccl object (else args, else this layer), so it dies with the model
        instead of outliving the device in a process-level cache. `make` runs on first use (host write)."""
        for holder in (self.tt_ccl, self.args, self):
            if holder is None:
                continue
            try:
                cache = holder.__dict__.setdefault("_qwen36_gdn_batched_consts", {})
            except AttributeError:
                continue
            if key not in cache:
                cache[key] = make()
            return cache[key]
        return make()

    def _replicated(self, t, dtype, layout):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _batched_consts(self, B):
        """Width-B constants (shared across layers), built once (host writes: before any trace capture).

        Every user owns a 32-row block (the chunk size): rows 0..K-2 = its conv history, row K-1 = the live token,
        the rest zero. The live row is _FUSED_LIVE_ROW = K-1 = 3 in every block.
          P    [1, 32B, 32] bf16: placement, P[32u+3, u] = 1. P @ t places row u of a zero-padded [1, 32, N] tensor at
               the live row of block u and zeros elsewhere (exact copy: one nonzero bf16 product, fp32 accumulate).
          Q    [1, B, 32B] bf16: selection, Q[u, 32u+3] = 1. Q @ t compacts the live rows of a [1, 32B, N] tensor.
          ind  [1, 32B, Nv] fp32: 1 at every live row (folded into -exp(A_log) per layer, see _nea_mask_for).
          zeros [B, 32-K, C] bf16 RM: the zero tail of each block of the conv input.
        All four exist only for widths the fused path handles (B <= _scan_max_b); wider widths run the original decode.
        """
        live, blk = self.K - 1, 32

        def make():
            c = {}
            if B <= self._scan_max_b:
                P = torch.zeros(1, blk * B, blk, dtype=torch.bfloat16)
                ind = torch.zeros(1, blk * B, self.Nv, dtype=torch.float32)
                for u in range(B):
                    P[0, blk * u + live, u] = 1.0
                    ind[0, blk * u + live, :] = 1.0
                c["P"] = self._replicated(P, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                c["ind"] = self._replicated(ind, ttnn.float32, ttnn.TILE_LAYOUT)
                Q = torch.zeros(1, B, blk * B, dtype=torch.bfloat16)
                for u in range(B):
                    Q[0, u, blk * u + live] = 1.0
                c["Q"] = self._replicated(Q, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                c["zeros"] = self._rm_zeros((B, blk - self.K, self.qkv_dim_tp))
            return c

        return self._shared_const(("widths", B, self.K, self.Nv, self.qkv_dim_tp, self._scan_max_b), make)

    def _nea_mask_for(self, B):
        """This layer's [1, 32B, Nv] fp32 -exp(A_log) at each user's live row and exactly 0 at every other row, so
        g = nea_mask * softplus(a_placed + dt_bias) is exactly 0 on the padded steps (identity recurrence steps).
        Built once per width (device op: eager, before any trace capture)."""
        if B not in self._nea_mask:
            self._nea_mask[B] = ttnn.mul(
                self._batched_consts(B)["ind"], self.tw["neg_exp_A_fp32"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        return self._nea_mask[B]

    def _decode_widths(self):
        """Every width a bucketed decode can run: powers of two below the capacity, and the capacity itself."""
        widths, w = [], 1
        while w < self._decode_B:
            widths.append(w)
            w *= 2
        widths.append(self._decode_B)
        return widths

    def _prepare_batched_widths(self):
        """Eagerly build the constants of every decode width (idempotent) so that no width allocates inside a trace
        capture. Other widths (non-bucket callers) are built lazily on first use, which is only safe eagerly."""
        for w in self._decode_widths():
            if w <= self._scan_max_b:
                self._batched_consts(w)
                self._nea_mask_for(w)

    def _matmul_cfg(self):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _place_live_rows(self, t, B, dtype):
        """[1, B, N] TILE (bf16) -> [1, 32B, N] TILE `dtype` DRAM: row u lands on the live row of user u's 32-row
        block, every other row is exactly zero. CONSUMES t.

        t is first padded to the 32-row tile height with ttnn.pad, which (as in the B=1 path) returns a VIEW of
        the same buffer and zero-fills the implicit tile padding in place, so the matmul's K padding is zero (a
        NaN/Inf in stale tile padding would otherwise poison the one-hot product). The view aliases t: exactly ONE
        of the two handles is deallocated (after the matmul), never both. HiFi4 + fp32 accumulate: the product of a
        bf16 value with 1.0 (plus exact zeros) is an exact copy; fp32 out is exact for a bf16 source."""
        tp = t
        same = True
        if B < 32:
            tp = ttnn.pad(t, [(0, 0), (0, 32 - B), (0, 0)], value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Decide aliasing while BOTH handles are alive (a view shares t's buffer).
            same = (tp is t) or (tp.buffer_address() == t.buffer_address())
        out = ttnn.matmul(
            self._batched_consts(B)["P"],
            tp,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._matmul_cfg(),
        )
        ttnn.deallocate(tp)  # frees the shared buffer when tp is a view of t (or is t)
        if not same:
            ttnn.deallocate(t)  # pad returned a fresh buffer (not a view): free the source too
        return out

    def _select_live_rows(self, t, B, dtype, memory_config):
        """[1, 32B, N] TILE -> [1, B, N]: row u = live row of user u's block (exact one-hot select). CONSUMES t."""
        out = ttnn.matmul(
            self._batched_consts(B)["Q"],
            t,
            dtype=dtype,
            memory_config=memory_config,
            compute_kernel_config=self._matmul_cfg(),
        )
        ttnn.deallocate(t)
        return out

    def _fused_conv_batched(self, qkv, B):
        """Causal conv1d + SiLU + q/k/v split for width B on the persistent unified history, ONE kda program.

        qkv [1, B, C] TILE (consumed). Builds the RM input joined[u] = [hist[u] (K-1 rows) | this token | zeros] =
        32 rows per user, flattened to [1, 32B, C], and runs the op with a ZERO history: the live token of every user
        sees its own K-1 previous inputs as ordinary preceding rows, and the rows of one user's block never feed
        another user's live row. Returns q [1, 32B, kd], k, v [1, 32B, vd] bf16 TILE DRAM (live row = 3 of each 32-row
        block) and updates the history IN PLACE (rows [0:B]): joined[:, 1:K] = [hist[1:], this token]. Trace-safe:
        no persistent allocation, the history address is fixed."""
        _DRAM = ttnn.DRAM_MEMORY_CONFIG
        C, kd, vd = self.qkv_dim_tp, self.key_dim_tp, self.value_dim_tp
        Kh = self.K - 1
        hist = self._conv_hist_rm
        assert hist is not None, "fused batched decode needs the conv history (reset_state on the decode binding)"
        zeros = self._batched_consts(B)["zeros"]
        if qkv.dtype != ttnn.bfloat16:
            qkv = ttnn.typecast(qkv, ttnn.bfloat16)
        rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)  # [1, B, C] RM
        ttnn.deallocate(qkv)
        row = ttnn.reshape(rm, (B, 1, C))  # RM view: user u's token as its own row
        hist_b = hist if B == self._decode_B else ttnn.slice(hist, (0, 0, 0), (B, Kh, C), memory_config=_DRAM)
        joined = ttnn.concat([hist_b, row, zeros], dim=1, memory_config=_DRAM)  # [B, 32, C] RM
        if hist_b is not hist:
            ttnn.deallocate(hist_b)
        ttnn.deallocate(row)  # frees rm (view)
        flat = ttnn.reshape(joined, (1, 32 * B, C))  # RM view
        zero_hist = self._kda_zero_history  # [1, K-1, C]
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            flat,
            zero_hist,
            *self.tw["conv_taps"],
            kd,
            kd,
            vd,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C, cap=256)),
            actual_start=self._kda_actual_start,
            predecessor_carry=zero_hist,  # no sequence parallelism: alias history (see kda_conv_prefill)
            memory_config=_DRAM,
        )
        # Next history = the K-1 newest raw inputs of each user = block rows 1..K-1 (hist[1:], this token).
        new_hist = ttnn.slice(joined, (0, 1, 0), (B, self.K, C), memory_config=_DRAM)  # [B, K-1, C] RM
        ttnn.deallocate(joined)  # frees flat (view)
        if B == self._decode_B:
            ttnn.copy(new_hist, hist)
        else:
            # Prefix write: rows [0:B] only, idle users' history untouched. RM interleaved slice_write, in place.
            ttnn.experimental.slice_write(new_hist, hist, [0, 0, 0], [B, Kh, C], [1, 1, 1])
        ttnn.deallocate(new_hist)
        return q, k, v

    def _forward_decode_fused_batched(self, x, decode_ar=None):
        """Fused decode at width B (1 <= B <= _scan_max_b) of a max_batch_size > 1 model (state rows [0:B]).

        Same kernels and math as _forward_decode_fused, run over B users at once: every user owns a 32-row block
        with its live token at row K-1 = 3 (see _batched_consts); g / beta / z are placed there with exact one-hot
        matmuls and are ZERO on every other row (g = beta = 0 -> identity recurrence steps), so the single chunk_
        gated_delta_rule over [B, 32] tokens updates each user's rec_state row by exactly one real step. The gated
        norm output is compacted back to [1, B, value_dim] with a one-hot select and goes through the unchanged
        out-proj + all-reduce. rec_state is [Bmax, Nv, Dk, Dv] fp32 in the SAME format as the original path (rows
        [0:B] updated in place), so widths can be switched between steps. conv_states is not maintained; the conv
        history is _conv_hist_rm. out-proj input is bf16 (as the B=1 fused path)."""
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE

        tw, Nv = self.tw, self.Nv
        _L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        kd, vd = self.key_dim_tp, self.value_dim_tp
        Bmax = self.B
        B = x.shape[-2]
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        assert 1 <= B <= self._scan_max_b and Bmax == self._decode_B

        qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)

        # ---- z / beta / g placed on the live row of each user's 32-row block, zero elsewhere ----
        if z.dtype != ttnn.bfloat16:  # sigmoid_gated_rms_norm's gate contract
            z = ttnn.typecast(z, ttnn.bfloat16, memory_config=_L1)
        z_p = ttnn.reshape(self._place_live_rows(z, B, ttnn.bfloat16), (B, _FUSED_CHUNK_SIZE, vd))
        beta = ttnn.sigmoid(b, memory_config=_L1)  # sigmoid BEFORE placement: padded rows must stay exactly 0
        ttnn.deallocate(b)
        beta_p = ttnn.reshape(self._place_live_rows(beta, B, ttnn.bfloat16), (B, _FUSED_CHUNK_SIZE, Nv))
        if a.dtype != ttnn.bfloat16:
            a = ttnn.typecast(a, ttnn.bfloat16, memory_config=_L1)
        a_p = self._place_live_rows(a, B, ttnn.float32)  # fp32 [1, 32B, Nv]: exact copy of the bf16 `a`
        a_dt = ttnn.add(a_p, tw["dt_bias_fp32"], memory_config=_DRAM)
        ttnn.deallocate(a_p)
        # nea_mask is -exp(A_log) at live rows and 0 elsewhere: g is exactly 0 on every padded row.
        g = ttnn.mul(
            self._nea_mask_for(B),
            a_dt,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            memory_config=_DRAM,
        )
        ttnn.deallocate(a_dt)
        g_p = ttnn.reshape(g, (B, _FUSED_CHUNK_SIZE, Nv))

        # ---- causal conv1d + SiLU + q/k/v split over the per-user blocks ----
        q, k, v = self._fused_conv_batched(qkv, B)
        q = ttnn.reshape(q, (B, _FUSED_CHUNK_SIZE, kd))
        k = ttnn.reshape(k, (B, _FUSED_CHUNK_SIZE, kd))
        v = ttnn.reshape(v, (B, _FUSED_CHUNK_SIZE, vd))

        # ---- gated delta rule over [B, 32] tokens: fp32 state, rows [0:B] in place ----
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        eye, tril, ones, masks = self._fused_const_tiles
        o, new_rec = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g_p,
            beta_p,
            scale=self.scale,
            initial_state=init_state,
            output_final_state=True,
            chunk_size=_FUSED_CHUNK_SIZE,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        if init_state is not self.rec_state:
            ttnn.deallocate(init_state)
        for t in (q, k, v, g_p, beta_p):
            ttnn.deallocate(t)
        if B == Bmax:
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self._write_recurrent_state_prefix(new_rec, B)

        # ---- gated RMSNorm (no +1) * silu(z) over the blocks, then compact the live rows ----
        normed = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o, z_p, tw["norm_w_1d"], Nv, epsilon=1e-6, output_dtype=ttnn.bfloat16
        )  # [B, 32, vd]
        ttnn.deallocate(o)
        gated = ttnn.mul(normed, z_p, memory_config=_DRAM)
        ttnn.deallocate(normed)
        ttnn.deallocate(z_p)
        gated = self._select_live_rows(ttnn.reshape(gated, (1, _FUSED_CHUNK_SIZE * B, vd)), B, ttnn.bfloat16, _DRAM)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here (as the B=1 fused path)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_decode_step_op(self, x, decode_ar=None):
        """Decode at width B (1 <= B <= _decode_B <= 32; state rows / slots [0:B]) through ONE op,
        ttnn.experimental.kda.gdn_decode_step: causal conv (packed history, updated in place) + SiLU + q/k/v split +
        gating (beta, g) + recurrence on rec_state rows [0:B] (in place) + gated RMSNorm * silu(z). Then the unchanged
        out-proj + all-reduce. Needs the "packed" conv format (see _require_conv_fmt)."""
        tw = self.tw
        B = x.shape[-2]
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        assert 1 <= B <= self._decode_B == self.B
        # the fused [qkv|z|a|b] decode projection (1D matmul) straight into L1; the op consumes the whole row.
        qkvzab = tpc.matmul_1d_decode(
            x,
            tw["qkvz"],
            self.args.gdn_qkvz_decode_1d_progcfg,
            self.cfg_in,
            out_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        qkvzab = ttnn.reshape(qkvzab, (1, B, qkvzab.shape[-1]))
        gated = ttnn.experimental.kda.gdn_decode_step(
            qkvzab,
            tw["dt_bias_fp32"],
            tw["neg_exp_A_fp32"],
            self.rec_state,
            tw["norm_w_1d"],
            self.Nv,
            self.Nk,
            self.Dk,
            self.Dv,
            scale=self.scale,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=ttnn.bfloat16,
            conv_hist=self._conv_hist_packed,
            conv_taps=tw["conv_taps_packed"],
            qkvz_dim=self.qkvz_dim_tp,
            fast_mode=True,
        )  # [1, B, Nv*Dv] bf16
        ttnn.deallocate(qkvzab)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here (as the fused paths)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def snapshot_fused_decode_state(self):
        """Host copy of the fused-decode conv history (None when the fused path is off). conv_states is NOT the
        decode conv state when the fused path is on, so a caller that snapshots/restores GDN state around a
        throwaway decode run (demo trace capture) must also snapshot/restore this: pair with
        restore_fused_decode_state. Mirrors the demos' rec_state/conv_states host snapshot."""
        if not self._fused_decode or self._conv_hist_rm is None:
            return None
        if self._conv_fmt == "packed":
            self._unpack_conv_hist()  # RM becomes current too (fmt stays "packed")
        return ttnn.to_torch(self._conv_hist_rm, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))

    def restore_fused_decode_state(self, snap):
        """In-place restore (fixed address, trace-safe) of a snapshot_fused_decode_state result."""
        if snap is None:
            return
        src = ttnn.from_torch(
            snap,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(src, self._conv_hist_rm)
        ttnn.deallocate(src)
        self._conv_fmt = "hist"  # conv_states is not part of the snapshot

    def _row_proj(self, x, weight):
        """Row-parallel out projection: DRAM-sharded decode/prefill matmul (K=gdn_value_dim_tp),
        matching the in-proj. Falls back to plain interleaved on single device (no sharded memcfg)."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x, weight, self.args.gdn_out_decode_1d_progcfg, self.cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if not self._out_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                # Prefill non-fused arm (single device, or out-sharded): tuned 2D config vs ttnn-auto.
                # fp32 [seq,dim] output too big for L1 (42MB) -> DRAM out; separate tt_all_reduce does the RS.
                # max_cols = device width (11 on BH): wide grid (~10-wide), fp32-neutral.
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x, weight, compute_kernel_config=self.cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            self.args.gdn_out_progcfg,
            self.args.act_shard_gdn_value,
            self.args.prefill_progcfg,
            self.args.gdn_value_dim_tp,
        )

    def _project_qkvzab(self, x, S, out_mc=None):
        """Project x → (qkv, z, a, b). Fused path: one [qkv|z|a|b] matmul then slice.
        out_mc: placement of the qkvzab matmul + slices. None → DRAM; prefill+decode now pass L1 to
        keep qkvzab + q/k/v/z/a/b resident (was DRAM to spare NoC traffic — re-measure if reverting)."""
        Nv, qz, az = self.Nv, self.qkv_dim_tp, self.qkvz_dim_tp
        _proj_mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG
        if self._fuse_ab:
            # Prefill: x is K-sharded (norm skipped its AG) -> fused all-gather + qkvzab matmul.
            if self._fuse_agmm and S > tpc.TILE_SIZE:
                qkvzab = tpc.all_gather_matmul_prefill(
                    x,
                    self.tw["qkvz"],
                    self.tt_ccl,
                    self.cfg,
                    self.args.ccl_topology(),
                    out_memory_config=_proj_mc,
                )
                qkvzab = ttnn.reshape(qkvzab, (1, S, qkvzab.shape[-1]))
            elif getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:
                # Decode: small-grid 1D matmul on the interleaved fused weight (beats the DRAM-sharded grid).
                qkvzab = tpc.matmul_1d_decode(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvz_decode_1d_progcfg,
                    self.cfg_in,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                qkvzab = self._col_proj(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvzab_progcfg,
                    out_memory_config=_proj_mc,
                    compute_cfg=self.cfg_in if S <= tpc.TILE_SIZE else self.cfg,
                )
            qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
            # z (output gate) lives across the chunk kernel (gated = out_f * silu(z)); L1 z (6MB@S=2048)
            # clashes with the scan kernel CBs -> keep DRAM in chunk-prefill; decode (small S) keeps out_mc.
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=_z_mc)
            # a,b end mid-tile; slicing straight from qkvzab untilizes the full 4120-wide tensor.
            # Grab the enclosing tile-aligned block once (no untilize), then split a/b from it (test_gdn_slice_opt).
            _ab_end = min(az + -(-2 * Nv // tpc.TILE_SIZE) * tpc.TILE_SIZE, qkvzab.shape[-1])  # 2*Nv up to a tile
            ab = ttnn.slice(qkvzab, (0, 0, az), (1, S, _ab_end), memory_config=out_mc)
            ttnn.deallocate(qkvzab)
            a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
            b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv), memory_config=out_mc)
            ttnn.deallocate(ab)
            return qkv, z, a, b
        _cfg_in = self.cfg_in if S <= tpc.TILE_SIZE else self.cfg
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg, compute_cfg=_cfg_in)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=_cfg_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(
        self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False, prefill_masks=None
    ):
        """Causal chunk-prefill from scratch. x [1,1,T,dim]: K-sharded (dim/tp per device) when the
        fused in-proj AG-matmul path is active (``_fuse_agmm`` and T>TILE — the norm skips its
        post-AG); replicated otherwise. Output reduce-scattered.

        valid_len: real token count (rest is padding). capture_state: save rec/conv state for decode.
        return_state: when True (per-user batched prefill), return
        ``(output, final_state, conv_new_state)`` for one user's from-scratch B=1
        pass and skip all self.* writeback; the caller stitches per-user states via
        assemble_batched_state(). Single-sequence behavior is unchanged when False.
        prefill_masks: traced masked-bucket prefill (valid_len stays None; the real length lives in PERSISTENT device
        tensors the caller rewrites per request): dict with "bg" [1,T,1] fp32 TILE (1 for t < valid_len; multiplied
        into beta/g), "conv_sel" [1,K-1,T] bf16 TILE one-hot of the last K-1 real conv-input rows (conv_new_state),
        and optionally "qkv" (q/k/v mask, parity only). Requires the KDA conv and a from-scratch (zeroed) carry.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[1]
        # Pass the RAW valid_len (may be None) to the conv-FIR / seq kernels below — NOT a
        # `valid_len or T` coercion. A full chunk (valid_len is None) must take the kernels'
        # valid_len-None path (a static last-(K-1) slice for the conv state), which is trace-safe;
        # the valid_len-set path builds a one-hot via ttnn.from_torch (a host write) that TT_FATALs
        # ("Writes are not supported during trace capture") inside the captured chunk-outer trace.
        # Masked buckets still pass a real valid_len (< T) so their exact masking is unchanged, and
        # for a full chunk the None slice and the valid_len==T one-hot select the identical rows.

        # Cross-chunk carry (chunk-outer prefill): the recurrent + conv state continue from the
        # persistent buffers (zeroed at sequence start by reset_state_inplace, so a from-scratch
        # single pass reads zeros == None).
        # Per-user prefill (return_state) is always from scratch: must not carry the shared
        # batched buffer (other users' state) as its initial recurrent/conv state.
        carry = not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        # Prefill qkvzab in L1: keeps proj + q/k/v/z/a/b resident for conv+gate prep.
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=ttnn.L1_MEMORY_CONFIG)

        # Causal conv + SiLU; conv_state = previous chunk's last K-1 inputs (None/zero from scratch).
        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        _cstate = self.conv_carry if carry else None
        kd = self.key_dim_tp
        if self._gdn_kda_conv and valid_len is None:
            # KDA op: conv + SiLU + q/k/v split in one program; its outputs are already the three
            # token-major tensors the flat-qkv path wants (masked buckets keep the MAC FIR below).
            q, k, v, conv_new_state = self._kda_conv_prefill(
                qkv, T, _cstate, conv_sel=None if prefill_masks is None else prefill_masks["conv_sel"]
            )
            ttnn.deallocate(qkv)
            if self._gdn_flat_qkv:
                _qkv_head_dims = (Nk, Dk, Nv, Dv)
            else:
                q = ttnn.reshape(q, (1, T, Nk, Dk))
                k = ttnn.reshape(k, (1, T, Nk, Dk))
                v = ttnn.reshape(v, (1, T, Nv, Dv))
                _qkv_head_dims = None
        else:
            assert prefill_masks is None, "persistent prefill_masks need the KDA conv (QWEN_GDN_CONV=kda, K=4)"
            # The MAC FIR: masked buckets (their one-hot new_state selection) and QWEN_GDN_CONV=fir.
            conv, conv_new_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                self.K,
                self.mesh,
                # Conv in L1 (output freed before chunk kernel; new_state lands in DRAM internally)
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_state=_cstate,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=valid_len,
            )
            ttnn.deallocate(qkv)
            if self._gdn_flat_qkv:
                # Flat q/k/v: adapter splits heads inside untilize
                q = ttnn.slice(conv, (0, 0, 0), (1, T, kd))
                k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd))
                v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp))
                _qkv_head_dims = (Nk, Dk, Nv, Dv)
            else:
                q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
                k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
                v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
                _qkv_head_dims = None
            ttnn.deallocate(conv)
        # GQA late-expand: adapter L2-norms at Nk, expands to Nv after
        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"])), (1, T, Nv))
        ttnn.deallocate(a)

        # Fused chunk_gated_delta_rule; also used for masked valid_len.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        # const_tiles / program_config / wy_inverse only apply to the fused op; the seq adapter has none of them.
        _extra = (
            {
                "const_tiles": self._fused_const_tiles,
                "program_config": self.gdn_program_config,
                "wy_inverse": self.gdn_wy_inverse,
            }
            if _use_fused
            else {}
        )
        if prefill_masks is not None:
            assert _use_fused, "persistent prefill_masks need the fused chunk op"
            _extra["masks"] = (prefill_masks["bg"], prefill_masks.get("qkv"))
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_len,
            qkv_head_dims=_qkv_head_dims,
            return_o_bh=self._gdn_fuse_out,
            **_extra,
        )
        B, D = 1, self.qkv_dim_tp
        captured = None
        if return_state:
            # Per-user prefill: return this user's state for assemble_batched_state to stitch
            # into the batched buffers. No self.* writeback; tensors are not deallocated here.
            captured = (final_state, conv_new_state)
        else:
            # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
            # In place (ttnn.copy) so the addresses the prefill/decode traces
            # baked in stay valid across execute_trace replays and across sequences.
            ttnn.copy(final_state, self.rec_state)
            ttnn.deallocate(final_state)
            ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
            # ---- Finalize the decode conv window (last chunk / short prompt). ----
            # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
            # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
            if capture_state:
                if self.conv_states is None:
                    self.reset_state()
                if self._zero_conv0 is not None:
                    ttnn.copy(self._zero_conv0, self.conv_states[0])
                else:
                    zero = ttnn.from_torch(
                        torch.zeros(1, B, D, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    )
                    ttnn.copy(zero, self.conv_states[0])
                    ttnn.deallocate(zero)
                for j in range(self.K - 1):
                    src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                    ttnn.copy(src, self.conv_states[j + 1])
                if self._hist_hook_active() and self._decode_B == 1:
                    # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
                    # (A batched model prefills per user into the B=1 scratch; write_slot / assemble_batched_state
                    # then move the state into the decode history, so the hook is single-user only.)
                    self._write_conv_hist(conv_new_state)
                # Prefill wrote the taps directly, so they are the truth and the [1,K,C] window
                # mirror the fullbatch verify reads its carry from is now behind (re-seeded lazily
                # by _ensure_conv_win, on the next EAGER verify — the spec loop's seed).
                self._conv_taps_stale = False
                self._conv_win_stale = True
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj
        _L1 = ttnn.L1_MEMORY_CONFIG
        if self._gdn_fuse_out:
            # Fuse adapter relayout with per-head rms_norm + head-flatten.
            # TILE-native head->token relayout (transpose + fold), dropping the
            # TILE->ROW_MAJOR->TILE round-trip. o is head-major (1,Nv,T,Dv).
            n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            n = ttnn.reshape(n, (1, Nv, T, Dv))
            # Fused head->token relayout: [1,Nv,T,Dv] -> [1,1,T,Nv*Dv].
            n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        else:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_L1)
            ttnn.deallocate(out_n)
        if self._out_colpar_prefill:
            # Column-parallel out-proj: the gate multiply emits the AGMM input directly as bf16 (the
            # only numerics change vs the fp32 MMRS arm: activation quantized to bf16 before the
            # matmul, as every other projection in the model already does).
            gated = _silu_mul(out_f, z, _L1, dtype=ttnn.bfloat16)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z)
            # TODO(#57458): switch to the op's barrier_semaphore once it is wired up (see tpc.agmm_gather_buffer).
            out = tpc.all_gather_matmul_prefill(
                gated,
                tw["out_colpar"],
                self.tt_ccl,
                self.cfg,
                self.args.ccl_topology(),
                out_memory_config=_L1,
                persistent_output_buffer=tpc.agmm_gather_buffer(self.tt_ccl, gated),
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        gated = _silu_mul(out_f, z, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        # Prefill: fused out-proj matmul + reduce-scatter (matmul_reduce_scatter_async), flag-gated.
        if self._fuse_out_mmrs_prefill:
            x_out = ttnn.reshape(gated, (1, 1, T, gated.shape[-1]))
            # fp32 output is load-bearing: o_proj is row-parallel, so the RS SUMS 4 per-device partials
            # across devices — bf16 there tanks PCC to ~0.69 even at ISL 2048 (test_oproj_dtype_isl). Keep fp32.
            out = tpc.matmul_reduce_scatter_prefill(
                x_out, tw["out"], self.tt_ccl, self.cfg, self.args.ccl_topology(), self.args.num_devices, ttnn.float32
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if return_state:
            return out, captured[0], captured[1]
        return out

    def forward_prefill_collect(self, x, chunk_size=128, valid_len=None):
        """Per-user prefill that stashes this user's B=1 state for later assembly.

        Called once per user; finalize_pending() then stitches the collected states into the
        batched decode buffers. Returns the user's prefill output (needed for residual + MLP)."""
        out, rec, conv = self.forward_prefill(x, chunk_size=chunk_size, valid_len=valid_len, return_state=True)
        self._pending.append((rec, conv))
        return out

    def finalize_pending(self):
        """Assemble the per-user states collected by forward_prefill_collect into the batched
        decode buffers (row u = user u), then clear the accumulator."""
        assert self._pending, "finalize_pending called with no collected per-user states"
        rec_list = [r for (r, _) in self._pending]
        conv_list = [c for (_, c) in self._pending]
        self.assemble_batched_state(rec_list, conv_list)
        self._pending = []

    def assemble_batched_state(self, rec_list, conv_new_list):
        """Stitch B per-user prefill states (from forward_prefill(return_state=True)) into the
        batched decode buffers.

        rec_list[u]: [1, Nv, Dk, Dv] recurrent state; conv_new_list[u]: [1, K-1, qkv_dim_tp]
        last-(K-1) conv inputs. Row u of rec_state and conv_states[1..K-1] becomes user u's state;
        conv_states[0] is zeroed (shifted-out tap). ttnn has no in-place row write, so buffers are
        built by concat along the batch dim (rec: dim 0; conv: dim 1).

        The result is copied into the fixed-address buffers.
        """
        if self.rec_state is None:
            self.reset_state()
        assert len(rec_list) == self.B and len(conv_new_list) == self.B, "need one state per batch row"
        D = self.qkv_dim_tp
        # B == 1: concat of a single tensor returns its input, which is freed with rec_list below: copy it.
        rec_batched = ttnn.clone(rec_list[0]) if len(rec_list) == 1 else ttnn.concat(rec_list, dim=0)
        conv_states = [
            ttnn.from_torch(
                torch.zeros(1, self.B, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        ]
        for m in range(1, self.K):  # conv_states[m] row u = conv_new_list[u][:, m-1]
            rows = [
                ttnn.reshape(ttnn.slice(conv_new_list[u], (0, m - 1, 0), (1, m, D)), (1, 1, D)) for u in range(self.B)
            ]
            # B == 1: concat of a single tensor returns its input (a view of conv_new_list[u], freed at the end):
            # take an independent copy and leave the row to that free.
            cs = ttnn.clone(rows[0]) if len(rows) == 1 else ttnn.concat(rows, dim=1)  # [1, B, D]
            if len(rows) > 1:
                for r in rows:
                    ttnn.deallocate(r)
            conv_states.append(cs)

        rec_src = (
            rec_batched
            if rec_batched.dtype == self.rec_state.dtype
            else ttnn.typecast(rec_batched, self.rec_state.dtype)
        )
        ttnn.copy(rec_src, self.rec_state)
        if rec_src is not rec_batched:
            ttnn.deallocate(rec_src)
        ttnn.deallocate(rec_batched)
        for m in range(self.K):
            ttnn.copy(conv_states[m], self.conv_states[m])
            ttnn.deallocate(conv_states[m])
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True
        if self._hist_hook_active():
            # Fused decode reads its conv history from the RM buffer, not conv_states.
            if self._decode_B == 1:
                if len(conv_new_list) == 1:
                    self._write_conv_hist(conv_new_list[0])
            else:
                self._write_conv_hist_users(conv_new_list)  # full replace: row u = user u
            self._conv_fmt = "both"  # conv_states[1..K-1] were written above, hist just now
        for t in rec_list:
            ttnn.deallocate(t)
        for t in conv_new_list:
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ #
    # Per-slot state edits for vLLM continuous batching.
    # ------------------------------------------------------------------ #
    # The demo prefills all B users up front and assembles the whole batch at
    # once (assemble_batched_state). vLLM instead prefills ONE user at a time
    # into its decode slot while the other rows are mid-decode, and condenses
    # the batch when a request finishes. GDN's recurrent+conv state is a fixed
    # [B,...] buffer indexed by physical slot (not paged), so both events need a
    # single-row edit that preserves the other (live) rows. ttnn has no in-place
    # row write, so — exactly like assemble_batched_state — these rebuild the
    # buffer by slice+concat and ttnn.copy the result back (the copy preserves
    # the decode trace's baked buffer address).
    def _slice_along(self, buf, dim, lo, hi):
        """ttnn.slice of buf along `dim` for indices [lo, hi), other dims kept full."""
        start = [0] * len(buf.shape)
        end = list(buf.shape)
        start[dim] = lo
        end[dim] = hi
        return ttnn.slice(buf, tuple(start), tuple(end))

    def _write_recurrent_state_prefix(self, new_rec, B):
        """Write active rows [0:B] without reading or copying idle rows."""
        grid_size = self.mesh.compute_with_storage_grid_size()
        assert (
            grid_size.x >= 8 and grid_size.y >= 6
        ), f"GDN prefix state write needs an 8x6 core rectangle, got {grid_size.x}x{grid_size.y}"
        nhw = B * self.Nv * self.Dk
        assert (
            nhw % ttnn.TILE_SIZE == 0
        ), f"GDN prefix state rows B={B}, Nv={self.Nv}, Dk={self.Dk} -> {nhw} is not tile-aligned"
        n_tiles = nhw // ttnn.TILE_SIZE

        # Prefer the tuned 8x6=48-core rectangle, which every TP=4 shape hits (Nv=12 -> nhw=B*1536
        # -> 48*B tiles). At TP=8 Nv halves to 6, so B=1 gives only 24 tiles and cannot fill 48
        # cores with tile-aligned shards — fall back to the largest core count that divides evenly.
        if n_tiles % 48 == 0:
            num_cores = 48
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
        else:
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

        shard_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid,
                (nhw // num_cores, self.Dv),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        src = (
            new_rec
            if new_rec.dtype == self.rec_state.dtype
            else ttnn.typecast(new_rec, self.rec_state.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        sharded = ttnn.to_memory_config(src, shard_memcfg)
        ttnn.experimental.slice_write(
            sharded,
            self.rec_state,
            [0, 0, 0, 0],
            [B, self.Nv, self.Dk, self.Dv],
            [1, 1, 1, 1],
        )
        ttnn.deallocate(sharded)
        if src is not new_rec:
            ttnn.deallocate(src)
        ttnn.deallocate(new_rec)

    def _write_index(self, buf, src, idx, dim, consume_src=True):
        """Replace slice `idx` of `buf` along `dim` with `src` (extent 1 along `dim`), preserving
        the other slices, via an in-place copy into `buf`. Consumes `src` (and the temporary
        slices) unless consume_src=False. `src` must already match `buf`'s dtype."""
        n = buf.shape[dim]
        if n == 1:
            ttnn.copy(src, buf)
            if consume_src:
                ttnn.deallocate(src)
            return
        parts = []
        if idx > 0:
            parts.append(self._slice_along(buf, dim, 0, idx))
        parts.append(src)
        if idx < n - 1:
            parts.append(self._slice_along(buf, dim, idx + 1, n))
        new = ttnn.concat(parts, dim=dim)
        ttnn.copy(new, buf)
        ttnn.deallocate(new)
        for p in parts:
            if p is src and not consume_src:
                continue
            ttnn.deallocate(p)

    def write_slot(self, slot, rec, convs, consume=True):
        """Write one user's B=1 prefill state into decode `slot`, preserving every other (live)
        row. The per-slot analogue of assemble_batched_state for vLLM continuous batching.

        rec:   [1, Nv, Dk, Dv] the user's recurrent state.
        convs: list of K [1, 1, qkv_dim_tp] the user's conv taps (conv_states[m] column). Unlike
               assemble_batched_state (which zeroes tap 0), every tap is written straight from the
               user's B=1 prefill state, so decode continues from exactly the produced shift register.
        Consumes rec and convs unless consume=False (device-resident sources that must survive, e.g. the persistent
        B=1 prefill scratch: the slot write then costs no host round trip). Requires the batched buffers
        (allocate_kv_caches(batch_size=B))."""
        assert self.rec_state is not None and self.conv_states is not None, "batched GDN state not allocated"
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        if self._hist_hook_active() and self._decode_B > 1:
            # Fused batched decode reads its conv history from the RM buffer: write ONLY row `slot` (the K-1 newest
            # taps, convs[1..K-1], oldest first) BEFORE the loop below consumes convs. Live rows are untouched.
            if self._conv_fmt == "packed":
                # Direct packed write: the format stays "packed" (RM history / conv_states are stale; every RM reader
                # converts packed -> RM first), so no unpack here and no re-pack at the next decode.
                self._write_conv_hist_slot_packed(slot, convs)
            elif self._conv_fmt in ("both", "hist"):
                self._write_conv_hist_slot(slot, convs)
        self.sync_conv_taps()  # read-modify-write of the taps: they must be current first
        rec_src = rec if rec.dtype == self.rec_state.dtype else ttnn.typecast(rec, self.rec_state.dtype)
        if rec_src is not rec and consume:
            ttnn.deallocate(rec)
        if self.rec_state.shape[0] > 1:
            # In-place row write (no slice + concat + copy of the whole [B, Nv, Dk, Dv] state; bit-identical).
            ttnn.experimental.slice_write(
                rec_src,
                self.rec_state,
                [slot, 0, 0, 0],
                [slot + 1] + list(self.rec_state.shape)[1:],
                [1, 1, 1, 1],
            )
            if consume or rec_src is not rec:
                ttnn.deallocate(rec_src)
        else:
            self._write_index(self.rec_state, rec_src, slot, dim=0, consume_src=consume or rec_src is not rec)
        # conv_states are stale (and fully rewritten by the hist -> states sync) while the fused history is the
        # authoritative format ("hist" / "packed"): skip the K slice+concat+copy writes then.
        write_states = not (self._hist_hook_active() and self._decode_B > 1 and self._conv_fmt in ("hist", "packed"))
        for m in range(self.K):
            c = convs[m]
            if not write_states:
                if consume:
                    ttnn.deallocate(c)
                continue
            c_src = c if c.dtype == self.conv_states[m].dtype else ttnn.typecast(c, self.conv_states[m].dtype)
            if c_src is not c and consume:
                ttnn.deallocate(c)
            self._write_index(self.conv_states[m], c_src, slot, dim=1, consume_src=consume or c_src is not c)
        self._conv_win_stale = True
        if self._hist_hook_active() and self._decode_B == 1:
            # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
            self._sync_conv_hist_from_states()

    def remap_slots(self, remap):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops). Mirrors
        seed_manager.apply_slot_remap for GDN's per-slot recurrent+conv state, which the plugin's
        slot_remap does not itself move. In-place copy into the fixed buffers (preserves the decode
        trace's baked addresses).

        Format-aware: only the conv format(s) that are VALID per `_conv_fmt` are gathered. rec_state is always
        gathered. With the fused batched hook active the live history is _conv_hist_rm; under tag "hist" conv_states
        is stale (the full hist->states sync in prepare_decode_width rewrites every row before conv_states is read
        again), under "states" _conv_hist_rm is stale (symmetric), under "both" both are gathered. Without the hook
        only conv_states exists/matters. The tag is left unchanged."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self.sync_conv_taps()  # read-modify-write of the taps: they must be current first
        self._gather_indices(self.rec_state, idx, dim=0)
        if self._conv_fmt == "packed" and self._hist_hook_active() and self._decode_B > 1:
            # Parity-aware single gather over the packed rows; RM history / conv_states are stale and stay so.
            self._remap_packed(idx)
            self._conv_win_stale = True
            return
        hist_used = self._hist_hook_active() and self._decode_B > 1 and self._conv_hist_rm is not None
        if not hist_used:
            for m in range(self.K):
                self._gather_indices(self.conv_states[m], idx, dim=1)
        elif self._conv_fmt == "hist":
            self._gather_indices(self._conv_hist_rm, idx, dim=0)
        elif self._conv_fmt == "states":
            for m in range(self.K):
                self._gather_indices(self.conv_states[m], idx, dim=1)
        else:  # "both"
            for m in range(self.K):
                self._gather_indices(self.conv_states[m], idx, dim=1)
            self._gather_indices(self._conv_hist_rm, idx, dim=0)
        self._conv_win_stale = True

    def _gather_indices(self, buf, idx, dim):
        """Rebuild `buf` so slice i along `dim` becomes old slice idx[i], then copy back in place.
        idx is a permutation; it is decomposed into maximal runs (dst_lo, src_lo, n) with idx[dst_lo+k] == src_lo+k,
        so one slice per run is taken instead of one per row. Identity => no-op.
        `new` is fully materialized before the copy, so gathering from `buf` into itself is safe."""
        runs = []
        for d, s in enumerate(idx):
            if runs and runs[-1][1] + runs[-1][2] == s and runs[-1][0] + runs[-1][2] == d:
                runs[-1][2] += 1
            else:
                runs.append([d, s, 1])
        if len(runs) == 1 and runs[0][0] == runs[0][1]:
            return
        pieces = [self._slice_along(buf, dim, s, s + n) for (_d, s, n) in runs]
        new = ttnn.concat(pieces, dim=dim)
        ttnn.copy(new, buf)
        # ttnn may return views/aliases (e.g. a single full-length run): compare addresses BEFORE any deallocate.
        keep = {new.buffer_address(), buf.buffer_address()}
        addrs = [pc.buffer_address() for pc in pieces]
        pieces_free = [pc for pc, a in zip(pieces, addrs) if pc is not new and a not in keep]
        ttnn.deallocate(new)
        for pc in pieces_free:
            ttnn.deallocate(pc)

    def forward_prefill_batched(self, x, chunk_size=128, valid_lens=None, carry=False):
        """Batched prefill: all B users in one pass (no per-user Python loop).

        The chunk-seq GDN kernel scans a leading BH = B*H batch dim, each (user, head) row an
        independent causal scan, so B is a true batch dim (not a time concat). Runs projection /
        conv-FIR / chunk-parallel recurrence over [B, T, *] and writes straight into the batched
        decode buffers (rec_state[B,Nv,Dk,Dv], conv_states[*][1,B,D]); row u == user u.

        x:          [B, T, dim] replicated (all users padded to a common bucket length T).
        valid_lens: optional list of B real token counts (< T => right-padding masked per row);
                    None => every row is full length T.
        carry:      False (default) => from scratch (single-shot). True => CHUNK-OUTER carry: read
                    the recurrent state (self.rec_state) and conv left-context (self._batched_conv_carry)
                    from the previous chunk and write the updated ones back, so a long prompt can be
                    prefilled chunk-by-chunk over the batch. Mirrors the B=1 forward_prefill carry;
                    the caller zeroes rec_state (reset_state_inplace) + _batched_conv_carry at
                    sequence start, so the first chunk reads zeros (== from scratch).

        KERNEL CAP: gated_delta_attn_seq maps one BH = B*Nv_tp row per core and is L1-bound, so BH
        must stay <= ~32 (at TP=4, Nv_tp=8 => B <= 4). Larger B trips an L1 clash (B=8) or the
        kernel's `BH <= compute_grid` assert (B=32); B>4 would need grouped launches (groups <=4).
        The model currently prefills per-user instead (see prefill_paged_peruser).
        """
        if self.rec_state is None:
            self.reset_state()
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[-3], x.shape[-2], x.shape[-1]))  # [.,B,T,dim] -> [B,T,dim]
        B, T = x.shape[0], x.shape[1]
        D = self.qkv_dim_tp

        # Route through the shared per-token projection (handles _fuse_ab/_fuse_agmm — required
        # when the caller's norm skipped its post-AG and x arrives K-sharded, e.g. prefill_paged_
        # grouped). A plain ttnn.linear(x, tw["qkvz"]) here would (a) mismatch the K-sharded width
        # against the fused-weight's full-K height, and (b) KeyError on tw["ab"], which doesn't
        # exist when _fuse_ab folds a/b into tw["qkvz"]. Flatten the batch dim into the token dim
        # (the projection is per-token; user boundaries don't matter to a linear layer) since
        # _project_qkvzab's slicing assumes a leading dim of 1.
        x_flat = ttnn.reshape(x, (1, B * T, x.shape[-1]))
        qkv_flat, z_flat, a_flat, b_flat = self._project_qkvzab(x_flat, B * T, out_mc=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.reshape(qkv_flat, (B, T, D))
        z = ttnn.reshape(z_flat, (B, T, self.qkvz_dim_tp - D))
        a = ttnn.reshape(a_flat, (B, T, Nv))
        b = ttnn.reshape(b_flat, (B, T, Nv))

        # FIR causal conv1d + SiLU over each user's sequence (per-row valid_len picks each user's
        # decode conv window). Chunk-outer carry: left-context = previous chunk's last K-1 inputs.
        if carry and getattr(self, "_batched_conv_carry", None) is None:
            # First chunk of a chunk-outer prefill: zeroed left-context (== from scratch).
            self._batched_conv_carry = ttnn.from_torch(
                torch.zeros(B, self.K - 1, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        conv_carry_in = self._batched_conv_carry if carry else None
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=conv_carry_in,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_lens,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        # Flat token-major q/k/v (no host head-split / GQA): the fused op does in-kernel L2-norm and
        # GQA (Nk->Nv) from qkv_head_dims, matching the single-user forward_prefill fused path.
        q = ttnn.slice(conv, (0, 0, 0), (B, T, kd))
        k = ttnn.slice(conv, (0, 0, kd), (B, T, 2 * kd))
        v = ttnn.slice(conv, (0, 0, 2 * kd), (B, T, D))
        ttnn.deallocate(conv)

        beta = ttnn.reshape(ttnn.sigmoid(b), (B, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"])), (B, T, Nv))
        ttnn.deallocate(a)

        # Chunk-parallel recurrence over the BH = B*Nv batch (each row an independent scan). Fused
        # chunk_gated_delta_rule (same op as single-user prefill); per-row valid_lens mask each user.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        _extra = (
            {
                "const_tiles": self._fused_const_tiles,
                "program_config": self.gdn_program_config,
                "wy_inverse": self.gdn_wy_inverse,
            }
            if _use_fused
            else {}
        )
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_lens,
            qkv_head_dims=(Nk, Dk, Nv, Dv),
            **_extra,
        )

        # ---- write the batched decode state directly (row u == user u) ----
        rec_src = (
            final_state
            if final_state.dtype == self.rec_state.dtype
            else ttnn.typecast(final_state, self.rec_state.dtype)
        )
        ttnn.copy(rec_src, self.rec_state)
        if rec_src is not final_state:
            ttnn.deallocate(rec_src)
        ttnn.deallocate(final_state)
        # conv_states[0] = shifted-out zero; conv_states[m] row u = conv_new_state[u, m-1].
        zero0 = ttnn.from_torch(
            torch.zeros(1, B, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        new_conv = [zero0]
        for m in range(1, self.K):
            cs = ttnn.reshape(ttnn.slice(conv_new_state, (0, m - 1, 0), (B, m, D)), (1, B, D))  # [1,B,D]
            new_conv.append(cs)
        if carry:
            # Preserve this chunk's last K-1 inputs as the next chunk's left-context (replace the
            # buffer just consumed by the FIR above).
            if conv_carry_in is not None:
                ttnn.deallocate(conv_carry_in)
            self._batched_conv_carry = conv_new_state  # [B, K-1, D]
        else:
            ttnn.deallocate(conv_new_state)
        for m in range(self.K):
            ttnn.copy(new_conv[m], self.conv_states[m])
            ttnn.deallocate(new_conv[m])
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True
        if self._hist_hook_active() and B == self._decode_B:
            # Fused decode reads its conv history from the RM buffer, not conv_states: full replace from the
            # just-written conv_states (single-user: the [1, K-1, C] row; batched: every user's row).
            self._sync_conv_hist_from_states()
            self._conv_fmt = "both"

        # ---- output (gated RMSNorm + SiLU(z) gate + row-parallel out proj + all-reduce) ----
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (B, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, B, T, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, x, decode_ar=None):
        """decode_ar: tp_common.DecodeResidualAllReduce (QWEN36_DECODE_ALLREDUCE=1) -> x is the replicated
        residual-norm output and the out-proj partial is all-reduced (replicated RES40, bf16 unless
        QWEN36_DECODE_AR_GDN_FP32=1) instead of reduce-scattered."""
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        Bmax = self.B
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode conv→recurrence→norm/gate chain L1-resident
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))

        # Active decode width, taken from the input. Normally == Bmax. BUCKETED decode: a request
        # feeds B<Bmax tokens and the whole step runs on state rows [0:B]; idle rows [B:Bmax] are
        # preserved. Conv taps are per-channel (broadcast over batch), so the conv weighted-sum
        # works at any width. The B==Bmax path is byte-identical to before.
        B = x.shape[-2]
        if self._decode_op and not self.use_fused_recurrent_decode and B <= 32 and Bmax == self._decode_B:
            # gdn_decode_step op (QWEN36_GDN_DECODE_STEP_OP): every width 1..Bmax on the packed conv history.
            self._require_conv_fmt("packed")
            return self._forward_decode_step_op(x, decode_ar)
        if self._fused_decode and Bmax == 1 and B == 1 and not self.use_fused_recurrent_decode:
            # (Spec decode sets use_fused_recurrent_decode: its verify advances conv_states + rec_state with the fused
            # recurrent op, so decode must stay on that path too -- the fused-conv decode below keeps its history in
            # _conv_hist_rm instead.)
            # QWEN36_GDN_FUSED_DECODE (default on), single-user: fused conv / chunk-delta-rule / gated-norm kernels.
            return self._forward_decode_fused(x, decode_ar)
        if self._fused_batched:
            # Batched fused decode (max_batch_size > 1, every width 1..Bmax). The conv history is the unified
            # _conv_hist_rm [Bmax, K-1, C]; conv_states is NOT maintained in this mode.
            assert Bmax == self._decode_B, (
                f"fused batched GDN decode needs the decode binding (B={Bmax}, max_batch_size={self._decode_B}); "
                "set QWEN36_GDN_FUSED_DECODE=0 to run another batch size"
            )
            if B <= self._scan_max_b:
                self._require_conv_fmt("hist")
                return self._forward_decode_fused_batched(x, decode_ar)
            # Wide widths (B > SCAN_MAX): the ENTIRE original path below (shift-register conv on conv_states +
            # original recurrence on the shared rec_state); conv_states must be the valid format.
            self._require_conv_fmt("states")

        qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)
        q, k, v = self._decode_conv_original(qkv, B, Bmax)

        # GQA expand Q/K Nk→Nv; recurrence L2-norms + scales internally
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        # Decode: hand q/k/v to the recurrent kernel in L1. The kernel typecasts + does a LOCAL
        # l2-norm (no cross-device gather), so placement is output-neutral here (unlike SDPA-q,
        # which hard-requires DRAM, and unlike the residual→DistributedNorm all-gather).
        q = ttnn.reshape(q, (B, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(k, (B, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)

        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))

        # fp32 decode step by default (QWEN35_GDN_DECODE_BF16=1 reverts)
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        if self.use_fused_recurrent_decode:
            # Spec-decode path. The whole recurrence (decay->k.S->delta->outer->q.S) is ONE fused
            # device op rather than the ~13-op composite: 0.345 vs 0.541 ms, and closer to the FLA
            # reference (PCC 0.999991 vs 0.999981). Decode runs under trace, so the dispatch saving
            # is small — the reason spec decode selects it is CONSISTENCY. Spec verify advances GDN
            # with this same op, so decode and verify must use identical math or every greedy
            # near-tie flips between them and acceptance drops (measured 2.82 -> 2.00 /3 when the
            # two paths disagreed at ~1e-5).
            assert B == Bmax, "fused recurrent decode path supports full-batch only (spec decode, B=1)"
            o, new_rec = fused_recurrent_gated_delta_rule_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=self.rec_state,
                device=self.mesh,
                high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
            )
        else:
            o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
            )
        if init_state is not self.rec_state:
            ttnn.deallocate(init_state)
        # In-place update preserves rec_state address for decode trace replay
        if B == Bmax:
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self._write_recurrent_state_prefix(new_rec, B)

        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f, z, _L1)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is fp32 here
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def _decode_conv_original(self, qkv, B, Bmax):
        """The original decode conv: shift-register conv_states + weighted sum + SiLU, then the q/k/v split.
        Returns q [B, Nk, Dk], k [B, Nk, Dk], v [B, Nv, Dv]. Consumes qkv. Unchanged from the pre-batched code."""
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1 = ttnn.L1_MEMORY_CONFIG
        # Conv1d shift-register + weighted sum + SiLU.
        # The K taps below ARE the shift register, so they must be current: a preceding fullbatch
        # verify advanced only the [1,K,C] window mirror. No-op (zero device ops, so trace-safe)
        # unless a spec verify actually ran — see sync_conv_taps.
        self.sync_conv_taps()
        st = self.conv_states
        if B < Bmax:
            # Bucketed decode: active requests occupy a contiguous prefix [0:B]; idle rows [B:Bmax]
            # hold no live request (a slot is re-initialized by prefill/write_slot when reused), so
            # they are don't-care. Pad the width-B new input up to Bmax and run the SAME full-width
            # shift-register as below -- the conv sum's active rows [0:B] are exact and the downstream
            # q/k/v slices take [0:B]. This keeps the op COUNT identical to the baseline path (just a
            # single pad), vs a per-row slice/concat that added ~4*K ops/layer and erased the width win.
            qkv_p = ttnn.pad(qkv, [(0, 0), (0, Bmax - B), (0, 0)], value=0.0, memory_config=_L1)
            ttnn.deallocate(qkv)
            qkv = qkv_p
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        self._conv_win_stale = True  # decode shifted the taps; the window mirror is now behind
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

        return q, k, v

    def _forward_decode_fused(self, x, decode_ar=None):
        """Single-user (max_batch_size == 1, active width 1) decode through the fused kernels (see
        QWEN36_GDN_FUSED_DECODE). Same math as the original path, fewer ops:
          * ttnn.experimental.kda.qkv_causal_conv1d_silu: 4-tap causal conv + SiLU + q/k/v split in one program,
            history = the persistent RM buffer _conv_hist_rm (the last K-1 raw conv inputs, oldest first);
          * ttnn.transformer.chunk_gated_delta_rule with chunk_size=32 over ONE live token padded to a 32-row tile:
            beta and g are zero-padded, so the 31 padded steps are identity state updates (the qkv padding rows
            only feed those identity steps), and it does the q/k L2-norm, GQA expansion and the fp32 state update;
          * ttnn.experimental.kda.sigmoid_gated_rms_norm (= norm * w * sigmoid(z)) then * z -> norm * w * silu(z).
        Differences from the original path (all follow the validated fused decoder, models/demos/qwen38_27b_qb2):
        a and g stay fp32 (dt_bias / -exp(A_log) are fp32 copies), and the out-proj input is bf16 (the original
        feeds fp32; the out-proj partial is therefore bf16 too). conv_states is not updated here -- only
        _conv_hist_rm and rec_state are. conv input / q / k / v are DRAM; the padded z_p / g_p / beta_p are views of
        their (L1) sources and stay in that memory space until their last consumer. No allocation of persistent
        state: trace-safe.
        """
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE

        tw, Nv = self.tw, self.Nv
        _L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        C, kd, vd = self.qkv_dim_tp, self.key_dim_tp, self.value_dim_tp
        # The kernels run on one 32-row tile; row 0 is the live token. Zero rows: identity recurrence steps.
        # NOTE: ttnn.pad of a [1,1,C] TILE tensor up to the 32-row tile height returns a VIEW of the SAME device
        # buffer (the tile already holds the 32 rows; pad only zero-fills the implicit tile padding in place and
        # ignores memory_config). So the padded tensors (z_p, g_p, beta_p, qkv_p) must NEVER be freed by
        # deallocating their SOURCE (that frees the shared buffer under the live view -> later allocations
        # overwrite it -> garbage): the source is left alone and exactly ONE handle (the padded one) is
        # deallocated, after the padded tensor's last consumer.
        pad_rows = [(0, 0), (0, _FUSED_CHUNK_SIZE - 1), (0, 0)]

        qkv, z, a, b = self._project_qkvzab(x, 1, out_mc=_L1)

        # ---- z (the out gate) and the gates, ahead of the kernels: DRAM, 32 rows, zero padded ----
        if z.dtype != ttnn.bfloat16:  # sigmoid_gated_rms_norm's gate contract
            z = ttnn.typecast(z, ttnn.bfloat16, memory_config=_L1)
        z_p = ttnn.pad(z, pad_rows, value=0.0, memory_config=_DRAM)  # view of z; freed once, after the gate mul
        # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), a/g in fp32
        beta = ttnn.sigmoid(b, memory_config=_L1)
        ttnn.deallocate(b)
        a32 = ttnn.typecast(a, ttnn.float32, memory_config=_L1)
        ttnn.deallocate(a)
        a_dt = ttnn.add(a32, tw["dt_bias_fp32"], memory_config=_L1)
        ttnn.deallocate(a32)
        g = ttnn.mul(
            tw["neg_exp_A_fp32"],
            a_dt,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            memory_config=_L1,
        )
        ttnn.deallocate(a_dt)
        g_p = ttnn.pad(g, pad_rows, value=0.0, memory_config=_DRAM)  # view of g; freed once, after the chunk op
        beta_p = ttnn.pad(beta, pad_rows, value=0.0, memory_config=_DRAM)  # view of beta; freed once after chunk

        # ---- causal conv1d + SiLU + q/k/v split: one program on the persistent RM history ----
        hist = self._ensure_conv_hist()
        qkv_p = ttnn.pad(qkv, pad_rows, value=0.0, memory_config=_DRAM)  # view of qkv
        row_qkv = ttnn.to_layout(qkv_p, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)  # new (RM, DRAM) buffer
        ttnn.deallocate(qkv_p)  # last consumer of the tile view done: frees the shared qkv buffer, once
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            row_qkv,
            hist,
            *tw["conv_taps"],
            kd,
            kd,
            vd,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C, cap=256)),
            actual_start=self._kda_actual_start,
            # No sequence parallelism on the TP mesh: alias history (see kda_conv_prefill).
            predecessor_carry=hist,
            memory_config=_DRAM,
        )
        # History for the next token: [hist[1:], this token's raw conv input], updated in place (fixed address).
        keep = ttnn.slice(hist, (0, 1, 0), (1, self.K - 1, C), memory_config=_DRAM)
        cur = ttnn.slice(row_qkv, (0, 0, 0), (1, 1, C), memory_config=_DRAM)
        ttnn.deallocate(row_qkv)
        new_hist = ttnn.concat([keep, cur], dim=1, memory_config=_DRAM)
        ttnn.deallocate(keep)
        ttnn.deallocate(cur)
        ttnn.copy(new_hist, hist)
        ttnn.deallocate(new_hist)

        # ---- gated delta rule: one op, fp32 state in place (decode/prefill traces keep rec_state's address) ----
        # No program_config: gdn_program_config is a prefill-geometry knob; the op picks its own for this shape.
        eye, tril, ones, masks = self._fused_const_tiles
        o, new_rec = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g_p,
            beta_p,
            scale=self.scale,
            initial_state=self.rec_state,
            output_final_state=True,
            chunk_size=_FUSED_CHUNK_SIZE,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        for t in (q, k, v, g_p, beta_p):
            ttnn.deallocate(t)
        ttnn.copy(new_rec, self.rec_state)
        ttnn.deallocate(new_rec)

        # ---- gated RMSNorm (no +1) * silu(z) = (norm * w * sigmoid(z)) * z, bf16 into the out-proj ----
        normed = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o, z_p, tw["norm_w_1d"], Nv, epsilon=1e-6, output_dtype=ttnn.bfloat16
        )
        ttnn.deallocate(o)
        gated = ttnn.mul(normed, z_p, memory_config=_DRAM)
        ttnn.deallocate(normed)
        ttnn.deallocate(z_p)
        # Keep only the live row logically; the physical tile geometry is unchanged (a free view).
        gated = ttnn.reshape(gated, ttnn.Shape([1, 1, vd]), gated.padded_shape)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, 1, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here (fp32 cast only if asked)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_verify_recurrent(self, x, valid_len, pre_gathered=False):
        """Hybrid spec-decode verify for GDN: advance the recurrent state token-by-token over the
        first ``valid_len`` rows of the bucket. This is BIT-EXACT to ``valid_len`` sequential
        ``forward_decode`` steps (identical conv shift-register + recurrent kernel + in-place state
        updates), so verify uses the SAME kernel as decode instead of the lossy chunk kernel.

        Rows past ``valid_len`` are zero and are never read downstream: full attention is causal
        (real queries < valid_len never attend to padded keys) and verify only row-selects rows
        < valid_len. The rest of the layer stack (attn/MLP/norm/lm_head) still runs batched over the
        bucket, so only the GDN recurrence is sequential — that is the whole point of the hybrid.

        x : [1, 1, bucket, dim] prefill-normed input (same shape forward_prefill receives).
        Returns [1, 1, bucket, dim] full-dim (matches forward_prefill's output for the layer add).

        pre_gathered : the caller already handed us a FULL-dim activation (decode-config verify runs
        the layer norms in Mode.DECODE, which gathers pre-norm), so skip the internal all-gather.
        """
        assert valid_len <= tpc.TILE_SIZE, f"verify bucket {valid_len} exceeds one tile"
        return self._forward_verify_recurrent_batched(x, valid_len, pre_gathered=pre_gathered)

    def _forward_verify_recurrent_batched(self, x, valid_len, pre_gathered=False):
        """Batched hybrid verify — BIT-IDENTICAL to the per-token forward_decode loop, but with
        valid_len x fewer matmul/all-reduce launches. Key fact: the decode matmul (matmul_1d_decode)
        is row-independent and processes a full 32-row M-tile regardless of how many rows are real, so
        packing all `valid_len` (<= TILE) tokens into ONE decode matmul gives per-row-identical results
        while collapsing valid_len separate launches into one. Structure (cf. the reference's
        fused_sigmoid_gating_delta_rule_update: project once, loop the recurrence, output once):

          1. Gather the valid_len rows to full dim, then ONE decode qkvzab matmul (same kernel + weights
             forward_decode uses per token -> per-row bit-identical q/k/v/z/a/b).
          2. Per-token loop over valid_len: conv shift-register + fp32 recurrence step (the ONLY
             sequential part; carries rec_state/conv_states, so slot capture is unchanged).
          3. ONE gated-norm + out-proj (decode kernel) + all-reduce over the valid_len rows.

        Because every matmul is the decode kernel and row-independent, this is numerically identical to
        the per-token loop (verified: same accept rate + same trajectory), NOT an approximation. The
        AGMM prefill projection was avoided precisely because it rounds differently and drifts the state.
        """
        tw, B, Nk, Nv, Dk, Dv = self.tw, self.B, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        if self.conv_states is None:
            self.reset_state()
        # Decode-config verify hands us the DECODE attn-norm output, which is L1 WIDTH-SHARDED.
        # We slice valid rows out of it below, so interleave first (row-slicing a width-shard is not
        # supported). We then own that copy and must free it.
        _x_owned = False
        if pre_gathered and x.is_sharded():
            x = ttnn.to_memory_config(x, mc)
            _x_owned = True
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        bucket = x.shape[-2]  # x is K-sharded [1, bucket, dim/tp] (full-dim when pre_gathered)
        T = valid_len
        kd = self.key_dim_tp

        # 1) Gather only the valid_len rows to full dim (like the per-token loop's gather, but over
        #    valid_len rows not the whole bucket), then ONE decode qkvzab matmul. S=valid_len <= TILE
        #    routes _project_qkvzab through matmul_1d_decode — the exact per-token decode projection.
        x_valid = x if T == bucket else ttnn.slice(x, (0, 0, 0), (1, T, x.shape[-1]))
        if pre_gathered:
            # Decode-config verify: the layer already ran its norm in Mode.DECODE, which gathers
            # PRE-norm, so x is already full-dim [1, bucket, dim]. Gathering again would quadruple
            # the feature dim. Only free xg below if we own it (i.e. it is the slice, not the caller's x).
            xg = x_valid
        else:
            # x is 3D [1, T, dim/tp] here (reshaped above), so gather the LAST (feature) dim = -1, not 3.
            xg = tt_all_gather(
                x_valid,
                self.mesh,
                self.tt_ccl,
                cluster_axis=None,
                dim=-1,
                topology=self.args.ccl_topology(),
                memory_config=mc,
            )
            if x_valid is not x:
                ttnn.deallocate(x_valid)
        qkv_all, z_all, a_all, b_all = self._project_qkvzab(xg, T, out_mc=mc)
        if xg is not x:
            ttnn.deallocate(xg)
        if _x_owned:
            ttnn.deallocate(x)

        # 2) Sequential conv + recurrence per token — identical building blocks to forward_decode, so
        #    self.conv_states / self.rec_state advance exactly as decode does (bit-exact slot capture).
        capture = getattr(self, "_capture_slots", False)
        if capture:
            self._ensure_verify_slot_bufs(T)
            self._verify_slots = self._slot_bufs
        else:
            self._verify_slots = None
        # The recurrence is ALWAYS the fused device op now: the T sequential dispatches (~13 ops
        # each) collapse into one. The conv shift-register (a cheap FIR) stays sequential — it
        # produces per-token q/k/v that we collect, then run the whole recurrence in a single call
        # that also emits the state AFTER every token (output_per_token_state) for slot acceptance.
        # Measured: verify marginal 18.0 -> 15.2 ms/candidate (the recurrence is only ~15% of it).
        # self.use_fullbatch_verify (default True): eliminate the per-token loop ENTIRELY (marginal
        # -> 2.3 ms). The ~18 ms/candidate is ~50 device ops per token per GDN layer at a few us each
        # — launch-bound with no hot spot — so the only fix is to stop launching them. Three pieces
        # make that possible without any sliding-window slicing:
        #   conv    -> the KDA op (kda_conv_prefill), the T tokens right-padded to a multiple of 32
        #              (causal: rows [0, T) exact);
        #   q/k/v   -> split by the KDA op itself; the padded rows are sliced off and q/k
        #              repeat_interleaved once for all T;
        #   beta/g  -> a_all/b_all are already [1,T,Nv], so the gating is one sigmoid / softplus pass;
        #   recur.  -> the C++ fused_recurrent_gated_delta_rule kernel consumes [B,T,Nv,D] for all T
        #              tokens in ONE dispatch and emits per-token state for slot acceptance.
        if self.use_fullbatch_verify:
            return self._verify_fullbatch(qkv_all, z_all, a_all, b_all, T, bucket, capture)
        # Per-token path: the taps below ARE the shift register (same contract as forward_decode).
        self.sync_conv_taps()
        self._conv_win_stale = True
        rf = Nv // Nk
        out_f_rows = []
        q_seq, k_seq, v_seq, beta_seq, g_seq = [], [], [], [], []
        for t in range(T):
            qkv_t = ttnn.reshape(ttnn.slice(qkv_all, (0, t, 0), (1, t + 1, self.qkv_dim_tp)), (1, B, self.qkv_dim_tp))
            st = self.conv_states
            for j in range(self.K - 1):
                ttnn.copy(st[j + 1], st[j])
            ttnn.copy(qkv_t, st[self.K - 1])
            ttnn.deallocate(qkv_t)
            conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
            for j in range(1, self.K):
                conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
            conv = ttnn.silu(conv, memory_config=_L1)

            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
            ttnn.deallocate(conv)
            q = ttnn.reshape(ttnn.repeat_interleave(q, rf, dim=1), (B, 1, Nv, Dk), memory_config=_L1)
            k = ttnn.reshape(ttnn.repeat_interleave(k, rf, dim=1), (B, 1, Nv, Dk), memory_config=_L1)
            v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)

            a_t = ttnn.reshape(ttnn.slice(a_all, (0, t, 0), (1, t + 1, Nv)), (1, B, Nv))
            b_t = ttnn.reshape(ttnn.slice(b_all, (0, t, 0), (1, t + 1, Nv)), (1, B, Nv))
            beta = ttnn.reshape(ttnn.sigmoid(b_t, memory_config=_L1), (B, 1, Nv))
            ttnn.deallocate(b_t)
            g = ttnn.reshape(
                ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_t, tw["dt_bias"]), memory_config=_L1), (B, 1, Nv)
            )
            ttnn.deallocate(a_t)

            # Defer the recurrence: collect this token's inputs. conv_states slot capture stays
            # per-token here (the shift-register is inherently sequential); rec_state slots come
            # from the fused call's per-token output below.
            q_seq.append(q)
            k_seq.append(k)
            v_seq.append(v)
            beta_seq.append(beta)
            g_seq.append(g)
            if capture:
                _, conv_bufs = self._slot_bufs[t]
                for j, c in enumerate(self.conv_states):
                    ttnn.copy(c, conv_bufs[j])

        # ONE recurrence over all T tokens. q/k/v -> [B,T,Nv,D]; beta/g -> [B,T,Nv]. The wrapper
        # applies the L2-norm + query scale + exp(g) internally (same contract as the per-token op).
        def _stack(seq, d):
            if T == 1:
                return seq[0]
            cat = ttnn.concat(seq, dim=1, memory_config=mc)
            for x in seq:
                ttnn.deallocate(x)
            return cat

        q_all = _stack(q_seq, Dk)
        k_all = _stack(k_seq, Dk)
        v_all = _stack(v_seq, Dv)
        beta_all = _stack(beta_seq, Nv)
        g_all = _stack(g_seq, Nv)
        o_all, states = fused_recurrent_gated_delta_rule_ttnn(
            q_all,
            k_all,
            v_all,
            beta_all,
            g_all,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            output_per_token_state=capture,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        ttnn.deallocate(q_all)
        ttnn.deallocate(k_all)
        ttnn.deallocate(v_all)
        ttnn.deallocate(beta_all)
        ttnn.deallocate(g_all)
        # Advance durable rec_state to the last token's state; keep the per-token states for slot
        # acceptance. The kernel writes them token-major, so slot t is a contiguous row block of
        # `states` and NOTHING has to be copied here: hold the tensor and let commit_verify_slot
        # slice the one accepted slot. The old code ran T x (slice + reshape + copy + dealloc) per
        # layer — ~770 device ops per verify across 48 GDN layers, for state that is thrown away
        # for every slot except the accepted one.
        if capture:
            self._verify_states = self._verify_states_buf = states  # [B,T,Nv,Dk,Dv]
            last = ttnn.reshape(ttnn.slice(states, (0, T - 1, 0, 0, 0), (B, T, Nv, Dk, Dv)), (B, Nv, Dk, Dv))
            ttnn.copy(last, self.rec_state)
            ttnn.deallocate(last)
        else:
            ttnn.copy(states, self.rec_state)
            ttnn.deallocate(states)
        # Per-token gated-norm to build out_f_rows (identical to the sequential tail).
        for t in range(T):
            o_t = ttnn.reshape(ttnn.slice(o_all, (0, t, 0, 0), (B, t + 1, Nv, Dv)), (B, Nv, Dv))
            out_n = ttnn.rms_norm(o_t, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o_t)
            out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
            ttnn.deallocate(out_n)
            out_f_rows.append(ttnn.to_layout(out_f, rm))
            ttnn.deallocate(out_f)
        ttnn.deallocate(o_all)

        ttnn.deallocate(qkv_all)
        ttnn.deallocate(a_all)
        ttnn.deallocate(b_all)

        # 3) Output tail: one gated SiLU + one out-proj + one all-reduce over the valid rows.
        if T == 1:
            out_f_b = ttnn.to_layout(out_f_rows[0], ttnn.TILE_LAYOUT)
            for r in out_f_rows:
                ttnn.deallocate(r)
        else:
            cat = ttnn.concat(out_f_rows, dim=1, memory_config=mc)  # [1, T, value_dim_tp] ROW_MAJOR
            out_f_b = ttnn.to_layout(cat, ttnn.TILE_LAYOUT)
            ttnn.deallocate(cat)
            for r in out_f_rows:
                ttnn.deallocate(r)
        # z_all is already [1, T, value_dim_tp] (projected over exactly the valid rows).
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        o_red = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=mc,
        )
        # Pad the valid rows out to the full bucket (rows >= valid_len are never read downstream).
        if T < bucket:
            o_rm = ttnn.to_layout(o_red, rm)
            ttnn.deallocate(o_red)
            # Trace-safe pad: ttnn.zeros is a host write that TT_FATALs inside a captured trace, so use
            # a PERSISTENT zero buffer (allocated once, fixed address) when verify is being traced. The
            # values are identical to ttnn.zeros; only the allocation site differs.
            pad = self._verify_pad_buf(bucket - T, o_rm.shape[-1], o_rm.dtype, rm, mc)
            o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
            ttnn.deallocate(o_rm)
            o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
            ttnn.deallocate(o_full)
        return o_red

    def _verify_fullbatch(self, qkv_all, z_all, a_all, b_all, T, bucket, capture):
        """Fully-batched hybrid verify: NO per-token loop. See the use_fullbatch_verify note above.

        Inputs are the already-projected [1,T,*] tensors. Returns the same padded [1,1,bucket,dim]
        the per-token path returns, and advances conv_states / rec_state identically.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        kd, C, rf = self.key_dim_tp, self.qkv_dim_tp, Nv // Nk
        self._ensure_conv_win()  # no-op after the first (eager) call; never allocates inside a trace
        self._ensure_kda_consts()  # same: allocated on the eager seed, before any trace capture

        # 1) Causal conv over all T tokens. The carry is the shift register's previous K-1 inputs,
        #    i.e. conv_states[1:] (conv_states[K-1] is the most recent input). Constant-offset slice
        #    of the persistent shift register — trace-safe (fixed address, fixed offsets) and what
        #    lets commit write ONE buffer.
        carry = ttnn.slice(self._conv_win_buf, (0, 1, 0), (1, self.K, C))
        # Window E = [carry(K-1) ; tokens(T)]: the state stash in step 5 slices commit windows out of it.
        E = ttnn.concat([carry, qkv_all], dim=1, memory_config=mc)  # [1, K-1+T, C]
        # The KDA op (conv + SiLU + q/k/v split) needs tile-aligned rows: right-pad the T tokens to Tp.
        # The conv is causal, so rows [0, T) are exact and the padded tail is sliced off below.
        Tp = -(-T // tpc.TILE_SIZE) * tpc.TILE_SIZE
        xin = qkv_all if Tp == T else ttnn.pad(qkv_all, [(0, 0), (0, Tp - T), (0, 0)], 0.0)
        q_all, k_all, v_all, _ = kda_conv_prefill(
            xin, Tp, carry, tw["conv_taps"], (kd, kd, self.value_dim_tp), self._kda_actual_start, emit_state=False
        )
        ttnn.deallocate(carry)
        if xin is not qkv_all:
            ttnn.deallocate(xin)

        # 2) q/k/v for all T: drop the padded rows, then one repeat_interleave each.
        q_all = ttnn.reshape(ttnn.slice(q_all, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k_all = ttnn.reshape(ttnn.slice(k_all, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        v_all = ttnn.reshape(ttnn.slice(v_all, (0, 0, 0), (1, T, self.value_dim_tp)), (1, T, Nv, Dv))
        if rf != 1:
            q_all = ttnn.repeat_interleave(q_all, rf, dim=2)
            k_all = ttnn.repeat_interleave(k_all, rf, dim=2)

        # 3) Gating for all T at once (a_all/b_all are already [1,T,Nv]).
        beta_all = ttnn.sigmoid(b_all, memory_config=_L1)
        g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_all, tw["dt_bias"]), memory_config=_L1)

        # 4) ONE recurrence dispatch over all T tokens, emitting per-token state when capturing.
        o_all, states = fused_recurrent_gated_delta_rule_ttnn(
            q_all,
            k_all,
            v_all,
            beta_all,
            g_all,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            output_per_token_state=capture,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        for t in (q_all, k_all, v_all, beta_all, g_all):
            ttnn.deallocate(t)

        # 5) State bookkeeping. rec_state slots come from the kernel's per-token output; conv slots
        #    are rows [t, t+K) of the window E built in step 1, stashed with ONE copy.
        if capture:
            ttnn.copy(E, self._verify_win_buf)
            self._win_captured = True
            # Token-major kernel output: keep it and let commit_verify_slot slice the accepted slot.
            self._verify_states = self._verify_states_buf = states
            last = ttnn.reshape(ttnn.slice(states, (0, T - 1, 0, 0, 0), (self.B, T, Nv, Dk, Dv)), (self.B, Nv, Dk, Dv))
            ttnn.copy(last, self.rec_state)
            ttnn.deallocate(last)
        else:
            ttnn.copy(states, self.rec_state)
            ttnn.deallocate(states)
        # Durable shift register = the window's last K rows (what T sequential shifts would leave),
        # written to the persistent [1,K,C] buffer the next replay's carry reads. ONE slice + copy.
        #
        # The K conv_states taps are deliberately NOT refilled here. Nothing in the verify loop reads
        # them, and doing it inside the trace cost K x (slice + copy) per layer x 48 layers ~= 10 ms
        # of the iteration. They are rebuilt from this window on demand instead (sync_conv_taps), at
        # every consumer: the snapshot/restore round-trip, forward_decode's shift register, and the
        # end of a spec generate. `_conv_taps_stale` below is what arms that — and note it is a HOST
        # side effect, so it only fires on the eager passes; the per-REPLAY marking lives in
        # model.verify_traced (python does not re-run during execute_trace).
        tail = ttnn.slice(E, (0, T - 1, 0), (1, T - 1 + self.K, C))
        ttnn.copy(tail, self._conv_win_buf)
        if T > 1:
            # At T == 1 (the one-token eager seed) E is exactly K rows, so that slice is FULL-SPAN
            # and ttnn.slice hands back an ALIAS of E — deallocating it would double-free with E.
            ttnn.deallocate(tail)
        ttnn.deallocate(E)
        self._conv_taps_stale, self._conv_win_stale = True, False

        # 6) Batched output tail: rms_norm normalises over the last dim, so one call covers all T.
        out_n = ttnn.rms_norm(ttnn.reshape(o_all, (T, Nv, Dv)), weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(o_all)
        out_f_b = ttnn.reshape(out_n, (1, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        o_red = tt_all_reduce(
            partial, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(), memory_config=mc
        )
        ttnn.deallocate(qkv_all)
        ttnn.deallocate(a_all)
        ttnn.deallocate(b_all)
        if T < bucket:
            o_rm = ttnn.to_layout(o_red, rm)
            ttnn.deallocate(o_red)
            pad = self._verify_pad_buf(bucket - T, o_rm.shape[-1], o_rm.dtype, rm, mc)
            o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
            ttnn.deallocate(o_rm)
            o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
            ttnn.deallocate(o_full)
        return o_red

    def _verify_pad_buf(self, rows, width, dtype, layout, mc):
        """Persistent zero pad for the trace-safe verify output (bucket - valid_len rows). Allocated
        once per (rows,width,dtype) at a fixed address so the pad concat is trace-capturable; ttnn.zeros
        allocates a fresh buffer each call, which is a host write that TT_FATALs inside a captured trace.
        First call (verify warmup, before begin_trace_capture) allocates; later calls reuse the buffer."""
        cache = getattr(self, "_verify_pad_cache", None)
        if cache is None:
            cache = self._verify_pad_cache = {}
        key = (rows, width, dtype, layout)
        buf = cache.get(key)
        if buf is None:
            buf = ttnn.zeros([1, 1, rows, width], device=self.mesh, dtype=dtype, layout=layout, memory_config=mc)
            cache[key] = buf
        return buf

    def _ensure_verify_slot_bufs(self, n):
        """Lazily allocate n persistent per-token slot buffers (rec_state + conv_states shaped). Verify
        copies state INTO these fixed addresses so a captured trace replays without allocating (clones
        would mint new addresses each call). Reallocated only if fewer than n slots exist."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        # Conv window buffer, ONLY for the fully-batched conv path: [1, K-1+n, qkv_dim_tp], matching
        # the concat that builds E. Allocated HERE (before the trace warmup) so the warmup and
        # captured passes take the identical ttnn.copy branch — a lazy allocate-then-copy makes
        # capture hit an uncompiled program. Gated so the per-token A/B path allocates nothing.
        if self.use_fullbatch_verify:
            _wrows = self.K - 1 + n
            if self._verify_win_buf is None or self._verify_win_buf.shape[-2] != _wrows:
                if self._verify_win_buf is not None:
                    ttnn.deallocate(self._verify_win_buf)
                self._verify_win_buf = ttnn.zeros(
                    [1, _wrows, self.qkv_dim_tp],
                    device=self.mesh,
                    dtype=self.conv_states[0].dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mc,
                )
            self._ensure_conv_win()
        if self._slot_bufs is not None and len(self._slot_bufs) >= n:
            return
        if self._slot_bufs is not None:
            for rec, convs in self._slot_bufs:
                ttnn.deallocate(rec)
                for c in convs:
                    ttnn.deallocate(c)
        self._slot_bufs = [
            (ttnn.clone(self.rec_state, memory_config=mc), [ttnn.clone(c, memory_config=mc) for c in self.conv_states])
            for _ in range(n)
        ]

    def _ensure_conv_win(self):
        """Allocate the persistent [1, K, qkv_dim_tp] shift register once, seeded from conv_states.

        Allocate-and-seed happens on the FIRST eager fullbatch verify (the spec loop's seed forward),
        never lazily inside a captured trace: by capture time the buffer exists and the trace body
        only ever slices/copies it, which the warmup pass has already compiled.

        Re-seeds an EXISTING buffer whose taps moved underneath it (`_conv_win_stale`: a new prompt's
        prefill, a reset, a plain decode step). That re-seed likewise only ever happens on an eager
        call — by capture time the flag is clear (the pre-capture _restore_gdn_verify syncs), so the
        trace body records no copy and a replay can never clobber the window with stale taps."""
        if self._conv_win_buf is None:
            self._conv_win_buf = ttnn.zeros(
                [1, self.K, self.qkv_dim_tp],
                device=self.mesh,
                dtype=self.conv_states[0].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._conv_win_stale = True
        if self._conv_win_stale:
            self.sync_conv_win()

    def sync_conv_taps(self):
        """Rebuild conv_states[0..K-1] from the persistent window — the inverse of sync_conv_win.

        The traced fullbatch verify advances ONLY _conv_win_buf (see _verify_fullbatch), so the K tap
        buffers go stale for as long as nothing reads them. Every tap CONSUMER calls this first;
        it is a no-op — zero device ops, so calling it from a traced body is safe — unless a
        fullbatch verify actually ran since the taps were last written.

        K x (slice + copy) per layer, eager, and only on the handful of iterations that read taps
        (snapshot, an eager decode step, the end of a generate) instead of every verify replay."""
        if not self._conv_taps_stale:
            return
        self._conv_taps_stale = False
        if self._conv_win_buf is None or self.conv_states is None:
            return
        C = self.qkv_dim_tp
        for j in range(self.K):
            row = ttnn.slice(self._conv_win_buf, (0, j, 0), (1, j + 1, C))
            ttnn.copy(row, self.conv_states[j])
            ttnn.deallocate(row)

    def sync_conv_win(self):
        """Mirror conv_states[0..K-1] into the persistent [1,K,qkv_dim_tp] shift-register buffer.

        conv_states stay the source of truth outside the spec loop (prefill fills them, snapshot /
        restore round-trips them, the decode path reads them); this buffer is the copy the traced
        fullbatch verify reads its carry from, so it has to be re-seeded whenever conv_states are
        set from outside — at slot-buffer setup and after every _restore_gdn_verify."""
        if self._conv_win_buf is None or self.conv_states is None:
            return
        if self._conv_taps_stale:
            # The WINDOW is the truth here (a fullbatch verify advanced it and the taps were left
            # behind), so copying the taps over it would undo the verify. Bring the taps forward
            # instead; both mirrors then agree and there is nothing left to copy.
            self.sync_conv_taps()
            self._conv_win_stale = False
            return
        c = ttnn.concat(list(self.conv_states), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(c, self._conv_win_buf)
        ttnn.deallocate(c)
        self._conv_win_stale = False

    def commit_verify_slot(self, idx):
        """Set the recurrent state to the buffered verify slot `idx` (state after consuming the
        accepted-prefix's last token). Copies in place (preserves buffer addresses), so no commit
        forward runs. Slot buffers are persistent (reused next verify), so they are NOT freed here.

        The recurrent state comes straight out of the kernel's token-major per-token output, so only
        the ONE accepted slot is ever touched — the verify no longer copies all T states aside.

        COST NOTE. This is the one EAGER step of the spec iteration that scales with layer count: 48
        GDN layers x a handful of device ops, at this stack's ~35 us eager op cost. The ops move a
        few KB each — it is dispatch count, not data — so the only thing that matters here is how
        many ttnn calls the loop makes. On the fully-batched verify it used to make ten per layer
        (rec slice+copy, then K conv taps x slice+copy+deallocate) and cost ~17 ms, ~3x its per-token
        self, whose trace had already materialised the per-token conv slots. Two changes took it to
        ~5 ms: the full-acceptance early-out below, and committing the conv shift register as ONE
        [1,K,qkv_dim_tp] buffer (_conv_win_buf) that the traced verify reads its carry from, instead
        of K separate conv_states taps."""
        assert self._verify_states is not None, "commit_verify_slot called without a captured verify"
        # A verify (traced or eager) just advanced the window; the taps are behind either way, so
        # arm the rebuild BEFORE the full-acceptance early-out below.
        if self._win_captured:
            self._conv_taps_stale, self._conv_win_stale = True, False
        if idx == self._verify_states.shape[1] - 1:
            # Full acceptance: the verify already LEFT the durable state at the last token (rec_state
            # = states[T-1], _conv_win_buf = window rows [T-1, T-1+K)), so every copy below would write
            # what is already there. 48 layers x ~6 device ops saved on those iterations.
            self._verify_states = None
            return
        st = ttnn.reshape(
            ttnn.slice(self._verify_states, (0, idx, 0, 0, 0), (self.B, idx + 1, self.Nv, self.Dk, self.Dv)),
            (self.B, self.Nv, self.Dk, self.Dv),
        )
        ttnn.copy(st, self.rec_state)
        ttnn.deallocate(st)
        convs = self._slot_bufs[idx][1] if self._slot_bufs is not None else None
        if self._win_captured:
            # Batched-conv path: conv slots were not materialised per token. The shift-register as of
            # token idx is rows [idx, idx+K) of the stashed window (see _forward_verify_recurrent_batched).
            #
            # ONE slice + ONE copy into the persistent shift register the next replay's carry reads.
            # This used to write the K conv_states taps individually (K x slice+copy+deallocate per
            # layer, x48 layers) and that dispatch count WAS the fb=1 commit regression. conv_states
            # themselves are left at the end-of-window state the trace wrote; nothing in the spec
            # loop reads them (see sync_conv_win).
            w = ttnn.slice(self._verify_win_buf, (0, idx, 0), (1, idx + self.K, self.qkv_dim_tp))
            ttnn.copy(w, self._conv_win_buf)
            ttnn.deallocate(w)
        else:
            for j, c in enumerate(convs):
                ttnn.copy(c, self.conv_states[j])
            self._conv_taps_stale, self._conv_win_stale = False, True
        self._verify_states = None

    # ------------------------------------------------------------------ #
    # Traced commit
    # ------------------------------------------------------------------ #
    # commit_verify_slot is the one eager step that scales with layer count (4 ops x 48 layers), almost all host launch over a few KB.
    # Ops are identical every iteration for a given accepted-prefix index because every tensor is persistent: read _verify_states_buf (verify-trace per-token state; handle re-armed each verify_traced replay, capture-allocated BUFFER address is fixed), read _verify_win_buf (_ensure_verify_slot_bufs), write rec_state (in-place ttnn.copy), write _conv_win_buf (_ensure_conv_win). Capture once per idx and replay — still dispatches those programs on device, but stops paying host launch.
    # commit_verify_slot_ops is the device half (trace body); commit_verify_slot_host is the host half (staleness flags + dropping the verify handle) and still runs every iteration.

    def commit_verify_slot_ops(self, idx):
        """Device half of commit_verify_slot(idx): the ops a commit trace captures.

        Byte-identical work to commit_verify_slot's device ops on the batched-conv (window) path,
        minus the full-acceptance early-out (the caller skips idx == T-1 entirely) and minus every
        host side effect, so a replay of this body is exactly what the eager commit would have done.

        Reads self._verify_states_buf rather than self._verify_states: the latter is nulled by each
        commit and re-armed by verify_traced, while the former is the stable handle on the buffer
        whose address the trace bakes in.
        """
        assert self._verify_states_buf is not None, "traced commit needs a captured verify"
        assert self._win_captured, "traced commit is the batched-conv (window) verify path only"
        st = ttnn.reshape(
            ttnn.slice(self._verify_states_buf, (0, idx, 0, 0, 0), (self.B, idx + 1, self.Nv, self.Dk, self.Dv)),
            (self.B, self.Nv, self.Dk, self.Dv),
        )
        ttnn.copy(st, self.rec_state)
        ttnn.deallocate(st)
        w = ttnn.slice(self._verify_win_buf, (0, idx, 0), (1, idx + self.K, self.qkv_dim_tp))
        ttnn.copy(w, self._conv_win_buf)
        ttnn.deallocate(w)

    def commit_verify_slot_host(self, idx):
        """Host half of commit_verify_slot(idx): the staleness marks and the verify handle drop.

        Runs every iteration whether the device half was eager or a trace replay — python does not
        re-run inside execute_trace, so these marks have to be set from here either way.
        """
        assert self._verify_states is not None, "commit_verify_slot called without a captured verify"
        if self._win_captured:
            self._conv_taps_stale, self._conv_win_stale = True, False
        self._verify_states = None

    def _traced_commit_blockers(self):
        """Which traced-commit preconditions this layer fails, as a list of names. Empty == ready."""
        return [
            name
            for name, ok in (
                ("use_fullbatch_verify", self.use_fullbatch_verify),
                ("_win_captured", self._win_captured),
                ("_verify_states_buf", self._verify_states_buf is not None),
                ("_verify_win_buf", self._verify_win_buf is not None),
                ("_conv_win_buf", self._conv_win_buf is not None),
                ("rec_state", self.rec_state is not None),
            )
            if not ok
        ]

    def traced_commit_ready(self):
        """Whether this layer's commit can be traced: the batched-conv verify window has to be live
        and every buffer the commit ops touch has to be a persistent (fixed-address) one."""
        return not self._traced_commit_blockers()

    def traced_commit_why(self):
        """Human-readable reason traced_commit_ready() is False (for the capture-time log)."""
        return ", ".join(self._traced_commit_blockers()) or "ready"
