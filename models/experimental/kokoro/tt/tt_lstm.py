# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN LSTM helpers (TT-prefixed params, ``tt_`` entrypoints).

Host-driven timestep loop using ``ttnn.linear`` and activations (no high-level LSTM op).
The perf report's many slice ops are L per-step copies (one per timestep) — the LSTM
recurrence is serial (step t reads step t-1's state) so it can't be batched over time.
See the per-slice comments below; the only wall-clock lever is tracing, not cutting slices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch

import ttnn

from .tt_trace_prep import (
    prep_cache_get as _prep_cache_get,
    prep_cache_set as _prep_cache_set,
    trace_weight_prep_enabled as _trace_weight_prep_enabled,
    traced_zeros as _traced_zeros,
)

_TILE = 32
# Per-step LSTM state/gate tensors are tiny at bring-up widths (H<=64); keep them in L1 when
# tile-padded storage fits (gx precompute buffers stay on the caller's memory_config).
# At Kokoro-82M H=256 the recurrent ``h @ W_h^T`` matmul CB footprint clashes with L1
# tensor allocations on BH (see kmodel L1 circular-buffer overlap at ~137 KiB).
_L1_STEP_BUDGET_BYTES = 512 * 1024
_L1_STEP_HIDDEN_MAX = 64
# Direction-fused loop only: max ``[L, B, 2H]`` output-accumulation footprint kept on L1-interleaved
# per-step tensors (the sole L-scaling term; the rest of the working set is fixed-size). 4 MiB ≈
# L<=128 at B=2/H=256 — verified to fit BH L1 with margin (tested at L=48 ~1.5 MiB); longer
# sequences fall back to DRAM. Only consulted when a tuned recurrent program config is supplied.
_FUSED_L1_ACCUM_BUDGET_BYTES = 4 * 1024 * 1024
# Per-direction (non-fused) fp32 path P1+P2: max ``2 * L * [B,H]`` output-accumulation footprint kept
# on L1 (the two direction output lists are the sole L-scaling term; the [L,B,4H] gx buffer stays
# DRAM, only its per-step [B,4H] slice is L1). 8 MiB ≈ L<=128 at B=1/H=256 fp32 — F0Ntrain has ample
# free L1 (decoder not yet resident); longer sequences fall back to the DRAM per-step path. Tunable.
_PERDIR_L1_ACCUM_BUDGET_BYTES = 8 * 1024 * 1024


def _dtype_nbytes(dtype) -> int:
    return 4 if dtype == ttnn.float32 else 2


def _tile_padded_volume(shape: tuple[int, ...] | list[int]) -> int:
    n = 1
    for d in shape:
        n *= math.ceil(int(d) / _TILE) * _TILE
    return n


def _tensor_nbytes(shape: tuple[int, ...] | list[int], dtype) -> int:
    return _tile_padded_volume(shape) * _dtype_nbytes(dtype)


def _lstm_step_memory_config(
    *,
    batch: int,
    hidden: int,
    dtype,
    fp32_state: bool,
    fallback: ttnn.MemoryConfig,
) -> ttnn.MemoryConfig:
    """L1-interleaved for per-step ops when small enough (see ``tt_matmul_memory`` sweep note)."""
    if hidden > _L1_STEP_HIDDEN_MAX:
        return fallback
    state_dtype = ttnn.float32 if fp32_state else dtype
    peak = 3 * _tensor_nbytes((batch, 4 * hidden), dtype) + 4 * _tensor_nbytes((batch, hidden), state_dtype)
    if peak <= _L1_STEP_BUDGET_BYTES:
        return ttnn.L1_MEMORY_CONFIG
    return fallback


@dataclass(frozen=True)
class TTLSTMParams:
    """One-direction LSTM gate weights for ``x @ W_x + h @ W_h + b``.

    ``W_*`` are stored **pre-transposed** (``[in, 4H]``) so the matmuls run with the default
    ``transpose_b=False``. PyTorch's ``weight_*h`` are ``[4H, in]`` (laid out for ``x @ W^T``);
    transposing once at upload avoids re-transposing ``W_h`` on every recurrent timestep (one
    ``TransposeDeviceOperation``/step otherwise).
    """

    w_x: ttnn.Tensor
    w_h: ttnn.Tensor
    b: ttnn.Tensor
    hidden_size: int


def preprocess_tt_lstm_1layer(
    lstm: torch.nn.LSTM,
    mesh_device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> Tuple[TTLSTMParams, Optional[TTLSTMParams]]:
    """
    Convert a 1-layer PyTorch ``nn.LSTM`` into TTNN gate tensors.

    Returns ``(forward_params, reverse_params_or_none)``.
    """
    assert lstm.num_layers == 1, "Only 1-layer LSTM supported in bring-up helper"
    hidden_size = lstm.hidden_size

    def pack(prefix: str) -> TTLSTMParams:
        w_ih = getattr(lstm, f"weight_ih_{prefix}").detach().cpu()
        w_hh = getattr(lstm, f"weight_hh_{prefix}").detach().cpu()
        b_ih = getattr(lstm, f"bias_ih_{prefix}").detach().cpu()
        b_hh = getattr(lstm, f"bias_hh_{prefix}").detach().cpu()
        b = (b_ih + b_hh).reshape(1, 1, 1, -1)

        # Store transposed (PyTorch gives [4H, in]; we want [in, 4H] for transpose_b=False).
        w_x = ttnn.from_torch(
            w_ih.t().contiguous(),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w_h = ttnn.from_torch(
            w_hh.t().contiguous(),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b_tt = ttnn.from_torch(
            b, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return TTLSTMParams(w_x=w_x, w_h=w_h, b=b_tt, hidden_size=hidden_size)

    fwd = pack("l0")
    rev = pack("l0_reverse") if lstm.bidirectional else None
    return fwd, rev


def build_fused_recurrent_weight(
    lstm: torch.nn.LSTM,
    mesh_device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> Optional[ttnn.Tensor]:
    """Block-diagonal recurrent weight that fuses both BiLSTM directions into one matmul.

    The forward and reverse passes share no recurrent state, so a single step that advances
    both at once is exact: stack the two hidden states into ``h_comb = [h_f | h_r]`` ``[B, 2H]``
    and multiply by a ``[2H, 8H]`` block-diagonal weight whose off-diagonal blocks are zero.
    The zeros contribute exactly ``0.0`` to the fp32 matmul accumulation, so the fused gates are
    bit-identical to two separate ``h @ W_h`` matmuls — but the loop runs one matmul / sigmoid /
    add / set-of-slices per step instead of two (see :func:`tt_bilstm_nlc`).

    Output columns are laid out **gate-major, direction-minor**:
    ``[i_f i_r f_f f_r g_f g_r o_f o_r]`` (each ``H`` wide). This keeps the per-step gate slices
    contiguous over both directions (``i = W[:, 0:2H]`` etc.), so the cell math runs once over
    ``[B, 2H]`` tensors. ``gates_x`` must be interleaved into the same order (see ``tt_bilstm_nlc``).

    Returns ``None`` for a unidirectional LSTM (nothing to fuse).
    """
    if not lstm.bidirectional:
        return None
    H = lstm.hidden_size
    wf = getattr(lstm, "weight_hh_l0").detach().cpu().t().contiguous()  # [H, 4H], cols [i f g o]
    wr = getattr(lstm, "weight_hh_l0_reverse").detach().cpu().t().contiguous()  # [H, 4H]
    W = torch.zeros(2 * H, 8 * H, dtype=wf.dtype)
    for gi in range(4):  # gate order i, f, g, o
        # forward gate gi -> column group 2*gi (rows for h_f), reverse gate gi -> group 2*gi+1 (rows for h_r)
        W[0:H, (2 * gi) * H : (2 * gi + 1) * H] = wf[:, gi * H : (gi + 1) * H]
        W[H : 2 * H, (2 * gi + 1) * H : (2 * gi + 2) * H] = wr[:, gi * H : (gi + 1) * H]
    return ttnn.from_torch(
        W,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _lstm_step(
    gates_x: ttnn.Tensor,
    h: ttnn.Tensor,
    c: ttnn.Tensor,
    params: TTLSTMParams,
    *,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    fold_bias: bool = False,
    program_config=None,
    out_dtype=None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    # ``gates_x`` (= x_t @ W_x^T + b, shape [B, 4H]) is precomputed for the whole sequence
    # by the caller (one batched matmul with the bias folded into its epilogue, see
    # tt_bilstm_nlc) — it doesn't depend on the recurrent state. Only the recurrent
    # ``h @ W_h^T`` is per-step, and the gate sum is then a single add.
    #
    # ``program_config`` (P2, L1 fp32 per-direction path only) bounds the recurrent matmul CBs so
    # it can run L1-resident at H=256 without the CB-overlap that otherwise forces DRAM. Its output
    # is width-sharded over the config's grid; pin the dtype to the state dtype (``out_dtype``) so
    # the L1 placement is numerics-neutral — ttnn.matmul otherwise defaults a *sharded* fp32 output
    # to bf16, which would silently downcast the F0-sensitive fp32 cell-state chain. The gate ``add``
    # below writes ``memory_config`` (interleaved L1), absorbing the shard->interleaved relayout into
    # its own kernel (no extra per-step reshard op).
    _mm_mem = memory_config
    _mm_dtype = None
    if program_config is not None:
        _grid = program_config.compute_with_storage_grid_size
        _padded_m = ((int(h.shape[0]) + _TILE - 1) // _TILE) * _TILE
        _mm_mem = ttnn.create_sharded_memory_config(
            (_padded_m, int(params.w_h.shape[-1])),  # [pad(B), 4H] width-sharded over the matmul grid
            core_grid=ttnn.CoreGrid(y=_grid.y, x=_grid.x),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        _mm_dtype = out_dtype or h.dtype
    if fold_bias:
        # Fold gates_x into the recurrent matmul bias epilogue: gates = h@W_h + gates_x in one
        # fp32 epilogue add — drops one BinaryNg/step AND is closer to the torch fp32 reference
        # (one rounding vs the separate bf16 matmul + bf16 add). Valid only for B==1 (gates_x
        # [1,4H] is the matmul bias row).
        gates = ttnn.linear(
            h,
            params.w_h,
            bias=gates_x,
            memory_config=_mm_mem,
            dtype=_mm_dtype,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )
    else:
        gates_h = ttnn.linear(
            h,
            params.w_h,
            bias=None,
            memory_config=_mm_mem,
            dtype=_mm_dtype,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )
        gates = ttnn.add(gates_x, gates_h, memory_config=memory_config)
        ttnn.deallocate(gates_h)

    gs = tuple(int(s) for s in gates.shape)
    if len(gs) > 2:
        batch_leading = math.prod(gs[:-1])
        gates = ttnn.reshape(gates, [batch_leading, gs[-1]], memory_config=memory_config)

    H4 = gates.shape[-1]
    H = params.hidden_size
    assert H4 == 4 * H

    # Per-step gate split (4 slices/step). Gate order is [i, f, g, o] with i,f,o -> sigmoid
    # and g -> tanh. One sigmoid over the whole [B, 4H] tensor + slicing i/f/o is bit-identical
    # to three separate sigmoids but one op instead of three (g takes tanh on the raw slice).
    # The slices are the price of fusing the activation — dropping them re-adds ops.
    sig = ttnn.sigmoid(gates, memory_config=memory_config)
    i = ttnn.slice(sig, [0, 0], [sig.shape[0], H], [1, 1])
    f = ttnn.slice(sig, [0, H], [sig.shape[0], 2 * H], [1, 1])
    o = ttnn.slice(sig, [0, 3 * H], [sig.shape[0], 4 * H], [1, 1])
    ttnn.deallocate(sig)
    g_raw = ttnn.slice(gates, [0, 2 * H], [gates.shape[0], 3 * H], [1, 1], memory_config=memory_config)

    # c_new = f*c + i*tanh(g). Kept as separate mul/mul/add: the fused addcmul (MAC) changes
    # the cell-state accumulation, which feeds the F0 curve (amplified ~1885x by the
    # vocoder) and dropped end-to-end PCC below the 0.84 floor (0.855 -> 0.825). The two
    # tanh activations are instead folded into their consuming multiply via the operand
    # activation kwarg (same SFPU tanh, one kernel instead of two) — this leaves the
    # add's accumulation order untouched, so it is bit-identical, and drops 2 unary
    # ops/timestep (and their host-dispatch gap).
    c_new = ttnn.add(
        ttnn.multiply(f, c, memory_config=memory_config),
        ttnn.multiply(g_raw, i, input_tensor_a_activations=[ttnn.UnaryOpType.TANH], memory_config=memory_config),
        memory_config=memory_config,
    )
    h_new = ttnn.multiply(c_new, o, input_tensor_a_activations=[ttnn.UnaryOpType.TANH], memory_config=memory_config)
    return h_new, c_new


def _lstm_step_fused(
    gates_x: ttnn.Tensor,
    h: ttnn.Tensor,
    c: ttnn.Tensor,
    w_h_block: ttnn.Tensor,
    H: int,
    *,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    fold_bias: bool = False,
    program_config=None,
    fuse_cell_math: bool = False,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """One fused BiLSTM step advancing both directions at once.

    ``h``/``c`` are the stacked states ``[h_f | h_r]`` / ``[c_f | c_r]`` shaped ``[B, 2H]``;
    ``gates_x`` is the precomputed ``[B, 8H]`` input projection in gate-major direction-minor
    order ``[i_f i_r f_f f_r g_f g_r o_f o_r]`` and ``w_h_block`` is the block-diagonal recurrent
    weight from :func:`build_fused_recurrent_weight`. Identical cell math to :func:`_lstm_step`,
    just run once over ``2H``-wide tensors instead of twice over ``H``-wide ones.

    ``program_config`` (optional) overrides the recurrent ``h @ W_h_block`` matmul's program config.
    A tuned 1D mcast config beats the default for the ``[B, 2H] @ [2H, 8H]`` shape on Blackhole
    (~8 vs ~11 µs at H=256, PCC unchanged) while keeping ``in0``/``out`` interleaved — i.e. no
    per-step reshard. See :func:`models.experimental.kokoro.tt.tt_text_encoder` and the matmul sweep.
    """
    # Emit the recurrent matmul output L1 width-sharded whenever the tuned program config is active
    # (the validated H=256 / 8x8 TextEncoder shape — _fused_recurrent_program_config returns None for
    # any other BiLSTM, so this never touches the F0-sensitive prosody LSTMs). This is the fastest
    # bf16 sweep config for [B,2H]@[2H,8H] (1D_in0 l1/dram/ws: ~7.7 vs 8.1 µs isolated). The output
    # tiles 1-per-core over the matmul's own compute grid (per_core_N=1), and the immediate consumer
    # below (sigmoid, memory_config-interleaved) absorbs the shard->interleaved relayout into its own
    # kernel — so no extra per-step reshard op is added (verified in the perf report).
    _mm_mem = memory_config
    # Preserve the matmul's output dtype across the L1-sharded path. ttnn.matmul defaults a *sharded*
    # output to bfloat16 even when the inputs are fp32 (an interleaved output instead preserves the
    # input dtype), which would silently downcast the whole downstream cell-state chain to bf16 — a
    # precision change the F0-sensitive prosody BiLSTM rejects, and a dtype mismatch against the
    # fp32 anti-identity reorder at the final concat. Pin the output to the state dtype so the sharding
    # is a pure placement change (bit-identical numerics), not a dtype switch.
    _mm_dtype = None
    if program_config is not None:
        _grid = program_config.compute_with_storage_grid_size
        # Tile-pad the height (B -> ceil to 32): the matmul tile-pads B internally and computes a
        # [32, 8H/cores] shard, so passing the logical B (e.g. 2) yields a [2, ...] spec that mismatches
        # and gets silently overridden ("Using computed config" warning). Padding here matches it exactly.
        _padded_m = ((h.shape[0] + _TILE - 1) // _TILE) * _TILE
        _mm_mem = ttnn.create_sharded_memory_config(
            (_padded_m, w_h_block.shape[-1]),  # [pad(B), 8H], width-sharded over the matmul's grid
            core_grid=ttnn.CoreGrid(y=_grid.y, x=_grid.x),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        _mm_dtype = h.dtype
    if fold_bias:
        # Fold gates_x into the recurrent matmul bias epilogue (one fp32 epilogue add): drops one
        # BinaryNg/step and is closer to the torch fp32 reference. B==1 only (see _lstm_step).
        gates = ttnn.linear(
            h,
            w_h_block,
            bias=gates_x,
            memory_config=_mm_mem,
            dtype=_mm_dtype,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )
    else:
        gates_h = ttnn.linear(
            h,
            w_h_block,
            bias=None,
            memory_config=_mm_mem,
            dtype=_mm_dtype,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )
        # Keep gates_x as operand-a: ttnn binary ops take their output dtype from the first operand.
        # gates_x is the fp32 input projection while gates_h (the recurrent matmul) may be bf16, so
        # operand-a=gates_x keeps ``gates`` fp32 — both bit-faithful to the fp32 reference and dtype-
        # consistent with the fp32 anti-identity output reorder at the final concat (see below).
        gates = ttnn.add(gates_x, gates_h, memory_config=memory_config)
        ttnn.deallocate(gates_h)

    H2 = 2 * H
    sig = ttnn.sigmoid(gates, memory_config=memory_config)
    i = ttnn.slice(sig, [0, 0], [sig.shape[0], H2], [1, 1])
    f = ttnn.slice(sig, [0, H2], [sig.shape[0], 2 * H2], [1, 1])
    o = ttnn.slice(sig, [0, 3 * H2], [sig.shape[0], 4 * H2], [1, 1])
    ttnn.deallocate(sig)
    g_raw = ttnn.slice(gates, [0, 2 * H2], [gates.shape[0], 3 * H2], [1, 1], memory_config=memory_config)

    # Fold tanh(g) and tanh(c_new) into their consuming multiply (operand activation kwarg):
    # same SFPU tanh, one kernel instead of two, leaving the add's accumulation order
    # untouched (bit-identical) — drops 2 unary ops/timestep. See _lstm_step.
    itg = ttnn.multiply(g_raw, i, input_tensor_a_activations=[ttnn.UnaryOpType.TANH], memory_config=memory_config)
    if fuse_cell_math:
        # c_new = f*c + tanh(g)*i collapsed into ONE op: ttnn.addcmul(a,b,c) = a + b*c lowers to a single
        # TernaryDeviceOperation (verified via graph capture), so addcmul(itg, f, c) = itg + f*c replaces
        # the separate mul(f,c) + add — genuinely -1 device op/step. (NOTE: ttnn.mac is NOT used here — it
        # is a python/cpp *composite* that lowers to 2 BinaryNg device ops, i.e. mul+add, giving ZERO
        # reduction; the perf report showed no change with mac. addcmul is the real fused kernel.)
        # The fused MAC changes the cell-state accumulation rounding (one fused fp32 op vs a separate bf16
        # multiply then bf16 add), so it is **opt-in and used ONLY by the ASR TextEncoder**: the
        # prosody/duration BiLSTMs share this step and feed the F0 curve (amplified ~1885x by the
        # vocoder), which rejects any numeric change. itg keeps its tanh folded, so only the cell add moves.
        c_new = ttnn.addcmul(itg, f, c, memory_config=memory_config)
    else:
        c_new = ttnn.add(ttnn.multiply(f, c, memory_config=memory_config), itg, memory_config=memory_config)
    h_new = ttnn.multiply(c_new, o, input_tensor_a_activations=[ttnn.UnaryOpType.TANH], memory_config=memory_config)
    return h_new, c_new


def _gatex_program_config(*, batch: int, seq_len: int, four_hidden: int, in_dim: int, device):
    """Tuned 2D mcast config for the gate-precompute matmul ``[B, L, in] @ [in, 4H]``.

    The sweep (``perf/test_gatex_matmul_sweep.py``) found ``gy=total_m, gx=8, per_core_M=1,
    per_core_N=4, out_subblock=1x4, in0_block_w=8`` fastest for B=2/L=48/H=256: **7.97µs vs the
    default's 17.3µs (-54%)**. Two knobs drove it beyond the first-pass 11.6µs: (1) spreading the
    fuse-batched ``B*ceil(L/32)`` M-tiles ONE-PER-ROW (``gy=total_m, per_core_M=1``, 32 cores) beats
    packing 2/row on 16 cores; (2) a wide ``out_subblock_w=4`` (vs 1x1) packs 4 output tiles per
    compute call. ``gx=8`` splits the 4H output into 8 col-groups (``per_core_N=4`` at H=256),
    ``in0_block_w=8`` is two K-steps. Numerics are config-invariant (out_subblock/grid only reschedule
    tiles; the 1x4 subblock = 4 DST tiles fits the fp32_dest_acc cap, so it is bit-identical to 1x1).
    Guarded to shapes that map cleanly (H%64==0 so 4H-tiles divide 8; K%8==0; a small fuse-batched M
    that fits the core grid one-per-row) — else ``None`` (default), so odd lengths and the F0-sensitive
    prosody/duration BiLSTMs (which don't wire a gate-precompute config) are untouched."""
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        return None
    n_tiles = four_hidden // 32
    k_tiles = in_dim // 32
    total_m = batch * math.ceil(seq_len / _TILE)  # fuse_batch folds B into the M dim
    if four_hidden % 32 or in_dim % 32 or n_tiles % 8 or k_tiles % 8:
        return None
    if not (1 <= total_m <= 8):  # gy=total_m one M-tile/row; bound to the core grid + L1 footprint
        return None
    per_core_n = n_tiles // 8
    # out_subblock_w: largest divisor of per_core_N that is <= 4 (the fp32_dest_acc DST cap); h=1 (pm=1).
    sub_w = next(d for d in (4, 2, 1) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, total_m),  # gx=8, gy=total_m
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        per_core_M=1,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def _perdir_recurrent_program_config(*, batch: int, hidden: int, device):
    """1D-mcast config for the per-direction recurrent matmul ``[B, H] @ [H, 4H]`` (P2).

    Used ONLY by the L1-resident fp32 per-direction path (the shared F0/N BiLSTM). Its sole job is
    to bound the matmul's circular-buffer footprint so the recurrent ``h @ W_h`` can run with L1
    in0/out without the CB-overlap that forces H=256 to DRAM under the default config (see the
    ``_L1_STEP_HIDDEN_MAX`` note). It is **schedule/placement only**: ``in0_block_w`` is the FULL K
    (``hidden//32`` tiles → a single accumulation pass, no K-reordering), and grid/subblock choices
    do not change the math — so it must not perturb the F0-sensitive gate arithmetic. The output is
    width-sharded over the grid (per_core_N tiles/core); the consumer add relayouts it back to
    interleaved (see :func:`_lstm_step`). Returns ``None`` off Blackhole or if 4H-tiles don't map."""
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        return None
    if (4 * hidden) % 32 or hidden % 32:
        return None
    n_tiles = (4 * hidden) // 32
    k_tiles = hidden // 32
    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = int(grid.x), int(grid.y)
    # Largest core grid whose total cores divide n_tiles evenly (per_core_N integer), N along gx.
    best = None
    for gy in range(1, gy_max + 1):
        for gx in range(1, gx_max + 1):
            cores = gx * gy
            if cores <= n_tiles and n_tiles % cores == 0:
                if best is None or cores > best[0] * best[1]:
                    best = (gx, gy)
    if best is None:
        return None
    gx, gy = best
    per_core_n = n_tiles // (gx * gy)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=k_tiles,  # full K in one block: single accumulation pass, no K-reorder
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _length_valid_mask_b1(
    *,
    batch: int,
    seq_len: int,
    sequence_lengths: Sequence[int],
    device,
    memory_config: ttnn.MemoryConfig,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    # Laid out time-major ``[L, B, 1]`` (not ``[B, L, 1]``) so the per-step mask is a leading-dim
    # slice ``valid[t]``. ``L`` is not a tiled dim here, so that slice is a clean page copy; slicing
    # the middle ``L`` of a ``[B, L, 1]`` tile tensor instead forces an untilize+slice+tilize
    # round-trip every masked step (same reason ``gx_comb`` is stored ``[L, B, 4H]``).
    vm = torch.zeros(seq_len, batch, 1, dtype=torch.float32)
    for bi, le in enumerate(sequence_lengths):
        le = max(0, min(int(le), seq_len))
        vm[:le, bi, 0] = 1.0
    return ttnn.from_torch(
        vm,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def tt_bilstm_nlc(
    *,
    x_nlc: ttnn.Tensor,
    fwd: TTLSTMParams,
    rev: TTLSTMParams,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    sequence_lengths: Optional[Sequence[int]] = None,
    fp32_state: bool = False,
    w_h_block: Optional[ttnn.Tensor] = None,
    fold_gates_bias: bool = False,
    recurrent_program_config=None,
    fuse_cell_math: bool = False,
    out_memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    1-layer BiLSTM over sequence for NLC ``[B, L, in]``.

    With ``sequence_lengths``, timesteps ``t >= length[b]`` yield zero output and frozen state
    (``pack_padded_sequence`` semantics). Returns ``[B, L, 2H]``.

    When ``fp32_state`` is True, hidden/cell states are accumulated in fp32 to avoid
    Hz-level drift in long F0/N decoder chains (shared BiLSTM in ``F0Ntrain``).

    ``fold_gates_bias`` folds the per-step ``gates_x`` into the recurrent matmul bias epilogue
    (one fp32 add instead of a separate BinaryNg/step). It alters the gate-sum rounding, so it is
    **opt-in and enabled ONLY on the ASR TextEncoder LSTM** — the duration/F0 BiLSTMs reject any
    numeric change (flipped durations / amplified F0, see the LSTM-gate-matmul notes). Applied only
    when ``B == 1`` (gates_x is one bias row) and ``not fp32_state`` (keeps dtypes aligned).

    ``fuse_cell_math`` folds the per-step ``f*c + tanh(g)*i`` cell-state update into a single
    ``ttnn.mac`` (one fewer BinaryNg/step). Like ``fold_gates_bias`` it alters the gate-math rounding,
    so it is **opt-in and enabled ONLY on the ASR TextEncoder** (direction-fused path); the F0-feeding
    prosody/duration BiLSTMs reject it (see :func:`_lstm_step_fused`).

    ``w_h_block`` (from :func:`build_fused_recurrent_weight`) enables the **direction-fused**
    loop: a single step advances both passes at once over ``[B, 2H]`` state, halving the
    per-timestep matmul / activation / elementwise op (and host-dispatch) count. It is bit-exact
    and used only when no padding mask is needed (``min_len == L``); the padded path falls back
    to the two separate per-direction loops below.
    """
    B, L, C_in = x_nlc.shape
    H = fwd.hidden_size
    H4 = 4 * H
    state_dtype = ttnn.float32 if fp32_state else ttnn.bfloat16
    # ``[L, B, 4H]`` gate projections are much larger than per-step state — keep them DRAM
    # even when the caller passes L1 for pipeline activations.
    gx_mc = ttnn.DRAM_MEMORY_CONFIG if memory_config.buffer_type == ttnn.BufferType.L1 else memory_config
    step_mc = _lstm_step_memory_config(
        batch=B,
        hidden=H,
        dtype=x_nlc.dtype,
        fp32_state=fp32_state,
        fallback=memory_config,
    )
    # Bias-fold the per-step gate add into the recurrent matmul (opt-in, TextEncoder only): drops
    # one BinaryNg/step. Guards: B==1 (gates_x is one bias row), bf16 state (matmul+bias epilogue
    # dtype-consistent), and **DRAM step state** — the matmul-with-bias TT_FATALs when in0/out are
    # L1 (matmul_utilities.cpp), which is why this is unsafe for the L1-state prosody LSTMs (H<=64)
    # and fine for the TextEncoder (H=256 -> step_mc falls back to DRAM).
    do_fold = fold_gates_bias and B == 1 and not fp32_state and step_mc.buffer_type == ttnn.BufferType.DRAM

    # Input gate projections (``x @ W_x^T``) don't depend on the recurrent state, so compute
    # them for the WHOLE sequence in one batched matmul per direction (bit-identical: matmul
    # rows are independent). Permute to ``[L, B, 4H]`` so the per-timestep extraction is a
    # cheap leading-dim slice — this removes the per-step untilize+slice+tilize churn that
    # dominated the loop (slicing a single timestep out of the TILE-laid [B,L,C] sequence).
    def _precompute_gates_x_of(p: TTLSTMParams, x_in: ttnn.Tensor, program_config=None) -> ttnn.Tensor:
        # Fold the gate bias into the matmul epilogue here (once for the whole sequence)
        # instead of adding it per timestep — saves one elementwise add per step. The
        # stored bias is [1,1,1,4H] (rank 4); reshape to [1,1,4H] so the linear output
        # stays rank-3 [B,L,4H] for the permute below.
        # L1 in0 + DRAM out is valid (sweep: matmul reads L1 activations); gx buffer stays DRAM.
        # ``program_config`` (fused path only) is the tuned 2D mcast config for the [B,L,in]@[in,4H]
        # gate precompute — 11.6µs vs 17.3µs default (see _gatex_program_config / the matmul sweep).
        bias = ttnn.reshape(p.b, [1, 1, H4], memory_config=gx_mc)
        gx = ttnn.linear(
            x_in,
            p.w_x,
            bias=bias,
            memory_config=gx_mc,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )  # [B, L, 4H]
        gx_t = ttnn.permute(gx, (1, 0, 2), memory_config=gx_mc)  # [L, B, 4H]
        ttnn.deallocate(gx)
        return gx_t

    def _precompute_gates_x(p: TTLSTMParams) -> ttnn.Tensor:
        return _precompute_gates_x_of(p, x_nlc)

    def _gates_x_at(gx_all: ttnn.Tensor, t: int) -> ttnn.Tensor:
        # Per-step gate-row hand-off (1 slice/step): the state-independent input projection is
        # batched out of the loop (gx_all), so only extracting step t's row stays per step.
        # Irreducible — the recurrence serializes the L steps. Same as _gates_comb_at (fused path).
        # Slice the timestep directly into the per-step memory config (L1 when small enough).
        # The DRAM gx buffer is read once and the [B, 4H] row lands where the gate add wants it,
        # so the per-step DRAM->L1 copy that a follow-up to_memory_config would emit is avoided.
        return ttnn.reshape(
            ttnn.slice(gx_all, [t, 0, 0], [t + 1, B, H4], [1, 1, 1], memory_config=step_mc),
            [B, H4],
            memory_config=step_mc,
        )

    h0 = _traced_zeros(
        [B, H],
        dtype=state_dtype,
        device=x_nlc.device(),
        memory_config=step_mc,
        key=(id(fwd.w_h), "lstm_h0", B, H, str(state_dtype), str(step_mc)),
    )
    c0 = _traced_zeros(
        [B, H],
        dtype=state_dtype,
        device=x_nlc.device(),
        memory_config=step_mc,
        key=(id(fwd.w_h), "lstm_c0", B, H, str(state_dtype), str(step_mc)),
    )

    valid_all = None
    # Timesteps t < min(lengths) have every batch row valid, so the pack-padded blend
    # is the identity (valid=1 -> new) and is skipped — bit-exact and avoids ~11 binary
    # ops/timestep. Only t >= min_len (where some row is padded) needs masking. With no
    # padding (all lengths == L, the common case) masking is skipped entirely.
    min_len = L
    if sequence_lengths is not None:
        assert len(sequence_lengths) == B, "sequence_lengths must have one entry per batch row"
        min_len = max(0, min(min(int(n) for n in sequence_lengths), L))
        if min_len < L:
            valid_all = _length_valid_mask_b1(
                batch=B,
                seq_len=L,
                sequence_lengths=sequence_lengths,
                device=x_nlc.device(),
                memory_config=step_mc,
                dtype=state_dtype,
            )

    # ---- Direction-fused fast path (no padding mask) ----------------------------------------
    # Advance forward and reverse passes together: one [B,2H]@[2H,8H] matmul, one sigmoid, one
    # set of slices and one cell-math chain per step instead of two. Bit-identical to the
    # per-direction loops (the block-diagonal recurrent weight's zeros add 0.0 in fp32 accum).
    if w_h_block is not None and valid_all is None:
        # Per-step tensors (state/gates/slices) on L1-interleaved instead of DRAM: the small
        # elementwise/slice/sigmoid ops are memory-bound, so L1 cuts their device time ~10-15%
        # (matmul is launch-bound, unaffected). L1-interleaved spreads across banks, so it needs
        # NO per-step reshard (unlike width-sharding) — interleaved in/out throughout.
        # Guard: only with ``recurrent_program_config`` set — its bounded matmul CBs avoid the L1
        # circular-buffer clash that forced DRAM at H>64 with the default config. The ``[L,B,2H]``
        # output accumulation is the only L-scaling term, so gate on a sequence-length budget and
        # fall back to DRAM for long sequences (keeps L1 footprint bounded).
        #
        # L1 AND the B==1 bias-fold (``do_fold``) now COMPOSE: the fold folds the per-step gates_x add
        # into the recurrent matmul's bias epilogue (one fewer BinaryNg/step — the highest-dispatch-gap
        # op in the loop), and the L1 in0/out + L1 bias matmul that this needs is verified to run on BH
        # (the old "bias-epilogue TT_FATALs with L1 in0/out" limitation that forced us to pick L1 *or*
        # fold has been lifted in ttnn). So for B==1 we keep both: L1 cuts the matmul (~8->6 us) and
        # every per-step elementwise/slice, AND the fold drops the gate-add. (The fold's fp32-epilogue
        # rounding differs from the separate bf16 add — fine for the tolerant ASR TextEncoder, which is
        # the only B==1 fused-path caller; the B>1 standalone test can't broadcast the [B,8H] gates_x as
        # a bias row so it keeps do_fold=False and the separate add, unchanged.)
        l1_weights: list[ttnn.Tensor] = []
        if (
            recurrent_program_config is not None
            and not fp32_state
            and step_mc.buffer_type != ttnn.BufferType.L1
            and L * _tensor_nbytes((B, 2 * H), state_dtype) <= _FUSED_L1_ACCUM_BUDGET_BYTES
        ):
            # Keep ``do_fold`` as computed above (True for B==1 + fold_gates_bias): L1 in0/out + L1
            # bias is verified to run on BH, so the fold and L1 now compose (see comment above).
            step_mc = ttnn.L1_MEMORY_CONFIG
            # Co-locate the gate-projection buffer (gx_comb [L,B,8H]) + its per-step slices/concat on
            # L1 too (same seq budget bounds both it and the [L,B,2H] state accumulation), and stage
            # the matmul weights (w_x for the gate precompute, w_h_block for the recurrent step) to
            # L1 for the duration of this forward. L1 in1 cuts the matmul device time substantially
            # (gate precompute + recurrent: ~438->343 us). The weight copies are transient (freed at
            # the end of this call), so no persistent L1 footprint leaks into the kmodel pipeline.
            gx_mc = ttnn.L1_MEMORY_CONFIG

            def _stage_l1(t: ttnn.Tensor) -> ttnn.Tensor:
                """Place ``t`` in L1 for this forward. Constant weights pre-staged to L1 (see
                ``preprocess_tt_text_encoder``) are used as-is — no per-forward DRAM->L1 copy and NOT
                registered for dealloc (they outlive the call). Only DRAM tensors are copied + tracked
                for the end-of-forward free, so no persistent L1 footprint leaks for the transients."""
                if t.memory_config().buffer_type == ttnn.BufferType.L1:
                    return t
                t_l1 = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
                l1_weights.append(t_l1)
                return t_l1

            w_h_block = _stage_l1(w_h_block)
            fwd_wx = _stage_l1(fwd.w_x)
            rev_wx = _stage_l1(rev.w_x)
            fwd = TTLSTMParams(w_x=fwd_wx, w_h=fwd.w_h, b=fwd.b, hidden_size=H)
            rev = TTLSTMParams(w_x=rev_wx, w_h=rev.w_h, b=rev.b, hidden_size=H)
            # Stage the LSTM input to L1 too so the gate-precompute (in0=x_nlc) and the reverse-input
            # reorder (in1=x_nlc) read L1 instead of DRAM — removes the last dram-interleaved matmul.
            # (x_nlc is recomputed every forward, so it is always a transient DRAM->L1 copy.)
            x_nlc = _stage_l1(x_nlc)
        # Output-assembly tensors (reverse-output reorder + the [B,L,2H] stack/slices) share the
        # step memory: L1 when the fused L1 path is active, else the caller's config.
        asm_mc = step_mc if step_mc.buffer_type == ttnn.BufferType.L1 else memory_config
        H2 = 2 * H
        H8 = 8 * H

        # Reverse the sequence in time with an anti-identity matmul (each output row is exactly
        # one input row * 1.0, so it is a bit-exact reordering). Then ``gx_rev`` computed from the
        # reversed input is already indexed in the order the reverse pass consumes it, so combined
        # step ``k`` pairs forward position ``k`` with reverse position ``L-1-k`` by a plain slice.
        # NOTE: keep these reorders at the caller's fidelity (HiFi3). The cliff is LoFi-specific: a LoFi
        # variant looks bit-exact in isolation (anti is 0/1) and passes the standalone encoder PCC, but
        # ``tt_bilstm_nlc`` is SHARED with the prosody DurationEncoder — running its reverse-input
        # reorder at LoFi shifts the bf16 activations enough to perturb F0/source phase, which the
        # vocoder amplifies: full kmodel config-E PCC collapses 0.872 -> 0.451. HiFi2 is fine here
        # (config-E 0.872, unchanged) but saves only ~0.2 µs/reorder on these 3 µs one-shot ops — moot
        # on a dispatch-bound model — so there is no reason to deviate. Validated via test_tt_kmodel_pcc.py.
        # The anti-identity reversal matrix is a constant (depends only on L); uploading it each call
        # is a host->device write. Under trace weight prep, build+cache once and keep it alive
        # (``_own_anti`` guards the deallocation below); prep off = build+free each call.
        _anti_key = (id(fwd.w_h), "lstm_anti", B, L, str(x_nlc.dtype), str(gx_mc))
        _own_anti = not _trace_weight_prep_enabled()
        anti_tt = _prep_cache_get(_anti_key) if _trace_weight_prep_enabled() else None
        if anti_tt is None:
            anti = torch.eye(L, dtype=torch.float32).flip(0).reshape(1, L, L).expand(B, L, L).contiguous()
            anti_tt = ttnn.from_torch(
                anti, dtype=x_nlc.dtype, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=gx_mc
            )
            if _trace_weight_prep_enabled():
                _prep_cache_set(_anti_key, anti_tt)
        x_rev = ttnn.matmul(anti_tt, x_nlc, memory_config=gx_mc, compute_kernel_config=compute_kernel_config)
        # anti_tt is kept alive — reused to re-order the reverse-pass outputs back into natural time.

        # Tuned 2D mcast config for the two [B,L,in]@[in,4H] gate-precompute matmuls (fused path only).
        gatex_pc = _gatex_program_config(
            batch=B, seq_len=L, four_hidden=H4, in_dim=int(x_nlc.shape[-1]), device=x_nlc.device()
        )
        gx_f = _precompute_gates_x_of(fwd, x_nlc, program_config=gatex_pc)  # [L, B, 4H] natural-order x
        gx_r = _precompute_gates_x_of(rev, x_rev, program_config=gatex_pc)  # [L, B, 4H] reversed-x order
        ttnn.deallocate(x_rev)

        # One-shot setup (NOT per step): interleave into gate-major direction-minor order
        # [i_f i_r f_f f_r g_f g_r o_f o_r] to match the fused recurrent weight's columns.
        # Done once so each step reads a contiguous [B,8H] row.
        chunks = []
        for gi in range(4):
            chunks.append(ttnn.slice(gx_f, [0, 0, gi * H], [L, B, (gi + 1) * H], [1, 1, 1], memory_config=gx_mc))
            chunks.append(ttnn.slice(gx_r, [0, 0, gi * H], [L, B, (gi + 1) * H], [1, 1, 1], memory_config=gx_mc))
        gx_comb = ttnn.concat(chunks, dim=-1, memory_config=gx_mc)  # [L, B, 8H]
        for ch in chunks:
            ttnn.deallocate(ch)
        ttnn.deallocate(gx_f)
        ttnn.deallocate(gx_r)

        def _gates_comb_at(t: int) -> ttnn.Tensor:
            # Per-step gate-row hand-off (1 slice/step): the batchable input projection is hoisted
            # out of the loop into gx_comb [L,B,8H], so only extracting step t's row stays per step.
            # L is the leading untiled dim, so this is a page copy (not untilize+slice+tilize) —
            # already at the floor. Irreducible: the recurrence serializes the L steps.
            return ttnn.reshape(
                ttnn.slice(gx_comb, [t, 0, 0], [t + 1, B, H8], [1, 1, 1], memory_config=step_mc),
                [B, H8],
                memory_config=step_mc,
            )

        hc = _traced_zeros(
            [B, H2],
            dtype=state_dtype,
            device=x_nlc.device(),
            memory_config=step_mc,
            key=(id(fwd.w_h), "lstm_hc", B, H2, str(state_dtype), str(step_mc)),
        )
        cc = _traced_zeros(
            [B, H2],
            dtype=state_dtype,
            device=x_nlc.device(),
            memory_config=step_mc,
            key=(id(fwd.w_h), "lstm_cc", B, H2, str(state_dtype), str(step_mc)),
        )
        ttnn.deallocate(h0)
        ttnn.deallocate(c0)

        outs_comb = []
        for t in range(L):
            gxt = _gates_comb_at(t)
            hc, cc = _lstm_step_fused(
                gxt,
                hc,
                cc,
                w_h_block,
                H,
                compute_kernel_config=compute_kernel_config,
                memory_config=step_mc,
                fold_bias=do_fold,
                program_config=recurrent_program_config,
                fuse_cell_math=fuse_cell_math,
            )
            # hc = [h_f@pos t | h_r@pos L-1-t]: forward output for position t and reverse output for
            # position L-1-t. Stash the whole [B,2H] state and split in bulk after the loop (avoids
            # 2 per-step output slices).
            outs_comb.append(hc)
        ttnn.deallocate(cc)
        ttnn.deallocate(gx_comb)

        # Stack the L combined states -> [B, L, 2H], then split the two direction halves in one slice
        # each. The reverse half is in reverse-pass order (row t holds position L-1-t); re-order it to
        # natural time with the same anti-identity matmul used for the input (bit-exact 0/1 reorder).
        # Keep this transient [B, L*2H] in L1 (asm_mc) instead of letting concat default to DRAM: it is
        # consumed immediately by the L1 reshape->slice below, so a DRAM output would force a needless
        # write + read-back of the ~1.5 MiB tensor. Placement only — bit-identical.
        cat = ttnn.concat(outs_comb, dim=-1, memory_config=asm_mc)  # [B, L*2H]
        for o in outs_comb:
            ttnn.deallocate(o)
        # The [B, L*2H] -> [B, L, 2H] de-interleave is a genuine tile relayout (L moves from the width
        # axis, where the per-step concat placed it, onto the height axis) ~22µs. A row-major round-trip
        # (untilize -> RM-view -> retilize) was measured *worse* (~32µs: the padded RM reshape isn't a
        # free view, plus the two layout ops), so keep the single TILE reshape.
        hcomb = ttnn.reshape(cat, [B, L, H2], memory_config=asm_mc)  # [B, L, 2H]
        ttnn.deallocate(cat)
        hs_f = ttnn.slice(hcomb, [0, 0, 0], [B, L, H], [1, 1, 1], memory_config=asm_mc)
        hs_b_rev = ttnn.slice(hcomb, [0, 0, H], [B, L, H2], [1, 1, 1], memory_config=asm_mc)
        ttnn.deallocate(hcomb)
        # NOTE: no program_config here. This anti-identity reorder is a *batched* matmul (B>1 leading
        # dim), and ttnn sizes its output blocks as B*ceil(L/32) along the core-grid y; a fixed/derived
        # config that satisfies ``num_blocks_y <= grid.y`` couples to both B and L (which vary per
        # input) and TT_FATALs on mismatch. Unlike the recurrent matmul (M=batch -> one tile row, so a
        # robust config transfers), this 3-6 µs one-shot op stays on the default config.
        hs_b = ttnn.matmul(anti_tt, hs_b_rev, memory_config=asm_mc, compute_kernel_config=compute_kernel_config)
        ttnn.deallocate(hs_b_rev)
        if _own_anti:
            ttnn.deallocate(anti_tt)
        # Return in the caller's memory_config (preserve the DRAM contract; assembly stayed L1).
        out = ttnn.concat([hs_f, hs_b], dim=2, memory_config=out_memory_config or memory_config)
        for _w in l1_weights:  # free the transient L1 weight copies (no persistent L1 footprint)
            ttnn.deallocate(_w)
        return out
    # ---- End fused fast path ----------------------------------------------------------------

    # P1+P2: L1-resident fp32 per-direction path (the shared F0/N BiLSTM — ``w_h_block`` is None so it
    # never takes the fused loop, and ``fp32_state`` is True). The per-step elementwise/slice/sigmoid
    # and the recurrent matmul are ~68% of F0Ntrain device time and run on DRAM at H=256 under the
    # ``_lstm_step_memory_config`` ``hidden<=64`` gate. Move them to L1: (P1) ``step_mc``=L1 for the
    # per-step tensors; (P2) a bounded recurrent program config (``_perdir_recurrent_program_config``)
    # so the matmul's circular buffers don't overlap the L1 tensors (the clash that forced DRAM). The
    # big ``[L,B,4H]`` gx buffer stays DRAM (only the per-step ``[B,4H]`` slice is L1); ``w_h`` (read
    # every step) is staged to L1 for the call and freed after. Gated on an L1 accumulation budget so
    # long sequences fall back to the DRAM path. Applies ONLY to the fp32 shared BiLSTM (``fp32_state``)
    # — the bf16 fused/duration LSTMs are untouched. Numerics: L1 placement + full-K program config are
    # intended numerics-neutral, but this feeds the F0-sensitive vocoder, so it is validated end-to-end.
    recur_pc = None
    out_dtype_step = None
    l1_weights: list[ttnn.Tensor] = []
    if (
        fp32_state
        and step_mc.buffer_type != ttnn.BufferType.L1
        and 2 * L * _tensor_nbytes((B, H), state_dtype) <= _PERDIR_L1_ACCUM_BUDGET_BYTES
    ):
        recur_pc = _perdir_recurrent_program_config(batch=B, hidden=H, device=x_nlc.device())
        if recur_pc is not None:
            step_mc = ttnn.L1_MEMORY_CONFIG
            out_dtype_step = state_dtype

            def _stage_l1(t: ttnn.Tensor) -> ttnn.Tensor:
                if t.memory_config().buffer_type == ttnn.BufferType.L1:
                    return t
                t_l1 = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
                l1_weights.append(t_l1)
                return t_l1

            # Stage only w_h (recurrent, read every step) to L1; w_x is used once (batched gate
            # precompute) so it stays DRAM to bound the transient L1 footprint.
            fwd = TTLSTMParams(w_x=fwd.w_x, w_h=_stage_l1(fwd.w_h), b=fwd.b, hidden_size=H)
            rev = TTLSTMParams(w_x=rev.w_x, w_h=_stage_l1(rev.w_h), b=rev.b, hidden_size=H)

    gx_fwd = _precompute_gates_x(fwd)
    gx_rev = _precompute_gates_x(rev)

    h_f = h0
    c_f = c0
    outs_f = []
    for t in range(L):
        gxt = _gates_x_at(gx_fwd, t)
        h_new, c_new = _lstm_step(
            gxt,
            h_f,
            c_f,
            fwd,
            compute_kernel_config=compute_kernel_config,
            memory_config=step_mc,
            fold_bias=do_fold,
            program_config=recur_pc,
            out_dtype=out_dtype_step,
        )
        # Forward padding is a *suffix* (t >= length): once a row is padded it never returns to a
        # valid position, and batch rows don't mix (per-row recurrence). So a padded row's evolving
        # state is never read by any valid output — the state needs no freeze; only the output at
        # padded positions must be zeroed. Skipping the two ``_blend_state`` calls saves 6 binary
        # ops/step on the padded tail; valid-position outputs are bit-identical.
        h_f, c_f = h_new, c_new
        if valid_all is not None and t >= min_len:
            vt = ttnn.slice(valid_all, [t, 0, 0], [t + 1, B, 1], [1, 1, 1], memory_config=step_mc)
            vt_b1 = ttnn.reshape(vt, [B, 1], memory_config=step_mc)
            outs_f.append(ttnn.multiply(vt_b1, h_new, memory_config=step_mc))
        else:
            outs_f.append(h_f)

    h_b = h0
    c_b = c0
    outs_b_rev = []
    for t in reversed(range(L)):
        gxt = _gates_x_at(gx_rev, t)
        h_new, c_new = _lstm_step(
            gxt,
            h_b,
            c_b,
            rev,
            compute_kernel_config=compute_kernel_config,
            memory_config=step_mc,
            fold_bias=do_fold,
            program_config=recur_pc,
            out_dtype=out_dtype_step,
        )
        if valid_all is not None and t >= min_len:
            # Reverse hits the suffix padding *first*, so the recurrent state is still 0 across the
            # whole padded region (it starts at 0 and each padded step multiplies it back to 0).
            # With old == 0 the blend ``old + valid*(new - old)`` collapses to ``valid*new`` bit-for-
            # bit (subtracting/adding 0 is exact), so one multiply replaces the 3-op blend per state
            # and the masked hidden state doubles as the masked output — 7 ops/step -> 2.
            vt = ttnn.slice(valid_all, [t, 0, 0], [t + 1, B, 1], [1, 1, 1], memory_config=step_mc)
            vt_b1 = ttnn.reshape(vt, [B, 1], memory_config=step_mc)
            h_b = ttnn.multiply(vt_b1, h_new, memory_config=step_mc)
            c_b = ttnn.multiply(vt_b1, c_new, memory_config=step_mc)
            outs_b_rev.append(h_b)
        else:
            h_b, c_b = h_new, c_new
            outs_b_rev.append(h_b)

    if valid_all is not None:
        ttnn.deallocate(valid_all)
    ttnn.deallocate(gx_fwd)
    ttnn.deallocate(gx_rev)
    for _w in l1_weights:  # free the transient L1 w_h copies (no persistent L1 footprint leaks)
        ttnn.deallocate(_w)

    outs_b = list(reversed(outs_b_rev))

    # Assemble per-timestep [B, H] outputs into [B, L, H]. Concatenate along the *last* dim
    # (H=64 is tile-aligned) -> [B, L*H], then a single reshape -> [B, L, H]. Concatenating on
    # dim 0 instead would force an unpad of every input (its dim-0 size 1 is tile-padded to 32),
    # emitting one UntilizeWithUnpadding per timestep; the width concat is a clean tile copy and
    # drops the trailing permute too. Pure data movement (bit-identical).
    def _stack_time(outs):
        cat = ttnn.concat(outs, dim=-1)  # [B, L*H]
        return ttnn.reshape(cat, [B, L, H], memory_config=memory_config)  # [B, L, H]

    hs_f = _stack_time(outs_f)
    hs_b = _stack_time(outs_b)
    return ttnn.concat([hs_f, hs_b], dim=2, memory_config=out_memory_config or ttnn.DRAM_MEMORY_CONFIG)
