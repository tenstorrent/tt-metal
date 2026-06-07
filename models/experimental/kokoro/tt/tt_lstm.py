# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN LSTM helpers (TT-prefixed params, ``tt_`` entrypoints).

Host-driven timestep loop using ``ttnn.linear`` and activations (no high-level LSTM op).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch

import ttnn

_TILE = 32
# Per-step LSTM state/gate tensors are tiny at bring-up widths (H<=64); keep them in L1 when
# tile-padded storage fits (gx precompute buffers stay on the caller's memory_config).
# At Kokoro-82M H=256 the recurrent ``h @ W_h^T`` matmul CB footprint clashes with L1
# tensor allocations on BH (see kmodel L1 circular-buffer overlap at ~137 KiB).
_L1_STEP_BUDGET_BYTES = 512 * 1024
_L1_STEP_HIDDEN_MAX = 64


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


def _lstm_step(
    gates_x: ttnn.Tensor,
    h: ttnn.Tensor,
    c: ttnn.Tensor,
    params: TTLSTMParams,
    *,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    # ``gates_x`` (= x_t @ W_x^T + b, shape [B, 4H]) is precomputed for the whole sequence
    # by the caller (one batched matmul with the bias folded into its epilogue, see
    # tt_bilstm_nlc) — it doesn't depend on the recurrent state. Only the recurrent
    # ``h @ W_h^T`` is per-step, and the gate sum is then a single add.
    gates_h = ttnn.linear(
        h,
        params.w_h,
        bias=None,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
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

    # Gate order is [i, f, g, o] with i,f,o -> sigmoid and g -> tanh. Sigmoid is
    # elementwise, so applying it once over the whole [B, 4H] tensor and slicing the
    # i/f/o parts is bit-identical to three separate sigmoids but uses one op instead
    # of three. g still needs tanh on the raw (pre-sigmoid) slice.
    sig = ttnn.sigmoid(gates, memory_config=memory_config)
    i = ttnn.slice(sig, [0, 0], [sig.shape[0], H], [1, 1])
    f = ttnn.slice(sig, [0, H], [sig.shape[0], 2 * H], [1, 1])
    o = ttnn.slice(sig, [0, 3 * H], [sig.shape[0], 4 * H], [1, 1])
    ttnn.deallocate(sig)
    g = ttnn.tanh(ttnn.slice(gates, [0, 2 * H], [gates.shape[0], 3 * H], [1, 1]), memory_config=memory_config)

    # c_new = f*c + i*g. Kept as separate mul/mul/add: the fused addcmul (MAC) changes
    # the cell-state accumulation, which feeds the F0 curve (amplified ~1885x by the
    # vocoder) and dropped end-to-end PCC below the 0.84 floor (0.855 -> 0.825).
    c_new = ttnn.add(
        ttnn.multiply(f, c, memory_config=memory_config),
        ttnn.multiply(i, g, memory_config=memory_config),
        memory_config=memory_config,
    )
    h_new = ttnn.multiply(o, ttnn.tanh(c_new, memory_config=memory_config), memory_config=memory_config)
    return h_new, c_new


def _length_valid_mask_b1(
    *,
    batch: int,
    seq_len: int,
    sequence_lengths: Sequence[int],
    device,
    memory_config: ttnn.MemoryConfig,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    vm = torch.zeros(batch, seq_len, 1, dtype=torch.float32)
    for bi, le in enumerate(sequence_lengths):
        le = max(0, min(int(le), seq_len))
        vm[bi, :le, 0] = 1.0
    return ttnn.from_torch(
        vm,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _blend_state(
    valid_b1: ttnn.Tensor,
    new_state: ttnn.Tensor,
    old_state: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Blend LSTM state; ``valid_b1`` is ``[B, 1]`` (broadcasts over hidden dim).

    ``old + valid*(new - old)`` — 3 ops vs 5 for the ``valid*new + (1-valid)*old``
    form, and bit-exact for the 0/1 mask (valid=1 -> new, valid=0 -> old).
    """
    diff = ttnn.subtract(new_state, old_state, memory_config=memory_config)
    return ttnn.add(old_state, ttnn.multiply(valid_b1, diff, memory_config=memory_config), memory_config=memory_config)


def tt_bilstm_nlc(
    *,
    x_nlc: ttnn.Tensor,
    fwd: TTLSTMParams,
    rev: TTLSTMParams,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    sequence_lengths: Optional[Sequence[int]] = None,
    fp32_state: bool = False,
) -> ttnn.Tensor:
    """
    1-layer BiLSTM over sequence for NLC ``[B, L, in]``.

    With ``sequence_lengths``, timesteps ``t >= length[b]`` yield zero output and frozen state
    (``pack_padded_sequence`` semantics). Returns ``[B, L, 2H]``.

    When ``fp32_state`` is True, hidden/cell states are accumulated in fp32 to avoid
    Hz-level drift in long F0/N decoder chains (shared BiLSTM in ``F0Ntrain``).
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

    # Input gate projections (``x @ W_x^T``) don't depend on the recurrent state, so compute
    # them for the WHOLE sequence in one batched matmul per direction (bit-identical: matmul
    # rows are independent). Permute to ``[L, B, 4H]`` so the per-timestep extraction is a
    # cheap leading-dim slice — this removes the per-step untilize+slice+tilize churn that
    # dominated the loop (slicing a single timestep out of the TILE-laid [B,L,C] sequence).
    def _precompute_gates_x(p: TTLSTMParams) -> ttnn.Tensor:
        # Fold the gate bias into the matmul epilogue here (once for the whole sequence)
        # instead of adding it per timestep — saves one elementwise add per step. The
        # stored bias is [1,1,1,4H] (rank 4); reshape to [1,1,4H] so the linear output
        # stays rank-3 [B,L,4H] for the permute below.
        # L1 in0 + DRAM out is valid (sweep: matmul reads L1 activations); gx buffer stays DRAM.
        bias = ttnn.reshape(p.b, [1, 1, H4], memory_config=gx_mc)
        gx = ttnn.linear(
            x_nlc,
            p.w_x,
            bias=bias,
            memory_config=gx_mc,
            compute_kernel_config=compute_kernel_config,
        )  # [B, L, 4H]
        gx_t = ttnn.permute(gx, (1, 0, 2), memory_config=gx_mc)  # [L, B, 4H]
        ttnn.deallocate(gx)
        return gx_t

    gx_fwd = _precompute_gates_x(fwd)
    gx_rev = _precompute_gates_x(rev)

    def _gates_x_at(gx_all: ttnn.Tensor, t: int) -> ttnn.Tensor:
        # Slice the timestep directly into the per-step memory config (L1 when small enough).
        # The DRAM gx buffer is read once and the [B, 4H] row lands where the gate add wants it,
        # so the per-step DRAM->L1 copy that a follow-up to_memory_config would emit is avoided.
        return ttnn.reshape(
            ttnn.slice(gx_all, [t, 0, 0], [t + 1, B, H4], [1, 1, 1], memory_config=step_mc),
            [B, H4],
            memory_config=step_mc,
        )

    h0 = ttnn.zeros([B, H], dtype=state_dtype, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=step_mc)
    c0 = ttnn.zeros([B, H], dtype=state_dtype, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=step_mc)

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

    h_f = h0
    c_f = c0
    outs_f = []
    for t in range(L):
        gxt = _gates_x_at(gx_fwd, t)
        h_old, c_old = h_f, c_f
        h_new, c_new = _lstm_step(
            gxt,
            h_f,
            c_f,
            fwd,
            compute_kernel_config=compute_kernel_config,
            memory_config=step_mc,
        )
        if valid_all is not None and t >= min_len:
            vt = ttnn.slice(valid_all, [0, t, 0], [B, t + 1, 1], [1, 1, 1])
            vt_b1 = ttnn.reshape(vt, [B, 1], memory_config=step_mc)
            h_f = _blend_state(vt_b1, h_new, h_old, memory_config=step_mc)
            c_f = _blend_state(vt_b1, c_new, c_old, memory_config=step_mc)
            outs_f.append(ttnn.multiply(vt_b1, h_new, memory_config=step_mc))
        else:
            h_f, c_f = h_new, c_new
            outs_f.append(h_f)

    h_b = h0
    c_b = c0
    outs_b_rev = []
    for t in reversed(range(L)):
        gxt = _gates_x_at(gx_rev, t)
        h_old, c_old = h_b, c_b
        h_new, c_new = _lstm_step(
            gxt,
            h_b,
            c_b,
            rev,
            compute_kernel_config=compute_kernel_config,
            memory_config=step_mc,
        )
        if valid_all is not None and t >= min_len:
            vt = ttnn.slice(valid_all, [0, t, 0], [B, t + 1, 1], [1, 1, 1])
            vt_b1 = ttnn.reshape(vt, [B, 1], memory_config=step_mc)
            h_b = _blend_state(vt_b1, h_new, h_old, memory_config=step_mc)
            c_b = _blend_state(vt_b1, c_new, c_old, memory_config=step_mc)
            outs_b_rev.append(ttnn.multiply(vt_b1, h_new, memory_config=step_mc))
        else:
            h_b, c_b = h_new, c_new
            outs_b_rev.append(h_b)

    if valid_all is not None:
        ttnn.deallocate(valid_all)
    ttnn.deallocate(gx_fwd)
    ttnn.deallocate(gx_rev)

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
    return ttnn.concat([hs_f, hs_b], dim=2)
