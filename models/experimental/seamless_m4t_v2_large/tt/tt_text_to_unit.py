# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2TextToUnitForConditionalGeneration`]: encoder + decoder + ``lm_head``.

**Parity target (Hugging Face):** ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` — same dataflow as
``modeling_seamless_m4t_v2``: encoder on ``inputs_embeds``; decoder char upsample, duration predictor,
unit upsample, self-attn + conv decoder stack; ``lm_head`` logits in the field HF names
``last_hidden_state``.

**Implementation policy:** All math in this file runs through **ttnn** (no ``torch`` / no ``numpy`` /
no Transformers helpers). Host-side control flow uses Python ``int`` / ``Sequence[int]`` for repeat
counts and sequence lengths. Small host readbacks use ``to_torch_replicated_first_shard`` as a
host transport only — torch is not used for any math — to unpack float32 values into integer repeat
counts for ``ttnn.repeat_interleave``, matching HF ``round(expm1(...))`` with clamp. I/O tensors stay
on device except for that readback.

**Whole-model compatibility:** Callers should pass ``char_count_per_id`` as a length-``enc_seq`` list of
non-negative integers (batch size 1), matching HF ``char_count_per_id.sum(-1)`` semantics; build the
encoder 4D additive mask with the same helpers used for the text encoder PCC tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import ttnn

from models.common.utility_functions import nearest_32
from models.experimental.seamless_m4t_v2_large.tt.common import (
    core_grid,
    matmul_program_config,
    MATMUL_1D_SEQ_THRESHOLD,
    sdpa_program_config,
    to_torch_replicated_first_shard,
)

# HF ``torch.finfo(torch.bfloat16).min`` additive padding mask floor (approx.).
_BF16_MASK_FLOOR = -3.3895313892565356e38


def _linear_token_rows(x: ttnn.Tensor) -> int:
    if len(x.shape) == 3:
        return int(x.shape[0]) * int(x.shape[1])
    if len(x.shape) == 2:
        return int(x.shape[0])
    return int(x.shape[-2])


def _upload_repeat_cumsum_tiles(device: ttnn.Device, repeats: Sequence[int]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Inclusive / exclusive cum boundaries as TILE float32 ``[1, len(repeats)]`` for ``_hard_upsample_nlc``."""
    enc_seq = len(repeats)
    cum_inc_list: list[float] = []
    acc = 0
    for r in repeats:
        acc += int(r)
        cum_inc_list.append(float(acc))
    prev_list = [0.0] + cum_inc_list[:-1]
    cumsum_inc_rm = ttnn.Tensor(
        cum_inc_list,
        [1, enc_seq],
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )
    cumsum_prev_rm = ttnn.Tensor(
        prev_list,
        [1, enc_seq],
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )
    cumsum_inc = ttnn.to_layout(cumsum_inc_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cumsum_prev = ttnn.to_layout(cumsum_prev_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cumsum_inc_rm)
    ttnn.deallocate(cumsum_prev_rm)
    return cumsum_inc, cumsum_prev


def _unit_hidden_pad_tail(
    device: ttnn.Device,
    *,
    batch: int,
    pad_len: int,
    hidden_size: int,
    cache: Dict[Tuple[int, int, int], ttnn.Tensor],
) -> ttnn.Tensor:
    """Pre-built zero tail ``[B, pad_len, H]`` for tile-aligned unit-seq padding (one ``concat`` vs ``pad``)."""
    key = (batch, pad_len, hidden_size)
    tail = cache.get(key)
    if tail is None:
        tail = ttnn.zeros(
            (batch, pad_len, hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cache[key] = tail
    return tail


def _row_major_host_f32_flat(host_tensor: ttnn.Tensor, *, num_floats: int) -> list[float]:
    """Flat list of float32 values from a host-side ttnn tensor.

    Uses ``to_torch_replicated_first_shard`` for the host readback because the ``HostBuffer`` iterator yields
    ``std::byte``, which pybind does not auto-convert (``bytes(shard)`` raises TypeError). The
    decoded values feed integer repeat counts for ``ttnn.repeat_interleave`` — torch is only used
    as a host transport, no math runs through it.
    """
    flat = to_torch_replicated_first_shard(host_tensor).to(torch.float32).reshape(-1)
    n = int(num_floats)
    if int(flat.numel()) < n:
        raise RuntimeError(f"Host tensor has {int(flat.numel())} floats; need {n}.")
    return flat[:n].tolist()


def _mask_row_valid_prefix(device: ttnn.Device, width: int, valid_len: int) -> ttnn.Tensor:
    """``[1, width]`` bfloat16 tile: 1 where column index < ``valid_len``, else 0 (batch 1)."""
    idx_rm = ttnn.arange(
        0,
        width,
        step=1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    idx = ttnn.reshape(idx_rm, (1, width))
    idx_t = ttnn.to_layout(idx, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    lim = ttnn.full(
        (1, width),
        float(valid_len),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ones = ttnn.full(
        (1, width),
        1.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros = ttnn.full(
        (1, width),
        0.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.where(ttnn.lt(idx_t, lim), ones, zeros, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


def _expand_4d_padding_additive_b1(
    device: ttnn.Device, mask_2d_tile: ttnn.Tensor, seq_len: int, width: int
) -> ttnn.Tensor:
    """
    HF ``AttentionMaskConverter._expand_mask`` for a 2D padding mask (1 = keep), batch 1.

    Returns additive mask ``[1, 1, seq_len, width]`` (tile bf16): 0 keep, ``_BF16_MASK_FLOOR`` masked.
    """
    if mask_2d_tile.get_layout() != ttnn.TILE_LAYOUT:
        m = ttnn.to_layout(mask_2d_tile, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        m = mask_2d_tile
    row = ttnn.reshape(m, (1, 1, 1, width))
    expanded = ttnn.repeat_interleave(row, seq_len, dim=2)
    ones = ttnn.full(
        (1, 1, seq_len, width),
        1.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inverted = ttnn.add(
        ones,
        ttnn.multiply(expanded, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    floor = ttnn.full(
        (1, 1, seq_len, width),
        _BF16_MASK_FLOOR,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros = ttnn.full(
        (1, 1, seq_len, width),
        0.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.where(ttnn.gt(inverted, 0.5), floor, zeros, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


@dataclass
class T2UTraceHardUpsampleCumsums:
    """Device tensors for T2U when ``hard_upsample_cums`` is used (built **outside** trace capture).

    The four ``*_inc`` / ``*_prev`` TILE float32 rows feed ``_hard_upsample_nlc`` (no Python-list
    uploads). When the optional ``char_frame_idx_f32`` … fields are all set, the forward also skips
    ``ttnn.arange`` / ``ttnn.pad`` / mask builders during capture — those allocate device buffers and
    can enqueue writes while ``begin_trace_capture`` is active (Metal forbids that).
    """

    char_inc: ttnn.Tensor
    char_prev: ttnn.Tensor
    unit_inc: ttnn.Tensor
    unit_prev: ttnn.Tensor
    char_frame_idx_f32: Optional[ttnn.Tensor] = None
    unit_frame_idx_f32: Optional[ttnn.Tensor] = None
    char_pos_ids: Optional[ttnn.Tensor] = None
    unit_pos_ids: Optional[ttnn.Tensor] = None
    char_pad_bf16_tile: Optional[ttnn.Tensor] = None
    pad_unit_bf16_tile: Optional[ttnn.Tensor] = None
    attn_4d_bf16_tile: Optional[ttnn.Tensor] = None
    pad_unit_3d_tt: Optional[ttnn.Tensor] = None
    unit_hidden_pad_tail_bf16: Optional[ttnn.Tensor] = None


def make_t2u_trace_prealloc_tensors(
    device: ttnn.Device,
    *,
    pad_token_id: int,
    hidden_size: int,
    char_w: int,
    cc_list: Sequence[int],
    ref_durs: Sequence[int],
    char_inc: ttnn.Tensor,
    char_prev: ttnn.Tensor,
    unit_inc: ttnn.Tensor,
    unit_prev: ttnn.Tensor,
) -> T2UTraceHardUpsampleCumsums:
    """Pre-build T2U tensors that must not be allocated during trace capture (run before ``compile``)."""
    char_len = int(sum(int(x) for x in cc_list))
    unit_seq = int(sum(int(x) for x in ref_durs))
    padded_unit_seq = ((unit_seq + 31) // 32) * 32
    dur_sum = unit_seq

    char_frame = ttnn.arange(0, char_len, step=1, dtype=ttnn.float32, device=device)
    unit_frame = ttnn.arange(0, unit_seq, step=1, dtype=ttnn.float32, device=device)

    char_seq_total = char_len
    char_pad = _mask_row_valid_prefix(device, char_w, char_seq_total)
    if char_seq_total < char_w:
        char_pad = ttnn.slice(char_pad, [0, 0], [1, char_seq_total], [1, 1])
    if char_len < char_w:
        char_pad = ttnn.slice(char_pad, [0, 0], [1, char_len], [1, 1])

    char_pos = ttnn.reshape(
        ttnn.arange(
            pad_token_id + 1,
            pad_token_id + 1 + char_len,
            step=1,
            dtype=ttnn.uint32,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        (1, char_len),
    )
    unit_pos = ttnn.reshape(
        ttnn.arange(
            pad_token_id + 1,
            pad_token_id + 1 + unit_seq,
            step=1,
            dtype=ttnn.uint32,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        (1, unit_seq),
    )

    pad_unit = _mask_row_valid_prefix(device, padded_unit_seq, dur_sum)
    attn_4d = _expand_4d_padding_additive_b1(device, pad_unit, padded_unit_seq, padded_unit_seq)
    pad_unit_3d = ttnn.reshape(pad_unit, (1, padded_unit_seq, 1))

    tail: Optional[ttnn.Tensor] = None
    if padded_unit_seq > unit_seq:
        tail = ttnn.zeros(
            (1, padded_unit_seq - unit_seq, hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return T2UTraceHardUpsampleCumsums(
        char_inc=char_inc,
        char_prev=char_prev,
        unit_inc=unit_inc,
        unit_prev=unit_prev,
        char_frame_idx_f32=char_frame,
        unit_frame_idx_f32=unit_frame,
        char_pos_ids=char_pos,
        unit_pos_ids=unit_pos,
        char_pad_bf16_tile=char_pad,
        pad_unit_bf16_tile=pad_unit,
        attn_4d_bf16_tile=attn_4d,
        pad_unit_3d_tt=pad_unit_3d,
        unit_hidden_pad_tail_bf16=tail,
    )


def _conv1d_prep_tensor_id(t: ttnn.Tensor) -> int:
    if t.is_allocated() and t.storage_type() == ttnn.StorageType.DEVICE:
        return int(t.buffer_address())
    return id(t)


def _hard_upsample_nlc(
    enc: ttnn.Tensor,
    repeats: Sequence[int],
    *,
    device: ttnn.Device,
    hidden_size: int,
    cumsum_inc_tile: Optional[ttnn.Tensor] = None,
    cumsum_prev_tile: Optional[ttnn.Tensor] = None,
    frame_idx_f32: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """HF ``_hard_upsample`` for batch 1: ``enc`` is ``[1, T, H]`` tile; ``repeats`` length ``T``.

    Variable repeat-interleave as one device matmul.

    The previous per-slot Python loop (``slice`` + ``repeat_interleave`` +
    pairwise ``concat``) dispatched ~100 device ops per call, with each row
    Slice/Concat triggering a Tilize round-trip (~70-90 us each, ~15 ms total
    across both upsampler calls per forward).  We replace it with the same
    cumsum + comparisons + matmul pattern that ``tt_code_hifigan.py`` already
    uses to implement HF's ``repeat_interleave``:

      1. Compute inclusive / exclusive cumulative duration boundaries on host
         (Python ints; no torch).
      2. Upload them as ROW_MAJOR float32 device tensors via the
         ``ttnn.Tensor(data_list, ...)`` Python binding, which accepts a
         Python list directly -- no ``torch`` import needed.
      3. Build the expansion matrix ``H[1, sum_r, T]`` on device where
         ``H[f, t] = 1`` iff ``cumsum_prev[t] <= f < cumsum_inc[t]``.  Each
         output row picks exactly one input row.
      4. ``out = H @ enc`` gives ``[1, sum_r, H]`` in one matmul.

    Same math as ``enc[i]`` repeated ``r[i]`` times: every output row is a
    linear combination of input rows with a single 1.0 selecting one row.
    """
    enc_seq = len(repeats)
    repeats_int = [int(r) for r in repeats]
    if any(r < 0 for r in repeats_int):
        raise ValueError(f"_hard_upsample_nlc: negative repeat in {repeats_int!r}")

    owns_cum_boundary_tensors = not (cumsum_inc_tile is not None and cumsum_prev_tile is not None)
    if cumsum_inc_tile is not None and cumsum_prev_tile is not None:
        cumsum_inc = cumsum_inc_tile
        cumsum_prev = cumsum_prev_tile
    else:
        cumsum_inc, cumsum_prev = _upload_repeat_cumsum_tiles(device, repeats_int)

    sum_r = int(sum(repeats_int))
    if sum_r <= 0:
        raise ValueError("_hard_upsample_nlc: empty output (all repeat counts zero).")

    if frame_idx_f32 is not None:
        frame_idx = frame_idx_f32
        owns_frame_idx = False
    else:
        frame_idx = ttnn.arange(
            start=0,
            end=sum_r,
            step=1,
            dtype=ttnn.float32,
            device=device,
        )
        owns_frame_idx = True

    # Broadcast layout: cumsum_* -> [1, 1, T]; frame_idx -> [1, sum_r, 1].
    # ge/lt broadcast to [1, sum_r, T]; the resulting H[f, t] is 1 iff the
    # half-open boundary range ``[cumsum_prev[t], cumsum_inc[t])`` contains f.
    c_b = ttnn.reshape(cumsum_inc, (1, 1, enc_seq))
    cp_b = ttnn.reshape(cumsum_prev, (1, 1, enc_seq))
    f_b = ttnn.reshape(frame_idx, (1, sum_r, 1))
    if owns_frame_idx:
        ttnn.deallocate(frame_idx)

    lower = ttnn.ge(f_b, cp_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    upper = ttnn.lt(f_b, c_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # ``reshape`` may alias the TILE cum vectors; do not free views of caller-owned
    # ``hard_upsample_cums.*`` (trace pack tensors must survive across forwards).
    if owns_cum_boundary_tensors:
        ttnn.deallocate(c_b)
        ttnn.deallocate(cp_b)
    if owns_frame_idx:
        ttnn.deallocate(f_b)

    H_mask = ttnn.logical_and(lower, upper, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(lower)
    ttnn.deallocate(upper)
    H = ttnn.typecast(H_mask, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(H_mask)

    # H: [1, sum_r, T] bf16 TILE; enc: [1, T, hidden] bf16 TILE.
    # out: [1, sum_r, hidden]
    token_m = max(sum_r, 1)
    mm_pc = matmul_program_config(device, token_rows=token_m, in_dim=enc_seq, out_dim=int(enc.shape[-1]))
    out = ttnn.matmul(H, enc, program_config=mm_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(H)
    return out


def _discrete_duration_counts(log_dur: ttnn.Tensor, *, batch: int, seq: int) -> list[int]:
    """HF-exact ``clamp(round(expm1(log_dur)), min=1).long()`` per position.

    Caller (``_duration_predictor``) produces ``log_dur`` already in float32 — no bf16→fp32 typecast
    is needed, and the bf16 write-out that previously perturbed ``log_dur`` is gone.

    Round mode: ``ttnn.round`` uses the SFPU ``_round_even_`` op (round-half-to-even, banker's
    rounding) — bit-identical to ``torch.round`` semantics, so the on-device pipeline below is
    HF-equivalent ``expm1`` → banker's-round → clamp. The host loop only converts fp32 → Python int
    and applies a defensive ``max(1, ...)`` (no further rounding); we use ``int(round(...))`` rather
    than ``int(...)`` because ``int()`` truncates toward zero and could surprise on negative values
    (which clamp rules out anyway, but the explicit ``round`` is cheap insurance).
    """
    ld = ttnn.reshape(log_dur, (int(batch), int(seq)))
    if ld.dtype != ttnn.float32:
        ld = ttnn.typecast(ld, ttnn.float32)
    x = ttnn.expm1(ld, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.round(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.clamp(x, min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
    else:
        x_rm = x
    host_x = ttnn.from_device(x_rm)
    ttnn.deallocate(x_rm)
    x_h = _row_major_host_f32_flat(host_x, num_floats=int(batch) * int(seq))
    return [max(1, int(round(v))) for v in x_h]


def _conv1d_same(
    device: ttnn.Device,
    x_tile: ttnn.Tensor,
    *,
    sequence_length: int,
    weight_rm: ttnn.Tensor,
    bias_rm: Optional[ttnn.Tensor],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    activation: Optional[str] = None,
    prep_cache: Optional[Dict[Tuple[Any, ...], Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]]] = None,
) -> ttnn.Tensor:
    """Same-padding Conv1d stride 1 via ``ttnn.conv1d`` (activations ``[B,S,C]`` NLC).

    When ``activation`` is provided (e.g. ``"relu"``) it is fused into the
    conv kernel via ``Conv1dConfig.activation``; the SFPU applies it at the
    writeback stage of each output tile so no separate UnaryDeviceOperation
    is dispatched.  Numerically equivalent to a post-conv ``ttnn.relu``.
    """
    batch = int(x_tile.shape[0])
    seq = int(sequence_length)
    if x_tile.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        x_rm = ttnn.to_layout(x_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_tile)
    else:
        x_rm = x_tile
    # Host ROW_MAJOR conv weights: pass through to conv1d so conv2d prepares + uploads (no invalid device weights).

    # stream conv weights from DRAM as ``bfloat8_b`` (block-float8).
    # The decoder convs are DRAM-bandwidth-bound -- ``PM FPU UTIL`` is ~5% per
    # call and the kernel spends most of its time streaming a ``[7168, 1024]``
    # weight tensor (14 MB at bf16) per call across 12 conv calls per forward.
    # bf8 halves the per-weight byte count, which directly halves the DRAM
    # bytes the kernel waits on.  Activations and accumulators stay at bf16/fp32
    # (``fp32_dest_acc_en=True`` in the compute config), so per-tile rounding is
    # preserved -- the bf8 hit is a one-time weight quantization, the same
    # Same recipe used for FFN/attention matmuls elsewhere in this model.
    #
    # enable conv-side double buffering.
    # ``enable_weights_double_buffer`` allocates two L1 slots for incoming
    # weight tiles so the data-movement kernel can prefetch tile ``N+1`` while
    # the matrix engine multiplies tile ``N``.  For a bandwidth-bound conv this
    # turns the serial ``read -> compute -> read -> compute`` pattern into an
    # overlapped pipeline (~30% less DRAM-stall time per call).
    # ``enable_act_double_buffer`` does the same for activation tiles.
    # Both flags are pure scheduling optimizations; no math changes, so PCC is
    # unaffected.
    #
    # ``enable_activation_reuse`` is intentionally NOT enabled: the runtime
    # check ``act_block_h_ntiles > output_image_width_ntiles`` fails for these
    # convs (auto-sharding gives ``act_block_h=1 tile`` while output width is
    # 18 tiles), so the kernel refuses the flag with ``TT_FATAL``.
    conv_kwargs = dict(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=None,
        deallocate_activation=True,
        enable_weights_double_buffer=True,
        enable_act_double_buffer=True,
    )
    if activation:
        # ``Conv1dConfig.activation`` is bound as ``UnaryWithParam`` in Python;
        # the friendly ``"relu"`` string is mapped here so call sites remain
        # compact.  Extend the mapping if/when other fused activations are needed.
        _ACTIVATION_OP_TYPES = {
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
            "gelu": ttnn.UnaryOpType.GELU,
        }
        op_type = _ACTIVATION_OP_TYPES.get(activation.lower())
        if op_type is None:
            raise ValueError(f"_conv1d_same: unsupported activation {activation!r}")
        conv_kwargs["activation"] = ttnn.UnaryWithParam(op_type)
    if seq > 64 or in_channels >= 512:
        conv_kwargs["act_block_h_override"] = 32
    conv_config = ttnn.Conv1dConfig(**conv_kwargs)
    # Prepared conv weights depend on the *weight* tensor and conv geometry, not the activation's
    # DRAM/L1 memory_config. Including activation layout in the key caused cache misses when the
    # standalone T2U probe ran before the full E2E graph (or between compile vs trace replay), which
    # forced ``return_weights_and_bias=True`` and ``write_shard_to_device`` — illegal during trace capture.
    cache_key = (
        _conv1d_prep_tensor_id(weight_rm),
        _conv1d_prep_tensor_id(bias_rm) if bias_rm is not None else 0,
        batch,
        seq,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation or "",
        int(conv_kwargs.get("act_block_h_override", 0)),
    )
    if prep_cache is not None:
        cached = prep_cache.get(cache_key)
        if cached is not None:
            prep_w, prep_b = cached
            out_tt, _out_len = ttnn.conv1d(
                input_tensor=x_rm,
                weight_tensor=prep_w,
                in_channels=in_channels,
                out_channels=out_channels,
                device=device,
                bias_tensor=prep_b,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                batch_size=batch,
                input_length=seq,
                conv_config=conv_config,
                compute_config=compute_kernel_config,
                groups=1,
                dtype=ttnn.bfloat16,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            packed = ttnn.conv1d(
                input_tensor=x_rm,
                weight_tensor=weight_rm,
                in_channels=in_channels,
                out_channels=out_channels,
                device=device,
                bias_tensor=bias_rm,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                batch_size=batch,
                input_length=seq,
                conv_config=conv_config,
                compute_config=compute_kernel_config,
                groups=1,
                dtype=ttnn.bfloat16,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            out_tt, _out_len, wb = packed
            if isinstance(wb, (list, tuple)) and len(wb) >= 1:
                pw = wb[0]
                pb = wb[1] if len(wb) > 1 else None
            else:
                pw, pb = wb, None
            # ``conv1d`` may return prepared weights tied to internal buffers. After the compile
            # forward frees the output graph, those handles can become invalid while ``prep_cache``
            # still references them; the trace replay then hits ``cache hit`` but
            # ``is_valid_device_conv_weights`` fails and conv2d re-prepares (``write_shard_to_device``),
            # which is illegal during trace capture. Own copies in DRAM for the cache.
            prep_cache[cache_key] = (
                ttnn.clone(pw, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ttnn.clone(pb, memory_config=ttnn.DRAM_MEMORY_CONFIG) if pb is not None else None,
            )
    else:
        out_tt, _out_len = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=weight_rm,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
            bias_tensor=bias_rm,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            batch_size=batch,
            input_length=seq,
            conv_config=conv_config,
            compute_config=compute_kernel_config,
            groups=1,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )
    if out_tt.get_layout() != ttnn.TILE_LAYOUT:
        out_tile = ttnn.to_layout(out_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_tt)
        return out_tile
    return out_tt


class TTSeamlessM4Tv2TextToUnitEncoder:
    """
    Encoder stack inside HF ``SeamlessM4Tv2TextToUnitForConditionalGeneration.model`` —
    ``SeamlessM4Tv2Encoder(..., is_t2u_encoder=True)``: ``inputs_embeds`` only, then transformer + ``layer_norm``.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        *,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_size: int,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._sdpa_pc_cache: dict = {}
        self._matmul_pc_cache: dict = {}

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        return sdpa_program_config(self.device, seq_q, seq_k, self._sdpa_pc_cache)

    def _matmul_pc(self, token_rows: int, in_dim: int, out_dim: int) -> Optional[ttnn.ProgramConfig]:
        if token_rows > MATMUL_1D_SEQ_THRESHOLD:
            return None
        key = (token_rows, in_dim, out_dim)
        cached = self._matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        cached = matmul_program_config(
            self.device,
            token_rows=token_rows,
            in_dim=in_dim,
            out_dim=out_dim,
        )
        self._matmul_pc_cache[key] = cached
        return cached

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: str | None = None,
        program_config: Optional[ttnn.ProgramConfig] = None,
    ) -> ttnn.Tensor:
        if program_config is None:
            program_config = self._matmul_pc(
                _linear_token_rows(x),
                int(weight.shape[-2]),
                int(weight.shape[-1]),
            )
        kwargs: dict = dict(
            bias=bias,
            activation=activation,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        if program_config is not None:
            kwargs["program_config"] = program_config
        else:
            kwargs["core_grid"] = core_grid(self.device)
        return ttnn.linear(x, weight, **kwargs)

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        if ttnn.is_sharded(x):
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _self_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module,
        attn_mask: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        token_m = batch * seq
        pc_qkv = self._matmul_pc(token_m, hidden_size, 3 * hidden_size)
        qkv = self._linear(
            hidden_states,
            attn_module.qkv.weight,
            attn_module.qkv.bias,
            program_config=pc_qkv,
        )
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq, 3 * hidden_size))

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_4d)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0 / math.sqrt(head_dim),
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq, hidden_size))

        pc_out = self._matmul_pc(token_m, hidden_size, hidden_size)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            program_config=pc_out,
        )
        ttnn.deallocate(merged)
        return proj

    def forward(self, inputs_embeds: ttnn.Tensor, attention_mask_4d: ttnn.Tensor) -> ttnn.Tensor:
        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers

        batch = int(inputs_embeds.shape[0])
        seq = int(inputs_embeds.shape[1])

        hidden = inputs_embeds
        sdpa_cfg = self._sdpa_program_config(seq, seq)
        token_m = batch * seq
        ffn_dim = int(parameters.layers[0].ffn.fc1.weight.shape[-1])
        pc_ffn_fc1 = self._matmul_pc(token_m, hidden_size, ffn_dim)
        pc_ffn_fc2 = self._matmul_pc(token_m, ffn_dim, hidden_size)

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
            )
            attn_out = self._self_attention(
                normed,
                layer.self_attn,
                attention_mask_4d,
                batch=batch,
                seq=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_cfg,
            )
            ttnn.deallocate(normed)
            hidden = ttnn.add(hidden, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
            )
            ff = self._linear(
                normed,
                layer.ffn.fc1.weight,
                layer.ffn.fc1.bias,
                activation="relu",
                program_config=pc_ffn_fc1,
            )
            ttnn.deallocate(normed)
            ff = self._linear(
                ff,
                layer.ffn.fc2.weight,
                layer.ffn.fc2.bias,
                program_config=pc_ffn_fc2,
            )
            hidden = ttnn.add(hidden, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ff)

        hidden = self._layer_norm(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
        )
        return hidden


class TTSeamlessM4Tv2TextToUnitForConditionalGeneration:
    """
    TTNN port of HF ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` (encoder + decoder + ``lm_head``).

    Conv1d blocks use ``ttnn.conv1d`` (same padding, stride 1). If a deployment hits L1 limits, tune
    ``Conv1dConfig.act_block_h_override`` inside ``_conv1d_same``.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Any,
        *,
        layer_norm_eps: float,
        encoder_layers: int,
        encoder_attention_heads: int,
        decoder_layers: int,
        decoder_attention_heads: int,
        hidden_size: int,
        pad_token_id: int,
        variance_predictor_embed_dim: int,
        variance_predictor_hidden_dim: int,
        variance_predictor_kernel_size: int,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.variance_predictor_embed_dim = variance_predictor_embed_dim
        self.variance_predictor_hidden_dim = variance_predictor_hidden_dim
        self.variance_predictor_kernel_size = variance_predictor_kernel_size

        self.encoder = TTSeamlessM4Tv2TextToUnitEncoder(
            device,
            parameters.encoder,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            hidden_size=hidden_size,
        )
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._conv_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._conv1d_prepared_cache: Dict[Tuple[Any, ...], Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]] = {}
        self._sdpa_pc_cache: dict = {}
        self._matmul_pc_cache: dict = {}
        self._unit_pad_tail_cache: Dict[Tuple[int, int, int], ttnn.Tensor] = {}
        self._decoder_mask_cache: Dict[Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]] = {}
        self._repeat_cum_cache: Dict[Tuple[int, ...], Tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def _cached_repeat_cumsums(self, repeats: Sequence[int]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        key = tuple(int(r) for r in repeats)
        cached = self._repeat_cum_cache.get(key)
        if cached is not None:
            return cached
        inc, prev = _upload_repeat_cumsum_tiles(self.device, repeats)
        self._repeat_cum_cache[key] = (inc, prev)
        return inc, prev

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        key = (seq_q, seq_k)
        cached = self._sdpa_pc_cache.get(key)
        if cached is not None:
            return cached

        m = max(seq_q, seq_k)
        if m > 96:
            cap = 32
        elif m > 64:
            cap = 64
        elif m > 32:
            cap = 128
        else:
            cap = 256
        q_chunk = max(32, min(cap, nearest_32(seq_q)))
        k_chunk = max(32, min(cap, nearest_32(seq_k)))
        out = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        self._sdpa_pc_cache[key] = out
        return out

    def _matmul_pc(self, token_rows: int, in_dim: int, out_dim: int) -> Optional[ttnn.ProgramConfig]:
        if token_rows > MATMUL_1D_SEQ_THRESHOLD:
            return None
        key = (token_rows, in_dim, out_dim)
        cached = self._matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        cached = matmul_program_config(
            self.device,
            token_rows=token_rows,
            in_dim=in_dim,
            out_dim=out_dim,
        )
        self._matmul_pc_cache[key] = cached
        return cached

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: str | None = None,
        program_config: Optional[ttnn.ProgramConfig] = None,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        dtype: Optional[ttnn.DataType] = None,
    ) -> ttnn.Tensor:
        if program_config is None:
            program_config = self._matmul_pc(
                _linear_token_rows(x),
                int(weight.shape[-2]),
                int(weight.shape[-1]),
            )
        mem = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        kwargs: dict = dict(
            bias=bias,
            activation=activation,
            memory_config=mem,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        if program_config is not None:
            kwargs["program_config"] = program_config
        else:
            kwargs["core_grid"] = core_grid(self.device)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ttnn.linear(x, weight, **kwargs)

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        if ttnn.is_sharded(x):
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _decoder_masks(self, *, padded_unit_seq: int, dur_sum: int) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Cached ``pad_unit`` 2D/3D + additive 4D mask for a fixed (padded_len, valid_len) pair."""
        key = (padded_unit_seq, dur_sum)
        cached = self._decoder_mask_cache.get(key)
        if cached is not None:
            return cached
        pad_unit = _mask_row_valid_prefix(self.device, padded_unit_seq, dur_sum)
        attn_4d = _expand_4d_padding_additive_b1(self.device, pad_unit, padded_unit_seq, padded_unit_seq)
        pad_unit_3d = ttnn.reshape(pad_unit, (1, padded_unit_seq, 1))
        cached = (pad_unit, attn_4d, pad_unit_3d)
        self._decoder_mask_cache[key] = cached
        return cached

    def _decoder_self_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        attn_mask: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        token_m = batch * seq
        pc_qkv = self._matmul_pc(token_m, hidden_size, 3 * hidden_size)
        qkv = self._linear(
            hidden_states,
            attn_module.qkv.weight,
            attn_module.qkv.bias,
            program_config=pc_qkv,
        )
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq, 3 * hidden_size))

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_4d)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0 / math.sqrt(head_dim),
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq, hidden_size))

        pc_out = self._matmul_pc(token_m, hidden_size, hidden_size)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            program_config=pc_out,
        )
        ttnn.deallocate(merged)
        return proj

    def _duration_predictor(
        self, char_hidden: ttnn.Tensor, char_padding_mask_tt: ttnn.Tensor, *, seq: int
    ) -> ttnn.Tensor:
        p = self.parameters.decoder.duration_predictor
        batch = int(char_hidden.shape[0])
        mask_bc = ttnn.reshape(char_padding_mask_tt, (batch, seq, 1))
        h = ttnn.multiply(char_hidden, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        k = self.variance_predictor_kernel_size
        pad = k // 2
        h = _conv1d_same(
            self.device,
            h,
            sequence_length=seq,
            weight_rm=p.conv1.weight,
            bias_rm=p.conv1.bias,
            in_channels=self.variance_predictor_embed_dim,
            out_channels=self.variance_predictor_hidden_dim,
            kernel_size=k,
            padding=pad,
            compute_kernel_config=self._conv_compute_cfg,
            activation="relu",
            prep_cache=self._conv1d_prepared_cache,
        )
        h = self._layer_norm(h, weight=p.ln1.weight, bias=p.ln1.bias)
        h = ttnn.multiply(h, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        h = _conv1d_same(
            self.device,
            h,
            sequence_length=seq,
            weight_rm=p.conv2.weight,
            bias_rm=p.conv2.bias,
            in_channels=self.variance_predictor_hidden_dim,
            out_channels=self.variance_predictor_hidden_dim,
            kernel_size=k,
            padding=pad,
            compute_kernel_config=self._conv_compute_cfg,
            activation="relu",
            prep_cache=self._conv1d_prepared_cache,
        )
        h = self._layer_norm(h, weight=p.ln2.weight, bias=p.ln2.bias)

        # Final projection in float32 (weights uploaded as fp32 in preprocessing). Cast activations
        # once here so ``log_dur`` matches HF near rounding boundaries.
        h_fp32 = ttnn.typecast(h, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        pc_proj = self._matmul_pc(batch * seq, self.variance_predictor_hidden_dim, 1)
        log_dur = self._linear(
            h_fp32,
            p.proj.weight,
            bias=getattr(p.proj, "bias", None),
            program_config=pc_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )
        ttnn.deallocate(h_fp32)
        return log_dur

    def _decoder_layer(
        self,
        hidden: ttnn.Tensor,
        attention_mask_4d: ttnn.Tensor,
        padding_mask_1d: ttnn.Tensor,
        layer: Any,
        *,
        batch: int,
        seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        residual = hidden
        attn_out = self._decoder_self_attention(
            hidden,
            layer.self_attn,
            attention_mask_4d,
            batch=batch,
            seq=seq,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            sdpa_cfg=sdpa_cfg,
        )
        hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        hidden = self._layer_norm(
            hidden,
            weight=layer.self_attn_layer_norm.weight,
            bias=layer.self_attn_layer_norm.bias,
        )

        residual = hidden
        mask_bc = ttnn.reshape(padding_mask_1d, (batch, seq, 1))
        hidden = ttnn.multiply(hidden, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden = _conv1d_same(
            self.device,
            hidden,
            sequence_length=seq,
            weight_rm=layer.conv1.weight,
            bias_rm=layer.conv1.bias,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=7,
            padding=3,
            compute_kernel_config=self._conv_compute_cfg,
            activation="relu",
            prep_cache=self._conv1d_prepared_cache,
        )
        hidden = ttnn.multiply(hidden, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = _conv1d_same(
            self.device,
            hidden,
            sequence_length=seq,
            weight_rm=layer.conv2.weight,
            bias_rm=layer.conv2.bias,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=7,
            padding=3,
            compute_kernel_config=self._conv_compute_cfg,
            prep_cache=self._conv1d_prepared_cache,
        )
        hidden = ttnn.add(residual, hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = self._layer_norm(
            hidden,
            weight=layer.conv_layer_norm.weight,
            bias=layer.conv_layer_norm.bias,
        )
        return hidden

    def forward(
        self,
        inputs_embeds: ttnn.Tensor,
        encoder_attention_mask_4d: ttnn.Tensor,
        char_input_ids: ttnn.Tensor,
        char_count_per_id: Sequence[int],
        *,
        reference_discrete_durations: Optional[Sequence[int]] = None,
        hard_upsample_cums: Optional[T2UTraceHardUpsampleCumsums] = None,
        trace_no_profiler: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            inputs_embeds: ``[1, enc_seq, hidden]`` tile bf16 on device.
            encoder_attention_mask_4d: encoder self-attention additive mask ``[1, 1, enc_seq, enc_seq]``.
            char_input_ids: ``uint32`` ``[1, char_len]`` on device (padded).
            char_count_per_id: length-``enc_seq`` sequence of non-negative ints (sum = ``char_len``), batch 1.
            reference_discrete_durations: optional per-character integer durations (length ``char_len``),
                e.g. from [`hf_discrete_duration_counts_batch1`], to match HF unit length in PCC tests while
                the TTNN duration predictor is converged. When ``None``, durations come from the TT predictor.
            hard_upsample_cums: optional pre-tilized cumsum rows for both hard upsamples (trace capture).
                For **full** trace capture (no device writes during ``begin_trace_capture``), build the pack
                with ``make_t2u_trace_prealloc_tensors`` so ``char_frame_idx_f32`` … ``pad_unit_3d_tt``
                are populated; cumsum-only packs still use dynamic ``arange`` / masks (not trace-safe).
            trace_no_profiler: when True, skip ``ReadDeviceProfiler`` (not allowed during trace capture).

        Returns:
            ``(lm_logits, padding_mask)`` both tile bf16 on device (``padding_mask`` is ``1`` = valid),
            matching HF logits and ``padding_mask`` semantics.
        """
        dec = self.parameters.decoder
        batch = int(inputs_embeds.shape[0])
        enc_seq = int(inputs_embeds.shape[1])
        if batch != 1:
            raise NotImplementedError("batch > 1 not supported for TT text-to-unit.")

        cc_list = [int(x) for x in char_count_per_id]
        if len(cc_list) != enc_seq:
            raise ValueError(f"char_count_per_id length {len(cc_list)} must equal enc_seq {enc_seq}.")

        tb = hard_upsample_cums
        full_trace_prebuf = (
            tb is not None
            and tb.char_pad_bf16_tile is not None
            and tb.char_frame_idx_f32 is not None
            and tb.char_pos_ids is not None
            and tb.unit_frame_idx_f32 is not None
            and tb.unit_pos_ids is not None
            and tb.pad_unit_bf16_tile is not None
            and tb.attn_4d_bf16_tile is not None
            and tb.pad_unit_3d_tt is not None
        )

        enc_out = self.encoder.forward(inputs_embeds, encoder_attention_mask_4d)

        char_w = int(char_input_ids.shape[1])
        char_seq_total = int(sum(cc_list))
        if full_trace_prebuf:
            char_pad = tb.char_pad_bf16_tile
        else:
            char_pad = _mask_row_valid_prefix(self.device, char_w, char_seq_total)
            if char_seq_total < char_w:
                char_pad = ttnn.slice(char_pad, [0, 0], [1, char_seq_total], [1, 1])

        char_frame_f32 = tb.char_frame_idx_f32 if full_trace_prebuf else None
        if tb is not None:
            char_inc, char_prev = tb.char_inc, tb.char_prev
        else:
            char_inc, char_prev = self._cached_repeat_cumsums(cc_list)
        up1 = _hard_upsample_nlc(
            enc_out,
            cc_list,
            device=self.device,
            hidden_size=self.hidden_size,
            cumsum_inc_tile=char_inc,
            cumsum_prev_tile=char_prev,
            frame_idx_f32=char_frame_f32,
        )
        # Upsampled character length equals ``sum(char_count_per_id)`` (batch 1); do not rely on
        # ``up1.shape[1]`` alone — tile layout can report incorrect logical widths to Python.
        char_len = char_seq_total
        if int(up1.shape[1]) != char_len:
            raise RuntimeError(
                f"char upsample width mismatch: up1.shape[1]={int(up1.shape[1])} vs sum(char_count)={char_len}."
            )
        if not full_trace_prebuf and char_len < char_w:
            char_pad = ttnn.slice(char_pad, [0, 0], [1, char_len], [1, 1])
        elif char_len > char_w:
            raise ValueError(f"Upsampled char length {char_len} exceeds char_input_ids width {char_w}; pad HF inputs.")

        # HF-style character padding: prefix ones for real characters (matches ``_mask_row_valid_prefix``
        # + slices above when ``char_len == char_seq_total``). Avoids reading ``char_pad`` after reshape
        # views may share storage with ``char_pad_tt``.
        char_pad_valid_host = [1.0] * char_len

        if full_trace_prebuf:
            pos_ids = tb.char_pos_ids
        else:
            pos_ids = ttnn.reshape(
                ttnn.arange(
                    self.pad_token_id + 1,
                    self.pad_token_id + 1 + char_len,
                    step=1,
                    dtype=ttnn.uint32,
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                (1, char_len),
            )
        pos_emb_tt = ttnn.embedding(
            pos_ids,
            weight=dec.embed_char_positions.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        if not full_trace_prebuf:
            ttnn.deallocate(pos_ids)

        char_ids_slice = char_input_ids
        if char_w > char_len:
            char_ids_slice = ttnn.slice(char_input_ids, [0, 0], [batch, char_len], [1, 1])
        char_emb_tt = ttnn.embedding(char_ids_slice, weight=dec.embed_char.weight, layout=ttnn.TILE_LAYOUT)

        char_h = ttnn.add(
            ttnn.multiply(dec.pos_emb_alpha_char, pos_emb_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            char_emb_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(pos_emb_tt)
        ttnn.deallocate(char_emb_tt)
        char_h = ttnn.add(char_h, up1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up1)

        char_pad_tt = ttnn.reshape(char_pad, (batch, char_len, 1))

        if reference_discrete_durations is None:
            log_dur = self._duration_predictor(char_h, char_pad_tt, seq=char_len)
            dur_list = _discrete_duration_counts(log_dur, batch=batch, seq=char_len)
            ttnn.deallocate(log_dur)
        else:
            dur_list = [int(x) for x in reference_discrete_durations]
            if len(dur_list) != char_len:
                raise ValueError(f"reference_discrete_durations length {len(dur_list)} must equal char_len {char_len}.")

        for j in range(char_len):
            if j < len(char_pad_valid_host) and char_pad_valid_host[j] < 0.5:
                dur_list[j] = 0
        if not full_trace_prebuf:
            ttnn.deallocate(char_pad)

        # Drain the device profiler buffer at the natural encoder/decoder boundary.  The
        # duration readback above already syncs host<->device, so this adds no extra wait
        # in profiler builds and compiles to a no-op in normal builds.  Without it, the
        # 12000-marker on-device buffer overflows on a full forward (~1300 ops) and
        # ``python -m tracy`` fails post-run with "Op N not present in cpp_device_perf_report".
        if not trace_no_profiler:
            ttnn.ReadDeviceProfiler(self.device)

        unit_frame_f32 = tb.unit_frame_idx_f32 if full_trace_prebuf else None
        if tb is not None:
            unit_inc, unit_prev = tb.unit_inc, tb.unit_prev
        else:
            unit_inc, unit_prev = self._cached_repeat_cumsums(dur_list)
        up2 = _hard_upsample_nlc(
            char_h,
            dur_list,
            device=self.device,
            hidden_size=self.hidden_size,
            cumsum_inc_tile=unit_inc,
            cumsum_prev_tile=unit_prev,
            frame_idx_f32=unit_frame_f32,
        )
        ttnn.deallocate(char_h)
        unit_seq = int(sum(dur_list))
        if int(up2.shape[1]) != unit_seq:
            raise RuntimeError(
                f"unit upsample width mismatch: up2.shape[1]={int(up2.shape[1])} vs sum(dur_list)={unit_seq}."
            )

        if full_trace_prebuf:
            pos_ids2 = tb.unit_pos_ids
        else:
            pos_ids2 = ttnn.reshape(
                ttnn.arange(
                    self.pad_token_id + 1,
                    self.pad_token_id + 1 + unit_seq,
                    step=1,
                    dtype=ttnn.uint32,
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                (1, unit_seq),
            )
        pos2_tt = ttnn.embedding(
            pos_ids2,
            weight=dec.embed_positions.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        if not full_trace_prebuf:
            ttnn.deallocate(pos_ids2)

        hidden = ttnn.add(
            up2,
            ttnn.multiply(dec.pos_emb_alpha, pos2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(pos2_tt)
        ttnn.deallocate(up2)

        # Pad the unit sequence to a tile-aligned length so the decoder SDPA does not score real
        # queries against tile-padded garbage keys. ``pad_unit`` carries the valid-prefix length so
        # the additive mask drives padded keys to -inf and the post-SDPA gating zeros padded queries.
        dur_sum = int(sum(dur_list))
        assert dur_sum == unit_seq
        padded_unit_seq = ((unit_seq + 31) // 32) * 32
        if padded_unit_seq > unit_seq:
            if full_trace_prebuf:
                tail_tt = tb.unit_hidden_pad_tail_bf16
                if tail_tt is None:
                    raise RuntimeError(
                        "T2U trace prebuf: padded_unit_seq > unit_seq but unit_hidden_pad_tail_bf16 is None."
                    )
            else:
                tail_tt = _unit_hidden_pad_tail(
                    self.device,
                    batch=batch,
                    pad_len=padded_unit_seq - unit_seq,
                    hidden_size=self.hidden_size,
                    cache=self._unit_pad_tail_cache,
                )
            hidden = ttnn.concat([hidden, tail_tt], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if full_trace_prebuf:
            pad_unit = tb.pad_unit_bf16_tile
            attn_4d_tt = tb.attn_4d_bf16_tile
            pad_unit_tt = tb.pad_unit_3d_tt
        else:
            pad_unit, attn_4d_tt, pad_unit_tt = self._decoder_masks(padded_unit_seq=padded_unit_seq, dur_sum=dur_sum)

        num_heads = self.decoder_attention_heads
        head_dim = self.hidden_size // num_heads
        sdpa_cfg = self._sdpa_program_config(padded_unit_seq, padded_unit_seq)

        for i in range(self.decoder_layers):
            hidden = self._decoder_layer(
                hidden,
                attn_4d_tt,
                pad_unit_tt,
                dec.layers[i],
                batch=batch,
                seq=padded_unit_seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=self.hidden_size,
                sdpa_cfg=sdpa_cfg,
            )

        hidden = self._layer_norm(
            hidden,
            weight=dec.layer_norm.weight,
            bias=dec.layer_norm.bias,
        )
        vocab = int(self.parameters.lm_head.weight.shape[-1])
        pc_lm = self._matmul_pc(batch * padded_unit_seq, self.hidden_size, vocab)
        logits = self._linear(
            hidden,
            self.parameters.lm_head.weight,
            bias=None,
            program_config=pc_lm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        # Slice back to the logical unit-seq so callers see ``[..., unit_seq, vocab]``. ``ttnn.slice``
        # requires the begins/ends/strides to match the input rank, which can be 3D or 4D depending
        # on the linear output layout.
        if padded_unit_seq > unit_seq:
            logits_shape = tuple(logits.shape)
            vocab = int(logits_shape[-1])
            rank = len(logits_shape)
            begins = [0] * rank
            ends = list(logits_shape)
            ends[-2] = unit_seq
            ends[-1] = vocab
            strides = [1] * rank
            logits = ttnn.slice(logits, begins, ends, strides)

        pad_out = pad_unit
        return logits, pad_out
