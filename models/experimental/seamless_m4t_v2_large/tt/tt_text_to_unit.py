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
    all_reduce_sum_replicate,
    core_grid,
    dram_matmul_program_config,
    ensure_l1_width_sharded_activation,
    matmul_program_config,
    MATMUL_1D_SEQ_THRESHOLD,
    sdpa_program_config,
    TILE,
    to_torch_replicated_first_shard,
    width_sharded_to_l1_interleaved,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import get_tp, mesh_cluster_axis

# Chunk ``H @ enc`` along upsampled rows (same row count as speech-encoder long mel matmuls).
_HARD_UPSAMPLE_MATMUL_CHUNK_ROWS = TILE

# HF ``torch.finfo(torch.bfloat16).min`` additive padding mask floor (approx.).
_BF16_MASK_FLOOR = -3.3895313892565356e38


def _linear_token_rows(x: ttnn.Tensor) -> int:
    if len(x.shape) == 3:
        return int(x.shape[0]) * int(x.shape[1])
    if len(x.shape) == 2:
        return int(x.shape[0])
    return int(x.shape[-2])


def _weight_is_dram_width_sharded(weight: ttnn.Tensor) -> bool:
    mc = weight.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def _bias_token_rows(bias: ttnn.Tensor) -> int:
    if len(bias.shape) == 4:
        return int(bias.shape[2])
    return TILE


def _t2u_sdpa_program_config(
    device: ttnn.Device,
    seq_q: int,
    seq_k: int,
    cache: dict,
) -> ttnn.SDPAProgramConfig:
    """Capped SDPA chunks for long T2U encoder/decoder sequences (avoids L1 overflow at seq 4096)."""
    key = (seq_q, seq_k)
    cached = cache.get(key)
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
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    cache[key] = out
    return out


def _t2u_layer_norm(
    x: ttnn.Tensor,
    *,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    epsilon: float,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    long_seq: bool,
) -> ttnn.Tensor:
    """Layer norm with DRAM activations on long sequences (speech-encoder long-audio policy)."""
    if ttnn.is_sharded(x):
        mc = ttnn.DRAM_MEMORY_CONFIG if long_seq else ttnn.L1_MEMORY_CONFIG
        x = ttnn.sharded_to_interleaved(x, mc, output_dtype=ttnn.bfloat16)
    elif long_seq and x.memory_config().buffer_type != ttnn.BufferType.DRAM:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    out_mc = ttnn.DRAM_MEMORY_CONFIG if long_seq else ttnn.L1_MEMORY_CONFIG
    return ttnn.layer_norm(
        x,
        weight=weight,
        bias=bias,
        epsilon=epsilon,
        memory_config=out_mc,
        compute_kernel_config=compute_kernel_config,
    )


def _pad_token_rows(x: ttnn.Tensor, m_actual: int, m_padded: int) -> ttnn.Tensor:
    if m_actual >= m_padded:
        return x
    k = int(x.shape[-1])
    pad_rows = m_padded - m_actual
    pad = ttnn.full(
        [pad_rows, k],
        0.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=x.device(),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    padded = ttnn.concat([x, pad], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(pad)
    return padded


def _linear_dram_chunked(
    device: ttnn.Device,
    *,
    dram_matmul_pc_cache: dict,
    chunked_linear_compute_cfg: Optional[ttnn.DeviceComputeKernelConfig],
    long_seq_mc: Optional[ttnn.MemoryConfig],
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    activation: Optional[str],
    logical_out_dim: int,
    batch: int,
    seq: int,
    m_actual: int,
    m: int,
    k: int,
    n: int,
) -> tuple[ttnn.Tensor, Optional[ttnn.DeviceComputeKernelConfig]]:
    """DRAM-sharded matmul in TILE-row chunks (text-encoder long-seq path)."""
    fused_activation = ttnn.UnaryOpType.RELU if activation == "relu" else None
    pc_key = (m, k, n, fused_activation)
    pc = dram_matmul_pc_cache.get(pc_key)
    if pc is None:
        pc = dram_matmul_program_config(device, m, k, n, fused_activation=fused_activation)
        dram_matmul_pc_cache[pc_key] = pc

    if chunked_linear_compute_cfg is None:
        chunked_linear_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    if ttnn.is_sharded(x):
        x_inter = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
    else:
        x_inter = x
    if len(x_inter.shape) == 3:
        x_inter = ttnn.reshape(x_inter, (m_actual, k))
    elif len(x_inter.shape) != 2:
        x_inter = ttnn.reshape(x_inter, (m_actual, k))

    chunks: list[ttnn.Tensor] = []
    num_chunks = (m_actual + m - 1) // m
    for i in range(num_chunks):
        start = i * m
        end = min(start + m, m_actual)
        chunk_rows = end - start

        chunk = ttnn.slice(x_inter, [start, 0], [end, k], [1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        if chunk_rows < m:
            chunk = _pad_token_rows(chunk, chunk_rows, m)
        chunk_sharded = ensure_l1_width_sharded_activation(device, chunk, m, k, n)
        out_sharded = ttnn.linear(
            chunk_sharded,
            weight,
            bias=bias,
            program_config=pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=chunked_linear_compute_cfg,
        )
        if chunk_sharded is not chunk:
            ttnn.deallocate(chunk_sharded)
        if chunk is not x_inter:
            ttnn.deallocate(chunk)
        out_inter = width_sharded_to_l1_interleaved(out_sharded)
        if long_seq_mc is ttnn.DRAM_MEMORY_CONFIG:
            out_dram = ttnn.to_memory_config(out_inter, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out_inter)
            out_inter = out_dram
        ttnn.deallocate(out_sharded)
        if len(out_inter.shape) == 4 and int(out_inter.shape[1]) == 1:
            out_inter = ttnn.reshape(out_inter, (int(out_inter.shape[2]), int(out_inter.shape[-1])))
        if chunk_rows < m:
            trim_mc = long_seq_mc or ttnn.L1_MEMORY_CONFIG
            out_inter = ttnn.slice(
                out_inter,
                [0, 0],
                [chunk_rows, int(out_inter.shape[-1])],
                [1, 1],
                memory_config=trim_mc,
            )
        chunks.append(out_inter)

    concat_mc = long_seq_mc or ttnn.L1_MEMORY_CONFIG
    if len(chunks) == 1:
        out_concat = chunks[0]
        if concat_mc is ttnn.DRAM_MEMORY_CONFIG and out_concat.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out_dram = ttnn.to_memory_config(out_concat, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out_concat)
            out_concat = out_dram
    else:
        out_concat = ttnn.concat(chunks, dim=0, memory_config=concat_mc)
        for c in chunks:
            ttnn.deallocate(c)

    padded_n = int(out_concat.shape[-1])
    if padded_n > logical_out_dim:
        out_concat = ttnn.slice(out_concat, [0, 0], [m_actual, logical_out_dim], [1, 1], memory_config=concat_mc)
    out = ttnn.reshape(out_concat, (batch, seq, logical_out_dim))
    return out, chunked_linear_compute_cfg


def _linear_matmul_1d_chunked(
    device: ttnn.Device,
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor],
    *,
    activation: Optional[str],
    batch: int,
    seq: int,
    m_actual: int,
    k: int,
    n: int,
    logical_out_dim: int,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
) -> ttnn.Tensor:
    """Chunked ``ttnn.linear`` with 1D multicast (interleaved weights, DRAM activations)."""
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (batch, seq, k))
    elif len(x.shape) == 2:
        x = ttnn.reshape(x, (batch, seq, k))

    chunk_m = TILE
    chunks: list[ttnn.Tensor] = []
    for start in range(0, m_actual, chunk_m):
        end = min(start + chunk_m, m_actual)
        chunk_rows = end - start
        x_chunk = ttnn.slice(
            x,
            [0, start, 0],
            [batch, end, k],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pc = matmul_program_config(device, token_rows=chunk_rows, in_dim=k, out_dim=n)
        out_chunk = ttnn.linear(
            x_chunk,
            weight,
            bias=bias,
            activation=activation,
            program_config=pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(x_chunk)
        if chunk_rows < chunk_m:
            out_chunk = ttnn.slice(
                out_chunk,
                [0, 0, 0],
                [batch, chunk_rows, int(out_chunk.shape[-1])],
                [1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        chunks.append(out_chunk)

    if len(chunks) == 1:
        out = chunks[0]
    else:
        out = ttnn.concat(chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for c in chunks:
            ttnn.deallocate(c)

    padded_n = int(out.shape[-1])
    if padded_n > logical_out_dim:
        rank = len(out.shape)
        begins = [0] * rank
        ends = list(out.shape)
        ends[-2] = m_actual
        ends[-1] = logical_out_dim
        strides = [1] * rank
        out = ttnn.slice(out, begins, ends, strides, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


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


def _conv1d_config_for(
    *,
    sequence_length: int,
    in_channels: int,
    activation: Optional[str] = None,
) -> ttnn.Conv1dConfig:
    conv_kwargs: dict = dict(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=None,
        deallocate_activation=True,
        enable_weights_double_buffer=True,
        enable_act_double_buffer=True,
    )
    if activation:
        _ACTIVATION_OP_TYPES = {
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
            "gelu": ttnn.UnaryOpType.GELU,
        }
        op_type = _ACTIVATION_OP_TYPES.get(activation.lower())
        if op_type is None:
            raise ValueError(f"_conv1d_config_for: unsupported activation {activation!r}")
        conv_kwargs["activation"] = ttnn.UnaryWithParam(op_type)
    if sequence_length > 64 or in_channels >= 512:
        conv_kwargs["act_block_h_override"] = 32
    if sequence_length > 256:
        conv_kwargs["enable_act_double_buffer"] = False
        conv_kwargs["enable_weights_double_buffer"] = False
    return ttnn.Conv1dConfig(**conv_kwargs)


def _conv1d_prep_cache_key(
    *,
    weight_rm: ttnn.Tensor,
    bias_rm: Optional[ttnn.Tensor],
    batch: int,
    sequence_length: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    activation: Optional[str] = None,
) -> Tuple[Any, ...]:
    act_block_h = 32 if (sequence_length > 64 or in_channels >= 512) else 0
    return (
        _conv1d_prep_tensor_id(weight_rm),
        _conv1d_prep_tensor_id(bias_rm) if bias_rm is not None else 0,
        batch,
        sequence_length,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation or "",
        act_block_h,
    )


def _prepare_conv1d_weights_on_device(
    device: ttnn.Device,
    *,
    weight_rm: ttnn.Tensor,
    bias_rm: Optional[ttnn.Tensor],
    batch: int,
    sequence_length: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    activation: Optional[str] = None,
    prep_cache: Dict[Tuple[Any, ...], Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]],
) -> None:
    """``prepare_conv_weights`` + DRAM ``clone`` only — no Conv2d forward (speech-encoder recipe)."""
    cache_key = _conv1d_prep_cache_key(
        weight_rm=weight_rm,
        bias_rm=bias_rm,
        batch=batch,
        sequence_length=sequence_length,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
    )
    if cache_key in prep_cache:
        return

    conv_config = _conv1d_config_for(
        sequence_length=sequence_length,
        in_channels=in_channels,
        activation=activation,
    )
    prep_w = ttnn.prepare_conv_weights(
        weight_tensor=weight_rm,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch,
        input_height=1,
        input_width=sequence_length,
        kernel_size=(1, kernel_size),
        stride=(1, 1),
        padding=(0, padding),
        dilation=(1, 1),
        has_bias=bias_rm is not None,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_kernel_config,
    )
    prep_b = None
    if bias_rm is not None:
        prep_b = ttnn.prepare_conv_bias(
            bias_tensor=bias_rm,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch,
            input_height=1,
            input_width=sequence_length,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, padding),
            dilation=(1, 1),
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=compute_kernel_config,
        )
    prep_cache[cache_key] = (
        ttnn.clone(prep_w, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.clone(prep_b, memory_config=ttnn.DRAM_MEMORY_CONFIG) if prep_b is not None else None,
    )


def _cached_frame_idx_f32(
    device: ttnn.Device,
    sum_r: int,
    cache: Dict[int, ttnn.Tensor],
) -> ttnn.Tensor:
    key = int(sum_r)
    cached = cache.get(key)
    if cached is not None:
        return cached
    frame_idx = ttnn.arange(
        start=0,
        end=key,
        step=1,
        dtype=ttnn.float32,
        device=device,
    )
    cache[key] = frame_idx
    return frame_idx


def _hard_upsample_matmul(
    H: ttnn.Tensor,
    enc: ttnn.Tensor,
    device: ttnn.Device,
    *,
    sum_r: int,
    enc_seq: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """``H @ enc`` with 1D multicast chunks when ``sum_r`` exceeds the 2D L1 budget."""
    if sum_r <= MATMUL_1D_SEQ_THRESHOLD:
        mm_pc = matmul_program_config(device, token_rows=sum_r, in_dim=enc_seq, out_dim=hidden_size)
        return ttnn.matmul(H, enc, program_config=mm_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    chunk_m = _HARD_UPSAMPLE_MATMUL_CHUNK_ROWS
    chunks: list[ttnn.Tensor] = []
    for start in range(0, sum_r, chunk_m):
        end = min(start + chunk_m, sum_r)
        chunk_rows = end - start
        H_chunk = ttnn.slice(
            H,
            [0, start, 0],
            [1, end, enc_seq],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mm_pc = matmul_program_config(
            device,
            token_rows=chunk_rows,
            in_dim=enc_seq,
            out_dim=hidden_size,
        )
        out_chunk = ttnn.matmul(H_chunk, enc, program_config=mm_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(H_chunk)
        if chunk_rows < chunk_m:
            out_chunk = ttnn.slice(
                out_chunk,
                [0, 0, 0],
                [1, chunk_rows, hidden_size],
                [1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        chunks.append(out_chunk)

    if len(chunks) == 1:
        return chunks[0]
    out = ttnn.concat(chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for c in chunks:
        ttnn.deallocate(c)
    return out


def _hard_upsample_nlc(
    enc: ttnn.Tensor,
    repeats: Sequence[int],
    *,
    device: ttnn.Device,
    hidden_size: int,
    cumsum_inc_tile: Optional[ttnn.Tensor] = None,
    cumsum_prev_tile: Optional[ttnn.Tensor] = None,
    frame_idx_f32: Optional[ttnn.Tensor] = None,
    frame_idx_cache: Optional[Dict[int, ttnn.Tensor]] = None,
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
    elif frame_idx_cache is not None:
        frame_idx = _cached_frame_idx_f32(device, sum_r, frame_idx_cache)
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
    hidden_size = int(enc.shape[-1])
    out = _hard_upsample_matmul(
        H,
        enc,
        device,
        sum_r=sum_r,
        enc_seq=enc_seq,
        hidden_size=hidden_size,
    )
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


# Chunk wide decoder conv along sequence when single-shot L1 still overflows (hidden=1024, seq=4096).
_CONV1D_CHUNK_ROWS = MATMUL_1D_SEQ_THRESHOLD


def _conv1d_out_to_nlc(out: ttnn.Tensor, *, batch: int, out_channels: int) -> ttnn.Tensor:
    """Normalize ``ttnn.conv1d`` output to NLC ``[batch, seq, C]``."""
    rank = len(out.shape)
    if rank == 3:
        return out
    if rank == 4 and int(out.shape[1]) == 1:
        return ttnn.reshape(out, (batch, int(out.shape[2]), out_channels))
    if rank == 4:
        return ttnn.reshape(out, (batch, int(out.shape[1]), out_channels))
    raise RuntimeError(f"Unexpected conv1d output rank {rank}, shape {tuple(out.shape)}")


def _conv1d_same_impl(
    device: ttnn.Device,
    x_rm: ttnn.Tensor,
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
    deallocate_input: bool = False,
) -> ttnn.Tensor:
    """Single-shot same-padding Conv1d (``x_rm`` is NLC row-major)."""
    batch = int(x_rm.shape[0])
    seq = int(sequence_length)
    conv_config = _conv1d_config_for(
        sequence_length=seq,
        in_channels=in_channels,
        activation=activation,
    )
    if prep_cache is not None:
        cache_key = _conv1d_prep_cache_key(
            weight_rm=weight_rm,
            bias_rm=bias_rm,
            batch=batch,
            sequence_length=seq,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
        )
        cached = prep_cache.get(cache_key)
        if cached is None:
            _prepare_conv1d_weights_on_device(
                device,
                weight_rm=weight_rm,
                bias_rm=bias_rm,
                batch=batch,
                sequence_length=seq,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                compute_kernel_config=compute_kernel_config,
                activation=activation,
                prep_cache=prep_cache,
            )
            cached = prep_cache[cache_key]
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
    if deallocate_input:
        ttnn.deallocate(x_rm)
    out_nlc = _conv1d_out_to_nlc(out_tt, batch=batch, out_channels=out_channels)
    if out_nlc is not out_tt:
        ttnn.deallocate(out_tt)
    if out_nlc.get_layout() != ttnn.TILE_LAYOUT:
        out_tile = ttnn.to_layout(out_nlc, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_nlc)
        return out_tile
    return out_nlc


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

    For ``sequence_length > 256``, ``_conv1d_config_for`` disables conv double-buffering.
    Wide decoder conv (``in_channels >= 512``) at longer seq is additionally tiled along S.
    """
    batch = int(x_tile.shape[0])
    seq = int(sequence_length)
    if x_tile.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        x_rm = ttnn.to_layout(x_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_tile)
    else:
        x_rm = x_tile

    if seq <= _CONV1D_CHUNK_ROWS or in_channels < 512:
        return _conv1d_same_impl(
            device,
            x_rm,
            sequence_length=seq,
            weight_rm=weight_rm,
            bias_rm=bias_rm,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            compute_kernel_config=compute_kernel_config,
            activation=activation,
            prep_cache=prep_cache,
        )

    chunk_size = 256 if in_channels >= 512 else _CONV1D_CHUNK_ROWS
    chunks: list[ttnn.Tensor] = []
    for start in range(0, seq, chunk_size):
        end = min(start + chunk_size, seq)
        chunk_rows = end - start
        in_start = max(0, start - padding)
        in_end = min(seq, end + padding)
        win_len = in_end - in_start
        x_win = ttnn.slice(
            x_rm,
            [0, in_start, 0],
            [batch, in_end, in_channels],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_win = _conv1d_same_impl(
            device,
            x_win,
            sequence_length=win_len,
            weight_rm=weight_rm,
            bias_rm=bias_rm,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            compute_kernel_config=compute_kernel_config,
            activation=activation,
            prep_cache=prep_cache,
            deallocate_input=True,
        )
        out_start = start - in_start
        if out_start > 0 or chunk_rows < win_len:
            out_chunk = ttnn.slice(
                out_win,
                [0, out_start, 0],
                [batch, out_start + chunk_rows, out_channels],
                [1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(out_win)
        else:
            out_chunk = out_win
        chunks.append(out_chunk)

    ttnn.deallocate(x_rm)
    if len(chunks) == 1:
        return chunks[0]
    out = ttnn.concat(chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for c in chunks:
        ttnn.deallocate(c)
    return out


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
        # TP: number of devices participating in tensor parallelism.
        self._tp = get_tp(device)
        self._cluster_axis = mesh_cluster_axis(device)
        # local head count = total_heads / tp (for single device tp=1, unchanged).
        self._num_local_heads = num_attention_heads // self._tp
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
        self._dram_matmul_pc_cache: dict = {}
        self._chunked_linear_compute_cfg: Optional[ttnn.DeviceComputeKernelConfig] = None
        self._long_seq_mc: Optional[ttnn.MemoryConfig] = None

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
        batch: Optional[int] = None,
        seq: Optional[int] = None,
    ) -> ttnn.Tensor:
        k = int(weight.shape[-2])
        n = int(weight.shape[-1])
        logical_out_dim = n

        if len(x.shape) == 3:
            batch = batch if batch is not None else int(x.shape[0])
            seq = seq if seq is not None else int(x.shape[1])
            m_actual = batch * seq
        elif len(x.shape) == 2:
            m_actual = int(x.shape[0])
            batch = batch if batch is not None else 1
            seq = seq if seq is not None else m_actual
        else:
            m_actual = _linear_token_rows(x)
            batch = batch if batch is not None else 1
            seq = seq if seq is not None else m_actual

        m = _bias_token_rows(bias) if bias is not None else TILE
        if m_actual > m or _weight_is_dram_width_sharded(weight):
            out, self._chunked_linear_compute_cfg = _linear_dram_chunked(
                self.device,
                dram_matmul_pc_cache=self._dram_matmul_pc_cache,
                chunked_linear_compute_cfg=self._chunked_linear_compute_cfg,
                long_seq_mc=self._long_seq_mc,
                x=x,
                weight=weight,
                bias=bias,
                activation=activation,
                logical_out_dim=logical_out_dim,
                batch=batch,
                seq=seq,
                m_actual=m_actual,
                m=m,
                k=k,
                n=n,
            )
            return out

        if program_config is None:
            program_config = self._matmul_pc(m_actual, k, n)
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
        long_seq = self._long_seq_mc is ttnn.DRAM_MEMORY_CONFIG
        return _t2u_layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            compute_kernel_config=self._linear_ln_compute_cfg,
            long_seq=long_seq,
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
        """TP-aware self-attention.

        For TP>1 the preprocessed QKV weight is column-parallel ``[H, 3H//tp]`` and
        out_proj is row-parallel ``[H//tp, H]``. ``num_heads`` must already be
        ``num_local_heads = total_heads // tp``.
        """
        token_m = batch * seq
        # Read TP-local QKV output dim from weight shape (works for tp=1 and tp>1).
        qkv_out_dim = int(attn_module.qkv.weight.shape[-1])
        pc_qkv = self._matmul_pc(token_m, hidden_size, qkv_out_dim)
        qkv = self._linear(
            hidden_states,
            attn_module.qkv.weight,
            attn_module.qkv.bias,
            program_config=pc_qkv,
            batch=batch,
            seq=seq,
        )
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq, qkv_out_dim))

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
        # For TP>1: merged dim = num_local_heads * head_dim = hidden_size // tp.
        local_hidden = num_heads * head_dim
        merged = ttnn.reshape(merged_4d, (batch, seq, local_hidden))

        local_in_dim = int(attn_module.out_proj.weight.shape[-2])  # H or H//tp
        pc_out = self._matmul_pc(token_m, local_in_dim, hidden_size)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            program_config=pc_out,
            batch=batch,
            seq=seq,
        )
        ttnn.deallocate(merged)
        # TP all_reduce: row-parallel out_proj gives partial sums; sum to full hidden.
        if self._tp > 1:
            proj = all_reduce_sum_replicate(proj, self.device, cluster_axis=self._cluster_axis)
        return proj

    def forward(self, inputs_embeds: ttnn.Tensor, attention_mask_4d: ttnn.Tensor) -> ttnn.Tensor:
        parameters = self.parameters
        # For TP>1 use local head count; head_dim is per-head and does not change.
        num_heads = self._num_local_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // self.num_attention_heads
        num_layers = self.num_hidden_layers

        batch = int(inputs_embeds.shape[0])
        seq = int(inputs_embeds.shape[1])

        self._long_seq_mc = ttnn.DRAM_MEMORY_CONFIG if seq > TILE else None

        hidden = inputs_embeds
        sdpa_cfg = self._sdpa_program_config(seq, seq)
        token_m = batch * seq
        # For TP>1 fc1 is column-parallel → local ffn_dim = ffn_dim // tp.
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
                batch=batch,
                seq=seq,
            )
            ttnn.deallocate(normed)
            ff = self._linear(
                ff,
                layer.ffn.fc2.weight,
                layer.ffn.fc2.bias,
                program_config=pc_ffn_fc2,
                batch=batch,
                seq=seq,
            )
            # TP all_reduce: row-parallel fc2 gives partial sums; sum to full hidden.
            if self._tp > 1:
                ff = all_reduce_sum_replicate(ff, self.device, cluster_axis=self._cluster_axis)
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
        # TP: number of devices in tensor-parallel group.
        self._tp = get_tp(device)
        self._cluster_axis = mesh_cluster_axis(device)
        self._num_local_dec_heads = decoder_attention_heads // self._tp

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
        self._dram_matmul_pc_cache: dict = {}
        self._chunked_linear_compute_cfg: Optional[ttnn.DeviceComputeKernelConfig] = None
        self._long_seq_mc: Optional[ttnn.MemoryConfig] = None
        self._unit_pad_tail_cache: Dict[Tuple[int, int, int], ttnn.Tensor] = {}
        self._decoder_mask_cache: Dict[Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]] = {}
        self._repeat_cum_cache: Dict[Tuple[int, ...], Tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._frame_idx_cache: Dict[int, ttnn.Tensor] = {}
        self._char_prefill_cache: Dict[Tuple[int, int, int], Tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._lm_head_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _cached_repeat_cumsums(self, repeats: Sequence[int]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        key = tuple(int(r) for r in repeats)
        cached = self._repeat_cum_cache.get(key)
        if cached is not None:
            return cached
        inc, prev = _upload_repeat_cumsum_tiles(self.device, repeats)
        self._repeat_cum_cache[key] = (inc, prev)
        return inc, prev

    def prewarm_conv1d_weights(self, *, char_len: int, padded_unit_seq: int) -> None:
        """Prepare conv1d weights for duration + decoder stacks (host upload only, no Conv2d forward).

        Call before trace capture so replay avoids first-hit weight prep inside ``ttnn.conv1d``.
        """
        batch = 1
        p = self.parameters.decoder.duration_predictor
        dec = self.parameters.decoder
        k = self.variance_predictor_kernel_size
        pad = k // 2
        hidden = self.hidden_size

        def _prep(
            *,
            weight_rm: ttnn.Tensor,
            bias_rm: Optional[ttnn.Tensor],
            seq: int,
            in_ch: int,
            out_ch: int,
            kernel: int,
            padding: int,
            act: Optional[str],
        ) -> None:
            _prepare_conv1d_weights_on_device(
                self.device,
                weight_rm=weight_rm,
                bias_rm=bias_rm,
                batch=batch,
                sequence_length=seq,
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel,
                padding=padding,
                compute_kernel_config=self._conv_compute_cfg,
                activation=act,
                prep_cache=self._conv1d_prepared_cache,
            )

        _prep(
            weight_rm=p.conv1.weight,
            bias_rm=p.conv1.bias,
            seq=char_len,
            in_ch=self.variance_predictor_embed_dim,
            out_ch=self.variance_predictor_hidden_dim,
            kernel=k,
            padding=pad,
            act="relu",
        )
        _prep(
            weight_rm=p.conv2.weight,
            bias_rm=p.conv2.bias,
            seq=char_len,
            in_ch=self.variance_predictor_hidden_dim,
            out_ch=self.variance_predictor_hidden_dim,
            kernel=k,
            padding=pad,
            act="relu",
        )
        for layer in dec.layers:
            _prep(
                weight_rm=layer.conv1.weight,
                bias_rm=layer.conv1.bias,
                seq=padded_unit_seq,
                in_ch=hidden,
                out_ch=hidden,
                kernel=7,
                padding=3,
                act="relu",
            )
            _prep(
                weight_rm=layer.conv2.weight,
                bias_rm=layer.conv2.bias,
                seq=padded_unit_seq,
                in_ch=hidden,
                out_ch=hidden,
                kernel=7,
                padding=3,
                act=None,
            )

    def _cached_char_prefill(self, *, char_w: int, char_len: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Reuse char padding mask + position ids when ``(char_w, char_len, pad_token)`` is stable."""
        key = (char_w, char_len, int(self.pad_token_id))
        cached = self._char_prefill_cache.get(key)
        if cached is not None:
            return cached
        char_pad = _mask_row_valid_prefix(self.device, char_w, char_len)
        if char_len < char_w:
            char_pad = ttnn.slice(char_pad, [0, 0], [1, char_len], [1, 1])
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
        cached = (char_pad, pos_ids)
        self._char_prefill_cache[key] = cached
        return cached

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        return _t2u_sdpa_program_config(self.device, seq_q, seq_k, self._sdpa_pc_cache)

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
        batch: Optional[int] = None,
        seq: Optional[int] = None,
    ) -> ttnn.Tensor:
        k = int(weight.shape[-2])
        n = int(weight.shape[-1])
        logical_out_dim = n

        if len(x.shape) == 3:
            batch = batch if batch is not None else int(x.shape[0])
            seq = seq if seq is not None else int(x.shape[1])
            m_actual = batch * seq
        elif len(x.shape) == 2:
            m_actual = int(x.shape[0])
            batch = batch if batch is not None else 1
            seq = seq if seq is not None else m_actual
        else:
            m_actual = _linear_token_rows(x)
            batch = batch if batch is not None else 1
            seq = seq if seq is not None else m_actual

        m = _bias_token_rows(bias) if bias is not None else TILE
        if dtype is None and (m_actual > m or _weight_is_dram_width_sharded(weight)):
            out, self._chunked_linear_compute_cfg = _linear_dram_chunked(
                self.device,
                dram_matmul_pc_cache=self._dram_matmul_pc_cache,
                chunked_linear_compute_cfg=self._chunked_linear_compute_cfg,
                long_seq_mc=self._long_seq_mc,
                x=x,
                weight=weight,
                bias=bias,
                activation=activation,
                logical_out_dim=logical_out_dim,
                batch=batch,
                seq=seq,
                m_actual=m_actual,
                m=m,
                k=k,
                n=n,
            )
            return out

        if program_config is None:
            program_config = self._matmul_pc(m_actual, k, n)
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
        long_seq = self._long_seq_mc is ttnn.DRAM_MEMORY_CONFIG
        return _t2u_layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            compute_kernel_config=self._linear_ln_compute_cfg,
            long_seq=long_seq,
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
        """TP-aware decoder self-attention.

        ``num_heads`` must already be ``num_local_heads = total_heads // tp``.
        QKV weight is column-parallel ``[H, 3H//tp]``; out_proj is row-parallel ``[H//tp, H]``.
        """
        token_m = batch * seq
        qkv_out_dim = int(attn_module.qkv.weight.shape[-1])
        pc_qkv = self._matmul_pc(token_m, hidden_size, qkv_out_dim)
        qkv = self._linear(
            hidden_states,
            attn_module.qkv.weight,
            attn_module.qkv.bias,
            program_config=pc_qkv,
            batch=batch,
            seq=seq,
        )
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq, qkv_out_dim))

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
        # For TP>1: local hidden = num_local_heads * head_dim = hidden_size // tp.
        local_hidden = num_heads * head_dim
        merged = ttnn.reshape(merged_4d, (batch, seq, local_hidden))

        local_in_dim = int(attn_module.out_proj.weight.shape[-2])
        pc_out = self._matmul_pc(token_m, local_in_dim, hidden_size)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            program_config=pc_out,
            batch=batch,
            seq=seq,
        )
        ttnn.deallocate(merged)
        # TP all_reduce: sum partial row-parallel results across devices.
        if self._tp > 1:
            proj = all_reduce_sum_replicate(proj, self.device, cluster_axis=self._cluster_axis)
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
            char_pad, _cached_pos = self._cached_char_prefill(char_w=char_w, char_len=char_seq_total)

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
            frame_idx_cache=None if full_trace_prebuf else self._frame_idx_cache,
        )
        # Upsampled character length equals ``sum(char_count_per_id)`` (batch 1); do not rely on
        # ``up1.shape[1]`` alone — tile layout can report incorrect logical widths to Python.
        char_len = char_seq_total
        if int(up1.shape[1]) != char_len:
            raise RuntimeError(
                f"char upsample width mismatch: up1.shape[1]={int(up1.shape[1])} vs sum(char_count)={char_len}."
            )
        elif char_len > char_w:
            raise ValueError(f"Upsampled char length {char_len} exceeds char_input_ids width {char_w}; pad HF inputs.")

        # HF-style character padding: prefix ones for real characters (matches ``_mask_row_valid_prefix``
        # + slices above when ``char_len == char_seq_total``). Avoids reading ``char_pad`` after reshape
        # views may share storage with ``char_pad_tt``.
        char_pad_valid_host = [1.0] * char_len

        if full_trace_prebuf:
            pos_ids = tb.char_pos_ids
        else:
            _, pos_ids = self._cached_char_prefill(char_w=char_w, char_len=char_len)
        pos_emb_tt = ttnn.embedding(
            pos_ids,
            weight=dec.embed_char_positions.weight,
            layout=ttnn.TILE_LAYOUT,
        )

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
        if not trace_no_profiler:
            # Drain profiler when tracing (no-op otherwise).
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
            frame_idx_cache=None if full_trace_prebuf else self._frame_idx_cache,
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
        self._long_seq_mc = ttnn.DRAM_MEMORY_CONFIG if max(char_len, unit_seq) > TILE else None
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

        # For TP>1 use local head count; head_dim is per-head and does not change.
        num_heads = self._num_local_dec_heads
        head_dim = self.hidden_size // self.decoder_attention_heads
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
        lm_weight = self.parameters.lm_head.weight
        if batch * padded_unit_seq > TILE:
            logits = _linear_matmul_1d_chunked(
                self.device,
                hidden,
                lm_weight,
                None,
                activation=None,
                batch=batch,
                seq=padded_unit_seq,
                m_actual=batch * padded_unit_seq,
                k=self.hidden_size,
                n=vocab,
                logical_out_dim=vocab,
                compute_kernel_config=self._lm_head_compute_cfg,
            )
        else:
            pc_lm = self._matmul_pc(batch * padded_unit_seq, self.hidden_size, vocab)
            lm_kwargs: dict = dict(
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self._lm_head_compute_cfg,
            )
            if pc_lm is not None:
                lm_kwargs["program_config"] = pc_lm
            else:
                lm_kwargs["core_grid"] = core_grid(self.device)
            logits = ttnn.linear(hidden, lm_weight, bias=None, **lm_kwargs)
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

        # Return a copy so callers may ``deallocate`` without invalidating ``_decoder_mask_cache``.
        pad_out = ttnn.clone(pad_unit, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, pad_out
