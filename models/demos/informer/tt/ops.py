# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn

from .config import TILE_SIZE, InformerConfig, align_to_tile

_ARANGE_CACHE: dict[tuple[int, int, int, int, ttnn.DataType, ttnn.Layout], ttnn.Tensor] = {}
_SUBTRACT_WEIGHT_CACHE: dict[tuple[int, ttnn.DataType], ttnn.Tensor] = {}


def get_cached_arange(
    *,
    start: int = 0,
    end: int,
    step: int = 1,
    device,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
) -> ttnn.Tensor:
    key = (id(device), int(start), int(end), int(step), dtype, layout)
    if key not in _ARANGE_CACHE:
        _ARANGE_CACHE[key] = ttnn.arange(start=start, end=end, step=step, device=device, dtype=dtype, layout=layout)
    return _ARANGE_CACHE[key]


def get_cached_subtract_weight(*, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    key = (id(device), dtype)
    if key not in _SUBTRACT_WEIGHT_CACHE:
        weight = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        _SUBTRACT_WEIGHT_CACHE[key] = ttnn.from_torch(weight, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return _SUBTRACT_WEIGHT_CACHE[key]


def compute_sparsity_float32(
    max_scores: ttnn.Tensor,
    mean_scores: ttnn.Tensor,
    *,
    device,
) -> ttnn.Tensor:
    batch, heads, q_len = max_scores.shape
    max_rm = ttnn.to_layout(max_scores, ttnn.ROW_MAJOR_LAYOUT)
    mean_rm = ttnn.to_layout(mean_scores, ttnn.ROW_MAJOR_LAYOUT)
    max_rm = ttnn.reshape(max_rm, (batch * heads * q_len, 1))
    mean_rm = ttnn.reshape(mean_rm, (batch * heads * q_len, 1))
    stacked = ttnn.concat([max_rm, mean_rm], dim=1)
    stacked = ttnn.to_layout(stacked, ttnn.TILE_LAYOUT)
    weight = get_cached_subtract_weight(device=device, dtype=ttnn.float32)
    diff = linear(stacked, weight, bias=None, dtype=ttnn.float32)
    diff = ttnn.to_layout(diff, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(diff, (batch * heads, q_len))


def mask_invalid_queries_float32(
    sparsity_rm: ttnn.Tensor,
    *,
    q_valid_len: int,
    mask_value: float,
) -> ttnn.Tensor:
    q_len = sparsity_rm.shape[1]
    if q_valid_len >= q_len:
        return sparsity_rm
    device = sparsity_rm.device()
    invalid_len = q_len - q_valid_len
    idx = get_cached_arange(
        start=q_valid_len,
        end=q_len,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    idx = ttnn.reshape(idx, (1, invalid_len))
    idx = ttnn.repeat(idx, (sparsity_rm.shape[0], 1))
    mask_full = ttnn.add(ttnn.mul(sparsity_rm, 0.0), float(mask_value))
    mask_full = ttnn.to_layout(mask_full, ttnn.ROW_MAJOR_LAYOUT)
    slice_start = [0, q_valid_len]
    slice_end = [mask_full.shape[0], q_len]
    src = ttnn.slice(mask_full, slice_start, slice_end)
    return ttnn.scatter(sparsity_rm, dim=1, index=idx, src=src)


def to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x)


def make_causal_mask(
    length: int, *, batch: int, heads: int, device, dtype: ttnn.DataType, mask_value: float
) -> ttnn.Tensor:
    idx = get_cached_arange(end=length, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    row = ttnn.reshape(idx, (length, 1))
    col = ttnn.reshape(idx, (1, length))
    row = ttnn.repeat(row, (1, length))
    col = ttnn.repeat(col, (length, 1))
    mask = ttnn.gt(col, row, dtype=ttnn.bfloat16)
    mask = ttnn.mul(mask, float(mask_value))
    mask = ttnn.reshape(mask, (1, 1, length, length))
    if batch > 1 or heads > 1:
        mask = ttnn.repeat(mask, (batch, heads, 1, 1))
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    if dtype != mask.dtype:
        mask = ttnn.typecast(mask, dtype)
    return mask


def make_causal_mask_with_offset(
    q_length: int,
    k_length: int,
    past_length: int,
    *,
    batch: int,
    heads: int,
    device,
    dtype: ttnn.DataType,
    mask_value: float,
) -> ttnn.Tensor:
    q_idx = get_cached_arange(end=q_length, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    q_idx = ttnn.add(q_idx, float(past_length))
    q_idx = ttnn.reshape(q_idx, (1, 1, q_length, 1))

    k_idx = get_cached_arange(end=k_length, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_idx = ttnn.reshape(k_idx, (1, 1, 1, k_length))

    mask = ttnn.gt(k_idx, q_idx, dtype=ttnn.bfloat16)
    mask = ttnn.mul(mask, float(mask_value))
    if batch > 1 or heads > 1:
        mask = ttnn.repeat(mask, (batch, heads, 1, 1))
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    if dtype != mask.dtype:
        mask = ttnn.typecast(mask, dtype)
    return mask


def make_padding_mask(
    *,
    valid_length: int,
    key_length: int,
    batch: int,
    heads: int,
    device,
    dtype: ttnn.DataType,
    mask_value: float,
) -> ttnn.Tensor:
    idx = get_cached_arange(end=key_length, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    valid_len = ttnn.add(ttnn.mul(idx, 0.0), float(valid_length))
    valid = ttnn.lt(idx, valid_len, dtype=ttnn.bfloat16)
    valid = ttnn.reshape(valid, (1, 1, 1, key_length))
    invalid = ttnn.add(ttnn.mul(valid, -1.0), 1.0)
    mask = ttnn.mul(invalid, float(mask_value))
    if batch > 1 or heads > 1:
        mask = ttnn.repeat(mask, (batch, heads, 1, 1))
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    if dtype != mask.dtype:
        mask = ttnn.typecast(mask, dtype)
    return mask


def sinusoidal_position_encoding(length: int, d_model: int, *, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros((length, d_model), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return ttnn.from_torch(pe, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def pad_to_multiple(x: ttnn.Tensor, *, dim: int, multiple: int, value: float = 0.0) -> Tuple[ttnn.Tensor, int]:
    length = x.shape[dim]
    pad = (multiple - (length % multiple)) % multiple
    if pad == 0:
        return x, length
    padding = [(0, 0)] * len(x.shape)
    padding[dim] = (0, pad)
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.pad(x, padding, value=value, memory_config=ttnn.DRAM_MEMORY_CONFIG), length


def pad_last_dim_to_multiple(x: ttnn.Tensor, *, multiple: int, value: float = 0.0) -> Tuple[ttnn.Tensor, int]:
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    dim = len(x_rm.shape) - 1
    x_rm, length = pad_to_multiple(x_rm, dim=dim, multiple=multiple, value=value)
    x = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    return x, length


def slice_to_length(x: ttnn.Tensor, *, dim: int, length: int) -> ttnn.Tensor:
    if x.shape[dim] == length:
        return x
    slice_start = [0] * len(x.shape)
    slice_end = list(x.shape)
    slice_end[dim] = length
    return ttnn.slice(x, slice_start, slice_end)


def slice_sequence(x: ttnn.Tensor, *, dim: int, start: int, end: int) -> ttnn.Tensor:
    if start == 0 and end == x.shape[dim]:
        return x
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    slice_start = [0] * len(x_rm.shape)
    slice_end = list(x_rm.shape)
    slice_start[dim] = start
    slice_end[dim] = end
    sliced = ttnn.slice(x_rm, slice_start, slice_end)
    return ttnn.to_layout(sliced, ttnn.TILE_LAYOUT)


def topk_indices_argmax(values: ttnn.Tensor, k: int, *, mask_value: float) -> ttnn.Tensor:
    work = values
    indices_tiles: list[ttnn.Tensor] = []
    batch = work.shape[0]
    use_row_major = work.dtype == ttnn.float32
    if use_row_major and work.layout != ttnn.ROW_MAJOR_LAYOUT:
        work = ttnn.to_layout(work, ttnn.ROW_MAJOR_LAYOUT)
    for _ in range(k):
        idx = ttnn.argmax(work, dim=-1)
        idx_rm = idx if idx.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
        idx_rm = ttnn.reshape(idx_rm, (batch, 1))
        indices_tiles.append(idx_rm)

        if use_row_major:
            src = ttnn.full(
                idx_rm.shape,
                float(mask_value),
                dtype=work.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=work.device(),
            )
            work = ttnn.scatter(work, dim=1, index=idx_rm, src=src)
        else:
            idx_tile = ttnn.to_layout(idx_rm, ttnn.TILE_LAYOUT)
            zeros = ttnn.mul(work, 0.0)
            idx_float = ttnn.typecast(idx_tile, work.dtype)
            ones = ttnn.add(ttnn.mul(idx_float, 0.0), 1.0)
            mask = ttnn.scatter(zeros, dim=1, index=idx_tile, src=ones)
            work = work + mask * float(mask_value)

    indices = indices_tiles[0] if len(indices_tiles) == 1 else ttnn.concat(indices_tiles, dim=1)
    return indices


def select_topk_indices(
    values: ttnn.Tensor,
    k: int,
    *,
    dim: int,
    mask_value: float,
    prefer_argmax: bool = False,
) -> ttnn.Tensor:
    # Runtime topk support is dtype/layout dependent. For FP32 and explicit compatibility
    # requests, use deterministic argmax-based selection.
    if prefer_argmax or values.dtype == ttnn.float32:
        return topk_indices_argmax(values, k, mask_value=mask_value)
    k_pad = align_to_tile(k)
    k_call = k_pad if k_pad <= values.shape[dim] else k
    _, topk_idx = ttnn.topk(values, k=k_call, dim=dim, largest=True, sorted=False)
    topk_idx = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
    if k_call != k:
        topk_idx = slice_to_length(topk_idx, dim=dim, length=k)
    return topk_idx


def ensure_uint32_indices(idx: ttnn.Tensor, *, length: int) -> ttnn.Tensor:
    if idx.dtype == ttnn.uint32:
        return idx
    idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
    idx_rm, _ = pad_to_multiple(idx_rm, dim=1, multiple=TILE_SIZE, value=0)
    idx_tile = ttnn.to_layout(idx_rm, ttnn.TILE_LAYOUT)
    idx_tile = ttnn.typecast(idx_tile, ttnn.uint32)
    idx_rm = ttnn.to_layout(idx_tile, ttnn.ROW_MAJOR_LAYOUT)
    return slice_to_length(idx_rm, dim=1, length=length)


def pad_qh_for_sdpa(
    qh: ttnn.Tensor,
    mask: Optional[ttnn.Tensor],
    *,
    mask_value: float,
) -> tuple[ttnn.Tensor, Optional[ttnn.Tensor], int]:
    q_len = qh.shape[2]
    if q_len % TILE_SIZE == 0:
        return qh, mask, q_len
    qh_rm = ttnn.to_layout(qh, ttnn.ROW_MAJOR_LAYOUT)
    qh_rm, _ = pad_to_multiple(qh_rm, dim=2, multiple=TILE_SIZE, value=0.0)
    qh = ttnn.to_layout(qh_rm, ttnn.TILE_LAYOUT)
    if mask is not None:
        mask_rm = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
        mask_rm, _ = pad_to_multiple(mask_rm, dim=2, multiple=TILE_SIZE, value=mask_value)
        mask = ttnn.to_layout(mask_rm, ttnn.TILE_LAYOUT)
    return qh, mask, q_len


def safe_softmax(x: ttnn.Tensor, *, dim: int, numeric_stable: bool = True) -> ttnn.Tensor:
    # Use manual implementation for very large rows where runtime softmax support can be limited.
    if x.shape[dim] < 4096:
        return ttnn.softmax(x, dim=dim, numeric_stable=numeric_stable)
    max_vals = ttnn.max(x, dim=dim, keepdim=True) if numeric_stable else None
    shifted = x if max_vals is None else x - max_vals
    exp = ttnn.exp(shifted)
    sum_vals = ttnn.sum(exp, dim=dim, keepdim=True)
    sum_vals = ttnn.add(sum_vals, 1.0e-6)
    inv = ttnn.reciprocal(sum_vals)
    return ttnn.mul(exp, inv)


def pad_attention_mask(
    mask: ttnn.Tensor,
    *,
    q_length: int,
    k_length: int,
    mask_value: float,
) -> ttnn.Tensor:
    mask_rm = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
    curr_q = mask_rm.shape[2]
    curr_k = mask_rm.shape[3]
    if curr_q > q_length:
        mask_rm = slice_to_length(mask_rm, dim=2, length=q_length)
        curr_q = q_length
    if curr_k > k_length:
        mask_rm = slice_to_length(mask_rm, dim=3, length=k_length)
        curr_k = k_length
    if curr_q == q_length and curr_k == k_length:
        return ttnn.to_layout(mask_rm, ttnn.TILE_LAYOUT)
    padding = [(0, 0)] * len(mask_rm.shape)
    padding[2] = (0, q_length - curr_q)
    padding[3] = (0, k_length - curr_k)
    mask_rm = ttnn.pad(mask_rm, padding, value=float(mask_value), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_layout(mask_rm, ttnn.TILE_LAYOUT)


def make_sample_indices(length: int, sample_k: int, *, device, random_sampling: bool) -> ttnn.Tensor:
    length = max(1, int(length))
    sample_k = max(1, min(int(sample_k), length))
    if random_sampling:
        idx_torch = torch.randint(0, length, (sample_k,), dtype=torch.int64)
        idx = ttnn.from_torch(idx_torch, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(idx, (1, 1, sample_k))
    step = max(1, length // sample_k)
    end = step * sample_k
    idx = get_cached_arange(start=0, end=end, step=step, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(idx, (1, 1, sample_k))


def precompute_trace_constants(config: "InformerConfig", *, device) -> None:
    """Precompute tensors needed during trace capture to avoid host writes."""
    lengths: set[int] = set()
    distil_lengths: list[int] = []
    lengths.add(int(config.seq_len))
    if config.distil.enabled:
        length = int(config.seq_len)
        distil_lengths.append(length)
        for _ in range(max(0, config.e_layers - 1)):
            length = max(
                1, (length + 2 * config.distil.padding - config.distil.kernel_size) // config.distil.stride + 1
            )
            lengths.add(length)
            distil_lengths.append(length)
    lengths.add(int(config.label_len + config.pred_len))

    for length in lengths:
        padded = align_to_tile(length)
        get_cached_arange(end=length, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        get_cached_arange(end=length, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        get_cached_arange(end=padded, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        get_cached_arange(end=padded, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        if config.attention_type == "prob":
            sample_k = min(length, max(1, int(config.factor * math.log(max(2, length)))))
            step = max(1, length // sample_k)
            end = step * sample_k
            get_cached_arange(
                start=0, end=end, step=step, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        if config.attention_type == "prob" and config.hf_compat:
            get_cached_subtract_weight(device=device, dtype=ttnn.float32)

    if config.distil.enabled:
        kernel = int(config.distil.kernel_size)
        stride = int(config.distil.stride)
        padding = int(config.distil.padding)
        for length in distil_lengths:
            length_padded = length + 2 * padding
            out_len = max(1, (length_padded - kernel) // stride + 1)
            for offset in range(kernel):
                end = offset + out_len * stride
                get_cached_arange(
                    start=offset,
                    end=end,
                    step=stride,
                    device=device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )


def linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    *,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    core_grid: Optional[ttnn.CoreGrid] = None,
) -> ttnn.Tensor:
    kwargs = {"bias": bias, "transpose_b": True, "dtype": dtype}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    if core_grid is not None:
        kwargs["core_grid"] = core_grid
    return ttnn.linear(x, weight, **kwargs)


def max_pool1d(
    x: ttnn.Tensor,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    batch, length, channels = x.shape
    if length < kernel:
        return x

    pad_value = -1.0e9
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if padding > 0:
        pad_spec = [(0, 0), (padding, padding), (0, 0)]
        x_rm = ttnn.pad(x_rm, pad_spec, value=pad_value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    length_padded = length + 2 * padding
    out_len = max(1, (length_padded - kernel) // stride + 1)
    x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)

    pooled = None
    slices: list[ttnn.Tensor] = []
    for offset in range(kernel):
        end = offset + out_len * stride
        idx = get_cached_arange(
            start=offset,
            end=end,
            step=stride,
            device=x.device(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        idx = ttnn.reshape(idx, (1, out_len, 1))
        idx = ttnn.repeat(idx, (batch, 1, channels))
        idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        slice_out = ttnn.gather(x_tile, dim=1, index=idx)
        slice_out = ttnn.reshape(slice_out, (batch, out_len, 1, channels))
        slices.append(slice_out)

    if not slices:
        return x
    stacked = slices[0] if len(slices) == 1 else ttnn.concat(slices, dim=2)
    pooled = ttnn.max(stacked, dim=2)
    if pooled.dtype != dtype:
        pooled = ttnn.typecast(pooled, dtype)
    return pooled


def apply_dropout(x: ttnn.Tensor, p: float) -> ttnn.Tensor:
    if p == 0.0:
        return x
    raise NotImplementedError("Dropout is not implemented for TTNN bring-up; set dropout=0.0.")


def make_sequence_mask(
    length: int,
    valid_length: int,
    *,
    device,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    idx = get_cached_arange(end=length, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    valid_len = ttnn.add(ttnn.mul(idx, 0.0), float(valid_length))
    mask = ttnn.lt(idx, valid_len, dtype=ttnn.bfloat16)
    mask = ttnn.reshape(mask, (1, 1, length, 1))
    mask = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
    if dtype != mask.dtype:
        mask = ttnn.typecast(mask, dtype)
    return mask


def base_context_mean(v: ttnn.Tensor, *, valid_length: int) -> ttnn.Tensor:
    batch, heads, length, _ = v.shape
    mask = make_sequence_mask(length, valid_length, device=v.device(), dtype=ttnn.bfloat16)
    mask = ttnn.repeat(mask, (batch, heads, 1, 1))
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    masked = v * mask
    mean = ttnn.sum(masked, dim=2, keepdim=True, scalar=1.0 / float(max(1, valid_length)))
    return ttnn.repeat(mean, (1, 1, length, 1))


def base_context_cumsum(v: ttnn.Tensor) -> ttnn.Tensor:
    cumsum = ttnn.cumsum(v, dim=2, dtype=ttnn.float32)
    if cumsum.dtype != v.dtype:
        cumsum = ttnn.typecast(cumsum, v.dtype)
    return cumsum
