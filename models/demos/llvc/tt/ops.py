# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN op helpers for LLVC.

Conventions
-----------
* Activations flow as ``[B, T, C]`` on device.
* Convolutions consume/emit ROW_MAJOR ``[B, T, C]``; matmul/attention/layernorm
  consume/emit TILE ``[B, T, C]``.
* Streaming causal convolution uses :func:`causal_window` + :func:`apply_taps`
  (kernel-tap shifted multiply-accumulate), which mirrors the reference's
  per-layer ring buffers and avoids the L1 pressure of ``ttnn.conv1d`` on long
  sequences.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn
from models.demos.llvc.tt.config import TILE_SIZE

# Additive mask fill for SDPA key-padding (fused kernel ignores structural causality).
_SDPA_PAD_MASK_VALUE = -1e4

# Device masks cached by shape so fused SDPA stays trace-safe (no host upload in
# ``forward_chunk`` after the warmup chunk has populated the cache).
_SDPA_PAD_MASK_CACHE: dict[tuple, ttnn.Tensor] = {}


def to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x)


def as_row_major(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


def as_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout != ttnn.TILE_LAYOUT:
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return x


def relu(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.relu(x)


def tanh(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.tanh(x)


def sigmoid(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.sigmoid(x)


def concat_time(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    """Concatenate two ``[B, T, C]`` tensors along the time axis (dim=1)."""
    a_rm = as_row_major(a)
    b_rm = as_row_major(b)
    return ttnn.concat([a_rm, b_rm], dim=1)


def slice_time(x: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    x_rm = as_row_major(x)
    starts = [0, start, 0]
    ends = [x_rm.shape[0], end, x_rm.shape[2]]
    return ttnn.slice(x_rm, starts, ends)


def linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    *,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    activation: Optional[str] = None,
) -> ttnn.Tensor:
    """Row-wise affine transform. ``weight`` is stored ``[out, in]`` (transpose_b)."""
    kwargs = {"bias": bias, "transpose_b": True, "dtype": dtype}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    if activation is not None:
        kwargs["activation"] = activation
    return ttnn.linear(as_tile(x), weight, **kwargs)


def layernorm_channels(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    *,
    eps: float = 1e-5,
) -> ttnn.Tensor:
    """LayerNorm over the last (channel) dim of a ``[B, T, C]`` tensor."""
    return ttnn.layer_norm(as_tile(x), weight=weight, bias=bias, epsilon=eps)


def conv1d(
    x_btc: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor],
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int,
    device,
    weights_dtype: ttnn.DataType,
    output_dtype: ttnn.DataType,
    math_fidelity: ttnn.MathFidelity,
    dilation: int = 1,
    activation: str = "",
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
) -> tuple[ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
    """Wrap ``ttnn.conv1d`` for a ROW_MAJOR ``[B, T, Cin]`` tensor.

    ``activation`` (e.g. ``"relu"``) is applied after the conv as a separate op.
    (Conv-fused activation via ``UnaryWithParam`` is a Stage-2 optimization.)
    Returns ``([B, Tout, Cout] ROW_MAJOR, prepared_weight, prepared_bias)``; store
    *both* prepared tensors and pass them back so subsequent calls (and trace
    capture) reuse them instead of re-uploading host weights/bias to device.
    """
    x_rm = as_row_major(x_btc)
    batch = x_rm.shape[0]
    length = x_rm.shape[1]

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        deallocate_activation=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=math_fidelity)

    out, out_length, [weight_dev, bias_dev] = ttnn.conv1d(
        input_tensor=x_rm,
        weight_tensor=weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=bias,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        batch_size=batch,
        input_length=length,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        dtype=output_dtype,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    out = ttnn.sharded_to_interleaved(out) if out.is_sharded() else out
    if activation:
        if activation == "relu":
            out = ttnn.relu(out)
        else:
            raise ValueError(f"Unsupported conv activation: {activation}")
    out = as_row_major(out)
    out = ttnn.reshape(out, (batch, out_length, out_channels))
    return out, weight_dev, bias_dev


def conv_transpose1d(
    x_btc: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor],
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    device,
    weights_dtype: ttnn.DataType,
    output_dtype: ttnn.DataType,
    math_fidelity: ttnn.MathFidelity,
) -> tuple[ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
    """1D transposed conv via ``ttnn.conv_transpose2d`` with a singleton width.

    The kernel lives on the height axis, so the time length must too: input
    ``[B, T, Cin]`` -> ``[B, T, 1, Cin]`` (NHWC), weight ``[Cin, Cout, kernel, 1]``.
    Returns ``([B, Tout, Cout], weight, bias)``.
    """
    x_rm = as_row_major(x_btc)
    batch = x_rm.shape[0]
    length = x_rm.shape[1]
    x_nhwc = ttnn.reshape(x_rm, (batch, length, 1, in_channels))

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=math_fidelity, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    out, [out_h, out_w], [weight_dev, bias_dev] = ttnn.conv_transpose2d(
        input_tensor=x_nhwc,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        padding=(padding, 0),
        output_padding=(0, 0),
        batch_size=batch,
        input_height=length,
        input_width=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    out = ttnn.sharded_to_interleaved(out) if out.is_sharded() else out
    out = as_row_major(out)
    out = ttnn.reshape(out, (batch, out_h, out_channels))
    return out, weight_dev, bias_dev


def causal_window(
    x_btc: ttnn.Tensor,
    ctx_btc: ttnn.Tensor,
    *,
    buf_len: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Prepend the cached context and return ``(x_ext TILE, new_ctx)``.

    ``x_ext = concat(ctx, x)`` is the input a causal conv consumes with
    ``padding=0``; ``new_ctx`` is the trailing ``buf_len`` frames to carry to the
    next chunk. Several tap sets (e.g. the prenet's filter *and* gate) can share
    one window build instead of rebuilding the concat/slice/tilize per path.
    """
    x_ext = concat_time(ctx_btc, x_btc)  # [B, buf_len + T, Cin]
    new_ctx = slice_time(x_ext, x_ext.shape[1] - buf_len, x_ext.shape[1])
    return as_tile(x_ext), new_ctx


def apply_taps(
    x_ext_tile: ttnn.Tensor,
    seq_len: int,
    taps: list[ttnn.Tensor],
    biases: Optional[list[ttnn.Tensor]],
    *,
    dilation: int,
) -> ttnn.Tensor:
    """Shifted matmul-accumulate over kernel taps on a prepared TILE window.

    ``y[t] = sum_j x_ext[t + j*dilation] @ taps[j]`` for ``t`` in ``[0, seq_len)``.
    ``taps[j]`` is ``[Cin, Cout]`` (transposed for ``ttnn.linear``); ``biases`` is
    added per-tap (conv bias is folded onto tap 0 by the caller) or ``None``.
    """
    out = None
    for j in range(len(taps)):
        offset = j * dilation
        window = ttnn.slice(x_ext_tile, [0, offset, 0], [x_ext_tile.shape[0], offset + seq_len, x_ext_tile.shape[2]])
        bias = biases[j] if biases is not None else None
        term = ttnn.linear(window, taps[j], bias=bias, transpose_b=True)
        out = term if out is None else ttnn.add(out, term)
    return out


def depthwise_causal_conv1d(
    x_btc: ttnn.Tensor,
    weight_taps: list[ttnn.Tensor],
    bias: Optional[ttnn.Tensor],
    *,
    dilation: int,
) -> ttnn.Tensor:
    """Depthwise (groups==channels) causal conv1d via shifted per-channel MAC.

    ``x_btc`` already has the causal context prepended (``padding=0`` semantics),
    so with kernel ``K`` and ``dilation`` the output length is
    ``T_out = T_in - (K-1)*dilation``. ``weight_taps[j]`` is the per-channel tap
    vector ``[1, 1, C]`` and ``bias`` (added once) is ``[1, 1, C]`` or ``None``.

    ``ttnn.conv1d`` cannot find a valid shard/slice config for these depthwise
    layers, so we express the conv directly: ``y[t] = b + sum_j x[t+j*d] * w_j``.
    This mirrors the reference exactly and stays in TILE layout.
    """
    x_tile = as_tile(x_btc)
    b, t_in, c = x_tile.shape
    kernel = len(weight_taps)
    t_out = t_in - (kernel - 1) * dilation
    out = None
    for j in range(kernel):
        offset = j * dilation
        window = ttnn.slice(x_tile, [0, offset, 0], [b, offset + t_out, c])
        term = ttnn.mul(window, weight_taps[j])
        out = term if out is None else ttnn.add(out, term)
    if bias is not None:
        out = ttnn.add(out, bias)
    return out


def sinusoidal_position_encoding(length: int, d_model: int, *, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    """Positional encoding matching the reference (speechbrain layout)."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros((length, d_model), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return ttnn.from_torch(pe, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def split_heads(x: ttnn.Tensor, n_heads: int) -> ttnn.Tensor:
    """``[B, T, D]`` -> ``[B, H, T, D/H]``."""
    b, t, d = x.shape
    x = ttnn.reshape(x, (b, t, n_heads, d // n_heads))
    return ttnn.permute(x, (0, 2, 1, 3))


def merge_heads(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[B, H, T, D/H]`` -> ``[B, T, D]``."""
    b, h, t, hd = x.shape
    x = ttnn.permute(x, (0, 2, 1, 3))
    return ttnn.reshape(x, (b, t, h * hd))


def _pad_heads_seq(x: ttnn.Tensor, *, seq_dim: int = 2) -> tuple[ttnn.Tensor, int]:
    """Right-pad ``[B, H, S, D]`` along ``S`` to a TILE_SIZE multiple. Returns (padded, orig_S)."""
    seq_len = int(x.shape[seq_dim])
    pad = (TILE_SIZE - (seq_len % TILE_SIZE)) % TILE_SIZE
    if pad == 0:
        return as_tile(x), seq_len
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    padding = [(0, 0)] * len(x_rm.shape)
    padding[seq_dim] = (0, pad)
    x_rm = ttnn.to_memory_config(x_rm, ttnn.DRAM_MEMORY_CONFIG)
    x_rm = ttnn.pad(x_rm, padding, value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return as_tile(x_rm), seq_len


def _sdpa_key_pad_mask(
    *,
    batch: int,
    n_heads: int,
    q_padded: int,
    k_len: int,
    k_padded: int,
    device,
    dtype: ttnn.DataType,
) -> Optional[ttnn.Tensor]:
    """Additive mask ``[B, H, Qpad, Kpad]`` that suppresses padded key positions.

    Cached on device after first build so later (traced) calls do no host upload.
    """
    if k_padded == k_len:
        return None
    cache_key = (id(device), batch, n_heads, q_padded, k_len, k_padded, dtype)
    cached = _SDPA_PAD_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mask = torch.zeros((batch, n_heads, q_padded, k_padded), dtype=torch.float32)
    mask[:, :, :, k_len:] = _SDPA_PAD_MASK_VALUE
    dev_mask = ttnn.from_torch(mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    _SDPA_PAD_MASK_CACHE[cache_key] = dev_mask
    return dev_mask


def _manual_scaled_dot_product_attention(
    qh: ttnn.Tensor,
    kh: ttnn.Tensor,
    vh: ttnn.Tensor,
    *,
    scale: float,
) -> ttnn.Tensor:
    """Fallback matmul+softmax SDPA (no pad) for configs the fused kernel rejects."""
    scores = ttnn.matmul(qh, ttnn.permute(kh, (0, 1, 3, 2)))
    scores = ttnn.mul(scores, scale)
    probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
    return ttnn.matmul(probs, vh)


def scaled_dot_product_attention(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    n_heads: int,
    head_dim: int,
) -> ttnn.Tensor:
    """Multi-head SDPA via fused ``ttnn.transformer.scaled_dot_product_attention``.

    Causality is structural (decoder windows already encode history); no causal
    mask is applied. Sequence dims are padded to ``TILE_SIZE`` for the fused
    kernel; padded key slots are masked out so numerics match the unpadded case.
    Falls back to manual matmul+softmax if the fused op rejects the shapes
    (e.g. tiny CI configs with ``head_dim < TILE_SIZE``).
    Inputs are ``[B, Tq/Tk, D]`` TILE tensors.
    """
    qh = split_heads(as_tile(q), n_heads)
    kh = split_heads(as_tile(k), n_heads)
    vh = split_heads(as_tile(v), n_heads)
    scale = 1.0 / math.sqrt(head_dim)

    # Fused SDPA wants tile-aligned head_dim and sequence pads; keep the old path
    # for the small CI config (head_dim=16) so PCC/perf tests still run.
    if head_dim % TILE_SIZE != 0:
        return merge_heads(_manual_scaled_dot_product_attention(qh, kh, vh, scale=scale))

    qh_p, q_len = _pad_heads_seq(qh)
    kh_p, k_len = _pad_heads_seq(kh)
    vh_p, _ = _pad_heads_seq(vh)
    attn_mask = _sdpa_key_pad_mask(
        batch=int(qh_p.shape[0]),
        n_heads=n_heads,
        q_padded=int(qh_p.shape[2]),
        k_len=k_len,
        k_padded=int(kh_p.shape[2]),
        device=q.device(),
        dtype=q.dtype,
    )
    ctx = ttnn.transformer.scaled_dot_product_attention(
        qh_p,
        kh_p,
        vh_p,
        attn_mask=attn_mask,
        is_causal=False,
        scale=scale,
    )
    if int(ctx.shape[2]) != q_len:
        ctx = ttnn.slice(ctx, [0, 0, 0, 0], [ctx.shape[0], ctx.shape[1], q_len, ctx.shape[3]])
    return merge_heads(ctx)


def pad_time_to_multiple(x: ttnn.Tensor, multiple: int, value: float = 0.0) -> tuple[ttnn.Tensor, int]:
    """Right-pad ``[B, T, C]`` along T to a multiple. Returns (padded, original_T)."""
    x_rm = as_row_major(x)
    length = x_rm.shape[1]
    pad = (multiple - (length % multiple)) % multiple
    if pad == 0:
        return x_rm, length
    padding = [(0, 0), (0, pad), (0, 0)]
    x_rm = ttnn.to_memory_config(x_rm, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.pad(x_rm, padding, value=value, memory_config=ttnn.DRAM_MEMORY_CONFIG), length
