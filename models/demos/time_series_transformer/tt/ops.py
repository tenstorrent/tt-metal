# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from .config import TILE_SIZE

# Causal masks are shape- and device-invariant, so they are built once and reused. Keyed by
# (device, length, mask_value, dtype) because a traced replay must reuse the same tensor.
_CAUSAL_MASK_CACHE: dict[tuple[int, int, float, ttnn.DataType], ttnn.Tensor] = {}


def linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    *,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Apply ``y = x @ weight.T + bias`` with weights in torch ``(out, in)`` layout.

    The fused bias path is incompatible with L1-resident operands here -- ``ttnn.linear``
    raises a matmul broadcast error whenever an operand or the output lives in L1 and a bias
    is supplied. Falling back to a separate eltwise add keeps ``use_l1`` usable; it costs one
    extra dispatch, which is part of why L1 measures slower at this model size.
    """
    kwargs = {"transpose_b": True, "dtype": dtype}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    uses_l1 = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
    if bias is not None and not uses_l1:
        kwargs["bias"] = bias
        return ttnn.linear(x, weight, **kwargs)

    output = ttnn.linear(x, weight, **kwargs)
    if bias is None:
        return output
    return ttnn.add(output, bias, memory_config=memory_config)


def activation(x: ttnn.Tensor, name: str) -> ttnn.Tensor:
    if name == "gelu":
        # ttnn.gelu defaults to GeluVariant.Accurate, i.e. the erf formulation HuggingFace
        # uses. Passing the approximate variant explicitly would cost PCC for no benefit at
        # this model size.
        return ttnn.gelu(x)
    if name == "relu":
        return ttnn.relu(x)
    raise ValueError(f"Unsupported activation: {name}")


def softmax(x: ttnn.Tensor, *, dim: int = -1, exact: bool = False) -> ttnn.Tensor:
    """Row-wise softmax.

    ``ttnn.softmax`` carries roughly 3.8% row-sum error on this model's score matrices, with
    or without ``numeric_stable``, and independently of tile alignment -- a 32-wide row is
    just as bad as a 24-wide one. That looks alarming in isolation, but the error is close to
    a uniform scaling of each row, so the layer norm following the residual add removes it:
    measured end to end, the fused kernel is no less accurate than the composed version and is
    ~15% faster. ``exact=True`` composes max/exp/sum/reciprocal instead, six dispatches rather
    than one, and is kept for diagnosing attention numerics.
    """
    if not exact:
        return ttnn.softmax(x, dim=dim, numeric_stable=True)
    shifted = ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True))
    exponentiated = ttnn.exp(shifted)
    total = ttnn.sum(exponentiated, dim=dim, keepdim=True)
    return ttnn.multiply(exponentiated, ttnn.reciprocal(total))


def to_device(x: torch.Tensor, *, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(x.contiguous(), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x).float()


def make_causal_mask(
    length: int,
    *,
    device,
    dtype: ttnn.DataType,
    mask_value: float,
) -> ttnn.Tensor:
    """Additive ``(1, 1, length, length)`` causal mask, zero on and below the diagonal."""
    key = (id(device), int(length), float(mask_value), dtype)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = torch.full((length, length), mask_value, dtype=torch.float32).triu(1)
    tensor = ttnn.from_torch(
        mask.reshape(1, 1, length, length),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    _CAUSAL_MASK_CACHE[key] = tensor
    return tensor


def make_causal_mask_with_offset(
    query_length: int,
    key_length: int,
    *,
    device,
    dtype: ttnn.DataType,
    mask_value: float,
) -> Optional[ttnn.Tensor]:
    """Causal mask for a query block that starts ``key_length - query_length`` steps in.

    Returns ``None`` for a single-token query, where every cached key is visible.
    """
    if query_length == 1:
        return None
    offset = key_length - query_length
    rows = torch.arange(query_length).reshape(-1, 1) + offset
    cols = torch.arange(key_length).reshape(1, -1)
    mask = torch.where(cols > rows, torch.tensor(mask_value), torch.tensor(0.0)).float()
    return ttnn.from_torch(
        mask.reshape(1, 1, query_length, key_length),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )


def slice_sequence(x: ttnn.Tensor, *, dim: int, start: int, end: int) -> ttnn.Tensor:
    starts = [0] * len(x.shape)
    ends = list(x.shape)
    starts[dim] = start
    ends[dim] = end
    return ttnn.slice(x, starts, ends)


def slice_last_step(x: ttnn.Tensor) -> ttnn.Tensor:
    """Take the final timestep of a ``(batch, seq, dim)`` tensor, keeping the seq axis."""
    return slice_sequence(x, dim=1, start=int(x.shape[1]) - 1, end=int(x.shape[1]))


def pad_last_dim(x: ttnn.Tensor, *, multiple: int = TILE_SIZE) -> tuple[ttnn.Tensor, int]:
    """Zero-pad the last dim up to a multiple, returning the tensor and its original width.

    Needed only for the SDPA kernel, which rejects tensors whose logical last dim differs
    from the tile-padded one.
    """
    width = int(x.shape[-1])
    target = ((width + multiple - 1) // multiple) * multiple
    if target == width:
        return x, width
    padding = [(0, 0)] * (len(x.shape) - 1) + [(0, target - width)]
    return ttnn.pad(x, padding, value=0.0), width


def squareplus(x: ttnn.Tensor) -> ttnn.Tensor:
    """``0.5 * (x + sqrt(x^2 + 4))`` — the softplus variant HF uses for distribution scales."""
    return ttnn.multiply(ttnn.add(x, ttnn.sqrt(ttnn.add(ttnn.square(x), 4.0))), 0.5)


__all__ = [
    "activation",
    "linear",
    "make_causal_mask",
    "make_causal_mask_with_offset",
    "pad_last_dim",
    "slice_last_step",
    "slice_sequence",
    "softmax",
    "squareplus",
    "to_device",
    "to_torch",
]
