# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Constant mask tiles for the Qwen3.8 Gated DeltaNet.

The primitives the delta rule needs that ttml was missing -- ``cumsum``,
``l2_norm``, ``softplus``, ``sum_over_dim``, ``transpose`` and
``shift_along_dim`` -- are now first-class ttml ops in ``ttml::ops`` (see
``sources/ttml/ops/unary_ops.{hpp,cpp}``) with C++ backwards, exposed as
``ttml.ops.unary.*``.  Nothing in this module is an op any more.

What is left here is *data*: the constant triangular tiles the delta rule
multiplies by.  ``masked_fill`` with a constant mask needs no op of its own --
it is a multiply by a 0/1 tensor -- so these are plain tensors, memoized
because the delta rule would otherwise rebuild them once per layer.

The generic slice/concat/split wrappers are shared with the DeepSeek model
rather than duplicated; they are re-exported below.
"""

from __future__ import annotations

import numpy as np

import ttnn
import ttml

# Model-agnostic despite living under models/deepseek; reuse rather than
# maintaining a second copy. ttnn's slice/concat/split have no ttml backward.
from ttml.models.deepseek.autograd_ops import (  # noqa: F401
    autograd_concat,
    autograd_slice,
    autograd_split,
)

__all__ = [
    "tril_ones",
    "causal_mask",
    "identity",
    "clear_const_cache",
    "autograd_slice",
    "autograd_concat",
    "autograd_split",
]

_CONST_CACHE: dict = {}


def _const_tile(key, build):
    """Memoize a constant device tensor; these are rebuilt every layer otherwise."""
    cached = _CONST_CACHE.get(key)
    if cached is None:
        cached = build()
        _CONST_CACHE[key] = cached
    return cached


def _upload(mask, batch_shape, chunk_size, dtype):
    full = np.broadcast_to(mask, tuple(batch_shape) + (chunk_size, chunk_size)).copy()
    return ttml.autograd.Tensor.from_numpy(full, ttnn.Layout.TILE, dtype)


def tril_ones(chunk_size, batch_shape=(1, 1), *, strict=False, dtype=ttnn.DataType.BFLOAT16):
    """Lower-triangular ones, shaped ``(*batch_shape, C, C)``, memoized.

    ttnn's matmul does not broadcast batch dims (it requires ``BCMK @ BCKN``),
    so constants have to be materialized at the operand's batch shape rather
    than relying on a ``[1, 1, C, C]`` broadcast.

    Args:
        strict: exclude the diagonal (``j < i`` instead of ``j <= i``).
    """
    batch_shape = tuple(int(b) for b in batch_shape)
    key = ("tril", chunk_size, batch_shape, bool(strict), str(dtype))

    def build():
        mask = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32), k=-1 if strict else 0)
        return _upload(mask, batch_shape, chunk_size, dtype)

    return _const_tile(key, build)


def causal_mask(chunk_size, batch_shape=(1, 1), *, diagonal=0, dtype=ttnn.DataType.BFLOAT16):
    """0/1 keep-mask, shaped ``(*batch_shape, C, C)``, memoized.

    ``diagonal=0`` keeps strictly below the diagonal (the reference's
    ``masked_fill(triu(diagonal=0), 0)``); ``diagonal=1`` keeps the diagonal too
    (``masked_fill(triu(diagonal=1), 0)``).
    """
    batch_shape = tuple(int(b) for b in batch_shape)
    key = ("keep", chunk_size, batch_shape, diagonal, str(dtype))

    def build():
        mask = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32), k=diagonal - 1)
        return _upload(mask, batch_shape, chunk_size, dtype)

    return _const_tile(key, build)


def identity(chunk_size, batch_shape=(1, 1), dtype=ttnn.DataType.BFLOAT16):
    """Identity matrix, shaped ``(*batch_shape, C, C)``, memoized."""
    batch_shape = tuple(int(b) for b in batch_shape)
    key = ("eye", chunk_size, batch_shape, str(dtype))

    def build():
        return _upload(np.eye(chunk_size, dtype=np.float32), batch_shape, chunk_size, dtype)

    return _const_tile(key, build)


def clear_const_cache():
    """Drop memoized constants (call before closing the device/mesh)."""
    _CONST_CACHE.clear()
