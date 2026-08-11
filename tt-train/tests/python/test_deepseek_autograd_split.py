# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``autograd_split`` against the ``autograd_slice`` it replaces.

``autograd_split`` is the efficient drop-in for splitting one tensor into
contiguous chunks (used by the MLA kv_down path). It must be value- and
gradient-identical to calling ``autograd_slice`` once per chunk, which is the
reference these tests pin it to.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models.deepseek.autograd_ops import autograd_slice, autograd_split


# (shape, chunk sizes along dim, dim) -- chunks tile the whole axis.
CASES = [
    ((2, 1, 256, 160), [128, 32], 3),  # MLA kv_down split (kv_lora=128, qk_rope=32)
    ((1, 1, 128, 160), [96, 32, 32], 3),  # 3-way, uneven
    ((1, 1, 64, 192), [64, 64, 64], 3),  # 3-way, even
    ((1, 4, 64, 32), [1, 3], 1),  # split on a non-last dim
    ((1, 1, 64, 96), [32, 64], 3),  # asymmetric 2-way
]


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _make_leaf(x_np):
    t = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    t.set_requires_grad(True)
    return t


def _weighted_loss(chunks):
    # Distinct per-chunk weights so an ordering/placement bug in backward
    # produces a different input gradient than the slice reference.
    loss = None
    for i, chunk in enumerate(chunks):
        term = ttml.ops.unary.mean(chunk) * float(i + 1)
        loss = term if loss is None else loss + term
    return loss


def _run(x_np, split_fn):
    leaf = _make_leaf(x_np)
    chunks = split_fn(leaf)
    _weighted_loss(chunks).backward(False)
    return [c.to_numpy() for c in chunks], leaf.get_grad_tensor().to_numpy()


@pytest.mark.requires_device
@pytest.mark.parametrize("shape,sizes,dim", CASES)
def test_split_matches_slice(shape, sizes, dim):
    ctx = ttml.autograd.AutoContext.get_instance()
    x_np = np.random.default_rng(0).standard_normal(shape, dtype=np.float32)

    def via_split(leaf):
        return list(autograd_split(leaf, sizes, dim))

    def via_slice(leaf):
        chunks = []
        offset = 0
        for size in sizes:
            start = [0] * len(shape)
            end = list(shape)
            start[dim] = offset
            end[dim] = offset + size
            chunks.append(autograd_slice(leaf, start, end))
            offset += size
        return chunks

    ctx.reset_graph()
    split_chunks, split_grad = _run(x_np, via_split)
    ctx.reset_graph()
    slice_chunks, slice_grad = _run(x_np, via_slice)

    for sc, lc in zip(split_chunks, slice_chunks):
        np.testing.assert_array_equal(sc, lc)  # pure copy -> bit-exact
    np.testing.assert_allclose(split_grad, slice_grad, rtol=1e-3, atol=1e-3)
