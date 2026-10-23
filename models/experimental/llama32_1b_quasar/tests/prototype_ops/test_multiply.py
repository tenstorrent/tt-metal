# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.multiply``.

Model call site (modules/embedding/embedding_1d.py):
  * L147  forward — embedding output scaling: multiply(out, embed_scale, memory_config=...)
           (a tensor * python-scalar multiply, run only when embed_scale != 1.0)

The embedding output is [1, 1, seq/batch, DIM]; ``embed_scale`` is a float. We also
cover the tensor * tensor form of the same op for completeness. Reference is the
elementwise product in torch.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

_SHAPES = [pytest.param((1, 1, seq, U.DIM), id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS] + [
    pytest.param((1, 1, batch, U.DIM), id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES
]

# Representative embedding scale (Llama does not scale by default, so exercise a
# non-unit factor to make the op meaningful).
_EMBED_SCALE = 2.0


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SHAPES)
def test_multiply_scalar(ttnn_mesh_device, reset_seeds, shape):
    """Embedding output scaling: multiply(out, embed_scale) (embedding_1d.py:147)."""
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.experimental.quasar.multiply(x, _EMBED_SCALE)

    ref = x_torch.float() * _EMBED_SCALE
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SHAPES)
def test_multiply_tensor(ttnn_mesh_device, reset_seeds, shape):
    """Elementwise tensor * tensor form of ttnn.multiply."""
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.experimental.quasar.multiply(a, b)

    ref = a_torch.float() * b_torch.float()
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
