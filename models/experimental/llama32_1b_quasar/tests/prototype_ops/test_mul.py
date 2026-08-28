# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.mul``.

Model call sites (modules/mlp/mlp_1d.py):
  * L256  decode_forward  — SwiGLU gate: mul(w1_out, w3_out, input_tensor_a_activations=[SILU])
  * L344  prefill_forward — same, prefill path

Both fuse the SiLU activation on ``input_tensor_a`` (the gate projection) and then
multiply by the up projection (w3), i.e. ``silu(w1_out) * w3_out``. The operands are
[1, 1, M, INTERMEDIATE] (the MLP hidden width). Reference is the same in torch.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

_M_SIZES = [pytest.param((1, 1, seq, U.INTERMEDIATE), id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS] + [
    pytest.param((1, 1, batch, U.INTERMEDIATE), id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _M_SIZES)
def test_mul_silu_gate(ttnn_mesh_device, reset_seeds, shape):
    """SwiGLU gate: silu(w1_out) * w3_out (mlp_1d.py:256 / :344)."""
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)  # w1_out (gate)
    b_torch = U.torch_rand(shape)  # w3_out (up)

    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.experimental.quasar.multiply(
        a,
        b,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
    )

    ref = torch.nn.functional.silu(a_torch.float()) * b_torch.float()
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
