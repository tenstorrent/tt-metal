# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.add``.

Model call sites (models/llama32_1b/model.py):
  * L130  decode_forward  — residual + attn_out, [1, 1, batch, DIM]
  * L140  decode_forward  — residual + mlp_out,  [1, 1, batch, DIM]
  * L169  prefill_forward — residual + attn_out, [1, 1, seq, DIM]
  * L178  prefill_forward — residual + mlp_out,  [1, 1, seq, DIM]

These are the elementwise residual adds. The model passes ``memory_config`` (and
sometimes ``dtype``); the emulator-friendly interleaved-DRAM path is exercised
here. Reference is torch ``a + b``.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 1, seq, U.DIM), id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS]
    + [pytest.param((1, 1, batch, U.DIM), id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_add(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)

    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.experimental.quasar.add(a, b)

    ref = a_torch.float() + b_torch.float()
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
