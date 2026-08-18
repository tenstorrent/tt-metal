# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.rms_norm``  (CANONICAL EXEMPLAR for tests/ops/).

Model call sites (modules/rmsnorm/rmsnorm_1d.py):
  * L206  _decode_local_sharded   — sharded, with program_config/memory_config
  * L238  _decode_local_interleaved — interleaved, program_config=None
  * L261  _prefill_local          — interleaved, program_config=None

This file covers the interleaved path (the emulator-friendly one) for the two
tensor rank/shape families the model drives through RMSNorm: prefill
[1, 1, seq, DIM] and decode [1, 1, batch, DIM]. The reference is torch.nn.RMSNorm.

Weight layout matches the module (rmsnorm_1d.py:369-370):
    (dim,) -> (1, 1, dim // 32, 32)
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ref = torch.nn.RMSNorm(x.shape[-1], eps=eps, dtype=torch.float32)
    ref.weight.data.copy_(weight.float())
    return ref(x.float())


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 1, seq, U.DIM), id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS]
    + [pytest.param((1, 1, batch, U.DIM), id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_rms_norm(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    dim = shape[-1]

    x_torch = U.torch_rand(shape)
    w_torch = U.torch_rand((dim,))

    x = U.to_tt(x_torch, mesh)
    # weight: (dim,) -> (1, 1, dim // TILE, TILE) in ROW_MAJOR, matching the module's
    # norm-weight LazyWeight (rmsnorm_1d.py:369-383, layout=ROW_MAJOR_LAYOUT). ttnn.rms_norm
    # flattens the ROW_MAJOR gamma to width `dim`; a TILE-layout gamma is read as width 32
    # and trips "gamma logical width >= input logical width".
    w = U.to_tt(w_torch.reshape(1, 1, dim // U.TILE, U.TILE), mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.rms_norm(x, epsilon=U.NORM_EPS, weight=w)

    ref = _torch_rms_norm(x_torch, w_torch, U.NORM_EPS)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
