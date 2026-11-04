# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.transpose(x, -2, -1)`` (TILE, bf16 and fp32).

Shape source (kernel cache): ``transpose_wh`` compute + ``reader_unary_transpose_wh_interleaved_start_id``
(``transpose_wh_program_factory.cpp:260-333``): bf16 L1 -> L1 and fp32 from DRAM [H dtype/memory].
Shapes are runtime args (lost) [L]. Value-preserving: checked bit-exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, dt, mem",
    [
        pytest.param((1, 1, 256, 320), "bf16", "l1", id="X1-bf16-l1"),
        pytest.param((1, 1, 32, 64), "fp32", "dram", id="X2-fp32-dram"),
    ],
)
def test_transpose(ttnn_mesh_device, reset_seeds, shape, dt, mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt])
    out = ttnn.transpose(x, -2, -1, memory_config=U.resolve_memcfg(mem, mesh))
    U.assert_lossless(x_torch.transpose(-2, -1).contiguous(), out, mesh_device=mesh)
