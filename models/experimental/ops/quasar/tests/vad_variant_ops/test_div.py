# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.div`` — tensor/tensor (incl. column broadcast) and tensor/scalar.

Shape source (kernel cache): bf16 DIV routes to the SFPU in the fork.
  * COL_B broadcast div ``eltwise_binary_sfpu/113078`` + ``reader_interleaved_col_bcast/829483``:
    the spatial-cross-attention normalisation ``slots / count[..., None]`` on [1, 10000, 256] /
    [1, 10000, 1], both L1 interleaved   [M]
  * no-bcast bf16 ``eltwise_binary_sfpu_no_bcast/175804`` (a DRAM, b L1)   [L shape]
  * scalar bf16 ``eltwise_binary_sfpu_scalar/910494``, fp32 ``106700``     [L shape / value]
Divisors are kept away from 0 (|b| + 0.5); the model's count is clamped to >= 1.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED

# fmt: off
CASES = [
    pytest.param((1, Q, E), (1, Q, 1), "bf16", "bf16", "l1", "l1", "l1", id="B13-sca-count-colbcast-113078"),
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", "dram", "l1", "dram", id="B16-bf16-175804"),
]
SCALAR_CASES = [
    pytest.param((1, Q, E), "bf16", "l1", 2.0, id="S5-bf16-910494"),
    pytest.param((1, 300, 32), "fp32", "l1", 2.0, id="S7-fp32-106700"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem", CASES)
def test_div(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    U.run_binary(
        ttnn_mesh_device,
        ttnn.div,
        lambda x, y: x / y,
        a_shape,
        b_shape,
        a_dt,
        b_dt,
        a_mem,
        b_mem,
        out_mem,
        b_transform=lambda b: b.abs() + 0.5,
        pcc=0.999,
    )


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, mem, scalar", SCALAR_CASES)
def test_div_scalar(ttnn_mesh_device, reset_seeds, shape, dt, mem, scalar):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt])
    out = ttnn.div(x, scalar, memory_config=U.resolve_memcfg(mem, mesh))
    U.assert_pcc(x_torch.float() / scalar, out, pcc=0.999, mesh_device=mesh)
