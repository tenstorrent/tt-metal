# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.multiply`` — tensor*tensor (bf16 / fp32), row-broadcast, and tensor*scalar.

Shape source (kernel cache): bf16 MUL routes to the SFPU in the fork
(``eltwise_binary_sfpu_no_bcast/167837``, ``eltwise_binary_sfpu_row_bcast/455011``,
``eltwise_binary_sfpu_scalar/115322``); fp32 ``eltwise_binary_sfpu_no_bcast/636314``,
scalar ``139989``. All interleaved => op/dtype/memory are cache-determined [H], shapes are
inferred from the BEV encoder (queries [1, 10000, 256]) [M/L]. Scalar values are runtime args
(lost); representative values are used.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED

# fmt: off
CASES = [
    pytest.param((1, Q, E), (1, Q, E), "bf16", "bf16", "l1", "dram", "l1", id="B15-bf16-167837"),
    pytest.param((1, Q, E), (1, 1, E), "bf16", "bf16", "l1", "dram", "l1", id="B12-rowbcast-455011"),
    pytest.param((1, 300, 32), (1, 300, 32), "fp32", "fp32", "l1", "l1", "l1", id="B21-fp32-636314"),
]
SCALAR_CASES = [
    pytest.param((1, Q, E), "bf16", "dram", 0.5, id="S4-bf16-dram-115322"),
    pytest.param((1, Q, E), "bf16", "l1", 0.5, id="S4-bf16-l1-115322"),
    pytest.param((1, 300, 32), "fp32", "l1", 2.0, id="S6-fp32-139989"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem", CASES)
def test_multiply(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    U.run_binary(
        ttnn_mesh_device,
        ttnn.multiply,
        lambda x, y: x * y,
        a_shape,
        b_shape,
        a_dt,
        b_dt,
        a_mem,
        b_mem,
        out_mem,
        pcc=0.999,
    )


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, mem, scalar", SCALAR_CASES)
def test_multiply_scalar(ttnn_mesh_device, reset_seeds, shape, dt, mem, scalar):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt])
    out = ttnn.multiply(x, scalar, memory_config=U.resolve_memcfg(mem, mesh))
    U.assert_pcc(x_torch.float() * scalar, out, pcc=0.999, mesh_device=mesh)


@U.with_default_mesh()
# fp32 ROW_MAJOR DRAM scalar op, last dim <= 16 (reader_interleaved_rm_scalar_op/127631).
@pytest.mark.parametrize("shape", [pytest.param((1, 300, 2), id="S-fp32-rowmajor-127631")])
def test_multiply_scalar_fp32_row_major(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape, dtype=torch.float32)
    x = U.to_tt(x_torch, mesh, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.multiply(x, 2.0)
    U.assert_pcc(x_torch * 2.0, out, pcc=0.9999, mesh_device=mesh)
