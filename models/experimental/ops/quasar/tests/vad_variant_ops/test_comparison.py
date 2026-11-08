# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.gt`` / ``ttnn.lt`` against a scalar, and ``ttnn.logical_and``.

Shape source (kernel cache):
  * ``gt(x, s)`` -> FPU sub + ``gtz`` (``eltwise_binary_scalar/560157``). In the spatial-cross-
    attention count pipeline ``bev_mask.sum(-1) > 0`` on [6, 1, 10000, 1] (bf16, L1)   [M]
  * ``lt(x, s)`` -> sub + ``ltz`` (``140291``)                                       [L shape]
  * ``logical_and`` -> mul + nez on inputs/output (``eltwise_binary_no_bcast/381505``),
    bf16 L1; the camera-visibility mask combine on [6, 1, 10000, 4]               [L shape]
Comparison outputs are exactly 0/1, so these are checked for exact equality.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

C, Q = U.NUM_CAMS, U.NUM_QUERY


def _assert_exact(ref, out, mesh):
    got = U.from_tt(out, mesh).reshape(-1)[: ref.numel()].reshape(ref.shape)
    assert torch.equal(got, ref.float()), f"{(got != ref.float()).sum().item()}/{ref.numel()} mismatches"


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, shape, mem, scalar",
    [
        pytest.param("gt", (C, 1, Q, 1), "l1", 0.0, id="S2-gt-sca-count-560157"),
        pytest.param("lt", (1, Q, U.EMBED), "dram", 0.0, id="S3-lt-140291"),
    ],
)
def test_compare_scalar(ttnn_mesh_device, reset_seeds, op, shape, mem, scalar):
    mesh = ttnn_mesh_device
    # Integer-valued input (like a hit count) so the comparison is unambiguous in bf16.
    x_torch = torch.randint(-2, 3, shape).to(torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, mem)
    out = getattr(ttnn, op)(x, scalar, memory_config=U.resolve_memcfg(mem, mesh))
    ref = (x_torch > scalar) if op == "gt" else (x_torch < scalar)
    _assert_exact(ref, out, mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((C, 1, Q, 4), id="B18-logical_and-381505")])
def test_logical_and(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    a_torch = torch.randint(0, 2, shape).to(torch.bfloat16)
    b_torch = torch.randint(0, 2, shape).to(torch.bfloat16)
    a = U.to_tt_mem(a_torch, mesh, "l1")
    b = U.to_tt_mem(b_torch, mesh, "l1")
    out = ttnn.logical_and(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)
    _assert_exact(torch.logical_and(a_torch, b_torch), out, mesh)
