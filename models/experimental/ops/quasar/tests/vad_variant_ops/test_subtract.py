# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.subtract`` — fp32 tensor-tensor and bf16 tensor-scalar.

Shape source (kernel cache): fp32 SFPU sub ``eltwise_binary_sfpu_no_bcast/151244`` (L1);
bf16 scalar sub on the FPU ``eltwise_binary_scalar/182098`` (L1 / DRAM interleaved).
Op / dtype / memory are cache-determined; shapes and scalar values are runtime args (lost) [L].
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape", [pytest.param((1, 300, 32), id="B20-fp32-151244")])
def test_subtract(ttnn_mesh_device, reset_seeds, a_shape):
    U.run_binary(
        ttnn_mesh_device,
        ttnn.subtract,
        lambda x, y: x - y,
        a_shape,
        a_shape,
        "fp32",
        "fp32",
        "l1",
        "l1",
        "l1",
        pcc=0.9999,
    )


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, mem",
    [
        pytest.param((1, U.NUM_QUERY, U.EMBED), "l1", id="S1-bf16-l1-182098"),
        pytest.param((1, U.NUM_QUERY, U.EMBED), "dram", id="S1-bf16-dram-182098"),
    ],
)
def test_subtract_scalar(ttnn_mesh_device, reset_seeds, shape, mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, mem)
    out = ttnn.subtract(x, 0.5, memory_config=U.resolve_memcfg(mem, mesh))
    U.assert_pcc(x_torch.float() - 0.5, out, pcc=0.999, mesh_device=mesh)
