# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import pad_by_zero


def test_with_ops(device):
    torch.manual_seed(0)

    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = a @ b
    output = a @ b

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_devices(
    mesh_device,
):
    logger.debug("Testing All Devices")


def test_with_sfpu(mesh_device):
    file_path = "tests/ttnn/tracy/test_with_sfpu_x_tensor.pt"
    # torch.manual_seed(1234)
    # shape = [1, 1, 32, 32]
    # x = torch.randn(shape).bfloat16().float()
    # torch.save(x, file_path)
    x = torch.load(file_path)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(mesh_device)

    layernorm_output = ttnn.layer_norm(
        xt,
        epsilon=1e-5,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        memory_config=mem,
    )

    layernorm_output = ttnn.mul(
        layernorm_output,
        layernorm_output,
    )

    layernorm_output = ttnn.add(
        layernorm_output,
        layernorm_output,
    )
