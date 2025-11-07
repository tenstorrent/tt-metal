#  SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0


from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

DEEPSEEK_SHAPE_PAIRS = [([1, 1, 1, 32], [1, 1, 4096, 64]), ([1, 1, 1, 32], [1, 1, 129280, 224])]

VOCAB_SIZE = 2048


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize("shape_pair", DEEPSEEK_SHAPE_PAIRS)
@pytest.mark.parametrize("input_dtype", [ttnn.uint32])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_embedding(
    mesh_device,
    shape_pair,
    input_dtype,
    weights_dtype,
    mem_config,
    layout,
):
    torch.manual_seed(1234)

    input_shape, weights_shape = shape_pair

    torch_input_tensor = torch.randint(0, VOCAB_SIZE, tuple(input_shape[-2:]))
    torch_weights = torch.rand(weights_shape[-2:]).bfloat16()
    torch_reference = torch.nn.functional.embedding(torch_input_tensor, torch_weights).repeat(
        [prod(mesh_device.shape)] + [1] * (len(input_shape) - 2)
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, device=mesh_device, dtype=input_dtype, memory_config=mem_config
    )
    tt_weights = ttnn.from_torch(torch_weights, device=mesh_device, dtype=weights_dtype, memory_config=mem_config)

    output_tensor = ttnn.embedding(tt_input_tensor, tt_weights, layout=layout)
    output_tensor = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert_with_pcc(torch_reference, output_tensor)
