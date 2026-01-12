#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#  SPDX-License-Identifier: Apache-2.0


from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, maybe_trace

DEEPSEEK_SHAPE_PAIRS = [([1, 1, 1, 32], [1, 1, 4096, 64]), ([1, 1, 1, 32], [1, 1, 129280, 224])]

VOCAB_SIZE = 2048


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("shape_pair", DEEPSEEK_SHAPE_PAIRS)
@pytest.mark.parametrize("input_dtype", [ttnn.uint32])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_embedding(
    mesh_device,
    shape_pair,
    input_dtype,
    weights_dtype,
    mem_config,
    layout,
    enable_trace,
):
    torch.manual_seed(1234)

    input_shape, weights_shape = shape_pair

    torch_input_tensor = torch.randint(0, VOCAB_SIZE, tuple(input_shape[-2:]))
    torch_weights = torch.rand(weights_shape[-2:]).bfloat16()
    torch_reference = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    torch_reference = torch_reference.repeat([prod(mesh_device.shape)] + [1] * (len(input_shape) - 2))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, device=mesh_device, dtype=input_dtype, memory_config=mem_config
    )
    tt_weights = ttnn.from_torch(torch_weights, device=mesh_device, dtype=weights_dtype, memory_config=mem_config)

    def run_op():
        return ttnn.embedding(tt_input_tensor, tt_weights, layout=layout)

    output_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    output_tensor = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_with_pcc(torch_reference, output_tensor)
