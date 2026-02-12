# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, maybe_trace

DEEPSEEK_SHAPES_DTYPES = [[(1, 1, 32, 256), ttnn.bfloat16, (1, 1, 32, 8), ttnn.uint16]]


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("shapes_dtypes", DEEPSEEK_SHAPES_DTYPES)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_gather_deepseek(mesh_device, shapes_dtypes, dim, layout, mem_config, enable_trace):
    torch.manual_seed(0)

    input_shape, input_dtype, index_shape, index_dtype = shapes_dtypes

    torch_dtype = torch.bfloat16
    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    torch_index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)

    torch_gather = torch.gather(torch_input, dim, torch_index)

    ttnn_input = ttnn.from_torch(
        torch_input,
        input_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn_index = ttnn.from_torch(
        torch_index,
        index_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.gather(ttnn_input, dim, index=ttnn_index)

    tt_out_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_out_tensors.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, ttnn_gather in zip(coords, ttnn.get_device_tensors(tt_out_tensors)):
        if view is not None and not view.is_local(coord):
            continue
        assert ttnn_gather.shape == torch_index.shape
        assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))
