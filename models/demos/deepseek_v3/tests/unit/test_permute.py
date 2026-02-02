# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal, maybe_trace

DEEPSEEK_SHAPE_PERM_LAYOUT_MEM = [
    (
        (32, 1, 16, 128),
        (1, 2, 0, 3),
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        (7168, 1, 32, 8),
        (3, 1, 2, 0),
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.L1_MEMORY_CONFIG,
    ),
]


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("test_config", DEEPSEEK_SHAPE_PERM_LAYOUT_MEM)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_permute_deepseek(mesh_device, test_config, dtype, enable_trace):
    torch.manual_seed(2005)

    shape, perm, layout, memory_config = test_config

    torch_input = torch.rand(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    def run_op():
        return ttnn.permute(tt_input, perm)

    tt_output_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    torch_ref = torch.permute(torch_input, perm)
    coords = list(tt_output_tensors.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, tt_out_tensor in zip(coords, ttnn.get_device_tensors(tt_output_tensors)):
        if view is not None and not view.is_local(coord):
            continue
        torch_out = ttnn.to_torch(tt_out_tensor)
        assert_equal(torch_ref, torch_out)
