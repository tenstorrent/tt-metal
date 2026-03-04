# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace


def _get_tensors(input_shape, dim, dtype, memory_config, layout, cluster_axis, device):
    mesh_shape = tuple(device.shape)
    cluster_size = prod(mesh_shape) if cluster_axis is None else mesh_shape[cluster_axis]

    assert input_shape[dim] % cluster_size == 0

    torch.manual_seed(0)
    torch_input = torch.rand(input_shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )

    dim_per_device = input_shape[dim] // cluster_size

    torch_reference_slices = []
    for x in range(mesh_shape[0]):
        for y in range(mesh_shape[1]):
            i, j = (x, y) if cluster_axis == 1 else (y, x)

            torch_reference_slice = torch_input.split(dim_per_device, dim=dim)[j]
            torch_reference_slices.append(torch_reference_slice)

    return tt_input, torch_reference_slices


DEEPSEEK_SHAPES = [
    [1, 32, 128, 576],
    [1, 32, 32, 576],
    [1, 128, 32, 512],
]

DIM = 1
CLUSTER_AXIS = 1


@pytest.mark.requires_device(["N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("dim", [DIM])
@pytest.mark.parametrize("shape", DEEPSEEK_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("cluster_axis", [CLUSTER_AXIS])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_mesh_partition_deepseek(mesh_device, shape, dim, dtype, mem_config, layout, cluster_axis, enable_trace):
    tt_input, torch_references = _get_tensors(shape, dim, dtype, mem_config, layout, cluster_axis, mesh_device)

    def run_op():
        return ttnn.mesh_partition(tt_input, dim=dim, cluster_axis=cluster_axis)

    tt_out_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_out_tensors.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, tt_out, torch_ref in zip(coords, ttnn.get_device_tensors(tt_out_tensors), torch_references):
        if view is not None and not view.is_local(coord):
            continue
        torch_out = ttnn.to_torch(tt_out)
        eq, output = comp_equal(torch_out, torch_ref)
        assert eq, f"Output mismatch between torch and ttnn all_broadcast: {output}"
