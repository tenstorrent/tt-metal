# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace


def _get_tensors(input_shape, dtype, memory_config, layout, device):
    torch.manual_seed(0)
    torch_input = torch.rand(input_shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )
    return tt_input, torch_input


DEEPSEEK_SHAPES = [
    [1, 16, 32, 576],
    [1, 32, 128, 576],
    [1, 32, 32, 576],
    [1, 4, 128, 512],
    [1, 128, 32, 512],
    [1, 16, 32, 128],
]

CLUSTER_AXIS = 1


@pytest.mark.requires_device(["N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("shape", DEEPSEEK_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("cluster_axis", [CLUSTER_AXIS])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_all_broadcast_deepseek(
    mesh_device, shape, dtype, mem_config, layout, num_links, cluster_axis, topology, enable_trace
):
    tt_input, torch_reference = _get_tensors(shape, dtype, mem_config, layout, mesh_device)

    def run_op():
        return ttnn.all_broadcast(
            tt_input,
            num_links=num_links,
            cluster_axis=cluster_axis,
            memory_config=mem_config,
            topology=topology,
        )

    tt_out_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for tt_out_tensor in tt_out_tensors:
        coords = list(tt_out_tensor.tensor_topology().mesh_coords())
        for coord, tt_out in zip(coords, ttnn.get_device_tensors(tt_out_tensor)):
            if view is not None and not view.is_local(coord):
                continue
            torch_out = ttnn.to_torch(tt_out)
            eq, output = comp_equal(torch_out, torch_reference)
            assert eq, f"Output mismatch between torch and ttnn all_broadcast: {output}"
