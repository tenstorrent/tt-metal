# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace


def _get_tensors(input_shape_list, dim, dtype, memory_config, layout, device):
    torch.manual_seed(0)
    torch_inputs = [torch.rand(shape).bfloat16() for shape in input_shape_list]

    tt_inputs = [
        ttnn.from_torch(
            torch_input,
            layout=layout,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            device=device,
        )
        for torch_input in torch_inputs
    ]
    return tt_inputs, torch.cat(torch_inputs, dim=dim)


DEEPSEEK_SHAPE_LISTS = [
    [(1, 32, 32, 512), (1, 32, 32, 64)],
    [
        (1, 16, 32, 576),
        (1, 16, 32, 576),
        (1, 16, 32, 576),
        (1, 16, 32, 576),
        (1, 16, 32, 576),
        (1, 16, 32, 576),
        (1, 16, 32, 576),
        (1, 16, 32, 576),
    ],
    [(1, 1, 32, 512), (1, 1, 32, 64)],
    [
        (1, 4, 128, 512),
        (1, 4, 128, 512),
        (1, 4, 128, 512),
        (1, 4, 128, 512),
        (1, 4, 128, 512),
        (1, 4, 128, 512),
        (1, 4, 128, 512),
        (1, 4, 128, 512),
    ],
    [
        (1, 16, 32, 128),
        (1, 16, 32, 128),
        (1, 16, 32, 128),
        (1, 16, 32, 128),
        (1, 16, 32, 128),
        (1, 16, 32, 128),
        (1, 16, 32, 128),
        (1, 16, 32, 128),
    ],
]
CLUSTER_AXIS = 1


@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("shape_list", DEEPSEEK_SHAPE_LISTS)
@pytest.mark.parametrize("dim", [3, 1])  # slightly overkill
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_concat_deepseek(mesh_device, shape_list, dim, dtype, mem_config, layout, enable_trace):
    if len(set(tuple(x for i, x in enumerate(shape) if i != dim) for shape in shape_list)) != 1:
        pytest.skip("Invalid concat shapes")

    tt_inputs, torch_reference = _get_tensors(shape_list, dim, dtype, mem_config, layout, mesh_device)

    def run_op():
        return ttnn.concat(tt_inputs, dim=dim)

    tt_out_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_out_tensors.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, tt_out_tensor in zip(coords, ttnn.get_device_tensors(tt_out_tensors)):
        if view is not None and not view.is_local(coord):
            continue
        torch_out = ttnn.to_torch(tt_out_tensor)
        eq, output = comp_equal(torch_out, torch_reference)
        assert eq, f"Output mismatch between torch and ttnn all_broadcast: {output}"
