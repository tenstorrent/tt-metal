# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace

DEEPSEEK_SHAPE_PADDED_FILL_MEM = [
    ((1, 1, 32, 576), (1, 32, 32, 576), 0, ttnn.DRAM_MEMORY_CONFIG),
    (
        (1, 32, 8, 32),
        (1, 32, 32, 64),
        float("-inf"),
        ttnn.L1_MEMORY_CONFIG,
    ),
    (
        (1, 1, 32, 8),
        (1, 1, 32, 64),
        float("-inf"),
        ttnn.L1_MEMORY_CONFIG,
    ),
]


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("test_config", DEEPSEEK_SHAPE_PADDED_FILL_MEM)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_pad_deepseek(mesh_device, test_config, dtype, layout, enable_trace):
    shape, padded_shape, pad_value, memory_config = test_config

    torch.manual_seed(0)
    torch_input = torch.rand(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    def run_op():
        return ttnn.pad(tt_input, padded_shape, [0, 0, 0, 0], value=pad_value, use_multicore=True)

    tt_outputs = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    shape_diff = list(map(lambda x, y: x - y, padded_shape, shape))
    torch_ref = torch.nn.functional.pad(torch_input, sum([[0, pd] for pd in reversed(shape_diff)], []), value=pad_value)

    coords = list(tt_outputs.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, tt_out in zip(coords, ttnn.get_device_tensors(tt_outputs)):
        if view is not None and not view.is_local(coord):
            continue
        torch_out = ttnn.to_torch(tt_out)
        eq, output = comp_equal(torch_out, torch_ref)
        assert eq, f"Output mismatch between torch and ttnn all_broadcast: {output}"
