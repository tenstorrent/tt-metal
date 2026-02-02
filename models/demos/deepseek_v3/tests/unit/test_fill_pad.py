# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.data_movement.test_fill_pad import (
    create_nd_padded_tiled_tensor,
    ttnn_dtype_to_torch_dtype,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, maybe_trace

DEEPSEEK_SHAPE_DTYPE_FILL_LIST = [
    (
        (
            1,
            32,
            8,
            32,
        ),
        ttnn.bfloat16,
        -float("inf"),
    ),
    ((1, 32, 8, 64), ttnn.bfloat16, 1.1754944e-38),
    ((1, 32, 8, 2), ttnn.bfloat16, 0),
    ((1, 1, 32, 8), ttnn.bfloat16, -float("inf")),
    ((1, 1, 32, 8), ttnn.uint16, 0),
    ((1, 1, 32, 8), ttnn.bfloat16, 0),
]


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("shape_dtype_fill", DEEPSEEK_SHAPE_DTYPE_FILL_LIST)
@pytest.mark.parametrize("mem_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("enable_trace", [True, False])
def test_fill_pad_deepseek(mesh_device, shape_dtype_fill, mem_config, enable_trace):
    shape, dtype, fill_value = shape_dtype_fill

    torch.manual_seed(1234)
    torch_input_tensor, padded_torch_tensor = create_nd_padded_tiled_tensor(
        shape, 32, fill_value, ttnn_dtype_to_torch_dtype[dtype]
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.fill_implicit_tile_padding(input_tensor, fill_value)

    tt_out_tensors = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    coords = list(tt_out_tensors.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    for coord, t in zip(coords, ttnn.get_device_tensors(tt_out_tensors)):
        if view is not None and not view.is_local(coord):
            continue
        padded_torch_output_tensor = ttnn.from_device(t).to_torch_with_padded_shape()
        assert_with_pcc(padded_torch_tensor, padded_torch_output_tensor)
