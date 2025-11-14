# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev


# Enumerate the post-commit cases explicitly
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, layout, input_dtype",
    [
        (4, 1, [1, 1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_all_broadcast(
    bh_1d_mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    output_tensors = []
    for k in range(num_devices):
        output_tensor = torch.rand(output_shape).bfloat16()
        output_tensors.append(output_tensor)
    temp_output_tensor = torch.cat(output_tensors, -1)
    input_tensor = ttnn.from_torch(
        temp_output_tensor,
        device=bh_1d_mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            bh_1d_mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(1, num_devices)),
        ),
    )
    tt_out_tensors = ttnn.all_broadcast(
        input_tensor,
        num_links=1,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(bh_1d_mesh_device)
