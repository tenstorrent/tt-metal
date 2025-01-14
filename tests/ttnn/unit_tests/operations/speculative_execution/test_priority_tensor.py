# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from tests.ttnn.unit_tests.operations.speculative_execution.sfd_common import (
    get_buffer_address,
)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 64, 128],
    ],
)
def test_priority_tensor(
    t3k_mesh_device,
    shape,
    function_level_defaults,
):
    num_devices = t3k_mesh_device.get_num_devices()
    num_devices_to_skip = 3
    num_cores = 64  # TODO: Get this from the device

    # Set up the core range set
    storage_grid = t3k_mesh_device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(num_cores, storage_grid, row_wise=True)

    # Create the priority tensor
    p_tensor = torch.ones((num_devices, num_cores))
    p_tensor[:num_devices_to_skip] = 0  # Set skipping priority
    p_tensor_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    p_tensor_tt = ttnn.from_torch(
        p_tensor,
        device=t3k_mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=p_tensor_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=0),
    )
    pt_tensor_address = get_buffer_address(p_tensor_tt)
    logger.info(f"Priority tensor address: {pt_tensor_address}")

    for i in range(num_devices):
        t3k_mesh_device.get_device(i).set_speculation_state(True, pt_tensor_address)
        logger.info(f"Device {i} speculation state: {t3k_mesh_device.get_device(i).get_speculation_state()}")

    pt_input = torch.randn((shape), dtype=torch.float32)

    tt_a = ttnn.from_torch(
        pt_input,
        device=t3k_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )

    tt_b = ttnn.from_torch(
        pt_input,
        device=t3k_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    logger.info(f"tt_input shape: {tt_a.shape}")

    tt_out = ttnn.add(tt_a, tt_b)
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(t3k_mesh_device, dim=-1))
    logger.info(f"tt_out shape: {tt_out.shape}")

    pt_out = torch.add(pt_input, pt_input)

    num_passing = 0
    expected_num_passing = num_devices - num_devices_to_skip
    for i in range(num_devices):
        tt_out_ = tt_out[..., i * shape[-1] : (i + 1) * shape[-1]]
        passing, output = comp_pcc(pt_out, tt_out_)
        logger.info(f"{i}: {output}")
        num_passing += int(passing)

    assert num_passing == expected_num_passing, f"Expected {expected_num_passing} passing, got {num_passing}"
