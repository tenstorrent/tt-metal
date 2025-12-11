# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
##
from itertools import combinations
from math import prod
from time import sleep

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from models.common.utility_functions import skip_for_wormhole_b0


def run_with_trace(
    mesh_device,
    sender_coord,
    receiver_coord,
    topology,
    input_tensor,
    num_iter=20,
):
    """Run point_to_point with trace capture for performance testing"""
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.point_to_point(
        input_tensor,
        sender_coord,
        receiver_coord,
        topology=topology,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.point_to_point(
            input_tensor,
            sender_coord,
            receiver_coord,
            topology=topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


TEST_SHAPES = [
    (1, 1, 1, 7168),
]

MESH_SHAPE = (4, 1)


def _linear_coord(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


torch.set_printoptions(threshold=10000)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 217872}], indirect=True
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("trace_mode", [False])
def test_point_to_point(bh_2d_mesh_device, dtype, num_iters, trace_mode):
    shape = [1, 128]  # Per-device tensor shape
    coords = [(0, 0), (1, 0)]
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))
    num_devices = 4
    topology = ttnn.Topology.Linear

    validate_test(num_devices, topology, bh_2d_mesh_device.shape, 0)
    mesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    print("mesh device shape: ", mesh_device.shape)

    lcoord0, lcoord1 = (_linear_coord(c, list(mesh_device.shape)) for c in coords)
    coord0, coord1 = (ttnn.MeshCoordinate(c) for c in coords)

    idx_start0, idx_end0 = lcoord0 * shape[0], (lcoord0 + 1) * shape[0]
    idx_start1, idx_end1 = lcoord1 * shape[0], (lcoord1 + 1) * shape[0]

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
        }
    )
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [1, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config)

    # Run multiple iterations
    for iter_idx in range(num_iters):
        # Create per-device tensors like reduce_to_root does
        data_per_device = []
        for device_idx in range(num_devices):
            if device_idx == lcoord0:
                # Sender device gets non-zero data
                device_data = torch.linspace(1, prod(shape), prod(shape)).reshape(shape).to(dtype=dtype)
            else:
                # Other devices get zeros
                device_data = torch.zeros(shape, dtype=dtype)
            data_per_device.append(device_data)

        # Stack into [num_devices, ...shape] like reduce_to_root does
        input_tensor_torch = torch.stack(data_per_device, dim=0)
        print("input tensor torch shape: ", input_tensor_torch.shape)

        # Create tensor with tile specified - same pattern as reduce_to_root
        input_tensor = ttnn.from_torch(
            input_tensor_torch,
            layout=layout,
            tile=tile,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )
        print("input tensor tt shape: ", input_tensor.shape)

        if trace_mode and iter_idx == 0:
            # Use trace for all iterations after the first setup
            sent_tensor = run_with_trace(
                mesh_device,
                coord0,
                coord1,
                ttnn.Topology.Linear,
                input_tensor,
                num_iter=num_iters,
            )
            sent_tensor_torch = ttnn.to_torch(sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            # Compare sender's data with receiver's data
            expected = data_per_device[lcoord0]
            received = sent_tensor_torch[lcoord1]  # Index into the concatenated mesh tensor
            assert_equal(expected, received)
            # Trace mode runs all iterations in one go, so break after first
            break
        else:
            # Non-trace mode: run each iteration individually
            print("before point_to_point")
            sent_tensor = ttnn.point_to_point(
                input_tensor,
                coord0,
                coord1,
                topology=ttnn.Topology.Linear,
            )
            print("after point_to_point")
            sent_tensor_torch = ttnn.to_torch(sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            print("expected vs sent:")
            expected = data_per_device[lcoord0]
            received = sent_tensor_torch[lcoord1]  # Index into the concatenated mesh tensor
            print(expected)
            print(received)
            assert_equal(expected, received)

    logger.info(f"Waiting for op")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Done op")
