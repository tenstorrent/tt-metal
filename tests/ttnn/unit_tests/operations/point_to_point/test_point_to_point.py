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


TEST_SHAPES = [
    (1, 1, 1, 16),
    (1, 1, 8, 16),
    (1, 1, 1, 64),
    (1, 1, 3, 128),
    (1, 13, 1, 32),
    (1, 1, 1, 32),
    (100, 1, 1, 16),
    (1, 1, 1, 24),
    # (1, 1, 2, 8), alignment issues here too
    # (1, 1, 2, 17), TODO make non-aligned tensors work
    (1, 1, 1, 7168),
    (1, 1, 32, 7168),
]

MESH_SHAPE = (2, 4)


def _linear_coord(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


def _get_test_coords_and_shapes(mesh_shape, tensor_shapes):
    coords = [(i, j) for i in range(mesh_shape[0]) for j in range(mesh_shape[1])]

    # only test all coordinate permutations against the first tensor shape to keep test time reasonable
    for i, shape in enumerate(tensor_shapes):
        if i == 0:
            # here we do coordinate filtering appropriate for 1D fabric
            one_d_filter = lambda cc: cc[0][0] == cc[1][0] or cc[0][1] == cc[1][1]
            yield from map(lambda cc: (shape, cc), filter(one_d_filter, combinations(coords, 2)))
        else:
            yield (shape, (coords[0], coords[1]))


torch.set_printoptions(threshold=10000)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("shape_coords", _get_test_coords_and_shapes(MESH_SHAPE, TEST_SHAPES))
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_point_to_point(mesh_device, shape_coords, layout, dtype):
    shape, coords = shape_coords

    devices = prod(list(mesh_device.shape))
    multi_device_shape = tuple(s * (devices if i == 0 else 1) for i, s in enumerate(shape))

    lcoord0, lcoord1 = (_linear_coord(c, list(mesh_device.shape)) for c in coords)
    coord0, coord1 = (ttnn.MeshCoordinate(c) for c in coords)

    idx_start0, idx_end0 = lcoord0 * shape[0], (lcoord0 + 1) * shape[0]
    idx_start1, idx_end1 = lcoord1 * shape[0], (lcoord1 + 1) * shape[0]

    input_tensor_torch = torch.zeros(multi_device_shape, dtype=dtype)
    input_tensor_torch[idx_start0:idx_end0, :, :, :] = (
        torch.linspace(1, prod(shape), prod(shape)).reshape(shape).to(dtype=dtype)
    )
    input_tensor = ttnn.from_torch(
        input_tensor_torch,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    sent_tensor = ttnn.point_to_point(
        input_tensor,
        coord1,
        coord0,
        ttnn.Topology.Linear,
    )
    sent_tensor_torch = ttnn.to_torch(sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_equal(input_tensor_torch[idx_start0:idx_end0, :, :, :], sent_tensor_torch[idx_start1:idx_end1, :, :, :])

    # test optional output tensor
    return_tensor = ttnn.from_torch(
        torch.zeros(sent_tensor_torch.shape, dtype=dtype),
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.point_to_point(
        sent_tensor,
        coord0,
        coord1,
        ttnn.Topology.Linear,
        optional_output_tensor=return_tensor,
    )

    torch_return_tensor = ttnn.to_torch(return_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_equal(input_tensor_torch[idx_start0:idx_end0, :, :, :], torch_return_tensor[idx_start0:idx_end0, :, :, :])


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 500000}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("shape_coords", [((1, 1, 1, 16), ((0, 0), (0, 1)))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_point_to_point_with_device_delay(mesh_device, shape_coords, layout, dtype):
    shape, coords = shape_coords

    devices = prod(list(mesh_device.shape))
    multi_device_shape = tuple(s * (devices if i == 0 else 1) for i, s in enumerate(shape))

    lcoord0, lcoord1 = (_linear_coord(c, list(mesh_device.shape)) for c in coords)
    coord0, coord1 = (ttnn.MeshCoordinate(c) for c in coords)

    idx_start0, idx_end0 = lcoord0 * shape[0], (lcoord0 + 1) * shape[0]
    idx_start1, idx_end1 = lcoord1 * shape[0], (lcoord1 + 1) * shape[0]

    input_tensor_torch = torch.zeros(multi_device_shape, dtype=dtype)
    input_tensor_torch[idx_start0:idx_end0, :, :, :] = (
        torch.linspace(1, prod(shape), prod(shape)).reshape(shape).to(dtype=dtype)
    )
    input_tensor = ttnn.from_torch(
        input_tensor_torch,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    input_tensor2 = ttnn.from_torch(
        input_tensor_torch * 2,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    delays = []
    for i in range(MESH_SHAPE[0]):
        delay_at_i = []
        for j in range(MESH_SHAPE[1]):
            delay_at_i.append(0)
        delays.append(delay_at_i)
    delays[coord1[0]][coord1[1]] = 800000

    # Compile programs
    sent_tensor = ttnn.point_to_point(
        input_tensor,
        coord1,
        coord0,
        ttnn.Topology.Linear,
    )
    ttnn.apply_device_delay(
        mesh_device, delays
    )  # tests for a potential race by having receive attempt to increment the semaphore before the send is done

    # Capture trace
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    sent_tensor = ttnn.point_to_point(
        input_tensor,
        coord1,
        coord0,
        ttnn.Topology.Linear,
    )
    ttnn.apply_device_delay(
        mesh_device, delays
    )  # tests for a potential race by having receive attempt to increment the semaphore before the send is done
    sent_tensor2 = ttnn.point_to_point(
        input_tensor2,
        coord1,
        coord0,
        ttnn.Topology.Linear,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute trace
    for i in range(10):
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    # Verify results
    sent_tensor_torch = ttnn.to_torch(sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_equal(input_tensor_torch[idx_start0:idx_end0, :, :, :], sent_tensor_torch[idx_start1:idx_end1, :, :, :])
    sent_tensor2_torch = ttnn.to_torch(sent_tensor2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_equal(input_tensor_torch[idx_start0:idx_end0, :, :, :] * 2, sent_tensor2_torch[idx_start1:idx_end1, :, :, :])

    # 1 for send/receive and 1 for device delay
    assert (
        mesh_device.num_program_cache_entries() == 2
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"
