# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from math import prod
from itertools import permutations
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweeps.ccl.common import device_context, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45
NUM_DEVICES = ttnn.get_num_devices()
FABRIC_CONFIGS = [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING]


# heuristic
def _coords_set_iterator(mesh_shape):
    coords = {(0, 0), (mesh_shape[0] - 1, 0), (0, mesh_shape[1] - 1), (mesh_shape[0] - 1, mesh_shape[1] - 1)}
    yield from set(permutations(coords, 2))


def _mesh_and_coords_iterator(num_devices, mesh_limit=None):
    for mesh_shape in mesh_shape_iterator(num_devices, mesh_limit):
        for coords in _coords_set_iterator(mesh_shape):
            yield mesh_shape, coords


parameters = {
    "suite_1": {
        "mesh_shape_and_coords": _mesh_and_coords_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 32],
            [1, 1, 32, 1280],
            [1, 1, 32, 31],
            [1, 1, 1, 32, 32],
            [2, 32, 32],
            [1, 1, 32, 16384],
            [1, 1, 1, 2048],
        ],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # fabric 2D not yet supported #TICKET
    if test_vector["fabric_config"] == ttnn.FabricConfig.FABRIC_2D:
        return True, "2D Fabric not yet supported"

    _, (coord0, coord1) = test_vector["mesh_shape_and_coords"]
    if test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_2D and (
        coord0[0] != coord1[0] and coord0[1] != coord1[1]
    ):
        return True, "Invalid coordinates for 1D fabric"

    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _linear_coord(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


def _shard_range(coord, shape, mesh_shape):
    lcoord = _linear_coord(coord, mesh_shape)
    return lcoord * shape[0], (lcoord + 1) * shape[0]


def _get_tensors(input_shape, coord0, dtype, layout, device):
    mesh_shape = tuple(device.shape)
    num_devices = prod(mesh_shape)

    idx_start0, idx_end0 = _shard_range(coord0, input_shape, mesh_shape)

    device_shard_shape = tuple(s * (num_devices if i == 0 else 1) for i, s in enumerate(input_shape))
    input_tensor_torch = torch.zeros(device_shard_shape).bfloat16()
    input_tensor_torch[idx_start0:idx_end0, :, :, :] = torch.randn(input_shape).bfloat16()

    tt_input = ttnn.from_torch(
        input_tensor_torch,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )

    return tt_input, input_tensor_torch


def run(
    mesh_shape_and_coords,
    fabric_config,
    input_shape,
    num_links,
    input_dtype,
    layout,
    num_iters,
    topology,
    *,
    device,  # unused
) -> list:
    logger.info(vars())

    mesh_shape, (coord0, coord1) = mesh_shape_and_coords

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None

        logger.info("device set up")

        tt_input, torch_reference = _get_tensors(input_shape, coord0, input_dtype, layout, device)

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphore = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.point_to_point(
                    tt_input,
                    ttnn.MeshCoordinate(coord1),
                    ttnn.MeshCoordinate(coord0),
                    topology=topology,
                    semaphore=semaphore,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                raise RuntimeError(f"Execution failed: {e}")

            logger.info(f"Done iteration {i}")

        output_tensor_torch = ttnn.to_torch(tt_out_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))

        idx_start0, idx_end0 = _shard_range(coord0, input_shape, mesh_shape)
        idx_start1, idx_end1 = _shard_range(coord1, input_shape, mesh_shape)

        output_tensor_torch = output_tensor_torch[idx_start1:idx_end1]
        torch_reference = torch_reference[idx_start0:idx_end0]

        eq, output = (comp_equal if input_dtype == ttnn.bfloat16 else comp_pcc)(output_tensor_torch, torch_reference)
        if not eq:
            logger.error(f"output mismatch")
        return [(eq, output), e2e_perf]
