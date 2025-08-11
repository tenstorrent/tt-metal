# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.scripts.common import get_updated_device_params
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case
from ttnn import ShardTensorToMesh

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

NUM_DEVICES = ttnn.get_num_devices()


parameters = {
    "suite_1": {
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 1280],
            [8, 5, 13, 512],
            [8, 5, 32, 512],
            [1, 1, 32, 16384],
        ],
        "dim": [1,2,3],
        "layout": [ttnn.TILE_LAYOUT],#[ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
        "topology": [ttnn.Topology.Linear], #,ttnn.ccl.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def mesh_shape_iterator(num_devices: int=NUM_DEVICES, limit = None):
    assert num_devices%2==0


def mesh_device_fixture():
    logger.info("ALL GATHER: OPENING device mesh")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        #mesh_shape=ttnn.MeshShape(2, 4), **get_updated_device_params({})
    )
    logger.info("ALL GATHER: Opened device mesh")

    yield (mesh_device, "T3000 Mesh")

    ttnn.close_mesh_device(mesh_device)
    del mesh_device


def run(
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    topology,
    *,
    device,
) -> list:
    
    logger.info("STARTING SWEEP")
    
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")
    
    if input_shape[dim] % NUM_DEVICES != 0:
        input_shape[dim] *= NUM_DEVICES

    input_tensor = torch.rand(input_shape).bfloat16()

    ttnn_tensor = ttnn.from_torch(
        input_tensor, layout=layout, mesh_mapper=ShardTensorToMesh(device, dim=dim)
    )
    input_tensor_mesh = ttnn.to_device(ttnn_tensor, device)
    
    compute_grid_size = device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphore=ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)
    
    for i in range(num_iters):
        try:
            start_time = start_measuring_time()
            tt_out_tensor = ttnn.experimental.all_gather_async(
                input_tensor_mesh, 
                dim, 
                num_links=num_links, 
                memory_config=mem_config, 
                topology=topology,
                multi_device_global_semaphore=semaphore
            )
            e2e_perf = stop_measuring_time(start_time)
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")

        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        return [(eq, output), e2e_perf]
