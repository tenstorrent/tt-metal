# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import device_context, get_mem_config, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()

FABRIC_CONFIGS = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
    ttnn.FabricConfig.FABRIC_2D,
    ttnn.FabricConfig.FABRIC_2D_DYNAMIC,
]


parameters = {
    "generality_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 32],
            [1, 1, 32, 31],
            [1, 1, 1, 32, 32],
            [2, 32, 32],
            [1, 1, 32, 16384],
            [1, 1, 1, 2048],
        ],
        "dim": [0, 1, 2, 3, 4],
        "cluster_axis": [0, 1, None],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM],
        "shard_shape": [None],
        "shard_strategy": [None],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
    "lead_model_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 1440],  # GPT-OSS 20B. Dim: 3, cluster_axis 1
            [1, 1, 32, 32],  # Qwen3
            [1, 8, 8, 128],  # Qwen3
            [3, 1, 4096, 192],  # Gemma3 Dim: 3
            [3, 1, 4096, 144],  # Gemma3 Dim: 3
        ],
        "dim": [1, 3],
        "cluster_axis": [0, 1],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
        "shard_shape": [None],  # TODO (32,32),(32,128)],
        "shard_strategy": [None],  # TODO ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.TensorMemoryLayout.HEIGHT_SHARDED],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # Kind of hardcode the model specific sharding configs
    input_shape = test_vector["input_shape"]
    shard_shape = test_vector["shard_shape"]
    shard_strategy = test_vector["shard_strategy"]

    if shard_shape is None and shard_strategy is not None:
        return True, "Invalid application of shard config"
    elif shard_shape == (32, 32):
        if input_shape != [1, 1, 32, 32]:
            return True, "Invalid application of shard config"
        if shard_strategy != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return True, "Invalid application of shard config"
    elif shard_shape == (32, 128):
        if input_shape != [1, 8, 8, 128]:
            return True, "Invalid application of shard config"
        if shard_strategy != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            return True, "Invalid application of shard config"

    # hardcode for 6U
    if test_vector["mesh_shape"] in [(16, 2), (2, 16)]:
        return True, "Invalid mesh shape for 6U"

    cluster_axis = test_vector["cluster_axis"]
    if cluster_axis is not None and test_vector["mesh_shape"][cluster_axis] == 1:
        return True, "Only one device along axis"

    if test_vector["dim"] >= len(input_shape):
        return True, "Dim greater than rank"
    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(
    input_shape, mesh_shape, dim, cluster_axis, dtype, layout, buffer_type, shard_shape, shard_strategy, device
):
    torch_input = torch.rand(input_shape).bfloat16()

    mem_config = get_mem_config(buffer_type, shard_shape, shard_strategy, device)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

    return tt_input, torch_reference


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    dim,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    shard_shape,
    shard_strategy,
    num_iters,
    topology,
    *,
    device,  # unused
) -> list:
    logger.info("STARTING SWEEP")

    logger.info(vars())

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None

        logger.info("device set up")

        tt_input, torch_reference = _get_tensors(
            input_shape,
            mesh_shape,
            dim,
            cluster_axis,
            input_dtype,
            layout,
            buffer_type,
            shard_shape,
            shard_strategy,
            device,
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphores = [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(2)]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    tt_input,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    topology=topology,
                    multi_device_global_semaphore=semaphores,
                    num_links=num_links,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                raise RuntimeError(f"Execution failed: {e}")

            logger.info(f"Done iteration {i}")

        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            logger.info("Bringing tensor back to host")
            tt_output_tensor = ttnn.to_torch(t)
            logger.info("Brought tensor back from host")

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, torch_reference)
            else:
                eq, output = comp_pcc(tt_output_tensor, torch_reference)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
            return [(eq, output), e2e_perf]
