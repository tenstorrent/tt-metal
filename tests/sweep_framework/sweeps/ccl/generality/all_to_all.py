# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import device_context, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

# Get the number of available devices to dynamically generate mesh shapes
NUM_DEVICES = ttnn.get_num_devices()

FABRIC_CONFIGS = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
    ttnn.FabricConfig.FABRIC_2D_DYNAMIC,
]

# Define the parameter space for the sweep test
parameters = {
    "generality_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
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
        "in_dim": [0, 1, 2, 3, 4],
        "out_dim": [0, 1, 2, 3, 4],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Prunes the test space by invalidating known unsupported or problematic configurations.
    """

    # hardcode for 6U
    if test_vector["mesh_shape"] in [(16, 2), (2, 16)]:
        return True, "Invalid mesh shape for 6U"

    if test_vector["in_dim"] >= len(test_vector["input_shape"]) or test_vector["out_dim"] >= len(
        test_vector["input_shape"]
    ):
        return True, "in_dim or out_dim is out of range for the given input_shape rank"

    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring topology requires FABRIC_1D_RING fabric config"
    # invalidate when input and output dim are the same
    if test_vector["in_dim"] == test_vector["out_dim"]:
        return True, "in_dim and out_dim cannot be the same"

    return False, None


# Dummy device fixture required by the sweep runner
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(input_shape, in_dim, out_dim, mesh_shape, dtype, layout, device, mem_config):
    """
    Generates sharded input tensors for the mesh and computes the golden reference tensors.
    """

    num_devices = prod(mesh_shape)
    if dtype == ttnn.uint32:
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(input_shape).bfloat16()

    # The input tensor is sharded across devices along the in_dim
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(device, in_dim),
        memory_config=mem_config,
        device=device,
        dtype=dtype,
    )

    # The golden output is the input tensor chunked along the out_dim.
    # The all-to-all operation effectively transposes the sharding from in_dim to out_dim.
    torch_golden_per_device = torch.chunk(torch_input, num_devices, dim=out_dim)

    return tt_input, torch_golden_per_device


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    in_dim,
    out_dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    topology,
    *,
    device,  # unused
) -> list:
    """
    The main run function for the all_to_all sweep test.
    """
    logger.info("STARTING All-to-All SWEEP")

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        if device_err is not None:
            return False, device_err, None, None

        assert tuple(device.shape) == mesh_shape
        logger.info(f"Running test with vector: {locals()}")

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        semaphore = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)

        # Create persistent buffers required by the API
        output_shape = list(input_shape)
        output_shape[out_dim] //= prod(mesh_shape)
        persistent_intermediate_buffer = ttnn.allocate_tensor_on_device(
            output_shape, input_dtype, layout, device, mem_config
        )

        tt_input, torch_golden_per_device = _get_tensors(
            input_shape, in_dim, out_dim, mesh_shape, input_dtype, layout, device, mem_config
        )
        persistent_output_buffer = ttnn.allocate_tensor_on_device(output_shape, input_dtype, layout, device, mem_config)

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                ttnn.experimental.all_to_all_async(
                    tt_input,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    persistent_intermediate_buffer=persistent_intermediate_buffer,
                    persistent_output_buffer=persistent_output_buffer,
                    multi_device_global_semaphore=semaphore,
                    num_links=num_links,
                    memory_config=mem_config,
                    topology=topology,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                raise

            logger.info(f"Done iteration {i}")

        # The output is sharded, so we compare each device's output to its corresponding golden chunk
        for i, t in enumerate(ttnn.get_device_tensors(persistent_output_buffer)):
            logger.info(f"Bringing tensor {i} from device {device.get_device_ids()[i]} back to host")
            tt_output_tensor = ttnn.to_torch(t)
            golden_output = torch_golden_per_device[i]

            eq, output = comp_pcc(tt_output_tensor, golden_output)
            if not eq:
                logger.error(f"Output mismatch for tensor {i}")
                # Return on first mismatch
                return [(eq, output), e2e_perf]

        # If all comparisons passed
        return [(True, "PCC check passed"), e2e_perf]
