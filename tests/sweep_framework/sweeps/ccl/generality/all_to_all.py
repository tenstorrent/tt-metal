# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweeps.ccl.common import device_context, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

# Get the number of available devices to dynamically generate mesh shapes
NUM_DEVICES = ttnn.get_num_devices()


# Define the parameter space for the sweep test
parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": [
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricConfig.FABRIC_1D_RING,
        ],  # default is ring for this one
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 2048],
            [1, 1, 2048, 256],
        ],
        "in_dim": [0, 1, 2, 3, 4],  # only 2 and 3 are supported
        "out_dim": [0, 1, 2, 3, 4],  # only 2 and 3 are supported
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ],
        "topology": [ttnn.Topology.Ring, ttnn.Topology.Linear],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Prunes the test space by invalidating known unsupported or problematic configurations.
    """
    if test_vector["in_dim"] >= len(test_vector["input_shape"]) or test_vector["out_dim"] >= len(
        test_vector["input_shape"]
    ):
        return True, "in_dim or out_dim is out of range for the given input_shape rank"

    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring topology requires FABRIC_1D_RING fabric config"

    return False, None


# Dummy device fixture required by the sweep runner
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(input_shape, in_dim, out_dim, mesh_shape, dtype, layout, device):
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
        mesh_mapper=ttnn.ShardTensorToMesh(in_dim),
        device=device,
        dtype=dtype,
    )

    # The golden output is a transpose of the input, where each device receives a chunk
    # of the input tensor from every other device, scattered along the out_dim.
    torch_golden_chunks = torch.chunk(torch_input, num_devices, dim=in_dim)
    torch_golden_per_device = [torch.cat(torch_golden_chunks, dim=out_dim)] * num_devices

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

        tt_input, torch_golden_per_device = _get_tensors(
            input_shape, in_dim, out_dim, mesh_shape, input_dtype, layout, device
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        # all_to_all_async requires 1 semaphore
        semaphore = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)

        # Create persistent buffers required by the API
        output_shape = list(input_shape)
        output_shape[in_dim] //= prod(mesh_shape)
        persistent_intermediate_buffer = ttnn.allocate_tensor_on_device(
            output_shape, input_dtype, layout, device, mem_config
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
