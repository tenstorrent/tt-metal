# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweeps.ccl.common import device_context, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

# Get the number of available devices to dynamically generate mesh shapes
NUM_DEVICES = ttnn.get_num_devices()


# Define the parameter space for the sweep test
parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": [ttnn.FabricConfig.FABRIC_1D],
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 256],
            [1, 1, 4096, 32],
            [1, 1, 32, 16384],
            # [1, 1, 1, 4096],
            # [1, 1, 1, 2048],
            # [1, 32, 2048, 8],
            # [8, 32, 2048, 8],
            # [1, 32, 2048, 16],
            # [2, 32, 2048, 16],
            # [1, 32, 4096, 16],
            # [1, 32, 2048, 64],
            # [8, 32, 2048, 64],
            # [1, 1, 32, 32],
            # [1, 1, 1, 32],
            # [1, 1, 1, 1],
            # [1, 1, 1, 8],
            # [1, 1, 1, 16],
            # [1, 1, 8, 8],
            # [1, 1, 16, 16]
        ],
        "cluster_axis": [0, 1],
        "math_op": [ttnn.ReduceType.Sum],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ],
        "topology": [ttnn.Topology.Linear],
        "num_iters": [1],
    },
}


def _get_num_devices_in_cluster(cluster_axis, mesh_shape):
    """Helper to determine how many devices are in the CCL cluster"""
    if cluster_axis is None:
        return prod(mesh_shape)
    return mesh_shape[cluster_axis]


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Prunes the test space by invalidating known unsupported or problematic configurations.
    """
    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"
    if (test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT) and (test_vector["input_dtype"] == ttnn.bfloat8_b):
        return True, "Row major not supported for bfloat8_b"

    # Invalidate known hanging cases
    if (
        test_vector["mesh_shape"] == (8, 1)
        and test_vector["input_shape"] == [1, 1, 32, 16384]
        and test_vector["cluster_axis"] == 0
    ):
        return True, "Known hang case for 8-device all-reduce"

    return False, None


# Dummy device fixture required by the sweep runner
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _reference_map_op(math_op):
    """Maps the ttnn reduce type to a torch function"""
    if math_op == ttnn.ReduceType.Sum:
        return torch.sum
    else:
        raise NotImplementedError(f"Math op: {math_op} not yet implemented in sweep")


def _get_tensors(input_shape, cluster_axis, mesh_shape, math_op, dtype, layout, device):
    """
    Generates a replicated input tensor for the mesh and computes the golden reference tensor.
    """
    num_devices_in_cluster = _get_num_devices_in_cluster(cluster_axis, mesh_shape)

    # Create a single torch tensor and replicate it across all devices in the mesh
    if dtype == ttnn.uint32:
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(input_shape).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(device), device=device, dtype=dtype
    )

    # For all-reduce, the golden output is the sum of all replicated inputs.
    # Since the input is the same on all devices, this is equivalent to input * num_devices.
    # The final result is then replicated back to all devices.
    torch_reference = _reference_map_op(math_op)(torch.stack([torch_input] * num_devices_in_cluster), dim=0)

    return tt_input, torch_reference


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    topology,
    math_op,
    *,
    device,  # unused
) -> list:
    """
    The main run function for the all_reduce sweep test.
    """
    logger.info("STARTING All-Reduce SWEEP")

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        if device_err is not None:
            return False, device_err, None, None

        assert tuple(device.shape) == mesh_shape
        logger.info(f"Running test with vector: {locals()}")

        tt_input, torch_reference = _get_tensors(
            input_shape, cluster_axis, mesh_shape, math_op, input_dtype, layout, device
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        # all_reduce_async requires 3 semaphores
        semaphores = [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(3)]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.all_reduce_async(
                    tt_input,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    topology=topology,
                    from_remote_multi_device_global_semaphore=semaphores[0],
                    to_remote_multi_device_global_semaphore=semaphores[1],
                    gather_multi_device_global_semaphore=semaphores[2],
                    math_op=math_op,
                    num_links=num_links,
                    memory_config=mem_config,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                raise

            logger.info(f"Done iteration {i}")

        # The output is replicated, so we compare each device's output to the same golden reference
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            logger.info(f"Bringing tensor {i} from device {device.get_device_ids()[i]} back to host")
            tt_output_tensor = ttnn.to_torch(t)

            eq, output = comp_pcc(tt_output_tensor, torch_reference)
            if not eq:
                logger.error(f"Output mismatch for tensor {i}")
                # Return on first mismatch
                return [(eq, output), e2e_perf]

        # If all comparisons passed
        return [(True, "PCC check passed"), e2e_perf]
