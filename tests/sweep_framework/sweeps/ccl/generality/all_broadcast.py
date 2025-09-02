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
TIMEOUT = 45

# Get the number of available devices to dynamically generate mesh shapes
NUM_DEVICES = ttnn.get_num_devices()


# Define the parameter space for the sweep test
parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING],
        "num_links": [1],
        "cluster_axis": [0, 1],
        "input_shape": [
            # [1, 1, 32, 256],
            [1, 1, 4096, 32],
        ],
        "layout": [ttnn.TILE_LAYOUT],  # , ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],  # , ttnn.bfloat8_b, ttnn.uint32],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ],
        "topology": [ttnn.Topology.Linear],  # , ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Prunes the test space by invalidating known unsupported or problematic configurations.
    """
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, "Row major not supported for bfloat8_b"

    if test_vector["cluster_axis"] >= len(test_vector["mesh_shape"]):
        return True, "cluster_axis is out of range for the given mesh_shape rank"

    return False, None


# Dummy device fixture required by the sweep runner
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(input_shape, mesh_shape, dtype, layout, device):
    """
    Generates a replicated input tensor for the mesh and a list of golden reference tensors.
    """
    num_devices = prod(mesh_shape)
    # Create a single torch tensor that will be replicated to all devices.
    if dtype == ttnn.uint32:
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(input_shape).bfloat16()

    # The golden output is a list of identical tensors, one for each device.
    torch_golden_list = [torch_input] * num_devices

    # Create a tt_input tensor by replicating the torch_input to all devices in the mesh.
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
        dtype=dtype,
    )

    return tt_input, torch_golden_list


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    num_links,
    cluster_axis,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    topology,
    *,
    device,  # unused
) -> list:
    """
    The main run function for the all_broadcast sweep test.
    """
    logger.info("STARTING All-Broadcast SWEEP")

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        if device_err is not None:
            return False, device_err, None, None

        assert tuple(device.shape) == mesh_shape
        logger.info(f"Running test with vector: {locals()}")

        # Setup subdevice and stall group, similar to the unit test
        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        sub_device_manager = device.create_sub_device_manager([worker_sub_device], 0)
        device.load_sub_device_manager(sub_device_manager)
        device.set_sub_device_stall_group(sub_device_stall_group)

        try:
            tt_input, torch_golden_list = _get_tensors(input_shape, mesh_shape, input_dtype, layout, device)

            for i in range(num_iters):
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.all_broadcast_async(
                    tt_input,
                    num_links=num_links,
                    memory_config=mem_config,
                    topology=topology,
                    subdevice_id=worker_sub_device_id,
                    cluster_axis=cluster_axis,
                )
                e2e_perf = stop_measuring_time(start_time)
                logger.info(f"Done iteration {i}")

            logger.info("Waiting for op to complete")
            ttnn.synchronize_device(device, sub_device_ids=sub_device_stall_group)
            logger.info("Op complete")

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise
        finally:
            # Clean up subdevice resources
            device.reset_sub_device_stall_group()
            device.clear_loaded_sub_device_manager()

        # The output is replicated, so we compare each device's output to the same golden reference
        num_tensors = len(tt_out_tensor)
        for dev in range(num_tensors):
            for i, (t, ref) in enumerate(zip(ttnn.get_device_tensors(tt_out_tensor[dev]), torch_golden_list)):
                logger.info(f"Bringing tensor {i} from device {device.get_device_ids()[i]} back to host")
                tt_output_tensor = ttnn.to_torch(t)
                torch_golden = ref

                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, torch_golden)
                else:
                    eq, output = comp_pcc(tt_output_tensor, torch_golden)

                if not eq:
                    logger.error(f"Output mismatch for tensor {i}")
                    # Return on first mismatch
                    return [(eq, output), e2e_perf]

        # If all comparisons passed
        return [(True, "Comparison passed"), e2e_perf]
