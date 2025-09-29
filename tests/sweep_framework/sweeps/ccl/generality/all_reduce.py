# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import (
    device_context,
    get_mem_configs,
    get_serializable_shard_specs,
    mesh_shape_iterator,
    validate_serializable_shard_spec,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

# Get the number of available devices to dynamically generate mesh shapes
NUM_DEVICES = ttnn.get_num_devices()


# parameters from:
# https://docs.google.com/spreadsheets/d/18lQ_dJpodMkoDFZjt7TfHdt0cEGsa5GCxxRKDzErGvM/edit?usp=sharing
TRAINING_SHAPES = [
    [1, 1, 1, 4096],
    [1, 1, 1, 2048],
    ["B", 32, 2048, 8],
    ["B", 32, 2048, 16],
    ["B", 32, 4096, 16],
    ["B", 32, 2048, 64],
    [1, 1, 32, 32],
    ["B", 32, 4096, 32],
    ["B", 32, 4096, 64],
    [1, 1, 32, 32],
    [1, 1, 1, 32],
    [1, 1, 1, 1],
    [1, 1, 1, 16],
    [1, 1, 1, 8],
    [1, 1, 8, 8],
    [1, 1, 16, 16],
    [1, 1, 1, 1],
    [1, 1, 1, 16],
    [1, 1, 16, 16],
]

TRAINING_BATCH = [1, 4, 8, 16, 32]


def _model_shape_iterator(model_shapes, batch_params):
    for shape in model_shapes:
        if "B" in shape:
            for b in batch_params:
                yield [b if x == "B" else x for x in shape]
        else:
            yield shape


# TODO set up test suite with training shapes

LEAD_MODEL_SHARD_SPECS = [
    get_serializable_shard_specs(
        input_shape=(32, 64),
        input_cores=(4, 6),
        input_strategy="w",
        output_shape=None,
        output_cores=(1, 10),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 1280]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 128),
        input_cores=(4, 6),
        input_strategy="w",
        output_shape=None,
        output_cores=(1, 10),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 2560]],
    ),
]


FABRIC_CONFIGS = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
    ttnn.FabricConfig.FABRIC_2D,
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
            [1, 1, 32, 31 * NUM_DEVICES],
            [1, 1, 1, 32, 32],
            [2, 32, 32],
            [1, 1, 32, 16384],
            [1, 1, 1, 2048],
        ],
        "cluster_axis": [0, 1, None],
        "math_op": [ttnn.ReduceType.Sum],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM],
        "shard_specs": [None],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
    "lead_model_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 1280],  # Qwen3 Galaxy. cluster_axis: 0
            [1, 1, 32, 2560],  # Qwen3 2x8. Cluster axis 0
        ],
        "cluster_axis": [0],
        "math_op": [ttnn.ReduceType.Sum],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
        "shard_specs": [None] + LEAD_MODEL_SHARD_SPECS,
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def _get_num_devices_in_cluster(cluster_axis, mesh_shape):
    """Helper to determine how many devices are in the CCL cluster"""
    if cluster_axis is None:
        return prod(mesh_shape)
    return mesh_shape[cluster_axis]


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # L1 sharding only
    if test_vector["shard_specs"] is not None and test_vector["buffer_type"] == ttnn.BufferType.DRAM:
        return True, "L1 Sharding only"

    if not validate_serializable_shard_spec(test_vector["input_shape"], test_vector["shard_specs"]):
        return True, "Invalid shard spec"

    mesh_shape, cluster_axis = test_vector["mesh_shape"], test_vector["cluster_axis"]
    if cluster_axis and mesh_shape[cluster_axis] == 1:
        return True, "Unit cluster axis"

    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"
    if (test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT) and (test_vector["input_dtype"] == ttnn.bfloat8_b):
        return True, "Row major not supported for bfloat8_b"

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


def _get_tensors(input_shape, cluster_axis, mesh_shape, math_op, dtype, layout, buffer_type, shard_specs, device):
    """
    Generates a replicated input tensor for the mesh and computes the golden reference tensor.
    """
    num_devices_in_cluster = _get_num_devices_in_cluster(cluster_axis, mesh_shape)

    # Create a single torch tensor and replicate it across all devices in the mesh
    if dtype == ttnn.uint32:
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(input_shape).bfloat16()

    # For all-reduce, the golden output is the sum of all replicated inputs.
    # Since the input is the same on all devices, this is equivalent to input * num_devices.
    # The final result is then replicated back to all devices.
    torch_reference = _reference_map_op(math_op)(torch.stack([torch_input] * num_devices_in_cluster), dim=0)

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, shard_specs, torch_reference.shape)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        memory_config=input_memory_config,
        device=device,
        dtype=dtype,
    )

    return tt_input, torch_reference, output_memory_config


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    shard_specs,
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

        tt_input, torch_reference, output_memory_config = _get_tensors(
            input_shape, cluster_axis, mesh_shape, math_op, input_dtype, layout, buffer_type, shard_specs, device
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        # all_reduce_async requires 3 semaphores
        semaphores = [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(7)]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.all_reduce_async(
                    tt_input,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    barrier_semaphores=semaphores[:2],
                    rs_global_semaphores=semaphores[2:5],
                    ag_global_semaphores=semaphores[5:],
                    math_op=math_op,
                    num_links=num_links,
                    memory_config=output_memory_config,
                    topology=topology,
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
