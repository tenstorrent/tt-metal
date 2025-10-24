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
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import is_unsupported_case

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()

FABRIC_CONFIGS = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
    ttnn.FabricConfig.FABRIC_2D_DYNAMIC,
]

LEAD_MODEL_SHARD_SPECS = [
    get_serializable_shard_specs(
        input_shape=(32, 128),
        input_cores=(2, 5),
        input_strategy="w",
        output_shape=None,
        output_cores=(2, 5),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 1280]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 160),
        input_cores=(4, 6),
        input_strategy="w",
        output_shape=None,
        output_cores=(4, 8),  # production is (5,6) because they pad the output.
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 3584]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 64),
        input_cores=(4, 6),
        input_strategy="w",
        output_shape=None,
        output_cores=(2, 5),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 1280]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 160),
        input_cores=(4, 6),
        input_strategy="w",
        output_shape=None,  # (32, 32) in production on Galaxy
        output_cores=(5, 5),  # production is (5,6) because they pad the output.
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 3200]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 128),
        input_cores=(7, 8),
        input_strategy="w",
        output_shape=None,
        output_cores=(4, 4),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 7168]],
    ),
]

parameters = {
    "generality_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 256],
            [1, 1, 32, 248],
            [1, 1, 1, 32, 256],
            [2, 32, 256],
            [1, 1, 32, 16384],
        ],
        "dim": [0, 1, 2, 3, 4],
        "cluster_axis": [0, 1, None],
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
            [1, 1, 32, 2880],  # GPT-OSS 20B. Dim: 3, cluster_axis 1
            [1, 1, 32, 1280],  # Qwen3 dim: 2 cluster_axis: 1; Llama glx dim: 2 cluster_axis 1
            [1, 1, 32, 3200],  # Qwen3 dim: 3 cluster_axis: 1
            [1, 1, 32, 3584],  # Llama Glx. dim:3 cluster_axis:1
            [1, 1, 32, 7168],  # DeepSeek dim:3 cluster_axis 1
            [1, 1, 32, 1536],  # DeepSeek dim:3 cluster_axis 1
            [1, 32, 128, 576],  # DeepSeek dim: 1 cluster_axis 1
        ],
        "dim": [1, 2, 3],
        "cluster_axis": [0, 1],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
        "shard_specs": [None] + LEAD_MODEL_SHARD_SPECS,
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape, **kwargs):
    return input_shape[dim] % (NUM_DEVICES if cluster_axis is None else mesh_shape[cluster_axis]) == 0


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # L1 sharding only
    if test_vector["shard_specs"] is not None and test_vector["buffer_type"] == ttnn.BufferType.DRAM:
        return True, "L1 Sharding only"

    input_shape = test_vector["input_shape"]
    cluster_axis = test_vector["cluster_axis"]
    mesh_shape = test_vector["mesh_shape"]
    input_shape = test_vector["input_shape"]
    dim = test_vector["dim"]

    cluster_size = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)

    if not validate_serializable_shard_spec(input_shape, test_vector["shard_specs"], dim, cluster_size, "scatter"):
        return True, "Invalid shard spec"

    # hardcode for 6U
    if mesh_shape in [(16, 2), (2, 16)]:
        return True, "Invalid mesh shape for 6U"

    if cluster_axis is not None and mesh_shape[cluster_axis] == 1:
        return True, "Only one device along axis"

    if dim >= len(input_shape):
        return True, "Dim greater than rank"
    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"

    if not _valid_cluster_div(**test_vector):
        return True, "Shape at given dim not divisible by cluster devices"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _reference_map_op(math_op):
    if math_op == ttnn.ReduceType.Sum:
        return torch.sum
    else:
        raise NotImplementedError(f"Math op: {math_op} not yet implemented in sweep")


def _get_tensors(
    input_shape,
    mesh_shape,
    dim,
    cluster_axis,
    dtype,
    buffer_type,
    shard_specs,
    layout,
    device,
    math_op=ttnn.ReduceType.Sum,
):
    assert _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape)

    torch_input = torch.randn(input_shape).bfloat16()

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    per_device_dim = input_shape[dim] // replicate_dim

    torch_reference = torch_input.unsqueeze(0).repeat([replicate_dim] + [1] * len(input_shape))
    torch_references = _reference_map_op(math_op)(torch_reference, dim=0).split(per_device_dim, dim=dim)

    input_memory_config, output_memory_config = get_mem_configs(
        buffer_type, shard_specs, layout, torch_references[0].shape
    )

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )

    return tt_input, torch_references, output_memory_config


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
    shard_specs,
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

        tt_input, torch_references, output_memory_config = _get_tensors(
            input_shape,
            mesh_shape,
            dim,
            cluster_axis,
            input_dtype,
            buffer_type,
            shard_specs,
            layout,
            device,
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphores = [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(3)]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                    tt_input,
                    dim=dim,
                    multi_device_global_semaphore=semaphores,
                    cluster_axis=cluster_axis,
                    topology=topology,
                    num_links=num_links,
                    memory_config=output_memory_config,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                raise RuntimeError(f"Execution failed: {e}")

            logger.info(f"Done iteration {i}")

        for i, (t, ref) in enumerate(zip(ttnn.get_device_tensors(tt_out_tensor), torch_references)):
            logger.info("Bringing tensor back to host")
            tt_output_tensor = ttnn.to_torch(t)
            logger.info(f"Brought tensor {i} back from host. Shape: {tt_output_tensor.shape}")

            eq, output = comp_pcc(tt_output_tensor, ref)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
            return [(eq, output), e2e_perf]
