# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import device_context, mesh_shape_iterator
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()


parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": [ttnn.FabricConfig.FABRIC_1D],
        # TODO this seem to reliably cause hangs, and we can't recover from hangs right now
        # "fabric_config": [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricConfig.FABRIC_2D],
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
        "math_op": [ttnn.ReduceType.Sum],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
        "topology": [ttnn.Topology.Linear],  # , ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape, **kwargs):
    return input_shape[dim] % (NUM_DEVICES if cluster_axis is None else mesh_shape[cluster_axis]) == 0


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["dim"] >= len(test_vector["input_shape"]):
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


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, math_op, dtype, layout, device):
    assert _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape)

    torch_input = torch.randn(input_shape).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(device), device=device
    )

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    per_device_dim = input_shape[dim] // replicate_dim

    torch_reference = torch_input.unsqueeze(0).repeat([replicate_dim] + [1] * len(input_shape))
    torch_references = _reference_map_op(math_op)(torch_reference, dim=0).split(per_device_dim, dim=dim)

    return tt_input, torch_references


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    dim,
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
    logger.info("STARTING SWEEP")

    logger.info(vars())

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None

        logger.info("device set up")

        tt_input, torch_references = _get_tensors(
            input_shape, mesh_shape, dim, cluster_axis, math_op, input_dtype, layout, device
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphores = [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(2)]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.reduce_scatter_async(
                    tt_input,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    topology=topology,
                    from_remote_multi_device_global_semaphore=semaphores[0],
                    to_remote_multi_device_global_semaphore=semaphores[0],
                    math_op=math_op,
                    num_links=num_links,
                    memory_config=mem_config,
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
