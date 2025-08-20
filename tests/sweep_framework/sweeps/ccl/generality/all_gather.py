# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from itertools import count, takewhile
from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.scripts.common import get_updated_device_params
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()


def mesh_shape_iterator(num_devices: int = NUM_DEVICES, limit=None):
    assert num_devices % 2 == 0

    for r in takewhile(lambda x: x <= num_devices, (2**i for i in (range(limit) if limit else count()))):
        yield (num_devices // r, r)


parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(),
        "fabric_config": [ttnn.FabricConfig.FABRIC_1D],
        # TODO this seem to reliably cause hangs, and we can't recover from hangs right now
        #        "fabric_config": [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricConfig.FABRIC_2D],
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 32],
            [1, 1, 32, 1280],
            [1, 1, 32, 31],
            [1, 1, 1, 32, 32],
            [2, 32, 32],
            [1, 1, 32, 16384],
            [1, 1, 1, 2048],  # the following shapes are from training
            [
                1,
                1,
                1,
                4096,
            ],  # https://docs.google.com/spreadsheets/d/18lQ_dJpodMkoDFZjt7TfHdt0cEGsa5GCxxRKDzErGvM/edit?usp=sharing
            [1, 32, 2048, 8],
            [1, 32, 2048, 16],
            [1, 32, 4096, 16],
            [1, 32, 2048, 64],
            [1, 32, 4096, 32],
            [1, 32, 4096, 64],
            [1, 1, 1, 1],
            [1, 1, 1, 8],
            [1, 1, 1, 16],
            [1, 1, 1, 32],
            [1, 1, 8, 8],
            [1, 1, 16, 16],
        ],
        "dim": [0, 1, 2, 3, 4],
        "cluster_axis": [0, 1, None],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["dim"] >= len(test_vector["input_shape"]):
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


@contextmanager
def device_context(mesh_shape, fabric_config):
    mesh_device = None
    try:
        logger.info("Setting up device")
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape), **get_updated_device_params({}))
        yield mesh_device, None
    except AssertionError as e:
        logger.error(f"Device error: {e}")
        yield None, f"Device error {e}"
    finally:
        logger.info("Tearing down device")
        if mesh_device:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            del mesh_device


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, dtype, layout, device):
    torch_input = torch.rand(input_shape).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(device), device=device
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
    mem_config,
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
            input_shape, mesh_shape, dim, cluster_axis, input_dtype, layout, device
        )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphore = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    tt_input,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    topology=topology,
                    multi_device_global_semaphore=semaphore,
                    num_links=num_links,
                    memory_config=mem_config,
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
