# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import ttnn

from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import device_context, mesh_shape_iterator
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import run_all_to_all_combine_test


# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()


def _pd(val: int):
    return val * NUM_DEVICES


parameters = {
    "suite_1": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricConfig.FABRIC_2D],
        "input_shape": [
            [_pd(1), 1, 8, 32],
            [_pd(1), 1, 2, 1280],
            [_pd(1), 1, 8, 31],
            [_pd(8), 1, 2, 7168],
            [_pd(16), 1, 2, 7168],
            [_pd(1), 1, 2, 16384],
        ],
        "experts": [_pd(i) for i in [2, 4, 8]],
        "select_experts_k": [2, 4, 8],
        "local_reduce": [False, True],
        "cluster_axis": [0, 1, None],
        "num_links": [1, 2, 3],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["select_experts_k"] > test_vector["experts"]:
        return True, "k greater than experts"

    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    experts,
    select_experts_k,
    local_reduce,
    cluster_axis,
    num_links,
    input_dtype,
    mem_config,
    num_iters,
    topology,
    *,
    device,  # unused
) -> list:
    logger.info("STARTING SWEEP")

    logger.info(vars())

    batch, _, seq, hidden = tuple(input_shape)

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None
        logger.info("device set up")

        try:
            run_all_to_all_combine_test(
                mesh_device=device,
                mesh_shape=mesh_shape,
                axis=cluster_axis,
                batch=batch,
                seq=seq,
                local_reduce=local_reduce,
                experts=experts,
                select_experts_k=select_experts_k,
                hidden_size=hidden,
                num_iters=num_iters,
                num_links=num_links,
                dtype=input_dtype,
                topology=topology,
                input_memory_config=mem_config,
                output_memory_config=mem_config,
            )
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")

        except AssertionError as e:
            return False, e

        return True, None
