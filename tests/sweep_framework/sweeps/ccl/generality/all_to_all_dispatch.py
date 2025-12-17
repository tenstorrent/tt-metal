# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import ttnn

from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import device_context, mesh_shape_iterator
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import run_all_to_all_dispatch_test


# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

NUM_DEVICES = ttnn.get_num_devices()


FABRIC_CONFIGS_1D = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
]

FABRIC_CONFIGS_2D = [
    ttnn.FabricConfig.FABRIC_2D,
]

FABRIC_CONFIGS = FABRIC_CONFIGS_1D + FABRIC_CONFIGS_2D


def _pd(val: int):
    return val * NUM_DEVICES


GENERALITY_PARAMETERS = {
    "mesh_shape": list(mesh_shape_iterator(NUM_DEVICES)),
    "fabric_config": FABRIC_CONFIGS,
    "input_shape": [
        [_pd(1), 1, 8, 32],
        [_pd(1), 1, 32, 2880],  # GPT-OSS
        [_pd(1), 1, 2, 31],
        [_pd(8), 1, 2, 7168],
        [_pd(16), 1, 2, 7168],
        [_pd(1), 1, 2, 16384],
    ],
    "experts": [_pd(i) for i in [2, 4, 8]],
    "select_experts_k": [2, 4, 8],
    "cluster_axis": [0, 1, None],
    "num_links": [1],
    "input_dtype": [ttnn.bfloat16],
    "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
    "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
    "num_iters": [1],
}

parameters = {
    "generality_suite": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS},
    "generality_suite_fabric_1d": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS_1D},
    "generality_suite_fabric_2d": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS_2D},
    "lead_model_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "input_shape": [[_pd(1), 1, 32, 2880], [_pd(4), 1, 1, 7168]],  # GPT-OSS  # deepseek cluster_axis=0
        "experts": [_pd(i) for i in [2, 4, 8]],
        "select_experts_k": [2, 4, 8],
        "cluster_axis": [0, 1],
        "num_links": [1],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # hardcode for 6U
    if test_vector["mesh_shape"] in [(16, 2), (2, 16)]:
        return True, "Invalid mesh shape for 6U"

    mesh_shape, cluster_axis = test_vector["mesh_shape"], test_vector["cluster_axis"]
    if cluster_axis and mesh_shape[cluster_axis] == 1:
        return True, "Unit cluster axis"

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


DEVICE_PARAMS = {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}


def run(
    mesh_shape,
    fabric_config,
    input_shape,
    experts,
    select_experts_k,
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

    with device_context(mesh_shape, fabric_config, device_params=DEVICE_PARAMS) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None
        logger.info("device set up")

        try:
            run_all_to_all_dispatch_test(
                mesh_device=device,
                mesh_shape=mesh_shape,
                batch=batch,
                experts=experts,
                select_experts_k=select_experts_k,
                hidden_size=hidden,
                seq_len=seq,
                num_iters=num_iters,
                warmup_iters=0,
                trace_mode=False,
                num_links=num_links,
                dtype=input_dtype,
                topology=None,
                input_memory_config=mem_config,
                output_memory_config=mem_config,
                cluster_axis=cluster_axis,
            )
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")

        except AssertionError as e:
            return False, e

        return True, None
