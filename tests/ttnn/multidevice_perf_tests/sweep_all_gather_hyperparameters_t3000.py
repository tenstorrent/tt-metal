# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case
from models.utility_functions import skip_for_blackhole

from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tracy import signpost

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl


def get_max_chunks_per_sync(num_devices, ag_output_shape, num_links):
    packet_elems = 2048
    total_elems = math.prod(ag_output_shape)
    return (total_elems // packet_elems) // (num_devices * num_links)


CONFIGS = [
    (8, [1, 1, 11264, 12288], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    (8, [1, 1, 12288, 4096], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    (8, [1, 1, 12288, 3072], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    (8, [1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    (8, [1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    (8, [1, 1, 8192, 16384], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    (8, [1, 1, 8192, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
]
CONFIGS_IDS = [f"ag_output_shape{i}" for i in range(len(CONFIGS))]

CHUNKS_PER_SYNC = ["MAX", 160]  # ["MAX", 320, 160, 80, 40, 20, 10]
CHUNKS_PER_SYNC_IDS = [f"{chunk}-chunks" for chunk in CHUNKS_PER_SYNC]

WORKERS_PER_LINK = [1, 4]  # [8, 4, 2, 1]
WORKERS_PER_LINK_IDS = [f"{worker}-workers" for worker in WORKERS_PER_LINK]

TOPOLOGY = ["ring", "linear"]


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    CONFIGS,
    ids=CONFIGS_IDS,
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 20),
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=TOPOLOGY,
)
@pytest.mark.parametrize(
    "chunks_per_sync",
    CHUNKS_PER_SYNC,
    ids=CHUNKS_PER_SYNC_IDS,
)
@pytest.mark.parametrize(
    "num_workers_per_link",
    WORKERS_PER_LINK,
    ids=WORKERS_PER_LINK_IDS,
)
def test_all_gather_chunks_per_sync(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
):
    if chunks_per_sync == "MAX":
        chunks_per_sync = get_max_chunks_per_sync(num_devices, ag_output_shape, num_links)

    logger.info(f"Running with chunks_per_sync: {chunks_per_sync}")

    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_barrier=True,
        use_persistent_buffers=False,
        chunks_per_sync=chunks_per_sync,
        skip_check=True,
        num_workers_per_link=num_workers_per_link,
    )
