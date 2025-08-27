# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from tests.ttnn.multidevice_perf_tests.sweep_all_gather_hyperparameters_T3K import get_max_chunks_per_sync
from models.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


CONFIGS = [
    ([1, 1, 11264, 3072], 1, 3, 4, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 3072, 8192], 1, 2, 4, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 11264, 12288], 0, 3, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 22528, 3072], 1, 3, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 12288, 4096], 0, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 3072, 16384], 1, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
]

CONFIGS_IDS = [f"ag_output_shape{i}" for i in range(len(CONFIGS))]
WORKERS_PER_LINK = [4, 2, 1]
WORKERS_PER_LINK_IDS = [f"{worker}-workers" for worker in WORKERS_PER_LINK]
CHUNKS_PER_SYNC = ["MAX", 320, 160, 80, 40, 20, 10]
CHUNKS_PER_SYNC_IDS = [f"{chunk}-chunks" for chunk in CHUNKS_PER_SYNC]
TOPOLOGY = ["ring", "linear"]


@pytest.mark.parametrize(
    "ag_output_shape, cluster_axis, dim, num_links, layout, ag_input_dtype",
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
    "enable_trace, num_iters",
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
@pytest.mark.parametrize("chunks_per_sync", CHUNKS_PER_SYNC, ids=CHUNKS_PER_SYNC_IDS)
@pytest.mark.parametrize("num_workers_per_link", WORKERS_PER_LINK, ids=WORKERS_PER_LINK_IDS)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_chunks_per_sync(
    mesh_device,
    ag_output_shape,
    cluster_axis,
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
    num_devices = 4
    if cluster_axis == 0:
        num_devices = 8
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        num_devices = 4
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))

    if chunks_per_sync == "MAX":
        chunks_per_sync = get_max_chunks_per_sync(num_devices, ag_output_shape, num_links)

    run_all_gather_impl(
        submesh_device,
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
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        skip_check=True,
    )
