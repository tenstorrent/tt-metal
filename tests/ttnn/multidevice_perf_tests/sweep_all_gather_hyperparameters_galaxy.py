# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from tests.ttnn.multidevice_perf_tests.sweep_all_gather_hyperparameters_t3000 import get_max_chunks_per_sync
from models.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


CONFIGS = [
    ([1, 1, 11264, 12288], 0, 3),
    ([1, 1, 22528, 3072], 1, 3),
    ([1, 1, 12288, 4096], 0, 2),
    ([1, 1, 3072, 16384], 1, 2),
    ([1, 1, 12288, 3072], 0, 2),
    ([1, 1, 3072, 12288], 1, 2),
    ([1, 1, 11264, 3072], 1, 3),
    ([1, 1, 12288, 2304], 0, 2),
    ([1, 1, 3072, 9216], 1, 2),
    ([1, 1, 3072, 8192], 1, 2),
    ([1, 1, 2048, 12288], 0, 3),
    ([1, 1, 8192, 3072], 1, 3),
    ([1, 1, 3072, 8192], 1, 2),
    ([1, 1, 3072, 6144], 1, 2),
    ([1, 1, 12288, 1536], 0, 2),
    ([1, 1, 3072, 6144], 1, 2),
    ([1, 1, 5632, 3072], 0, 3),
    ([1, 1, 5632, 3072], 0, 3),
    ([1, 1, 3072, 4608], 1, 2),
    ([1, 1, 1536, 9216], 1, 2),
    ([1, 1, 6144, 2304], 0, 2),
    ([1, 1, 4096, 3072], 1, 3),
    ([1, 1, 1536, 8192], 1, 2),
    ([1, 1, 6144, 2048], 0, 2),
    ([1, 1, 4096, 3072], 1, 3),
    ([1, 1, 3072, 3072], 1, 2),
    ([1, 1, 12288, 768], 0, 2),
    ([1, 1, 3072, 3072], 1, 2),
    ([1, 1, 1536, 4608], 1, 2),
    ([1, 1, 1536, 4608], 1, 2),
    ([1, 1, 3072, 2304], 1, 2),
    ([1, 1, 1536, 4096], 1, 2),
    ([1, 1, 1024, 6144], 0, 3),
    ([1, 1, 4096, 1536], 1, 3),
    ([1, 1, 1536, 4096], 1, 2),
    ([1, 1, 3072, 1536], 1, 2),
    ([1, 1, 12288, 384], 0, 2),
    ([1, 1, 3072, 1536], 1, 2),
    ([1, 1, 3072, 1152], 0, 2),
    ([1, 1, 1536, 2304], 1, 2),
    ([1, 1, 2048, 1536], 1, 3),
    ([1, 1, 1536, 2048], 1, 2),
    ([1, 1, 2048, 1536], 1, 3),
    ([1, 1, 3072, 768], 1, 2),
    ([1, 1, 3072, 768], 1, 2),
    ([1, 1, 1536, 1152], 0, 2),
    ([1, 1, 128, 12288], 0, 3),
    ([1, 1, 32, 49152], 0, 3),
    ([1, 1, 1536, 1024], 0, 2),
    ([1, 1, 1024, 1536], 1, 3),
    ([1, 1, 3072, 384], 0, 2),
    ([1, 1, 3072, 384], 1, 2),
    ([1, 1, 512, 1536], 0, 3),
    ([1, 1, 3072, 192], 0, 2),
    ([1, 1, 128, 3072], 1, 3),
    ([1, 1, 32, 12288], 1, 3),
    ([1, 1, 128, 1536], 1, 3),
    ([1, 1, 32, 6144], 1, 3),
    ([1, 1, 128, 1536], 1, 3),
    ([1, 1, 32, 6144], 1, 3),
]

CONFIGS_IDS = [f"ag_output_shape{i}_" for i in range(len(CONFIGS))]
WORKERS_PER_LINK = [None, 1]  # [4, 2, 1]
WORKERS_PER_LINK_IDS = [f"{worker}-workers" for worker in WORKERS_PER_LINK]
CHUNKS_PER_SYNC = [None, "MAX"]  # ["MAX", 320, 160, 80, 40, 20, 10]
CHUNKS_PER_SYNC_IDS = [f"{chunk}-chunks" for chunk in CHUNKS_PER_SYNC]
TOPOLOGY = ["ring", "linear"]


@pytest.mark.parametrize(
    "ag_output_shape, cluster_axis, dim",
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
        (True, 30),
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
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("num_links", [4], ids=["4link"])
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
