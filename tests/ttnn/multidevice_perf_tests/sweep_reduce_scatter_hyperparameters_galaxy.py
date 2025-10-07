# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
import math


# (num_devices, num_links, rs_input_shape, dim)
CONFIGS = [
    (4, 1, [1, 1, 22528, 3072], 3),  # important shape
    (2, 1, [1, 1, 11264, 3072], 3),
    (2, 4, [1, 1, 5632, 3072], 3),
    (8, 4, [1, 1, 71680, 3072], 3),
    (8, 4, [1, 1, 63488, 3072], 3),
    (8, 4, [1, 1, 47104, 1536], 3),
    (8, 4, [1, 1, 55296, 3072], 3),
    (8, 4, [1, 1, 47104, 3072], 3),
    (8, 4, [1, 1, 38912, 3072], 3),
    (8, 4, [1, 1, 34816, 3072], 3),
    (4, 4, [1, 1, 30720, 3072], 3),
    (8, 4, [1, 1, 30720, 1536], 3),
    (4, 4, [1, 1, 30720, 1536], 3),
    (4, 4, [1, 1, 18432, 3072], 3),
    (8, 4, [1, 1, 17408, 3072], 3),
    (8, 4, [1, 1, 16384, 3072], 3),
    (8, 4, [1, 1, 15360, 3072], 3),
    (8, 4, [1, 1, 14336, 3072], 3),
    (8, 4, [1, 1, 13312, 3072], 3),
    (8, 4, [1, 1, 12288, 3072], 3),
    (8, 4, [1, 1, 11264, 3072], 3),  # important shape
    (8, 4, [1, 1, 10752, 3072], 3),
    (8, 4, [1, 1, 10240, 3072], 3),
    (8, 4, [1, 1, 9728, 3072], 3),
    (8, 4, [1, 1, 8278, 3072], 3),
    (8, 4, [1, 1, 9216, 3072], 3),
    (8, 4, [1, 1, 8192, 3072], 3),
    (8, 4, [1, 1, 7168, 3072], 3),
    (8, 4, [1, 1, 7040, 3072], 3),
    (8, 4, [1, 1, 6912, 3072], 3),
    (8, 4, [1, 1, 6784, 3072], 3),
    (8, 4, [1, 1, 6656, 3072], 3),
    (8, 4, [1, 1, 6528, 3072], 3),
    (8, 4, [1, 1, 6400, 3072], 3),
    (8, 4, [1, 1, 6272, 3072], 3),
    (8, 4, [1, 1, 6144, 3072], 3),
    (8, 4, [1, 1, 6016, 3072], 3),
    (8, 4, [1, 1, 5888, 3072], 3),
    (8, 4, [1, 1, 5760, 3072], 3),
    (8, 4, [1, 1, 5632, 3072], 3),  # important shape
    (8, 4, [1, 1, 5504, 3072], 3),
    (8, 4, [1, 1, 5376, 3072], 3),
    (8, 4, [1, 1, 5248, 3072], 3),
    (8, 4, [1, 1, 5120, 3072], 3),
    (8, 4, [1, 1, 4992, 3072], 3),
    (8, 4, [1, 1, 4864, 3072], 3),
    (8, 4, [1, 1, 4736, 3072], 3),
    (8, 4, [1, 1, 4608, 3072], 3),
    (8, 4, [1, 1, 4480, 3072], 3),
    (8, 4, [1, 1, 4352, 3072], 3),
    (8, 4, [1, 1, 4224, 3072], 3),
    (8, 4, [1, 1, 4096, 3072], 3),
    (8, 4, [1, 1, 3968, 3072], 3),
    (8, 4, [1, 1, 3840, 3072], 3),
    (8, 4, [1, 1, 3712, 3072], 3),  # important shape
    (8, 4, [1, 1, 3072, 3072], 3),
    (8, 4, [1, 1, 2048, 3072], 3),
    (8, 4, [1, 1, 1024, 3072], 3),
    (4, 4, [1, 1, 5632, 3072], 3),  # important shape
    (2, 1, [1, 1, 128, 1536], 3),
    (4, 1, [1, 1, 128, 1536], 3),
    (2, 4, [1, 1, 128, 1536], 3),
    (4, 4, [1, 1, 128, 1536], 3),  # important shape
    # (8,4,[1, 1, 128, 1536],3),
]

CONFIGS_IDS = [f"rs_input_shape{i}_" for i in range(len(CONFIGS))]

WORKERS_PER_LINK = [None]
WORKERS_PER_LINK_IDS = [f"{worker}-workers" for worker in WORKERS_PER_LINK]

CHUNKS_PER_SYNC = [None]
CHUNKS_PER_SYNC_IDS = [f"{chunk}-chunks" for chunk in CHUNKS_PER_SYNC]

TOPOLOGY = ["ring", "linear"]


@pytest.mark.parametrize(
    "num_devices, num_links, rs_input_shape, dim",
    CONFIGS,
    ids=CONFIGS_IDS,
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
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
        (True, 4),
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 2000000}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 2000000}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=TOPOLOGY,
)
@pytest.mark.parametrize("chunks_per_sync", CHUNKS_PER_SYNC, ids=CHUNKS_PER_SYNC_IDS)
@pytest.mark.parametrize("num_workers_per_link", WORKERS_PER_LINK, ids=WORKERS_PER_LINK_IDS)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("num_outer_iters", [4])
def test_reduce_scatter_chunks_per_sync(
    mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    rs_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_outer_iters,
):
    submesh_device = mesh_device
    cluster_axis = 0
    if num_devices == 4:
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
        cluster_axis = 1
    elif num_devices == 8:
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    elif num_devices == 2:
        if rs_topology == ttnn.Topology.Ring:
            pytest.skip("Ring topology is not supported for 2 devices")
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    else:
        pytest.skip("Unsupported number of devices")

    for _ in range(num_outer_iters):
        run_reduce_scatter_impl(
            submesh_device,
            num_devices,
            rs_input_shape,
            dim,
            num_links,
            rs_input_dtype,
            layout,
            mem_config_input,
            mem_config_rs,
            rs_topology=rs_topology,
            enable_trace=enable_trace,
            num_iters=num_iters,
            cluster_axis=cluster_axis,
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=None,
            verify_output=False,
            use_persistent_buffers=False,
        )
