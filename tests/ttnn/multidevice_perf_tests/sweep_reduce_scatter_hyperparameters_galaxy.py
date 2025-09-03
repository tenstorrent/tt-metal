# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl


# (num_devices, num_links, rs_input_shape, dim)
CONFIGS = [
    (4, 1, [1, 1, 22528, 3072], 3),
    (2, 1, [1, 1, 11264, 3072], 3),
    # (2,4,[1, 1, 5632, 3072],3),
    # (8,4,[1, 1, 11264, 3072],3),
    # (4,4,[1, 1, 5632, 3072],3),
    # (2,1,[1, 1, 128, 1536],3),
    # (4,1,[1, 1, 128, 1536],3),
    # (2,4,[1, 1, 128, 1536],3),
    # (4,4,[1, 1, 128, 1536],3),
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
        (True, 15),
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
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
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
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
    )
