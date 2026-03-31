# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    run_minimal_matmul_strided_reduce_scatter_impl,
)
from tests.nightly.t3000.ccl.test_strided_reduce_scatter_async import run_reduce_scatter_impl


def _make_fabric_router_config(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Strided reduce-scatter: TG (4×8)
# Representative: same 128×16 tile / 8-core / 6-worker config on a multi-chip mesh
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": _make_fabric_router_config(8192),
                "trace_region_size": 1531456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_ring_8kib_payload"],
)
def test_strided_reduce_scatter_async_tg(mesh_device, num_links, cluster_axis, topology):
    num_devices = mesh_device.shape[cluster_axis]
    if num_devices == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device, reduce-scatter ring size must be > 1")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, num_devices))

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_reduce_scatter_impl(
        submesh,
        num_devices,
        [4, 1, 4096, 4096],
        3,
        num_links,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mem_config,
        mem_config,
        rs_topology=topology,
        enable_trace=False,
        num_iters=1,
        small_random_ints=True,
        use_barrier=True,
        use_persistent_buffers=True,
        use_strided=True,
        verify_output_shape=True,
        verify_output_pcc=True,
        mm_cores_y=8,
        mm_block_ht=4,
        mm_block_wt=4,
        mm_N_full_block_wt=8,
        chunk_width_in_mm_blocks=1,
        num_workers_per_link=6,
        cluster_axis=cluster_axis,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Minimal matmul + strided reduce-scatter: TG (4×8), Blackhole grid
# Representative: M=9472, N=5120, 12×8 BH core grid, subblock_h=2 subblock_w=1
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": _make_fabric_router_config(8192),
                "trace_region_size": 1531456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_ring_8kib_payload"],
)
def test_minimal_matmul_strided_reduce_scatter_async_tg_bh(mesh_device, num_links, cluster_axis, topology):
    if is_wormhole_b0():
        pytest.skip("Blackhole-only config: compute grid 12x8 exceeds wormhole_b0 limit (8x8)")
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device, reduce-scatter ring size must be > 1")

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=9472,
        K=3456,
        N=5120,
        dim=3,
        num_links=num_links,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config,
        mem_config_mm=mem_config,
        mem_config_rs=mem_config,
        topology=topology,
        mm_block_m=256,
        mm_block_k=128,
        mm_block_n=256,
        subblock_h=2,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(12, 8),
        chunk_width_in_mm_blocks=1,
        num_workers_per_link=5,
        rs_core_grid_offset=ttnn.CoreCoord(0, 8),
        rs_mode="fused",
        cluster_axis=cluster_axis,
    )
