# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0


from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    MinimalMatmulStridedReduceScatterTestConfig,
    run_minimal_matmul_strided_reduce_scatter_impl,
)


def _make_fabric_router_config(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    return config


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=128,
                K=256,
                N=512,
                dim=3,
                mm_block_m=64,
                mm_block_k=64,
                mm_block_n=64,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=1,
            ),
            id="small_Nwt2_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=128,
                K=256,
                N=1024,
                dim=3,
                mm_block_m=64,
                mm_block_k=64,
                mm_block_n=64,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=1,
            ),
            id="medium_Nwt4_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=512,
                K=512,
                N=2048,
                dim=3,
                mm_block_m=128,
                mm_block_k=128,
                mm_block_n=128,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=2,
            ),
            id="large_Nwt8_cwimb2",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=512,
                K=256,
                N=2560,
                dim=3,
                mm_block_m=64,
                mm_block_k=64,
                mm_block_n=64,
                mm_core_grid=ttnn.CoreCoord(8, 2),
                chunk_width_in_mm_blocks=4,
            ),
            id="large_Nwt10_cwimb4",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4096,
                K=512,
                N=2048,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 4),
                chunk_width_in_mm_blocks=1,
            ),
            id="xlarge_4k_Nwt8_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4096,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 4),
                chunk_width_in_mm_blocks=2,
            ),
            id="xlarge_4k_Nwt16_cwimb2",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=3072,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 6),
                chunk_width_in_mm_blocks=2,
            ),
            id="xlarge_4k_y6_Nwt16_cwimb2",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=3072,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 6),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=6,
            ),
            id="xlarge_4k_y6_Nwt16_cwimb2_rs6",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=3584,
                K=512,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 7),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_3584_y7_Nwt16_cwimb2_rs3_fullgrid",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=9472,
                K=3456,
                N=5120,
                dim=3,
                mm_block_m=256,
                mm_block_k=128,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(8, 7),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_9472_3456_5120_y7_cwimb1_rs3_fullgrid",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=9472,
                K=3456,
                N=5120,
                dim=3,
                mm_block_m=256,
                mm_block_k=128,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(7, 7),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="xlarge_9472_3456_5120_x7_y7_cwimb1_rs3_fullgrid",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=512,
                K=256,
                N=1536,
                dim=3,
                mm_block_m=128,
                mm_block_k=64,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(6, 2),
                chunk_width_in_mm_blocks=1,
            ),
            id="non_div_Wt_6x2_cwimb1",
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=3072,
                K=512,
                N=5120,
                dim=3,
                mm_block_m=256,
                mm_block_k=128,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(5, 6),
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=4,
            ),
            id="non_div_Wt_large_5x6_cwimb2_rs4",
        ),
        # Blackhole-only: BH compute grid is 12x10 (vs wormhole's 8x8).
        # x12_y8: 96 MM cores, RS cores at row 8 (within BH's 10-row grid).
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=9472,
                K=3456,
                N=5120,
                dim=3,
                mm_block_m=256,
                mm_block_k=128,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(12, 8),
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=5,
            ),
            id="bh_xlarge_9472_3456_5120_x12_y8_cwimb1_rs5",
        ),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "rs_mode",
    [
        "fused",
    ],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_minimal_matmul_strided_reduce_scatter_async(
    mesh_device,
    test_config,
    num_links,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    enable_trace,
    topology,
    num_iters,
    rs_mode,
    cluster_axis,
):
    cfg = test_config

    if is_blackhole() and cluster_axis == 0:
        pytest.skip("cluster_axis=0 not tested on blackhole")
    if is_wormhole_b0() and (cfg.mm_core_grid.x > 8 or cfg.mm_core_grid.y > 8):
        pytest.skip("core grid exceeds wormhole_b0 compute grid (8x8), blackhole-only config (BH grid is 12x10)")
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device in this mesh, reduce-scatter ring size must be > 1")

    TILE_SIZE = 32
    Nt = cfg.N // TILE_SIZE
    Nt_per_core = Nt // cfg.mm_core_grid.x
    assert Nt_per_core >= (
        cfg.mm_block_n // TILE_SIZE
    ), f"block_n size is {cfg.mm_block_n // TILE_SIZE} tiles, but only {Nt_per_core} tiles of work per core"

    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        cfg.M,
        cfg.K,
        cfg.N,
        cfg.dim,
        num_links,
        cfg.input_dtype,
        cfg.layout,
        mem_config_input,
        mem_config_mm,
        mem_config_rs,
        topology=topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=cfg.num_workers_per_link,
        mm_block_m=cfg.mm_block_m,
        mm_block_k=cfg.mm_block_k,
        mm_block_n=cfg.mm_block_n,
        subblock_h=cfg.subblock_h,
        subblock_w=cfg.subblock_w,
        mm_core_grid=cfg.mm_core_grid,
        chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
        rs_mode=rs_mode,
        cluster_axis=cluster_axis,
    )


# Blackhole-specific test: large-packet fabric router config (8 KiB payloads).
# Grid 12x8 with subblocks h=32,w=16 and 2 links, 5 workers/link.
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
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
    ids=["fabric_ring_8kib_payload"],
)
def test_minimal_matmul_strided_reduce_scatter_async_bh_large_packet(
    mesh_device,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    topology,
    cluster_axis,
):
    if is_wormhole_b0():
        pytest.skip("Blackhole-only config: compute grid 12x8 exceeds wormhole_b0 limit (8x8)")
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device in this mesh, reduce-scatter ring size must be > 1")

    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=9472,
        K=3456,
        N=5120,
        dim=3,
        num_links=2,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config_input,
        mem_config_mm=mem_config_mm,
        mem_config_rs=mem_config_rs,
        topology=topology,
        enable_trace=False,
        num_iters=1,
        num_workers_per_link=5,
        num_buffers_per_channel=None,
        mm_block_m=256,
        mm_block_k=128,
        mm_block_n=256,
        subblock_h=1,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(12, 8),
        chunk_width_in_mm_blocks=1,
        rs_core_grid_offset=ttnn.CoreCoord(0, 8),
        rs_mode="fused",
        cluster_axis=cluster_axis,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_acc=True,
    )
