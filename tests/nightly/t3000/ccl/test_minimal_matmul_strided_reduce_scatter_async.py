# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.minimal_matmul_strided_reduce_scatter import (
    MinimalMatmulStridedReduceScatterTestConfig,
    create_global_semaphores,
    run_minimal_matmul_strided_reduce_scatter_impl,
)

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole

from tracy import signpost


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
        ),
        # Same shape as above but with explicit RS worker counts to explore
        # the MM/RS core budget tradeoff in the fused case.
        # RS cores = 2 + 2*num_workers_per_link (mux + workers, 1 link).
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
                num_workers_per_link=6,  # 14 RS cores, 62 total
            ),
            id="xlarge_4k_y6_Nwt16_cwimb2_rs6",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                # Non-divisible slice_Wt: mm_core_grid.x=6 (not a power of 2) breaks the
                # usual alignment between slice_Wt and mm_N_full_block_wt.
                # N_tiles=48, mm_N_full_block_wt=48/6=8, slice_Wt=48/8=6. 6 % 8 ≠ 0.
                # chip k: skip_cols_left = (k*6) % 8 = {0,6,4,2,0,6,4,2} — 4 distinct values.
                # chunk_width_in_tiles=8 > effective_advance=2 → cross-column 2-tile packets
                # in the fused reader/writer. Tests all three rs_mode paths.
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
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                # Non-divisible slice_Wt with multi-worker cross-column packets.
                # mm_core_grid.x=5, N_tiles=160, mm_N_full_block_wt=32, slice_Wt=20. 20%32≠0.
                # chip k: skip = (k*20)%32 = {0,20,8,28,16,4,24,12} — 8 distinct skip values.
                # chunk_width_in_mm_blocks=2 → chunk_width_in_tiles=16; num_workers=4 →
                # effective_advance=8. 8 % 16 ≠ 0 → consecutive packet slots visit different
                # columns; with skip=20 a packet (col=0,col=8) spans (invalid,invalid) but
                # (col=4,col=12) spans (invalid,valid) — exercises the CB-packing fix.
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
            marks=pytest.mark.skip(reason="run manually"),
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
        # "separate",
        # "separate_strided",
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
@skip_for_blackhole("t3000 tests are wormhole_b0 only")
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


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("broadcast_gate", [True, False], ids=["broadcast_gate", "full_gate"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@skip_for_blackhole("t3000 tests are wormhole_b0 only")
def test_minimal_matmul_strided_reduce_scatter_addcmul(mesh_device, broadcast_gate, topology):
    """Test fused addcmul in minimal_matmul_strided_reduce_scatter with broadcast and full gate."""
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=128,
        K=256,
        N=512,
        dim=3,
        num_links=1,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config,
        mem_config_mm=mem_config,
        mem_config_rs=mem_config,
        topology=topology,
        mm_block_m=64,
        mm_block_k=64,
        mm_block_n=64,
        subblock_h=1,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(8, 2),
        chunk_width_in_mm_blocks=1,
        rs_mode="fused",
        cluster_axis=1,
        addcmul_scalar=1.0,
        broadcast_gate=broadcast_gate,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@skip_for_blackhole("t3000 tests are wormhole_b0 only")
def test_minimal_matmul_strided_reduce_scatter_aliased_addcmul_counter_cache_hit(mesh_device, topology):
    """Cache hit with aliased full-shape addcmul inputs and a replacement counter allocation."""
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=512,
        K=256,
        N=512,
        dim=3,
        num_links=1,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config,
        mem_config_mm=mem_config,
        mem_config_rs=mem_config,
        topology=topology,
        mm_block_m=64,
        mm_block_k=64,
        mm_block_n=64,
        subblock_h=1,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(8, 2),
        chunk_width_in_mm_blocks=1,
        rs_mode="fused",
        cluster_axis=1,
        num_iters=2,
        addcmul_scalar=1.0,
        broadcast_gate=False,
        alias_addcmul=True,
        mm_window_blocks=2,
        replace_counters_on_hit=True,
        check_program_cache=True,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@skip_for_blackhole("t3000 tests are wormhole_b0 only")
def test_minimal_matmul_strided_reduce_scatter_fused_concat_non_aligned(mesh_device, topology):
    """MMRS fused concat with non-tile-aligned Ka and Kb (t3000/WH).

    K=32 per device (logical), Ka=11 (non-aligned prefix), Kb=21 (non-aligned suffix).
    Weight is per-segment tile-padded: prefix rows [0..11) real, zeros to tile 32,
    suffix rows [32..53) real, zeros to tile 64.  K_padded=64 (2 tiles), mm_block_k=64.
    """
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=128,
        K=32,  # logical per-device K: Ka=11 + Kb=21
        N=512,
        dim=3,
        num_links=1,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config,
        mem_config_mm=mem_config,
        mem_config_rs=mem_config,
        topology=topology,
        mm_block_m=128,
        mm_block_k=64,  # K_padded=64=2t, 2 divides 2
        mm_block_n=64,
        subblock_h=1,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(8, 2),
        chunk_width_in_mm_blocks=1,
        rs_mode="fused",
        cluster_axis=1,
        fused_concat=True,
        fused_concat_ka=11,  # Ka=11: non-tile-aligned prefix (padded to 32)
    )


# Optional-feature tests (addcmul, fused_activation, bias) live in
# models/tt_dit/tests/models/wan2_2/test_strided_reduce_scatter_wan_t3000.py
