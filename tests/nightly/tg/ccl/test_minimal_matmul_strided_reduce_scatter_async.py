# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import ttnn
from loguru import logger
from models.common.utility_functions import is_wormhole_b0

from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    MinimalMatmulStridedReduceScatterTestConfig,
    run_minimal_matmul_strided_reduce_scatter_impl,
)


def _make_fabric_router_config(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    return config


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])
@pytest.mark.parametrize(
    "test_config",
    [
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
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
            marks=pytest.mark.skip(reason="run manually"),
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
        # LTX video FFN ff2 (RowParallel reduce-scatter): per-device [4864,4096]@[4096,4096]
        # (K = ffn_dim/TP = 16384/4 = 4096, N = dim = 4096), HiFi2 (impl default).
        # LTX has no fused_mmrs_configs entry for this shape (falls to the (8,7) default), which is a
        # too-small grid. Start from the Wan-style (12,8)=96-core MM grid with a plain 8/8/8 blocking
        # (subblock 2x2), leaving rows 8-9 for the RS workers; tune from here.
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4864,
                K=4096,
                N=4096,
                dim=3,
                mm_block_m=256,
                mm_block_k=256,
                mm_block_n=256,
                mm_core_grid=ttnn.CoreCoord(12, 8),
                chunk_width_in_mm_blocks=1,
                subblock_h=2,
                subblock_w=2,
                num_workers_per_link=3,
            ),
            id="ltx_ff2_4864_4096_4096_x12_y8_b888",
        ),
        # LTX ff2, tuned: on the (12,8) grid (rows 8-9 reserved for RS workers), the default block is
        # M/K/N = 4/5/7 tiles (subblock 4x1); run_ff2_block_sweep.py sweeps shapes to revisit.
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4864,
                K=4096,
                N=4096,
                dim=3,
                mm_block_m=128,  # 4 tiles
                mm_block_k=160,  # 5 tiles
                mm_block_n=224,  # 7 tiles
                mm_core_grid=ttnn.CoreCoord(12, 8),
                chunk_width_in_mm_blocks=1,
                subblock_h=4,
                subblock_w=1,
                num_workers_per_link=3,
            ),
            id="ltx_ff2_4864_4096_4096_x12_y8_b457",
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
        # Run the op twice back-to-back (no trace): iter 0 pays program build/first-dispatch, iter 1
        # is a program-cache hit -> use the 2nd op in the profiler to rule out compile/first-run cost.
        (False, 2),
    ],
    ids=["check", "iter2"],
)
@pytest.mark.parametrize(
    "rs_mode",
    [
        "fused",
        # Unfused: run the standalone minimal_matmul (its own profiler op = the standalone matmul
        # time) followed by a separate reduce-scatter. Note the standalone matmul uses the FULL
        # compute grid, not the (12,8) the fused op reserves for RS workers.
        "separate_strided",
        "separate",
    ],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": _make_fabric_router_config(8192),
                "trace_region_size": 1531456,
            },
            ttnn.Topology.Ring,
            marks=pytest.mark.skipif(is_wormhole_b0(), reason="fabric_router_config=8192 not supported on wormhole_b0"),
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_ring_8kib_payload"],
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


@pytest.mark.skip(reason="Sweep test - skipped from nightly")
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "subblock_h, subblock_w",
    [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (4, 1),
        (1, 4),
    ],
    ids=["sh1_sw1", "sh2_sw1", "sh1_sw2", "sh2_sw2", "sh4_sw1", "sh1_sw4"],
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
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": _make_fabric_router_config(8192),
                "trace_region_size": 1531456,
            },
            ttnn.Topology.Ring,
            marks=pytest.mark.skipif(is_wormhole_b0(), reason="fabric_router_config=8192 not supported on wormhole_b0"),
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_ring_8kib_payload"],
)
def test_minimal_matmul_strided_reduce_scatter_async_bh_large_packet(
    mesh_device,
    subblock_h,
    subblock_w,
    mem_config_input,
    mem_config_mm,
    mem_config_rs,
    topology,
):
    cluster_axis = 0
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
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=ttnn.CoreCoord(12, 8),
        chunk_width_in_mm_blocks=1,
        rs_core_grid_offset=ttnn.CoreCoord(0, 8),
        rs_mode="fused",
        cluster_axis=cluster_axis,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_acc=True,
    )


# ---------------------------------------------------------------------------
# Block-shape sweep: fixed LTX video ff2 shape (4864/4096/4096) on the (12,8) MM grid, HiFi2.
# The mm_block_m / mm_block_k / mm_block_n ranges come from env vars so a driver can shard the
# (large) sweep into small, profiler-friendly batches:
#   SWEEP_M_BLOCKS / SWEEP_K_BLOCKS / SWEEP_N_BLOCKS  (each "lo:hi" inclusive, or "a,b,c"; tiles)
# If none are set the parameter set is empty -> the test skips (empty_parameter_set_mark=skip),
# so it never runs in nightly. N block is capped by the per-core N tiles (Nt // grid.x = 10 here);
# subblock is auto-picked as the largest DST-fitting (sh*sw <= 4) divisor of the block.
# Drive it with tests/nightly/tg/ccl/run_ff2_block_sweep.py.
# ---------------------------------------------------------------------------
_SWEEP_M, _SWEEP_K, _SWEEP_N = 4864, 4096, 4096
_SWEEP_GRID = (12, 8)
_DEFAULT_BLOCK_RANGE = tuple(range(2, 17))  # 2..16 tiles (used per-axis when its env var is unset)
_DST_MAX_TILES = 4  # matches the subblocks the other configs in this file use (sh*sw <= 4)
# Evaluated at collection so the sweep skips (without opening a device) unless a driver set the env.
_HAS_SWEEP_ENV = any(os.environ.get(v) for v in ("SWEEP_M_BLOCKS", "SWEEP_K_BLOCKS", "SWEEP_N_BLOCKS"))


def _parse_block_env(name):
    """Parse a block-size range env var: 'lo:hi' (inclusive) or 'a,b,c'. Returns None if unset."""
    import os

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    if ":" in raw:
        lo, hi = (int(x) for x in raw.split(":"))
        return tuple(range(lo, hi + 1))
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _largest_valid_subblock(block_m_t, block_n_t, dst_max_tiles=_DST_MAX_TILES):
    """Largest (subblock_h, subblock_w) that divides the block and fits the DST register;
    ties broken toward a balanced (square-ish) subblock."""
    best = (1, 1)
    for sh in range(1, block_m_t + 1):
        if block_m_t % sh:
            continue
        for sw in range(1, block_n_t + 1):
            if block_n_t % sw or sh * sw > dst_max_tiles:
                continue
            cur, bp = sh * sw, best[0] * best[1]
            if cur > bp or (cur == bp and abs(sh - sw) < abs(best[0] - best[1])):
                best = (sh, sw)
    return best


def _gen_block_sweep_configs():
    import math

    m_blocks = _parse_block_env("SWEEP_M_BLOCKS")
    k_blocks = _parse_block_env("SWEEP_K_BLOCKS")
    n_blocks = _parse_block_env("SWEEP_N_BLOCKS")
    if m_blocks is None and k_blocks is None and n_blocks is None:
        return []  # no sweep requested -> empty parameter set skips cleanly (keeps nightly clean)
    m_blocks = m_blocks or _DEFAULT_BLOCK_RANGE
    k_blocks = k_blocks or _DEFAULT_BLOCK_RANGE
    n_blocks = n_blocks or _DEFAULT_BLOCK_RANGE

    TILE = 32
    gx, gy = _SWEEP_GRID
    mt_per_core = math.ceil((_SWEEP_M // TILE) / gy)  # fused RS-matmul: M on grid.y (no transpose)
    kt = _SWEEP_K // TILE
    nt_per_core = (_SWEEP_N // TILE) // gx  # matches the Nt_per_core assert in the impl/test
    configs = []
    for bm in m_blocks:
        if bm > mt_per_core:
            continue
        for bk in k_blocks:
            if bk > kt:
                continue
            for bn in n_blocks:
                if bn > nt_per_core:
                    continue
                sh, sw = _largest_valid_subblock(bm, bn)
                configs.append(
                    MinimalMatmulStridedReduceScatterTestConfig(
                        M=_SWEEP_M,
                        K=_SWEEP_K,
                        N=_SWEEP_N,
                        dim=3,
                        mm_block_m=bm * TILE,
                        mm_block_k=bk * TILE,
                        mm_block_n=bn * TILE,
                        mm_core_grid=ttnn.CoreCoord(gx, gy),
                        chunk_width_in_mm_blocks=1,
                        subblock_h=sh,
                        subblock_w=sw,
                        num_workers_per_link=3,
                    )
                )
    return configs


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0], ids=["axis_0"])
@pytest.mark.skipif(
    not _HAS_SWEEP_ENV, reason="block sweep: set SWEEP_{M,K,N}_BLOCKS env (drive with run_ff2_block_sweep.py)"
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
    ids=["DRAM"],
)
@pytest.mark.parametrize("enable_trace, num_iters", [(False, 1)], ids=["check"])
@pytest.mark.parametrize("rs_mode", ["fused"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": _make_fabric_router_config(8192),
                "trace_region_size": 1531456,
            },
            ttnn.Topology.Ring,
            marks=pytest.mark.skipif(is_wormhole_b0(), reason="fabric_router_config=8192 not supported on wormhole_b0"),
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_ring_8kib_payload"],
)
def test_minimal_matmul_strided_reduce_scatter_block_sweep(
    mesh_device,
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
    if is_wormhole_b0() and (_SWEEP_GRID[0] > 8 or _SWEEP_GRID[1] > 8):
        pytest.skip("core grid exceeds wormhole_b0 compute grid (8x8), blackhole-only sweep")
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device in this mesh")

    configs = _gen_block_sweep_configs()
    if not configs:
        pytest.skip("no SWEEP_{M,K,N}_BLOCKS env set; drive with run_ff2_block_sweep.py")

    # Run every config back-to-back on the SAME device (opened once by the fixture) so the sweep is
    # not dominated by per-config device open/close. Each op records its block config in the
    # profiler ATTRIBUTES, so the driver still maps timings back to configs.
    for idx, cfg in enumerate(configs):
        logger.info(
            f"[block sweep {idx + 1}/{len(configs)}] M_block={cfg.mm_block_m // 32} "
            f"K_block={cfg.mm_block_k // 32} N_block={cfg.mm_block_n // 32} "
            f"subblock=({cfg.subblock_h},{cfg.subblock_w})"
        )
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
            check_correctness=False,  # perf sweep: skip golden matmul + PCC
        )
