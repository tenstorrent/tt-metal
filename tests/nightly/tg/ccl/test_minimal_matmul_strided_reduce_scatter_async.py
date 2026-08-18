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
        # Same 8/8/8 blocking, but handed over through a 2-block window. Unwindowed it does not fit:
        # the 19x11 tile resident shard leaves less L1 than 8/8/8's ~1.05 MB of matmul CBs need, and
        # the op throws at program validation. The window shrinks the shard to 2*8 x 11 tiles, which
        # is what makes the blocking reachable at all.
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
                mm_window_blocks=2,
            ),
            id="ltx_ff2_4864_4096_4096_x12_y8_b888_window2",
        ),
        # LTX ff2, tuned: on the (12,8) grid (rows 8-9 reserved for RS workers)
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
        # Same shape/blocking, but the MM output is handed over through a rolling 3-block L1 window
        # instead of staying fully resident. Mt_per_core=19 with mm_block_ht=4 gives 5 M blocks per
        # core, so a 3-deep window really does recycle slots (blocks 3,4 reuse slots 0,1) and
        # exercises the RS->MM credit path. Only the RS output is PCC-checked; see the impl.
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4864,
                K=4096,
                N=4096,
                dim=3,
                mm_block_m=128,
                mm_block_k=160,
                mm_block_n=224,
                mm_core_grid=ttnn.CoreCoord(12, 8),
                chunk_width_in_mm_blocks=1,
                subblock_h=4,
                subblock_w=1,
                num_workers_per_link=3,
                mm_window_blocks=3,
            ),
            id="ltx_ff2_4864_4096_4096_x12_y8_b457_window3",
        ),
        # Control for the window: W == M_blocks_per_core (5), so no slot is ever recycled and the
        # matmul's credit wait never fires. Isolates the windowed ADDRESSING (row remap on both
        # sides, shortened output tensor) from the recycling handshake — if this passes and window3
        # fails, the addressing is right and the bug is in the credit path.
        pytest.param(
            MinimalMatmulStridedReduceScatterTestConfig(
                M=4864,
                K=4096,
                N=4096,
                dim=3,
                mm_block_m=128,
                mm_block_k=160,
                mm_block_n=224,
                mm_core_grid=ttnn.CoreCoord(12, 8),
                chunk_width_in_mm_blocks=1,
                subblock_h=4,
                subblock_w=1,
                num_workers_per_link=3,
                mm_window_blocks=5,
            ),
            id="ltx_ff2_4864_4096_4096_x12_y8_b457_window5_norecycle",
        ),
        # Larger M (11520) on the same (12,8) grid. Mt_per_core=45 with mm_block_ht=4 gives 12 M
        # blocks per core. Without a window the resident shard is 45x14 tiles = 1,290,240 B, leaving
        # only ~171 KB of the 1,461,504 B L1 bank for circular buffers — less than the RS workers
        # alone need (~213 KB), so EVERY blocking fails the CB-vs-L1 clash check. This unwindowed
        # entry is kept as the demonstration of that; the windowed ones below are the fix.
        *[
            pytest.param(
                MinimalMatmulStridedReduceScatterTestConfig(
                    M=11520,
                    K=3456,
                    N=5120,
                    dim=3,
                    mm_block_m=128,
                    mm_block_k=160,
                    mm_block_n=224,
                    mm_core_grid=ttnn.CoreCoord(12, 8),
                    chunk_width_in_mm_blocks=1,
                    subblock_h=4,
                    subblock_w=1,
                    num_workers_per_link=3,
                    mm_window_blocks=w,
                ),
                id=f"x11520_3456_5120_x12_y8_b457_{'nowindow' if w is None else f'window{w}'}",
            )
            for w in (None, 2, 3, 4)
        ],
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
        # Run the op twice back-to-back (no trace): iter 0 pays program build/first-dispatch
        (False, 2),
    ],
    ids=["check", "iter2"],
)
@pytest.mark.parametrize(
    "rs_mode",
    [
        "fused",
        # Unfused: run the standalone minimal_matmul (its own profiler op = the standalone matmul time) followed
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
        mm_window_blocks=cfg.mm_window_blocks,
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


# --------------------------------------------------------------------------- Block-shape sweep: fixed LTX video ff2
_SWEEP_M, _SWEEP_K, _SWEEP_N = 2656, 3456, 5120
_SWEEP_GRID = (12, 8)
# Hand the MM output to the RS through a rolling L1 window this many M blocks deep (clamped to the
# number of blocks a core actually has). 2 is the shallowest depth that still lets the matmul run a
# block ahead of the readers, and measured perf is flat in this knob, so it is chosen to give the
# circular buffers as much of the L1 bank as possible.
_SWEEP_WINDOW_BLOCKS = 2
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
                # Whether a blocking's circular buffers actually fit in L1 alongside the resident
                # window is left to the device to decide: the sweep loop catches and logs the ones
                # it rejects. Predicting it here needs a model of every CB both programs create,
                # which is easy to get wrong and drifts as those factories change.
                window = min(_SWEEP_WINDOW_BLOCKS, math.ceil(mt_per_core / bm))
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
                        mm_window_blocks=window,
                    )
                )
    return configs


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0], ids=["axis_0"])
# One batch runs ~100 blockings back-to-back, far past pytest.ini's 300s per-test timeout. Without
# this the test is killed mid-batch and the driver silently gets results only for the configs that
# ran before the cutoff — biased toward the low K/N end, since that is the iteration order.
@pytest.mark.timeout(0)
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

    skipped = []
    # Run every config back-to-back on the SAME device (opened once by the fixture) so the sweep is not dominated
    for idx, cfg in enumerate(configs):
        logger.info(
            f"[block sweep {idx + 1}/{len(configs)}] M_block={cfg.mm_block_m // 32} "
            f"K_block={cfg.mm_block_k // 32} N_block={cfg.mm_block_n // 32} "
            f"subblock=({cfg.subblock_h},{cfg.subblock_w})"
        )
        # A config the L1 filter let through can still be rejected on device. Log and carry on
        # rather than aborting: the batch's remaining configs are still worth timing, and bailing
        # here would leave the driver with no profiler CSV at all for this batch.
        try:
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
                mm_window_blocks=cfg.mm_window_blocks,
            )
        except Exception as e:
            skipped.append((cfg.mm_block_m // 32, cfg.mm_block_k // 32, cfg.mm_block_n // 32))
            logger.warning(f"[block sweep] config failed on device, skipping: {str(e).splitlines()[0]}")

    if skipped:
        logger.warning(f"[block sweep] {len(skipped)}/{len(configs)} configs failed on device: {skipped}")


# --- Cosmos3 trunk-shape regression cases -----------------------------------
#
# Bisect ladder from the passing LTX config toward cosmos3's trunk usage on the
# BH-Galaxy 10x10 power-clamped grid. Dimensions stepped: ring size (cluster
# axis), shape/grid/workers, trace capture. Retained as regression guards for
# the (now-fixed) semaphore-wipe replay race and submesh ring-neighbor
# resolution in minimal_ring_strided_reduce_scatter_async_{reader,writer}.cpp.

_COSMOS3_DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
_COSMOS3_DEVICE_PARAMS = [
    ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000000}, ttnn.Topology.Ring)
]


def _run_cosmos3_case(
    mesh_device,
    topology,
    M,
    K,
    N,
    mm_core_grid,
    num_workers_per_link,
    cluster_axis,
    enable_trace,
    # This suite reproduces credit/replay protocol bugs, not blocking perf: small M
    # blocks + a 2-deep window keep the L1-resident MM shard far from the CB region
    # on every case (the op keeps its output L1-resident; at M=22144 a full shard
    # is ~2x the bank).
    mm_block_m=128,
    sub_h=2,
    sub_w=1,
    ops_per_trace=1,
    mm_window_blocks=2,
):
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=M,
        K=K,
        N=N,
        dim=3,
        num_links=2,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=_COSMOS3_DRAM,
        mem_config_mm=_COSMOS3_DRAM,
        mem_config_rs=_COSMOS3_DRAM,
        topology=topology,
        enable_trace=enable_trace,
        # Production replays the captured graph 35x (one per denoise step). The
        # (now-fixed) semaphore wipe race hung at replay #4, so shallow replay
        # counts miss the regression this suite guards.
        num_iters=35 if enable_trace else 1,
        num_workers_per_link=num_workers_per_link,
        ops_per_trace=ops_per_trace,
        mm_window_blocks=mm_window_blocks,
        num_buffers_per_channel=None,
        mm_block_m=mm_block_m,
        mm_block_k=128,
        mm_block_n=256,
        subblock_h=sub_h,
        subblock_w=sub_w,
        mm_core_grid=mm_core_grid,
        chunk_width_in_mm_blocks=1,
        rs_core_grid_offset=ttnn.CoreCoord(0, 8),
        rs_mode="fused",
        cluster_axis=cluster_axis,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_acc=True,
        allowed_pcc=0.999,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology", _COSMOS3_DEVICE_PARAMS, indirect=["device_params"], ids=["fabric_ring"]
)
@pytest.mark.parametrize(
    "M, K, N, mm_core_grid, num_workers_per_link, cluster_axis, enable_trace, mm_block_m, sub_h, sub_w",
    [
        # A: passing LTX shape/grid/workers, but ring size 8 (axis 1) instead of 4.
        (9472, 3456, 5120, ttnn.CoreCoord(12, 8), 5, 1, False, 256, 2, 1),
        # B: cosmos3 shape + power-clamped grid + derived workers, ring size 4 (axis 0).
        (22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 0, False, 256, 2, 1),
        # C: cosmos3 everything, ring size 8, eager.
        (22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 256, 2, 1),
        # D: cosmos3 everything under trace capture+replay.
        (22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, True, 256, 2, 1),
        # E/F/G: the other three trunk RowParallel shapes (to_out K=1024; und M=2720 has
        # ragged M-blocks: 85 tiles / M_block 4).
        (22144, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 256, 2, 1),
        # E under trace: the trunk bisect isolated the fused corruption to this shape;
        # case D (K=3200) is trace-clean, so K=1024 has its own traced coverage.
        # Passes in isolation — the trunk corruption needs more context.
        (22144, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, True, 256, 2, 1),
        (2720, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 128, 2, 2),
        (2720, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, False, 128, 2, 2),
    ],
    ids=[
        "A_ring8_ltx",
        "B_ring4_cosmos3",
        "C_ring8_cosmos3",
        "D_ring8_cosmos3_trace",
        "E_gen_to_out",
        "E_gen_to_out_trace",
        "F_und_down_proj",
        "G_und_to_out",
    ],
)
@pytest.mark.timeout(900)
def test_mmrs_cosmos3_shapes(
    mesh_device,
    topology,
    M,
    K,
    N,
    mm_core_grid,
    num_workers_per_link,
    cluster_axis,
    enable_trace,
    mm_block_m,
    sub_h,
    sub_w,
):
    _run_cosmos3_case(
        mesh_device,
        topology,
        M,
        K,
        N,
        mm_core_grid,
        num_workers_per_link,
        cluster_axis,
        enable_trace,
        mm_block_m=mm_block_m,
        sub_h=sub_h,
        sub_w=sub_w,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology", _COSMOS3_DEVICE_PARAMS, indirect=["device_params"], ids=["fabric_ring"]
)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.timeout(900)
def test_mmrs_cosmos3_submesh(mesh_device, topology, enable_trace):
    """Cosmos3 runs the op on a 2x8 submesh of the 4x8 parent (cfg-parallel split on
    axis 0); the full-mesh cases pass while the trunk corrupts, so exercise the op's
    ring-neighbor resolution under a submesh."""
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(2, 8))
    _run_cosmos3_case(submeshes[0], topology, 22144, 3200, 5120, ttnn.CoreCoord(10, 8), 4, 1, enable_trace)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology", _COSMOS3_DEVICE_PARAMS, indirect=["device_params"], ids=["fabric_ring"]
)
@pytest.mark.timeout(900)
def test_mmrs_cosmos3_adjacent_trace(mesh_device, topology):
    """Two K=1024 instances back-to-back in one captured graph — the trunk
    to_out/to_add_out adjacency, where a later instance's matmul cores start
    while the earlier instance's RS cores still drain. Both instances verified.
    Passes — the trunk's K=1024 corruption needs context beyond this pairing."""
    _run_cosmos3_case(mesh_device, topology, 22144, 1024, 5120, ttnn.CoreCoord(10, 8), 4, 1, True, ops_per_trace=2)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology", _COSMOS3_DEVICE_PARAMS, indirect=["device_params"], ids=["fabric_ring"]
)
@pytest.mark.timeout(900)
def test_strided_rs_batch2_trace(mesh_device, topology):
    """B=2 non-fused strided RS under 35 traced replays: exercises the per-batch
    batch_ready barrier decrement and the out_ready credit target growing across
    batches — the multi-batch half of the credit-decrement protocol, unreachable
    through the fused op (which requires B=1)."""
    import torch

    from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
        create_global_semaphores,
    )

    B, M, N = 2, 4096, 5120
    cluster_axis = 1
    num_devices = tuple(mesh_device.shape)[cluster_axis]
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    sems = create_global_semaphores(mesh_device, all_cores, 0)
    barrier = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    torch.manual_seed(0)
    torch_input = torch.randn(B, 1, M, N, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=_COSMOS3_DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.experimental.strided_reduce_scatter_async(
            tt_input,
            None,
            3,
            sems,
            barrier_semaphore=barrier,
            num_links=2,
            memory_config=_COSMOS3_DRAM,
            topology=topology,
            cluster_axis=cluster_axis,
            num_workers_per_link=4,
            num_buffers_per_channel=None,
            mm_cores_y=8,
            mm_block_ht=8,
            mm_block_wt=8,
            mm_N_full_block_wt=N // 32 // 10,
            chunk_width_in_mm_blocks=1,
        )

    run_op()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = run_op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    for _ in range(35):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

    # Inputs are replicated, so every device's scatter slice is num_devices * its
    # input chunk; batch content differs, so cross-batch credit mixups corrupt.
    slice_n = N // num_devices
    shards = ttnn.get_device_tensors(tt_out)
    for dev in range(num_devices):
        out = ttnn.to_torch(shards[dev]).to(torch.float32)  # row 0 of the mesh
        ref = num_devices * torch_input[:, :, :, dev * slice_n : (dev + 1) * slice_n]
        pcc = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1].item()
        assert pcc >= 0.999, f"device {dev}: B=2 traced RS PCC {pcc:.6f}"
