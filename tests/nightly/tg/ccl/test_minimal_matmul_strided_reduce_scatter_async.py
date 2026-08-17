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


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_minimal_matmul_strided_reduce_scatter_fused_concat_non_aligned(mesh_device, num_links, cluster_axis, topology):
    """MMRS fused concat with non-tile-aligned Ka and Kb (TG/galaxy).

    K=32 per device (logical), Ka=11 (non-aligned prefix), Kb=21 (non-aligned suffix).
    Weight is per-segment tile-padded: prefix rows [0..11) real, zeros to tile 32,
    suffix rows [32..53) real, zeros to tile 64.  K_padded=64 (2 tiles), mm_block_k=64.
    """
    if mesh_device.shape[cluster_axis] == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device in this mesh, reduce-scatter ring size must be > 1")

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=128,
        K=32,  # logical per-device K: Ka=11 + Kb=21
        N=512,
        dim=3,
        num_links=num_links,
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
        cluster_axis=cluster_axis,
        fused_concat=True,
        fused_concat_ka=11,  # Ka=11: non-tile-aligned prefix (padded to 32)
    )
