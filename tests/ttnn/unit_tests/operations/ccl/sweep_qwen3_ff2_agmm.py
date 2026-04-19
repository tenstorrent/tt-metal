# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fast single-process block-size sweep for Qwen3-32B FF2 fused AG+MM
(all_gather_minimal_matmul_async) on WH Galaxy.

Design:
  * Uses pytest's `mesh_device` + `device_params` fixtures so conftest owns
    the (evolving) fabric-setup API -- we just parametrize the shape and
    the block-size combos.
  * Single pytest process per (M, grid). Mesh opens once, tensors once,
    the hot loop only varies MinimalMatmulConfig.
  * Timing = Python wallclock around 1 warmup + N measured iters. Each
    iter ends with ttnn.synchronize_device, so noise is small relative to
    op time (>100 ms).
  * CSV append-only, flushed BEFORE moving to the next combo -> a hang or
    C++ abort loses at most the running combo.
  * Op invocation mirrors `line_all_gather_matmul` in llama_ccl.py, so
    measured times are representative of what the Qwen model sees.
  * Config space pruned from what we already learned about Qwen FF2:
      K_block in {1, 5, 25}  (divisors of K_tiles_per_device = 25)
      N_block in divisors(40) clipped to [4, 20]
      M_block in divisors(M_tiles) clipped to [2, 32]
      subblock auto-picked (max h*w <= 8, h|M_block, w|N_block)
      grids: (6,8), (6,9), (5,8)  -> num_links {3, 3, 1}

Qwen3-32B FF2 per-device shapes (cluster_axis=1, 4-way TP ring):
  K_full (post-AG) = 3200, K_per_device = 800  (25 K-tiles per device)
  N_per_device     = 1280  (40 N-tiles)
  M                = seq_len

Prerequisites:
  * Run from the tt-metal repo root (CSV is written to cwd).
  * Galaxy fabric must be clean. A plain `tt-smi-metal -r 0..7` only resets 8
    chips; the 32-chip WH Galaxy needs:
        tt-smi-metal -glx_reset_auto
    Symptoms of a dirty fabric: "TT_FATAL: Graph specified in MGD could not
    fit in the discovered physical topology" or `IndexError: unordered_map::at`
    during mesh_device setup.

Usage:
  tt-smi-metal -glx_reset_auto                                        # ~60 s
  pytest tests/ttnn/unit_tests/operations/ccl/sweep_qwen3_ff2_agmm.py -s  # ~9 min

  # One shape only:
  pytest tests/.../sweep_qwen3_ff2_agmm.py -s -k "4096_6x8"

  # Smoke test (2 combos, ~10 s):
  QWEN_AGMM_SMOKE=1 pytest tests/.../sweep_qwen3_ff2_agmm.py -s -k "4096_6x8"

  # Custom CSV path:
  QWEN_AGMM_CSV=/tmp/my_sweep.csv pytest ...

Outputs:
  sweep_results_qwen3_ff2.csv (crash-safe append, one row per combo):
    M,K_full,N,gx,gy,M_block,K_block,N_block,sub_h,sub_w,
    num_links,workers_per_link,iters,mean_ms,min_ms,status
"""

import csv
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

# ----------------------------------------------------------------------
# Constants from Qwen3-32B FF2 on WH Galaxy (8x4 mesh)
# ----------------------------------------------------------------------
K_FULL = 3200  # intermediate_dim / 2 / 4  (gated down-proj, 4-way TP)
N_PER_DEVICE = 1280  # hidden_dim / 4
RING_SIZE = 4
CLUSTER_AXIS = 1
K_TILES_PER_DEVICE = (K_FULL // RING_SIZE) // 32  # = 25
N_TILES = N_PER_DEVICE // 32  # = 40

CSV_PATH = os.environ.get("QWEN_AGMM_CSV", "sweep_results_qwen3_ff2.csv")
CSV_COLUMNS = [
    "M",
    "K_full",
    "N",
    "gx",
    "gy",
    "M_block",
    "K_block",
    "N_block",
    "sub_h",
    "sub_w",
    "num_links",
    "workers_per_link",
    "iters",
    "mean_ms",
    "min_ms",
    "status",
]

# (M, gx, gy) -- one pytest test per entry.
SHAPES = [
    (4096, 6, 8),
    (4096, 6, 9),
    (4096, 5, 8),
    (8192, 6, 8),
    (8192, 6, 9),
]
SHAPE_IDS = [f"{M}_{gx}x{gy}" for M, gx, gy in SHAPES]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def divisors(n, cap=None):
    ds = [i for i in range(1, n + 1) if n % i == 0]
    return [d for d in ds if cap is None or d <= cap]


def pick_subblock(m_block, n_block, max_vol=8):
    best = (1, 1)
    for h in divisors(m_block, cap=max_vol):
        for w in divisors(n_block, cap=max_vol):
            if h * w <= max_vol and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def derive_num_links(gx):
    for nl in (4, 3, 2, 1):
        if gx % nl == 0:
            return nl
    return 1


def derive_workers_per_link(gx, num_links):
    return max(1, min(8 // num_links, gx // num_links))


def build_configs(M_tiles, gx):
    m_blocks = [m for m in divisors(M_tiles, cap=32) if m >= 2]
    k_blocks = divisors(K_TILES_PER_DEVICE)  # {1, 5, 25}
    n_blocks = [n for n in divisors(N_TILES) if 4 <= n <= 20]
    combos = []
    for mb in m_blocks:
        for kb in k_blocks:
            for nb in n_blocks:
                sh, sw = pick_subblock(mb, nb)
                combos.append((mb, kb, nb, sh, sw))
    return combos


def write_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append_row(row):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[k] for k in CSV_COLUMNS])


def setup_tensors(mesh_device, M, gx, gy):
    """Mirror of llama_ccl.line_all_gather_matmul input layout."""
    dtype = ttnn.bfloat8_b
    mesh_shape = tuple(mesh_device.shape)

    tt_input = ttnn.from_torch(
        torch.randn((M, K_FULL), dtype=torch.float32),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=[None, CLUSTER_AXIS]),
    )
    tt_weight = ttnn.from_torch(
        torch.randn((K_FULL, N_PER_DEVICE), dtype=torch.float32),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    ccl_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    # 8 semaphores = enough for any num_links in {1..4} * 2.
    sems = [ttnn.create_global_semaphore(mesh_device, ccl_cores, 0) for _ in range(8)]
    return tt_input, tt_weight, sems


def run_once(tt_input, tt_weight, sems, matmul_config, compute_config, num_links, workers):
    inp = ttnn.reshape(tt_input, (1, 1, tt_input.shape[-2], tt_input.shape[-1]))
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=inp,
        weight_tensor=tt_weight,
        config=matmul_config,
        compute_kernel_config=compute_config,
        multi_device_global_semaphore=sems[: num_links * 2],
        num_links=num_links,
        topology=ttnn.Topology.Ring,
        cluster_axis=CLUSTER_AXIS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
    )
    # Release the output so DRAM doesn't accumulate across combos.
    if isinstance(out, (list, tuple)):
        for o in out:
            if o is not None:
                ttnn.deallocate(o)
    else:
        ttnn.deallocate(out)


# ----------------------------------------------------------------------
# The sweep (one pytest test per shape)
# ----------------------------------------------------------------------
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_qwen3_ff2_agmm_sweep(mesh_device, shape, reset_seeds):
    M, gx, gy = shape
    M_tiles = M // 32

    iters = int(os.environ.get("QWEN_AGMM_ITERS", "3"))
    smoke = os.environ.get("QWEN_AGMM_SMOKE", "0") == "1"

    configs = build_configs(M_tiles, gx)
    if smoke:
        # Current baked-in winner + one close relative.
        configs = [(8, 5, 5, 1, 5), (8, 5, 8, 1, 4)]
        configs = [c for c in configs if c[0] <= M_tiles and M_tiles % c[0] == 0]

    num_links = derive_num_links(gx)
    workers = derive_workers_per_link(gx, num_links)

    logger.info(
        f"Qwen FF2 AG+MM sweep: M={M} K_full={K_FULL} N={N_PER_DEVICE} "
        f"grid=({gx},{gy}) num_links={num_links} workers={workers} "
        f"-> {len(configs)} combos, iters={iters}"
    )

    write_csv_header()

    tt_input, tt_weight, sems = setup_tensors(mesh_device, M, gx, gy)
    # Matches model's compute_kernel_config_hifi2_fp16 (qwen_model_config.py).
    # fp32_dest_acc_en=False -> max_dest_volume=8, so subblock_h*subblock_w<=8.
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    results = []
    for i, (mb, kb, nb, sh, sw) in enumerate(configs, 1):
        row = {
            "M": M,
            "K_full": K_FULL,
            "N": N_PER_DEVICE,
            "gx": gx,
            "gy": gy,
            "M_block": mb,
            "K_block": kb,
            "N_block": nb,
            "sub_h": sh,
            "sub_w": sw,
            "num_links": num_links,
            "workers_per_link": workers,
            "iters": 0,
            "mean_ms": -1.0,
            "min_ms": -1.0,
            "status": "PENDING",
        }
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        )

        try:
            # Warmup also triggers kernel compile; sync before timing.
            run_once(tt_input, tt_weight, sems, matmul_config, compute_config, num_links, workers)
            ttnn.synchronize_device(mesh_device)

            per_iter = []
            for _ in range(iters):
                t0 = time.perf_counter()
                run_once(tt_input, tt_weight, sems, matmul_config, compute_config, num_links, workers)
                ttnn.synchronize_device(mesh_device)
                per_iter.append((time.perf_counter() - t0) * 1000.0)

            row["iters"] = iters
            row["mean_ms"] = sum(per_iter) / iters
            row["min_ms"] = min(per_iter)
            row["status"] = "OK"
            results.append(row)
            logger.info(
                f"  [{i}/{len(configs)}] M={mb} K={kb} N={nb} sb=({sh},{sw}) "
                f"-> min={row['min_ms']:.2f}ms mean={row['mean_ms']:.2f}ms"
            )
        except Exception as e:
            msg = str(e).splitlines()[0][:140]
            row["status"] = f"FAIL: {msg}"
            logger.warning(f"  [{i}/{len(configs)}] M={mb} K={kb} N={nb} sb=({sh},{sw}) -> {msg}")

        append_row(row)

    ok = [r for r in results if r["status"] == "OK"]
    ok.sort(key=lambda r: r["min_ms"])
    logger.info(f"==== DONE M={M} grid=({gx},{gy}): " f"{len(ok)}/{len(configs)} OK, top-5 by min_ms ====")
    for rank, r in enumerate(ok[:5], 1):
        logger.info(
            f"  #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
            f"sb=({r['sub_h']},{r['sub_w']}) -> min={r['min_ms']:.2f}ms "
            f"mean={r['mean_ms']:.2f}ms"
        )
