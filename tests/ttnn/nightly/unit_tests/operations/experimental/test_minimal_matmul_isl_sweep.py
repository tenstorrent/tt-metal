# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Util vs KV ISL for minimal_matmul dynamic-M, Kimi K2.7 MLA shape (sp=8, tp=4).

Global ISL 5k..256k -> per-device M = ISL/8. K=576, N=2048, bf8->bf8, HiFi2, BH 11x10.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tracy.process_model_log import run_device_profiler

from tests.ttnn.nightly.unit_tests.operations.experimental.test_minimal_matmul import (
    _uint32_scalar,
    post_process_ops_log,
)

K, N, HEAD_DIM, SP = 576, 2048, 128, 8
K_T, N_T = K // 32, N // 32
CLOCK_GHZ = 1.35
BF8_TILE_BYTES = 1088  # 1024 data + 64 exponent
DRAM_BPS = 512 * 1024**3  # 512 GiB/s
ITERS = 30

ISLS = [5120, 10240, 20480, 32768, 51200, 65536, 98304, 131072, 196608, 262144]
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

# (label, M_block, K_block, N_block, sh, sw, out_mem)
VARIANTS = [
    ("M20K2N3_s1x3_L1", 20, 2, 3, 1, 3, L1),
    ("M20K2N3_s1x3_DR", 20, 2, 3, 1, 3, DRAM),
    ("M10K2N3_s2x3_L1", 10, 2, 3, 2, 3, L1),
    ("M4K2N3_s2x3_L1", 4, 2, 3, 2, 3, L1),
    ("M4K2N3_s2x3_DR", 4, 2, 3, 2, 3, DRAM),
]
CASES = {f"{isl}__{v[0]}": (isl, v) for isl in ISLS for v in VARIANTS}


def roofline(m_t):
    tiles = m_t * K_T * N_T
    compute_ns = tiles * 32 / 110 / CLOCK_GHZ
    dm_bytes = (m_t * K_T + K_T * N_T + m_t * N_T) * BF8_TILE_BYTES
    return tiles, compute_ns, dm_bytes / DRAM_BPS * 1e9


@pytest.mark.parametrize("case", list(CASES))
def test_run_case(device, case):
    isl, (_, mbs, kbs, nbs, sh, sw, out_mem) = CASES[case]
    m = isl // SP
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mbs,
        K_block_size=kbs,
        N_block_size=nbs,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
    )
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    tt_cache = ttnn.from_torch(
        torch.randn(1, 1, m, K, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_w = ttnn.from_torch(
        torch.randn(K, N, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_valid = _uint32_scalar(device, m)
    for _ in range(ITERS):
        ttnn.deallocate(
            ttnn.experimental.minimal_matmul(
                tt_cache,
                tt_w,
                config=cfg,
                compute_kernel_config=ck,
                dtype=ttnn.bfloat8_b,
                memory_config=out_mem,
                valid_rows_tensor=tt_valid,
                out_head_dim=HEAD_DIM,
            )
        )
    ttnn.synchronize_device(device)


def test_isl_sweep_table():
    res = {}
    for case, (isl, v) in CASES.items():
        subdir = f"mm_isl_{case}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_isl_sweep.py::test_run_case[case={case}]"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=False
            )
            res[case] = (int(r["CORE COUNT"].max()), float(r["DEVICE KERNEL DURATION [ns]"].min()))
        except Exception as e:
            logger.error(f"{case} FAILED: {str(e)[:120]}")

    print("\n| global ISL | M local | M_t | tiles | best config | cores | us | util % | compute us | DM us | bound |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for isl in ISLS:
        m = isl // SP
        tiles, c_ns, d_ns = roofline(m // 32)
        cands = [
            (res[f"{isl}__{v[0]}"][1], v[0], res[f"{isl}__{v[0]}"][0]) for v in VARIANTS if f"{isl}__{v[0]}" in res
        ]
        if not cands:
            print(
                f"| {isl} | {m} | {m//32} | {tiles} | ALL FAILED | - | - | - | {c_ns/1000:.1f} | {d_ns/1000:.1f} | - |"
            )
            continue
        ns, label, cores = min(cands)
        util = 100 * (tiles * 32 / cores / CLOCK_GHZ) / ns
        bound = "compute" if c_ns > d_ns else "DM"
        print(
            f"| {isl} | {m} | {m//32} | {tiles} | {label} | {cores} | {ns/1000:.1f} | {util:.1f} "
            f"| {c_ns/1000:.1f} | {d_ns/1000:.1f} | {bound} |"
        )

    print("\n| global ISL | " + " | ".join(v[0] for v in VARIANTS) + " |")
    print("|---" * (len(VARIANTS) + 1) + "|")
    for isl in ISLS:
        cells = []
        for v in VARIANTS:
            k = f"{isl}__{v[0]}"
            cells.append(f"{res[k][1]/1000:.1f}" if k in res else "fail")
        print(f"| {isl} | " + " | ".join(cells) + " |")
