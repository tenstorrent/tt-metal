# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config tuning for minimal_matmul dynamic-M on the Kimi K2.7 MLA KV-cache shape.

Per-device shape at global ISL 50k with sp=8 / tp=4:
  M_cap = 6400 (50k/8), K = 576 (kv_lora 512 + rope 64), N = 2048 (64 heads/4 x 128)
  in0 = bf8_b KV cache, in1 = bf8_b weight, out = bf8_b, HiFi2, BH 11x10.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tracy.process_model_log import run_device_profiler

from tests.ttnn.nightly.unit_tests.operations.experimental.test_minimal_matmul import (
    _uint32_scalar,
    assert_quality,
    post_process_ops_log,
)

M_CAP, K, N, HEAD_DIM = 6400, 576, 2048, 128
M_T, K_T, N_T = M_CAP // 32, K // 32, N // 32
TILES = M_T * K_T * N_T
FIDELITY_CYCLES = 32  # HiFi2
CLOCK_GHZ = 1.35
ITERS = 50

# (name, M_block, K_block, N_block, subblock_h, subblock_w, fp32_acc, grid_x, grid_y)
CONFIGS = [
    ("base_M8K9N8_s2x2_fp32", 8, 9, 8, 2, 2, True, 11, 10),
    ("M10K9N6_s2x3", 10, 9, 6, 2, 3, False, 11, 10),
    ("M20K9N6_s2x3", 20, 9, 6, 2, 3, False, 11, 10),
    ("M10K18N6_s2x3", 10, 18, 6, 2, 3, False, 11, 10),
    ("M20K18N6_s2x3", 20, 18, 6, 2, 3, False, 11, 10),
    ("M10K9N6_s1x6", 10, 9, 6, 1, 6, False, 11, 10),
    ("M8K9N6_s4x2", 8, 9, 6, 4, 2, False, 11, 10),
    ("M10K6N6_s2x3", 10, 6, 6, 2, 3, False, 11, 10),
    ("M5K9N6_s1x6", 5, 9, 6, 1, 6, False, 11, 10),
    ("M20K9N3_s2x3", 20, 9, 3, 2, 3, False, 11, 10),
    ("M10K9N6_s2x2_fp32", 10, 9, 6, 2, 2, True, 11, 10),
    ("M10K9N8_s2x4_grid8", 10, 9, 8, 2, 4, False, 8, 10),
    ("M20K9N6_s5x1", 20, 9, 6, 5, 1, False, 11, 10),
]
BY_NAME = {c[0]: c for c in CONFIGS}


def build(device, name, m_cap=M_CAP):
    _, mbs, kbs, nbs, sh, sw, fp32, gx, gy = BY_NAME[name]
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mbs,
        K_block_size=kbs,
        N_block_size=nbs,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    cache = torch.randn(1, 1, m_cap, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    tt_cache = ttnn.from_torch(cache, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    return cache, w, tt_cache, tt_w, cfg, ck


def run_op(tt_cache, tt_w, cfg, ck, tt_valid):
    return ttnn.experimental.minimal_matmul(
        tt_cache,
        tt_w,
        config=cfg,
        compute_kernel_config=ck,
        dtype=ttnn.bfloat8_b,
        valid_rows_tensor=tt_valid,
        out_head_dim=HEAD_DIM,
    )


@pytest.mark.parametrize("name", [c[0] for c in CONFIGS])
def test_run_cfg(device, name):
    _, _, tt_cache, tt_w, cfg, ck = build(device, name)
    tt_valid = _uint32_scalar(device, M_CAP)
    for _ in range(ITERS):
        ttnn.deallocate(run_op(tt_cache, tt_w, cfg, ck, tt_valid))
    ttnn.synchronize_device(device)


def test_tune_table():
    rows = []
    for name, *_ in CONFIGS:
        subdir = f"mm_kv50k_{name}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_kv50k_tune.py::test_run_cfg[name={name}]"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir,
                float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"],
                sum_vals=False,
                has_signposts=False,
            )
            d = r["DEVICE KERNEL DURATION [ns]"]
            cores = int(r["CORE COUNT"].max())
            ns = float(d.min())
            ideal = TILES * FIDELITY_CYCLES / cores / CLOCK_GHZ
            rows.append((name, cores, ns / 1000, 100 * ideal / ns))
        except Exception as e:
            rows.append((name, 0, float("nan"), float("nan")))
            logger.error(f"{name} FAILED: {str(e)[:200]}")

    compute_floor = TILES * FIDELITY_CYCLES / 110 / CLOCK_GHZ / 1000
    print("")
    print(f"# M={M_CAP} K={K} N={N} bf8->bf8 HiFi2 | tiles={TILES} | 110-core compute floor {compute_floor:.1f} us")
    print(f"# 70% util target = {compute_floor/0.70:.1f} us")
    print("")
    print("| config | cores | us | util % |")
    print("|---|---|---|---|")
    for name, cores, us, util in sorted(rows, key=lambda r: (r[2] != r[2], r[2])):
        if us != us:
            print(f"| {name} | - | FAILED | - |")
        else:
            print(f"| {name} | {cores} | {us:.2f} | {util:.1f} |")
