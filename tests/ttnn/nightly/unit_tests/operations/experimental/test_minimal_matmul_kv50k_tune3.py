# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 3: L1 placement + finest blocking for minimal_matmul dynamic-M, Kimi K2.7 KV shape."""

import pytest
import torch
import ttnn
from loguru import logger

from tracy.process_model_log import run_device_profiler

from tests.ttnn.nightly.unit_tests.operations.experimental.test_minimal_matmul import (
    _uint32_scalar,
    post_process_ops_log,
)

M_CAP, K, N, HEAD_DIM = 6400, 576, 2048, 128
TILES = (M_CAP // 32) * (K // 32) * (N // 32)
CLOCK_GHZ = 1.35
ITERS = 50
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def C(mbs, kbs, nbs, sh, sw, gx=11, gy=10, act=DRAM, out=DRAM):
    return dict(mbs=mbs, kbs=kbs, nbs=nbs, sh=sh, sw=sw, gx=gx, gy=gy, act=act, out=out)


CONFIGS = {
    # L1 placement on the round-2 winner (M20 K2 N3 s2x3)
    "w_dram_dram": C(20, 2, 3, 2, 3),
    "w_actL1": C(20, 2, 3, 2, 3, act=L1),
    "w_outL1": C(20, 2, 3, 2, 3, out=L1),
    "w_bothL1": C(20, 2, 3, 2, 3, act=L1, out=L1),
    # finest blocking, DRAM
    "M20K1N3_s2x3": C(20, 1, 3, 2, 3),
    "M20K2N2_s2x2": C(20, 2, 2, 2, 2),
    "M20K2N6_s2x3": C(20, 2, 6, 2, 3),
    "M10K2N3_s2x3": C(10, 2, 3, 2, 3),
    "M20K2N3_s1x3": C(20, 2, 3, 1, 3),
    "M20K2N3_s4x3": C(20, 2, 3, 4, 3),
    "M20K2N8_grid8": C(20, 2, 8, 2, 4, gx=8, gy=10),
    # finest blocking + both-L1
    "M20K1N3_bothL1": C(20, 1, 3, 2, 3, act=L1, out=L1),
    "M20K2N6_bothL1": C(20, 2, 6, 2, 3, act=L1, out=L1),
    "M20K4N3_bothL1": C(20, 4, 3, 2, 3, act=L1, out=L1),
    "M20K2N8_grid8_bothL1": C(20, 2, 8, 2, 4, gx=8, gy=10, act=L1, out=L1),
}


def build(device, name):
    c = CONFIGS[name]
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=c["mbs"],
        K_block_size=c["kbs"],
        N_block_size=c["nbs"],
        subblock_h=c["sh"],
        subblock_w=c["sw"],
        compute_with_storage_grid_size=ttnn.CoreCoord(c["gx"], c["gy"]),
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
        torch.randn(1, 1, M_CAP, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=c["act"],
    )
    tt_w = ttnn.from_torch(
        torch.randn(K, N, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    return c, tt_cache, tt_w, cfg, ck


@pytest.mark.parametrize("name", list(CONFIGS))
def test_run_cfg(device, name):
    c, tt_cache, tt_w, cfg, ck = build(device, name)
    tt_valid = _uint32_scalar(device, M_CAP)
    for _ in range(ITERS):
        ttnn.deallocate(
            ttnn.experimental.minimal_matmul(
                tt_cache,
                tt_w,
                config=cfg,
                compute_kernel_config=ck,
                dtype=ttnn.bfloat8_b,
                memory_config=c["out"],
                valid_rows_tensor=tt_valid,
                out_head_dim=HEAD_DIM,
            )
        )
    ttnn.synchronize_device(device)


def test_tune3_table():
    rows = []
    for name in CONFIGS:
        subdir = f"mm_kv50k3_{name}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_kv50k_tune3.py::test_run_cfg[name={name}]"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=False
            )
            cores = int(r["CORE COUNT"].max())
            ns = float(r["DEVICE KERNEL DURATION [ns]"].min())
            rows.append((name, cores, ns / 1000, 100 * (TILES * 32 / cores / CLOCK_GHZ) / ns))
        except Exception as e:
            rows.append((name, 0, float("nan"), float("nan")))
            logger.error(f"{name} FAILED: {str(e)[:160]}")
    floor = TILES * 32 / 110 / CLOCK_GHZ / 1000
    print(f"\n# M={M_CAP} K={K} N={N} bf8->bf8 HiFi2 | floor {floor:.1f} us | 70% target {floor/0.70:.1f} us\n")
    print("| config | cores | us | util % |")
    print("|---|---|---|---|")
    for name, cores, us, util in sorted(rows, key=lambda r: (r[2] != r[2], r[2])):
        print(f"| {name} | - | FAILED | - |" if us != us else f"| {name} | {cores} | {us:.2f} | {util:.1f} |")
