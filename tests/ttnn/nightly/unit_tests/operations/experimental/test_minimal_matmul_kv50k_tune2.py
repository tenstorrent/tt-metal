# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 2: diagnostics + finer blocking for minimal_matmul dynamic-M on the Kimi K2.7 KV shape."""

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
FID = {
    "LoFi": (ttnn.MathFidelity.LoFi, 16),
    "HiFi2": (ttnn.MathFidelity.HiFi2, 32),
    "HiFi4": (ttnn.MathFidelity.HiFi4, 64),
}


# name: dict(mbs,kbs,nbs,sh,sw,fp32,gx,gy,fid,dyn,in_dt,out_dt)
def C(mbs, kbs, nbs, sh, sw, fp32=False, gx=11, gy=10, fid="HiFi2", dyn=True, in_dt="bf8", out_dt="bf8"):
    return dict(
        mbs=mbs, kbs=kbs, nbs=nbs, sh=sh, sw=sw, fp32=fp32, gx=gx, gy=gy, fid=fid, dyn=dyn, in_dt=in_dt, out_dt=out_dt
    )


CONFIGS = {
    # --- diagnostics on the round-1 winner ---
    "best_HiFi2": C(20, 9, 3, 2, 3),
    "best_LoFi": C(20, 9, 3, 2, 3, fid="LoFi"),
    "best_HiFi4": C(20, 9, 3, 2, 3, fid="HiFi4"),
    "best_static": C(20, 9, 3, 2, 3, dyn=False),
    "best_outbf16": C(20, 9, 3, 2, 3, out_dt="bf16"),
    "best_inbf16": C(20, 9, 3, 2, 3, in_dt="bf16"),
    # --- finer blocking ---
    "M20K9N2_s2x2": C(20, 9, 2, 2, 2),
    "M20K9N1_s2x1": C(20, 9, 1, 2, 1),
    "M20K6N3_s2x3": C(20, 6, 3, 2, 3),
    "M20K3N3_s2x3": C(20, 3, 3, 2, 3),
    "M20K2N3_s2x3": C(20, 2, 3, 2, 3),
    "M10K9N3_s2x3": C(10, 9, 3, 2, 3),
    "M4K9N3_s2x3": C(4, 9, 3, 2, 3),
    "M20K9N3_s1x3": C(20, 9, 3, 1, 3),
    "M20K9N3_s4x1": C(20, 9, 3, 4, 1),
    "M20K6N2_s2x2": C(20, 6, 2, 2, 2),
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
        math_fidelity=FID[c["fid"]][0],
        math_approx_mode=False,
        fp32_dest_acc_en=c["fp32"],
        packer_l1_acc=True,
    )
    dt = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    torch.manual_seed(0)
    tt_cache = ttnn.from_torch(
        torch.randn(1, 1, M_CAP, K, dtype=torch.bfloat16), dtype=dt[c["in_dt"]], layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_w = ttnn.from_torch(
        torch.randn(K, N, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    return c, tt_cache, tt_w, cfg, ck, dt[c["out_dt"]]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_run_cfg(device, name):
    c, tt_cache, tt_w, cfg, ck, out_dt = build(device, name)
    kw = {}
    if c["dyn"]:
        kw = dict(valid_rows_tensor=_uint32_scalar(device, M_CAP), out_head_dim=HEAD_DIM)
    for _ in range(ITERS):
        ttnn.deallocate(
            ttnn.experimental.minimal_matmul(tt_cache, tt_w, config=cfg, compute_kernel_config=ck, dtype=out_dt, **kw)
        )
    ttnn.synchronize_device(device)


def test_tune2_table():
    rows = []
    for name in CONFIGS:
        subdir = f"mm_kv50k2_{name}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_kv50k_tune2.py::test_run_cfg[name={name}]"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=False
            )
            cores = int(r["CORE COUNT"].max())
            ns = float(r["DEVICE KERNEL DURATION [ns]"].min())
            fc = FID[CONFIGS[name]["fid"]][1]
            rows.append((name, cores, ns / 1000, 100 * (TILES * fc / cores / CLOCK_GHZ) / ns))
        except Exception as e:
            rows.append((name, 0, float("nan"), float("nan")))
            logger.error(f"{name} FAILED: {str(e)[:160]}")
    floor = TILES * 32 / 110 / CLOCK_GHZ / 1000
    print(f"\n# M={M_CAP} K={K} N={N} | HiFi2 110-core compute floor {floor:.1f} us | 70% target {floor/0.70:.1f} us\n")
    print("| config | cores | us | util % (own fidelity) |")
    print("|---|---|---|---|")
    for name, cores, us, util in sorted(rows, key=lambda r: (r[2] != r[2], r[2])):
        print(f"| {name} | - | FAILED | - |" if us != us else f"| {name} | {cores} | {us:.2f} | {util:.1f} |")
