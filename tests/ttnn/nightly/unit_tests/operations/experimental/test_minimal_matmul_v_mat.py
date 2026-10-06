# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V-materialization matmul: latent KV cache -> per-head V, ahead of ring SDPA.

in0 = [1,1,M_cap,512] bf8 latent cache (kv_lora_rank), M runtime-bounded by valid_rows.
in1 = wkv_b2 [1,heads,512,128] flattened over heads -> [512, heads*128].
out = [1,heads,M_cap,128] bf8 head-major.  Kimi K2.7, tp=4 -> heads=16, N=2048.
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

K, N, HEAD_DIM = 512, 2048, 128
M_CAP = int(__import__("os").environ.get("VMAT_M", 6400))
CLOCK_GHZ = 1.35
ITERS = 50
L1 = ttnn.L1_MEMORY_CONFIG


def C(kbs, nbs, sh, sw, mbs=20):
    return dict(mbs=mbs, kbs=kbs, nbs=nbs, sh=sh, sw=sw)


CONFIGS = {
    "K2_N6_s1x6": C(2, 6, 1, 6),
    "K2_N7_s1x7": C(2, 7, 1, 7),
}


def build(device, name, m_cap=M_CAP):
    c = CONFIGS[name]
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=c["mbs"],
        K_block_size=c["kbs"],
        N_block_size=c["nbs"],
        subblock_h=c["sh"],
        subblock_w=c["sw"],
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
    lat = torch.randn(1, 1, m_cap, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    return (
        lat,
        w,
        ttnn.from_torch(lat, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
        cfg,
        ck,
    )


def call(tt_lat, tt_w, cfg, ck, tt_valid):
    return ttnn.experimental.minimal_matmul(
        tt_lat,
        tt_w,
        config=cfg,
        compute_kernel_config=ck,
        dtype=ttnn.bfloat8_b,
        memory_config=L1,
        valid_rows_tensor=tt_valid,
        out_head_dim=HEAD_DIM,
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_run_cfg(device, name):
    _, _, tt_lat, tt_w, cfg, ck = build(device, name)
    tt_valid = _uint32_scalar(device, M_CAP)
    for _ in range(ITERS):
        ttnn.deallocate(call(tt_lat, tt_w, cfg, ck, tt_valid))
    ttnn.synchronize_device(device)


def test_pcc(device):
    lat, w, tt_lat, tt_w, cfg, ck = build(device, next(iter(CONFIGS)))
    out = ttnn.to_torch(call(tt_lat, tt_w, cfg, ck, _uint32_scalar(device, M_CAP))).float()
    ref = (lat[0, 0].float() @ w.float()).reshape(M_CAP, -1, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    q = assert_quality(ref, out)
    logger.info(f"V-mat PCC={q['pcc']:.6f} rel_rmse={q['relative_rmse']:.4f}")
    assert q["pcc"] > 0.99


def test_vmat_table():
    tiles = (M_CAP // 32) * (K // 32) * (N // 32)
    rows = []
    for name in CONFIGS:
        subdir = f"mm_vmat_{name}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_v_mat.py::test_run_cfg[name={name}]"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir,
                float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "DEVICE BRISC KERNEL DURATION [ns]"],
                sum_vals=False,
                has_signposts=False,
            )
            cores = int(r["CORE COUNT"].max())
            ns = float(r["DEVICE KERNEL DURATION [ns]"].min())
            br = float(r["DEVICE BRISC KERNEL DURATION [ns]"].max())
            rows.append((name, cores, ns / 1000, 100 * (tiles * 32 / cores / CLOCK_GHZ) / ns, 100 * br / ns))
        except Exception as e:
            rows.append((name, 0, float("nan"), float("nan"), float("nan")))
            logger.error(f"{name} FAILED: {str(e)[:160]}")
    floor = tiles * 32 / 110 / CLOCK_GHZ / 1000
    print(
        f"\n# V-mat M={M_CAP} K={K} N={N} bf8->bf8 HiFi2 | tiles={tiles} | floor {floor:.1f} us | 70% target {floor/0.7:.1f} us\n"
    )
    print("| config | cores | us | util % | BRISC busy % |")
    print("|---|---|---|---|---|")
    for name, cores, us, util, br in sorted(rows, key=lambda r: (r[2] != r[2], r[2])):
        print(
            f"| {name} | - | FAILED | - | - |"
            if us != us
            else f"| {name} | {cores} | {us:.2f} | {util:.1f} | {br:.0f} |"
        )
