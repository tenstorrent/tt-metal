# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 4: fidelity + K=512 (kv_lora only) on the tuned config, plus a PCC gate."""

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

M_CAP, N, HEAD_DIM = 6400, 2048, 128
CLOCK_GHZ = 1.35
ITERS = 50
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
FID = {"LoFi": (ttnn.MathFidelity.LoFi, 16), "HiFi2": (ttnn.MathFidelity.HiFi2, 32)}


def C(k, mbs, kbs, nbs, sh, sw, fid="HiFi2", out=L1):
    return dict(K=k, mbs=mbs, kbs=kbs, nbs=nbs, sh=sh, sw=sw, fid=fid, out=out)


CONFIGS = {
    "K576_s2x3_outL1": C(576, 20, 2, 3, 2, 3),
    "K576_s1x3_outL1": C(576, 20, 2, 3, 1, 3),
    "K576_K3_outL1": C(576, 20, 3, 3, 2, 3),
    "K576_M10_outL1": C(576, 10, 2, 3, 2, 3),
    "K576_s2x3_outL1_LoFi": C(576, 20, 2, 3, 2, 3, fid="LoFi"),
    "K512_s2x3_outL1": C(512, 20, 2, 3, 2, 3),
    "K512_K4_outL1": C(512, 20, 4, 3, 2, 3),
    "K512_K8_outL1": C(512, 20, 8, 3, 2, 3),
    "K512_s2x3_outL1_LoFi": C(512, 20, 2, 3, 2, 3, fid="LoFi"),
    "K576_N7_s1x7_outL1": C(576, 20, 2, 7, 1, 7),
    "K576_N7_s1x1_outL1": C(576, 20, 2, 7, 1, 1),
    "K576_N14_s1x7_outL1": C(576, 20, 2, 14, 1, 7),
    "K576_N7_K6_outL1": C(576, 20, 6, 7, 1, 7),
}


def build(device, name):
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
        math_fidelity=FID[c["fid"]][0],
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    cache = torch.randn(1, 1, M_CAP, c["K"], dtype=torch.bfloat16)
    w = torch.randn(c["K"], N, dtype=torch.bfloat16)
    tt_cache = ttnn.from_torch(cache, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    return c, cache, w, tt_cache, tt_w, cfg, ck


def _call(c, tt_cache, tt_w, cfg, ck, tt_valid):
    return ttnn.experimental.minimal_matmul(
        tt_cache,
        tt_w,
        config=cfg,
        compute_kernel_config=ck,
        dtype=ttnn.bfloat8_b,
        memory_config=c["out"],
        valid_rows_tensor=tt_valid,
        out_head_dim=HEAD_DIM,
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_run_cfg(device, name):
    c, _, _, tt_cache, tt_w, cfg, ck = build(device, name)
    tt_valid = _uint32_scalar(device, M_CAP)
    for _ in range(ITERS):
        ttnn.deallocate(_call(c, tt_cache, tt_w, cfg, ck, tt_valid))
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("name", ["K576_s2x3_outL1", "K576_N7_s1x7_outL1", "K576_N14_s1x7_outL1"])
def test_pcc(device, name):
    c, cache, w, tt_cache, tt_w, cfg, ck = build(device, name)
    tt_valid = _uint32_scalar(device, M_CAP)
    out = ttnn.to_torch(_call(c, tt_cache, tt_w, cfg, ck, tt_valid)).float()
    ref = (cache[0, 0].float() @ w.float()).reshape(M_CAP, -1, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    q = assert_quality(ref, out)
    logger.info(f"{name}: PCC={q['pcc']:.6f} rel_rmse={q['relative_rmse']:.4f}")
    assert q["pcc"] > 0.99, f"{name} PCC {q['pcc']}"


def test_tune4_table():
    rows = []
    for name in CONFIGS:
        subdir = f"mm_kv50k4_{name}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_kv50k_tune4.py::test_run_cfg[name={name}]"
        )
        try:
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=False
            )
            cores = int(r["CORE COUNT"].max())
            ns = float(r["DEVICE KERNEL DURATION [ns]"].min())
            c = CONFIGS[name]
            tiles = (M_CAP // 32) * (c["K"] // 32) * (N // 32)
            hifi2_floor = tiles * 32 / 110 / CLOCK_GHZ / 1000
            own = tiles * FID[c["fid"]][1] / cores / CLOCK_GHZ
            rows.append((name, cores, ns / 1000, 100 * own / ns, hifi2_floor))
        except Exception as e:
            rows.append((name, 0, float("nan"), float("nan"), 0))
            logger.error(f"{name} FAILED: {str(e)[:160]}")
    print("\n| config | cores | us | util % (own fid) | HiFi2 floor us | 70% target us |")
    print("|---|---|---|---|---|---|")
    for name, cores, us, util, fl in sorted(rows, key=lambda r: (r[2] != r[2], r[2])):
        print(
            f"| {name} | - | FAILED | - | - | - |"
            if us != us
            else f"| {name} | {cores} | {us:.2f} | {util:.1f} | {fl:.1f} | {fl/0.7:.1f} |"
        )
