# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Probe: the flat routed expert's output scale at GLM shapes (H 4096, I 2048, 36 local experts) on one chip, vs an
fp32 reference on the same quantized weights (scale coefficient <got, want> / <want, want>, rel L2, PCC).
GLM_PROBE_STD (weight std, default 0.02), GLM_PROBE_XSTD (x std, default 0.3)."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, act_ref

import ttnn

H, I, E, NG, M = 4096, 2048, 36, 288, 2048


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", ["bf4", "bf8"])
@pytest.mark.parametrize("act", ["clamped_silu", "silu"])
@pytest.mark.parametrize("down_fp32", [False, True], ids=["dn_bf16", "dn_fp32"])
def test_flat_scale_probe(device, wdtype, act, down_fp32):
    torch.manual_seed(0)
    std, xstd = float(os.environ.get("GLM_PROBE_STD", "0.02")), float(os.environ.get("GLM_PROBE_XSTD", "0.3"))
    dt = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b}[wdtype]
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
    weights = [[(torch.randn(H, I) * std, torch.randn(H, I) * std, torch.randn(I, H) * std) for _ in range(E)]]
    if os.environ.get("GLM_PROBE_PREQ") == "1":  # weights already on the dtype's grid: the host quantizer is exact
        weights = [[tuple(q(w_) for w_ in ws) for ws in weights[0]]]
    gids = [list(range(E))]
    op = FlatRoutedExpert(device, weights, m=M, H=H, I=I, gids=gids, n_global=NG, wdtype=wdtype, act=act, pin=1)
    counts = [200 + 37 * (e % 7) for e in range(E)]
    offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32
    x = torch.zeros(rows, H)
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e in range(E):
        x[offs[e] : offs[e] + counts[e]] = torch.randn(counts[e], H) * xstd
        c_[0, e], r_[0, e] = counts[e], offs[e]
    x = x.bfloat16().float()
    rm = lambda t, d: ttnn.from_torch(
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = ttnn.to_torch(
        op(rm(x, ttnn.bfloat16), rm(c_, ttnn.uint32), rm(r_, ttnn.uint32), y_row_major=True, down_fp32=down_fp32)
    ).float()
    got, want = [], []
    for e in range(0, E, 5):
        o, c = offs[e], counts[e]
        Wg, Wu, Wd = (q(w_) for w_ in weights[0][e])
        xe = x[o : o + c]
        want.append(act_ref(xe @ Wg, xe @ Wu, act) @ Wd)
        got.append(y[o : o + c])
    g, w = torch.cat(got).double(), torch.cat(want).double()
    coef = float((g * w).sum() / (w * w).sum())
    rel = float((g - w).norm() / w.norm())
    gc, wc = g - g.mean(), w - w.mean()
    pcc = float((gc * wc).sum() / (gc.norm() * wc.norm()))
    logger.info(
        f"PROBE preq={os.environ.get('GLM_PROBE_PREQ', '0')} {wdtype} {act} down_fp32={down_fp32} std {std} xstd {xstd}: coef {coef:.5f} rel {rel:.5f} pcc {pcc:.6f}"
    )
