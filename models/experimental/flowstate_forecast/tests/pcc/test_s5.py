# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""B0 and S5 checks on the path the model runs (D3 item 7, D0 B0 item 3, CONTRACT §6 B0 items 1-5).

test_scan_fused: route K in the model's fused layout (``lti_scan_fused``) on the golden fp32 Bu, every B0 case
(L = 682, 4096, 16384 x2) and captured layer (0, 3, 4, 5). Bounds: whole rel-L2 <= 1e-3 at L <= 4096, <= 1.5e-3 at
L = 16384 (item 1); per-channel p99 <= 1e-2 and every channel <= max(5e-2, 2x the reference's own error against
the float64 truth) (items 2, 4); no inf or NaN (item 5). Control: a bf16-rounded Lbar must fail item 1.
test_s5_block: the model's S5 block (``FlowStateTT.ssm_block``: device B projection, scan, C, gate, + D u) on the
golden bf16 u, both precisions: x against the golden head / tail (item 1's bound), Re(C x) through the fp32
reference C (rel-L2 <= 2e-3, PCC >= 0.99999) and the block output (PCC >= 0.9999 accuracy; the 0.999 module
floor in performance) (item 3). Results go to B0_OUT (default FLOWSTATE_PCC_OUT/b0) as json.
"""

import os

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.ops.lti_scan import op as lti
from models.experimental.flowstate_forecast.reference import weights
from models.experimental.flowstate_forecast.tests.b0.b0_common import B0_CASES, case_inputs, dump
from models.experimental.flowstate_forecast.tests.pcc import metrics
from models.experimental.flowstate_forecast.tests.pcc.test_modules import TILE, up
from models.experimental.flowstate_forecast.tt.model import FlowStateTT

OUT = os.environ.get("B0_OUT", os.path.join(os.environ.get("FLOWSTATE_PCC_OUT", "/tmp/flowstate_pcc"), "b0"))
LAYERS = (0, 3, 4, 5)


def x_bound(L):
    return 1.5e-3 if L > 4096 else 1e-3


def ch_err(a, b):
    """Per state channel relative L2 over (t, b)."""
    return ((a - b).abs().pow(2).sum((0, 1)).sqrt() / b.abs().pow(2).sum((0, 1)).sqrt().clamp(min=1e-30)).double()


def scan_fused(device, Bu, lbar):
    """Bu (L, B, 512) complex64 -> x (L, B, 512) complex64 through lti_scan_fused."""
    L, B, P = Bu.shape
    cat = torch.cat((Bu.real, Bu.imag), -1)
    x = lti.lti_scan_fused(up(device, cat, ttnn.float32), lti.lbar_tiles(device, lbar), L)
    x = ttnn.to_torch(x).float().reshape(L, -(-B // TILE) * TILE, 2 * P)[:, :B]
    return torch.complex(x[..., :P], x[..., P:])


def score_x(x, c):
    x64, ref64 = x.to(torch.complex128), c.ref.to(torch.complex128)
    e, own = ch_err(x64, ref64), ch_err(ref64, c.truth)
    lim = torch.clamp(2 * own, min=5e-2)
    return {
        "rel_l2": ((x64 - ref64).norm() / ref64.norm()).item(),
        "ch_p99": torch.quantile(e, 0.99).item(),
        "ch_max": e.max().item(),
        "ch_over_bound": int((e > lim).sum()),
        "ch_own_over_5e-2": [int(i) for i in (own > 5e-2).nonzero().flatten()],
        "ref_own_ch_max": own.max().item(),
        "nonfinite": int((~torch.isfinite(torch.view_as_real(x))).sum()),
    }


@pytest.mark.parametrize("case", B0_CASES)
def test_scan_fused(device, case):
    res = {}
    for layer in LAYERS:
        c = case_inputs(case, layer)
        m = res[layer] = score_x(scan_fused(device, c.Bu, c.lbar), c)
        print("K fused", case, layer, m, flush=True)
    dump(f"{OUT}/K_fused_{case}.json", res)
    for layer, m in res.items():
        assert m["rel_l2"] <= x_bound(c.L), (layer, m)
        assert m["ch_p99"] <= 1e-2 and m["ch_over_bound"] == 0 and m["nonfinite"] == 0, (layer, m)


def test_scan_fused_control(device):
    """Positive control: the fused path with a bf16-rounded Lbar must fail item 1."""
    c = case_inputs("b0_s1_L4096", 4)
    lb = torch.complex(c.lbar.real.bfloat16().float(), c.lbar.imag.bfloat16().float())
    m = score_x(scan_fused(device, c.Bu, lb), c)
    print("K fused control bf16 Lbar", m)
    dump(f"{OUT}/K_fused_control.json", m)
    assert m["rel_l2"] > x_bound(c.L), m


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("case", B0_CASES)
def test_s5_block(device, case, precision):
    tt = FlowStateTT(weights.load(), device, precision)
    res = {}
    for layer in LAYERS:
        c = case_inputs(case, layer)
        g, L, P = c.g, c.L, 512
        u = g[f"u{layer}"]
        B = u.shape[1]
        consts = tt.constants(c.s, L, 6)[0][layer]
        y, _, x = tt.ssm_block(layer, up(device, u), consts, L, -(-B // TILE) * TILE, 0, keep_x=True)
        xs = ttnn.to_torch(x).float().reshape(L, -1, 2 * P)[:, :B]
        ys = ttnn.to_torch(y).float().reshape(L, -1, P)[:, :B]
        ssm = tt.layers_ref[layer].ssm
        Ccat = torch.cat((ssm.C_tilde_r, -ssm.C_tilde_i), 1).detach().float()  # (512, 1024)
        m = {}
        for part, sl in (("head", slice(0, 64)), ("tail", slice(L - 64, L))):
            xc = torch.complex(xs[sl, :, :P], xs[sl, :, P:]).to(torch.complex128)
            gx = g[f"x_{part}{layer}"].to(torch.complex128)
            recx = xs[sl] @ Ccat.T
            gr = g[f"recx_{part}{layer}"].float()
            m[f"x_rel_l2_{part}"] = ((xc - gx).norm() / gx.norm()).item()
            m[f"recx_rel_l2_{part}"] = ((recx - gr).norm() / gr.norm()).item()
            m[f"recx_pcc_{part}"] = metrics.pearson(gr, recx)
            m[f"block_pcc_{part}"] = metrics.pearson(g[f"block_{part}{layer}"], ys[sl])
        res[layer] = m
        print("S5", precision, case, layer, {k: f"{v:.3e}" for k, v in m.items()}, flush=True)
        for t in (x, y):
            ttnn.deallocate(t)
    dump(f"{OUT}/S5_{precision}_{case}.json", res)
    floor = 0.9999 if precision == "accuracy" else 0.999
    for layer, m in res.items():
        for part in ("head", "tail"):
            assert m[f"x_rel_l2_{part}"] <= x_bound(L), (layer, m)
            assert m[f"recx_rel_l2_{part}"] <= 2e-3 and m[f"recx_pcc_{part}"] >= 0.99999, (layer, m)
            assert m[f"block_pcc_{part}"] >= floor, (layer, m)
