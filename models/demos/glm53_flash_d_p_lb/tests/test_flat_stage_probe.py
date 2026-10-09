# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Precision analysis: which device stage of the flat routed expert gives its output gain (+1.25% vs fp32 math on its
own weight bits at GLM layer 4, test_flat_diag.py). One chip, GLM layer 4's real experts (mesh chip 0's 36: global
0..35, as tt/experts_ag.py places them) and their real routed tokens (the component golden's x and routing), the
weights pre-quantized to the bfp4 grid on the host (so device and reference hold the same bits).

Variants (the Python builder FlatExpert, bit-identical to the C++ op, with its MIMO_FL_* stage switches; each set
before the build): base (fp32 full-sync down = the ag path), + bf16 h, + HiFi4 matmuls, no activation (h = raw
gate), and combinations; plus the C++ op (down_fp32, row-major y) as the anchor, without and with
pack_stochastic_rounding (cpp_srnd). Each is scored against fp32 CPU
math on the same weight bits (y = act(x Wg, x Wu) Wd; no-activation variants: y = (x Wg) Wd): scale coefficient and
rel L2 over every routed row. GLM_STAGE_VARIANTS (comma list) selects variants."""

import os

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing.harness import component_golden, spec

S = spec()
LAYER = 4
E, NG, H, I = 36, 288, 4096, 2048
VARIANTS = {
    "base": {"MIMO_FL_DN_ACC": "fp32full"},
    "h_bf16": {"MIMO_FL_DN_ACC": "fp32full", "MIMO_FL_H_BF16": "1"},
    "hifi4": {"MIMO_FL_DN_ACC": "fp32full", "MIMO_FL_FIDELITY": "HiFi4"},
    "h_bf16_hifi4": {"MIMO_FL_DN_ACC": "fp32full", "MIMO_FL_H_BF16": "1", "MIMO_FL_FIDELITY": "HiFi4"},
    "no_act": {"MIMO_FL_DN_ACC": "fp32full", "MIMO_FL_NO_ACT": "1"},
    "no_act_h_bf16": {"MIMO_FL_DN_ACC": "fp32full", "MIMO_FL_NO_ACT": "1", "MIMO_FL_H_BF16": "1"},
    "no_act_h_bf16_hifi4": {
        "MIMO_FL_DN_ACC": "fp32full",
        "MIMO_FL_NO_ACT": "1",
        "MIMO_FL_H_BF16": "1",
        "MIMO_FL_FIDELITY": "HiFi4",
    },
    "dn_bf16": {"MIMO_FL_DN_ACC": "bf16"},
}
SEL = os.environ.get("GLM_STAGE_VARIANTS", "cpp,cpp_srnd," + ",".join(VARIANTS)).split(",")


def _coef(g, w):
    g, w = g.double().reshape(-1), w.double().reshape(-1)
    return float((g * w).sum() / (w * w).sum()), float((g - w).norm() / w.norm())


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_stage_probe(device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatExpert, FlatRoutedExpert

    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PackedExpert

    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, "experts")
    gl = g.layer(c, LAYER)
    x, r = (gl[i].float() for i in st.inputs)
    x = x.reshape(-1, H).bfloat16().float()
    r = r.reshape(x.shape[0], -1)
    loader, _ = hooks._loader_cfg(S)
    q4 = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa

    W = []
    for e in range(E):
        gw, uw, dw = PackedExpert(loader, LAYER, e).weights(torch.float32)
        W.append(tuple(q4(w.T.contiguous().bfloat16().float()) for w in (gw, uw, dw)))  # the flat op's bits
    toks = [torch.nonzero(r[:, e] != 0).reshape(-1) for e in range(E)]
    counts = [len(t) for t in toks]
    m = max(256, -(-max(counts) // 32) * 32)
    offs = [sum(-(-cc // 32) * 32 for cc in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32
    xb = torch.zeros(rows, H)
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e in range(E):
        xb[offs[e] : offs[e] + counts[e]] = x[toks[e]]
        c_[0, e], r_[0, e] = counts[e], offs[e]
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_dev, c_dev, r_dev = rm(xb, ttnn.bfloat16), rm(c_, ttnn.uint32), rm(r_, ttnn.uint32)
    print(f"[stage] layer {LAYER}, {E} experts, {sum(counts)} routed rows, m {m}", flush=True)

    act = lambda g_, u_: torch.nn.functional.silu(g_.clamp(max=10.0)) * u_.clamp(-10.0, 10.0)  # noqa: E731
    want_act, want_lin = [], []
    for e in range(E):
        xe = x[toks[e]]
        Wg, Wu, Wd = W[e]
        g_ = xe @ Wg
        want_act.append(act(g_, xe @ Wu) @ Wd)
        want_lin.append(g_ @ Wd)
    want_act, want_lin = torch.cat(want_act), torch.cat(want_lin)
    pick = lambda y: torch.cat([y[offs[e] : offs[e] + counts[e]] for e in range(E)])  # noqa: E731

    keys = set(k for v in VARIANTS.values() for k in v)
    for name in SEL:
        saved = {k: os.environ.get(k) for k in keys}
        try:
            if name in ("cpp", "cpp_srnd"):
                op = FlatRoutedExpert(
                    device, [W], m=m, H=H, I=I, gids=[list(range(E))], n_global=NG, wdtype="bf4", act="clamped_silu"
                )
                y = op(
                    x_dev, c_dev, r_dev, y_row_major=True, down_fp32=True, pack_stochastic_rounding=name == "cpp_srnd"
                )
                y = ttnn.to_torch(y).float()
                lin = False
            else:
                for k in keys:
                    os.environ.pop(k, None)
                os.environ.update(VARIANTS[name])
                lin = "MIMO_FL_NO_ACT" in VARIANTS[name]
                op = FlatExpert(
                    device,
                    [W],
                    m=m,
                    H=H,
                    I=I,
                    gids=[list(range(E))],
                    n_global=NG,
                    wdtype="bf4",
                    act="gate_only" if lin else "clamped_silu",
                )
                y = ttnn.to_torch(op(x_dev, c_dev, r_dev)).float()
            cf, rl = _coef(pick(y), want_lin if lin else want_act)
            print(f"[stage] {name:22s} coef {cf:.5f} rel {rl:.5f}", flush=True)
            del op
        except Exception as ex:  # keep going: a variant may not fit this shape
            print(f"[stage] {name:22s} FAILED: {type(ex).__name__}: {str(ex).splitlines()[0][:200]}", flush=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
