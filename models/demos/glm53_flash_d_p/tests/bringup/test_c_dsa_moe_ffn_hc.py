# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.ffn_hc.test.1): the same op as attn_hc (hc_ffn_fn / base / scale of layer 3) on h_mid [S * 4, H]
(token-major). Output [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4], bf16-rounded in the golden (fp32
reference: part rel 0.0009 / 0.0017 / 0.0012, worst column 0.0035; golden comb column sums 0.9974..1.0027). The
h_mid streams differ (rel ~1.0 from stream 0), so PCC catches stream order (streams 0/1 swapped 0.597, stream-major
flatten 0.593, attn_hc weights 0.922). Unlike layer-3 attn_hc, post is not saturated everywhere: column 4 reaches
0.40, while columns 5..7 stay <= 0.009 (column 6 <= 0.0024). Flattened rows have RMS 0.013..0.050. Measured on this
golden (s4096 chunk 1, 2048 rows), PCC / part rel L2 (pre/post/comb) / part max abs / worst column rel L2, all passing
PCC 0.99: comb transposed 0.9945 / comb 0.103 / 0.11 / 1823 / column sums 0.92..1.08, softmax over the wrong axis
0.9981 / 0.073 / 0.082 / 5.2, 10 Sinkhorn iterations 0.9967 / 0.090 / 0.13 / 2.7, 18 iterations 0.0108 / 0.015 /
0.28, 19 iterations 0.0052 / 0.0081 / 0.147, 21 iterations 0.0048 / 0.161, hc_eps 1e-5 comb 0.0089 / 0.042 / 8.0,
hc_eps 0 worst column 1.0, comb base transposed 0.0065 / 0.011 / 0.95, rms eps 1e-6 post 0.0144 / 0.126, rms eps 2e-5
post 0.016 / 0.119, pre/post scales swapped 0.9991 / post 0.46, pre scale x1.02 pre 0.0081 / 0.037, post scale x1.02
post 0.044 / 0.119 / coefficient 0.957, comb scale x1.02 comb 0.0092 / 0.038 / 0.44, pre x1.02 pre 0.0196 / 0.022,
post x1.02 0.0201, comb x1.02 0.0200 / column sums 1.02, pre / post / comb x1.01 coefficient 1.0096 / 1.0100 /
1.0100 (rel 0.0096 / 0.0102 / ~0.010), post column 6 := column 7 worst column 8.2, last row zeroed max abs 1.0.
Mix truncated to bf16 (not rounded): post coefficient 1.0059. Not caught: rms eps 1.2e-5 (post 0.0037, worst column
0.026, coefficient 1.0028), pre without +hc_eps, x1.005 on one part (coefficient 1.0046..1.0051, at the limit).
Device-like noise: mix rounded to bf16 0.0010 / 0.0040 / 0.0013, max abs 3.7e-3 / 1.3e-3 / 5.9e-3, worst column 0.016;
0.3% element noise on the mix 0.0014 / 0.0072 / 0.0014, max abs 7.4e-3 / 2.9e-3 / 6.8e-3, worst column 0.032; 1%
noise fails (post 0.023, worst column 0.108). Coefficients under noise stay within 0.0006 of 1 (pre 0.9996 in the fp32
reference: bf16 rounding of pre entries just below 1.0 biases the golden up).
Extra checks (asserted, NaN fails): finite; rel L2 per part <= 0.01; max abs pre 0.02, post 5e-3, comb 0.02; worst
single-column rel L2 <= 0.07; per-part coefficient <got, want> / <want, want> in [0.995, 1.005]; every comb column
sums to 1 within 0.01; pre in (0, 1], post in [0, 2], comb >= 0.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "ffn_hc"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # ||got - want|| / ||want|| per part
MAX_ABS = {"pre": 0.02, "post": 5e-3, "comb": 0.02}  # worst element per part
MAX_COL_REL_L2 = 0.07  # worst single output column's rel L2 (post columns 5..7 are <= 0.009 here)
COEF = (0.995, 1.005)  # <got, want> / <want, want> per part
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Per-part checks (informational metrics, not in the runner's threshold list).
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    fails = []
    for name, sl in PARTS.items():
        a, b = got[:, sl], w[:, sl]
        rel = ((a - b).norm() / b.norm()).item()
        mx = (a - b).abs().max().item()
        coef = ((a * b).sum() / (b * b).sum()).item()
        metrics.record(f"rel_l2_{STEP}_{name}_L{LAYER:02d}", rel)
        metrics.record(f"max_abs_{STEP}_{name}_L{LAYER:02d}", mx)
        metrics.record(f"coef_{STEP}_{name}_L{LAYER:02d}", coef)
        print(
            f"{name}: rel_l2={rel:.5f} (<= {MAX_REL_L2[name]}) max_abs={mx:.2e} (<= {MAX_ABS[name]}) "
            f"coef={coef:.5f} (in {COEF})"
        )
        if not rel <= MAX_REL_L2[name]:
            fails.append(f"{name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if not mx <= MAX_ABS[name]:
            fails.append(f"{name} max abs {mx:.3g} > {MAX_ABS[name]}")
        if not COEF[0] <= coef <= COEF[1]:
            fails.append(f"{name} coefficient {coef:.5f} outside {COEF}")

    col_rel = (got - w).norm(dim=0) / w.norm(dim=0)
    worst = col_rel.max().item()
    metrics.record(f"worst_col_rel_l2_{STEP}_L{LAYER:02d}", worst)
    print(f"per-column rel L2: {[round(v, 4) for v in col_rel.tolist()]}")
    print(f"worst column rel L2 {worst:.4f} at column {col_rel.argmax().item()} (<= {MAX_COL_REL_L2})")
    if not worst <= MAX_COL_REL_L2:
        fails.append(f"column {col_rel.argmax().item()} rel L2 {worst:.4f} > {MAX_COL_REL_L2}")

    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record(f"comb_col_sum_err_{STEP}_L{LAYER:02d}", col_err)
    print(f"comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if not col_err <= COL_SUM_TOL:
        fails.append(f"comb column sums off 1 by {col_err:.4f} (transposed comb or missing final column step)")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append("negative comb entries")
    assert not fails, "; ".join(fails)
