# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.ffn_hc.test.1): the same op as attn_hc (hc_ffn_fn / base / scale of layer 4) on h_mid [S * 4, H]
(token-major). Output [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4], bf16-rounded in the golden (fp32
reference: part rel 0.0010 / 0.0017 / 0.0011, max abs 2.1e-3 / 3.7e-3 / 2.5e-3, worst column 0.0036). The layer-3
ffn_hc checks carry over; the limits are re-measured on this golden (s4096 chunk 1, 2048 rows). Unlike layer 3, post
column 4 reaches 1.70 (one bf16 ulp there is 7.8e-3), so layer 3's post max abs 5e-3 has no room for one device ulp;
post columns 5 / 6 / 7 stay <= 0.072 / 0.112 / 2.6e-5, and pre columns 2 / 3 are ~1e-6 (sigmoid ~0 plus hc_eps).
Flattened rows have RMS 0.030..0.102. The streams differ (rel ~1.0 from stream 0), so PCC catches stream order
(streams 0/1 swapped 0.476, stream-major flatten 0.558) and attn_hc weights (0.302). Every other bug passes PCC 0.99.
Measured PCC / part rel L2 / part max abs / coefficient / worst column rel L2 / comb column-sum error: comb
transposed 0.9980 / comb 0.063 / 0.095 / 362 / column sums off 0.074; softmax over the wrong axis 0.9992 / comb 0.047
/ 0.066 / 1.0225 / 1.36; 10 Sinkhorn iterations 0.9990 / comb 0.049 / 0.080 / 0.9765 / 0.82; 18 iterations comb
0.0061 / 0.0104 / 0.9969 / 0.159; 19 iterations 0.0031 / 0.0057 / 0.9985 / 0.081; 21 iterations 0.0029 / 1.0012 /
0.082; hc_eps 1e-5 comb 0.026 max abs / 8.8; hc_eps 0 worst column 1.0; pre without +hc_eps worst column 0.98
(pre columns 2/3 are ~hc_eps here, unlike layer 3); comb base transposed 0.9999 / comb 0.015 / 0.032 / 1.0071 /
1.99; pre/post scales swapped 0.9982 / post 0.47; pre scale x1.02 pre 0.0026 / 9.1e-3 / 1.0010 / 0.087; post scale
x1.02 post 0.019 / 0.015 / 0.9909 / 0.20; comb scale x1.02 comb 0.012 / 0.035 / 0.22; pre / post / comb x1.02 0.020 /
coefficient 1.020 (comb column sums 1.02); x1.01 on any part rel 0.0099..0.0102 / coefficient 1.0098..1.0100 (post
max abs 0.018); post column 6 := column 7 post 0.17 / worst column 1.0; last row zeroed max abs 1.0 / column sums off
by 1. Not caught: rms eps 1e-6 / 1.2e-5 / 2e-5 (post rel <= 0.0025, worst column <= 0.018, coefficient 0.9993..1.0008)
and x1.005 on one part (coefficient 1.0048..1.0050, at the limit). Device-like noise: mix rounded to bf16 0.0010 /
0.0024 / 0.0013, max abs 3.0e-3 / 4.1e-3 / 5.9e-3, worst column 0.019; mix truncated to bf16 post coefficient 1.0014,
worst column 0.041; 0.3% element noise on the mix 0.0011 / 0.0034 / 0.0017, max abs 3.7e-3 / 4.5e-3 / 9.8e-3, worst
column 0.022; 1% noise fails (post rel 0.0100, worst column 0.095). The existing device module (tt/mhc.py, built by
_device_step for any layer) scores PCC 0.999999, part rel 0.0011 / 0.0022 / 0.0019, max abs 2.9e-3 / 4.9e-3 / 6.8e-3,
coefficient 0.9995 / 1.0004 / 0.9990, worst column 0.0347 (column 15, comb [1, 3]), column sums 0.9969..1.0008.
Extra checks (asserted, NaN fails each): finite; rel L2 per part <= 0.01; max abs pre 0.02, post 0.01, comb 0.02;
worst single-column rel L2 <= 0.06 (1.7x over the device, under 19/21 iterations and pre scale x1.02); per-part
coefficient <got, want> / <want, want> in [0.995, 1.005]; every comb column sums to 1 within 0.01; pre in (0, 1],
post in [0, 2], comb >= 0.
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
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # ||got - want|| / ||want|| per part
MAX_ABS = {"pre": 0.02, "post": 0.01, "comb": 0.02}  # worst element per part (post reaches 1.70: ulp 7.8e-3)
MAX_COL_REL_L2 = 0.06  # worst single output column's rel L2 (device 0.035, 19 iterations 0.081)
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
