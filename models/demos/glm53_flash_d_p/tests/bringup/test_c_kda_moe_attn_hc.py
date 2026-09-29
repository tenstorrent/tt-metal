# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.attn_hc.test.1): the same op as dsa_moe attn_hc (hc_attn_fn / base / scale of layer 4) on `in`
[S * 4, H] (token-major). Output [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4], bf16-rounded in the golden
(fp32 reference: part rel 0.0017 / 0.0016 / 0.0015, max abs 2.3e-3 / 1.05e-3 / 2.3e-3, worst column 0.0025). The
layer-3 checks carry over; the limits are re-measured on this golden (s4096 chunk 1, 2048 rows). Unlike layer 3, post
is not saturated everywhere: post stream 1 reaches 0.455 (so layer 3's post max abs 5e-4 fails the fp32 reference),
while post streams 0 / 2 / 3 stay <= 5.3e-4 / 2.1e-3 / 8.8e-6. The streams differ (rel ~1.0 from stream 0), so PCC
catches stream order (streams 0/1 swapped 0.12, stream-major flatten 0.14, ffn_hc weights 0.30), comb transposed
(0.929), comb base transposed (0.931) and pre/post scales swapped (0.898). Every other coefficient bug passes PCC.
Measured PCC / part rel L2 / part max abs / worst column rel L2 / comb column-sum error: softmax over the wrong axis
0.99915 / comb 0.041 / 0.053 / 8.7; 10 Sinkhorn iterations 0.99890 / comb 0.045 / 0.092 / 1.07; 18 iterations comb
0.0051 / 0.0093 / 0.12; 19 iterations comb 0.0027 / 0.0053 / 0.061; hc_eps 1e-5 comb 0.0056 / 0.056 / 8.1; hc_eps 0
worst column 0.99; pre scale x1.02 pre 0.043 / 0.035; post scale x1.02 post 0.041 / 0.011 / 0.23; comb scale x1.02
comb 0.017 / 0.031 / 0.13; pre, post or comb x1.02 0.020 (comb column sums 1.02); last row zeroed pre max abs 0.87,
column sums off by 1. Not caught: rms eps 1e-6 (pre 0.0036, worst column 0.013), rms eps 1.2e-5 (0.0018 / 0.0041), and
x1.005 on any part (0.0052, about the device's own error). Device-like noise: mix rounded to bf16 0.0036 / 0.0040 /
0.0025, worst column 0.020; 0.3% element noise on the mix 0.0066 / 0.0064 / 0.0038, max abs 0.014 / 4.6e-3 / 9.7e-3,
worst column 0.026; 1% noise fails (0.021, worst column 0.12). The existing device module (tt/mhc.py, fp32 HiFi4
projection) scores PCC 0.999994, part rel 0.0047 / 0.0040 / 0.0031, max abs 6.3e-3 / 1.95e-3 (one bf16 ulp at 0.455)
/ 7.8e-3, worst column 0.0245 (column 9, comb [0, 1], entries <= 0.013), column sums 0.9965..1.0007.
Extra checks (asserted, NaN fails each): finite; rel L2 per part <= 0.01; max abs pre 0.02, post 6e-3, comb 0.02;
worst single-column rel L2 <= 0.05 (0.07 at layer 3; tightened so 19 iterations fail, 2x over the device); every comb
column sums to 1 within 0.01; pre in (0, 1], post in [0, 2], comb >= 0.
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
STEP = "attn_hc"
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # ||got - want|| / ||want|| per part
MAX_ABS = {"pre": 0.02, "post": 6e-3, "comb": 0.02}  # worst element per part
MAX_COL_REL_L2 = 0.05  # worst single output column's rel L2 (column 7, post stream 3, is ~1e-5 here)
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

    # Per-part checks (informational metrics, not in the runner's threshold list). Every limit is written
    # `not x <= lim`, so a NaN metric fails.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    fails = []
    for name, sl in PARTS.items():
        a, b = got[:, sl], w[:, sl]
        rel = ((a - b).norm() / b.norm()).item()
        mx = (a - b).abs().max().item()
        metrics.record(f"rel_l2_{STEP}_{name}_L{LAYER:02d}", rel)
        metrics.record(f"max_abs_{STEP}_{name}_L{LAYER:02d}", mx)
        print(f"{name}: rel_l2={rel:.5f} (<= {MAX_REL_L2[name]}) max_abs={mx:.2e} (<= {MAX_ABS[name]})")
        if not rel <= MAX_REL_L2[name]:
            fails.append(f"{name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if not mx <= MAX_ABS[name]:
            fails.append(f"{name} max abs {mx:.3g} > {MAX_ABS[name]}")

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
