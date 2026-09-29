# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.attn_hc.test.1): the same op as kda_dense attn_hc (hc_attn_fn / base / scale of layer 3) on
`in` [S * 4, H] (token-major). Output [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4], bf16-rounded in the
golden (fp32 reference: part rel 0.0013 / 0.0017 / 0.0012, worst column 0.0035). At layer 3 the streams differ (rel
~1.0 from stream 0), so stream order is caught by PCC (streams 0/1 swapped 0.64, stream-major flatten 0.60, ffn_hc
weights 0.92). But post is nearly saturated at 0 here (every post entry <= 0.024, column 6 ~1e-8), and every
coefficient bug scores PCC >= 0.9967 on this golden (s4096 chunk 1, 2048 rows). Measured PCC / part rel L2 / part max
abs / worst column rel L2: comb transposed 0.99926 / comb 0.037 / 0.092 / 7415, softmax over the wrong axis 0.99967 /
0.028 / 0.063 / 3.5, 10 Sinkhorn iterations 0.99966 / 0.027 / 0.074 / 0.86, 18 iterations 0.99999 / 0.0034 / 0.009 /
0.149, 19 iterations comb 0.0020 / 0.0053 / 0.076, hc_eps 1e-5 comb 0.0018 / 0.015 / 8.9, hc_eps 0 worst column 1.0,
comb base transposed 0.0034 / 0.010 / 0.33, rms eps 1e-6 post 0.026 / 9.4e-4 / 0.086, pre/post scales swapped 0.9967,
pre scale x1.02 pre 0.024 / 0.042, post scale x1.02 post 0.082 / 1.6e-3 / 0.30, comb scale x1.02 comb 0.018 / 0.039 /
0.20, pre x1.02 0.020, post x1.02 0.020 / 4.9e-4, comb x1.02 0.020 / 0.022 / column sums 1.02, last row zeroed pre
max abs 0.91. Not caught: rms eps 1.2e-5 (post 0.0061, worst column 0.021), hc_eps 0 on the part metrics (only the
column check sees it). Device-like noise: mix rounded to bf16 0.0023 / 0.0068 / 0.0015, worst column 0.028; 0.3%
element noise on the mix 0.0040 / 0.0129 / 0.0020, max abs 0.015 / 4.5e-4 / 0.011, worst column 0.049; 1% noise
fails (post 0.044, worst column 0.17). The existing device module (tt/mhc.py, fp32 HiFi4 projection) scores PCC
0.999998, part rel 0.0021 / 0.0030 / 0.0019, max abs 4.9e-3 / 1.4e-4 / 6.8e-3, worst column 0.034 (column 9, a comb
entry <= 3e-4), column sums 0.9969..1.0009.
Extra checks (asserted): finite; rel L2 per part <= 0.01; max abs pre 0.02, post 5e-4, comb 0.02 (post's own scale);
worst single-column rel L2 <= 0.07 (tiny post and comb entries show logit and eps errors that the part norms hide);
every comb column sums to 1 within 0.01; pre in (0, 1], post in [0, 2], comb >= 0.
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
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # ||got - want|| / ||want|| per part
MAX_ABS = {"pre": 0.02, "post": 5e-4, "comb": 0.02}  # worst element per part
MAX_COL_REL_L2 = 0.07  # worst single output column's rel L2 (post columns are ~1e-8..2e-2 here)
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
        metrics.record(f"rel_l2_{STEP}_{name}_L{LAYER:02d}", rel)
        metrics.record(f"max_abs_{STEP}_{name}_L{LAYER:02d}", mx)
        print(f"{name}: rel_l2={rel:.5f} (<= {MAX_REL_L2[name]}) max_abs={mx:.2e} (<= {MAX_ABS[name]})")
        if rel > MAX_REL_L2[name]:
            fails.append(f"{name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if mx > MAX_ABS[name]:
            fails.append(f"{name} max abs {mx:.3g} > {MAX_ABS[name]}")

    col_rel = (got - w).norm(dim=0) / w.norm(dim=0)
    worst = col_rel.max().item()
    metrics.record(f"worst_col_rel_l2_{STEP}_L{LAYER:02d}", worst)
    print(f"per-column rel L2: {[round(v, 4) for v in col_rel.tolist()]}")
    print(f"worst column rel L2 {worst:.4f} at column {col_rel.argmax().item()} (<= {MAX_COL_REL_L2})")
    if worst > MAX_COL_REL_L2:
        fails.append(f"column {col_rel.argmax().item()} rel L2 {worst:.4f} > {MAX_COL_REL_L2}")

    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record(f"comb_col_sum_err_{STEP}_L{LAYER:02d}", col_err)
    print(f"comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if col_err > COL_SUM_TOL:
        fails.append(f"comb column sums off 1 by {col_err:.4f} (transposed comb or missing final column step)")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append("negative comb entries")
    assert not fails, "; ".join(fails)
