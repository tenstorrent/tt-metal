# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.ffn_hc.test.1): the same op as attn_hc (hc_ffn_fn / base / scale) on h_mid [S * 4, H]
(token-major). Output [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4], bf16-rounded in the golden (fp32
reference: part rel 0.0013 / 0.0017 / 0.0014, comb column sums of the golden 0.9971..1.0029). Unlike `in`, the h_mid
streams differ at layer 0 (rel 0.60 / 1.47 / 1.35 from stream 0), so stream order is visible here (streams 0/1
swapped: PCC 0.912; stream-major flatten 0.548; attn_hc weights 0.547; rms eps 1e-6 0.936: all caught by PCC). The
flattened h_mid rows have RMS 0.0025..0.013, so the RMS eps (1e-5) matters. Measured on this golden (s4096 chunk 1,
2048 rows), PCC / part rel L2 / part max abs, all passing PCC 0.99: comb transposed 0.9945 / comb 0.113 / 0.26,
softmax over the wrong axis 0.9996 / 0.030 / 0.052, 10 Sinkhorn iterations 0.9988 / 0.054 / 0.17, hc_eps 1e-5
0.99995 / 0.0107 / 0.078, comb base transposed 0.99985 / 0.018 / 0.064, comb scale x1.02 0.9993 / 0.042, rms eps
1.2e-5 0.9979 / pre 0.045, pre scale x1.02 0.9998 / pre 0.031, post scale x1.02 0.99996 / post 0.0146, pre x1.02
pre 0.020, post x1.02 post 0.020, comb x1.02 comb 0.020 / column sums 1.02, last row zeroed 0.9996 / 0.02 / 0.99.
Not caught: 18..21 iterations (comb rel <= 0.006), hc_eps 0 (0.0023), pre without +1e-6. Device-like noise: mix
output rounded to bf16 rel 0.0026 / 0.0021 / 0.0024, max abs 0.007; 0.3% element noise on the mix 0.0049 / 0.0028 /
0.0038, max abs 0.023; 1% noise 0.016 / 0.0073 / 0.0117, max abs 0.070 (fails). The device attn_hc module scores
<= 0.0016 / 0.005 on attn_hc. The attn_hc test's pessimistic PCC-0.9989 mix noise gives 0.033 / 0.073 / 0.075 here,
far above what the fp32 HiFi4 projection does, so the limits are tighter than attn_hc's.
Extra checks (asserted): finite; rel L2 per part <= 0.01, max abs per part <= 0.05, every comb column sums to 1
within 0.01 (the last Sinkhorn step normalizes columns; transposed comb 0.93..1.05), pre in (0, 1], post in [0, 2],
comb >= 0.
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # ||got - want|| / ||want|| per part
MAX_ABS = {"pre": 0.05, "post": 0.05, "comb": 0.05}  # worst element per part
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
            fails.append(f"{name} max abs {mx:.3f} > {MAX_ABS[name]}")

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
