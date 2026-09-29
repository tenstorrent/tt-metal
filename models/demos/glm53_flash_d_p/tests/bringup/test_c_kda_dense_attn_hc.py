# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.attn_hc.test.1): the output is [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4]. The
golden holds bf16-rounded values (fp32 reference: PCC 0.999999, rel 0.0013). The three parts have very different
scales (pre in (0, 1], post in (0, 2], comb in [0, 1]), so a whole-matrix PCC hides bugs in one part. Measured on this
golden (s4096 chunk 1, 2048 rows), PCC / part rel L2 / part max abs: comb transposed (the kernel's output order)
0.9958 / comb 0.125 / 0.18, softmax over the wrong axis 0.9987 / 0.078 / 0.12, 10 Sinkhorn iterations 0.9958 /
0.130 / 0.21, hc_eps 1e-5 0.9996 / 0.042 / 0.15, pre/post scales swapped 0.9958 / pre 0.12 / 0.54, last row zeroed
0.9997 / 0.02 / 2.0: all pass PCC 0.99. Not caught: 19 iterations (comb rel 0.0066), hc_eps 0 (comb rel 0.010,
max abs 0.078), pre without +1e-6. Device noise: bf16, bfp8 or tf32 projection operands give rel <= 0.002 and max
abs <= 0.006; a pessimistic random mix error at PCC 0.9989 (the TT fp32-matmul ceiling quoted in
deepseek_v3_d_p/tests/pcc/test_mhc.py) gives pre rel 0.025 / post 0.020 / comb 0.0064, max abs 0.10 / 0.14 / 0.061.
Extra checks: rel L2 per part, max abs per part, and the Sinkhorn structure (the last step normalizes columns, so
every comb column sums to 1; a transposed comb has column sums 0.92..1.08).
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.03, "post": 0.03, "comb": 0.02}  # ||got - want|| / ||want|| per part
MAX_ABS = {"pre": 0.15, "post": 0.2, "comb": 0.1}  # worst element per part
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
