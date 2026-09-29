# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_collapse of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.attn_collapse.test.1): the same weightless op as dsa_moe attn_collapse: attn_in [S, H] =
sum_n pre[:, n] * x[:, n], x = `in` [S * 4, H] token-major (stream n of token s at row 4s + n), pre = attn_hc[:, 0:4].
At layer 4 the streams differ (rel 0.99..1.00 from stream 0) and pre spans 8.4e-5..0.98, so PCC catches stream and pre
order (no second layer needed, unlike layer 0). Measured on this golden (s4096 chunk 1, 2048 rows, row RMS
0.0016..0.030), PCC / rel L2 / per-token norm ratio / worst row rel L2: pre reversed 0.754 / 9.2, pre 0/1 swapped
0.760 / 5.2, stream-major rows 0.062 / 6.6, last stream dropped 0.956 / 0.31, post instead of pre 0.52, unweighted
mean 0.78, pre normalized to sum 1 0.971, one row's pre reversed 0.971 / 0.25 / ratio max 22.5 (all fail PCC);
bugs that pass PCC 0.99: output x1.005 0.0058 / [1.0016, 1.0084] / 0.0087, x1.01 0.0105 / [1.0066, 1.0134] / 0.0136,
x1.02 0.020, pre col 0 x1.01 0.0084 / [0.9997, 1.0126] / 0.0130, col 1 x1.01 0.0035 / row 0.0091, col 3 x1.01 0.0041
/ row 0.0098 (col 2 x1.01 is invisible: 0.0027 / row 0.0042), last 32 rows zero 0.098 / ratio min 0, last row zero
0.0103 / ratio min 0, last 32 columns zero 0.089 / worst row 0.12.
Noise: fp32 reference vs the bf16 golden 0.0027 / [0.9966, 1.0033] / 0.0042; bf16 products accumulated in bf16
0.0040 / [0.9965, 1.0035] / 0.0051; 0.3% element noise 0.0040 / [0.9967, 1.0035] / 0.0052.
Extra checks (asserted, written `not x <= lim` so NaN fails): finite; rel L2 <= 0.01; per-token norm ratio in
[0.99, 1.01] (catches x1.01); worst per-token rel L2 <= 0.008 (catches x1.005, a zeroed row or columns, pre col 0 / 1 / 3
x1.01).
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
STEP = "attn_collapse"
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.008  # worst per-token ||got - want|| / ||want||


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

    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    tag = f"L{LAYER:02d}"
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    metrics.record(f"row_rel_l2_max_{STEP}_{tag}", row)
    print(
        f"{tag}: rel_l2={rel:.5f} (<= {MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}] (in {RATIO}) "
        f"worst row rel_l2={row:.5f} (<= {MAX_ROW_REL_L2})"
    )
    fails = []
    if not rel <= MAX_REL_L2:
        fails.append(f"rel L2 {rel:.4f} > {MAX_REL_L2}")
    if not (RATIO[0] <= lo and hi <= RATIO[1]):
        fails.append(f"per-token norm ratio [{lo:.4f}, {hi:.4f}] outside {RATIO}")
    if not row <= MAX_ROW_REL_L2:
        fails.append(f"worst per-token rel L2 {row:.4f} > {MAX_ROW_REL_L2}")
    assert not fails, "; ".join(fails)
