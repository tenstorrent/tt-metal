# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_collapse of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.ffn_collapse.test.1): the weightless mHC collapse, ffn_in [S, H] = sum_n pre[:, n] * x[:, n],
x = `h_mid` [S * 4, H] token-major (stream n of token s at row 4s + n), pre = ffn_hc[:, 0:4]. At layer 3 the streams
differ (rel ~1.0 from stream 0) and pre spans 3e-6..1.0 (row sums 1.01..1.47), so stream and pre order are caught by
PCC here (no second layer needed). Measured on this golden (s4096 chunk 1, 2048 rows, row RMS 0.0016..0.0103),
PCC / rel L2 / worst row rel L2 / per-token norm ratio: pre reversed 0.32, pre 0/1 swapped 0.36, stream-major rows
0.03, last stream dropped 0.45, post instead of pre 0.26, unweighted mean 0.39, rows shifted by one 0.34, attn_hc's
pre 0.955, `in` instead of h_mid 0.83. Bugs that pass PCC 0.99: pre normalized to sum 1 rel 0.103 / ratio [0.68, 0.99];
output x1.01 0.0106 / 0.0124 / [1.008, 1.012]; x1.005 0.0059 / 0.0076 / [1.003, 1.007] (coefficient 1.0053); pre col
0 x1.01 0.0051 / 0.0100; pre col 1 x1.01 worst row 0.012; pre col 3 x1.01 0.0096 / [0.998, 1.011]; last row zero
0.0116 / 1.0; last 32 rows zero 0.115; last 32 columns zero 0.090 / 0.124; one row's pre reversed 0.25 / 17.7; one
row duplicated from its neighbour 0.013 / 1.02.
Noise: fp32 reference vs the bf16 golden rel 0.0025 / row 0.0036 / [0.9980, 1.0022] / coefficient 1.0003; bf16
products and bf16 sums 0.0031..0.0033 / 0.0049 / [0.9964, 1.0023] / 1.0002; 0.3% element noise 0.0039 / 0.0047.
Chunk 0 of layer 3 is the same (fp32 0.0026 / 0.0035, bf16 0.0032 / 0.0049 / [0.9966, 1.0022]).
Extra checks (asserted, NaN fails every one): finite; rel L2 <= 0.008; worst per-token rel L2 <= 0.008 (one bad or
zeroed row, pre col 0 x1.01); per-token norm ratio in [0.99, 1.01] (x1.01, a pre column x1.01); coefficient
<got, want> / <want, want> in [0.996, 1.004] (x1.005); all on chunk 1 and chunk 0 of layer 3.
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
STEP = "ffn_collapse"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want||
MAX_ROW_REL_L2 = 0.008  # worst per-token ||got - want|| / ||want||
RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
COEF = (0.996, 1.004)  # <got, want> / <want, want>


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    if out.numel() != want.numel():
        return [f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"]
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - w).norm() / w.norm()).item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    coef = ((got * w).sum() / (w * w).sum()).item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"row_rel_l2_max_{STEP}_{tag}", row)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    metrics.record(f"coef_{STEP}_{tag}", coef)
    print(
        f"{tag}: rel_l2={rel:.5f} (<= {MAX_REL_L2}) worst row {row:.5f} (<= {MAX_ROW_REL_L2}) "
        f"norm ratio [{lo:.4f}, {hi:.4f}] (in {RATIO}) coef {coef:.5f} (in {COEF})"
    )
    fails = []
    if not rel <= MAX_REL_L2:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {MAX_REL_L2}")
    if not row <= MAX_ROW_REL_L2:
        fails.append(f"{tag}: worst per-token rel L2 {row:.4f} > {MAX_ROW_REL_L2}")
    if not (RATIO[0] <= lo and hi <= RATIO[1]):
        fails.append(f"{tag}: per-token norm ratio [{lo:.4f}, {hi:.4f}] outside {RATIO}")
    if not (COEF[0] <= coef <= COEF[1]):
        fails.append(f"{tag}: coefficient {coef:.5f} outside {COEF}")
    return fails


def _run(fn, ref, g, chunk):
    gl = g.layer(chunk, LAYER)
    st = _step(ref, LAYER, STEP)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    return fn(reference_ctx(ref, LAYER, g, chunk), device_ctx(LAYER, g, chunk), *inputs), gl[st.output]


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out, want = _run(fn, ref, g, c)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    fails = _checks(f"L{LAYER:02d}", out, want)

    # Same module on the layer's other dumped chunk (start 0).
    if c != 0:
        out0, want0 = _run(fn, ref, g, 0)
        _, ok0 = compare(f"pcc_{STEP}_L{LAYER:02d}_c0", out0, want0, mode, thr)
        if not ok0:
            fails.append(f"chunk 0: PCC below {thr}")
        fails += _checks(f"L{LAYER:02d}_c0", out0, want0)
    assert not fails, "; ".join(fails)
