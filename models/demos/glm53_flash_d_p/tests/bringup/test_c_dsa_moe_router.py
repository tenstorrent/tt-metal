# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.router.test.1): HF Glm5NextTextTopkRouter (noaux_tc, n_group 1, top-8 of 288, sigmoid of fp32
logits, e_score_correction_bias for selection only, weights from the unbiased sigmoid, renormalized, then x
routed_scaling_factor 2.5, so every row sums to 2.5, not 1 as in MiMo). Golden [2048, 288] bf16, exactly 8 nonzeros
per row, row sums [2.4932, 2.5068], weights 0.029..2.14.
Layer 3 bias is 7.42..7.79 (std 0.048): the choice score reaches 8.77, where the bf16 step is 0.0625 and the TF32 step
0.0078, against a median 8th-9th choice gap of 0.0015 (822 of 2048 rows < 1e-3; chunk 0: 0.0017, 736 rows).
Measured on this golden (chunk 1; chunk 0 within 0.002) as PCC / mean selection overlap / worst-row overlap / matched-row
weight rel L2 / matched-row coefficient ``<got, want> / <want, want>`` / row sums:
  CPU reference, fp32 on the bf16 golden input: 0.99982 / 0.99695 / 0.875 / 0.0016 / 0.99995 / 2.5; bf16 output
  0.99982 / 0.99695 / 0.875 / 0.0012 / 1.00003 / [2.4939, 2.5068];
  logits rounded to bf16 0.99938 / 0.99011 / 0.75; logit noise 0.3% of std 0.99947 / 0.99310; 1% 0.99816 (passes PCC) /
  0.97693 (fails overlap); choice score truncated to TF32 (moe_grouped_topk keys) 0.99711 / 0.94727 (fails); choice
  score in bf16 0.98419 / 0.71387; correction bias in bf16 0.99071 (passes PCC!) / 0.87103; bias recentred by its mean
  (fp32) identical to the reference; recentred + TF32 choice 0.99978 / 0.99664; recentred + bf16 choice 0.99953 /
  0.99207 (passes); no routed scale 2.5 0.99982 (passes) / mrel 0.600 / sums 1.0; scale x1.01 0.99982 / mrel 0.0101 /
  coef 1.0097; scale x1.004 0.99982 / mrel 0.0043 (passes) / coef 1.00395 (fails) / sums 2.51; scale x0.998 coef
  0.99795; weights x1.02 mrel 0.0200; logits x1.01 0.99954 / 0.99487 / mrel 0.0081 / coef 1.0058; logits x1.1 0.99620
  / 0.96649; weights from biased scores 0.666; no renormalize 0.953 / sums 1.6..12.2; no bias in selection 0.878 /
  0.369; half bias 0.913; softmax instead of sigmoid 0.863; top-7 0.99595 (passes PCC) / nnz 7; top-9 0.99638 / nnz
  9; last row zeroed 0.99966 / nnz 0; last 32 rows zeroed 0.99289 (passes PCC); one row copied from its neighbour
  0.99951 / mean overlap 0.99646 (passes both) / worst row 0.0 (fails); last row's experts rolled worst row 0.125;
  rows shifted by one 0.059.
A "rows with a clear 8th-9th gap must match exactly" check was measured and dropped: only 34 rows have a gap >= 0.02,
and every bug above except bf16 choice / bias passes it.
The gated metric is PCC (pcc_router_L03, chunk 1). It misses a bf16 correction bias, a missing or wrong routed scale,
a TF32 choice score, top-k count errors and row bugs, so the test also asserts, on chunk 1 and on chunk 0 (start 0):
finite, shape, exactly 8 nonzeros per row, non-negative weights, mean selection overlap >= 0.985 (MiMo's limit), worst
per-row overlap >= 0.5, matched-row weight rel L2 <= 0.005, matched-row coefficient in [0.998, 1.002], and every row
sum within 0.015 of 2.5. Limits are written ``not x >= lim`` so NaN fails.
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
STEP = "router"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
TOP_K = 8
ROUTE_SCALE = 2.5  # routed_scaling_factor, applied after norm_topk_prob
MIN_SELECTION_OVERLAP = 0.985  # mean per-row |selected & golden selected| / 8
MIN_ROW_OVERLAP = 0.5  # worst row (reference 7/8, bf16 logits 6/8, a copied row 0/8)
MAX_MATCHED_REL_L2 = 0.005  # weights, rows whose selected set equals the golden's
MATCHED_COEF = (0.998, 1.002)  # <got, want> / <want, want> on the matched rows (uniform weight scale)
MAX_ROW_SUM_ERR = 0.015  # |sum(row) - 2.5|


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    if out.numel() != want.numel():
        return [f"{tag}: output has {out.numel()} elements, golden {tuple(want.shape)}"]
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    gs, ws = got != 0, w != 0
    nnz = gs.sum(-1)
    row_ov = (gs & ws).sum(-1).float() / ws.sum(-1)
    overlap, worst = row_ov.mean().item(), row_ov.min().item()
    m = (gs == ws).all(-1)
    if m.any():
        mrel = ((got[m] - w[m]).norm() / w[m].norm()).item()
        mcoef = ((got[m] * w[m]).sum() / (w[m] * w[m]).sum()).item()
    else:
        mrel, mcoef = float("inf"), float("nan")
    rsum = got.sum(-1)
    rmin, rmax = rsum.min().item(), rsum.max().item()
    metrics.record(f"selection_overlap_{STEP}_{tag}", overlap)
    metrics.record(f"worst_row_overlap_{STEP}_{tag}", worst)
    metrics.record(f"matched_rel_l2_{STEP}_{tag}", mrel)
    metrics.record(f"matched_coef_{STEP}_{tag}", mcoef)
    metrics.record(f"row_sum_min_{STEP}_{tag}", rmin)
    metrics.record(f"row_sum_max_{STEP}_{tag}", rmax)
    print(
        f"{tag}: nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K}) selection_overlap={overlap:.5f} "
        f"(>= {MIN_SELECTION_OVERLAP}) worst_row={worst:.3f} (>= {MIN_ROW_OVERLAP}) matched_rows={m.sum().item()}/"
        f"{m.numel()} matched_rel_l2={mrel:.5f} (<= {MAX_MATCHED_REL_L2}) matched_coef={mcoef:.5f} (in {MATCHED_COEF}) "
        f"row_sum=[{rmin:.4f}, {rmax:.4f}] ({ROUTE_SCALE} +- {MAX_ROW_SUM_ERR})"
    )
    fails = []
    if not (nnz == TOP_K).all():
        fails.append(f"{tag}: nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want exactly {TOP_K}")
    if not (got >= 0).all():
        fails.append(f"{tag}: negative routing weight")
    if not overlap >= MIN_SELECTION_OVERLAP:
        fails.append(
            f"{tag}: top-{TOP_K} selection overlap {overlap:.5f} < {MIN_SELECTION_OVERLAP} "
            "(choice score precision: the bias is ~7.6, keep sigmoid + bias in fp32 or recentre the bias)"
        )
    if not worst >= MIN_ROW_OVERLAP:
        fails.append(f"{tag}: worst per-row selection overlap {worst:.3f} < {MIN_ROW_OVERLAP} (a row bug)")
    if not mrel <= MAX_MATCHED_REL_L2:
        fails.append(
            f"{tag}: matched-row weight rel L2 {mrel:.5f} > {MAX_MATCHED_REL_L2} (weights from biased scores, "
            "scale or renorm bug)"
        )
    if not (MATCHED_COEF[0] <= mcoef <= MATCHED_COEF[1]):
        fails.append(f"{tag}: matched-row coefficient {mcoef:.5f} outside {MATCHED_COEF} (uniform weight scale)")
    if not (ROUTE_SCALE - MAX_ROW_SUM_ERR <= rmin and rmax <= ROUTE_SCALE + MAX_ROW_SUM_ERR):
        fails.append(
            f"{tag}: per-token routing sum in [{rmin:.4f}, {rmax:.4f}], want {ROUTE_SCALE} +- {MAX_ROW_SUM_ERR} "
            "(renormalization or routed_scaling_factor bug)"
        )
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

    # Selection, weight and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    fails = _checks(f"L{LAYER:02d}", out, want)

    # Same module on the layer's other dumped chunk (start 0).
    if c != 0:
        out0, want0 = _run(fn, ref, g, 0)
        _, ok0 = compare(f"pcc_{STEP}_L{LAYER:02d}_c0", out0, want0, mode, thr)
        if not ok0:
            fails.append(f"chunk 0: PCC below {thr}")
        fails += _checks(f"L{LAYER:02d}_c0", out0, want0)
    assert not fails, "; ".join(fails)
