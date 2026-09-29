# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.router.test.1): same router as dsa_moe (test_c_dsa_moe_router.py: HF Glm5NextTextTopkRouter,
noaux_tc, one group, top-8 of 288, sigmoid of fp32 logits, bias for selection only, renormalized weights x 2.5), with
layer 4 weights. Golden [2048, 288] bf16, exactly 8 nonzeros per row, row sums [2.4941, 2.5063] (chunk 0: [2.4946,
2.5054]), weights 0.032..1.77.
Layer 4 bias is 5.26..5.54 (std 0.044, layer 3: 7.42..7.79): the choice score reaches 6.53. The median 8th-9th choice
gap is 0.0040 (layer 3: 0.0015); 327 of 2048 rows are < 1e-3 and 143 are >= 0.02 (chunk 0: 329 / 154).
Measured on this golden (chunk 1; chunk 0 within 0.001 unless given) as PCC / mean selection overlap / worst-row
overlap / matched-row weight rel L2 / matched-row coefficient / row sums:
  CPU reference, fp32 on the bf16 golden input: 0.99980 / 0.99915 / 0.875 / 0.0017 / 1.00002 / 2.5; bf16 output
  0.99980 / 0.99915 / 0.875 / 0.0014 / 1.00001 / [2.4941, 2.5059];
  logits rounded to bf16 0.99910 / 0.99591; logit noise 0.3% of std 0.99872 / 0.99438; 1% 0.99479 / 0.98035 (fails
  overlap); choice score truncated to TF32 0.99593 (passes PCC) / 0.97998 (chunk 0 0.97919; passes layer 3's 0.985
  limit only by 0.005, fails 0.99); choice score in bf16 0.97181 / 0.86786; correction bias in bf16 0.98522 /
  0.93945; bias recentred by its mean identical to the reference; recentred + TF32 choice 0.99985 / 0.99915;
  recentred + bf16 choice 0.99867 / 0.99426 (chunk 0 0.99506, passes); no routed scale 2.5 PCC 0.99980 / coef 0.400
  / sums 1.0; scale x1.01 mrel 0.0102 / coef 1.010; scale x1.004 mrel 0.0044 (passes) / coef 1.00402 (fails) / sums
  2.51; scale x0.998 coef 0.99802 (passes, as at layer 3); logits x1.01 0.99964 / mrel 0.0067 / coef 1.0036; logits
  x1.1 0.99567 / 0.99139 / mrel 0.064; weights from biased scores 0.834; no renormalize 0.971 / sums 1.96..10.6; no
  bias in selection 0.795 / 0.469; half bias 0.881; softmax instead of sigmoid 0.784; top-7 0.98723 / nnz 7; top-9
  0.98831 / nnz 9; last row zeroed 0.99960 / nnz 0; last 32 rows zeroed 0.99222 (passes PCC); one row copied from its
  neighbour 0.99936 / 0.99866 (passes both) / worst row 0.0; last row's experts rolled worst row 0.125 (chunk 0 0.0);
  one row x1.02 only in its row sum (2.548); all x1.02 mrel 0.0201; rows shifted by one 0.075.
The gated metric is PCC (pcc_router_L04, chunk 1). It misses a TF32 choice score, a missing or wrong routed scale,
row bugs and a copied row, so the test also asserts, on chunk 1 and on chunk 0 (start 0): finite, shape, exactly 8
nonzeros per row, non-negative weights, mean selection overlap >= 0.99 (layer 3: 0.985; the reference is at 0.99915
here, TF32 and 1% logit noise sit at 0.980), worst per-row overlap >= 0.5, matched-row weight rel L2 <= 0.005,
matched-row coefficient in [0.998, 1.002], and every row sum within 0.015 of 2.5. Limits are written ``not x >= lim``
so NaN fails.
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
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
TOP_K = 8
ROUTE_SCALE = 2.5  # routed_scaling_factor, applied after norm_topk_prob
MIN_SELECTION_OVERLAP = 0.99  # mean per-row |selected & golden selected| / 8 (reference 0.99915, TF32 choice 0.980)
MIN_ROW_OVERLAP = 0.5  # worst row (reference 7/8, 1% logit noise 6/8, a copied row 0/8)
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
            "(choice score precision: the bias is ~5.5, keep sigmoid + bias in fp32 or recentre the bias)"
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
