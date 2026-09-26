# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_moe.router.test.1): same router as sliding_moe (HF MiMoV2MoEGate, noaux_tc, n_group 1, top-8, sigmoid
scores, e_score_correction_bias for selection only, renormalized, routed_scaling_factor 1.0), so the checks are those of
test_c_sliding_moe_router.py. Golden [2048, 256] bf16, exactly 8 nonzeros per row, row sums [0.9976, 1.0024].
Layer 5 bias is 0.72..1.09 (std 0.067); choice score up to 1.83; 8th-9th choice gap median 0.0028 (679 of 2048 rows
< 1e-3). Measured on this golden as PCC / top-8 selection overlap / matched-row weight rel L2 / row sums:
  CPU reference, fp32 on the bf16 golden input: 0.99819 / 0.99573 / 0.0015 / 1.0; bf16 output 0.99819 / 0.99573 /
  0.0011 / [0.9976, 1.0023];
  logits rounded to bf16 0.99770 / 0.99390; logit noise 0.3% of std 0.99356 / 0.98499; 1% 0.98293 (fails) / 0.953;
  correction bias rounded to bf16 0.99369 (passes PCC!) / 0.95190 (fails overlap); all-bf16 pipeline 0.980 / 0.881;
  weights x1.02 0.99819 (passes) / - / 0.0201 / 1.02; logits x1.01 0.99826 / 0.98755 / 0.0032; logits x1.1 0.99259
  (passes) / 0.90179; weights from scores+bias 0.7425 / - / 0.667; no renormalize 0.7325 / row sums 0.09-7.9;
  no bias in selection 0.813 / 0.805; half bias 0.839 / 0.835; top-7 0.978 / nnz 7; top-9 0.982 (overlap 0.9998) / nnz 9;
  last row zeroed 0.99806 (passes) / nnz 0; last 32 rows zeroed 0.99118 (passes); bias recentred by its mean: identical
  to the reference.
The gated metric is PCC. It misses a bf16 correction bias, a weight scale error, a few zeroed rows and top-k count errors,
so the test also asserts exactly 8 nonzeros per row, non-negative weights, mean selection overlap >= 0.985 (the layer-1
device router scored 0.9984 against a CPU 0.9988; bf16 bias 0.952, logits x1.1 0.902), matched-row weight rel L2
<= 0.005 and every row sum within 0.01 of 1.
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
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
TOP_K = 8
MIN_SELECTION_OVERLAP = 0.985  # mean per-row |selected & golden selected| / 8
MAX_MATCHED_REL_L2 = 0.005  # weights, rows whose selected set equals the golden's
MAX_ROW_SUM_ERR = 0.01  # |sum(row) - 1|: norm_topk_prob with routed_scaling_factor 1.0


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

    # Selection, weight and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    gs, ws = got != 0, w != 0
    nnz = gs.sum(-1)
    overlap = ((gs & ws).sum(-1).float() / ws.sum(-1)).mean().item()
    m = (gs == ws).all(-1)
    mrel = ((got[m] - w[m]).norm() / w[m].norm()).item() if m.any() else float("inf")
    rsum = got.sum(-1)
    rmin, rmax = rsum.min().item(), rsum.max().item()
    metrics.record(f"selection_overlap_{STEP}_L{LAYER:02d}", overlap)
    metrics.record(f"matched_rel_l2_{STEP}_L{LAYER:02d}", mrel)
    metrics.record(f"row_sum_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_sum_max_{STEP}_L{LAYER:02d}", rmax)
    print(
        f"nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K}) selection_overlap={overlap:.5f} (>= {MIN_SELECTION_OVERLAP}) "
        f"matched_rows={m.sum().item()}/{m.numel()} matched_rel_l2={mrel:.5f} (<= {MAX_MATCHED_REL_L2}) "
        f"row_sum=[{rmin:.4f}, {rmax:.4f}] (1 +- {MAX_ROW_SUM_ERR})"
    )
    assert (nnz == TOP_K).all(), f"nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want exactly {TOP_K}"
    assert (got >= 0).all(), "negative routing weight"
    assert (
        overlap >= MIN_SELECTION_OVERLAP
    ), f"top-{TOP_K} selection overlap {overlap:.5f} < {MIN_SELECTION_OVERLAP} (correction bias or choice-score precision?)"
    assert (
        mrel <= MAX_MATCHED_REL_L2
    ), f"matched-row weight rel L2 {mrel:.5f} > {MAX_MATCHED_REL_L2} (weights from biased scores, scale or renorm bug)"
    assert (
        1 - MAX_ROW_SUM_ERR <= rmin and rmax <= 1 + MAX_ROW_SUM_ERR
    ), f"per-token routing sum in [{rmin:.4f}, {rmax:.4f}], want 1 +- {MAX_ROW_SUM_ERR} (renormalization or scale bug)"
