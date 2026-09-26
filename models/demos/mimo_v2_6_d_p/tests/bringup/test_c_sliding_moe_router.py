# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type sliding_moe (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding_moe.router.test.1): router = dense [S, 256] routing matrix from ffn_norm (HF MiMoV2MoEGate,
noaux_tc, n_group 1): logits = x @ gate.weight^T in fp32 -> sigmoid scores -> top-8 on scores + e_score_correction_bias
-> weights are the unbiased sigmoid scores at those ids, renormalized to sum 1 (routed_scaling_factor null = 1.0),
scattered into zeros. Golden [2048, 256] bf16, exactly 8 nonzeros per row, row sums [0.998, 1.002].
Selection is fragile: the bias is 1.72..2.18 (std 0.04), so the choice score sits near 2 where the bf16 step is 0.0078,
and the 8th-9th choice gap has median 0.0017 (701 of 2048 rows < 1e-3). Measured on this golden as PCC / top-8 selection
overlap / matched-row weight rel L2 / row sums:
  CPU reference, fp32 on the bf16 golden input: 0.99933 / 0.99878 (20 rows differ, near ties) / 0.0016 / 1.0;
  same, bf16 output 0.99934 / 0.99878 / 0.0010 / [0.9976, 1.0020];
  logits rounded to bf16 0.99433 / 0.98975 / 0.0017; logit noise 0.3% of std 0.99441 / 0.98962; 1% 0.9810 (fails);
  correction bias rounded to bf16 0.9700 (fails) / 0.944; all-bf16 pipeline 0.910 / 0.830;
  weights x1.02 0.99956 (passes) / - / 0.0200 / 1.02; logits x1.01 0.99928 / 0.99872 / 0.0040; logits x1.1 0.99647
  (passes) / 0.99445 / 0.0366; weights from scores+bias 0.9594 / - / 0.275; no renormalize 0.9869 / row sums 2.1-4.0;
  no bias in selection 0.949 / 0.906; half bias 0.973 / 0.950; top-7 0.967 / nnz 7; top-9 0.970 / nnz 9;
  last row zeroed 0.99907 (passes) / nnz 0; last 32 rows zeroed 0.99088 (passes); softmax scoring 0.32; ids shifted 0.008.
The gated metric is PCC. It misses a weight scale error, a few zeroed rows and top-k count errors, so the test also asserts
exactly 8 nonzeros per row, non-negative weights, mean selection overlap >= 0.985 (between bf16-logit noise 0.990 and
half bias 0.950; the PCC gate alone already implies about 0.98), matched-row weight rel L2 <= 0.005 and every row sum
within 0.01 of 1 (the model renormalizes and scales by 1.0). The device needs fp32 logits and an fp32 (or
bias-recentred) choice score: bf16 rounding of the bias or of the choice score fails the PCC gate.
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
LAYER = 1
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
