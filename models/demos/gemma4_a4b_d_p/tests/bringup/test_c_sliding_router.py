# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type sliding (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding.router.test.1): router = dense [S, 128] routing matrix from h_mid: rms (no weight) * router.scale
* 2816^-0.5 -> proj -> softmax -> top-8 -> renormalize -> * per_expert_scale[id], scattered into zeros. The golden is bf16
with exactly 8 nonzeros per row; per_expert_scale is in [0.980, 1.023], so golden row sums are in [0.987, 1.011].
The gated metric is PCC (float output). PCC misses wrong top-k, zeroed rows and the per_expert_scale step, so the test also
asserts exactly 8 nonzeros per row, mean top-8 selection overlap >= 0.995, relative L2 of the weights <= 0.005 on the rows
whose selection matches the golden, and a per-row sum ratio within [0.99, 1.01]. Measured on this golden ([2048, 128]) as
PCC / selection overlap / matched-row rel L2 / row-sum ratio:
  CPU reference (bf16 golden input) 0.999965 / 0.99963 (6 rows differ, near ties) / 0.0018 / [0.9972, 1.0023];
  bf16 h and proj weight, bf16 out 0.999908 / 0.99921 / 0.0021 / [0.9952, 1.0041];
  logit noise 1% of std 0.99916 (passes) / 0.9930 / 0.0088; 3% 0.9974 (passes) / 0.979 / 0.026;
  no per_expert_scale 0.99991 (passes) / - / 0.0105 / [0.9887, 1.0129]; per_expert_scale by rank not id 0.99983 (passes) / - / 0.0168;
  renormalize after per_expert_scale 0.99996 (passes) / - / 0.0048 (passes) / [0.9887, 1.0129];
  top-7 0.9923 (passes) / nnz 7; top-9 0.9940 (passes) / nnz 9; row 0 or last row zeroed 0.9998 (passes) / nnz 0;
  last 32 rows zeroed 0.9918 (passes); no renormalize 0.955 / row sums [0.20, 0.90]; no router.scale 0.80; no rms 0.79;
  top-k on logits without softmax 0.87; expert ids shifted by one -0.007. Softmax over the top-8 logits is equivalent.
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
TOP_K = 8
MIN_SELECTION_OVERLAP = 0.995  # mean per-row |selected & golden selected| / 8
MAX_MATCHED_REL_L2 = 0.005  # weights, rows whose selected set equals the golden's
ROW_SUM_RATIO = (0.99, 1.01)  # per-token sum(got) / sum(want)


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

    # Selection, per-expert-scale and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    gs, ws = got != 0, w != 0
    nnz = gs.sum(-1)
    overlap = ((gs & ws).sum(-1).float() / ws.sum(-1)).mean().item()
    m = (gs == ws).all(-1)
    mrel = ((got[m] - w[m]).norm() / w[m].norm()).item() if m.any() else float("inf")
    ratio = got.sum(-1) / w.sum(-1)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"selection_overlap_{STEP}_L{LAYER:02d}", overlap)
    metrics.record(f"matched_rel_l2_{STEP}_L{LAYER:02d}", mrel)
    metrics.record(f"row_sum_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_sum_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    print(
        f"nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K}) selection_overlap={overlap:.5f} (>= {MIN_SELECTION_OVERLAP}) "
        f"matched_rows={m.sum().item()}/{m.numel()} matched_rel_l2={mrel:.5f} (<= {MAX_MATCHED_REL_L2}) "
        f"row_sum_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_SUM_RATIO})"
    )
    assert (nnz == TOP_K).all(), f"nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want exactly {TOP_K}"
    assert (got >= 0).all(), "negative routing weight"
    assert overlap >= MIN_SELECTION_OVERLAP, f"top-{TOP_K} selection overlap {overlap:.5f} < {MIN_SELECTION_OVERLAP}"
    assert (
        mrel <= MAX_MATCHED_REL_L2
    ), f"matched-row weight rel L2 {mrel:.5f} > {MAX_MATCHED_REL_L2} (per_expert_scale or renorm bug)"
    assert (
        ROW_SUM_RATIO[0] <= rmin and rmax <= ROW_SUM_RATIO[1]
    ), f"per-token routing sum ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_SUM_RATIO} (renormalization or per_expert_scale bug)"
