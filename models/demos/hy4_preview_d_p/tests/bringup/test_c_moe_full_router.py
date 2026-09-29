# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.router.test.1). HF HYV4TopkRouter: fp32 logits x @ W^T, sigmoid scores, top-8 on scores +
e_score_correction_bias (n_group 1, so the group mask is a no-op), weights = the unbiased sigmoids of the chosen 8,
renormalized (+1e-20), x routed_scaling_factor 2.827. Boundary: the dense [2048, 256] routing matrix (golden bf16,
exactly 8 nonzeros per row, row sums [2.8213, 2.8330] = 2.827 up to bf16 rounding). The input ffn_norm is bf16 and the
gate weight is stored bf16, so fp32 / TF32 operands are exact on the device.
Layer 1 bias is small: -0.097..0.032 (std 0.011), choice score <= 0.98; the median 8th-9th choice gap is 0.0028 (467 of
2048 rows < 1e-3). Measured on this golden (CPU study, /tmp/hy4_router1) as PCC / selection overlap vs golden / matched-row
weight rel L2 / row sums:
  CPU reference, fp32 on the bf16 golden input: 0.99934 / 0.99811 / 0.0017 / 2.827; bf16 output: 0.99934 / 0.99811 /
  0.0011 / [2.8213, 2.8330]; x or W truncated to TF32: identical;
  logits bf16 0.99800 / 0.99384; sigmoid bf16 0.99773 / 0.99341; choice score bf16 0.99666 / 0.99030; all-bf16 0.99581 /
  0.98773; logit noise 3e-3 x std 0.99766 / 0.99353 (1e-3: 0.99725, 1e-4: 0.99799); correction bias bf16 0.99927 /
  0.99799 (harmless here: the bias is small); no bias 0.972 / 0.917; half bias 0.985 / 0.956; bias misaligned 0.956;
  weights from scores + bias 0.99899 / - / 0.0263; no scale (x 1.0) 0.99934 (passes!) / - / 0.646 / 1.0; scale 2.5
  0.99967 / - / 0.116; weights x1.02 0.99956 / - / 0.0201; x1.005 0.99945 / - / 0.0053; logits x1.01 0.99920 / 0.99780
  / 0.0065; logits x1.1 0.99724 / 0.99567 / 0.063; logits + 0.1 0.99762 / 0.99402 / 0.027; no renormalize 0.967 / row
  sums 3.5-16.4; top-7 0.978 / nnz 7; top-9 0.980 / nnz 9; last row zeroed 0.99914 (passes) / nnz 0; last 32 rows
  zeroed 0.99189 (passes) / 0.98248; softmax scores 0.768; expert or row halves swapped ~0.
The gated metric is PCC. It misses the route scale, a weight scale, weights from biased scores, bf16 logits / scores /
choice keys and zeroed rows, so the test also asserts: exactly 8 nonzeros per row, non-negative finite weights, mean
selection overlap vs golden >= 0.995 and vs the CPU step on the same input >= 0.996 (reference 0.99811 / 1.0; the MiMo
fp32 device router lost 0.0004 against its CPU step; every bf16 stage above fails), matched-row weight rel L2 <= 0.005
vs golden and <= 0.004 vs the CPU step, and every row sum within 0.4% of 2.827 (bf16 rounding: 0.21%).
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
ROUTE_SCALE = 2.827  # routed_scaling_factor: every row sums to it (norm_topk_prob)
MIN_SELECTION_OVERLAP = 0.995  # mean per-row |selected & golden selected| / 8
MIN_CPU_SELECTION_OVERLAP = 0.996  # the same vs the CPU step on the same (golden) input
MAX_MATCHED_REL_L2 = 0.005  # weights on rows whose selected set equals the golden's
MAX_CPU_MATCHED_REL_L2 = 0.004  # the same vs the CPU step
MAX_ROW_SUM_ERR = 0.004  # |sum(row) / 2.827 - 1|


def _selection(got, want):
    """(mean selection overlap, matched-row rel L2, number of matched rows) of got against want."""
    gs, ws = got != 0, want != 0
    overlap = ((gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)).mean().item()
    m = (gs == ws).all(-1)
    mrel = ((got[m] - want[m]).norm() / want[m].norm()).item() if m.any() else float("inf")
    return overlap, mrel, int(m.sum().item())


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(
        fn, "cpu_bridge", False
    ), "device_component returned a CPU bridge; the router is not on the device"
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Selection, weight and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    cpu = ref.component(LAYER, STEP)(reference_ctx(ref, LAYER, g, c), *inputs).float().reshape(want.shape)
    assert torch.isfinite(got).all(), "non-finite output"
    nnz = (got != 0).sum(-1)
    overlap, mrel, nm = _selection(got, w)
    c_overlap, c_mrel, c_nm = _selection(got, cpu)
    rsum = got.sum(-1) / ROUTE_SCALE
    rmin, rmax = rsum.min().item(), rsum.max().item()
    metrics.record(f"selection_overlap_{STEP}_L{LAYER:02d}", overlap)
    metrics.record(f"selection_overlap_cpu_{STEP}_L{LAYER:02d}", c_overlap)
    metrics.record(f"matched_rel_l2_{STEP}_L{LAYER:02d}", mrel)
    metrics.record(f"matched_rel_l2_cpu_{STEP}_L{LAYER:02d}", c_mrel)
    metrics.record(f"row_sum_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_sum_max_{STEP}_L{LAYER:02d}", rmax)
    print(
        f"nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K})\n"
        f"vs golden: selection_overlap={overlap:.5f} (>= {MIN_SELECTION_OVERLAP}) matched_rows={nm}/{w.shape[0]} "
        f"matched_rel_l2={mrel:.5f} (<= {MAX_MATCHED_REL_L2})\n"
        f"vs CPU step: selection_overlap={c_overlap:.5f} (>= {MIN_CPU_SELECTION_OVERLAP}) matched_rows={c_nm}/{w.shape[0]} "
        f"matched_rel_l2={c_mrel:.5f} (<= {MAX_CPU_MATCHED_REL_L2})\n"
        f"row_sum / {ROUTE_SCALE} = [{rmin:.5f}, {rmax:.5f}] (1 +- {MAX_ROW_SUM_ERR})"
    )
    assert (nnz == TOP_K).all(), f"nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want exactly {TOP_K}"
    assert (got >= 0).all(), "negative routing weight"
    assert (
        overlap >= MIN_SELECTION_OVERLAP
    ), f"top-{TOP_K} selection overlap {overlap:.5f} < {MIN_SELECTION_OVERLAP} (bf16 logits / scores / choice keys, or the bias?)"
    assert (
        c_overlap >= MIN_CPU_SELECTION_OVERLAP
    ), f"top-{TOP_K} selection overlap vs the CPU step {c_overlap:.5f} < {MIN_CPU_SELECTION_OVERLAP}"
    assert (
        mrel <= MAX_MATCHED_REL_L2
    ), f"matched-row weight rel L2 {mrel:.5f} > {MAX_MATCHED_REL_L2} (weights from biased scores, scale or renorm bug)"
    assert (
        c_mrel <= MAX_CPU_MATCHED_REL_L2
    ), f"matched-row weight rel L2 vs the CPU step {c_mrel:.5f} > {MAX_CPU_MATCHED_REL_L2}"
    assert (
        1 - MAX_ROW_SUM_ERR <= rmin and rmax <= 1 + MAX_ROW_SUM_ERR
    ), f"per-token routing sum / {ROUTE_SCALE} in [{rmin:.5f}, {rmax:.5f}], want 1 +- {MAX_ROW_SUM_ERR} (route scale or renorm bug)"
