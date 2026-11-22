# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: router of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.router.test.1), from the layer-1 test (test_c_moe_full_router.py). The router of a moe_shared
layer is the same HF HYV4TopkRouter as on moe_full layers (only the attention differs): fp32 logits x @ W^T, sigmoid
scores, top-8 on scores + e_score_correction_bias (n_group 1), weights = the unbiased sigmoids of the chosen 8,
renormalized (+1e-20), x routed_scaling_factor 2.827. Boundary: the dense [2048, 256] routing matrix (golden bf16,
exactly 8 nonzeros per row, row sums [2.8203, 2.8320] = 2.827 up to bf16 rounding). The input ffn_norm is bf16 and the
gate weight is stored bf16, so fp32 / TF32 operands are exact on the device.
Layer 2 bias: -0.147..0.024 (std 0.017, wider than layer 1's), choice score <= 0.94; the median 8th-9th choice gap is
0.0017 (691 of 2048 rows < 1e-3; layer 1: 0.0028 / 467), so layer 2 is more precision-bound than layer 1. Measured on
this golden (CPU study, /tmp/hy4_router2/{study,mut,mut2}.py) as PCC / selection overlap vs golden / matched-row
weight rel L2 / row sums:
  CPU reference, fp32 on the bf16 golden input: 0.99894 / 0.99823 / 0.0017 / 2.827; bf16 output: 0.99894 / 0.99823 /
  0.0011 / [2.8223, 2.8320]; x or W truncated to TF32: identical (overlap vs the CPU step 1.0);
  logits bf16 0.99395 / 0.98969; sigmoid bf16 0.99280 / 0.98792; choice score bf16 0.99063 / 0.98395; all-bf16 0.98787 /
  0.97913; logit noise 3e-3 x std 0.99394 / 0.98944 (1e-3: overlap 0.99554, vs CPU 0.99622; 1e-4: vs CPU 0.99976);
  correction bias bf16 0.99905 / 0.99841 (harmless); no bias 0.930 / 0.881; half bias 0.964 / 0.939; bias misaligned
  0.883; weights from scores + bias 0.99810 / - / 0.0407; no scale (x 1.0) 0.99895 (passes!) / - / 0.646 / 1.0;
  scale 2.5 0.99930 / - / 0.116; weights x1.02 0.99916 / - / 0.0201; x1.005 0.99906 / - / 0.0053; logits x1.01
  0.99880 / 0.99805 / 0.0053; logits x1.1 0.99597 / 0.99481 / 0.051; logits + 0.1 0.99554 / 0.99268 / 0.019; no
  renormalize 0.976 / row sums 3.4-12.7; top-7 0.964 / nnz 7; top-9 0.967 / nnz 9; last row zeroed 0.99872 (passes) /
  nnz 0; last 32 rows zeroed 0.99077 (passes) / 0.98260; softmax scores 0.545; expert or row halves swapped ~0;
  rows 1023 / 1024 swapped: overlap vs the CPU step 0.99902 (passes the mean), worst row 0.
The gated metric is PCC. It misses the route scale, a weight scale, weights from biased scores, bf16 sigmoid, zeroed
rows and a row permutation, so the test also asserts: exactly 8 nonzeros per row, non-negative finite weights, mean
selection overlap vs golden >= 0.995 and vs the CPU step on the same input >= 0.996 (reference 0.99823 / 1.0; the
layer-1 fp32 device router scored 0.99982 vs its CPU step; every bf16 stage above fails both), worst per-row overlap vs
the CPU step >= 0.75 (a near-tie flip costs 1 of 8; adjacent rows share at most 5 of 8 experts here, so a swapped row
scores <= 0.625), matched-row weight rel L2 <= 0.005 vs golden and <= 0.004 vs the CPU step, and every row sum within
0.4% of 2.827 (bf16 rounding: 0.24%).
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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
TOP_K = 8
ROUTE_SCALE = 2.827  # routed_scaling_factor: every row sums to it (norm_topk_prob)
MIN_SELECTION_OVERLAP = 0.995  # mean per-row |selected & golden selected| / 8
MIN_CPU_SELECTION_OVERLAP = 0.996  # the same vs the CPU step on the same (golden) input
MAX_MATCHED_REL_L2 = 0.005  # weights on rows whose selected set equals the golden's
MAX_CPU_MATCHED_REL_L2 = 0.004  # the same vs the CPU step
MAX_ROW_SUM_ERR = 0.004  # |sum(row) / 2.827 - 1|
MIN_CPU_ROW_OVERLAP = 0.75  # worst per-row overlap vs the CPU step (catches row permutations)


def _selection(got, want):
    """(mean selection overlap, matched-row rel L2, number of matched rows) of got against want."""
    gs, ws = got != 0, want != 0
    overlap = ((gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)).mean().item()
    m = (gs == ws).all(-1)
    mrel = ((got[m] - want[m]).norm() / want[m].norm()).item() if m.any() else float("inf")
    return overlap, mrel, int(m.sum().item())


def _worst_row_overlap(got, want):
    gs, ws = got != 0, want != 0
    return ((gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)).min().item()


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
    c_worst = _worst_row_overlap(got, cpu)
    rsum = got.sum(-1) / ROUTE_SCALE
    rmin, rmax = rsum.min().item(), rsum.max().item()
    metrics.record(f"selection_overlap_{STEP}_L{LAYER:02d}", overlap)
    metrics.record(f"selection_overlap_cpu_{STEP}_L{LAYER:02d}", c_overlap)
    metrics.record(f"selection_overlap_cpu_worst_row_{STEP}_L{LAYER:02d}", c_worst)
    metrics.record(f"matched_rel_l2_{STEP}_L{LAYER:02d}", mrel)
    metrics.record(f"matched_rel_l2_cpu_{STEP}_L{LAYER:02d}", c_mrel)
    metrics.record(f"row_sum_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_sum_max_{STEP}_L{LAYER:02d}", rmax)
    print(
        f"nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K})\n"
        f"vs golden: selection_overlap={overlap:.5f} (>= {MIN_SELECTION_OVERLAP}) matched_rows={nm}/{w.shape[0]} "
        f"matched_rel_l2={mrel:.5f} (<= {MAX_MATCHED_REL_L2})\n"
        f"vs CPU step: selection_overlap={c_overlap:.5f} (>= {MIN_CPU_SELECTION_OVERLAP}) matched_rows={c_nm}/{w.shape[0]} "
        f"matched_rel_l2={c_mrel:.5f} (<= {MAX_CPU_MATCHED_REL_L2}) worst_row_overlap={c_worst:.3f} (>= {MIN_CPU_ROW_OVERLAP})\n"
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
        c_worst >= MIN_CPU_ROW_OVERLAP
    ), f"worst per-row selection overlap vs the CPU step {c_worst:.3f} < {MIN_CPU_ROW_OVERLAP} (rows permuted or zeroed?)"
    assert (
        mrel <= MAX_MATCHED_REL_L2
    ), f"matched-row weight rel L2 {mrel:.5f} > {MAX_MATCHED_REL_L2} (weights from biased scores, scale or renorm bug)"
    assert (
        c_mrel <= MAX_CPU_MATCHED_REL_L2
    ), f"matched-row weight rel L2 vs the CPU step {c_mrel:.5f} > {MAX_CPU_MATCHED_REL_L2}"
    assert (
        1 - MAX_ROW_SUM_ERR <= rmin and rmax <= 1 + MAX_ROW_SUM_ERR
    ), f"per-token routing sum / {ROUTE_SCALE} in [{rmin:.5f}, {rmax:.5f}], want 1 +- {MAX_ROW_SUM_ERR} (route scale or renorm bug)"
