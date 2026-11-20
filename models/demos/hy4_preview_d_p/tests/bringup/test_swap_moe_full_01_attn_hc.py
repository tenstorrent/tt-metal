# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type moe_full (layer 1) with attn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (moe_full) with these steps on the device and the rest on the CPU reference:
    attn_hc

Reviewed (S.moe_full.01.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). attn_hc's gates [S, 8] (pre 4 | post 4, hc_attn_layer of layer 1) feed
attn_hc_pre (attn_x = sum_j pre_j stream_j), attn_residual (h_mid = stream_j + post_j attn_out) and, through h_mid,
the MoE half (router top-8 of 256, experts, shared expert). Unlike layer 0 the four input streams are distinct, so
pre-gate bugs do reach the block. Measured on the CPU (golden s4096 chunk 1, 2048 rows, the gates replaced by
mutations of the fp32 reference; "g" = gates rel L2 / max per-column rel L2 / post worst row rel, "ax" / "hm" = rel
L2 / worst row (h_mid: worst (row, stream)) vs golden, "rt" = router top-8 selection overlap vs golden):

    variant                      g                     | ax             | hm             | rt     | out PCC   rel
    fp32 reference               0.0014 / 0.0019 / 0.0039 | 0.0022 / 0.0027 | 0.0021 / 0.0029 | 0.9987 | 0.999997  0.0024
    gates rounded to bf16        0.0004 / 0.0011 / 0.0078 | 0.0026 / 0.0050 | 0.0023 / 0.0036 | 0.9986 | 0.999997  0.0024
    gates + 1e-3 (random sign)   0.0031 / 1.30   / 0.034  | 0.0028 / 0.0046 | 0.0047 / 0.062  | 0.9909 | 0.999986  0.0052
    post sigmoid err 1e-4        0.0014 / 0.13   / 0.0057 | 0.0022 / 0.0027 | 0.0021 / 0.0065 | 0.9982 | 0.999997  0.0025
    post x 1.005                 0.0019 / 0.0055 / 0.0090 | 0.0022 / 0.0027 | 0.0035 / 0.0049 | 0.9970 | 0.999990  0.0044
    post x 1.02                  0.0054 / 0.020  / 0.024  | 0.0022 / 0.0027 | 0.011  / 0.017  | 0.9883 | 0.999896  0.014
    pre x 1.01                   0.0098 / 0.010  / 0.0039 | 0.010  / 0.011  | 0.0021 / 0.0030 | 0.9985 | 0.999997  0.0025
    pre x 1.02                   0.019  / 0.020  / 0.0039 | 0.020  / 0.020  | 0.0022 / 0.0036 | 0.9980 | 0.999996  0.0027
    zero stub                    1.0    / 1.0    / 1.0    | 1.0    / 1.0    | 0.54   / 0.83   | 0.37   | 0.860     0.53
    post = 1 * sigmoid (not 2x)  0.13   / 0.50   / 0.50   | 0.0022 / 0.0027 | 0.27   / 0.42   | 0.70   | 0.973     0.23
    post gate 4 zeroed           0.0016 / 1.0    / 0.019  | 0.0022 / 0.0027 | 0.0026 / 0.017  | 0.9987 | 0.999997  0.0026
    pre gate 1 = pre gate 0      0.020  / 1.26   / 0.0039 | 0.014  / 0.080  | 0.0026 / 0.0065 | 0.9977 | 0.999994  0.0033
    pre gate 3 = pre gate 2      0.43   / 0.85   / 0.0039 | 0.50   / 2.23   | 0.092  / 0.31   | 0.88   | 0.997     0.077
    fn rows 0 / 1 swapped (pre)  0.025  / 0.99   / 0.0039 | 0.0056 / 0.031  | 0.0025 / 0.0078 | 0.9976 | 0.999994  0.0035
    base 0 / 1 swapped (pre)     0.0069 / 0.24   / 0.0039 | 0.0036 / 0.015  | 0.0022 / 0.0030 | 0.9982 | 0.999996  0.0029
    fn rows 4 / 5 swapped        0.018  / 24     / 1.27   | 0.0022 / 0.0027 | 0.039  / 1.66   | 0.9974 | 0.999732  0.023
    base 4 / 5 swapped           0.018  / 15     / 1.19   | 0.0022 / 0.0027 | 0.037  / 1.64   | 0.9977 | 0.999516  0.032
    base 6 / 7 swapped           0.039  / 0.19   / 0.17   | 0.0022 / 0.0027 | 0.081  / 0.13   | 0.9568 | 0.998650  0.053
    fn rows 6 / 7 swapped        0.25   / 2.79   / 1.25   | 0.0022 / 0.0027 | 0.53   / 3.46   | 0.59   | 0.903     0.46
    fn streams 0 / 1 swapped     0.014  / 0.098  / 0.096  | 0.012  / 0.021  | 0.013  / 0.066  | 0.9759 | 0.999861  0.017
    fn streams 1 / 2 swapped     0.088  / 0.26   / 0.55   | 0.056  / 0.28   | 0.061  / 0.38   | 0.90   | 0.998921  0.047
    TP: one chip's sumsq         0.15   / 0.91   / 0.60   | 0.16   / 0.22   | 0.20   / 0.41   | 0.70   | 0.984300  0.18
    TP: one chip's partial mixes 0.38   / 24.7   / 3.20   | 0.24   / 0.34   | 0.52   / 3.12   | 0.56   | 0.913     0.42
    RMS over one stream          0.17   / 0.57   / 0.92   | 0.091  / 0.51   | 0.20   / 0.64   | 0.63   | 0.985618  0.17
    rms eps 1e-6 (not 1e-5)      0.031  / 0.081  / 0.38   | 0.0078 / 0.093  | 0.027  / 0.28   | 0.94   | 0.999838  0.018
    rows shifted by 1            0.32   / 1.30   / 6.61   | 0.24   / 1.60   | 0.41   / 5.51   | 0.66   | 0.958     0.30
    SP row halves swapped        0.34   / 1.31   / 6.75   | 0.27   / 1.54   | 0.40   / 7.25   | 0.68   | 0.954     0.31
    rows 1023 / 1024 swapped     0.0038 / 0.022  / 0.15   | 0.0041 / 0.31   | 0.0028 / 0.092  | 0.9983 | 0.999996  0.0030
    last row's gates zero        0.024  / 0.033  / 1.0    | 0.033  / 1.0    | 0.017  / 0.64   | 0.9984 | 0.999880  0.016
    last row = previous row      0.0072 / 0.019  / 0.46   | 0.011  / 0.32   | 0.0075 / 0.26   | 0.9986 | 0.999986  0.0053
    no hc_eps                    0.0014 / 0.0021 / 0.0039 | 0.0022 / 0.0027 | 0.0021 / 0.0029 | 0.9987 | 0.999997  0.0024

Every row except the zero stub and "post = 1 * sigmoid" passes the 0.98 out gate (PCC is dominated by the residual
streams). So the test also asserts (informational metrics):
  - the swapped step vs golden (the component test's limits): not a CPU bridge, element count and 8 columns, finite,
    rel L2 over [S, 8] <= 0.01, rel L2 per column (all 8) <= 0.01, worst row rel L2 over the post columns <= 0.015;
  - attn_x rel L2 <= 0.005 and worst row <= 0.02 (pre gates; catches pre x 1.01, base / fn 0 / 1, rows 1023 / 1024);
  - h_mid rel L2 <= 0.005 and worst (row, stream) rel L2 <= 0.02 (post gates; catches post x 1.02, 4 / 5 swaps);
  - router top-8 selection overlap vs golden >= 0.98 (a gross check; near-tie flips give 0.9987 on the reference);
  - block out finite and rel L2 <= 0.01.
Caught by nothing: dropping hc_eps, post x 1.005 (inside the bf16 golden noise). The out worst (row, stream) rel L2
is 0.056 even for the fp32 reference (a few near-tie tokens switch experts), so it is recorded, not asserted.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import run_block
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
BLOCK_TYPE = "moe_full"
SWAPPED = ["attn_hc"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
HC_MULT = 4
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs golden
GATES_MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
ATTN_X_MAX_REL = 0.005  # attn_x vs golden, whole tensor
ATTN_X_MAX_ROW_REL = 0.02  # attn_x, worst row
H_MID_MAX_REL = 0.005  # h_mid vs golden, whole tensor
H_MID_MAX_ROW_REL = 0.02  # h_mid, worst (row, stream)
ROUTER_MIN_OVERLAP = 0.98  # router top-8 selection overlap vs golden
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _worst_row_rel(got, want, streams=1):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)).max().item()


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    overrides = {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
        assert not getattr(mut, "cpu_bridge", False), f"device_component returned a CPU bridge for {name}"
        overrides[name] = lambda ctx, *x, mut=mut: mut(ctx, dctx, *x)
    gl = g.layer(c, layer)
    seen = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        rctx,
        gl["in"].float(),
        rec=lambda n, t: seen.__setitem__(n, t),
        overrides=overrides,
    )
    for n, t in seen.items():
        if n not in ("in", "out") and n in gl:
            compare(f"pcc_swap_{n}", t, gl[n], default_mode(gl[n]), 0.0)  # the trail, for diagnosis; not gated
    thr = threshold(S, "block") if THRESHOLD is None else THRESHOLD
    _, ok = compare("pcc_swap_out", seen["out"], gl["out"], "pcc", thr)

    # Extra checks (see the module docstring). Recorded as informational metrics, asserted here.
    failures = [] if ok else [f"pcc_swap_out below {thr}"]

    # The swapped step itself: the iHC gates [S, 8].
    for name in SWAPPED:
        o = _step(ref, layer, name).output
        want, got = gl[o].float(), seen[o]
        if got.numel() != want.numel() or got.shape[-1] != want.shape[-1]:
            failures.append(f"swapped step {name}: shape {tuple(got.shape)} vs golden {tuple(want.shape)}")
            continue
        got = got.float().reshape(want.shape)
        if not torch.isfinite(got).all():
            failures.append(f"swapped step {name}: non-finite output")
            continue
        err = got - want
        rel = _rel(got, want)
        col_rel = (err.norm(dim=0) / want.norm(dim=0).clamp_min(1e-12)).tolist()
        post_row = _worst_row_rel(got[:, HC_MULT:], want[:, HC_MULT:])
        metrics.record(f"rel_l2_swap_{o}", rel)
        metrics.record(f"max_col_rel_l2_swap_{o}", max(col_rel))
        metrics.record(f"post_worst_row_rel_l2_swap_{o}", post_row)
        print(
            f"swapped {name}: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) col rel (pre 0-3 | post 4-7)="
            f"{[round(v, 5) for v in col_rel]} (<= {GATES_MAX_COL_REL}) post worst row={post_row:.5f} "
            f"(<= {GATES_MAX_POST_ROW_REL})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"swapped step {name}: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_rel) if v > GATES_MAX_COL_REL]
        if bad:
            failures.append(f"swapped step {name}: rel L2 > {GATES_MAX_COL_REL} in columns {bad}")
        if post_row > GATES_MAX_POST_ROW_REL:
            failures.append(f"swapped step {name}: post worst row rel L2 {post_row:.5f} > {GATES_MAX_POST_ROW_REL}")

    # Its consumers: attn_x (pre gates) and h_mid (post gates).
    for n, lim, row_lim, streams in (
        ("attn_x", ATTN_X_MAX_REL, ATTN_X_MAX_ROW_REL, 1),
        ("h_mid", H_MID_MAX_REL, H_MID_MAX_ROW_REL, HC_MULT),
    ):
        if not torch.isfinite(seen[n]).all():
            failures.append(f"{n}: non-finite")
            continue
        rel = _rel(seen[n], gl[n])
        wr = _worst_row_rel(seen[n], gl[n], streams)
        metrics.record(f"rel_l2_swap_{n}", rel)
        metrics.record(f"worst_row_rel_l2_swap_{n}", wr)
        print(f"{n}: rel_l2={rel:.6f} (<= {lim}) worst_row_rel_l2={wr:.6f} (<= {row_lim})")
        if rel > lim:
            failures.append(f"{n} rel L2 {rel:.5f} > {lim}")
        if wr > row_lim:
            failures.append(f"{n} worst row rel L2 {wr:.5f} > {row_lim}")

    # Router selection (top-8 of 256; the dense routing weights are nonzero on the selected experts).
    want_sel = gl["router"] != 0
    got_sel = seen["router"].reshape(want_sel.shape) != 0
    overlap = ((got_sel & want_sel).sum(-1).float() / want_sel.sum(-1).clamp_min(1)).mean().item()
    metrics.record("router_overlap_swap", overlap)
    print(f"router top-8 overlap={overlap:.5f} (>= {ROUTER_MIN_OVERLAP})")
    if overlap < ROUTER_MIN_OVERLAP:
        failures.append(f"router selection overlap {overlap:.4f} < {ROUTER_MIN_OVERLAP}")

    # Block out.
    out_rel = _rel(seen["out"], gl["out"])
    out_row = _worst_row_rel(seen["out"], gl["out"], HC_MULT)
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("worst_row_rel_l2_swap_out", out_row)  # informational only (near-tie expert flips)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2}) worst (row, stream) rel={out_row:.5f} (not gated)")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
