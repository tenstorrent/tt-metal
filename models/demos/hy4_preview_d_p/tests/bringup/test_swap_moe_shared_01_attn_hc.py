# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type moe_shared (layer 2) with attn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 2 (moe_shared) with these steps on the device and the rest on the CPU reference:
    attn_hc

Reviewed (S.moe_shared.01.test.1). Built from the layer-1 swap test (test_swap_moe_full_01_attn_hc.py). The gated
metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC streams, spec block threshold 0.98). attn_hc's
gates [S, 8] (pre 4 | post 4, hc_attn_layer of layer 2) feed attn_hc_pre (attn_x = sum_j pre_j stream_j),
attn_residual (h_mid = stream_j + post_j attn_out) and, through h_mid, the MoE half (router top-8 of 256, experts,
shared expert).

A shared layer has no indexer: its topk_shared step returns the top-k of the latest full layer's current chunk,
which the reference only has if that layer ran first. The rendered test (run_swap_test) did not set it and raised a
KeyError in topk_shared, with the reference or the device. This test sets ctx.extra["shared_topk"] to the golden's
L{src}.topk (src = cfg.topk_source(2) = 1, int64 [2048, 2048]) in both the reference and the device context.

Layer 2's gates (see test_c_moe_shared_attn_hc.py): pre gate 0 sits at hc_eps (~2e-6), post gate 7 is small (mean
0.0028), and stream 3 (norm 340) is about 5x the others. So h_mid's whole-tensor rel L2 follows stream 3, which the
post-gate bugs barely touch; this test also checks h_mid per stream. Measured on the CPU (golden s4096 chunk 1, 2048
rows, the gates replaced by mutations of the fp32 reference; "g" = gates rel L2 / column 0 rel / max rel over columns
1-7 / post worst row rel, "ax" = attn_x rel L2 / worst row, "hm" = h_mid rel L2 / max per-stream rel / worst (row,
stream), "rt" = router top-8 selection overlap vs golden, out = PCC / rel L2 vs golden):

    variant                     g                              | ax              | hm                      | rt     | out
    fp32 reference              0.0013 / 0.0017 / 0.0018 / 0.0030 | 0.0023 / 0.0024 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030
    gates rounded to bf16       0.0004 / 0.0010 / 0.0011 / 0.0047 | 0.0025 / 0.0035 | 0.0009 / 0.0032 / 0.0052 | 0.9971 | 0.999986 0.0054
    gates + 1e-3 (random sign)  0.0029 / 468    / 0.17   / 0.027  | 0.0060 / 0.017  | 0.0014 / -      / 0.017  | 0.9883 | 0.999964 0.0085
    post sigmoid err 1e-4       0.0013 / 0.0017 / 0.017  / 0.0042 | 0.0023 / 0.0024 | 0.0009 / 0.0029 / 0.0033 | 0.9970 | 0.999996 0.0030
    post sigmoid err 3e-4       0.0014 / 0.0017 / 0.050  / 0.0089 | 0.0023 / 0.0024 | 0.0009 / -      / 0.0055 | 0.9958 | 0.999994 0.0033
    post x 1.005                0.0026 / 0.0017 / 0.0055 / 0.0079 | 0.0023 / 0.0024 | 0.0014 / 0.0060 / 0.0074 | 0.9944 | 0.999971 0.0081
    post x 1.01                 0.0048 / 0.0017 / 0.010  / 0.013  | 0.0023 / 0.0024 | 0.0022 / 0.011  / 0.014  | 0.9888 | 0.999909 0.014
    post x 1.02                 0.0092 / 0.0017 / 0.020  / 0.023  | 0.0023 / 0.0024 | 0.0042 / 0.022  / 0.028  | 0.9789 | 0.999833 0.021
    pre x 1.01                  0.0090 / 0.010  / 0.010  / 0.0030 | 0.010  / 0.011  | 0.0009 / -      / 0.0041 | 0.9970 | 0.999994 0.0034
    pre x 1.02                  0.018  / 0.020  / 0.020  / 0.0030 | 0.020  / 0.020  | 0.0009 / -      / 0.0069 | 0.9960 | 0.999993 0.0037
    zero stub                   1.0    / 1.0    / 1.0    / 1.0    | 1.0    / 1.0    | 0.20   / -      / 1.37   | 0.49   | 0.852    1.28
    post = 1 * sigmoid (not 2x) 0.23   / 0.0017 / 0.50   / 0.50   | 0.0023 / 0.0024 | 0.10   / -      / 0.69   | 0.67   | 0.950    0.53
    post gate 4 zeroed          0.24   / 0.0017 / 1.0    / 0.65   | 0.0023 / 0.0024 | 0.11   / -      / 1.22   | 0.89   | 0.992940 0.15
    post gate 7 zeroed          0.0056 / 0.0017 / 1.0    / 0.28   | 0.0023 / 0.0024 | 0.0020 / 0.0029 / 0.055  | 0.9878 | 0.999994 0.0034
    post 5 = post 4             0.055  / 0.0017 / 0.19   / 0.32   | 0.0023 / 0.0024 | 0.025  / -      / 0.43   | 0.9958 | 0.999803 0.021
    pre gate 0 zeroed           0.0013 / 1.0    / 0.0018 / 0.0030 | 0.0023 / 0.0024 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030
    pre gate 1 = pre gate 0     0.86   / 0.0017 / 1.0    / 0.0030 | 0.90   / 0.97   | 0.12   / -      / 0.99   | 0.59   | 0.974719 0.23
    pre gate 3 = pre gate 2     0.19   / 0.0017 / 4.7    / 0.0030 | 0.50   / 3.60   | 0.053  / -      / 1.02   | 0.59   | 0.993290 0.12
    fn rows 0 / 1 swapped       0.98   / 2.5e5  / 1.0    / 0.0030 | 0.80   / 0.89   | 0.073  / -      / 0.65   | 0.78   | 0.985069 0.19
    base 0 / 1 swapped          0.55   / 73     / 0.63   / 0.0030 | 0.80   / 0.89   | 0.072  / -      / 0.67   | 0.81   | 0.985748 0.18
    fn rows 4 / 5 swapped       0.096  / 0.0017 / 0.29   / 0.49   | 0.0023 / 0.0024 | 0.044  / -      / 0.47   | 0.9725 | 0.999147 0.043
    base 4 / 5 swapped          0.020  / 0.0017 / 0.055  / 0.063  | 0.0023 / 0.0024 | 0.0091 / -      / 0.092  | 0.9902 | 0.999961 0.0091
    fn rows 6 / 7 swapped       0.98   / 0.0017 / 3.8    / 24     | 0.0023 / 0.0024 | 0.35   / -      / 18     | 0.26   | 0.930    0.41
    base 6 / 7 swapped          1.08   / 0.0017 / 194    / 25     | 0.0023 / 0.0024 | 0.40   / -      / 4.5    | 0.30   | 0.166    9.8
    fn streams 0 / 1 swapped    0.072  / 0.012  / 0.31   / 0.12   | 0.036  / 0.32   | 0.0057 / -      / 0.21   | 0.9420 | 0.999838 0.020
    fn streams 1 / 2 swapped    0.12   / 0.053  / 0.49   / 0.27   | 0.052  / 0.46   | 0.017  / -      / 0.27   | 0.83   | 0.997025 0.087
    TP: one chip's sumsq        0.13   / 0.63   / 2.5    / 0.99   | 0.084  / 0.24   | 0.049  / -      / 0.63   | 0.65   | 0.975498 0.32
    TP: one chip's partial mixes 0.34  / 23     / 1.34   / 3.75   | 0.32   / 1.04   | 0.12   / -      / 2.59   | 0.15   | 0.950    0.34
    RMS over one stream         0.38   / 0.67   / 45     / 14     | 0.13   / 0.73   | 0.14   / -      / 2.17   | 0.47   | 0.831    1.00
    rms eps 1e-6 (not 1e-5)     0.010  / 0.016  / 0.26   / 0.23   | 0.0041 / 0.054  | 0.0022 / -      / 0.15   | 0.9528 | 0.999973 0.0074
    rows shifted by 1           0.49   / 0.81   / 1.35   / 13     | 0.39   / 1.28   | 0.18   / -      / 10.5   | 0.48   | 0.880    0.82
    SP row halves swapped       0.48   / 0.83   / 1.26   / 15.5   | 0.42   / 2.28   | 0.17   / -      / 12.1   | 0.50   | 0.882    0.80
    rows 1023 / 1024 swapped    0.0032 / 0.0017 / 0.0077 / 0.47   | 0.0024 / 0.056  | 0.0015 / -      / 0.46   | 0.9969 | 0.999991 0.0042
    last row's gates zero       0.024  / 0.021  / 0.035  / 1.0    | 0.031  / 1.0    | 0.0070 / -      / 1.17   | 0.9969 | 0.999767 0.022
    last row = previous row     0.0098 / 0.0094 / 0.024  / 0.62   | 0.0046 / 0.13   | 0.0044 / -      / 0.71   | 0.9969 | 0.999911 0.013
    no hc_eps                   0.0013 / 0.47   / 0.0018 / 0.0030 | 0.0023 / 0.0024 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030

("-" = not measured.) Every row except the zero stub, "post = 1 * sigmoid", "pre gate 1 = pre gate 0", "TP: one chip's
sumsq" and "RMS over one stream" passes the 0.98 out gate. So the test also asserts (informational metrics):
  - the swapped step vs golden (the component test's limits): not a CPU bridge, element count and 8 columns, finite,
    rel L2 over [S, 8] <= 0.01, rel L2 per column (all 8) <= 0.01, worst row rel L2 over the post columns <= 0.015
    (the per-column check is the only one that sees a dropped hc_eps or a zeroed pre gate 0; the post columns catch
    a sigmoid error of 1e-4);
  - attn_x rel L2 <= 0.005 and worst row <= 0.02 (pre gates);
  - h_mid rel L2 <= 0.005, max per-stream rel L2 <= 0.005 and worst (row, stream) rel L2 <= 0.02 (post gates; the
    per-stream check catches post x 1.005, which every other check misses: stream 1 at 0.0060 vs 0.0029);
  - router top-8 selection overlap vs golden >= 0.98 (a gross check; near-tie flips give 0.9973 on the reference);
  - block out finite and rel L2 <= 0.01.
Caught by nothing: gates rounded to bf16 (not a bug; its h_mid stream 0.0032 is the tightest reference-side margin).
The out worst (row, stream) rel L2 is 0.024 on the fp32 reference and 0.063 with bf16 gates (near-tie tokens switch
experts), so it is recorded, not asserted.
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
BLOCK_TYPE = "moe_shared"
SWAPPED = ["attn_hc"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
HC_MULT = 4
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs golden; pre 0 sits at hc_eps
GATES_MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
ATTN_X_MAX_REL = 0.005  # attn_x vs golden, whole tensor
ATTN_X_MAX_ROW_REL = 0.02  # attn_x, worst row
H_MID_MAX_REL = 0.005  # h_mid vs golden, whole tensor
H_MID_MAX_STREAM_REL = 0.005  # h_mid, rel L2 of each stream (stream 3 dominates the whole tensor at layer 2)
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


def _stream_rel(got, want, streams):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=(0, 2)) / want.norm(dim=(0, 2)).clamp_min(1e-12)).tolist()


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    # A shared layer reuses the latest full layer's top-k for this chunk; that layer does not run here, so take it
    # from the golden (the source layer's recorded topk, int64 [S, index_topk]).
    src = ref.cfg.topk_source(layer)
    shared_topk = g.layer(c, src)["topk"]
    assert src != layer and shared_topk.shape[0] == g.chunk, f"layer {src} topk {tuple(shared_topk.shape)}"
    rctx.extra["shared_topk"] = shared_topk
    dctx.extra["shared_topk"] = shared_topk
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
        if streams > 1:
            sr = _stream_rel(seen[n], gl[n], streams)
            metrics.record(f"max_stream_rel_l2_swap_{n}", max(sr))
            print(f"{n}: per-stream rel_l2={[round(v, 6) for v in sr]} (<= {H_MID_MAX_STREAM_REL})")
            bad = [j for j, v in enumerate(sr) if v > H_MID_MAX_STREAM_REL]
            if bad:
                failures.append(f"{n} streams {bad} rel L2 above {H_MID_MAX_STREAM_REL}: {sr}")

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
