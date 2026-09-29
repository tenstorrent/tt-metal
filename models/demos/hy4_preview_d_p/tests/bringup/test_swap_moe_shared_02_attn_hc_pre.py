# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type moe_shared (layer 2) with attn_hc_pre swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 2 (moe_shared) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre

Reviewed (S.moe_shared.02.test.1). Built from the layer-1 swap test (test_swap_moe_full_02_attn_hc_pre.py) with the
two layer-2 changes of swap 01 (test_swap_moe_shared_01_attn_hc.py): ctx.extra["shared_topk"] = the golden's
L{src}.topk (src = cfg.topk_source(2) = 1) in both contexts (a shared layer's topk_shared otherwise raises a KeyError,
since layer 1 does not run here), and a per-stream h_mid rel L2 limit (stream 3 is about 5x the others). The gated
metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC streams, spec block threshold 0.98).
attn_hc_pre makes attn_x [S, H] = sum_j pre_j stream_j from the block input and the (device) gates. At layer 2 the
pre gate column means are 1.7e-6 / 0.96 / 0.20 / 0.032: pre gate 0 sits at hc_eps, so stream 0 is invisible on the
golden, and attn_x is mostly stream 1. Measured on the CPU (golden s4096 chunk 1, 2048 rows; the gates from the fp32
CPU attn_hc, attn_hc_pre replaced by mutations of the fp32 reference; script /tmp/hy4_ss2/study.py). "ax" = attn_x
vs golden rel L2 / row norm ratio / worst row, "cpu" = attn_x vs the CPU hc_pre on the same inputs (rel / worst
row), "rot" = the step once more with each row's pre gates rotated by (row mod 4) vs the CPU step (rel / worst row),
"hm" = h_mid rel / max per-stream rel / worst (row, stream), "rt" = router top-8 selection overlap vs golden:

    variant                ax                           | cpu             | rot             | hm                     | rt     | out PCC  rel
    fp32 reference         0.0023 [0.9998, 1.0002] 0.0024 | 0      / 0      | 0      / 0      | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030
    bf16 output            0.0027 [0.9998, 1.0002] 0.0030 | 0.0017 / 0.0017 | 0.0017 / 0.0017 | 0.0009 / 0.0030 / 0.0036 | 0.9969 | 0.999992 0.0040
    bf16 accumulation      0.0034 [0.9967, 1.0024] 0.0045 | 0.0025 / 0.0039 | 0.0026 / 0.0054 | 0.0009 / 0.0031 / 0.0038 | 0.9968 | 0.999991 0.0043
    pre + 3e-4             0.0030 [1.0005, 1.0029] 0.0056 | 0.0020 / 0.0052 | 0.0008 / 0.0046 | 0.0009 / 0.0030 / 0.0054 | 0.9966 | 0.999993 0.0038
    pre x 1.005            0.0055 [1.0048, 1.0052] 0.0057 | 0.0050 / 0.0050 | 0.0050 / 0.0050 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999994 0.0034
    pre x 1.01             0.010  [1.0098, 1.0102] 0.011  | 0.010  / 0.010  | 0.010  / 0.010  | 0.0009 / 0.0029 / 0.0041 | 0.9970 | 0.999994 0.0034
    pre x 1.02             0.020  [1.0198, 1.0202] 0.020  | 0.020  / 0.020  | 0.020  / 0.020  | 0.0009 / 0.0030 / 0.0069 | 0.9960 | 0.999993 0.0037
    stream 0 dropped       0.0023 [0.9998, 1.0002] 0.0024 | 0      / 0      | 0.19   / 0.96   | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030
    stream 1 dropped       0.90   [0.0505, 0.5377] 0.97   | 0.90   / 0.97   | 0.17   / 0.96   | 0.12   / 0.62   / 0.99   | 0.5867 | 0.974719 0.23
    stream 2 dropped       0.10   [0.7224, 0.9928] 0.62   | 0.10   / 0.62   | 0.20   / 0.97   | 0.0089 / 0.045  / 0.30   | 0.9200 | 0.999780 0.023
    stream 3 dropped       0.11   [0.9078, 1.0609] 0.56   | 0.11   / 0.56   | 0.94   / 1.20   | 0.016  / 0.081  / 0.34   | 0.8905 | 0.999088 0.049
    streams 0 / 1 swapped  0.28   [1.0044, 1.1267] 0.42   | 0.28   / 0.42   | 0.073  / 0.41   | 0.037  / 0.19   / 0.31   | 0.8698 | 0.998498 0.055
    streams 0 / 2 swapped  0.038  [0.9901, 1.0604] 0.26   | 0.038  / 0.26   | 0.090  / 0.58   | 0.0064 / 0.032  / 0.15   | 0.9498 | 0.999814 0.020
    streams 0 / 3 swapped  0.11   [0.9414, 1.1722] 0.61   | 0.11   / 0.61   | 1.24   / 13.2   | 0.016  / 0.083  / 0.35   | 0.8886 | 0.999081 0.049
    streams 1 / 2 swapped  0.42   [0.9965, 1.2557] 0.74   | 0.42   / 0.74   | 0.12   / 0.73   | 0.065  / 0.34   / 0.56   | 0.7285 | 0.988162 0.17
    streams 2 / 3 swapped  0.51   [0.9758, 4.0297] 3.60   | 0.51   / 3.60   | 1.23   / 5.79   | 0.056  / 0.29   / 1.04   | 0.5858 | 0.993306 0.12
    no gating (pre = 1)    5.90   [2.6313, 17.499] 17.0   | 5.90   / 17.0   | 2.04   / 14.9   | 0.12   / 0.62   / 1.13   | 0.4365 | 0.971064 0.25
    post gates instead     0.92   [0.0939, 2.8559] 1.90   | 0.92   / 1.90   | 0.97   / 1.77   | 0.037  / 0.19   / 0.68   | 0.7781 | 0.998272 0.059
    gate j on stream j + 1 0.32   [0.9826, 1.1551] 0.61   | 0.32   / 0.61   | 1.24   / 13.2   | 0.042  / 0.22   / 0.40   | 0.8234 | 0.998059 0.065
    last row zeroed        0.031  [0.0,    1.0002] 1.0    | 0.031  / 1.0    | 0.013  / 1.0    | 0.0068 / 0.036  / 1.15   | 0.9969 | 0.999972 0.0075
    rows 1023 / 1024 swapped 0.019 [0.7411, 1.3492] 1.35  | 0.019  / 1.35   | 0.0089 / 1.62   | 0.0035 / 0.017  / 1.23   | 0.9966 | 0.999991 0.0043
    gate rows shifted by 1 0.39   [0.4979, 2.2082] 1.28   | 0.39   / 1.28   | 1.29   / 6.20   | 0.020  / 0.11   / 0.50   | 0.8876 | 0.998240 0.061
    SP row halves swapped  1.32   [0.0610, 16.395] 16.2   | 1.32   / 16.2   | 1.30   / 16.2   | 0.22   / 1.16   / 1.51   | 0.5439 | 0.969227 0.25
    TP column halves swapped 1.41 [0.9998, 1.0002] 1.46   | 1.41   / 1.46   | 1.41   / 1.45   | 0.22   / 1.16   / 1.55   | 0.5397 | 0.940345 0.50
    zero stub              1.0    [0.0,    0.0]    1.0    | 1.0    / 1.0    | 1.0    / 1.0    | 0.19   / 1.00   / 1.29   | 0.5915 | 0.942568 0.53

Only the zero stub, SP / TP swaps, "no gating" and a dropped stream 1 fail the 0.98 out gate; every other row passes
it. So the test also asserts (informational metrics):
  - attn_hc (the gates, as swap 01, the component test's limits): not a CPU bridge, 8 columns, finite, rel L2
    <= 0.01, rel L2 per column (all 8) <= 0.01, worst row rel L2 over the post columns <= 0.015;
  - attn_x vs golden: finite, element count, rel L2 <= 0.005, every row's norm ratio in [0.996, 1.004] (pre x 1.005:
    1.0048; bf16 accumulation 0.9967), worst row <= 0.01;
  - attn_x vs the CPU hc_pre on the same inputs (block input + the device gates): rel L2 <= 0.003, worst row
    <= 0.006 (the component test's limits; separates the step's own error from the device gates' error);
  - the attn_hc_pre module once more with the device gates' pre columns rotated by (row mod 4), vs the CPU step on
    the same inputs: rel L2 <= 0.004, worst row <= 0.01 (every stream meets the large gate 1 on a quarter of the
    rows; the only check that sees a dropped stream 0, 0.19);
  - h_mid rel L2 <= 0.005, max per-stream rel L2 <= 0.005, worst (row, stream) <= 0.02; router top-8 overlap
    >= 0.98; block out finite, rel L2 <= 0.01 (as swap 01).
Every mutation above fails at least one check, except bf16 output / accumulation (not bugs) and pre + 3e-4 (attn_x
vs CPU worst row 0.0052 against 0.006; below the bf16 rounding of gate 1, ~0.004). The out worst (row, stream) rel
L2 is recorded, not asserted (near-tie tokens switch experts, see swap 01).
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
SWAPPED = ["attn_hc", "attn_hc_pre"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
HC_MULT = 4
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs golden
GATES_MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
X_MAX_REL_L2 = 0.005  # attn_x vs golden, whole tensor
X_RATIO = (0.996, 1.004)  # attn_x per-row ||got|| / ||want|| vs golden
X_MAX_ROW_REL = 0.01  # attn_x vs golden, worst row
CPU_MAX_REL_L2 = 0.003  # attn_x vs the CPU hc_pre on the same inputs
CPU_MAX_ROW_REL = 0.006
ROT_MAX_REL_L2 = 0.004  # rotated pre gates, vs the CPU hc_pre on the same inputs
ROT_MAX_ROW_REL = 0.01
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


def _ratio(got, want):
    r = got.float().reshape(want.shape).norm(dim=-1) / want.float().norm(dim=-1).clamp_min(1e-12)
    return r.min().item(), r.max().item()


def _stream_rel(got, want, streams):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=(0, 2)) / want.norm(dim=(0, 2)).clamp_min(1e-12)).tolist()


def _rotated(gates):
    """Each row's pre gates rotated by (row mod 4), so every stream meets the large gate on a quarter of the rows."""
    n = gates.shape[0]
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, :HC_MULT] = torch.gather(gates[:, :HC_MULT], 1, idx)
    return gs


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
    overrides, muts = {}, {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
        assert not getattr(mut, "cpu_bridge", False), f"device_component returned a CPU bridge for {name}"
        muts[name] = mut
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

    def finite_shape(n):
        got, want = seen[n], gl[n]
        if got.numel() != want.numel() or got.shape[-1] != want.shape[-1]:
            failures.append(f"{n}: shape {tuple(got.shape)} vs golden {tuple(want.shape)}")
            return False
        if not torch.isfinite(got.float()).all():
            failures.append(f"{n}: non-finite")
            return False
        return True

    # attn_hc: the iHC gates [S, 8] (pre 0-3 | post 4-7), as swap 01.
    gates_ok = finite_shape("attn_hc")
    if gates_ok:
        want = gl["attn_hc"].float()
        got = seen["attn_hc"].float().reshape(want.shape)
        rel = _rel(got, want)
        col_rel = ((got - want).norm(dim=0) / want.norm(dim=0).clamp_min(1e-12)).tolist()
        post_row = _worst_row_rel(got[:, HC_MULT:], want[:, HC_MULT:])
        metrics.record("rel_l2_swap_attn_hc", rel)
        metrics.record("max_col_rel_l2_swap_attn_hc", max(col_rel))
        metrics.record("post_worst_row_rel_l2_swap_attn_hc", post_row)
        print(
            f"attn_hc: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) col rel (pre 0-3 | post 4-7)="
            f"{[round(v, 5) for v in col_rel]} (<= {GATES_MAX_COL_REL}) post worst row={post_row:.5f} "
            f"(<= {GATES_MAX_POST_ROW_REL})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"attn_hc: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_rel) if v > GATES_MAX_COL_REL]
        if bad:
            failures.append(f"attn_hc: rel L2 > {GATES_MAX_COL_REL} in columns {bad}")
        if post_row > GATES_MAX_POST_ROW_REL:
            failures.append(f"attn_hc: post worst row rel L2 {post_row:.5f} > {GATES_MAX_POST_ROW_REL}")

    # attn_x (the swapped step) vs golden.
    if finite_shape("attn_x"):
        rel = _rel(seen["attn_x"], gl["attn_x"])
        rmin, rmax = _ratio(seen["attn_x"], gl["attn_x"])
        row = _worst_row_rel(seen["attn_x"], gl["attn_x"])
        metrics.record("rel_l2_swap_attn_x", rel)
        metrics.record("worst_row_rel_l2_swap_attn_x", row)
        print(
            f"attn_x vs golden: rel_l2={rel:.6f} (<= {X_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(X_RATIO)}) worst_row_rel_l2={row:.5f} (<= {X_MAX_ROW_REL})"
        )
        if rel > X_MAX_REL_L2:
            failures.append(f"attn_x: rel L2 {rel:.5f} > {X_MAX_REL_L2}")
        if not (X_RATIO[0] <= rmin and rmax <= X_RATIO[1]):
            failures.append(f"attn_x: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(X_RATIO)}")
        if row > X_MAX_ROW_REL:
            failures.append(f"attn_x: worst row rel L2 {row:.5f} > {X_MAX_ROW_REL}")

        if gates_ok:
            # vs the CPU hc_pre on the same inputs (block input + the gates the device produced).
            cpu = ref.component(layer, "attn_hc_pre")
            gates = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            same = cpu(rctx, gl["in"].float(), gates).float()
            crel, crow = _rel(seen["attn_x"], same), _worst_row_rel(seen["attn_x"], same)
            metrics.record("rel_l2_swap_attn_x_vs_cpu", crel)
            print(
                f"attn_x vs CPU hc_pre on the same inputs: rel_l2={crel:.6f} (<= {CPU_MAX_REL_L2}) "
                f"worst_row_rel_l2={crow:.5f} (<= {CPU_MAX_ROW_REL})"
            )
            if crel > CPU_MAX_REL_L2 or crow > CPU_MAX_ROW_REL:
                failures.append(f"attn_x vs CPU on the same inputs: rel {crel:.5f} / worst row {crow:.5f}")

            # Stream order: the module again with each row's pre gates rotated by (row mod 4).
            rot = (gl["in"].float(), _rotated(gates))
            rot_want = cpu(rctx, *rot).float()
            rot_out = muts["attn_hc_pre"](rctx, dctx, *rot)
            if rot_out.numel() != rot_want.numel() or not torch.isfinite(rot_out.float()).all():
                failures.append(f"attn_hc_pre rotated gates: shape {tuple(rot_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(rot_out, rot_want), _worst_row_rel(rot_out, rot_want)
                metrics.record("rot_rel_l2_swap_attn_x", yrel)
                print(
                    f"attn_hc_pre with rotated pre gates vs CPU: rel_l2={yrel:.6f} (<= {ROT_MAX_REL_L2}) "
                    f"worst_row_rel_l2={yrow:.5f} (<= {ROT_MAX_ROW_REL})"
                )
                if yrel > ROT_MAX_REL_L2 or yrow > ROT_MAX_ROW_REL:
                    failures.append(f"attn_hc_pre rotated gates: rel {yrel:.5f} / worst row {yrow:.5f} (stream order?)")

    # h_mid.
    if finite_shape("h_mid"):
        rel = _rel(seen["h_mid"], gl["h_mid"])
        wr = _worst_row_rel(seen["h_mid"], gl["h_mid"], HC_MULT)
        metrics.record("rel_l2_swap_h_mid", rel)
        metrics.record("worst_row_rel_l2_swap_h_mid", wr)
        print(f"h_mid: rel_l2={rel:.6f} (<= {H_MID_MAX_REL}) worst (row, stream) rel={wr:.6f} (<= {H_MID_MAX_ROW_REL})")
        if rel > H_MID_MAX_REL:
            failures.append(f"h_mid rel L2 {rel:.5f} > {H_MID_MAX_REL}")
        if wr > H_MID_MAX_ROW_REL:
            failures.append(f"h_mid worst row rel L2 {wr:.5f} > {H_MID_MAX_ROW_REL}")
        sr = _stream_rel(seen["h_mid"], gl["h_mid"], HC_MULT)
        metrics.record("max_stream_rel_l2_swap_h_mid", max(sr))
        print(f"h_mid: per-stream rel_l2={[round(v, 6) for v in sr]} (<= {H_MID_MAX_STREAM_REL})")
        bad = [j for j, v in enumerate(sr) if v > H_MID_MAX_STREAM_REL]
        if bad:
            failures.append(f"h_mid streams {bad} rel L2 above {H_MID_MAX_STREAM_REL}: {sr}")

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
