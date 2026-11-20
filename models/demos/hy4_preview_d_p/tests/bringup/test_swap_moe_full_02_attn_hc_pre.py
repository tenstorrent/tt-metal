# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type moe_full (layer 1) with attn_hc_pre swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (moe_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre

Reviewed (S.moe_full.02.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). attn_hc_pre makes attn_x [S, H] = sum_j pre_j stream_j from the block input and
the (device) gates. attn_x feeds attn_norm and through it the attention, then h_mid and the MoE half. At layer 1 the
streams are distinct, but the pre gates are very unequal (column means 0.026 / 0.013 / 0.83 / 0.49), so attn_x is
mostly streams 2 and 3. Measured on the CPU (golden s4096 chunk 1, 2048 rows; the gates from the fp32 CPU attn_hc,
attn_hc_pre replaced by mutations of the fp32 reference). "ax" = attn_x vs golden rel L2 / row norm ratio / worst
row, "cpu" = attn_x vs the CPU hc_pre on the same inputs (rel / worst row), "rot" = the step once more with each
row's pre gates rotated by (row mod 4) vs the CPU step (rel / worst row), "hm" = h_mid rel / worst (row, stream),
"rt" = router top-8 selection overlap vs golden:

    variant                   ax                          | cpu           | rot           | hm            | rt     | out PCC   rel
    fp32 reference            0.0022 [0.9998, 1.0003] 0.0027 | 0      / 0      | 0      / 0      | 0.0021 / 0.0029 | 0.9987 | 0.999997  0.0024
    bf16 output               0.0026 [0.9998, 1.0004] 0.0032 | 0.0017 / 0.0017 | 0.0017 / 0.0017 | 0.0021 / 0.0026 | 0.9988 | 0.999997  0.0024
    pre x 1.005               0.0055 [1.0048, 1.0053] 0.0059 | 0.0050 / 0.0050 | 0.0050 / 0.0050 | 0.0021 / 0.0029 | 0.9985 | 0.999997  0.0024
    pre x 1.01                0.010  [1.0098, 1.0103] 0.011  | 0.010  / 0.010  | 0.010  / 0.010  | 0.0021 / 0.0030 | 0.9985 | 0.999997  0.0025
    pre x 1.02                0.020  [1.0198, 1.0203] 0.020  | 0.020  / 0.020  | 0.020  / 0.020  | 0.0022 / 0.0036 | 0.9980 | 0.999996  0.0027
    streams 0 / 1 swapped     0.0048 [0.9938, 1.0060] 0.027  | 0.0043 / 0.027  | 0.105  / 0.29   | 0.0024 / 0.0078 | 0.9978 | 0.999995  0.0031
    streams 0 / 2 swapped     0.058  [0.9887, 1.1922] 0.27   | 0.058  / 0.27   | 0.048  / 0.36   | 0.023  / 0.069  | 0.9576 | 0.999813  0.019
    streams 1 / 2 swapped     0.20   [0.9879, 1.3234] 0.47   | 0.20   / 0.47   | 0.12   / 0.60   | 0.055  / 0.14   | 0.9176 | 0.998868  0.048
    stream 0 dropped          0.022  [0.9124, 0.9994] 0.093  | 0.022  / 0.093  | 0.35   / 0.81   | 0.0041 / 0.014  | 0.9949 | 0.999986  0.0053
    stream 1 dropped          0.012  [0.9567, 0.9998] 0.050  | 0.012  / 0.050  | 0.32   / 0.81   | 0.0026 / 0.0095 | 0.9974 | 0.999992  0.0039
    stream 3 dropped          0.63   [0.1735, 0.9348] 1.13   | 0.63   / 1.13   | 0.58   / 1.13   | 0.46   / 0.88   | 0.4639 | 0.921     0.39
    no gating (pre = 1)       1.95   [1.7295, 3.8515] 2.99   | 1.95   / 2.99   | 1.92   / 3.85   | 0.059  / 0.21   | 0.9110 | 0.998562  0.054
    post gates instead        0.63   [0.0835, 0.6400] 0.92   | 0.63   / 0.92   | 0.71   / 1.11   | 0.36   / 0.80   | 0.5175 | 0.961     0.28
    gate j on stream j + 1    0.45   [0.5679, 1.0448] 1.15   | 0.45   / 1.15   | 0.57   / 3.60   | 0.22   / 0.63   | 0.6949 | 0.977     0.22
    last row zeroed           0.033  [0.0,    1.0003] 1.0    | 0.033  / 1.0    | 0.040  / 1.0    | 0.022  / 0.80   | 0.9985 | 0.999798  0.020
    rows 1023 / 1024 swapped  0.015  [0.6910, 1.4471] 1.35   | 0.015  / 1.35   | 0.028  / 2.70   | 0.011  / 0.51   | 0.9980 | 0.999965  0.0084
    gate rows shifted by 1    0.24   [0.4898, 2.1142] 1.60   | 0.24   / 1.60   | 0.57   / 2.37   | 0.070  / 0.42   | 0.8984 | 0.998279  0.060
    SP row halves swapped     1.26   [0.0736, 13.58]  13.3   | 1.26   / 13.3   | 1.27   / 18.8   | 0.55   / 1.00   | 0.4125 | 0.857     0.52
    TP column halves swapped  1.41   [0.9998, 1.0003] 1.46   | 1.41   / 1.46   | 1.41   / 1.45   | 0.61   / 1.12   | 0.3337 | 0.842     0.54
    zero stub                 1.0    [0.0,    0.0]    1.0    | 1.0    / 1.0    | 1.0    / 1.0    | 0.51   / 0.94   | 0.3967 | 0.871     0.49

Only the zero stub, SP / TP swaps, "post gates instead", "gate j on stream j + 1" and a dropped stream 3 fail the
0.98 out gate; pre x 1.02, a dropped stream 0 / 1, stream swaps and row bugs all pass it. So the test also asserts
(informational metrics):
  - attn_hc (the gates, from swap 01, the component test's limits): not a CPU bridge, 8 columns, finite, rel L2
    <= 0.01, rel L2 per column <= 0.01, worst row rel L2 over the post columns <= 0.015;
  - attn_x vs golden: finite, element count, rel L2 <= 0.005, every row's norm ratio in [0.996, 1.004] (pre x 1.005:
    1.0048), worst row <= 0.01 (streams 0 / 1 swapped 0.027);
  - attn_x vs the CPU hc_pre on the same inputs (block input + the device gates): rel L2 <= 0.003, worst row
    <= 0.006 (the component test's limits; separates the step's own error from the device gates' error);
  - the attn_hc_pre module once more with the device gates' pre columns rotated by (row mod 4), vs the CPU step on
    the same inputs: rel L2 <= 0.004, worst row <= 0.01 (every stream meets the large gate on a quarter of the rows;
    the clearest check of stream order, 0 / 1 swapped 0.105);
  - h_mid rel L2 <= 0.005, worst (row, stream) <= 0.02; router top-8 overlap >= 0.98; block out finite, rel L2
    <= 0.01 (as swap 01).
Every mutation above fails at least one check. The out worst (row, stream) rel L2 is recorded, not asserted (0.056
on the fp32 reference: a few near-tie tokens switch experts, see swap 01).
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
