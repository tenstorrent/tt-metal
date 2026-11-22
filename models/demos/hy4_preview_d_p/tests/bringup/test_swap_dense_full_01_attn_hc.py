# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type dense_full (layer 0) with attn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc

Reviewed (S.dense_full.01.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). attn_hc's gates [S, 8] (pre 4 | post 4) feed attn_hc_pre (attn_x = sum_j pre_j
stream_j), attn_residual (h_mid = stream_j + post_j attn_out) and, through h_mid, everything after. Measured on the
CPU (golden s4096 chunk 1, 2048 rows; the gates replaced by mutations of the fp32 reference):

    variant                      gates rel / col max | attn_x rel | h_mid rel / worst row | out PCC   rel
    fp32 reference               0.00052 / 0.0020    | 0.0010     | 0.0017 / 0.0017       | 0.999999  0.0017
    gates rounded to bf16        0.0     / 0.0       | 0.0010     | 0.0023 / 0.0042       | 0.999996  0.0030
    gates + 1e-3 (random sign)   0.0015  / 0.0030    | 0.0012     | 0.0039 / 0.020        | 0.999977  0.0068
    post x 1.005                 0.0012  / 0.0055    | 0.0010     | 0.0047 / 0.0069       | 0.999989  0.0052
    zero stub                    1.0     / 1.0       | 1.0        | 0.87   / 1.32         | 0.524     1.16
    post = 1 * sigmoid           0.11    / 0.39      | 0.0010     | 0.43   / 0.66         | 0.873     0.49
    one chip's partial mixes     0.12    / 0.72      | 0.051      | 0.40   / 2.49         | 0.928     0.44
    pre | post halves swapped    1.15    / 0.97      | 0.69       | 2.84   / 17.6         | 0.275     3.22
    rows shifted by 1            0.17    / 1.0       | 0.026      | 0.62   / 7.0          | 0.800     0.71
    SP row halves swapped        0.17    / 1.0       | 0.023      | 0.60   / 7.4          | 0.813     0.68
    fn rows 4 / 5 swapped        0.018   / 0.10      | 0.0010     | 0.069  / 0.11         | 0.992     0.125  (passes)
    base 4 / 5 swapped           0.0098  / 0.030     | 0.0010     | 0.038  / 0.067        | 0.998     0.070  (passes)
    post x 1.02                  0.0045  / 0.017     | 0.0010     | 0.017  / 0.026        | 0.99984   0.020  (passes)
    last row's gates zero        0.023   / 1.0       | 0.032      | 0.031  / 0.89         | 0.9993    0.038  (passes)
    pre x 1.02                   0.019   / 0.022     | 0.020      | 0.0017 / 0.0017       | 0.999999  0.0017 (passes)
    pre gate 3 = pre gate 2      0.030   / 1.0       | 0.019      | 0.0017 / 0.0017       | 0.999999  0.0017 (passes)
    fn rows 0 / 1 swapped (pre)  0.0011  / 0.035     | 0.0010     | 0.0017 / 0.0017       | 0.999999  0.0017 (passes)

"(passes)" = passes the 0.98 out gate. The pre gates are invisible downstream at layer 0: the four streams are
identical (each is the embedding), so attn_x = (sum_j pre_j) * embedding, and attn_norm (RMSNorm) removes that
per-row scale. Only attn_x and the gates themselves show a pre bug. So the test also asserts (informational metrics):
  - the swapped step vs golden: finite, 8 columns (a [S, 32] padded tile row would break hc_pre / hc_post), rel L2
    <= 0.01 and max abs error per column <= 0.015 (as test_c_dense_full_attn_hc.py; catches every row above except
    the noise rows);
  - attn_x rel L2 <= 0.01 (catches pre x 1.02, pre 3 = pre 2);
  - h_mid rel L2 <= 0.01 and worst row rel L2 <= 0.05 (catches post x 1.02, base 4 / 5, a zeroed row);
  - block out: finite, rel L2 <= 0.01 (catches fn / base 4 / 5 swaps, post x 1.02, a zeroed row).
The noise rows pass every check with margin. Blind spot (as in the component test): at layer 0 a stream-order bug
(fn's four stream blocks permuted, RMS over one stream) changes nothing; later layers must catch it.
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
BLOCK_TYPE = "dense_full"
SWAPPED = ["attn_hc"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_ABS_ERR = 0.015  # attn_hc, per column: max |got - want| (golden bf16 rounding up to 0.002)
MID_MAX_REL_L2 = 0.01  # attn_x and h_mid, whole tensor
MID_MAX_ROW_REL_L2 = 0.05  # h_mid, worst token row
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _worst_row_rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
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
        rel = _rel(got, want)
        col_err = (got - want).abs().amax(dim=0)
        metrics.record(f"rel_l2_swap_{o}", rel)
        metrics.record(f"max_abs_err_swap_{o}", col_err.max().item())
        print(
            f"swapped {name}: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) max_abs_err per column (pre 0-3 | post 4-7)="
            f"{[round(v, 5) for v in col_err.tolist()]} (<= {GATES_MAX_ABS_ERR})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"swapped step {name}: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_err.tolist()) if v > GATES_MAX_ABS_ERR]
        if bad:
            failures.append(f"swapped step {name}: max abs error > {GATES_MAX_ABS_ERR} in columns {bad}")

    # Its consumers: attn_x (pre gates) and h_mid (post gates).
    for n in ("attn_x", "h_mid"):
        if not torch.isfinite(seen[n]).all():
            failures.append(f"{n}: non-finite")
            continue
        rel = _rel(seen[n], gl[n])
        metrics.record(f"rel_l2_swap_{n}", rel)
        msg = f"{n}: rel_l2={rel:.6f} (<= {MID_MAX_REL_L2})"
        if rel > MID_MAX_REL_L2:
            failures.append(f"{n} rel L2 {rel:.5f} > {MID_MAX_REL_L2}")
        if n == "h_mid":
            wr = _worst_row_rel(seen[n], gl[n])
            metrics.record(f"worst_row_rel_l2_swap_{n}", wr)
            msg += f" worst_row_rel_l2={wr:.6f} (<= {MID_MAX_ROW_REL_L2})"
            if wr > MID_MAX_ROW_REL_L2:
                failures.append(f"{n} worst row rel L2 {wr:.5f} > {MID_MAX_ROW_REL_L2}")
        print(msg)

    # Block out.
    out_rel = _rel(seen["out"], gl["out"])
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
