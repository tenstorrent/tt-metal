# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type dense_full (layer 0) with attn_hc_pre swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre

Reviewed (S.dense_full.02.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). attn_hc_pre makes attn_x [S, H] = sum_j pre_j stream_j from the block input and
the (device) gates; attn_x feeds attn_norm (RMSNorm) and through it the attention. Measured on the CPU (golden s4096
chunk 1, 2048 rows; attn_hc_pre replaced by mutations of the fp32 reference, gates from the fp32 CPU attn_hc):

    variant                      attn_x rel / row ratio / worst row | attn_norm rel | h_mid rel | out PCC   rel
    fp32 reference               0.0010 / [0.9981, 1.0016] / 0.0024 | 0.0017        | 0.0017    | 0.999999  0.0017
    bf16 output                  0.0    / [1.0, 1.0]       / 0.0    | 0.0017        | 0.0017    | 0.999999  0.0017
    pre x 1.005                  0.0047 / [1.0030, 1.0066] / 0.0068 | 0.0017        | 0.0017    | 0.999999  0.0017 (passes)
    pre x 1.01                   0.0097 / [1.0080, 1.0116] / 0.0117 | 0.0017        | 0.0017    | 0.999999  0.0017 (passes)
    pre x 1.02                   0.0197 / [1.0180, 1.0216] / 0.0217 | 0.0017        | 0.0017    | 0.999999  0.0017 (passes)
    no gating (pre = 1)          0.0196 / [1.0000, 1.3321] / 0.33   | 0.0017        | 0.0017    | 0.999999  0.0017 (passes)
    gate rows shifted by 1       0.0264 / [0.7506, 1.3321] / 0.33   | 0.0017        | 0.0017    | 0.999999  0.0017 (passes)
    pre gate 3 dropped           0.249  / [0.7436, 0.9991] / 0.26   | 0.0027        | 0.0017    | 0.999998  0.0018 (passes)
    last row zeroed              0.0319 / [0.0,    1.0016] / 1.0    | 0.023         | 0.021     | 0.99961   0.028  (passes)
    post gates instead of pre    0.693  / [0.0578, 0.4809] / 0.94   | 0.234         | 0.065     | 0.99478   0.102  (passes)
    SP row halves swapped        1.36   / [0.0732, 13.67]  / 13.7   | 1.22          | 0.84      | 0.640     1.09
    TP column halves swapped     1.41   / [0.9981, 1.0016] / 1.44   | 1.42          | 0.74      | 0.706     0.94
    zero stub                    1.0    / [0.0,    0.0]    / 1.0    | 1.0           | 0.69      | 0.718     0.88

"(passes)" = passes the 0.98 out gate. At layer 0 the four streams are identical (each is the embedding), so every
pre-mix bug is a per-row scale of attn_x that attn_norm removes: nothing downstream sees it except a zeroed row. So
the test also asserts (informational metrics):
  - attn_hc (the gates, from swap 01): 8 columns, finite, rel L2 <= 0.01, per-column max abs error <= 0.015;
  - attn_x vs golden at the component limits (test_c_dense_full_attn_hc_pre.py): finite, rel L2 <= 0.004, every
    row's norm ratio in [0.995, 1.005], worst row rel L2 <= 0.01 (catches every bug row above; pre x 1.005 by its
    max ratio 1.0066 > 1.005 and rel 0.0047 > 0.004);
  - attn_x vs the CPU hc_pre on the same inputs (block input + the device gates): rel L2 <= 0.004, worst row <= 0.01
    (separates the step's own error from the device gates' error);
  - the attn_hc_pre module once more on synthetic distinct streams (golden stream rows rolled by 7 j, the device
    gates' pre columns rotated by row mod 4) vs the CPU hc_pre: rel L2 <= 0.008, worst row <= 0.02 (as the
    component test: stream swaps score 0.025 / 0.76), the only layer-0 check of stream order;
  - h_mid rel L2 <= 0.01, worst row <= 0.05; block out finite, rel L2 <= 0.01 (as swap 01).
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
SWAPPED = ["attn_hc", "attn_hc_pre"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
HC_MULT = 4
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_ABS_ERR = 0.015  # attn_hc, per column: max |got - want| (golden bf16 rounding up to 0.002)
X_MAX_REL_L2 = 0.004  # attn_x vs golden, and vs the CPU hc_pre on the same inputs
X_RATIO = (0.995, 1.005)  # attn_x per-row ||got|| / ||want|| vs golden
X_MAX_ROW_REL_L2 = 0.01  # attn_x worst row, vs golden and vs the CPU hc_pre on the same inputs
SYN_MAX_REL_L2 = 0.008  # attn_hc_pre on synthetic distinct streams vs the CPU hc_pre
SYN_MAX_ROW_REL_L2 = 0.02
MID_MAX_REL_L2 = 0.01  # h_mid, whole tensor
MID_MAX_ROW_REL_L2 = 0.05  # h_mid, worst token row
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor


def _errors(got, want):
    """rel L2, per-row norm ratio (min, max), worst row rel L2."""
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    wn = want.norm(dim=-1).clamp_min(1e-12)
    ratio = got.norm(dim=-1) / wn
    return rel, ratio.min().item(), ratio.max().item(), ((got - want).norm(dim=-1) / wn).max().item()


def _synthetic(streams, gates):
    """Distinct streams (stream 0 with rows rolled by 7 j) and per-row rotated pre gates (as the component test)."""
    n = streams.shape[0]
    base = streams.view(n, HC_MULT, -1)[:, 0]
    st = torch.stack([torch.roll(base, 7 * j, 0) for j in range(HC_MULT)], 1).reshape(n, -1).contiguous()
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, :HC_MULT] = torch.gather(gates[:, :HC_MULT], 1, idx)
    return st, gs


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

    # attn_hc: the iHC gates [S, 8] (pre 0-3 | post 4-7).
    gates_ok = finite_shape("attn_hc")
    if gates_ok:
        want = gl["attn_hc"].float()
        got = seen["attn_hc"].float().reshape(want.shape)
        rel = _errors(got, want)[0]
        col_err = (got - want).abs().amax(dim=0)
        metrics.record("rel_l2_swap_attn_hc", rel)
        metrics.record("max_abs_err_swap_attn_hc", col_err.max().item())
        print(
            f"attn_hc: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) max_abs_err per column (pre 0-3 | post 4-7)="
            f"{[round(v, 5) for v in col_err.tolist()]} (<= {GATES_MAX_ABS_ERR})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"attn_hc: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_err.tolist()) if v > GATES_MAX_ABS_ERR]
        if bad:
            failures.append(f"attn_hc: max abs error > {GATES_MAX_ABS_ERR} in columns {bad}")

    # attn_x (the swapped step) vs golden, at the component limits.
    if finite_shape("attn_x"):
        rel, rmin, rmax, row = _errors(seen["attn_x"], gl["attn_x"])
        metrics.record("rel_l2_swap_attn_x", rel)
        metrics.record("worst_row_rel_l2_swap_attn_x", row)
        print(
            f"attn_x vs golden: rel_l2={rel:.6f} (<= {X_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(X_RATIO)}) worst_row_rel_l2={row:.5f} (<= {X_MAX_ROW_REL_L2})"
        )
        if rel > X_MAX_REL_L2:
            failures.append(f"attn_x: rel L2 {rel:.5f} > {X_MAX_REL_L2}")
        if not (X_RATIO[0] <= rmin and rmax <= X_RATIO[1]):
            failures.append(f"attn_x: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(X_RATIO)}")
        if row > X_MAX_ROW_REL_L2:
            failures.append(f"attn_x: worst row rel L2 {row:.5f} > {X_MAX_ROW_REL_L2}")

        # vs the CPU hc_pre on the same inputs (block input + the gates the device produced).
        if gates_ok:
            cpu = ref.component(layer, "attn_hc_pre")
            gates = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            same = cpu(rctx, gl["in"].float(), gates).float()
            srel, _, _, srow = _errors(seen["attn_x"], same)
            metrics.record("rel_l2_swap_attn_x_vs_cpu", srel)
            print(f"attn_x vs CPU hc_pre on the same inputs: rel_l2={srel:.6f} worst_row_rel_l2={srow:.5f}")
            if srel > X_MAX_REL_L2 or srow > X_MAX_ROW_REL_L2:
                failures.append(f"attn_x vs CPU on the same inputs: rel {srel:.5f} / worst row {srow:.5f}")

            # Stream order: the module again on synthetic distinct streams.
            syn = _synthetic(gl["in"].float(), gates)
            syn_want = cpu(rctx, *syn).float()
            syn_out = muts["attn_hc_pre"](rctx, dctx, *syn)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"attn_hc_pre synthetic streams: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, _, _, yrow = _errors(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_attn_x", yrel)
                print(
                    f"attn_hc_pre on synthetic distinct streams: rel_l2={yrel:.6f} (<= {SYN_MAX_REL_L2}) "
                    f"worst_row_rel_l2={yrow:.5f} (<= {SYN_MAX_ROW_REL_L2})"
                )
                if yrel > SYN_MAX_REL_L2 or yrow > SYN_MAX_ROW_REL_L2:
                    failures.append(
                        f"attn_hc_pre synthetic streams: rel {yrel:.5f} / worst row {yrow:.5f} (stream order?)"
                    )

    # h_mid.
    if finite_shape("h_mid"):
        rel, _, _, wr = _errors(seen["h_mid"], gl["h_mid"])
        metrics.record("rel_l2_swap_h_mid", rel)
        metrics.record("worst_row_rel_l2_swap_h_mid", wr)
        print(f"h_mid: rel_l2={rel:.6f} (<= {MID_MAX_REL_L2}) worst_row_rel_l2={wr:.6f} (<= {MID_MAX_ROW_REL_L2})")
        if rel > MID_MAX_REL_L2:
            failures.append(f"h_mid rel L2 {rel:.5f} > {MID_MAX_REL_L2}")
        if wr > MID_MAX_ROW_REL_L2:
            failures.append(f"h_mid worst row rel L2 {wr:.5f} > {MID_MAX_ROW_REL_L2}")

    # Block out.
    out_rel = _errors(seen["out"], gl["out"])[0]
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
