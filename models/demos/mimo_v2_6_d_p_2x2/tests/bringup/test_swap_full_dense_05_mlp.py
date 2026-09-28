# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 5: block type full_dense (layer 0) with mlp swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (full_dense) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    attn_residual
    ffn_norm
    mlp

Reviewed (mimo_v2_6_d_p_2x2 S.full_dense.05.test.1): this run's reviewed swap 04 (golden s4096 chunk 1, KV prefix from
the golden state, plus the attn_residual addend check) with mlp added, as the 1x4 prior's frozen swap 05 does.
Gated metric pcc_swap_out (PCC, float [2048, 4096], spec block threshold 0.98), which on its own is weak: the residual
dominates `out`, and PCC misses mlp scale and row bugs (test_c_full_dense_mlp.py: 2x output, a missing TP shard,
zeroed rows all pass PCC). Extra asserted checks (informational metrics, not in the runner's threshold list):
  - each swapped float step's own output vs golden: finite, PCC >= spec component threshold, rel L2 <= 0.03
    (attn_norm, ffn_norm) / 0.015 (attention, mlp) / 0.01 (attn_residual), per-token norm ratio in [0.97, 1.03]
    (attn_norm, attention, ffn_norm), [0.98, 1.02] (mlp) or [0.99, 1.01] (attn_residual); attention and
    attn_residual also on the first 128 rows. mlp vs golden uses the mlp component test's limits although its input
    carries the upstream device error;
  - mlp vs the CPU mlp on the same device ffn_norm output (known_issues "What a swap test can and cannot see"): the
    component test's limits, rel L2 <= 0.015, per-token norm ratio in [0.98, 1.02], worst per-token rel L2 <= 0.05.
    This isolates the device mlp from upstream error, so a 1.02x scale, a missing or doubled TP shard (on 2x2: a
    down reduce over one mesh axis only) or zeroed rows fail even where block out cannot see them;
  - block out: finite, rel L2 <= 0.01, whole chunk and first 128 rows;
  - the attention term of h_mid (delta = h_mid - in vs the attn_out the residual received): coef in [0.95, 1.05],
    rel <= 0.3.
Measured on 2x2: BRINGUP_IMPL=reference out 0.999999 / rel 0.0017, mlp rel 0.0017, PASS; zero stub fails. Device
(TtRMSNorm, TtFullAttention, residual add, TtRMSNorm, TtDenseMLP): pcc_swap_out 0.999994, out rel 0.0048 / first rows
0.0045; mlp vs golden pcc 0.999994 rel 0.0037 ratio [0.9968, 1.0076]; mlp vs CPU on same inputs rel 0.0034 ratio
[1.0009, 1.0044] worst row 0.0048; upstream steps as in swap 04 (ffn_norm rel 0.0051), coef 1.0005.
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
BLOCK_TYPE = "full_dense"
SWAPPED = ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
# swapped step output, whole chunk
STEP_MAX_REL_L2 = {"attn_norm": 0.03, "attention": 0.015, "attn_residual": 0.01, "ffn_norm": 0.03, "mlp": 0.015}
HEAD_ROWS = 128  # first rows of the chunk: attend mostly into the KV prefix, most exposed to causality / RoPE offset
HEAD_ROW_STEPS = {"attention", "attn_residual"}  # stateful steps whose first HEAD_ROWS rows are checked separately
STEP_ROW_NORM_RATIO = {
    "attn_norm": (0.97, 1.03),
    "attention": (0.97, 1.03),
    "attn_residual": (0.99, 1.01),
    "ffn_norm": (0.97, 1.03),
    "mlp": (0.98, 1.02),
}
# swapped step output: per-token ||got|| / ||want||
OUT_MAX_REL_L2 = 0.01  # block output, whole chunk and first HEAD_ROWS rows
ATTN_COEF = (0.95, 1.05)  # <h_mid - in, attn_out> / ||attn_out||^2
MAX_ATTN_REL = 0.3  # ||(h_mid - in) - attn_out|| / ||attn_out||
# Stateless steps also checked against the CPU step on the same (device) inputs, at component-test limits:
# (max rel L2, per-token norm ratio, max worst per-token rel L2)
CPU_SAME_INPUT = {"mlp": (0.015, (0.98, 1.02), 0.05)}


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "swap chunk must start after 0 so the attention reads a real KV prefix"
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    overrides = {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
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
    want_out = gl["out"].float()
    got_out = seen["out"].float().reshape(want_out.shape)
    rows = min(HEAD_ROWS, want_out.shape[0])
    out_rel = _rel(got_out, want_out)
    out_rel_head = _rel(got_out[:rows], want_out[:rows])
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("rel_l2_head_rows_swap_out", out_rel_head)
    print(f"rel_l2_swap_out={out_rel:.6f} first_{rows}_rows={out_rel_head:.6f} (<= {OUT_MAX_REL_L2})")
    if not torch.isfinite(got_out).all():
        failures.append("block out non-finite")
    if out_rel > OUT_MAX_REL_L2 or out_rel_head > OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} / first {rows} rows {out_rel_head:.4f} > {OUT_MAX_REL_L2}")
    comp_thr = threshold(S, "component")
    for name in SWAPPED:
        st = _step(ref, layer, name)
        o = st.output
        want = gl[o]
        if not want.is_floating_point():
            continue
        if seen[o].numel() != want.numel():
            failures.append(f"swapped step {name}: shape {tuple(seen[o].shape)} vs golden {tuple(want.shape)}")
            continue
        got, w = seen[o].float().reshape(want.shape), want.float()
        v = metrics.pcc(got, w)
        rel = _rel(got, w)
        lim = STEP_MAX_REL_L2[name]
        ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
        rmin, rmax = ratio.min().item(), ratio.max().item()
        metrics.record(f"rel_l2_swap_{o}", rel)
        metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
        msg = (
            f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={rel:.6f} (<= {lim}) "
            f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {STEP_ROW_NORM_RATIO[name]})"
        )
        if not torch.isfinite(got).all():
            failures.append(f"swapped step {name}: non-finite output")
        if v < comp_thr or rel > lim:
            failures.append(f"swapped step {name}: pcc {v:.4f} / rel L2 {rel:.4f} > {lim} (scale, weight or state bug)")
        if not (STEP_ROW_NORM_RATIO[name][0] <= rmin and rmax <= STEP_ROW_NORM_RATIO[name][1]):
            failures.append(f"swapped step {name}: per-token norm ratio [{rmin:.4f}, {rmax:.4f}]")
        if name in HEAD_ROW_STEPS:
            r = min(HEAD_ROWS, w.shape[0])
            rel_head = _rel(got[:r], w[:r])
            metrics.record(f"rel_l2_head_rows_swap_{o}", rel_head)
            msg += f" first_{r}_rows={rel_head:.6f} (<= {lim})"
            if rel_head > lim:
                failures.append(
                    f"swapped step {name}: rel L2 on the first {r} rows {rel_head:.4f} > {lim} "
                    "(RoPE positions not offset by the chunk start, causal mask, or KV prefix mishandled?)"
                )
        print(msg)
        if name in CPU_SAME_INPUT:
            # The CPU step on the very inputs the device step saw: isolates this step from upstream device error.
            c_rel, c_ratio, c_row = CPU_SAME_INPUT[name]
            cpu = ref.component(layer, name)(rctx, *[seen[i] if i in seen else gl[i] for i in st.inputs])
            cw = cpu.float().reshape(want.shape)
            crel = _rel(got, cw)
            cn = cw.norm(dim=-1).clamp_min(1e-12)
            cr = got.norm(dim=-1) / cn
            crmin, crmax = cr.min().item(), cr.max().item()
            crow = ((got - cw).norm(dim=-1) / cn).max().item()
            metrics.record(f"rel_l2_vs_cpu_swap_{o}", crel)
            metrics.record(f"row_norm_ratio_min_vs_cpu_swap_{o}", crmin)
            metrics.record(f"row_norm_ratio_max_vs_cpu_swap_{o}", crmax)
            metrics.record(f"worst_row_rel_l2_vs_cpu_swap_{o}", crow)
            print(
                f"swapped {name} vs CPU on same inputs: rel_l2={crel:.6f} (<= {c_rel}) "
                f"row_norm_ratio=[{crmin:.4f}, {crmax:.4f}] (in {c_ratio}) worst_row_rel_l2={crow:.4f} (<= {c_row})"
            )
            if crel > c_rel:
                failures.append(f"swapped step {name}: rel L2 vs CPU on same inputs {crel:.4f} > {c_rel}")
            if not (c_ratio[0] <= crmin and crmax <= c_ratio[1]):
                failures.append(f"swapped step {name}: per-token norm ratio vs CPU [{crmin:.4f}, {crmax:.4f}]")
            if crow > c_row:
                failures.append(f"swapped step {name}: worst per-token rel L2 vs CPU {crow:.4f} > {c_row}")

    # The attention term of the residual, on h_mid - in, against the attn_out the residual step received.
    res = gl["in"].float()
    attn = seen["attn_out"].float().reshape(res.shape)
    delta = seen["h_mid"].float().reshape(res.shape) - res
    coef = ((delta * attn).sum() / (attn * attn).sum().clamp_min(1e-30)).item()
    arel = ((delta - attn).norm() / attn.norm().clamp_min(1e-30)).item()
    metrics.record("attn_coef_swap_h_mid", coef)
    metrics.record("attn_rel_l2_swap_h_mid", arel)
    print(f"attn term of h_mid: coef={coef:.4f} (in {ATTN_COEF}) rel={arel:.4f} (<= {MAX_ATTN_REL})")
    if not (ATTN_COEF[0] <= coef <= ATTN_COEF[1]):
        failures.append(
            f"attn_residual: attn_out coefficient {coef:.4f} outside {ATTN_COEF} (dropped or scaled attn_out?)"
        )
    if not arel <= MAX_ATTN_REL:
        failures.append(
            f"attn_residual: attn term rel L2 {arel:.4f} > {MAX_ATTN_REL} (attn_out misaligned or corrupted)"
        )
    assert not failures, "; ".join(failures)
