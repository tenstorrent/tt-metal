# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 6: block type full_dense (layer 0) with mlp_residual swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (full_dense) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    attn_residual
    ffn_norm
    mlp
    mlp_residual

Reviewed (S.full_dense.06.test.1): same body as test_swap_full_dense_05_mlp.py (golden s4096 chunk 1, KV prefix
from the golden state) with mlp_residual added; the whole dense block now runs on the device. mlp_residual's output
is the block output `out`, so the gated metric pcc_swap_out (PCC, float [2048, 4096], spec block threshold 0.98) is
also the swapped step's own output. PCC alone misses residual scale and row bugs (test_c_full_dense_mlp_residual.py:
2x output, a zeroed row, the last 32 rows zeroed all pass PCC 0.99). Extra asserted checks (informational metrics,
not in the runner's threshold list):
  - each swapped float step's own output vs golden: finite, PCC >= spec component threshold, rel L2 <= 0.03
    (attn_norm, ffn_norm) / 0.015 (attention, mlp) / 0.01 (attn_residual, mlp_residual), per-token norm ratio in
    [0.97, 1.03] (attn_norm, attention, ffn_norm), [0.98, 1.02] (mlp) or [0.99, 1.01] (attn_residual,
    mlp_residual); attention, attn_residual and mlp_residual also on the first 128 rows;
  - mlp and mlp_residual vs the CPU step on the same device inputs (known_issues "What a swap test can and cannot
    see"): mlp at the component test's limits (rel L2 <= 0.015, ratio [0.98, 1.02], worst per-token rel L2 <= 0.05);
    mlp_residual (out = h_mid + mlp_out on the device h_mid / mlp_out) rel L2 <= 0.01, ratio [0.99, 1.01], worst
    per-token rel L2 <= 0.02. This isolates the device add from upstream error, so a dropped operand, a scale or a
    zeroed row fails even where the golden comparison has headroom;
  - block out: finite, rel L2 <= 0.01, whole chunk and first 128 rows.
Measured (device: TtRMSNorm, TtFullAttention, residual add, TtRMSNorm, TtDenseMLP, residual add): pcc_swap_out
0.999993, out rel 0.0042 / first rows 0.0044, ratio [0.9988, 1.0064]; mlp_residual vs CPU on same inputs rel 0.0019
ratio [1.0004, 1.0019] worst row 0.0024; upstream steps as in swap 05 (mlp rel 0.0046).
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
SWAPPED = ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp", "mlp_residual"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
# swapped step output vs golden, whole chunk
STEP_MAX_REL_L2 = {
    "attn_norm": 0.03,
    "attention": 0.015,
    "attn_residual": 0.01,
    "ffn_norm": 0.03,
    "mlp": 0.015,
    "mlp_residual": 0.01,
}
HEAD_ROWS = 128  # first rows of the chunk: attend mostly into the KV prefix, most exposed to causality / RoPE offset
# steps whose first HEAD_ROWS rows are checked separately
HEAD_ROW_STEPS = {"attention", "attn_residual", "mlp_residual"}
STEP_ROW_NORM_RATIO = {
    "attn_norm": (0.97, 1.03),
    "attention": (0.97, 1.03),
    "attn_residual": (0.99, 1.01),
    "ffn_norm": (0.97, 1.03),
    "mlp": (0.98, 1.02),
    "mlp_residual": (0.99, 1.01),
}
# swapped step output: per-token ||got|| / ||want||
OUT_MAX_REL_L2 = 0.01  # block output, whole chunk and first HEAD_ROWS rows
# Stateless steps also checked against the CPU step on the same (device) inputs, at component-test limits:
# (max rel L2, per-token norm ratio, max worst per-token rel L2)
CPU_SAME_INPUT = {
    "mlp": (0.015, (0.98, 1.02), 0.05),
    "mlp_residual": (0.01, (0.99, 1.01), 0.02),
}


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
    assert not failures, "; ".join(failures)
