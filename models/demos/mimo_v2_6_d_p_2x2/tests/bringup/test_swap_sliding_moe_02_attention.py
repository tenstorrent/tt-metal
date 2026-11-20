# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type sliding_moe (layer 1) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (sliding_moe) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention

Reviewed (S.sliding_moe.02.test.1, 1x4 prior mimo_v2_6_d_p; carried unchanged to the 2x2 run, same golden and CPU
reference, so the measurements below are the prior's): golden s4096 chunk 1 (start 2048, KV prefix [0, 2048) from the golden state; the
sliding window 128 means only the first 127 query rows read the prefix). Same body as test_swap_full_dense_02_attention.py
with the C.sliding_moe.* component limits. The gated metric is pcc_swap_out (PCC, float [2048, 4096], spec block
threshold 0.98); on its own it is weak: the residual dominates `out`, and on this layer the attention sink takes most
of the softmax mass, so attn_out is small next to the residual. Extra asserted checks (informational metrics, not in
the runner's threshold list):
  - each swapped float step's own output vs golden: finite, PCC >= spec component threshold;
    attn_norm: rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03] (C.sliding_moe.attn_norm / swap 01 limits);
    attention: rel L2 <= 0.022 whole chunk and first 128 rows, per-token norm ratio in [0.95, 1.08], worst per-token
    rel L2 <= 0.08 (C.sliding_moe.attention limits 0.02 / [0.95, 1.05] / 0.08, widened here as noted below; here the
    attention input is the device attn_norm, so the check covers the chain);
  - attention window discriminator, as in the component test but against the CPU attention on the same (device)
    attn_norm input: rel L2 to the CPU attention at window 128 <= 0.015 (component noise estimate 0.0072, x1.02 0.020),
    and the output must be closer to it than to the CPU attention at windows 127 and 129 (an off-by-one window sits
    inside the size limits);
  - block out: finite, rel L2 <= 0.01 whole chunk and first 128 rows. Not 0.02 as in gemma4 swap_sliding_02: here the
    device scores 0.0031 and a zeroed attention output (the precompile pass) only 0.0142, since the sink keeps
    attn_out small next to the residual.
Measured: BRINGUP_IMPL=reference out 0.999996 / rel 0.0030 / first rows 0.0029; attention rel 0.0038 / first rows
0.0036, ratio [0.9891, 1.0107], worst row 0.011; vs CPU w128 0, w127 0.0085, w129 0.0093. Zero stub fails every
check. Device (TtRMSNorm + TtSlidingAttention): pcc_swap_out 0.999995, out rel 0.0031 / 0.0029; attn_norm rel 0.0030,
ratio [0.9934, 1.0058]; attention rel 0.0145 / first rows 0.0128, ratio [0.9882, 1.0420], worst row 0.042 (component
test on the golden input: 0.0086; the device attn_norm error is amplified by the sink); vs CPU on the same input w128
0.0070, w127 0.0081, w129 0.0141. The window-127 margin is small (0.0011): noise added to this path (bf8 KV, HiFi2,
a noisier attn_norm) must re-check it.
The K/V the attention writes to the state is not returned here; the state metrics check it.
Limits widened 2026-09-28 (owner decision): attention rel L2 0.02 -> 0.022, per-token norm ratio upper bound 1.05 ->
1.08; every other limit is unchanged. Why: the owner keeps the sliding SDPA preset "S" (P.2; fp32 accumulation off),
chosen for speed (sliding SDPA 2.5 ms against 19.7 ms at base). On its own, with the native norm, it measured attention
rel 0.0235, ratio [0.9791, 1.0729], worst row 0.073. The bring-up norm used to add a +0.1% scale bias on top: its
cross-chunk sum of squares was truncated to bf16 (fixed in ttnn.bringup.rms_norm, CHANGELOG 3). Measured at the
defaults (preset S, bring-up norm): before the norm fix, attention rel 0.0220 / first rows 0.0187, ratio
[0.9420, 1.0319], worst row 0.058; attn_norm rel 0.0033, ratio [0.9939, 1.0067]. After it, attention rel 0.0196 /
first rows 0.0151, ratio [0.9742, 1.0657], worst row 0.066, vs CPU w128 0.0132, w127 0.0166, w129 0.0150; attn_norm rel
0.0029, ratio [0.9935, 1.0061]; pcc_swap_out 0.999995, out rel 0.0032 / 0.0029. The new limits are those numbers plus
a small margin. Full-model accuracy is unchanged: ladder rung last, per-layer PCC L00 0.998576 ... L05 0.998414
(0.998435 before), state_min 0.999280 (0.999283).
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
BLOCK_TYPE = "sliding_moe"
SWAPPED = ["attn_norm", "attention"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = {"attn_norm": 0.03, "attention": 0.022}  # swapped step output, whole chunk (attention: see notes)
STEP_ROW_NORM_RATIO = {"attn_norm": (0.97, 1.03), "attention": (0.95, 1.08)}  # per-token ||got|| / ||want|| (notes)
STEP_MAX_WORST_ROW_REL_L2 = {"attention": 0.08}  # max over tokens of ||got_t - want_t|| / ||want_t||
HEAD_ROWS = 128  # first rows of the chunk: the only rows that attend into the KV prefix
HEAD_ROW_STEPS = {"attention"}  # stateful steps whose first HEAD_ROWS rows are checked separately
WINDOW_ALTERNATIVES = (-1, 1)  # attention must be closer to the CPU window-W attention than to W-1 and W+1
ATTN_MAX_REL_L2_VS_CPU = 0.015  # attention vs the CPU attention on the same device input (device 0.0070, x1.02 0.020)
OUT_MAX_REL_L2 = 0.01  # block output, whole chunk and first HEAD_ROWS rows (device 0.0031, zeroed attention 0.0142)


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _cpu_attention_at_window(ref, layer, g, c, x, window):
    """The CPU reference attention on input x with a different sliding window (restored afterwards)."""
    old = ref.cfg.sliding_window
    ref.cfg.sliding_window = window
    try:
        return ref.component(layer, "attention")(reference_ctx(ref, layer, g, c), x).float()
    finally:
        ref.cfg.sliding_window = old


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "swap chunk must start after 0 so the attention reads a real KV prefix"
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    assert ref.cfg.is_sliding(layer), f"layer {layer} must be a sliding-window layer"
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
        rlo, rhi = STEP_ROW_NORM_RATIO[name]
        ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
        rmin, rmax = ratio.min().item(), ratio.max().item()
        worst = ((got - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)).max().item()
        metrics.record(f"rel_l2_swap_{o}", rel)
        metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
        metrics.record(f"worst_row_rel_l2_swap_{o}", worst)
        msg = (
            f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={rel:.6f} (<= {lim}) "
            f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in [{rlo}, {rhi}]) worst_row_rel_l2={worst:.4f}"
        )
        if not torch.isfinite(got).all():
            failures.append(f"swapped step {name}: non-finite output")
        if v < comp_thr or rel > lim:
            failures.append(f"swapped step {name}: pcc {v:.4f} / rel L2 {rel:.4f} > {lim} (scale, weight or state bug)")
        if not (rlo <= rmin and rmax <= rhi):
            failures.append(f"swapped step {name}: per-token norm ratio [{rmin:.4f}, {rmax:.4f}]")
        if name in STEP_MAX_WORST_ROW_REL_L2 and worst > STEP_MAX_WORST_ROW_REL_L2[name]:
            failures.append(
                f"swapped step {name}: worst per-token rel L2 {worst:.4f} > {STEP_MAX_WORST_ROW_REL_L2[name]}"
            )
        if name in HEAD_ROW_STEPS:
            r = min(HEAD_ROWS, w.shape[0])
            rel_head = _rel(got[:r], w[:r])
            metrics.record(f"rel_l2_head_rows_swap_{o}", rel_head)
            msg += f" first_{r}_rows={rel_head:.6f} (<= {lim})"
            if rel_head > lim:
                failures.append(
                    f"swapped step {name}: rel L2 on the first {r} rows {rel_head:.4f} > {lim} "
                    "(KV prefix ignored, RoPE positions not offset by the chunk start, or window mask wrong?)"
                )
        print(msg)
        if name == "attention":
            # Window discriminator against the CPU attention on the same (device) input.
            x = seen[st.inputs[0]].float().reshape(gl[st.inputs[0]].shape)
            window = int(ref.cfg.sliding_window)
            d_w = _rel(got, _cpu_attention_at_window(ref, layer, g, c, x, window))
            metrics.record(f"rel_l2_vs_cpu_window_swap_{o}", d_w)
            if d_w > ATTN_MAX_REL_L2_VS_CPU:
                failures.append(
                    f"attention rel L2 vs the CPU attention on the same input {d_w:.4f} > {ATTN_MAX_REL_L2_VS_CPU}"
                )
            for dw in WINDOW_ALTERNATIVES:
                alt = window + dw
                d_alt = _rel(got, _cpu_attention_at_window(ref, layer, g, c, x, alt))
                print(f"attention rel_l2 vs CPU window {window}: {d_w:.6f}; vs CPU window {alt}: {d_alt:.6f}")
                if not d_w < d_alt:
                    failures.append(
                        f"attention output closer to a window-{alt} attention (rel {d_alt:.5f}) than to window "
                        f"{window} (rel {d_w:.5f}): key j must be visible to query i iff i - {window} < j <= i"
                    )
    assert not failures, "; ".join(failures)
