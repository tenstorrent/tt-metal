"""Swap test 4: block type full_dense (layer 0) with ffn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (full_dense) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    attn_residual
    ffn_norm

Reviewed (S.full_dense.04.test.1): same body as test_swap_full_dense_03_attn_residual.py (golden s4096 chunk 1, KV
prefix from the golden state) with ffn_norm added. Gated metric pcc_swap_out (PCC, float [2048, 4096], spec block
threshold 0.98), which on its own is weak (residual dominates `out`; a zero ffn_norm gives mlp(0) = 0, out = h_mid).
Extra asserted checks (informational metrics):
  - each swapped float step's own output vs golden: finite, PCC >= spec component threshold, rel L2 <= 0.03
    (attn_norm, ffn_norm) / 0.015 (attention) / 0.01 (attn_residual), per-token norm ratio in [0.97, 1.03]
    (attn_norm, attention, ffn_norm: the ffn_norm component test's limits, which catch sum-instead-of-mean,
    eps 1e-2, 1 + w and the wrong norm weight) or [0.99, 1.01] (attn_residual); attention and attn_residual also on
    the first 128 rows;
  - block out: finite, rel L2 <= 0.01, whole chunk and first 128 rows.
ffn_norm here reads h_mid with the device attn_norm + attention + attn_residual error.
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
SWAPPED = ["attn_norm", "attention", "attn_residual", "ffn_norm"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
# swapped step output, whole chunk
STEP_MAX_REL_L2 = {"attn_norm": 0.03, "attention": 0.015, "attn_residual": 0.01, "ffn_norm": 0.03}
HEAD_ROWS = 128  # first rows of the chunk: attend mostly into the KV prefix, most exposed to causality / RoPE offset
HEAD_ROW_STEPS = {"attention", "attn_residual"}  # stateful steps whose first HEAD_ROWS rows are checked separately
STEP_ROW_NORM_RATIO = {
    "attn_norm": (0.97, 1.03),
    "attention": (0.97, 1.03),
    "attn_residual": (0.99, 1.01),
    "ffn_norm": (0.97, 1.03),
}
# swapped step output: per-token ||got|| / ||want||
OUT_MAX_REL_L2 = 0.01  # block output, whole chunk and first HEAD_ROWS rows


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
        o = _step(ref, layer, name).output
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
    assert not failures, "; ".join(failures)
