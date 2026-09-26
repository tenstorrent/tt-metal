# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 4: block type global (layer 5) with attn_residual swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 5 (global) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    post_attn_norm
    attn_residual

Reviewed (S.global.04.test.1): same golden and checks as global swap 3 (test_swap_global_03_post_attn_norm.py, s4096
chunk 1, HEAD_ROWS = 128), with the isolation check extended to the residual as in sliding swap 4
(test_swap_sliding_04_attn_residual.py): PCC misses ``2 * (a + b)`` and zeroed rows (see test_c_sliding_attn_residual.py).
Gated: pcc_swap_out (PCC, float [2048, 2816], spec block threshold 0.98). Extra asserted checks:
  - each swapped float step's own output vs golden: PCC >= spec component threshold, rel L2 <= 0.03;
  - attention output over the first HEAD_ROWS rows: rel L2 <= 0.03, per-token norm ratio in [0.95, 1.05] (RoPE from 0);
  - block out rel L2 <= 0.02, whole chunk and first HEAD_ROWS rows;
  - every swapped norm or residual step vs the CPU step applied to the exact inputs the device step received (upstream
    device error does not count): rel L2 <= 0.03 and per-token norm ratio in [0.97, 1.03]. PCC ignores scale;
  - finite outputs.
CPU reference: pcc_swap_out 0.999996, block out rel 0.0029 / 0.0026, attn_residual rel 0.0023; zero stub fails (every check).
Device (first run): pcc_swap_out 0.999989, block out rel 0.0046 / 0.0035, attn_out rel 0.0079 / 0.0070, ratio [0.9964, 1.0117],
attn_residual rel 0.0034 vs golden, 0.0017 vs CPU add on its inputs, ratio [0.9990, 1.0019].
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
BLOCK_TYPE = "global"
SWAPPED = ["attn_norm", "attention", "post_attn_norm", "attn_residual"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = 0.03  # swapped step output: ||got - want|| / ||want||, whole chunk
HEAD_ROWS = 128  # first rows of the chunk: the largest share of their attention goes to the KV prefix
STEP_MAX_REL_L2_HEAD_ROWS = 0.03  # attention output, first HEAD_ROWS rows
STEP_ROW_NORM_RATIO = (0.95, 1.05)  # attention output, per-token ||got|| / ||want|| (as the component test)
OUT_MAX_REL_L2 = 0.02  # block output, whole chunk and first HEAD_ROWS rows (MoE routing amplifies attention error)
HEAD_ROW_STEPS = {"attention"}  # stateful steps whose first HEAD_ROWS rows and row norms are checked separately
ISO_STEP_KINDS = ("norm", "residual")  # steps checked against the CPU step on the same (device) inputs
ISO_MAX_REL_L2 = 0.03  # norm/residual step vs the CPU step applied to the same (device) input
ROW_NORM_RATIO = (0.97, 1.03)  # norm/residual step: per-token ||got|| / ||cpu step(same input)||


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
        got = seen[o].float().reshape(want.shape)
        v = metrics.pcc(got, want.float())
        rel = _rel(got, want)
        metrics.record(f"rel_l2_swap_{o}", rel)
        msg = f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={rel:.6f} (<= {STEP_MAX_REL_L2})"
        if not torch.isfinite(got).all():
            failures.append(f"swapped step {name}: non-finite output")
        if v < comp_thr or rel > STEP_MAX_REL_L2:
            failures.append(f"swapped step {name}: pcc {v:.4f} / rel L2 {rel:.4f} (scale, weight or state bug)")
        if name in HEAD_ROW_STEPS:
            w = want.float()
            r = min(HEAD_ROWS, w.shape[0])
            rel_head = _rel(got[:r], w[:r])
            ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
            rmin, rmax = ratio.min().item(), ratio.max().item()
            metrics.record(f"rel_l2_head_rows_swap_{o}", rel_head)
            metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
            metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
            msg += (
                f" first_{r}_rows={rel_head:.6f} (<= {STEP_MAX_REL_L2_HEAD_ROWS})"
                f" row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {list(STEP_ROW_NORM_RATIO)})"
            )
            if rel_head > STEP_MAX_REL_L2_HEAD_ROWS:
                failures.append(
                    f"swapped step {name}: rel L2 on the first {r} rows {rel_head:.4f} > {STEP_MAX_REL_L2_HEAD_ROWS} "
                    "(KV prefix ignored or RoPE positions not offset by the chunk start?)"
                )
            if not (STEP_ROW_NORM_RATIO[0] <= rmin and rmax <= STEP_ROW_NORM_RATIO[1]):
                failures.append(
                    f"swapped step {name}: per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {list(STEP_ROW_NORM_RATIO)}"
                )
        if _step(ref, layer, name).kind in ISO_STEP_KINDS:
            # Isolate the step from the upstream device error: the CPU step on the exact inputs the device step saw.
            st = _step(ref, layer, name)
            ins = [seen[i].float().reshape(gl[i].shape) if i in gl else seen[i].float() for i in st.inputs]
            iso = ref.component(layer, name)(rctx, *ins).float().reshape(want.shape)
            iso_rel = _rel(got, iso)
            ratio = got.norm(dim=-1) / iso.norm(dim=-1).clamp_min(1e-12)
            rmin, rmax = ratio.min().item(), ratio.max().item()
            metrics.record(f"rel_l2_iso_swap_{o}", iso_rel)
            metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
            metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
            msg += f" iso_rel_l2={iso_rel:.6f} (<= {ISO_MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {list(ROW_NORM_RATIO)})"
            if iso_rel > ISO_MAX_REL_L2 or not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
                failures.append(
                    f"swapped step {name}: vs CPU step on the same input rel L2 {iso_rel:.4f}, "
                    f"row-norm ratio [{rmin:.4f}, {rmax:.4f}] (weight or scale bug)"
                )
        print(msg)
    assert not failures, "; ".join(failures)
