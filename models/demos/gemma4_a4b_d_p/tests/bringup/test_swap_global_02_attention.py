# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type global (layer 5) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 5 (global) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention

Reviewed (S.global.02.test.1): same structure as test_swap_sliding_02_attention.py, with the global component test's
limits (test_c_global_attention.py). Golden s4096 chunk 1 (start 2048, KV prefix [0, 2048) from the golden state).
The gated metric is pcc_swap_out (PCC, float [2048, 2816], spec block threshold 0.98). On its own it misses the
stateful attention bugs: the residual dominates `out`, and on the global layer RoPE positions counted from 0 score
attn_out PCC 0.9991 (C.global.attention.test), most of the error on the first rows of the chunk. A global layer has no
window, so the "prefix rows" are the first HEAD_ROWS = 128 rows (RoPE-from-0 attn_out rel L2: whole 0.042, first 128
rows 0.062, norm ratio [0.85, 1.22]; bf16 act + bfp8 weights 0.0076).
Extra asserted checks (informational metrics, not in the runner's threshold list):
  - each swapped float step's own output vs golden: PCC >= spec component threshold, rel L2 <= 0.03 (whole chunk);
  - the attention output over the first HEAD_ROWS rows: rel L2 <= 0.03, and per-token norm ratio in [0.95, 1.05];
  - block out rel L2 <= 0.02, whole chunk and first HEAD_ROWS rows (as sliding swap 2: the MoE amplifies attention
    error through router flips, see known_issues "MoE amplifies attention error at block out");
  - finite outputs.
The K/V the attention writes to the state is not returned here; the state metrics check it.
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
SWAPPED = ["attn_norm", "attention"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = 0.03  # swapped step output: ||got - want|| / ||want||, whole chunk
HEAD_ROWS = 128  # first rows of the chunk: the largest share of their attention goes to the KV prefix
STEP_MAX_REL_L2_HEAD_ROWS = 0.03  # attention output, first HEAD_ROWS rows
STEP_ROW_NORM_RATIO = (0.95, 1.05)  # attention output, per-token ||got|| / ||want|| (as the component test)
OUT_MAX_REL_L2 = 0.02  # block output, whole chunk and first HEAD_ROWS rows (MoE routing amplifies attention error)
HEAD_ROW_STEPS = {"attention"}  # stateful steps whose first HEAD_ROWS rows and row norms are checked separately


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
        print(msg)
    assert not failures, "; ".join(failures)
