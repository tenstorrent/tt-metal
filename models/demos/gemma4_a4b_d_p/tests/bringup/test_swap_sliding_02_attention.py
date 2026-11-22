# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type sliding (layer 0) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (sliding) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention

Reviewed (S.sliding.02.test.1): golden s4096 chunk 1 (start 2048 > sliding window 1024, KV prefix from the golden state).
The gated metric is pcc_swap_out (PCC, float [2048, 2816], spec block threshold 0.98). On its own it misses the stateful
attention bugs: the residual dominates `out`, and only the first ``sliding_window`` query rows read the KV prefix.
Measured on the CPU (block out PCC / rel L2 whole / rel L2 first 1024 rows; attn_out rel L2 whole / first rows):
  reference        0.999996 / 0.0027 / 0.0033;  attn 0.0019 / 0.0019
  bf16 attn i/o    0.999995 / 0.0031 / 0.0035;  attn 0.0022 / 0.0022
  1% noise on attn 0.999975 / 0.0071 / 0.0066;  attn 0.0102 / 0.0101
  no KV prefix     0.998968 / 0.0454 / 0.0641;  attn 0.0641 / 0.0900   (passes 0.98)
  zeroed prefix    0.998904 / 0.0468 / 0.0661;  attn 0.0606 / 0.0850   (passes 0.98)
  RoPE from 0      0.997764 / 0.0669 / 0.0944;  attn 0.0912 / 0.1281   (passes 0.98)
  zero stub        0.603581 / 0.8444
Extra asserted checks (informational metrics, not in the runner's threshold list):
  - each swapped float step's own output vs golden: PCC >= spec component threshold, rel L2 <= 0.03 (whole chunk);
  - the attention output over the first ``sliding_window`` rows: rel L2 <= 0.03 (the rows that read the prefix);
  - block out rel L2 <= 0.02, whole chunk and first ``sliding_window`` rows (catches all the bugs above by > 2x). Not
    0.01 as in swap 1: the device run scores 0.0074 (attn_out 0.0052 grows through router flips in the MoE);
  - finite outputs.
Device (first implementation): pcc_swap_out 0.999973, block out rel 0.0074 / 0.0062, attn_out rel 0.0052 / 0.0052.
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
BLOCK_TYPE = "sliding"
SWAPPED = ["attn_norm", "attention"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = 0.03  # swapped step output: ||got - want|| / ||want||, whole chunk
STEP_MAX_REL_L2_PREFIX_ROWS = 0.03  # attention output, first sliding_window rows (the rows that read the KV prefix)
OUT_MAX_REL_L2 = 0.02  # block output, whole chunk and first sliding_window rows (device 0.0074: MoE routing amplifies)
PREFIX_ROW_STEPS = {"attention"}  # stateful steps whose first sliding_window rows are checked separately


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
    window = int(S.get("checkpoint.config.sliding_window", 1024))
    want_out = gl["out"].float()
    got_out = seen["out"].float().reshape(want_out.shape)
    rows = min(window, want_out.shape[0])
    out_rel = _rel(got_out, want_out)
    out_rel_head = _rel(got_out[:rows], want_out[:rows])
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("rel_l2_prefix_rows_swap_out", out_rel_head)
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
        if name in PREFIX_ROW_STEPS:
            r = min(window, want.shape[0])
            rel_head = _rel(got[:r], want[:r])
            metrics.record(f"rel_l2_prefix_rows_swap_{o}", rel_head)
            msg += f" first_{r}_rows={rel_head:.6f} (<= {STEP_MAX_REL_L2_PREFIX_ROWS})"
            if rel_head > STEP_MAX_REL_L2_PREFIX_ROWS:
                failures.append(
                    f"swapped step {name}: rel L2 on the first {r} rows {rel_head:.4f} > {STEP_MAX_REL_L2_PREFIX_ROWS} "
                    "(KV prefix ignored or RoPE positions not offset by the chunk start?)"
                )
        print(msg)
    assert not failures, "; ".join(failures)
