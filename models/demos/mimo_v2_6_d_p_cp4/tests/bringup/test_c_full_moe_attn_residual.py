# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_moe.attn_residual.test.1, ported from the prior mimo_v2_6_d_p; same golden, s4096 chunk 1,
[2048, 4096]): h_mid = in + attn_out (plain add, no scale). Layer 5 is full attention with no sink, so attn_out is
large (||in|| 121.4, ||attn_out|| 78.9, ||h_mid|| 126.3). Prior CPU measurements (PCC / rel L2): fp32 reference
0.999997 / 0.0025; bf16 add 0.999996 / 0.0030; 2 * (in + attn_out) 0.999997 (passes) / 1.0; last 32 rows zeroed
0.99247 (passes) / 0.12; attn_out shifted one row 0.9863 / 0.165; attn_out dropped 0.798. Asserted checks: output
size, finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01], and on delta = out - in vs attn_out: coefficient
<delta, attn_out> / ||attn_out||^2 in [0.98, 1.02] and ||delta - attn_out|| / ||attn_out|| <= 0.05 (as the prior
full_moe test; tighter than layer 1's [0.95, 1.05] / 0.3 because bf16 output rounding is small next to this attn_out).
CP=4 (this bring-up): chip r holds rows [r*S/4, (r+1)*S/4). Every check is also asserted per CP slice. Measured on
this golden (per slice: rel L2 / coef / attn rel): bf16 add 0.0030 / 1.0 / 0.0028; golden h_mid 0 / 1.0 / 0.0040;
attn_out slices 1 and 2 swapped 0.163-0.164 / 0.957-0.975 / 0.26; slice 0's attn_out written to slice 1 0.161 /
0.965 / 0.258; slice 3's attn_out dropped 0.626 / 0 / 1.0; slice 3's attn_out halved 0.313 / 0.5 / 0.5; attn_out
shifted one row 0.165 / 0.965 / 0.265 on every slice. Here the whole-chunk rel L2 already catches these (0.08-0.31);
the per-slice checks name the bad chip.
"""

import torch

from models.demos.common.bringup.core import metrics
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
STEP = "attn_residual"
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||, whole chunk and per CP slice
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
ATTN_COEF = (0.98, 1.02)  # <out - in, attn_out> / ||attn_out||^2, whole chunk and per CP slice
MAX_ATTN_REL = 0.05  # ||(out - in) - attn_out|| / ||attn_out||, whole chunk and per CP slice (bf16 rounding: 0.003)
CP = 4


def _attn_term(delta, attn):
    coef = ((delta * attn).sum() / (attn * attn).sum().clamp_min(1e-30)).item()
    arel = ((delta - attn).norm() / attn.norm().clamp_min(1e-30)).item()
    return coef, arel


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale, per-row and per-slice checks (PCC is scale-invariant and barely sees a few bad rows).
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()

    # Check the attention term directly on out - in (catches a scaled or misaligned attn_out).
    res, attn = (x.float().reshape(want.shape) for x in inputs)
    delta = got - res
    coef, arel = _attn_term(delta, attn)

    rows = w.shape[0]
    assert rows % CP == 0, f"chunk rows {rows} not divisible by CP={CP}"
    L = rows // CP
    sl = [slice(r * L, (r + 1) * L) for r in range(CP)]
    slice_rel = [((got[s] - w[s]).norm() / w[s].norm()).item() for s in sl]
    slice_attn = [_attn_term(delta[s], attn[s]) for s in sl]

    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"rel_l2_slice_max_{STEP}_L{LAYER:02d}", max(slice_rel))
    metrics.record(f"attn_coef_{STEP}_L{LAYER:02d}", coef)
    metrics.record(f"attn_rel_l2_{STEP}_L{LAYER:02d}", arel)
    metrics.record(f"attn_rel_l2_slice_max_{STEP}_L{LAYER:02d}", max(a for _, a in slice_attn))
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"slice_rel={[round(x, 5) for x in slice_rel]}"
    )
    print(
        f"attn term: coef={coef:.4f} (in {ATTN_COEF}) rel={arel:.4f} (<= {MAX_ATTN_REL}) "
        f"per slice={[(round(a, 4), round(b, 4)) for a, b in slice_attn]}"
    )

    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale bug, dropped or zeroed rows/columns)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed or padded rows?)"
    worst = max(range(CP), key=lambda r: slice_rel[r])
    assert (
        slice_rel[worst] <= MAX_REL_L2
    ), f"CP slice {worst} relative L2 {slice_rel[worst]:.4f} > {MAX_REL_L2} (slice order or gather bug?)"
    assert (
        ATTN_COEF[0] <= coef <= ATTN_COEF[1]
    ), f"attn_out coefficient {coef:.4f} outside {ATTN_COEF} (dropped or scaled attn_out?)"
    assert arel <= MAX_ATTN_REL, f"attn term rel L2 {arel:.4f} > {MAX_ATTN_REL} (attn_out misaligned or corrupted)"
    for r, (sc, sa) in enumerate(slice_attn):
        assert ATTN_COEF[0] <= sc <= ATTN_COEF[1], f"CP slice {r}: attn_out coefficient {sc:.4f} outside {ATTN_COEF}"
        assert sa <= MAX_ATTN_REL, f"CP slice {r}: attn term rel L2 {sa:.4f} > {MAX_ATTN_REL} (slice misplaced?)"
