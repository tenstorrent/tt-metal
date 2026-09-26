# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 7: block type sliding_moe (layer 1) with ffn_residual swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (sliding_moe) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    attn_residual
    ffn_norm
    router
    experts
    ffn_residual

Reviewed (S.sliding_moe.07.test.1): swap 06's checks (below) plus ffn_residual. Its output is the block output `out`
(out = h_mid + experts_out), so pcc_swap_out is also the swapped step's own output; PCC alone misses a scaled or
dropped experts term. The experts term is ~15% of ||out|| (test_c_sliding_moe_ffn_residual.py). Extra checks:
  - vs golden: finite, PCC >= spec component threshold, rel L2 <= 0.01 (whole chunk and first 128 rows, the
    block-out limit), per-token norm ratio in FFN_RES_ROW_NORM_RATIO_GOLDEN (looser: routing flips move rows);
  - vs the CPU add on the same device h_mid / experts_out ("iso"): rel L2 <= 0.01, per-token ratio in [0.99, 1.01],
    worst per-token rel L2 <= 0.02; on delta = out - h_mid: experts coefficient in [0.97, 1.03] and experts-term
    rel L2 <= 0.1 (component test limits).
Measured ffn_residual, golden rel / first rows / ratio; iso rel / ratio / worst row; coef / experts rel:
BRINGUP_IMPL=reference 0.0030 / 0.0029 / [0.9979, 1.0040]; 0 / [1, 1] / 0; 1.0 / 0. Device (all 7 steps on device):
pcc_swap_out 0.999991, 0.0044 / 0.0036 / [0.9958, 1.0042]; 0.0017 / [0.9997, 1.0011] / 0.0023; 1.0014 / 0.0114.
Zero stub fails every check.

Swap 06 review (S.sliding_moe.06.test.1): swap 05's checks (below, from S.sliding_moe.05.test.1) plus experts
(experts_out [2048, 4096], float). The experts input routing comes from the device router, and its near-tie flips
(22..39 of 2048 rows) change which experts run, so per-row checks against the golden are not meaningful. Instead:
  - vs golden: PCC >= 0.99 and whole-chunk rel L2 <= 0.03;
  - vs the CPU experts on the same device ffn_norm + router ("iso"): the component test limits, rel L2 <= 0.03,
    per-token norm ratio in [0.97, 1.03], worst per-token rel L2 <= 0.1 (dropped experts / pairs, scale, zeroed rows).
Measured experts, as golden rel / iso rel / iso ratio / iso worst row: BRINGUP_IMPL=reference 0.0126 / 0 / [1, 1] / 0;
device (TtExperts loop mode) pcc 0.99983, 0.0185 / 0.0098 / [0.9793, 1.0114] / 0.029; block out pcc 0.999992,
rel 0.0040 / first rows 0.0032. Zero stub fails every check.
Swap 05 review: swap 04's body (golden s4096 chunk 1, KV prefix [0, 2048) from the golden state,
window 128; test_swap_sliding_moe_04_ffn_norm.py) with router added. Gated metric pcc_swap_out (PCC, float [2048, 4096],
spec block threshold 0.98), weak on its own. Extra asserted checks (informational metrics):
  - attn_norm, attention, attn_residual, ffn_norm: as swap 04 (own output vs golden, window discriminator, CPU attention
    on the device input, attention term of the residual);
  - router (dense [2048, 256], 8 nonzeros per row, rows sum to 1): whole-matrix rel L2 is dominated by near-tie selection
    flips (8th/9th choice-score gap median 0.0017, test_c_sliding_moe_router.py), so instead: PCC vs golden >= spec
    component threshold, exactly 8 nonzeros per row, no negative weights, every row sum within ROUTER_MAX_ROW_SUM_ERR of 1
    (norm_topk_prob, routed_scaling_factor 1.0), and vs golden / vs the CPU router on the same device ffn_norm ("iso"):
    mean top-8 selection overlap and matched-row weight rel L2 (limits below). The iso check isolates the router from the
    upstream device error;
  - block out: finite, rel L2 <= 0.01 whole chunk and first 128 rows.
Measured: BRINGUP_IMPL=reference out 0.999997 / rel 0.0030; router pcc 0.99922, overlap 0.99866 (golden) / 1.0 (iso),
matched rel 0.0016 / 0.0, row sums 1.0. Zero stub fails every check (router nnz 0). Precompile pass (zero attention):
router overlap vs golden 0.976 fails, out rel 0.0142 fails. Device (TtRMSNorm + TtSlidingAttention + TtResidualAdd +
TtRMSNorm + TtRouter): pcc_swap_out 0.999993, out rel 0.0037 / first rows 0.0029; router pcc 0.99870, overlap 0.99762 /
0.99902, matched rows 2009 / 2032 of 2048, matched rel 0.00147 / 0.00157, nnz 8, row sums [0.9980, 1.0020].
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
BLOCK_TYPE = "sliding_moe"
SWAPPED = ["attn_norm", "attention", "attn_residual", "ffn_norm", "router", "experts", "ffn_residual"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
# swapped step output, whole chunk
STEP_MAX_REL_L2 = {"attn_norm": 0.03, "attention": 0.02, "attn_residual": 0.01, "ffn_norm": 0.03}
STEP_ROW_NORM_RATIO = {
    "attn_norm": (0.97, 1.03),
    "attention": (0.95, 1.05),
    "attn_residual": (0.99, 1.01),
    "ffn_norm": (0.97, 1.03),
}  # per-token ||got|| / ||want||
STEP_MAX_WORST_ROW_REL_L2 = {"attention": 0.08}  # max over tokens of ||got_t - want_t|| / ||want_t||
HEAD_ROWS = 128  # first rows of the chunk: the only rows that attend into the KV prefix
HEAD_ROW_STEPS = {"attention", "attn_residual"}  # stateful steps whose first HEAD_ROWS rows are checked separately
WINDOW_ALTERNATIVES = (-1, 1)  # attention must be closer to the CPU window-W attention than to W-1 and W+1
ATTN_MAX_REL_L2_VS_CPU = 0.015  # attention vs the CPU attention on the same device input (device 0.0070, x1.02 0.020)
ATTN_COEF = (0.95, 1.05)  # attn_residual: <h_mid - in, attn_out> / ||attn_out||^2
MAX_ATTN_REL = 0.3  # attn_residual: ||(h_mid - in) - attn_out|| / ||attn_out|| (bf16 rounding alone 0.15)
OUT_MAX_REL_L2 = 0.01  # block output, whole chunk and first HEAD_ROWS rows
ROUTER_STEPS = {"router"}  # dense [S, 256] routing: selection / weight checks instead of whole-matrix rel L2
TOP_K = int(S.get("checkpoint.config.num_experts_per_tok", 8))
ROUTER_MIN_OVERLAP_GOLDEN = 0.985  # mean top-8 selection overlap vs golden (component test limit)
ROUTER_MIN_OVERLAP_ISO = 0.99  # vs the CPU router on the same (device) ffn_norm
ROUTER_MAX_MATCHED_REL_L2_GOLDEN = 0.01  # weights on matched rows vs golden (upstream ffn_norm error adds)
ROUTER_MAX_MATCHED_REL_L2_ISO = 0.005  # vs the CPU router on the same ffn_norm (component test limit)
ROUTER_MAX_ROW_SUM_ERR = 0.01  # |sum(row) - 1|
EXPERTS_STEPS = {
    "experts"
}  # checked vs golden (loose: upstream routing flips) and vs the CPU experts on the same inputs
EXPERTS_MAX_REL_L2_GOLDEN = 0.03  # component limit; routing flips add (reference 0.0126, device 0.0185)
EXPERTS_MIN_PCC_GOLDEN = 0.99  # spec component threshold
EXPERTS_MAX_REL_L2_ISO = 0.03  # component test limits, vs the CPU experts on the device ffn_norm + device router
EXPERTS_ROW_NORM_RATIO_ISO = (0.97, 1.03)
EXPERTS_MAX_ROW_REL_L2_ISO = 0.1
FFN_RES_STEPS = {"ffn_residual"}  # out = h_mid + experts_out; checked vs golden and vs the CPU add on the device inputs
FFN_RES_MAX_REL_L2_GOLDEN = 0.01  # = OUT_MAX_REL_L2, whole chunk and first HEAD_ROWS rows
FFN_RES_ROW_NORM_RATIO_GOLDEN = (0.98, 1.02)  # per-token vs golden (routing flips move whole experts in a row)
FFN_RES_MAX_REL_L2_ISO = 0.01  # vs h_mid + experts_out on the device inputs (component test limits below)
FFN_RES_ROW_NORM_RATIO_ISO = (0.99, 1.01)
FFN_RES_MAX_ROW_REL_L2_ISO = 0.02
FFN_RES_EXPERTS_COEF = (0.97, 1.03)  # <out - h_mid, experts_out> / ||experts_out||^2
FFN_RES_MAX_EXPERTS_REL = 0.1  # ||(out - h_mid) - experts_out|| / ||experts_out||


def _routing_stats(got, want):
    """(nnz per row, mean selection overlap, matched rows mask, matched-row rel L2)."""
    gs, ws = got != 0, want != 0
    overlap = ((gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)).mean().item()
    m = (gs == ws).all(-1)
    mrel = ((got[m] - want[m]).norm() / want[m].norm().clamp_min(1e-12)).item() if m.any() else float("inf")
    return gs.sum(-1), overlap, m, mrel


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


def _check_experts(ref, layer, rctx, st, gl, seen, got, want, v):
    """Experts checks (see the module docstring); returns the failure messages."""
    name, o = st.name, st.output
    failures = []
    rel_g = _rel(got, want)
    metrics.record(f"rel_l2_swap_{o}", rel_g)
    msg = (
        f"swapped {name}: vs golden pcc={v:.6f} (>= {EXPERTS_MIN_PCC_GOLDEN}) rel_l2={rel_g:.6f} "
        f"(<= {EXPERTS_MAX_REL_L2_GOLDEN})"
    )
    if not torch.isfinite(got).all():
        failures.append(f"swapped step {name}: non-finite output")
    if v < EXPERTS_MIN_PCC_GOLDEN or rel_g > EXPERTS_MAX_REL_L2_GOLDEN:
        failures.append(f"swapped step {name}: vs golden pcc {v:.5f} / rel L2 {rel_g:.4f}")
    ins = [seen[i].float().reshape(gl[i].shape) for i in st.inputs]
    iso = ref.component(layer, name)(rctx, *ins).float().reshape(want.shape)
    wn = iso.norm(dim=-1).clamp_min(1e-12)
    rel = _rel(got, iso)
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = (got - iso).norm(dim=-1) / wn
    worst = row_rel.max().item()
    metrics.record(f"rel_l2_iso_swap_{o}", rel)
    metrics.record(f"row_norm_ratio_min_iso_swap_{o}", rmin)
    metrics.record(f"row_norm_ratio_max_iso_swap_{o}", rmax)
    metrics.record(f"worst_row_rel_l2_iso_swap_{o}", worst)
    msg += (
        f"\n  vs CPU experts on the same inputs: rel_l2={rel:.6f} (<= {EXPERTS_MAX_REL_L2_ISO}) "
        f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {EXPERTS_ROW_NORM_RATIO_ISO}) "
        f"worst_row_rel_l2={worst:.4f} (<= {EXPERTS_MAX_ROW_REL_L2_ISO}, row {row_rel.argmax().item()})"
    )
    if rel > EXPERTS_MAX_REL_L2_ISO:
        failures.append(f"swapped step {name}: rel L2 vs CPU experts {rel:.4f} > {EXPERTS_MAX_REL_L2_ISO}")
    lo, hi = EXPERTS_ROW_NORM_RATIO_ISO
    if not (lo <= rmin and rmax <= hi):
        failures.append(f"swapped step {name}: per-token norm ratio vs CPU experts [{rmin:.4f}, {rmax:.4f}]")
    if worst > EXPERTS_MAX_ROW_REL_L2_ISO:
        failures.append(
            f"swapped step {name}: worst per-token rel L2 vs CPU experts {worst:.4f} > {EXPERTS_MAX_ROW_REL_L2_ISO} "
            "(dropped (token, expert) pair?)"
        )
    print(msg)
    return failures


def _check_ffn_residual(ref, layer, rctx, st, gl, seen, got, want, v, comp_thr):
    """ffn_residual checks (see the module docstring); returns the failure messages."""
    name = st.name
    failures = []
    rel_g = _rel(got, want)
    r = min(HEAD_ROWS, want.shape[0])
    rel_head = _rel(got[:r], want[:r])
    ratio_g = got.norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)
    gmin, gmax = ratio_g.min().item(), ratio_g.max().item()
    metrics.record("row_norm_ratio_min_swap_out", gmin)
    metrics.record("row_norm_ratio_max_swap_out", gmax)
    msg = (
        f"swapped {name}: vs golden pcc={v:.6f} (>= {comp_thr}) rel_l2={rel_g:.6f} first_{r}_rows={rel_head:.6f} "
        f"(<= {FFN_RES_MAX_REL_L2_GOLDEN}) row_norm_ratio=[{gmin:.4f}, {gmax:.4f}] (in {FFN_RES_ROW_NORM_RATIO_GOLDEN})"
    )
    if not torch.isfinite(got).all():
        failures.append(f"swapped step {name}: non-finite output")
    if v < comp_thr or rel_g > FFN_RES_MAX_REL_L2_GOLDEN or rel_head > FFN_RES_MAX_REL_L2_GOLDEN:
        failures.append(f"swapped step {name}: vs golden pcc {v:.5f} / rel L2 {rel_g:.4f} / first rows {rel_head:.4f}")
    lo, hi = FFN_RES_ROW_NORM_RATIO_GOLDEN
    if not (lo <= gmin and gmax <= hi):
        failures.append(f"swapped step {name}: per-token norm ratio vs golden [{gmin:.4f}, {gmax:.4f}]")
    h_mid, eo = (seen[i].float().reshape(gl[i].shape) for i in st.inputs)
    iso = ref.component(layer, name)(rctx, h_mid, eo).float().reshape(want.shape)
    wn = iso.norm(dim=-1).clamp_min(1e-12)
    rel = _rel(got, iso)
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    worst = ((got - iso).norm(dim=-1) / wn).max().item()
    delta = got - h_mid
    coef = ((delta * eo).sum() / (eo * eo).sum().clamp_min(1e-30)).item()
    erel = ((delta - eo).norm() / eo.norm().clamp_min(1e-30)).item()
    metrics.record("rel_l2_iso_swap_out", rel)
    metrics.record("row_norm_ratio_min_iso_swap_out", rmin)
    metrics.record("row_norm_ratio_max_iso_swap_out", rmax)
    metrics.record("worst_row_rel_l2_iso_swap_out", worst)
    metrics.record("experts_coef_swap_out", coef)
    metrics.record("experts_rel_l2_swap_out", erel)
    msg += (
        f"\n  vs CPU add on the same inputs: rel_l2={rel:.6f} (<= {FFN_RES_MAX_REL_L2_ISO}) "
        f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {FFN_RES_ROW_NORM_RATIO_ISO}) worst_row_rel_l2={worst:.4f} "
        f"(<= {FFN_RES_MAX_ROW_REL_L2_ISO})"
        f"\n  experts term: coef={coef:.4f} (in {FFN_RES_EXPERTS_COEF}) rel={erel:.4f} (<= {FFN_RES_MAX_EXPERTS_REL})"
    )
    if rel > FFN_RES_MAX_REL_L2_ISO:
        failures.append(f"swapped step {name}: rel L2 vs CPU add {rel:.4f} > {FFN_RES_MAX_REL_L2_ISO}")
    lo, hi = FFN_RES_ROW_NORM_RATIO_ISO
    if not (lo <= rmin and rmax <= hi):
        failures.append(f"swapped step {name}: per-token norm ratio vs CPU add [{rmin:.4f}, {rmax:.4f}]")
    if worst > FFN_RES_MAX_ROW_REL_L2_ISO:
        failures.append(
            f"swapped step {name}: worst per-token rel L2 vs CPU add {worst:.4f} > {FFN_RES_MAX_ROW_REL_L2_ISO}"
        )
    if not (FFN_RES_EXPERTS_COEF[0] <= coef <= FFN_RES_EXPERTS_COEF[1]):
        failures.append(
            f"ffn_residual: experts_out coefficient {coef:.4f} outside {FFN_RES_EXPERTS_COEF} (dropped/scaled?)"
        )
    if not erel <= FFN_RES_MAX_EXPERTS_REL:
        failures.append(
            f"ffn_residual: experts term rel L2 {erel:.4f} > {FFN_RES_MAX_EXPERTS_REL} (misaligned/corrupted?)"
        )
    print(msg)
    return failures


def _check_router(ref, layer, rctx, st, gl, seen, got, want, v, comp_thr):
    """Router checks (see the module docstring); returns the failure messages."""
    name, o = st.name, st.output
    failures = []
    msg = f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={_rel(got, want):.6f} (not gated: near-tie flips)"
    metrics.record(f"rel_l2_swap_{o}", _rel(got, want))
    if not torch.isfinite(got).all():
        failures.append(f"swapped step {name}: non-finite output")
    if v < comp_thr:
        failures.append(f"swapped step {name}: pcc {v:.4f} < {comp_thr}")
    ins = [seen[i].float().reshape(gl[i].shape) for i in st.inputs]
    iso = ref.component(layer, name)(rctx, *ins).float().reshape(want.shape)
    for tag, w, min_ov, max_mrel in (
        ("golden", want, ROUTER_MIN_OVERLAP_GOLDEN, ROUTER_MAX_MATCHED_REL_L2_GOLDEN),
        ("iso", iso, ROUTER_MIN_OVERLAP_ISO, ROUTER_MAX_MATCHED_REL_L2_ISO),
    ):
        nnz, ov, m, mrel = _routing_stats(got, w)
        metrics.record(f"selection_overlap_{tag}_swap_{o}", ov)
        metrics.record(f"matched_rel_l2_{tag}_swap_{o}", mrel)
        msg += (
            f"\n  vs {tag}: selection_overlap={ov:.5f} (>= {min_ov}) matched_rows={m.sum().item()}/{m.numel()} "
            f"matched_rel_l2={mrel:.5f} (<= {max_mrel})"
        )
        if ov < min_ov:
            failures.append(f"swapped step {name}: top-{TOP_K} selection overlap vs {tag} {ov:.5f} < {min_ov}")
        if mrel > max_mrel:
            failures.append(
                f"swapped step {name}: matched-row weight rel L2 vs {tag} {mrel:.5f} > {max_mrel} "
                "(weights from biased scores, scale or renorm bug)"
            )
    rsum = got.sum(-1)
    rmin, rmax = rsum.min().item(), rsum.max().item()
    metrics.record(f"row_sum_min_swap_{o}", rmin)
    metrics.record(f"row_sum_max_swap_{o}", rmax)
    msg += (
        f"\n  nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K}) row_sum=[{rmin:.4f}, {rmax:.4f}] "
        f"(1 +- {ROUTER_MAX_ROW_SUM_ERR})"
    )
    if not (nnz == TOP_K).all():
        failures.append(
            f"swapped step {name}: nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want exactly {TOP_K}"
        )
    if (got < 0).any():
        failures.append(f"swapped step {name}: negative routing weight")
    if not (1 - ROUTER_MAX_ROW_SUM_ERR <= rmin and rmax <= 1 + ROUTER_MAX_ROW_SUM_ERR):
        failures.append(f"swapped step {name}: row sums [{rmin:.4f}, {rmax:.4f}], want 1 +- {ROUTER_MAX_ROW_SUM_ERR}")
    print(msg)
    return failures


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
        if name in ROUTER_STEPS:
            failures += _check_router(ref, layer, rctx, st, gl, seen, got, w, v, comp_thr)
            continue
        if name in EXPERTS_STEPS:
            failures += _check_experts(ref, layer, rctx, st, gl, seen, got, w, v)
            continue
        if name in FFN_RES_STEPS:
            failures += _check_ffn_residual(ref, layer, rctx, st, gl, seen, got, w, v, comp_thr)
            continue
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
        if name == "attn_residual":
            # The attention term is ~1% of h_mid: check it directly on delta = h_mid - in, against the inputs fed in.
            res, attn = (seen[i].float().reshape(gl[i].shape) for i in st.inputs)
            delta = got - res
            coef = ((delta * attn).sum() / (attn * attn).sum().clamp_min(1e-30)).item()
            arel = ((delta - attn).norm() / attn.norm().clamp_min(1e-30)).item()
            metrics.record(f"attn_coef_swap_{o}", coef)
            metrics.record(f"attn_rel_l2_swap_{o}", arel)
            print(f"attn term: coef={coef:.4f} (in {ATTN_COEF}) rel={arel:.4f} (<= {MAX_ATTN_REL})")
            if not (ATTN_COEF[0] <= coef <= ATTN_COEF[1]):
                failures.append(f"attn_residual: attn_out coefficient {coef:.4f} outside {ATTN_COEF} (dropped/scaled?)")
            if not arel <= MAX_ATTN_REL:
                failures.append(f"attn_residual: attn term rel L2 {arel:.4f} > {MAX_ATTN_REL} (misaligned/corrupted?)")
    assert not failures, "; ".join(failures)
