# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 7: block type full_moe (layer 5) with ffn_residual swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 5 (full_moe) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    attn_residual
    ffn_norm
    router
    experts
    ffn_residual

Reviewed (S.full_moe.07.test.1): swap 06's checks (below) plus ffn_residual, as test_swap_sliding_moe_07_ffn_residual.py.
Its output is the block output `out` (out = h_mid + experts_out), so pcc_swap_out is also the swapped step's own
output; PCC alone misses a dropped or scaled experts term (the experts term is ~6.7% of ||out|| on layer 5,
test_c_full_moe_ffn_residual.py). Extra checks:
  - vs golden: finite, PCC >= spec component threshold, rel L2 <= 0.01 (whole chunk and first 128 rows, the block-out
    limit). Per-token norm ratio in [0.98, 1.02] on rows whose device top-8 selection equals the golden one, and in
    [0.9, 1.1] on rows with a routing flip. Layer-5 experts outputs are large, so one flipped expert moves a row by
    up to 3%: the device run's worst row (0.9694) is a flipped row, and a single [0.98, 1.02] limit fails on it;
  - vs the CPU add on the same device h_mid / experts_out ("iso"): rel L2 <= 0.01, per-token ratio in [0.99, 1.01],
    worst per-token rel L2 <= 0.02; on delta = out - h_mid: experts coefficient in [0.97, 1.03] and experts-term
    rel L2 <= 0.1 (component test limits).
Measured ffn_residual, as golden rel / first rows / ratio on matched rows / ratio on flipped rows; iso rel / ratio /
worst row; coef / experts rel:
  BRINGUP_IMPL=reference 0.0025 / 0.0023 / 1993 rows [0.9990, 1.0010] / 55 rows [0.9947, 1.0106]; 0 / [1, 1] / 0;
  1.0 / 0;
  device (all 7 steps): pcc_swap_out 0.999985, 0.0055 / 0.0047 / 1864 rows [0.9970, 1.0034] / 184 rows
  [0.9694, 1.0023]; 0.0013 / [0.9995, 1.0011] / 0.0021; 1.0016 / 0.0195.
Zero stub fails every golden check.

Swap 06 review (S.full_moe.06.test.1): swap 05's checks (below) plus experts (experts_out [2048, 4096],
float), as test_swap_sliding_moe_06_experts.py. The experts routing comes from the device router, whose near-tie flips
(184 of 2048 rows differ from the golden selection) change which experts run. Layer-5 experts outputs vary widely
(expert 235 norm ~690), so a flip moves a row a lot: CPU experts on the device routing (swap 05 trail) already give
PCC 0.99869 / rel 0.0515 vs golden. Hence:
  - vs golden (gross errors only): finite, PCC >= 0.99, whole-chunk rel L2 <= EXPERTS_MAX_REL_L2_GOLDEN = 0.08 (layer 1
    uses 0.03; here the upstream flips alone give 0.0515);
  - vs the CPU experts on the same device ffn_norm + router ("iso"): the component test limits
    (test_c_full_moe_experts.py), rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03], worst per-token rel L2 <= 0.1
    (dropped expert / (token, expert) pair, scale, zeroed rows, bfp8 activations: rel 0.045 in the component test).
Measured experts, as golden pcc / golden rel / iso rel / iso ratio / iso worst row:
  BRINGUP_IMPL=reference 0.999875 / 0.0158 / 0 / [1, 1] / 0 (out rel 0.0025);
  device (TtExperts loop mode, bf16 x, HiFi4, fp32 mid) 0.998676 / 0.0515 / 0.0044 / [0.9967, 1.0058] / 0.0070;
  block pcc_swap_out 0.999986, out rel 0.0053 / first rows 0.0045. Zero stub fails every golden check.
Swap 05 review (S.full_moe.05.test.1): same body as test_swap_full_moe_03_attn_residual.py (golden s4096 chunk 1, start
2048, KV prefix [0, 2048) from the golden state; layer 5 is full causal GQA, no sink, no window) with ffn_norm added.
The gated metric is pcc_swap_out (PCC, float [2048, 4096], spec block threshold 0.98), weak on its own (the residual
dominates `out`). Extra asserted checks (informational metrics):
  - every swapped float step's own output vs golden, as swap 03 (attn_norm; attention incl. first 128 rows and worst
    row; attn_residual incl. first 128 rows and the attention-term coef / rel on delta = h_mid - in);
  - ffn_norm: finite, PCC >= spec component threshold, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03] (the
    limits of test_c_full_moe_ffn_norm.py, which catch sum-instead-of-mean, eps 1e-3, 1 + w, no weight, the
    attn_norm weight, zeroed rows). ffn_norm reads h_mid with the device attn_norm + attention + attn_residual error;
  - router (dense [2048, 256], 8 nonzeros per row, rows sum to 1): whole-matrix rel L2 is dominated by near-tie
    selection flips (layer-5 8th/9th choice gap median 0.0028, test_c_full_moe_router.py), so instead, as
    test_swap_sliding_moe_05_router.py: PCC vs golden >= spec component threshold, exactly 8 nonzeros per row, no
    negative weights, row sums within ROUTER_MAX_ROW_SUM_ERR of 1, and vs golden / vs the CPU router on the same device
    ffn_norm ("iso"): mean top-8 selection overlap and matched-row weight rel L2 (limits below). The iso check
    isolates the router from the upstream device error;
  - block out: finite, rel L2 <= 0.01 whole chunk and first 128 rows. The CPU experts now run on the device routing.
Measured: BRINGUP_IMPL=reference out rel 0.0025; router pcc 0.99865, overlap 0.99664 (golden) / 1.0 (iso), matched rel
0.00154 / 0.0. Zero stub fails every check. Precompile pass (zero attention): router overlap vs golden 0.137 fails.
Device: pcc_swap_out 0.999986, out rel 0.0053 / first rows 0.0045; router pcc 0.99552, overlap 0.98846 / 0.99890,
matched rows 1864 / 2030 of 2048, matched rel 0.00265 / 0.00147, nnz 8, row sums [0.9977, 1.0026]. The golden overlap
margin is thin (0.988 vs 0.985) because of the upstream device ffn_norm error (swap 04, with the CPU router on the
device ffn_norm, already had router pcc 0.9955); the iso check (0.9989) shows the device router itself is close.
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
BLOCK_TYPE = "full_moe"
SWAPPED = ["attn_norm", "attention", "attn_residual", "ffn_norm", "router", "experts", "ffn_residual"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = {
    "attn_norm": 0.03,
    "attention": 0.015,
    "attn_residual": 0.01,
    "ffn_norm": 0.03,
}  # swapped step output, whole chunk
HEAD_ROWS = 128  # first rows of the chunk: attend mostly into the KV prefix, most exposed to causality / RoPE offset
HEAD_ROW_STEPS = {"attention", "attn_residual"}  # stateful steps whose first HEAD_ROWS rows are checked separately
STEP_ROW_NORM_RATIO = {
    "attn_norm": (0.97, 1.03),
    "attention": (0.97, 1.03),
    "attn_residual": (0.99, 1.01),
    "ffn_norm": (0.97, 1.03),
}  # swapped step output: per-token ||got|| / ||want||
OUT_MAX_REL_L2 = 0.01  # block output, whole chunk and first HEAD_ROWS rows
STEP_MAX_WORST_ROW_REL_L2 = {
    "attention": 0.06
}  # max over tokens of ||got_t - want_t|| / ||want_t|| (C.full_moe.attention)
ATTN_COEF = (0.98, 1.02)  # attn_residual: <h_mid - in, attn_out> / ||attn_out||^2
MAX_ATTN_REL = 0.05  # attn_residual: ||(h_mid - in) - attn_out|| / ||attn_out|| (bf16 rounding alone 0.003)

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
EXPERTS_MIN_PCC_GOLDEN = 0.99  # spec component threshold
EXPERTS_MAX_REL_L2_GOLDEN = 0.08  # routing flips dominate, see the docstring
EXPERTS_MAX_REL_L2_ISO = 0.03  # component test limits, vs the CPU experts on the device ffn_norm + device router
EXPERTS_ROW_NORM_RATIO_ISO = (0.97, 1.03)
EXPERTS_MAX_ROW_REL_L2_ISO = 0.1
FFN_RES_STEPS = {"ffn_residual"}  # out = h_mid + experts_out; checked vs golden and vs the CPU add on the device inputs
FFN_RES_MAX_REL_L2_GOLDEN = 0.01  # = OUT_MAX_REL_L2, whole chunk and first HEAD_ROWS rows
FFN_RES_ROW_NORM_RATIO_GOLDEN = (0.98, 1.02)  # per-token vs golden, rows whose device routing matches the golden
FFN_RES_ROW_NORM_RATIO_GOLDEN_FLIPPED = (0.9, 1.1)  # rows with a routing flip (a whole expert changes in the row)
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
    # Rows whose device top-8 selection equals the golden one get the tight per-token limit; rows with a routing flip
    # (upstream near-tie, layer-5 experts outputs are large) only the loose one.
    ro = _step(ref, layer, "router").output
    m = ((seen[ro].float().reshape(gl[ro].shape) != 0) == (gl[ro].float() != 0)).all(-1)
    mr, fr = ratio_g[m], ratio_g[~m]
    mmin, mmax = (mr.min().item(), mr.max().item()) if m.any() else (1.0, 1.0)
    fmin, fmax = (fr.min().item(), fr.max().item()) if (~m).any() else (1.0, 1.0)
    metrics.record("row_norm_ratio_min_matched_swap_out", mmin)
    metrics.record("row_norm_ratio_max_matched_swap_out", mmax)
    metrics.record("row_norm_ratio_min_flipped_swap_out", fmin)
    metrics.record("row_norm_ratio_max_flipped_swap_out", fmax)
    msg += (
        f"\n  per-token ratio vs golden: {int(m.sum())} rows with golden routing [{mmin:.4f}, {mmax:.4f}] "
        f"(in {FFN_RES_ROW_NORM_RATIO_GOLDEN}, worst row {int(ratio_g.argmin())}), {int((~m).sum())} flipped rows "
        f"[{fmin:.4f}, {fmax:.4f}] (in {FFN_RES_ROW_NORM_RATIO_GOLDEN_FLIPPED})"
    )
    lo, hi = FFN_RES_ROW_NORM_RATIO_GOLDEN
    if not (lo <= mmin and mmax <= hi):
        failures.append(
            f"swapped step {name}: per-token norm ratio vs golden on matched-routing rows [{mmin:.4f}, {mmax:.4f}]"
        )
    lo, hi = FFN_RES_ROW_NORM_RATIO_GOLDEN_FLIPPED
    if not (lo <= fmin and fmax <= hi):
        failures.append(
            f"swapped step {name}: per-token norm ratio vs golden on flipped-routing rows [{fmin:.4f}, {fmax:.4f}]"
        )
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
    assert not ref.cfg.is_sliding(layer), f"layer {layer} must be a full-attention layer"
    assert ref.w[layer].sink is None, f"layer {layer} full attention has no sink"
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
        metrics.record(f"worst_row_rel_l2_swap_{o}", worst)
        metrics.record(f"rel_l2_swap_{o}", rel)
        metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
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
                    "(RoPE positions not offset by the chunk start, causal mask, or KV prefix mishandled?)"
                )
        print(msg)
        if name == "attn_residual":
            # The attention term, on delta = h_mid - in, against the inputs the swap actually fed in.
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
