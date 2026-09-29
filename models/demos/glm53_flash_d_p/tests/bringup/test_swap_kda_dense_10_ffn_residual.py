# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 10: block type kda_dense (layer 0) with ffn_residual swapped in last (every step on the device).

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (kda_dense) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    attention
    attn_residual
    ffn_hc
    ffn_collapse
    ffn_norm
    mlp
    ffn_residual

Reviewed (S.kda_dense.10.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). Block out is now the device ffn_residual (post * mlp_out + comb^T h_mid).
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 09 asserts (block out rel L2 <= 0.01 and per-token ratio [0.97, 1.03], attn_hc, attn_collapse,
    attn_norm, attention, KDA state, attn_residual, ffn_hc, ffn_collapse, ffn_norm, mlp, incl. the layer-1 runs and the
    clamp probe), same limits;
  - the block-out tails from the same device h_mid: the all-CPU ffn tail (now five device steps' share, limits
    unchanged 0.005 / [0.975, 1.025]); the collapse-, norm- and mlp-share tails now end in the device ffn_residual
    (swap 09 used the CPU one), limits unchanged, so each still isolates one step;
  - ffn_residual vs the fp32 CPU residual of the (h_mid, ffn_hc, mlp_out) it actually got, and the (weightless) module
    on layer 1's golden inputs vs the CPU residual of the same inputs: rel L2 <= 0.0035, per-row ratio
    [0.9975, 1.0025], per-stream rel <= 0.005, worst row <= 0.008, each term on its own (out minus the exact other
    term vs post * mlp_out and vs comb^T h_mid) coefficient [0.998, 1.002] and rel <= 0.012. These are tighter than
    the component test's (0.01 / [0.99, 1.01]) because the reference here is exact fp32, not the bf16-rounded golden.
Sensitivity of those limits (CPU host script on the layer-0 / layer-1 goldens, not kept), vs the exact residual, rel /
ratio / worst row / term coef: bf16 output (= the device) 0.0017 / [0.9998, 1.0002] / 0.0018 / 1.0000; all-bf16 mix
0.0029 / [0.9980, 1.0018] / 0.0054 / 1.0000 (term rel L1 0.0082); post x1.005 L0 0.0040 / 1.0079 / 0.011 / 1.005, L1
0.0015 / 1.0018 / 0.0071 / 1.005 (caught by the coefficient only); comb x1.005 0.0059 / 1.0073 / .. / 1.005 (L1
[1.0032, 1.0068]); 7-bit-mantissa output 0.0066 / [0.9940, 0.9948] / coef 0.9965; comb not transposed 0.098 (L1
0.038); last row zeroed 0.0052 / ratio 0 (L1 0.0036); last row x1.02 0.0001 / 1.02 / 0.02 (caught by ratio and worst
row).
Device (S.kda_dense.10.test.1): out PCC 0.999977, rel 0.0077, ratio [0.9797, 1.0082]; cpu tail 0.0038 /
[0.9873, 1.0131]; collapse-share 1e-4; norm-share 0.0017 / [0.9971, 1.0026]; mlp-share 0.0025 / [0.9985, 1.0005];
ffn_residual same input 0.0017 / [0.9998, 1.0002] / row 0.0018 / coef 1.0000 / term rel 0.0021 / 0.0014, layer 1 0.0017
/ term rel 0.0055 / 0.0016. Reference: out rel 0.0017, every same-input check 0. Stub: PCC 0 and every golden check
fails; the same-input checks score 0 on layer 0 (the CPU steps of zero inputs are zero) but fail on layer 1.
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
    impl_mode,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
BLOCK_TYPE = "kda_dense"
SWAPPED = [
    "attn_hc",
    "attn_collapse",
    "attn_norm",
    "attention",
    "attn_residual",
    "ffn_hc",
    "ffn_collapse",
    "ffn_norm",
    "mlp",
    "ffn_residual",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
OUT_MAX_REL_L2 = 0.01  # block out: ||got - want|| / ||want||
OUT_ROW_NORM_RATIO = (0.97, 1.03)  # block out: per-row ||got|| / ||want||
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.03, "post": 0.03, "comb": 0.02}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.15, "post": 0.2, "comb": 0.1}
FFN_HC_MAX_REL_L2 = {  # ffn_hc per part (cpu: the component test's limit)
    "cpu": {"pre": 0.01, "post": 0.01, "comb": 0.01},
    "golden": {"pre": 0.015, "post": 0.015, "comb": 0.015},
}
FFN_HC_MAX_ABS = {"cpu": {"pre": 0.05, "post": 0.05, "comb": 0.05}, "golden": {"pre": 0.06, "post": 0.06, "comb": 0.06}}
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|
TAIL_MAX_REL_L2 = 0.005  # block out vs the CPU tail with CPU ffn_hc + ffn_collapse + ffn_norm of the same h_mid
TAIL_RATIO = (0.975, 1.025)  # per-row ||out|| / ||tail||; bf16-mix noise gives [0.989, 1.010]
FC_TAIL_MAX_REL_L2 = 0.003  # block out vs the tail from the CPU collapse of the device (h_mid, ffn_hc), device norm
FC_TAIL_RATIO = (0.995, 1.005)  # per-row ||out|| / ||fc tail||
FN_TAIL_MAX_REL_L2 = 0.004  # block out vs the CPU tail from the CPU norm of the device ffn_in
FN_TAIL_RATIO = (0.99, 1.01)  # per-row ||out|| / ||fn tail||
FN_MAX_REL_L2 = {"cpu": 0.01, "golden": 0.015}  # ffn_norm (cpu: the component test's limits)
FN_RATIO = {"cpu": (0.99, 1.01), "golden": (0.98, 1.02)}  # ffn_norm per-token ||got|| / ||want||
FN_MAX_ROW_REL_L2 = {"cpu": 0.015, "golden": 0.03}  # ffn_norm worst per-token rel L2
FC_MAX_REL_L2 = {"cpu": 0.008, "golden": 0.02}  # ffn_collapse ffn_in (cpu: the component test's limit)
FC_RATIO = {"cpu": (0.99, 1.01), "golden": (0.97, 1.03)}  # ffn_collapse per-token ||got|| / ||want||
FC_MAX_ROW_REL_L2 = {"cpu": 0.03, "golden": 0.05}  # ffn_collapse worst per-token rel L2
COLLAPSE_MAX_REL_L2 = 0.01  # attn_collapse, as its component test
COLLAPSE_RATIO = (0.985, 1.015)  # attn_collapse per-token ||got|| / ||want||
DISTINCT_LAYER = 1  # a kda_dense layer whose streams differ
NORM_MAX_REL_L2 = 0.01  # attn_norm, as its component test
NORM_RATIO = (0.98, 1.02)  # attn_norm per-token ||got|| / ||want||
NORM_MAX_ROW_REL_L2 = 0.03  # attn_norm worst per-token rel L2
ATTN_MAX_REL_L2 = 0.02  # attention, as its component test
ATTN_RATIO = (0.98, 1.02)  # attention per-token ||got|| / ||want||
ATTN_BLOCK_ROWS = 128
ATTN_MAX_BLOCK_REL_L2 = 0.03  # every 128-row block: prefix state / SP carry / conv tail bugs hit the first rows
ATTN_MAX_ROW_REL_L2 = 0.1  # attention worst per-token rel L2
MAX_STATE_REL_L2 = {"kda_recurrent": 0.03, "kda_conv": 0.02}
MAX_STATE_HEAD_REL_L2 = 0.05  # kda_recurrent, worst head
RES_MAX_REL_L2 = {"cpu": 0.01, "golden": 0.015}  # attn_residual h_mid (cpu: the component test's limit)
RES_RATIO = {"cpu": (0.985, 1.015), "golden": (0.97, 1.03)}  # attn_residual per-row ||got|| / ||want||
RES_MAX_STREAM_REL = {"cpu": 0.015, "golden": 0.02}  # rel L2 over the rows of one stream
RES_MAX_ROW_REL = 0.05  # attn_residual worst row rel L2
MLP_MAX_REL_L2 = {"cpu": 0.015, "golden": 0.02}  # mlp mlp_out (cpu: the component test's limits)
MLP_RATIO = {"cpu": (0.99, 1.01), "golden": (0.98, 1.02)}  # mlp per-token ||got|| / ||want||
MLP_MAX_ROW_REL_L2 = {"cpu": 0.02, "golden": 0.03}  # mlp worst per-token rel L2
MLP_COEF = {"cpu": (0.995, 1.005), "golden": (0.99, 1.01)}  # mlp <got, want> / <want, want>
CLAMP_PROBE_SCALE = 48.0  # makes the swiglu clamps (limit 10) engage, as the component test
PROBE_MAX_REL_L2, PROBE_RATIO, PROBE_MAX_ROW_REL_L2 = 0.01, (0.99, 1.01), 0.03
MLP_TAIL_MAX_REL_L2 = 0.0035  # block out vs ffn_residual of the CPU mlp of the device ffn_norm
MLP_TAIL_RATIO = (0.995, 1.005)  # per-row ||out|| / ||mlp tail||
TERM_COEF = (0.98, 1.02)  # <out - other term, term> / ||term||^2, terms from the step's actual inputs
MAX_TERM_REL = 0.02  # ||out - other term - term|| / ||term||
# ffn_residual vs the fp32 CPU residual of the inputs it actually got (device = bf16 output rounding: 0.0017; an
# all-bf16 mix 0.0029 / [0.9980, 1.0018] / row 0.0054 / stream 0.0037 / term rel 0.0082; post or comb x1.005 fail).
FR_MAX_REL_L2 = 0.0035
FR_RATIO = (0.9975, 1.0025)  # per-row ||got|| / ||want||
FR_MAX_STREAM_REL = 0.005  # rel L2 over the rows of one stream
FR_MAX_ROW_REL = 0.008  # worst row rel L2
FR_TERM_COEF = (0.998, 1.002)  # <out - other term, term> / ||term||^2
FR_MAX_TERM_REL = 0.012  # ||out - other term - term|| / ||term||


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _f32(t):
    # Device steps hand back bf16; the CPU steps after them run in fp32 (bf16 @ fp32 raises).
    return t.float() if torch.is_tensor(t) and t.is_floating_point() else t


def _hc_checks(got, want, tag="attn_hc", max_rel=MAX_REL_L2, max_abs=MAX_ABS):
    fails = []
    if got.numel() != want.numel():
        return [f"{tag}: shape {tuple(got.shape)} vs {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    for name, sl in PARTS.items():
        a, b = got[:, sl], w[:, sl]
        rel = ((a - b).norm() / b.norm()).item()
        mx = (a - b).abs().max().item()
        metrics.record(f"rel_l2_swap_{tag}_{name}", rel)
        metrics.record(f"max_abs_swap_{tag}_{name}", mx)
        print(f"{tag} {name}: rel_l2={rel:.5f} (<= {max_rel[name]}) max_abs={mx:.2e} (<= {max_abs[name]})")
        if rel > max_rel[name]:
            fails.append(f"{tag} {name} rel L2 {rel:.4f} > {max_rel[name]}")
        if mx > max_abs[name]:
            fails.append(f"{tag} {name} max abs {mx:.3f} > {max_abs[name]}")
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record(f"comb_col_sum_err_swap_{tag}", col_err)
    print(f"{tag} comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if col_err > COL_SUM_TOL:
        fails.append(f"{tag} comb column sums off 1 by {col_err:.4f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"{tag} pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"{tag} post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append(f"{tag} negative comb entries")
    return fails


def _tail(ref, layer, ctx, h_mid, hc, x=None, xn=None):
    """CPU [ffn_collapse ->] [ffn_norm ->] mlp -> ffn_residual from (h_mid, ffn_hc[, ffn_in[, ffn_norm]])."""
    comp = lambda n: ref.component(layer, n)
    if xn is None:
        if x is None:
            x = comp("ffn_collapse")(ctx, h_mid, hc)
        xn = comp("ffn_norm")(ctx, x)
    y = comp("mlp")(ctx, xn)
    return comp("ffn_residual")(ctx, h_mid, hc, y)


def _tail_checks(tag, out, tail, max_rel, ratio):
    t_rel = _rel(out, tail)
    tr = out.norm(dim=-1) / tail.float().norm(dim=-1).clamp_min(1e-12)
    tlo, thi = tr.min().item(), tr.max().item()
    metrics.record(f"rel_l2_swap_out_vs_{tag}", t_rel)
    metrics.record(f"row_norm_ratio_min_swap_out_vs_{tag}", tlo)
    metrics.record(f"row_norm_ratio_max_swap_out_vs_{tag}", thi)
    print(f"block out vs {tag}: rel_l2={t_rel:.6f} (<= {max_rel}) ratio [{tlo:.4f}, {thi:.4f}] (in {ratio})")
    fails = []
    if t_rel > max_rel:
        fails.append(f"block out vs {tag} rel L2 {t_rel:.4f} > {max_rel}")
    if tlo < ratio[0] or thi > ratio[1]:
        fails.append(f"block out vs {tag} per-row ratio [{tlo:.4f}, {thi:.4f}] outside {ratio}")
    return fails


def _fc_checks(tag, lim, got, want):
    """ffn_collapse ffn_in [S, H] vs want at the `lim` limits."""
    if got.numel() != want.numel():
        return [f"ffn_collapse {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"ffn_collapse {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_swap_ffn_collapse_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_ffn_collapse_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_ffn_collapse_{tag}", hi)
    metrics.record(f"max_row_rel_l2_swap_ffn_collapse_{tag}", row)
    print(
        f"ffn_collapse {tag}: rel_l2={rel:.5f} (<= {FC_MAX_REL_L2[lim]}) norm ratio [{lo:.4f}, {hi:.4f}] "
        f"(in {FC_RATIO[lim]}) worst row {row:.4f} (<= {FC_MAX_ROW_REL_L2[lim]})"
    )
    fails = []
    if rel > FC_MAX_REL_L2[lim]:
        fails.append(f"ffn_collapse {tag} rel L2 {rel:.4f} > {FC_MAX_REL_L2[lim]}")
    if lo < FC_RATIO[lim][0] or hi > FC_RATIO[lim][1]:
        fails.append(f"ffn_collapse {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {FC_RATIO[lim]}")
    if row > FC_MAX_ROW_REL_L2[lim]:
        fails.append(f"ffn_collapse {tag} worst row rel L2 {row:.4f} > {FC_MAX_ROW_REL_L2[lim]}")
    return fails


def _fn_checks(tag, lim, got, want):
    """ffn_norm [S, H] vs want at the `lim` limits."""
    if got.numel() != want.numel():
        return [f"ffn_norm {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"ffn_norm {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_swap_ffn_norm_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_ffn_norm_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_ffn_norm_{tag}", hi)
    metrics.record(f"max_row_rel_l2_swap_ffn_norm_{tag}", row)
    print(
        f"ffn_norm {tag}: rel_l2={rel:.5f} (<= {FN_MAX_REL_L2[lim]}) norm ratio [{lo:.4f}, {hi:.4f}] "
        f"(in {FN_RATIO[lim]}) worst row {row:.4f} (<= {FN_MAX_ROW_REL_L2[lim]})"
    )
    fails = []
    if rel > FN_MAX_REL_L2[lim]:
        fails.append(f"ffn_norm {tag} rel L2 {rel:.4f} > {FN_MAX_REL_L2[lim]}")
    if lo < FN_RATIO[lim][0] or hi > FN_RATIO[lim][1]:
        fails.append(f"ffn_norm {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {FN_RATIO[lim]}")
    if row > FN_MAX_ROW_REL_L2[lim]:
        fails.append(f"ffn_norm {tag} worst row rel L2 {row:.4f} > {FN_MAX_ROW_REL_L2[lim]}")
    return fails


def _mlp_checks(tag, lim, got, want, max_rel=None, ratio=None, max_row=None, coef_lim=None):
    """mlp mlp_out [S, H] vs want at the `lim` limits (explicit limits override them)."""
    max_rel = MLP_MAX_REL_L2[lim] if max_rel is None else max_rel
    ratio = MLP_RATIO[lim] if ratio is None else ratio
    max_row = MLP_MAX_ROW_REL_L2[lim] if max_row is None else max_row
    if got.numel() != want.numel():
        return [f"mlp {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"mlp {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    coef = ((got * w).sum() / (w * w).sum().clamp_min(1e-30)).item()
    metrics.record(f"rel_l2_swap_mlp_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_mlp_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_mlp_{tag}", hi)
    metrics.record(f"max_row_rel_l2_swap_mlp_{tag}", row)
    metrics.record(f"scale_coef_swap_mlp_{tag}", coef)
    print(
        f"mlp {tag}: rel_l2={rel:.5f} (<= {max_rel}) norm ratio [{lo:.4f}, {hi:.4f}] (in {ratio}) worst row "
        f"{row:.4f} (<= {max_row}) coef {coef:.5f}" + (f" (in {coef_lim})" if coef_lim else "")
    )
    fails = []
    if rel > max_rel:
        fails.append(f"mlp {tag} rel L2 {rel:.4f} > {max_rel}")
    if lo < ratio[0] or hi > ratio[1]:
        fails.append(f"mlp {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {ratio}")
    if row > max_row:
        fails.append(f"mlp {tag} worst row rel L2 {row:.4f} > {max_row}")
    if coef_lim and not coef_lim[0] <= coef <= coef_lim[1]:
        fails.append(f"mlp {tag} scale coefficient {coef:.5f} outside {coef_lim}")
    return fails


def _collapse_checks(tag, got, want):
    if got.numel() != want.numel():
        return [f"attn_collapse {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"attn_collapse {tag}: non-finite output"]
    rel = _rel(got, w)
    r = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)
    lo, hi = r.min().item(), r.max().item()
    metrics.record(f"rel_l2_swap_attn_collapse_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_attn_collapse_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_attn_collapse_{tag}", hi)
    print(f"attn_collapse {tag}: rel_l2={rel:.5f} (<= {COLLAPSE_MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}]")
    fails = []
    if rel > COLLAPSE_MAX_REL_L2:
        fails.append(f"attn_collapse {tag} rel L2 {rel:.4f} > {COLLAPSE_MAX_REL_L2}")
    if lo < COLLAPSE_RATIO[0] or hi > COLLAPSE_RATIO[1]:
        fails.append(f"attn_collapse {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {COLLAPSE_RATIO}")
    return fails


def _norm_checks(tag, got, want):
    if got.numel() != want.numel():
        return [f"attn_norm {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"attn_norm {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row_rel = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_swap_attn_norm_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_attn_norm_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_attn_norm_{tag}", hi)
    metrics.record(f"max_row_rel_l2_swap_attn_norm_{tag}", row_rel)
    print(
        f"attn_norm {tag}: rel_l2={rel:.5f} (<= {NORM_MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}] "
        f"worst row {row_rel:.4f} (<= {NORM_MAX_ROW_REL_L2})"
    )
    fails = []
    if rel > NORM_MAX_REL_L2:
        fails.append(f"attn_norm {tag} rel L2 {rel:.4f} > {NORM_MAX_REL_L2}")
    if lo < NORM_RATIO[0] or hi > NORM_RATIO[1]:
        fails.append(f"attn_norm {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {NORM_RATIO}")
    if row_rel > NORM_MAX_ROW_REL_L2:
        fails.append(f"attn_norm {tag} worst row rel L2 {row_rel:.4f} > {NORM_MAX_ROW_REL_L2}")
    return fails


def _attn_checks(tag, got, want):
    if got.numel() != want.numel():
        return [f"attention {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"attention {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row_rel = ((got - w).norm(dim=-1) / wn).max().item()
    blocks = [
        _rel(got[i : i + ATTN_BLOCK_ROWS], w[i : i + ATTN_BLOCK_ROWS]) for i in range(0, w.shape[0], ATTN_BLOCK_ROWS)
    ]
    blk = max(blocks)
    blk_at = blocks.index(blk) * ATTN_BLOCK_ROWS
    metrics.record(f"rel_l2_swap_attention_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_attention_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_attention_{tag}", hi)
    metrics.record(f"max_block_rel_l2_swap_attention_{tag}", blk)
    metrics.record(f"max_row_rel_l2_swap_attention_{tag}", row_rel)
    print(
        f"attention {tag}: rel_l2={rel:.5f} (<= {ATTN_MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}] "
        f"worst block {blk:.4f} at rows {blk_at}.. (<= {ATTN_MAX_BLOCK_REL_L2}) worst row {row_rel:.4f} "
        f"(<= {ATTN_MAX_ROW_REL_L2})"
    )
    print(f"attention {tag} block rel_l2: " + " ".join(f"{b:.4f}" for b in blocks))
    fails = []
    if rel > ATTN_MAX_REL_L2:
        fails.append(f"attention {tag} rel L2 {rel:.4f} > {ATTN_MAX_REL_L2}")
    if lo < ATTN_RATIO[0] or hi > ATTN_RATIO[1]:
        fails.append(f"attention {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {ATTN_RATIO}")
    if blk > ATTN_MAX_BLOCK_REL_L2:
        fails.append(
            f"attention {tag} rows {blk_at}..{blk_at + ATTN_BLOCK_ROWS - 1} rel L2 {blk:.4f} > {ATTN_MAX_BLOCK_REL_L2}"
        )
    if row_rel > ATTN_MAX_ROW_REL_L2:
        fails.append(f"attention {tag} worst row rel L2 {row_rel:.4f} > {ATTN_MAX_ROW_REL_L2}")
    return fails


def _state_checks(tag, got_state, want_state):
    fails = []
    for name, lim in MAX_STATE_REL_L2.items():
        if name not in got_state:
            fails.append(f"state {tag}: no {name}")
            continue
        want, got = want_state[name].float(), got_state[name].float()
        if got.numel() != want.numel():
            fails.append(f"state {tag} {name}: {tuple(got.shape)}, want {tuple(want.shape)}")
            continue
        got = got.reshape(want.shape)
        if not torch.isfinite(got).all():
            fails.append(f"state {tag} {name} not finite")
            continue
        rel = _rel(got, want)
        metrics.record(f"rel_l2_swap_state_{name}_{tag}", rel)
        msg = f"state {tag} {name}: rel_l2={rel:.5f} (<= {lim})"
        if rel > lim:
            fails.append(f"state {tag} {name} rel L2 {rel:.4f} > {lim}")
        if name == "kda_recurrent":
            d = (got - want).flatten(1).norm(dim=1) / want.flatten(1).norm(dim=1).clamp_min(1e-12)
            head = d.max().item()
            metrics.record(f"max_head_rel_l2_swap_state_{name}_{tag}", head)
            msg += f" worst head {head:.5f} (<= {MAX_STATE_HEAD_REL_L2})"
            if head > MAX_STATE_HEAD_REL_L2:
                fails.append(f"state {tag} {name} worst head rel L2 {head:.4f} > {MAX_STATE_HEAD_REL_L2}")
        print(msg)
    return fails


def _terms(x, hc, y):
    """(post * attn_out, comb^T @ in) in fp32, [S * N, H] each."""
    h = x.shape[-1]
    post = hc[:, N : 2 * N].float()
    comb = hc[:, 2 * N :].float().reshape(-1, N, N)
    pterm = (post.unsqueeze(-1) * y.float().unsqueeze(-2)).reshape(-1, h)
    cterm = torch.matmul(comb.transpose(-1, -2), x.float().view(-1, N, h)).reshape(-1, h)
    return pterm, cterm


def _res_checks(tag, lim, got, want, inputs=None):
    """h_mid vs want at the `lim` limits; with `inputs` (in, attn_hc, attn_out) also each term on its own."""
    if got.numel() != want.numel():
        return [f"attn_residual {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"attn_residual {tag}: non-finite output"]
    h = w.shape[-1]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    gs, ws = got.view(-1, N, h), w.view(-1, N, h)
    srel = [_rel(gs[:, n], ws[:, n]) for n in range(N)]
    metrics.record(f"rel_l2_swap_attn_residual_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_attn_residual_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_attn_residual_{tag}", hi)
    metrics.record(f"max_row_rel_l2_swap_attn_residual_{tag}", row)
    metrics.record(f"max_stream_rel_l2_swap_attn_residual_{tag}", max(srel))
    print(
        f"attn_residual {tag}: rel_l2={rel:.5f} (<= {RES_MAX_REL_L2[lim]}) norm ratio [{lo:.4f}, {hi:.4f}] "
        f"(in {RES_RATIO[lim]}) worst row {row:.4f} (<= {RES_MAX_ROW_REL}) per-stream "
        f"{[round(s, 5) for s in srel]} (<= {RES_MAX_STREAM_REL[lim]})"
    )
    fails = []
    if rel > RES_MAX_REL_L2[lim]:
        fails.append(f"attn_residual {tag} rel L2 {rel:.4f} > {RES_MAX_REL_L2[lim]}")
    if lo < RES_RATIO[lim][0] or hi > RES_RATIO[lim][1]:
        fails.append(f"attn_residual {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {RES_RATIO[lim]}")
    if row > RES_MAX_ROW_REL:
        fails.append(f"attn_residual {tag} worst row rel L2 {row:.4f} > {RES_MAX_ROW_REL}")
    if max(srel) > RES_MAX_STREAM_REL[lim]:
        fails.append(f"attn_residual {tag} per-stream rel L2 {max(srel):.4f} > {RES_MAX_STREAM_REL[lim]}")
    if inputs is None:
        return fails
    pterm, cterm = _terms(*inputs)
    for name, term, other in (("post_term", pterm, cterm), ("comb_term", cterm, pterm)):
        d = got - other
        coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
        trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
        metrics.record(f"{name}_coef_swap_attn_residual_{tag}", coef)
        metrics.record(f"{name}_rel_l2_swap_attn_residual_{tag}", trel)
        print(f"attn_residual {tag}: {name} coef={coef:.4f} (in {TERM_COEF}) rel={trel:.4f} (<= {MAX_TERM_REL})")
        if not TERM_COEF[0] <= coef <= TERM_COEF[1]:
            fails.append(f"attn_residual {tag} {name} coefficient {coef:.4f} outside {TERM_COEF}")
        if trel > MAX_TERM_REL:
            fails.append(f"attn_residual {tag} {name} rel L2 {trel:.4f} > {MAX_TERM_REL}")
    return fails


def _fr_checks(tag, got, want, inputs):
    """ffn_residual out [S * N, H] vs want, and each term on its own from `inputs` (h_mid, ffn_hc, mlp_out)."""
    if got.numel() != want.numel():
        return [f"ffn_residual {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"ffn_residual {tag}: non-finite output"]
    h = w.shape[-1]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = _rel(got, w)
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    gs, ws = got.view(-1, N, h), w.view(-1, N, h)
    srel = [_rel(gs[:, n], ws[:, n]) for n in range(N)]
    metrics.record(f"rel_l2_swap_ffn_residual_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_ffn_residual_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_ffn_residual_{tag}", hi)
    metrics.record(f"max_row_rel_l2_swap_ffn_residual_{tag}", row)
    metrics.record(f"max_stream_rel_l2_swap_ffn_residual_{tag}", max(srel))
    print(
        f"ffn_residual {tag}: rel_l2={rel:.5f} (<= {FR_MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}] (in {FR_RATIO}) "
        f"worst row {row:.4f} (<= {FR_MAX_ROW_REL}) per-stream {[round(x, 5) for x in srel]} (<= {FR_MAX_STREAM_REL})"
    )
    fails = []
    if rel > FR_MAX_REL_L2:
        fails.append(f"ffn_residual {tag} rel L2 {rel:.4f} > {FR_MAX_REL_L2}")
    if lo < FR_RATIO[0] or hi > FR_RATIO[1]:
        fails.append(f"ffn_residual {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {FR_RATIO}")
    if row > FR_MAX_ROW_REL:
        fails.append(f"ffn_residual {tag} worst row rel L2 {row:.4f} > {FR_MAX_ROW_REL}")
    if max(srel) > FR_MAX_STREAM_REL:
        fails.append(f"ffn_residual {tag} per-stream rel L2 {max(srel):.4f} > {FR_MAX_STREAM_REL}")
    pterm, cterm = _terms(*inputs)
    for name, term, other in (("post_term", pterm, cterm), ("comb_term", cterm, pterm)):
        d = got - other
        coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
        trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
        metrics.record(f"{name}_coef_swap_ffn_residual_{tag}", coef)
        metrics.record(f"{name}_rel_l2_swap_ffn_residual_{tag}", trel)
        print(f"ffn_residual {tag}: {name} coef={coef:.4f} (in {FR_TERM_COEF}) rel={trel:.4f} (<= {FR_MAX_TERM_REL})")
        if not FR_TERM_COEF[0] <= coef <= FR_TERM_COEF[1]:
            fails.append(f"ffn_residual {tag} {name} coefficient {coef:.4f} outside {FR_TERM_COEF}")
        if trel > FR_MAX_TERM_REL:
            fails.append(f"ffn_residual {tag} {name} rel L2 {trel:.4f} > {FR_MAX_TERM_REL}")
    return fails


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    overrides, muts = {}, {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
        muts[name] = mut
        overrides[name] = lambda ctx, *x, mut=mut: _f32(mut(ctx, dctx, *x))
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

    failures = [] if ok else [f"pcc_swap_out below {thr}"]
    out, w = seen["out"].float().reshape(gl["out"].shape), gl["out"].float()
    out_rel = _rel(out, w)
    ratio = out.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("row_norm_ratio_min_swap_out", rmin)
    metrics.record("row_norm_ratio_max_swap_out", rmax)
    print(f"block out: rel_l2={out_rel:.6f} (<= {OUT_MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}]")
    if not torch.isfinite(out).all():
        failures.append("block out non-finite")
    if out_rel > OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2}")
    if not (OUT_ROW_NORM_RATIO[0] <= rmin and rmax <= OUT_ROW_NORM_RATIO[1]):
        failures.append(f"block out per-row norm ratio [{rmin:.4f}, {rmax:.4f}] outside {OUT_ROW_NORM_RATIO}")
    failures += _hc_checks(seen["attn_hc"], gl["attn_hc"])

    # attn_collapse: vs golden, and vs the CPU collapse of the inputs it actually got (golden in, swapped attn_hc).
    failures += _collapse_checks("vs_golden", seen["attn_in"], gl["attn_in"])
    cpu_in = ref.component(layer, "attn_collapse")(rctx, gl["in"].float(), seen["attn_hc"].float())
    failures += _collapse_checks("vs_cpu_same_input", seen["attn_in"], cpu_in)

    # Same module on a layer whose streams differ (layer 0's four streams are identical copies).
    assert S.block_type_of(DISTINCT_LAYER) == BLOCK_TYPE
    gd = g.layer(c, DISTINCT_LAYER)
    rctx_d = reference_ctx(ref, layer, g, c)
    out_d = muts["attn_collapse"](rctx_d, device_ctx(DISTINCT_LAYER, g, c), gd["in"].float(), gd["attn_hc"].float())
    failures += _collapse_checks(f"L{DISTINCT_LAYER:02d}_distinct_streams", out_d, gd["attn_in"])

    # attn_norm: vs golden, and vs the CPU norm of the attn_in it actually got (device collapse).
    failures += _norm_checks("vs_golden", seen["attn_norm"], gl["attn_norm"])
    cpu_norm = ref.component(layer, "attn_norm")(rctx, seen["attn_in"].float().reshape(gl["attn_in"].shape))
    failures += _norm_checks("vs_cpu_same_input", seen["attn_norm"], cpu_norm)

    # attention: vs golden, and vs the CPU KDA of the attn_norm it actually got (fresh ctx: same golden prefix state).
    failures += _attn_checks("vs_golden", seen["attn_out"], gl["attn_out"])
    rctx_a = reference_ctx(ref, layer, g, c)
    cpu_attn = ref.component(layer, "attention")(rctx_a, seen["attn_norm"].float().reshape(gl["attn_norm"].shape))
    failures += _attn_checks("vs_cpu_same_input", seen["attn_out"], cpu_attn)

    # KDA state after the chunk.
    want_state = g.state(layer, at=c * g.chunk + g.chunk)
    mode = impl_mode()
    if mode == "device":
        got_state = dctx.extra.get("state_out")
        if got_state is None:
            failures.append("device attention did not expose dctx.extra['state_out']")
        else:
            failures += _state_checks("vs_golden", got_state, want_state)
    elif mode == "reference":
        failures += _state_checks("vs_golden", ref.state_tensors(rctx.state, layer, g.seq), want_state)
    # attn_residual: vs golden h_mid (carries the upstream device error), vs the CPU residual of the inputs it actually
    # got (golden in, device attn_hc, device attn_out) with each term on its own, and on layer 1's golden.
    failures += _res_checks("vs_golden", "golden", seen["h_mid"], gl["h_mid"])
    res_in = [
        gl["in"].float(),
        seen["attn_hc"].float().reshape(gl["attn_hc"].shape),
        seen["attn_out"].float().reshape(gl["attn_out"].shape),
    ]
    cpu_res = ref.component(layer, "attn_residual")(reference_ctx(ref, layer, g, c), *res_in)
    failures += _res_checks("vs_cpu_same_input", "cpu", seen["h_mid"], cpu_res, res_in)
    res_in_d = [gd["in"].float(), gd["attn_hc"].float(), gd["attn_out"].float()]
    out_rd = muts["attn_residual"](reference_ctx(ref, layer, g, c), device_ctx(DISTINCT_LAYER, g, c), *res_in_d)
    failures += _res_checks(f"L{DISTINCT_LAYER:02d}_distinct_streams", "cpu", out_rd, gd["h_mid"], res_in_d)

    # ffn_hc: vs golden (carries the upstream h_mid error), and vs the CPU ffn_hc of the h_mid it actually got.
    lim = FFN_HC_MAX_REL_L2, FFN_HC_MAX_ABS
    failures += _hc_checks(seen["ffn_hc"], gl["ffn_hc"], "ffn_hc_vs_golden", lim[0]["golden"], lim[1]["golden"])
    h_mid = seen["h_mid"].float().reshape(gl["h_mid"].shape)
    cpu_hc = ref.component(layer, "ffn_hc")(reference_ctx(ref, layer, g, c), h_mid)
    failures += _hc_checks(seen["ffn_hc"], cpu_hc, "ffn_hc_vs_cpu_same_input", lim[0]["cpu"], lim[1]["cpu"])

    # ffn_collapse: vs golden (carries the upstream error), vs the CPU collapse of the (h_mid, ffn_hc) it actually got,
    # and on layer 1's golden.
    hc_dev = seen["ffn_hc"].float().reshape(gl["ffn_hc"].shape)
    failures += _fc_checks("vs_golden", "golden", seen["ffn_in"], gl["ffn_in"])
    cpu_fc = ref.component(layer, "ffn_collapse")(reference_ctx(ref, layer, g, c), h_mid, hc_dev)
    failures += _fc_checks("vs_cpu_same_input", "cpu", seen["ffn_in"], cpu_fc)
    out_fd = muts["ffn_collapse"](
        reference_ctx(ref, layer, g, c), device_ctx(DISTINCT_LAYER, g, c), gd["h_mid"].float(), gd["ffn_hc"].float()
    )
    failures += _fc_checks(f"L{DISTINCT_LAYER:02d}_distinct_streams", "cpu", out_fd, gd["ffn_in"])

    # ffn_norm: vs golden (carries the upstream error), vs the CPU norm of the ffn_in it actually got (device
    # collapse), and on layer 1's golden ffn_in (other row scales; layer-0 weights on both sides).
    fc_dev = seen["ffn_in"].float().reshape(gl["ffn_in"].shape)
    failures += _fn_checks("vs_golden", "golden", seen["ffn_norm"], gl["ffn_norm"])
    cpu_fn = ref.component(layer, "ffn_norm")(reference_ctx(ref, layer, g, c), fc_dev)
    failures += _fn_checks("vs_cpu_same_input", "cpu", seen["ffn_norm"], cpu_fn)
    out_nd = muts["ffn_norm"](reference_ctx(ref, layer, g, c), device_ctx(DISTINCT_LAYER, g, c), gd["ffn_in"].float())
    cpu_nd = ref.component(layer, "ffn_norm")(reference_ctx(ref, layer, g, c), gd["ffn_in"].float())
    failures += _fn_checks(f"L{DISTINCT_LAYER:02d}_input", "cpu", out_nd, cpu_nd)

    # mlp: vs golden (carries the upstream ffn_norm error), vs the CPU mlp of the ffn_norm it actually got, on layer 1's
    # golden ffn_norm (layer-0 weights on both sides), and the clamp probe (the golden never reaches the limit).
    fn_dev = seen["ffn_norm"].float().reshape(gl["ffn_norm"].shape)
    cpu_mlp = lambda x: ref.component(layer, "mlp")(reference_ctx(ref, layer, g, c), x)
    dev_mlp = lambda x, lyr=layer: _f32(muts["mlp"](reference_ctx(ref, layer, g, c), device_ctx(lyr, g, c), x))
    failures += _mlp_checks("vs_golden", "golden", seen["mlp_out"], gl["mlp_out"], coef_lim=MLP_COEF["golden"])
    cpu_y = cpu_mlp(fn_dev)
    failures += _mlp_checks("vs_cpu_same_input", "cpu", seen["mlp_out"], cpu_y, coef_lim=MLP_COEF["cpu"])
    xd = gd["ffn_norm"].float()
    failures += _mlp_checks(
        f"L{DISTINCT_LAYER:02d}_input", "cpu", dev_mlp(xd, DISTINCT_LAYER), cpu_mlp(xd), coef_lim=MLP_COEF["cpu"]
    )
    xp = fn_dev * CLAMP_PROBE_SCALE
    failures += _mlp_checks(
        "clamp_probe", "cpu", dev_mlp(xp), cpu_mlp(xp), PROBE_MAX_REL_L2, PROBE_RATIO, PROBE_MAX_ROW_REL_L2
    )

    # Shares of the block out: tails from the same device h_mid. (a) CPU ffn_hc + collapse + norm + mlp: all four
    # device ffn steps' share; (b) device ffn_hc, CPU collapse, device norm + mlp of it: ffn_collapse's share alone;
    # (c) device ffn_in, CPU norm, device mlp of it: ffn_norm's share alone; (d) device ffn_norm, CPU mlp: mlp's share.
    failures += _tail_checks(
        "cpu_tail",
        out,
        _tail(ref, layer, reference_ctx(ref, layer, g, c), h_mid, cpu_hc.float()),
        TAIL_MAX_REL_L2,
        TAIL_RATIO,
    )
    dev_res = lambda x, hc, y, lyr=layer: _f32(
        muts["ffn_residual"](reference_ctx(ref, layer, g, c), device_ctx(lyr, g, c), x, hc, y)
    ).reshape(gl["out"].shape if lyr == layer else gd["out"].shape)
    res = lambda y: dev_res(h_mid, hc_dev, y)
    dev_fn_of_cpu_fc = muts["ffn_norm"](reference_ctx(ref, layer, g, c), device_ctx(layer, g, c), cpu_fc.float())
    dev_fn_of_cpu_fc = dev_fn_of_cpu_fc.float().reshape(gl["ffn_norm"].shape)
    fc_tail = res(dev_mlp(dev_fn_of_cpu_fc).reshape(gl["mlp_out"].shape))
    failures += _tail_checks("cpu_fc_tail", out, fc_tail, FC_TAIL_MAX_REL_L2, FC_TAIL_RATIO)
    fn_tail = res(dev_mlp(cpu_fn.float()).reshape(gl["mlp_out"].shape))
    failures += _tail_checks("cpu_fn_tail", out, fn_tail, FN_TAIL_MAX_REL_L2, FN_TAIL_RATIO)
    failures += _tail_checks("cpu_mlp_tail", out, res(cpu_y.float()), MLP_TAIL_MAX_REL_L2, MLP_TAIL_RATIO)

    # ffn_residual: block out vs the CPU residual of the (h_mid, ffn_hc, mlp_out) it actually got, each term on its own,
    # and the (weightless) module on layer 1's golden.
    fr_in = [h_mid, hc_dev, seen["mlp_out"].float().reshape(gl["mlp_out"].shape)]
    cpu_fr = ref.component(layer, "ffn_residual")(reference_ctx(ref, layer, g, c), *fr_in)
    failures += _fr_checks("vs_cpu_same_input", out, cpu_fr, fr_in)
    fr_in_d = [gd["h_mid"].float(), gd["ffn_hc"].float(), gd["mlp_out"].float()]
    cpu_fr_d = ref.component(layer, "ffn_residual")(reference_ctx(ref, layer, g, c), *fr_in_d)
    failures += _fr_checks(
        f"L{DISTINCT_LAYER:02d}_distinct_streams", dev_res(*fr_in_d, DISTINCT_LAYER), cpu_fr_d, fr_in_d
    )
    assert not failures, "; ".join(failures)
