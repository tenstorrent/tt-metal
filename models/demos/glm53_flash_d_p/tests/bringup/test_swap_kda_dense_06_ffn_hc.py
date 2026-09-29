# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 6: block type kda_dense (layer 0) with ffn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (kda_dense) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    attention
    attn_residual
    ffn_hc

Reviewed (S.kda_dense.06.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). ffn_hc is the attn_hc op with the hc_ffn_* weights on h_mid; its output
[S, 24] (pre | post | comb) feeds ffn_collapse (pre, then ffn_norm hides scale errors) and ffn_residual
(out = post * mlp_out + comb^T @ h_mid; the h_mid streams differ at layer 0, so comb is visible in block out).
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 05 asserts (block out rel L2 <= 0.01 and per-token ratio [0.97, 1.03], attn_hc, attn_collapse,
    attn_norm, attention, KDA state, attn_residual incl. layer 1's golden), unchanged;
  - ffn_hc vs the CPU ffn_hc of the h_mid it actually got: the component test's limits (per part rel L2 <= 0.01,
    max abs <= 0.05, comb column sums within 0.01, pre in (0, 1], post in [0, 2], comb >= 0);
  - ffn_hc vs golden (carries the upstream h_mid error): per part rel L2 <= 0.015, max abs <= 0.06;
  - ffn_hc's share of block out alone: block out vs the CPU tail (ffn_collapse, ffn_norm, mlp, ffn_residual) run
    from the same device h_mid with the CPU ffn_hc. Every other difference cancels, so this isolates the swapped step:
    rel L2 <= 0.005, per-row ratio [0.975, 1.025].
Sensitivity (CPU host script on the layer-0 golden h_mid, not kept), tail rel L2 / per-row ratio: comb transposed
0.098 / [0.59, 1.40]; softmax wrong axis 0.035; 10 Sinkhorn iterations 0.067; hc_eps 1e-5 0.017 / [0.878, 1.008];
comb base transposed 0.0061 / [0.953, 1.029]; rms eps 1.2e-5 0.026; post x1.01 0.0080; post x1.02 0.016 /
[0.991, 1.032]; comb x1.02 0.024; pre x1.02 0.0023 / [0.9935, 1.0235] (caught only by the part rel 0.02 > 0.01);
last row zeroed 0.012; last row = previous row 0.0086 / [0.53, 1.36]; reversed post or pre stream order 0.94 / 0.35.
Noise: mix rounded to bf16 0.0018 / [0.989, 1.010]; output bf16 0.0021 / [0.994, 1.006]; 0.3% mix noise 0.0036 /
[0.979, 1.020]; 1% mix noise 0.012 / [0.931, 1.080] (fails). Reference tail vs golden out: 0.0023.
Device (S.kda_dense.06.test.1): out PCC 0.999982, rel 0.0069, ratio [0.9797, 1.0087] (swap 05: 0.9828; the
ffn_hc noise on small rows); ffn_hc vs CPU same input 0.0024 / 0.0013 / 0.0027, max abs <= 0.0087, column sums
[0.9965, 1.0008]; vs golden 0.0031 / 0.0027 / 0.0045, max abs <= 0.0156; tail rel 0.0020, ratio [0.9873, 1.0129].
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
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "attention", "attn_residual", "ffn_hc"]
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
TAIL_MAX_REL_L2 = 0.005  # block out vs the CPU tail with the CPU ffn_hc of the same h_mid: ffn_hc's share alone
TAIL_RATIO = (0.975, 1.025)  # per-row ||out|| / ||tail||; bf16-mix noise gives [0.989, 1.010]
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
TERM_COEF = (0.98, 1.02)  # <out - other term, term> / ||term||^2, terms from the step's actual inputs
MAX_TERM_REL = 0.02  # ||out - other term - term|| / ||term||


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


def _tail(ref, layer, ctx, h_mid, hc):
    """CPU ffn_collapse -> ffn_norm -> mlp -> ffn_residual from (h_mid, ffn_hc)."""
    comp = lambda n: ref.component(layer, n)
    x = comp("ffn_collapse")(ctx, h_mid, hc)
    y = comp("mlp")(ctx, comp("ffn_norm")(ctx, x))
    return comp("ffn_residual")(ctx, h_mid, hc, y)


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

    # ffn_hc's share of the block out alone: the CPU tail from the same h_mid with the CPU ffn_hc.
    tctx = reference_ctx(ref, layer, g, c)
    tail = _tail(ref, layer, tctx, h_mid, cpu_hc.float())
    t_rel = _rel(out, tail)
    tr = out.norm(dim=-1) / tail.float().norm(dim=-1).clamp_min(1e-12)
    tlo, thi = tr.min().item(), tr.max().item()
    metrics.record("rel_l2_swap_out_vs_cpu_tail", t_rel)
    metrics.record("row_norm_ratio_min_swap_out_vs_cpu_tail", tlo)
    metrics.record("row_norm_ratio_max_swap_out_vs_cpu_tail", thi)
    print(f"block out vs CPU tail (CPU ffn_hc): rel_l2={t_rel:.6f} (<= {TAIL_MAX_REL_L2}) ratio [{tlo:.4f}, {thi:.4f}]")
    if t_rel > TAIL_MAX_REL_L2:
        failures.append(f"block out vs CPU tail rel L2 {t_rel:.4f} > {TAIL_MAX_REL_L2}")
    if tlo < TAIL_RATIO[0] or thi > TAIL_RATIO[1]:
        failures.append(f"block out vs CPU tail per-row ratio [{tlo:.4f}, {thi:.4f}] outside {TAIL_RATIO}")
    assert not failures, "; ".join(failures)
