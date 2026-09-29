# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 5: block type kda_dense (layer 0) with attn_residual swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (kda_dense) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    attention
    attn_residual

Reviewed (S.kda_dense.05.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). attn_residual is h_mid = post * attn_out + comb^T @ in. At layer 0 the four
streams are identical and comb columns sum to 1, so comb^T @ in = in for any column-stochastic comb: an identity comb
scores h_mid PCC 0.999996 (component review). Block out cannot see comb bugs, so the swap test checks h_mid directly
and runs the weightless module again on layer 1's golden, as the component test does.
Extra asserted checks (informational metrics, not in the runner's list):
  - block out: finite, rel L2 <= 0.01, per-token norm ratio in [0.97, 1.03] (as swap 04);
  - attn_hc, attn_collapse (incl. layer 1's golden), attn_norm, attention, KDA post-chunk state: as swap 04;
  - h_mid vs the CPU residual of the inputs it actually got (golden in, device attn_hc, device attn_out), and the same
    module on layer 1's golden: the component test's limits (rel L2 <= 0.01, per-token ratio [0.985, 1.015],
    per-stream rel <= 0.015, worst row <= 0.05), plus each term on its own (out minus the exact other term vs
    post * attn_out and vs comb^T @ in: coefficient [0.98, 1.02], rel <= 0.02);
  - h_mid vs golden h_mid, which carries the upstream device error (mostly the attention's 0.995 scale, weighted by
    post): rel <= 0.015, ratio [0.97, 1.03], per-stream <= 0.02, worst row <= 0.05.
Device (S.kda_dense.05.test.1): out PCC 0.999982, rel 0.0064, ratio [0.9828, 1.0078]; h_mid vs golden 0.0084 /
[0.9883, 1.0042] / streams 0.0115, 0.0114, 0.0023, 0.0029 / worst row 0.021; vs CPU same input 0.0017, terms coef
1.0000 / rel 0.0015; layer 1 0.0031, terms rel 0.0023. Reference: out rel 0.0017, h_mid vs golden 0.0017, layer 1
0.0027. Stub: PCC 0, every check fails.
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
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "attention", "attn_residual"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
OUT_MAX_REL_L2 = 0.01  # block out: ||got - want|| / ||want||
OUT_ROW_NORM_RATIO = (0.97, 1.03)  # block out: per-row ||got|| / ||want||
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.03, "post": 0.03, "comb": 0.02}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.15, "post": 0.2, "comb": 0.1}
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|
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


def _hc_checks(got, want):
    fails = []
    if got.numel() != want.numel():
        return [f"attn_hc: shape {tuple(got.shape)} vs golden {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return ["attn_hc: non-finite output"]
    for name, sl in PARTS.items():
        a, b = got[:, sl], w[:, sl]
        rel = ((a - b).norm() / b.norm()).item()
        mx = (a - b).abs().max().item()
        metrics.record(f"rel_l2_swap_attn_hc_{name}", rel)
        metrics.record(f"max_abs_swap_attn_hc_{name}", mx)
        print(f"attn_hc {name}: rel_l2={rel:.5f} (<= {MAX_REL_L2[name]}) max_abs={mx:.2e} (<= {MAX_ABS[name]})")
        if rel > MAX_REL_L2[name]:
            fails.append(f"attn_hc {name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if mx > MAX_ABS[name]:
            fails.append(f"attn_hc {name} max abs {mx:.3f} > {MAX_ABS[name]}")
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record("comb_col_sum_err_swap_attn_hc", col_err)
    print(f"attn_hc comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if col_err > COL_SUM_TOL:
        fails.append(f"attn_hc comb column sums off 1 by {col_err:.4f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"attn_hc pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"attn_hc post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append("attn_hc negative comb entries")
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
    assert not failures, "; ".join(failures)
