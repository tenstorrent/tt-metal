# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 4: block type kda_dense (layer 0) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (kda_dense) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    attention

Reviewed (S.kda_dense.04.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). Block out is dominated by the residual (comb^T @ in) and PCC there misses
every KDA state bug (the attention component review: a zeroed recurrent prefix scores attn_out PCC 0.99935), so the
swap test checks the KDA output and its post-chunk state directly, at the attention component test's limits.
Extra asserted checks (informational metrics, not in the runner's list):
  - block out: finite, rel L2 <= OUT_MAX_REL_L2, per-token norm ratio in [0.97, 1.03];
  - attn_hc, attn_collapse (incl. layer 1's golden), attn_norm: as swap 03;
  - attn_out vs golden attn_out and vs the CPU KDA of the attn_norm it actually got (device norm, same golden prefix
    state): rel L2 <= 0.02, per-token norm ratio [0.98, 1.02], every 128-row block rel L2 <= 0.03, worst per-token
    rel L2 <= 0.1 (the component test's limits);
  - KDA state after the chunk vs the golden snapshot at start + chunk (recurrent rel <= 0.03, worst head <= 0.05, conv
    rel <= 0.02), from ``dctx.extra["state_out"]`` on the device and the CPU state in reference mode.
Block out at layer 0 follows the attention error (post * attn_out outweighs the residual here, unlike a sink-dominated
attention): CPU runs with only the attention perturbed give out rel L2 / per-token ratio: reference 0.0017 / [0.9998,
1.0002]; attn x0.99 0.0096 / [0.977, 1.008]; x1.02 0.0185 / [0.986, 1.047]; +3% noise 0.023; zeroed prefix state
0.045 / [0.65, 1.13]; zeroed attention 1.60. So block out keeps swap 03's limits (rel <= 0.01, ratio [0.97, 1.03]).
Device (S.kda_dense.04.test.1): out PCC 0.999984, rel 0.0062, ratio [0.9827, 1.0078]; attention vs golden 0.0073 /
[0.9895, 0.9978] / worst block 0.0074 / worst row 0.014, vs CPU same input 0.0068; state recurrent 0.0135 / worst
head 0.024, conv 0.0020.
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
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "attention"]
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
    assert not failures, "; ".join(failures)
