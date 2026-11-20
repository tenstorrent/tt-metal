# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 13: block type dsa_moe (layer 3) with shared_expert swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 3 (dsa_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    q_a
    indexer
    attention
    attn_residual
    ffn_hc
    ffn_collapse
    ffn_norm
    router
    experts
    shared_expert

Reviewed (S.dsa_moe.13.test.1): rewritten from swap 12's test. The gated metric is pcc_swap_out (PCC, float
[8192, 4096] = 4-stream residual [S * 4, H], spec block threshold 0.98). shared_expert is one clamped-swiglu expert of
width 2048, shared_out [S, 4096]; the CPU tail adds it to experts_out (moe_add) and ffn_residual puts post * mlp_out on
each stream. No norm follows, so its error reaches block out linearly, at about 0.26x (||shared_out|| is 0.37 of
||mlp_out||, ||post * mlp|| 0.70 of ||out||).
Share design: shares telescope, as in swaps 11 and 12. The new shared-share block (every device output up to experts
fixed, CPU shared_expert and tail) is the base of the shared expert's share (block out vs it; same routing by
construction). The experts share is now that block vs the experts-share block, and reproduces swap 12's experts share
exactly (0.00441 / [0.9982, 1.0034]). The earlier shares are unchanged.
Measured on the CPU (golden s4096 chunk 1; golden tail tensors, shared_out perturbed, tail recomputed). Columns:
shared rel L2 -> block out rel L2 / per-row ratio:
  Noise: 0.0024 elementwise (the component device rel) -> 0.00062 / [0.99993, 1.00009]; 0.0024 absolute -> 0.00074.
  Bugs: x1.003 -> 0.00077 / max 1.0018; x1.005 -> 0.0013 / max 1.0030; x1.01 -> 0.0026 / max 1.0060; x0.99 ->
  min 0.9940; second half x1.005 -> 0.0009 / max 1.0029; last row zeroed -> 0.0016 / min 0.962; last 32 rows ->
  0.032 / min 0.58; zero -> 0.26 / min 0.47; added twice -> 0.26 / max 1.62; rows rolled by one -> 0.35; negated 0.52.
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 12 asserts, limits unchanged (block out vs the all-CPU block, same-routing rel L2 <= 0.0075: the
    shared share adds about 0.0006 in quadrature);
  - experts share: the shared-share block vs the experts-share block (limits as swap 12);
  - shared share: block out vs the shared-share block: no routing difference, rel L2 <= 0.0015, ratio [0.998, 1.002]
    (catches x1.005 by ratio, x1.01 by both, a zeroed row by ratio);
  - shared_expert vs the fp32 CPU shared expert of the same ffn_norm: the component test's limits (rel L2 <= 0.008,
    ratio [0.993, 1.007], worst row <= 0.015, coefficient [0.997, 1.003], every 128-row block [0.996, 1.004]); also
    on chunk 0;
  - shared_expert vs golden (carries the upstream ffn_norm error): rel L2 <= 0.012, ratio [0.985, 1.015], worst row
    <= 0.03, coefficient [0.997, 1.003], blocks [0.995, 1.005] (the CPU shared expert of the device ffn_norm: 0.00286 /
    [0.9965, 1.0035] / 0.0157 / 0.99973; the CPU reference on the golden input already has worst row 0.0155).
  The swiglu clamp limits are covered by the component test's clamp probe, not repeated here.
Device (first run): PCC 0.999981 (c0 0.999975); every swap 12 number unchanged (experts share 0.00441 / [0.9982,
1.0034]); block out vs golden 0.00614, 74 flips / same-routing 0.00533 / [0.9907, 1.0070]; vs the all-CPU block 70 /
0.00500 / [0.9909, 1.0075]; shared share 0.00056 / [0.9998, 1.0001]; shared_expert vs CPU same input 0.00216 /
[0.9993, 1.0004] / 0.00264 / 0.99985 / blocks [0.99981, 0.99989] (c0 0.00216 / 0.00273 / 0.99983); vs golden
0.00351 / [0.9964, 1.0031] / 0.0157 / 0.99958 / blocks [0.99932, 0.99982].
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
BLOCK_TYPE = "dsa_moe"
SWAPPED = [
    "attn_hc",
    "attn_collapse",
    "attn_norm",
    "q_a",
    "indexer",
    "attention",
    "attn_residual",
    "ffn_hc",
    "ffn_collapse",
    "ffn_norm",
    "router",
    "experts",
    "shared_expert",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.975, 1.025)  # per-row ||got|| / ||want||, rows whose top-8 equals the golden's
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 96  # rows whose top-8 differs from the CPU block of the same `in` (bf16 noise; rel L2 gates)
CPU_MAX_REL_L2 = 0.0075  # block out vs CPU block, same-routing rows (swap 11 0.00222 + the experts share 0.0044)
CPU_ROW_RATIO = (0.98, 1.02)
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.02, "post": 5e-4, "comb": 0.02}
MAX_COL_REL_L2 = 0.07
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|
COL_GOLD = {"rel": 0.01, "ratio": (0.985, 1.015), "row": 0.02}  # attn_collapse vs golden attn_in (device attn_hc in)
COL_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.0065}  # vs the fp32 CPU collapse of the same inputs
SHARE_MAX_FLIPS = 24  # collapse share: rows whose top-8 differs from the CPU block with device attn_hc, CPU collapse
SHARE_MAX_REL_L2 = 0.0015  # collapse share, same-routing rows
SHARE_RATIO = (0.995, 1.005)
NORM_GOLD = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.015}  # attn_norm vs golden (device attn_in in)
NORM_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.008}  # vs the fp32 CPU norm of the same input
NORM_SHARE_MAX_FLIPS = 20  # norm share: rows whose top-8 differs from the CPU block with device hc + attn_in
NORM_SHARE_MAX_REL_L2 = 0.001
NORM_SHARE_RATIO = (0.995, 1.005)
QA_GOLD = {"rel": 0.01, "ratio": (0.995, 1.005), "row": 0.015}  # q_a vs golden q_resid (device attn_norm in)
QA_SAME = {"rel": 0.0045, "ratio": (0.997, 1.003), "row": 0.008}  # vs the fp32 CPU q_a of the same input
QA_SHARE_MAX_FLIPS = 12  # q_a share: rows whose top-8 differs from the CPU block with device hc, attn_in, attn_norm
QA_SHARE_MAX_REL_L2 = 0.0006
QA_SHARE_RATIO = (0.998, 1.002)
QA_SHARE_TOPK_OVERLAP = 0.9995  # indexer selection with the device q_resid vs with the CPU q_resid (same attn_norm)
KP = 4  # tokens per pool
SEL_POOLS = 512
IDX_GOLD = {"mean": 0.9975, "row": 0.98}  # indexer topk vs golden (device attn_norm, q_resid in): component limits
IDX_SAME = {"mean": 0.9985, "row": 0.988}  # vs the CPU indexer of the same inputs
KEY_GOLD = {"rel": 0.008, "ratio": (0.995, 1.005), "row": 0.02}  # chunk's pooled keys vs golden state
KEY_SAME = {"rel": 0.005, "ratio": (0.996, 1.004), "row": 0.01}  # vs the CPU pooled keys of the same attn_norm
ATTN_SHARE = {"rel": 0.0025, "ratio": (0.993, 1.007), "row": 0.1}  # attn_out: device topk vs CPU topk, same inputs
IDX_SHARE_MAX_FLIPS = 12  # indexer share: rows whose top-8 differs from the CPU block with device hc..q_a, CPU indexer
IDX_SHARE_MAX_REL_L2 = 0.0004
IDX_SHARE_RATIO = (0.998, 1.002)


def _f32(t):
    return t.float() if torch.is_tensor(t) and t.is_floating_point() else t


def _flipped(router, want_router):
    """Per token: top-8 selection (nonzero routing weights) differs."""
    return ((router.float() != 0) != (want_router.float() != 0)).any(dim=-1)


def _topk_overlap(got, want):
    """Mean per-row fraction of want's selected (non-negative) indices that got also selects."""
    got, want = got.long().reshape(want.shape), want.long()
    n = int(max(got.max().item(), want.max().item())) + 2
    mg = torch.zeros(want.shape[0], n, dtype=torch.bool).scatter_(1, got + 1, True)[:, 1:]
    mw = torch.zeros(want.shape[0], n, dtype=torch.bool).scatter_(1, want + 1, True)[:, 1:]
    return ((mg & mw).sum(-1).float() / mw.sum(-1).clamp_min(1).float()).mean().item()


def _row_checks(tag, out, want, flipped_tok, same_lim, flipped_lim, max_rel=None):
    fails = []
    flipped = flipped_tok.repeat_interleave(N)
    ratio = out.norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)
    same = ~flipped
    metrics.record(f"flipped_rows_swap_out_{tag}", int(flipped_tok.sum()))
    if same.any():
        r = ratio[same]
        rmin, rmax = r.min().item(), r.max().item()
        rel = ((out[same] - want[same]).norm() / want[same].norm().clamp_min(1e-12)).item()
        metrics.record(f"rel_l2_same_routing_swap_out_{tag}", rel)
        metrics.record(f"row_norm_ratio_min_swap_out_{tag}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_out_{tag}", rmax)
        print(
            f"block out vs {tag}: {int(flipped_tok.sum())} tokens with another top-8; same-routing rel_l2={rel:.5f}"
            f" ratio=[{rmin:.4f}, {rmax:.4f}] (limit {same_lim})"
        )
        if not (same_lim[0] <= rmin and rmax <= same_lim[1]):
            fails.append(f"block out vs {tag}: same-routing per-row ratio [{rmin:.4f}, {rmax:.4f}] outside {same_lim}")
        if max_rel is not None and not rel <= max_rel:
            fails.append(f"block out vs {tag}: same-routing rel L2 {rel:.4f} > {max_rel}")
    else:
        fails.append(f"block out vs {tag}: every token has another top-8")
    if flipped.any() and flipped_lim is not None:
        r = ratio[flipped]
        rmin, rmax = r.min().item(), r.max().item()
        print(f"block out vs {tag}: flipped rows ratio=[{rmin:.4f}, {rmax:.4f}] (limit {flipped_lim})")
        if not (flipped_lim[0] <= rmin and rmax <= flipped_lim[1]):
            fails.append(f"block out vs {tag}: flipped-row ratio [{rmin:.4f}, {rmax:.4f}] outside {flipped_lim}")
    return fails


def _out_checks(step, tag, got, want, lim):
    if got.numel() != want.numel():
        return [f"{step} {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"{step} {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_swap_{step}_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_{step}_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_{step}_{tag}", hi)
    metrics.record(f"row_rel_l2_max_swap_{step}_{tag}", row)
    print(
        f"{step} {tag}: rel_l2={rel:.5f} (<= {lim['rel']}) norm ratio [{lo:.4f}, {hi:.4f}] (in {lim['ratio']})"
        f" worst row {row:.5f} (<= {lim['row']})"
    )
    fails = []
    if not rel <= lim["rel"]:  # NaN fails
        fails.append(f"{step} {tag} rel L2 {rel:.4f} > {lim['rel']}")
    if not (lim["ratio"][0] <= lo and hi <= lim["ratio"][1]):
        fails.append(f"{step} {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {lim['ratio']}")
    if not row <= lim["row"]:
        fails.append(f"{step} {tag} worst row rel L2 {row:.4f} > {lim['row']}")
    return fails


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
        if not rel <= MAX_REL_L2[name]:
            fails.append(f"attn_hc {name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if not mx <= MAX_ABS[name]:
            fails.append(f"attn_hc {name} max abs {mx:.3g} > {MAX_ABS[name]}")
    col_rel = (got - w).norm(dim=0) / w.norm(dim=0)
    worst = col_rel.max().item()
    metrics.record("worst_col_rel_l2_swap_attn_hc", worst)
    print(f"attn_hc worst column rel L2 {worst:.4f} at column {col_rel.argmax().item()} (<= {MAX_COL_REL_L2})")
    if not worst <= MAX_COL_REL_L2:
        fails.append(f"attn_hc column {col_rel.argmax().item()} rel L2 {worst:.4f} > {MAX_COL_REL_L2}")
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record("comb_col_sum_err_swap_attn_hc", col_err)
    print(f"attn_hc comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if not col_err <= COL_SUM_TOL:
        fails.append(f"attn_hc comb column sums off 1 by {col_err:.4f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"attn_hc pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"attn_hc post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append("attn_hc negative comb entries")
    return fails


def _rowsets(t, n):
    """[S, W] ids (-1 = none) -> [S, n] bool membership; ids outside [0, n) count as none (_structure flags them)."""
    t = t.long()
    t = torch.where((t >= 0) & (t < n), t, torch.full_like(t, -1))
    return torch.zeros(t.shape[0], n + 1, dtype=torch.bool).scatter_(1, t + 1, True)[:, 1:]


def _row_overlap(got, want, n):
    mg, mw = _rowsets(got, n), _rowsets(want, n)
    return (mg & mw).sum(-1).float() / mw.sum(-1).clamp_min(1).float()


def _structure(got, want, start):
    """Exact checks of the indexer output against the golden's (as the component test)."""
    fails = []
    s = want.shape[0]
    n = start + s
    qpos = torch.arange(start, start + s)[:, None]
    bad = (got < -1) | (got >= n)
    if bad.any():
        return [f"indexer: {int(bad.sum())} ids outside [-1, {n})"]
    valid = got >= 0
    viol = valid & (got > qpos)
    if viol.any():
        fails.append(f"indexer: {int(viol.sum())} ids after their query position on {int(viol.any(-1).sum())} rows")
    srt = got.sort(dim=-1).values
    dup = (srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)
    if dup.any():
        fails.append(f"indexer: duplicate ids on {int(dup.any(-1).sum())} rows")
    cnt, wcnt = valid.sum(-1), (want >= 0).sum(-1)
    if not torch.equal(cnt, wcnt):
        fails.append(f"indexer: {int((cnt != wcnt).sum())} rows select another number of ids than the golden")
    pool_end = (qpos + 1) // KP * KP
    in_pools = valid & (got < pool_end)
    pid = torch.where(in_pools, got // KP, torch.zeros_like(got))
    pc = torch.zeros(s, n // KP + 1, dtype=torch.int64).scatter_add_(1, pid, in_pools.long())
    if ((pc != 0) & (pc != KP)).any():
        fails.append(f"indexer: incomplete pools selected on {int(((pc != 0) & (pc != KP)).any(-1).sum())} rows")
    full = ((qpos[:, 0] + 1) // KP) <= SEL_POOLS
    if full.any():
        neq = (_rowsets(got[full], n) != _rowsets(want[full], n)).any(-1)
        if neq.any():
            fails.append(f"indexer: {int(neq.sum())} of {int(full.sum())} all-visible rows select another set")
    print(f"indexer structure: {'ok' if not fails else '; '.join(fails)}")
    return fails


def _overlap_checks(tag, got, want, n, lim):
    ov = _row_overlap(got, want, n)
    mean, worst = ov.mean().item(), ov.min().item()
    metrics.record(f"topk_overlap_mean_swap_indexer_{tag}", mean)
    metrics.record(f"topk_overlap_worst_row_swap_indexer_{tag}", worst)
    print(f"indexer {tag}: overlap mean {mean:.6f} (>= {lim['mean']}) worst row {worst:.4f} (>= {lim['row']})")
    fails = []
    if not mean >= lim["mean"]:
        fails.append(f"indexer {tag}: overlap {mean:.5f} < {lim['mean']}")
    if not worst >= lim["row"]:
        fails.append(f"indexer {tag}: worst row overlap {worst:.4f} < {lim['row']}")
    return fails


def _pooled_keys(state, p0, p1, width):
    """Rows [p0, p1) of a state's index_key, or None."""
    if state is None or "index_key" not in state:
        return None
    k = state["index_key"].float()
    if k.numel() % width:
        return None
    k = k.reshape(-1, width)
    return k[p0:p1] if k.shape[0] >= p1 else None


ATTN_GOLD = {"rel": 0.012, "ratio": (0.99, 1.01), "row": 0.03}  # attn_out vs golden (device inputs: upstream error)
ATTN_SAME = {"rel": 0.006, "ratio": (0.994, 1.004), "row": 0.02}  # vs the fp32 CPU attention of the same inputs
C0_ATTN_SAME = ATTN_SAME  # chunk 0 (start 0, mostly -1 ids), same inputs
LAT_GOLD = {"rel": 0.006, "ratio": (0.997, 1.003), "row": 0.01}  # chunk's latent rows vs golden (device attn_norm in)
LAT_SAME = {"rel": 0.0035, "ratio": (0.997, 1.003), "row": 0.005}  # vs the CPU latent of the same attn_norm
LAT_PREFIX_MAX_REL_L2 = 1e-3  # prefix rows must stay the loaded golden prefix
ATTN_SHARE_MAX_FLIPS = 48  # attention share: rows whose top-8 differs from the CPU block with device hc..indexer
ATTN_SHARE_MAX_REL_L2 = 0.0012  # same-routing rows
ATTN_SHARE_RATIO = (0.996, 1.004)
RES_GOLD = {"rel": 0.006, "ratio": (0.985, 1.015), "row": 0.04, "stream": 0.008}  # h_mid vs golden (device inputs)
RES_SAME = {"rel": 0.0035, "ratio": (0.997, 1.003), "row": 0.008}  # vs the fp32 CPU residual of the same inputs
COMB_COEF = (0.998, 1.002)  # <out - post term, comb term> / ||comb term||^2, terms from the step's own inputs
COMB_MAX_REL = 0.0035  # ||out - post term - comb term|| / ||comb term||
POST_COEF = (0.99, 1.01)  # <out - comb term, post term> / ||post term||^2
POST_MAX_REL = 0.08  # bf16 output rounding alone gives 0.034
RES_SHARE_MAX_FLIPS = 64  # residual share: rows whose top-8 differs from the CPU block with device hc..attention
RES_SHARE_MAX_REL_L2 = 0.0015  # same-routing rows
RES_SHARE_RATIO = (0.997, 1.003)
FHC_SAME = {  # ffn_hc vs the fp32 CPU ffn_hc of the same h_mid: the component test's limits
    "rel": {"pre": 0.01, "post": 0.01, "comb": 0.01},
    "abs": {"pre": 0.02, "post": 5e-3, "comb": 0.02},
    "col": 0.07,
    "coef": (0.995, 1.005),
}
FHC_GOLD = FHC_SAME  # vs golden (device h_mid in); the CPU ffn_hc of that h_mid is at 0.0009 / 0.0021 / 0.0012
FHC_SHARE_MAX_FLIPS = 40  # ffn_hc share: rows whose top-8 differs from the CPU block with device hc..attn_residual
FHC_SHARE_MAX_REL_L2 = 0.0035  # same-routing rows
FHC_SHARE_RATIO = (0.99, 1.015)
FHC_SHARE_FLIPPED_RATIO = (0.9, 1.1)  # flipped rows (a zeroed row flips and hides from the same-routing check)
FC_SAME = {"rel": 0.008, "ratio": (0.99, 1.01), "row": 0.008, "coef": (0.996, 1.004)}  # vs the fp32 CPU collapse of
# the same (h_mid, ffn_hc): the component test's limits
FC_GOLD = {"rel": 0.008, "ratio": (0.99, 1.01), "row": 0.02, "coef": (0.995, 1.005)}  # vs golden ffn_in (device
# inputs; the CPU collapse of them scores 0.0044 / [0.9965, 1.0012] / row 0.0095 / coef 0.9990)
FC_SHARE_MAX_FLIPS = 64  # collapse share: rows whose top-8 differs from the CPU block with device hc..ffn_hc
FC_SHARE_MAX_REL_L2 = 0.0012  # same-routing rows
FC_SHARE_RATIO = (0.996, 1.004)
FC_SHARE_FLIPPED_RATIO = (0.95, 1.05)
FN_SAME = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.015, "coef": (0.996, 1.004)}  # vs the fp32 CPU norm of the
# same ffn_in: the component test's limits
FN_GOLD = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.02, "coef": (0.995, 1.005)}  # vs golden (device ffn_in in)
FN_SHARE_MAX_FLIPS = 64  # norm share: rows whose top-8 differs from the CPU block with device hc..ffn_collapse
FN_SHARE_MAX_REL_L2 = 0.0015  # same-routing rows
FN_SHARE_RATIO = (0.996, 1.004)
FN_SHARE_FLIPPED_RATIO = (0.92, 1.08)  # bf16 norm rounding alone gives [0.960, 1.009]; a zeroed row 0.88
TOP_K = 8
ROUTE_SCALE = 2.5  # routed_scaling_factor: every routing row sums to 2.5
RT_SAME = {"mean": 0.998, "row": 0.75, "mrel": 0.0025, "coef": (0.9995, 1.0005), "sum": 0.015}  # vs the fp32 CPU
# router of the same ffn_norm (bf16 output rounding alone: 1.0 / 1.0 / 0.00158 / 1.00008 / [2.4939, 2.5068])
RT_GOLD = {"mean": 0.985, "row": 0.5, "mrel": 0.005, "coef": (0.998, 1.002), "sum": 0.015}  # vs golden (device
# ffn_norm in): the component test's limits
RT_SHARE_MAX_FLIPS = 32  # router share: rows whose top-8 differs from the CPU block with device hc..ffn_norm
RT_SHARE_MAX_REL_L2 = 0.0015  # same-routing rows (bf16 weights alone 0.00086)
RT_SHARE_RATIO = (0.997, 1.003)  # bf16 weights alone [0.9980, 1.0022]
RT_SHARE_FLIPPED_RATIO = (0.95, 1.05)  # a zeroed, copied or rolled row gives 0.920..0.933
EX_SAME = {"rel": 0.012, "ratio": (0.985, 1.015), "row": 0.025, "coef": (0.997, 1.003), "block": (0.995, 1.005)}
# experts vs the fp32 CPU experts of the same (ffn_norm, router): the component test's limits
EX_GOLD = {"rel": 0.012, "ratio": (0.985, 1.015), "row": 0.03, "coef": (0.997, 1.003), "block": (0.995, 1.005)}
# experts vs golden on the rows whose top-8 equals the golden's (device ffn_norm and router in)
EX_GOLD_FLIPPED_RATIO = (0.8, 1.25)  # rows with another top-8 (a whole expert changes)
BLOCK_ROWS = 128
EX_SHARE_MAX_REL_L2 = 0.007  # experts share: block out vs the CPU block with device hc..router (device-like 0.0045)
EX_SHARE_RATIO = (0.994, 1.006)  # device [0.9982, 1.0034]; x1.01 experts gives 1.0079
SE_SAME = {"rel": 0.008, "ratio": (0.993, 1.007), "row": 0.015, "coef": (0.997, 1.003), "block": (0.996, 1.004)}
# shared_expert vs the fp32 CPU shared expert of the same ffn_norm: the component test's limits
SE_GOLD = {"rel": 0.012, "ratio": (0.985, 1.015), "row": 0.03, "coef": (0.997, 1.003), "block": (0.995, 1.005)}
# shared_expert vs golden (device ffn_norm in: carries the upstream error)
SE_SHARE_MAX_REL_L2 = 0.0015  # shared share: block out vs the CPU block with device hc..experts (device-like 0.0007)
SE_SHARE_RATIO = (0.998, 1.002)  # x1.005 shared gives max 1.0030, last row zeroed min 0.962


def _res_checks(tag, got, want, lim, inputs=None):
    """h_mid vs want; with ``inputs`` (in, attn_hc, attn_out) also each term on its own."""
    fails = _out_checks("attn_residual", tag, got, want, lim)
    if any("elements" in m or "non-finite" in m for m in fails):
        return fails
    got, w = got.float().reshape(want.shape), want.float()
    h = w.shape[-1]
    if "stream" in lim:
        gs, ws = got.view(-1, N, h), w.view(-1, N, h)
        srel = [((gs[:, n] - ws[:, n]).norm() / ws[:, n].norm()).item() for n in range(N)]
        metrics.record(f"worst_stream_rel_l2_swap_attn_residual_{tag}", max(srel))
        print(f"attn_residual {tag}: per-stream rel L2 {[round(v, 5) for v in srel]} (<= {lim['stream']})")
        if not max(srel) <= lim["stream"]:
            fails.append(f"attn_residual {tag} per-stream rel L2 {max(srel):.4f} > {lim['stream']}")
    if inputs is not None:
        x, hc, y = (t.float() for t in inputs)
        post = hc[:, N : 2 * N]
        comb = hc[:, 2 * N :].reshape(-1, N, N)
        pterm = (post.unsqueeze(-1) * y.reshape(-1, h).unsqueeze(-2)).reshape(-1, h)
        cterm = torch.matmul(comb.transpose(-1, -2), x.reshape(-1, N, h)).reshape(-1, h)
        for name, term, other, (clo, chi), max_rel in (
            ("post_term", pterm, cterm, POST_COEF, POST_MAX_REL),
            ("comb_term", cterm, pterm, COMB_COEF, COMB_MAX_REL),
        ):
            d = got - other
            coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
            trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
            metrics.record(f"{name}_coef_swap_attn_residual_{tag}", coef)
            metrics.record(f"{name}_rel_l2_swap_attn_residual_{tag}", trel)
            print(f"attn_residual {tag}: {name} coef={coef:.5f} (in [{clo}, {chi}]) rel={trel:.5f} (<= {max_rel})")
            if not (clo <= coef <= chi):
                fails.append(f"attn_residual {tag} {name} coefficient {coef:.5f} outside [{clo}, {chi}]")
            if not trel <= max_rel:
                fails.append(f"attn_residual {tag} {name} rel L2 {trel:.4f} > {max_rel}")
    return fails


def _fhc_checks(tag, got, want, lim):
    """ffn_hc [S, 24] vs want: per part rel L2, max abs and coefficient, worst column, comb column sums, ranges."""
    if got.numel() != want.numel():
        return [f"ffn_hc {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"ffn_hc {tag}: non-finite output"]
    fails = []
    for name, sl in PARTS.items():
        a, b = got[:, sl], w[:, sl]
        rel = ((a - b).norm() / b.norm()).item()
        mx = (a - b).abs().max().item()
        coef = ((a * b).sum() / (b * b).sum()).item()
        metrics.record(f"rel_l2_swap_ffn_hc_{name}_{tag}", rel)
        metrics.record(f"max_abs_swap_ffn_hc_{name}_{tag}", mx)
        metrics.record(f"coef_swap_ffn_hc_{name}_{tag}", coef)
        print(
            f"ffn_hc {tag} {name}: rel_l2={rel:.5f} (<= {lim['rel'][name]}) max_abs={mx:.2e} (<= {lim['abs'][name]})"
            f" coef={coef:.5f} (in {lim['coef']})"
        )
        if not rel <= lim["rel"][name]:
            fails.append(f"ffn_hc {tag} {name} rel L2 {rel:.4f} > {lim['rel'][name]}")
        if not mx <= lim["abs"][name]:
            fails.append(f"ffn_hc {tag} {name} max abs {mx:.3g} > {lim['abs'][name]}")
        if not lim["coef"][0] <= coef <= lim["coef"][1]:
            fails.append(f"ffn_hc {tag} {name} coefficient {coef:.5f} outside {lim['coef']}")
    col_rel = (got - w).norm(dim=0) / w.norm(dim=0)
    worst = col_rel.max().item()
    metrics.record(f"worst_col_rel_l2_swap_ffn_hc_{tag}", worst)
    print(f"ffn_hc {tag}: worst column rel L2 {worst:.4f} at column {col_rel.argmax().item()} (<= {lim['col']})")
    if not worst <= lim["col"]:
        fails.append(f"ffn_hc {tag} column {col_rel.argmax().item()} rel L2 {worst:.4f} > {lim['col']}")
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record(f"comb_col_sum_err_swap_ffn_hc_{tag}", col_err)
    if not col_err <= COL_SUM_TOL:
        fails.append(f"ffn_hc {tag} comb column sums off 1 by {col_err:.4f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"ffn_hc {tag} pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"ffn_hc {tag} post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append(f"ffn_hc {tag} negative comb entries")
    return fails


def _coef_checks(step, tag, got, want, lim):
    """[S, H] vs want: rel L2, per-token norm ratio, worst row and the coefficient <got, want> / <want, want>."""
    fails = _out_checks(step, tag, got, want, lim)
    if any("elements" in m or "non-finite" in m for m in fails):
        return fails
    a, b = got.float().reshape(want.shape), want.float()
    coef = ((a * b).sum() / (b * b).sum().clamp_min(1e-30)).item()
    metrics.record(f"coef_swap_{step}_{tag}", coef)
    print(f"{step} {tag}: coef={coef:.5f} (in {lim['coef']})")
    if not lim["coef"][0] <= coef <= lim["coef"][1]:
        fails.append(f"{step} {tag} coefficient {coef:.5f} outside {lim['coef']}")
    return fails


def _fc_checks(tag, got, want, lim):
    return _coef_checks("ffn_collapse", tag, got, want, lim)


def _fn_checks(tag, got, want, lim):
    return _coef_checks("ffn_norm", tag, got, want, lim)


def _router_checks(tag, got, want, lim):
    """Dense routing [S, E] vs want: 8 nonzeros per row, non-negative, selection overlap (mean and worst row), weight
    rel L2 and coefficient on rows with the same selection, row sums (as the component test)."""
    if got.numel() != want.numel():
        return [f"router {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"router {tag}: non-finite output"]
    gs, ws = got != 0, w != 0
    nnz = gs.sum(-1)
    row_ov = (gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)
    mean, worst = row_ov.mean().item(), row_ov.min().item()
    m = (gs == ws).all(-1)
    if m.any():
        mrel = ((got[m] - w[m]).norm() / w[m].norm()).item()
        mcoef = ((got[m] * w[m]).sum() / (w[m] * w[m]).sum()).item()
    else:
        mrel, mcoef = float("inf"), float("nan")
    rsum = got.sum(-1)
    rmin, rmax = rsum.min().item(), rsum.max().item()
    metrics.record(f"selection_overlap_swap_router_{tag}", mean)
    metrics.record(f"worst_row_overlap_swap_router_{tag}", worst)
    metrics.record(f"matched_rel_l2_swap_router_{tag}", mrel)
    metrics.record(f"matched_coef_swap_router_{tag}", mcoef)
    print(
        f"router {tag}: nnz/row {nnz.min().item()}..{nnz.max().item()} overlap={mean:.5f} (>= {lim['mean']}) worst"
        f" row={worst:.3f} (>= {lim['row']}) matched rows {int(m.sum())}/{m.numel()} mrel={mrel:.5f} (<= {lim['mrel']})"
        f" coef={mcoef:.5f} (in {lim['coef']}) row sums [{rmin:.4f}, {rmax:.4f}] ({ROUTE_SCALE} +- {lim['sum']})"
    )
    fails = []
    if not (nnz == TOP_K).all():
        fails.append(f"router {tag}: nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want {TOP_K}")
    if not (got >= 0).all():
        fails.append(f"router {tag}: negative routing weight")
    if not mean >= lim["mean"]:
        fails.append(f"router {tag}: selection overlap {mean:.5f} < {lim['mean']}")
    if not worst >= lim["row"]:
        fails.append(f"router {tag}: worst row overlap {worst:.3f} < {lim['row']}")
    if not mrel <= lim["mrel"]:
        fails.append(f"router {tag}: matched-row weight rel L2 {mrel:.5f} > {lim['mrel']}")
    if not lim["coef"][0] <= mcoef <= lim["coef"][1]:
        fails.append(f"router {tag}: matched-row coefficient {mcoef:.5f} outside {lim['coef']}")
    if not (ROUTE_SCALE - lim["sum"] <= rmin and rmax <= ROUTE_SCALE + lim["sum"]):
        fails.append(f"router {tag}: row sums [{rmin:.4f}, {rmax:.4f}] outside {ROUTE_SCALE} +- {lim['sum']}")
    return fails


def _experts_checks(tag, got, want, lim, rows=None, step="experts"):
    """experts_out (or shared_out, step="shared_expert") [S, H] vs want (on ``rows`` only if given): rel L2,
    per-token norm ratio, worst row, coefficient and the coefficient of every 128-row block (as the component test)."""
    if got.numel() != want.numel():
        return [f"{step} {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"{step} {tag}: non-finite output"]
    nb = w.shape[0] // BLOCK_ROWS
    gb, wb = got[: nb * BLOCK_ROWS].reshape(nb, -1), w[: nb * BLOCK_ROWS].reshape(nb, -1)
    keep = torch.ones(w.shape[0], dtype=torch.bool) if rows is None else rows
    if not keep.any():
        return [f"{step} {tag}: no rows to compare (every token has another top-8)"]
    if rows is not None:  # a block's coefficient over its kept rows only
        kb = keep[: nb * BLOCK_ROWS].reshape(nb, BLOCK_ROWS, 1).expand(nb, BLOCK_ROWS, w.shape[1]).reshape(nb, -1)
        gb, wb = gb * kb, wb * kb
    bcoef = (gb * wb).sum(-1) / (wb * wb).sum(-1).clamp_min(1e-30)
    bmin, bmax = bcoef.min().item(), bcoef.max().item()
    got, w = got[keep], w[keep]
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    coef = ((got * w).sum() / (w * w).sum()).item()
    for k, v in (("rel_l2", rel), ("norm_ratio_min", lo), ("norm_ratio_max", hi), ("row_rel_l2_max", row)):
        metrics.record(f"{k}_swap_{step}_{tag}", v)
    metrics.record(f"coef_swap_{step}_{tag}", coef)
    metrics.record(f"block_coef_min_swap_{step}_{tag}", bmin)
    metrics.record(f"block_coef_max_swap_{step}_{tag}", bmax)
    print(
        f"{step} {tag}: rows {int(keep.sum())}/{keep.numel()} rel_l2={rel:.5f} (<= {lim['rel']}) norm ratio"
        f" [{lo:.4f}, {hi:.4f}] (in {lim['ratio']}) worst row {row:.5f} (<= {lim['row']}) coef={coef:.5f}"
        f" (in {lim['coef']}) block coef [{bmin:.5f}, {bmax:.5f}] (in {lim['block']})"
    )
    fails = []
    if not rel <= lim["rel"]:
        fails.append(f"{step} {tag} rel L2 {rel:.4f} > {lim['rel']}")
    if not (lim["ratio"][0] <= lo and hi <= lim["ratio"][1]):
        fails.append(f"{step} {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {lim['ratio']}")
    if not row <= lim["row"]:
        fails.append(f"{step} {tag} worst row rel L2 {row:.4f} > {lim['row']}")
    if not lim["coef"][0] <= coef <= lim["coef"][1]:
        fails.append(f"{step} {tag} coefficient {coef:.5f} outside {lim['coef']}")
    if not (lim["block"][0] <= bmin and bmax <= lim["block"][1]):
        fails.append(
            f"{step} {tag} {BLOCK_ROWS}-row block coefficients [{bmin:.5f}, {bmax:.5f}] outside {lim['block']}"
        )
    return fails


def _latent_checks(tag, state, want, start, s, prefix=None):
    """The chunk's rows [start, start + s) of a state's kv_latent vs want [s, 512]; prefix rows vs ``prefix``."""
    if state is None or "kv_latent" not in state:
        return [f"attention {tag}: no kv_latent (dctx.extra['state_out']['kv_latent'] [>= start + S, 512])"]
    r = want.shape[-1]
    k = state["kv_latent"].float()
    if k.numel() % r or k.numel() // r < start + s:
        return [f"attention {tag}: kv_latent shape {tuple(k.shape)}, want [>= {start + s}, {r}]"]
    k = k.reshape(-1, r)
    fails = _out_checks("kv_latent", tag, k[start : start + s], want, LAT_GOLD if tag == "vs_golden" else LAT_SAME)
    if prefix is not None and start:
        prel = ((k[:start] - prefix).norm() / prefix.norm()).item()
        metrics.record("rel_l2_swap_kv_latent_prefix", prel)
        print(f"kv_latent prefix rows [0, {start}): rel_l2={prel:.2e} (<= {LAT_PREFIX_MAX_REL_L2})")
        if not prel <= LAT_PREFIX_MAX_REL_L2:
            fails.append(f"kv_latent prefix rows changed (rel L2 {prel:.2e} vs the loaded golden prefix)")
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
    # the indexer and the attention both write dctx.extra["state_out"]: keep the indexer's pooled keys of this run
    captured = {}

    def main_idx(ctx, x, q):
        y = muts["indexer"](ctx, dctx, x, q)
        captured["indexer"] = dctx.extra.get("state_out")
        return y

    overrides["indexer"] = main_idx
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
    # the caches the swapped steps wrote (the shares below run them again and overwrite state_out)
    mode = impl_mode()
    if mode == "device":
        idx_state, lat_state = captured.get("indexer"), dctx.extra.get("state_out")
    elif mode == "reference":
        idx_state = lat_state = ref.state_tensors(rctx.state, layer, g.seq)
    else:
        idx_state = lat_state = None
    for n, t in seen.items():
        if n not in ("in", "out") and n in gl:
            compare(f"pcc_swap_{n}", t, gl[n], default_mode(gl[n]), 0.0)  # the trail, for diagnosis; not gated
    thr = threshold(S, "block") if THRESHOLD is None else THRESHOLD
    _, ok = compare("pcc_swap_out", seen["out"], gl["out"], "pcc", thr)

    failures = [] if ok else [f"pcc_swap_out below {thr}"]
    out, w = seen["out"].float().reshape(gl["out"].shape), gl["out"].float()
    out_rel = ((out - w).norm() / w.norm()).item()
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"block out vs golden: rel_l2={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not torch.isfinite(out).all():
        failures.append("block out non-finite")
    if not out_rel <= OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2}")
    failures += _row_checks(
        "golden", out, w, _flipped(seen["router"], gl["router"]), OUT_ROW_RATIO_SAME, OUT_ROW_RATIO_FLIPPED
    )

    # Every device step's share: the all-CPU block of the same golden `in`.
    cpu = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: cpu.__setitem__(n, t),
    )
    flips = _flipped(seen["router"], cpu["router"])
    n_flips = int(flips.sum())
    if n_flips > CPU_MAX_FLIPS:
        failures.append(f"{n_flips} tokens route to another top-8 than the CPU block (> {CPU_MAX_FLIPS})")
    cpu_out = cpu["out"].float().reshape(w.shape)
    failures += _row_checks("cpu", out, cpu_out, flips, CPU_ROW_RATIO, None, CPU_MAX_REL_L2)

    failures += _hc_checks(seen["attn_hc"], gl["attn_hc"])

    # attn_collapse vs golden (its attn_hc input is the device one), and vs the CPU collapse of the same inputs.
    hc_dev = seen["attn_hc"].float()
    failures += _out_checks("attn_collapse", "vs_golden", seen["attn_in"], gl["attn_in"], COL_GOLD)
    cpu_in = ref.component(layer, "attn_collapse")(rctx, gl["in"].float(), hc_dev)
    failures += _out_checks("attn_collapse", "vs_cpu_same_input", seen["attn_in"], cpu_in, COL_SAME)

    dev_norm = lambda ctx, x: _f32(muts["attn_norm"](ctx, dctx, x))
    dev_qa = lambda ctx, x: _f32(muts["q_a"](ctx, dctx, x))
    dev_idx = lambda ctx, x, q: muts["indexer"](ctx, dctx, x, q)
    in_dev = seen["attn_in"].float().reshape(gl["attn_in"].shape)
    norm_dev = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
    qr_dev = seen["q_resid"].float().reshape(gl["q_resid"].shape)
    start, s = c * g.chunk, gl["topk"].shape[0]
    want_tk = gl["topk"].long()
    got_tk = seen["topk"]
    if got_tk.is_floating_point() or got_tk.numel() != want_tk.numel():
        failures.append(
            f"indexer: output {tuple(got_tk.shape)} {got_tk.dtype}, want {tuple(want_tk.shape)} integer ids"
        )
        assert not failures, "; ".join(failures)
    got_tk = got_tk.reshape(want_tk.shape).long()
    tk_dev = got_tk.to(torch.int32)

    # The attention share alone: the CPU block with the device attn_hc, attn_in, attn_norm, q_resid and topk and the
    # CPU attention. Its the base of the earlier steps' shares below: they re-run the device steps they do not
    # isolate, and a re-run device attention adds its own noise (rel ~0.004 on any input change) to every share.
    ashare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: ashare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
        },
    )
    # The residual-share block: every device output up to attn_out fixed, CPU attn_residual. Block out minus it is the
    # residual's share; it minus the attention-share block is the attention's share (as swap 06).
    ao_dev = seen["attn_out"].float().reshape(gl["attn_out"].shape)
    rshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: rshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
        },
    )
    # The ffn_hc-share block: every device output up to h_mid fixed, CPU ffn_hc. Block out minus it is the ffn_hc's
    # share; it minus the residual-share block is the residual's share (as swap 07).
    hm_dev = seen["h_mid"].float().reshape(gl["h_mid"].shape)
    hshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: hshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
        },
    )
    # The collapse-share block: every device output up to ffn_hc fixed, CPU ffn_collapse. Block out minus it is the
    # collapse's share; it minus the ffn_hc-share block is the ffn_hc's share (as swap 08).
    fhc_dev = seen["ffn_hc"].float().reshape(gl["ffn_hc"].shape)
    fshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: fshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
        },
    )
    # The norm-share block: every device output up to ffn_in fixed, CPU ffn_norm. Block out minus it is the norm's
    # share; it minus the collapse-share block is the collapse's share (as swap 09).
    fin_dev = seen["ffn_in"].float().reshape(gl["ffn_in"].shape)
    nfshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: nfshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
        },
    )
    # The router-share block: every device output up to ffn_norm fixed, CPU router. Block out minus it is the router's
    # share; it minus the norm-share block is the norm's share (as swap 10).
    fn_dev = seen["ffn_norm"].float().reshape(gl["ffn_norm"].shape)
    rtshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: rtshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
        },
    )
    # The experts-share block: every device output up to router fixed, CPU experts. Block out minus it is the experts'
    # share (the routing is the same by construction); it minus the router-share block is the router's share (as
    # swap 11).
    rt_dev = seen["router"].float().reshape(gl["router"].shape)
    exshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: exshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
            "router": lambda ctx, x: rt_dev,
        },
    )
    # The shared-share block: every device output up to experts fixed, CPU shared_expert. Block out minus it is the
    # shared expert's share (same routing by construction); it minus the experts-share block is the experts' share (as
    # swap 12).
    ex_dev = seen["experts_out"].float().reshape(gl["experts_out"].shape)
    seshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: seshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
            "indexer": lambda ctx, x, q: tk_dev,
            "attention": lambda ctx, x, q, t: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
            "router": lambda ctx, x: rt_dev,
            "experts": lambda ctx, x, r: ex_dev,
        },
    )
    se_flips = _flipped(seen["router"], seshare["router"])
    ex_flips = _flipped(seshare["router"], exshare["router"])
    if se_flips.any() or ex_flips.any():
        failures.append(
            f"share blocks route {int(se_flips.sum())} / {int(ex_flips.sum())} tokens differently (harness bug)"
        )
    seshare_out = seshare["out"].float().reshape(w.shape)
    exshare_out = exshare["out"].float().reshape(w.shape)
    failures += _row_checks("se_share", out, seshare_out, se_flips, SE_SHARE_RATIO, None, SE_SHARE_MAX_REL_L2)
    failures += _row_checks("ex_share", seshare_out, exshare_out, ex_flips, EX_SHARE_RATIO, None, EX_SHARE_MAX_REL_L2)

    # shared_expert vs the fp32 CPU shared expert of the same ffn_norm, and vs golden (device ffn_norm in).
    se = "shared_expert"
    failures += _experts_checks("vs_cpu_same_input", seen["shared_out"], seshare["shared_out"], SE_SAME, step=se)
    failures += _experts_checks("vs_golden", seen["shared_out"], gl["shared_out"], SE_GOLD, step=se)
    _experts_checks("cpu_vs_golden_info", seshare["shared_out"], gl["shared_out"], SE_GOLD, step=se)  # not asserted

    # experts vs the fp32 CPU experts of the same (ffn_norm, router), and vs golden on the rows whose top-8 is the
    # golden's (the router's flips each move a whole expert: those rows get only a loose norm-ratio check).
    failures += _experts_checks("vs_cpu_same_input", seen["experts_out"], exshare["experts_out"], EX_SAME)
    g_flips = _flipped(seen["router"], gl["router"])
    failures += _experts_checks("vs_golden", seen["experts_out"], gl["experts_out"], EX_GOLD, ~g_flips)
    _experts_checks("cpu_vs_golden_info", exshare["experts_out"], gl["experts_out"], EX_GOLD, ~g_flips)  # not asserted
    if g_flips.any():
        eg = seen["experts_out"].float().reshape(gl["experts_out"].shape)[g_flips]
        ew = gl["experts_out"].float()[g_flips]
        fr = eg.norm(dim=-1) / ew.norm(dim=-1).clamp_min(1e-12)
        lo, hi = fr.min().item(), fr.max().item()
        print(
            f"experts vs_golden: {int(g_flips.sum())} flipped rows ratio [{lo:.4f}, {hi:.4f}] (in {EX_GOLD_FLIPPED_RATIO})"
        )
        if not (EX_GOLD_FLIPPED_RATIO[0] <= lo and hi <= EX_GOLD_FLIPPED_RATIO[1]):
            failures.append(f"experts vs_golden flipped-row ratio [{lo:.4f}, {hi:.4f}] outside {EX_GOLD_FLIPPED_RATIO}")

    rt_flips = _flipped(exshare["router"], rtshare["router"])
    n_rt = int(rt_flips.sum())
    if n_rt > RT_SHARE_MAX_FLIPS:
        failures.append(f"{n_rt} tokens route to another top-8 than the router-share block (> {RT_SHARE_MAX_FLIPS})")
    rtshare_out = rtshare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "rt_share", exshare_out, rtshare_out, rt_flips, RT_SHARE_RATIO, RT_SHARE_FLIPPED_RATIO, RT_SHARE_MAX_REL_L2
    )

    # router vs golden (its ffn_norm input is the device one), and vs the fp32 CPU router of the same ffn_norm.
    failures += _router_checks("vs_golden", seen["router"], gl["router"], RT_GOLD)
    failures += _router_checks("vs_cpu_same_input", seen["router"], rtshare["router"], RT_SAME)
    _router_checks("cpu_vs_golden_info", rtshare["router"], gl["router"], RT_GOLD)  # upstream share, not asserted

    fn_flips = _flipped(rtshare["router"], nfshare["router"])
    n_fn = int(fn_flips.sum())
    if n_fn > FN_SHARE_MAX_FLIPS:
        failures.append(f"{n_fn} tokens route to another top-8 than the norm-share block (> {FN_SHARE_MAX_FLIPS})")
    nfshare_out = nfshare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "fn_share", rtshare_out, nfshare_out, fn_flips, FN_SHARE_RATIO, FN_SHARE_FLIPPED_RATIO, FN_SHARE_MAX_REL_L2
    )

    # ffn_norm vs golden (its ffn_in input is the device one), and vs the fp32 CPU norm of the same ffn_in.
    failures += _fn_checks("vs_golden", seen["ffn_norm"], gl["ffn_norm"], FN_GOLD)
    cpu_fn = ref.component(layer, "ffn_norm")(reference_ctx(ref, layer, g, c), fin_dev)
    failures += _fn_checks("vs_cpu_same_input", seen["ffn_norm"], cpu_fn, FN_SAME)
    _fn_checks("cpu_vs_golden_info", cpu_fn, gl["ffn_norm"], FN_GOLD)  # upstream share, not asserted

    f_flips = _flipped(nfshare["router"], fshare["router"])
    n_f = int(f_flips.sum())
    if n_f > FC_SHARE_MAX_FLIPS:
        failures.append(f"{n_f} tokens route to another top-8 than the collapse-share block (> {FC_SHARE_MAX_FLIPS})")
    fshare_out = fshare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "fc_share", nfshare_out, fshare_out, f_flips, FC_SHARE_RATIO, FC_SHARE_FLIPPED_RATIO, FC_SHARE_MAX_REL_L2
    )

    # ffn_collapse vs golden (its h_mid and ffn_hc inputs are the device ones), and vs the fp32 CPU collapse of the
    # same inputs.
    failures += _fc_checks("vs_golden", seen["ffn_in"], gl["ffn_in"], FC_GOLD)
    cpu_fc = ref.component(layer, "ffn_collapse")(reference_ctx(ref, layer, g, c), hm_dev, fhc_dev)
    failures += _fc_checks("vs_cpu_same_input", seen["ffn_in"], cpu_fc, FC_SAME)
    _fc_checks("cpu_vs_golden_info", cpu_fc, gl["ffn_in"], FC_GOLD)  # upstream share, not asserted

    h_flips = _flipped(fshare["router"], hshare["router"])
    n_h = int(h_flips.sum())
    if n_h > FHC_SHARE_MAX_FLIPS:
        failures.append(f"{n_h} tokens route to another top-8 than the ffn_hc-share block (> {FHC_SHARE_MAX_FLIPS})")
    hshare_out = hshare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "fhc_share", fshare_out, hshare_out, h_flips, FHC_SHARE_RATIO, FHC_SHARE_FLIPPED_RATIO, FHC_SHARE_MAX_REL_L2
    )

    r_flips = _flipped(hshare["router"], rshare["router"])
    n_r = int(r_flips.sum())
    if n_r > RES_SHARE_MAX_FLIPS:
        failures.append(f"{n_r} tokens route to another top-8 than the residual-share block (> {RES_SHARE_MAX_FLIPS})")
    rshare_out = rshare["out"].float().reshape(w.shape)
    failures += _row_checks("res_share", hshare_out, rshare_out, r_flips, RES_SHARE_RATIO, None, RES_SHARE_MAX_REL_L2)

    # ffn_hc vs golden (its h_mid input is the device one), and vs the fp32 CPU ffn_hc of the same h_mid.
    failures += _fhc_checks("vs_golden", seen["ffn_hc"], gl["ffn_hc"], FHC_GOLD)
    cpu_fhc = ref.component(layer, "ffn_hc")(reference_ctx(ref, layer, g, c), hm_dev)
    failures += _fhc_checks("vs_cpu_same_input", seen["ffn_hc"], cpu_fhc, FHC_SAME)
    _fhc_checks("cpu_vs_golden_info", cpu_fhc, gl["ffn_hc"], FHC_GOLD)  # upstream share, not asserted

    a_flips = _flipped(rshare["router"], ashare["router"])
    n_a = int(a_flips.sum())
    if n_a > ATTN_SHARE_MAX_FLIPS:
        failures.append(
            f"{n_a} tokens route to another top-8 than the attention-share block (> {ATTN_SHARE_MAX_FLIPS})"
        )
    ashare_out = ashare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "attn_share", rshare_out, ashare_out, a_flips, ATTN_SHARE_RATIO, None, ATTN_SHARE_MAX_REL_L2
    )

    # attn_residual vs golden h_mid (its attn_hc and attn_out inputs are the device ones), and vs the fp32 CPU
    # residual of the same inputs, with each term on its own.
    res_in = [gl["in"].float(), hc_dev, ao_dev]
    failures += _res_checks("vs_golden", seen["h_mid"], gl["h_mid"], RES_GOLD)
    cpu_res = ref.component(layer, "attn_residual")(reference_ctx(ref, layer, g, c), *res_in)
    failures += _res_checks("vs_cpu_same_input", seen["h_mid"], cpu_res, RES_SAME, res_in)
    _res_checks("cpu_vs_golden_info", cpu_res, gl["h_mid"], RES_GOLD)  # upstream share, not asserted

    # attention vs golden (its inputs are the device attn_norm, q_resid and topk), and vs the fp32 CPU attention of
    # the same inputs (fresh state from the golden prefix).
    failures += _out_checks("attention", "vs_golden", seen["attn_out"], gl["attn_out"], ATTN_GOLD)
    actx = reference_ctx(ref, layer, g, c)
    cpu_attn = ref.component(layer, "attention")(actx, norm_dev, qr_dev, tk_dev)
    failures += _out_checks("attention", "vs_cpu_same_input", seen["attn_out"], cpu_attn, ATTN_SAME)
    _out_checks("attention", "cpu_vs_golden_info", cpu_attn, gl["attn_out"], ATTN_GOLD)  # upstream share, not asserted

    # the chunk's latent rows vs the golden state and vs the CPU latent of the same attn_norm; prefix unchanged.
    want_state = g.state(layer)
    want_lat = want_state["kv_latent"].float()
    failures += _latent_checks("vs_golden", lat_state, want_lat[start : start + s], start, s, want_lat[:start])
    cpu_lat = ref.state_tensors(actx.state, layer, g.seq)["kv_latent"].float()[start : start + s]
    failures += _latent_checks("vs_cpu_same_input", lat_state, cpu_lat, start, s)

    # The earlier steps' shares, each vs the attention-share block (device steps through the indexer, CPU attention).
    base_router = ashare["router"]
    # The attn_collapse share alone: the CPU block with the device attn_hc, the CPU collapse, the device norm, q_a
    # and indexer.
    share = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: share.__setitem__(n, t),
        overrides={"attn_hc": lambda ctx, x: hc_dev, "attn_norm": dev_norm, "q_a": dev_qa, "indexer": dev_idx},
    )
    s_flips = _flipped(base_router, share["router"])
    n_s = int(s_flips.sum())
    if n_s > SHARE_MAX_FLIPS:
        failures.append(f"{n_s} tokens route to another top-8 than the collapse-share block (> {SHARE_MAX_FLIPS})")
    share_out = share["out"].float().reshape(w.shape)
    failures += _row_checks("collapse_share", ashare_out, share_out, s_flips, SHARE_RATIO, None, SHARE_MAX_REL_L2)

    # attn_norm vs golden (its attn_in input is the device one), and vs the CPU norm of the same input.
    failures += _out_checks("attn_norm", "vs_golden", seen["attn_norm"], gl["attn_norm"], NORM_GOLD)
    cpu_norm = ref.component(layer, "attn_norm")(rctx, in_dev)
    failures += _out_checks("attn_norm", "vs_cpu_same_input", seen["attn_norm"], cpu_norm, NORM_SAME)

    # The attn_norm share alone: the CPU block with the device attn_hc and attn_in, the CPU norm, the device q_a and
    # indexer.
    nshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: nshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "q_a": dev_qa,
            "indexer": dev_idx,
        },
    )
    n_flips_n = _flipped(base_router, nshare["router"])
    n_n = int(n_flips_n.sum())
    if n_n > NORM_SHARE_MAX_FLIPS:
        failures.append(f"{n_n} tokens route to another top-8 than the norm-share block (> {NORM_SHARE_MAX_FLIPS})")
    nshare_out = nshare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "norm_share", ashare_out, nshare_out, n_flips_n, NORM_SHARE_RATIO, None, NORM_SHARE_MAX_REL_L2
    )

    # q_a vs golden (its attn_norm input is the device one), and vs the CPU q_a of the same input.
    failures += _out_checks("q_a", "vs_golden", seen["q_resid"], gl["q_resid"], QA_GOLD)
    cpu_qa = ref.component(layer, "q_a")(rctx, norm_dev)
    failures += _out_checks("q_a", "vs_cpu_same_input", seen["q_resid"], cpu_qa, QA_SAME)

    # The q_a share alone: the CPU block with the device attn_hc, attn_in and attn_norm, the CPU q_a and the device
    # indexer.
    qshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: qshare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "indexer": dev_idx,
        },
    )
    ov = _topk_overlap(seen["topk"], qshare["topk"])
    metrics.record("topk_overlap_swap_qa_share", ov)
    print(f"indexer topk with device q_resid vs CPU q_resid: overlap {ov:.6f} (>= {QA_SHARE_TOPK_OVERLAP})")
    if not ov >= QA_SHARE_TOPK_OVERLAP:
        failures.append(f"q_a share: indexer topk overlap {ov:.5f} < {QA_SHARE_TOPK_OVERLAP}")
    q_flips = _flipped(base_router, qshare["router"])
    n_q = int(q_flips.sum())
    if n_q > QA_SHARE_MAX_FLIPS:
        failures.append(f"{n_q} tokens route to another top-8 than the q_a-share block (> {QA_SHARE_MAX_FLIPS})")
    qshare_out = qshare["out"].float().reshape(w.shape)
    failures += _row_checks("qa_share", ashare_out, qshare_out, q_flips, QA_SHARE_RATIO, None, QA_SHARE_MAX_REL_L2)

    # indexer vs golden (its inputs are the device attn_norm and q_resid): exact structure and set overlap.
    failures += _structure(got_tk, want_tk, start)
    failures += _overlap_checks("vs_golden", got_tk, want_tk, start + s, IDX_GOLD)

    # indexer vs the CPU indexer of the same inputs (fresh state from the golden prefix).
    ictx = reference_ctx(ref, layer, g, c)
    cpu_tk = ref.component(layer, "indexer")(ictx, norm_dev, qr_dev).long()
    failures += _overlap_checks("vs_cpu_same_input", got_tk, cpu_tk, start + s, IDX_SAME)

    # the chunk's pooled keys vs the golden state, and vs the CPU pooled keys of the same attn_norm.
    p0, p1 = start // KP, (start + s) // KP
    width = want_state["index_key"].shape[-1]
    got_k = _pooled_keys(idx_state, p0, p1, width)
    if got_k is None:
        failures.append("indexer: no pooled keys (dctx.extra['state_out']['index_key'] [>= (start+S)/4, 128])")
    else:
        failures += _out_checks("index_key", "vs_golden", got_k, want_state["index_key"].float()[p0:p1], KEY_GOLD)
        cpu_k = ref.state_tensors(ictx.state, layer, g.seq)["index_key"].float()[p0:p1]
        failures += _out_checks("index_key", "vs_cpu_same_input", got_k, cpu_k, KEY_SAME)

    # The indexer share at attn_out: the CPU attention of the same device attn_norm and q_resid, with the device topk
    # vs with the CPU topk (post <= 0.024 hides most of the attention from block out, so attn_out shows the selection).
    attn_cpu_tk = ref.component(layer, "attention")(reference_ctx(ref, layer, g, c), norm_dev, qr_dev, cpu_tk.int())
    failures += _out_checks("attn_out", "idx_share", cpu_attn, attn_cpu_tk, ATTN_SHARE)
    # ... and at block out: the CPU block with the device attn_hc, attn_in, attn_norm and q_resid and the CPU indexer.
    ishare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: ishare.__setitem__(n, t),
        overrides={
            "attn_hc": lambda ctx, x: hc_dev,
            "attn_collapse": lambda ctx, x, h: in_dev,
            "attn_norm": lambda ctx, x: norm_dev,
            "q_a": lambda ctx, x: qr_dev,
        },
    )
    i_flips = _flipped(base_router, ishare["router"])
    n_i = int(i_flips.sum())
    if n_i > IDX_SHARE_MAX_FLIPS:
        failures.append(f"{n_i} tokens route to another top-8 than the indexer-share block (> {IDX_SHARE_MAX_FLIPS})")
    ishare_out = ishare["out"].float().reshape(w.shape)
    failures += _row_checks("idx_share", ashare_out, ishare_out, i_flips, IDX_SHARE_RATIO, None, IDX_SHARE_MAX_REL_L2)

    # Chunk 0 (start 0, rows q < 2047 mostly -1 ids: masking bugs show only here): the swapped block vs the golden,
    # and the device attention vs the CPU attention of the same inputs.
    if c > 0 and 0 in g.dumped_chunks:
        gl0 = g.layer(0, layer)
        dctx0 = device_ctx(layer, g, 0)
        ov0 = {n: (lambda ctx, *x, mut=muts[n]: _f32(mut(ctx, dctx0, *x))) for n in SWAPPED}
        seen0 = {}
        run_block(
            steps,
            lambda n: ref.component(layer, n),
            reference_ctx(ref, layer, g, 0),
            gl0["in"].float(),
            rec=lambda n, t: seen0.__setitem__(n, t),
            overrides=ov0,
        )
        _, ok0 = compare("pcc_swap_out_c0", seen0["out"], gl0["out"], "pcc", thr)
        if not ok0:
            failures.append(f"chunk 0: block out PCC below {thr}")
        tk0 = seen0["topk"]
        if tk0.is_floating_point() or tk0.numel() != gl0["topk"].numel():
            failures.append(f"chunk 0: indexer output {tuple(tk0.shape)} {tk0.dtype}")
        else:
            tk0 = tk0.reshape(gl0["topk"].shape)
            failures += [f"chunk 0: {f}" for f in _structure(tk0.long(), gl0["topk"].long(), 0)]
            n0 = seen0["attn_norm"].float().reshape(gl0["attn_norm"].shape)
            q0 = seen0["q_resid"].float().reshape(gl0["q_resid"].shape)
            cpu_attn0 = ref.component(layer, "attention")(reference_ctx(ref, layer, g, 0), n0, q0, tk0.to(torch.int32))
            failures += _out_checks("attention", "c0_vs_cpu_same_input", seen0["attn_out"], cpu_attn0, C0_ATTN_SAME)
        res_in0 = [
            gl0["in"].float(),
            seen0["attn_hc"].float().reshape(gl0["attn_hc"].shape),
            seen0["attn_out"].float().reshape(gl0["attn_out"].shape),
        ]
        cpu_res0 = ref.component(layer, "attn_residual")(reference_ctx(ref, layer, g, 0), *res_in0)
        failures += _res_checks("c0_vs_cpu_same_input", seen0["h_mid"], cpu_res0, RES_SAME, res_in0)
        hm0 = seen0["h_mid"].float().reshape(gl0["h_mid"].shape)
        cpu_fhc0 = ref.component(layer, "ffn_hc")(reference_ctx(ref, layer, g, 0), hm0)
        failures += _fhc_checks("c0_vs_cpu_same_input", seen0["ffn_hc"], cpu_fhc0, FHC_SAME)
        fh0 = seen0["ffn_hc"].float().reshape(gl0["ffn_hc"].shape)
        cpu_fc0 = ref.component(layer, "ffn_collapse")(reference_ctx(ref, layer, g, 0), hm0, fh0)
        failures += _fc_checks("c0_vs_cpu_same_input", seen0["ffn_in"], cpu_fc0, FC_SAME)
        fi0 = seen0["ffn_in"].float().reshape(gl0["ffn_in"].shape)
        cpu_fn0 = ref.component(layer, "ffn_norm")(reference_ctx(ref, layer, g, 0), fi0)
        failures += _fn_checks("c0_vs_cpu_same_input", seen0["ffn_norm"], cpu_fn0, FN_SAME)
        fn0 = seen0["ffn_norm"].float().reshape(gl0["ffn_norm"].shape)
        cpu_rt0 = ref.component(layer, "router")(reference_ctx(ref, layer, g, 0), fn0)
        failures += _router_checks("c0_vs_cpu_same_input", seen0["router"], cpu_rt0, RT_SAME)
        rt0 = seen0["router"].float().reshape(gl0["router"].shape)
        cpu_ex0 = ref.component(layer, "experts")(reference_ctx(ref, layer, g, 0), fn0, rt0)
        failures += _experts_checks("c0_vs_cpu_same_input", seen0["experts_out"], cpu_ex0, EX_SAME)
        cpu_se0 = ref.component(layer, "shared_expert")(reference_ctx(ref, layer, g, 0), fn0)
        failures += _experts_checks("c0_vs_cpu_same_input", seen0["shared_out"], cpu_se0, SE_SAME, step="shared_expert")
    assert not failures, "; ".join(failures)
