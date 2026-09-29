# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 5: block type dsa_moe (layer 3) with indexer swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 3 (dsa_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    q_a
    indexer

Reviewed (S.dsa_moe.05.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). indexer: topk = indexer(attn_norm, q_resid), int32 [2048, 2051] (top 512
pools of 4 + tail, -1 = none), stateful (writes the chunk's pooled keys). The CPU attention consumes the device ids.
The trail's pcc_swap_topk ("match") is order-dependent and not a check; the indexer is compared by set overlap.
Measured on the CPU (golden s4096 chunk 1, all-CPU block; the indexer replaced by a perturbed CPU indexer on the CPU
attn_norm and q_resid). Columns: topk overlap mean / worst row vs the unperturbed indexer, attn_out rel L2 vs the
attention of the unperturbed topk, then block out: tokens with another top-8 / same-routing rel L2 / per-row ratio:
  Noise: bf16 q, keys, weights and scores 0.99927 / 0.9941, attn 0.00158, 6 / 0.00024 / [0.9987, 1.0006]; 1% score
  noise 0.99900 / 0.9941, 0.00158, 5 / 0.00026 / [0.9990, 1.0007]; 0.3% 0.99964, 0.00100, 3 / 0.00016.
  Score bugs: no ape 0.99702 / 0.9824, 0.00318, 19 / 0.00054; ape reversed 0.99633 / 0.9746, 0.00363, 21 / 0.00061;
  no relu 0.937, 0.022, 121; prefix pooled keys zero 0.761, 0.089, 504; lowest scores 0.501, 0.36; per-pool key scale
  0.97..1.03 0.99947, 0.00126, 3 (only the pooled-key checks see key scale: x1.01 on the chunk's keys, rel 0.010).
  Structure bugs (overlap stays high, attn_out moves a lot): pool visible one token early 0.99951, attn 0.131, 225
  flips; no tail 0.99927, 0.183, 572; tail one past q 0.99963, 0.236, 816 (also an id out of range); 511 pools 0.99805,
  0.0031. Row bugs: rows 512..1023 swapped with 1024..1535 (chip order) 0.859, 0.20; 32 rows shifted 0.99803 / worst
  row 0.766, 0.025; last row wrong 0.99969 / 0.373, 0.0057, 1 flip, same-routing block out unchanged; zeros 0.0005.
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 04 asserts (block out vs golden and vs the all-CPU block, now with five device steps' share;
    attn_hc parts; attn_collapse, attn_norm and q_a vs golden and vs the fp32 CPU step of their own inputs);
  - the collapse, norm and q_a shares now also run the device indexer, so each still isolates one step (limits as
    swap 04); the q_a share's topk overlap compares two device-indexer runs;
  - indexer vs golden (device attn_norm, q_resid in): exact structure (ids in range, causal, no duplicates, per-row
    count equal to the golden's, complete pools, exact sets on all-visible rows; catches every structure and count
    bug above) and the component limits, overlap >= 0.9975, worst row >= 0.98;
  - indexer vs the CPU indexer of the same inputs: overlap >= 0.9985, worst row >= 0.988 (fails the score and row
    bugs, noise passes);
  - the chunk's pooled keys (captured right after the main run: the shares overwrite state_out) vs the golden state
    (component limits: rel 0.008, ratio [0.995, 1.005], worst row 0.02) and vs the CPU keys of the same attn_norm:
    rel <= 0.005, ratio [0.996, 1.004], worst row <= 0.01;
  - the indexer share: the CPU block with the device attn_hc, attn_in, attn_norm and q_resid and the CPU indexer.
    attn_out rel L2 <= 0.0025, ratio [0.993, 1.007], worst row <= 0.1 (post <= 0.024 hides most of the attention
    from block out, so attn_out is where the selection shows); block out flips <= 12, same-routing rel L2 <= 0.0004,
    ratio [0.998, 1.002].
Device (first run): indexer vs golden 0.998571 / 0.9922, vs CPU same input 0.999194 / 0.9941; pooled keys vs golden
0.00317 / [0.9981, 1.0001], vs CPU 0.00203 / [0.9983, 0.9996] / 0.00268; indexer share attn_out 0.00156 /
[0.9969, 1.0013] / 0.0133, block out 8 flips / 0.00026 / [0.9987, 1.0005]; q_a share topk 0.99967 / 6 / 0.00021;
norm share 5 / 0.00026; collapse share 0 / 0.00004; block out vs CPU 29 flips / 0.00197 / [0.9924, 1.0075]; PCC
0.999995 (pcc_swap_topk match 0.0024). Reference: every same-input and share check exact (0 / 1.0). Stub: fails
PCC and every check.
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
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "q_a", "indexer"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.975, 1.025)  # per-row ||got|| / ||want||, rows whose top-8 equals the golden's
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 64  # rows whose top-8 differs from the CPU block of the same `in`
CPU_MAX_REL_L2 = 0.005  # block out vs CPU block, same-routing rows
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
    # the pooled keys the swapped indexer wrote (the shares below run it again and overwrite state_out)
    mode = impl_mode()
    if mode == "device":
        idx_state = dctx.extra.get("state_out")
    elif mode == "reference":
        idx_state = ref.state_tensors(rctx.state, layer, g.seq)
    else:
        idx_state = None
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
    s_flips = _flipped(seen["router"], share["router"])
    n_s = int(s_flips.sum())
    if n_s > SHARE_MAX_FLIPS:
        failures.append(f"{n_s} tokens route to another top-8 than the collapse-share block (> {SHARE_MAX_FLIPS})")
    share_out = share["out"].float().reshape(w.shape)
    failures += _row_checks("collapse_share", out, share_out, s_flips, SHARE_RATIO, None, SHARE_MAX_REL_L2)

    # attn_norm vs golden (its attn_in input is the device one), and vs the CPU norm of the same input.
    in_dev = seen["attn_in"].float().reshape(gl["attn_in"].shape)
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
    n_flips_n = _flipped(seen["router"], nshare["router"])
    n_n = int(n_flips_n.sum())
    if n_n > NORM_SHARE_MAX_FLIPS:
        failures.append(f"{n_n} tokens route to another top-8 than the norm-share block (> {NORM_SHARE_MAX_FLIPS})")
    nshare_out = nshare["out"].float().reshape(w.shape)
    failures += _row_checks("norm_share", out, nshare_out, n_flips_n, NORM_SHARE_RATIO, None, NORM_SHARE_MAX_REL_L2)

    # q_a vs golden (its attn_norm input is the device one), and vs the CPU q_a of the same input.
    norm_dev = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
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
    q_flips = _flipped(seen["router"], qshare["router"])
    n_q = int(q_flips.sum())
    if n_q > QA_SHARE_MAX_FLIPS:
        failures.append(f"{n_q} tokens route to another top-8 than the q_a-share block (> {QA_SHARE_MAX_FLIPS})")
    qshare_out = qshare["out"].float().reshape(w.shape)
    failures += _row_checks("qa_share", out, qshare_out, q_flips, QA_SHARE_RATIO, None, QA_SHARE_MAX_REL_L2)

    # indexer vs golden (its inputs are the device attn_norm and q_resid): exact structure and set overlap.
    start, s = c * g.chunk, gl["topk"].shape[0]
    want_tk = gl["topk"].long()
    got_tk = seen["topk"]
    if got_tk.is_floating_point() or got_tk.numel() != want_tk.numel():
        failures.append(
            f"indexer: output {tuple(got_tk.shape)} {got_tk.dtype}, want {tuple(want_tk.shape)} integer ids"
        )
        assert not failures, "; ".join(failures)
    got_tk = got_tk.reshape(want_tk.shape).long()
    failures += _structure(got_tk, want_tk, start)
    failures += _overlap_checks("vs_golden", got_tk, want_tk, start + s, IDX_GOLD)

    # indexer vs the CPU indexer of the same inputs (fresh state from the golden prefix).
    qr_dev = seen["q_resid"].float().reshape(gl["q_resid"].shape)
    ictx = reference_ctx(ref, layer, g, c)
    cpu_tk = ref.component(layer, "indexer")(ictx, norm_dev, qr_dev).long()
    failures += _overlap_checks("vs_cpu_same_input", got_tk, cpu_tk, start + s, IDX_SAME)

    # the chunk's pooled keys vs the golden state, and vs the CPU pooled keys of the same attn_norm.
    p0, p1 = start // KP, (start + s) // KP
    want_state = g.state(layer)
    width = want_state["index_key"].shape[-1]
    got_k = _pooled_keys(idx_state, p0, p1, width)
    if got_k is None:
        failures.append("indexer: no pooled keys (dctx.extra['state_out']['index_key'] [>= (start+S)/4, 128])")
    else:
        failures += _out_checks("index_key", "vs_golden", got_k, want_state["index_key"].float()[p0:p1], KEY_GOLD)
        cpu_k = ref.state_tensors(ictx.state, layer, g.seq)["index_key"].float()[p0:p1]
        failures += _out_checks("index_key", "vs_cpu_same_input", got_k, cpu_k, KEY_SAME)

    # The indexer share alone: the CPU block with the device attn_hc, attn_in, attn_norm and q_resid and the CPU
    # indexer. attn_out sees the selection directly (post <= 0.024 hides most of it from block out).
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
    failures += _out_checks("attn_out", "idx_share", seen["attn_out"], ishare["attn_out"].float(), ATTN_SHARE)
    i_flips = _flipped(seen["router"], ishare["router"])
    n_i = int(i_flips.sum())
    if n_i > IDX_SHARE_MAX_FLIPS:
        failures.append(f"{n_i} tokens route to another top-8 than the indexer-share block (> {IDX_SHARE_MAX_FLIPS})")
    ishare_out = ishare["out"].float().reshape(w.shape)
    failures += _row_checks("idx_share", out, ishare_out, i_flips, IDX_SHARE_RATIO, None, IDX_SHARE_MAX_REL_L2)
    assert not failures, "; ".join(failures)
