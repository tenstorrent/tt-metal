# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 6: block type dsa_moe (layer 3) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 3 (dsa_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    q_a
    indexer
    attention

Reviewed (S.dsa_moe.06.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). attention: attn_out = MLA(attn_norm, q_resid, topk) [2048, 4096], stateful
(writes the chunk's kv_latent rows [start, start + S) of [max_seq, 512]); enters block out as post * attn_out through
attn_residual, then the MoE. The trail's pcc_swap_* are for diagnosis only.
Share design: a re-run device attention moves by rel ~0.004 on any input change (fed the CPU topk instead of the
device topk it moves 0.0043, the CPU attention 0.0016), so shares of the earlier steps that re-run it measure the
attention's noise. The earlier shares are therefore taken against the attention-share block (device hc..indexer
outputs fixed, CPU attention), with the CPU attention downstream, as in swap 05; their limits and device numbers
are swap 05's.
Measured on the CPU (golden s4096 chunk 1, all-CPU block; the attention replaced by a perturbed copy of the CPU MLA).
Columns: attn_out rel L2 / per-token ratio / worst row vs the unperturbed attention, then block out: tokens with
another top-8 / same-routing rel L2 / ratio:
  Noise: bf16 latent, q_lat, probabilities, o_lat, heads 0.0021 / [0.9992, 1.0007] / 0.0030, 16 / 0.00038; 0.5%
  output noise 0.0050, 69 / 0.00076; 1% 0.010, 140 / 0.0015.
  Scale: output x0.995 [0.995], 24 / 0.00146 / [0.9971, 1.0032]; x1.01 57 / 0.0029; x1.02 114 / 0.0058; rows
  1536.. (chip 3) x1.01 15 / 0.0014 / [0.9938, 1.0052]; score scale x1.02 0.0196 / [1.0085, 1.0216], 123 / 0.0056;
  512^-0.5 0.29, 1173; latent x1.01 0.013 / [1.0008, 1.0172], 82 / 0.0041; kv eps 0 0.0061 / [1.0003, 1.0075], 30 /
  0.0018.
  Structure: -1 not masked (c1) 0.018 / worst row 0.086, 72 / 0.0041 (c0: rel 1.07, worst row 1.59); last complete
  pool dropped 0.0031 / worst row 0.135, 15 / 0.00027; latent written 32 rows late 0.26, 986; rows 512..1023 swapped
  with 1024..1535 0.61, 952; first or last row zero 0.020..0.025 / worst row 1.0, 1 flip, block out unchanged
  (post <= 0.024: only attn_out sees a row); zero 1.0, 1887.
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 05 asserts (block out vs golden and vs the all-CPU block, now with six device steps' share;
    attn_hc parts; attn_collapse, attn_norm, q_a and indexer vs golden and vs the CPU step of their own inputs; the
    pooled keys, captured from the indexer's own call: the attention overwrites dctx.extra["state_out"]);
  - attention vs golden (device inputs): rel <= 0.012, ratio [0.99, 1.01], worst row <= 0.03 (the component ratio
    [0.994, 1.006] is upstream error here: the CPU attention of the same device inputs scores [0.9939, 1.0001]);
  - attention vs the fp32 CPU attention of the same device inputs: rel <= 0.006, ratio [0.994, 1.004], worst row
    <= 0.02. With the device's -0.1% bias this also fails output x1.005 (1.0054), which the component test passes;
  - the chunk's latent rows vs the golden state (rel <= 0.006, ratio [0.997, 1.003], worst row <= 0.01; device
    attn_norm in) and vs the CPU latent of the same attn_norm (rel <= 0.0035, ratio [0.997, 1.003], worst row
    <= 0.005); prefix rows equal to the loaded golden prefix (rel <= 1e-3);
  - the attention share: the CPU block with the device attn_hc, attn_in, attn_norm, q_resid and topk and the CPU
    attention: flips <= 48, same-routing rel L2 <= 0.0012, ratio [0.996, 1.004];
  - the indexer share at attn_out is the CPU attention of the device topk vs of the CPU topk (same device attn_norm,
    q_resid; limits as swap 05), and at block out the CPU block with the CPU indexer vs the attention-share block;
  - chunk 0 (start 0, rows q < 2047 mostly -1 ids, where masking bugs show): the swapped block PCC >= 0.98 vs the
    golden, the indexer structure, and the attention vs the CPU attention of the same device inputs (as chunk 1).
Device (first run): PCC 0.999995 (c0 0.999994); attention vs CPU same input 0.00370 / [0.9954, 1.0004] / 0.00873 (c0
0.00384 / [0.9966, 1.0006] / 0.00684), vs golden 0.00654 / [0.9931, 0.9993] / 0.0159; latent vs golden 0.00422 /
[0.9987, 1.0007] / 0.00647, vs CPU 0.00204 / [0.9988, 1.0008] / 0.00237; attention share 30 flips / 0.00069 /
[0.9976, 1.0014]; block out vs the all-CPU block 29 flips / 0.00199 / [0.9924, 1.0075]; earlier shares and indexer
checks as swap 05. About 90 s. Reference: every same-input and share check exact. Stub: fails PCC and every check.
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
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "q_a", "indexer", "attention"]
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


ATTN_GOLD = {"rel": 0.012, "ratio": (0.99, 1.01), "row": 0.03}  # attn_out vs golden (device inputs: upstream error)
ATTN_SAME = {"rel": 0.006, "ratio": (0.994, 1.004), "row": 0.02}  # vs the fp32 CPU attention of the same inputs
C0_ATTN_SAME = ATTN_SAME  # chunk 0 (start 0, mostly -1 ids), same inputs
LAT_GOLD = {"rel": 0.006, "ratio": (0.997, 1.003), "row": 0.01}  # chunk's latent rows vs golden (device attn_norm in)
LAT_SAME = {"rel": 0.0035, "ratio": (0.997, 1.003), "row": 0.005}  # vs the CPU latent of the same attn_norm
LAT_PREFIX_MAX_REL_L2 = 1e-3  # prefix rows must stay the loaded golden prefix
ATTN_SHARE_MAX_FLIPS = 48  # attention share: rows whose top-8 differs from the CPU block with device hc..indexer
ATTN_SHARE_MAX_REL_L2 = 0.0012  # same-routing rows
ATTN_SHARE_RATIO = (0.996, 1.004)


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
    a_flips = _flipped(seen["router"], ashare["router"])
    n_a = int(a_flips.sum())
    if n_a > ATTN_SHARE_MAX_FLIPS:
        failures.append(
            f"{n_a} tokens route to another top-8 than the attention-share block (> {ATTN_SHARE_MAX_FLIPS})"
        )
    ashare_out = ashare["out"].float().reshape(w.shape)
    failures += _row_checks("attn_share", out, ashare_out, a_flips, ATTN_SHARE_RATIO, None, ATTN_SHARE_MAX_REL_L2)

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
    assert not failures, "; ".join(failures)
