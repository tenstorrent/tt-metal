# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 10: block type dsa_moe (layer 3) with ffn_norm swapped in last.

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

Reviewed (S.dsa_moe.10.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). ffn_norm is post_attention_layernorm w * rms(ffn_in), eps 1e-5, w nearly
constant at layer 3 (0.441..0.531); router, routed experts and shared expert read it without re-normalizing, so a
norm scale error reaches block out amplified ~1.8x (and flips routing).
Share design: shares telescope, as in swap 09. The new norm-share block (every device output up to ffn_in fixed, CPU
ffn_norm and tail) is the base of the norm's share (block out vs it). The collapse share is now that block vs the
collapse-share block, and reproduces swap 09's collapse share exactly. The earlier shares are unchanged.
Measured on the CPU (golden s4096 chunk 1; golden h_mid, ffn_hc and ffn_in, the norm output perturbed, CPU tail).
Columns: ffn_norm rel L2 / coef, then block out: tokens with another top-8 / same-routing rel L2 / per-row ratio /
flipped-row ratio:
  Noise: bf16 output 0.0017 / 1.0000, 36 / 0.00045 / [0.9995, 1.0008] / [0.960, 1.009]; 0.2% element noise 54 /
  0.00070 / [0.9988, 1.0015] / [0.981, 1.009]; 0.3% 82 / 0.00092 / [0.9989, 1.0023] / [0.966, 1.033]; 0.1% 50 /
  0.00052 / flipped max 1.033.
  Bugs: x0.999 2 / 0.0018 / min 0.9973; x1.0015 12 / 0.0028 / max 1.0041; x1.0025 0.0046 / 1.0068; x1.005 24 /
  0.0092 / 1.0137 (the norm checks see coef 1.005); x0.99 56 / 0.018; truncating bf16 46 / 0.0052 / min 0.9926;
  0.5% noise 122 / 0.0014; per-row rsqrt noise 1e-3 0.0019 / [0.9940, 1.0064]; bf16 square accumulate 0.0073 /
  [0.9775, 1.0322]; eps 1.05e-5 0.016; eps 9.5e-6 0.017; mean subtraction 231 / 0.0032; norm over 4 shards 219 /
  0.0033; w reversed 560 / 0.0077; cols of chip 3 x1.01 104 / 0.0048; rows of chip 3 x1.005 0.0045 / 1.0127; one row
  x1.02 0 flips / ratio 1.033; one row copied from its neighbour flips it (flipped ratio 1.247); last row zero 0.88,
  last 32 rows zero 0.33 (flipped-row ratio); last 32 columns zero 1350 flips; rows shifted 2047 flips.
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 09 asserts, limits unchanged (flips vs the all-CPU block stay <= 96: the device scores 70);
  - collapse share: the norm-share block vs the collapse-share block (limits as swap 09);
  - ffn_norm vs the fp32 CPU norm of the ffn_in it got, at the component test's limits: rel L2 <= 0.01, per-token
    ratio [0.99, 1.01], worst row <= 0.015, coefficient [0.996, 1.004]; also on chunk 0;
  - ffn_norm vs golden (carries the upstream ffn_in error): rel <= 0.01, ratio [0.99, 1.01], worst row <= 0.02,
    coefficient [0.995, 1.005] (the CPU norm of the device ffn_in: 0.0046 / [0.9980, 1.0007] / 0.0096 / 0.99966);
  - norm share: block out vs the norm-share block: flips <= 64, same-routing rel L2 <= 0.0015, ratio [0.996, 1.004],
    flipped-row ratio [0.92, 1.08] (catches every scale error from x0.999 / x1.0015 up, truncation, per-row rsqrt
    noise, bf16 square accumulation, eps, mean subtraction, sharded norm, w reversed, a changed or zeroed row).
    Not caught at block out: 0.5% element noise (0.0014, 122 flips > 64 does catch it).
Device (first run): PCC 0.999992 (c0 0.999993); every swap 09 number unchanged (collapse share 30 / 0.00044 /
[0.9994, 1.0011], flipped [0.9935, 1.0250]); block out vs golden 76 flips / 0.00274 / [0.9907, 1.0070]; vs the
all-CPU block 70 / 0.00202 / [0.9909, 1.0075]; ffn_norm vs CPU same input 0.00168 / [0.9990, 1.0005] / row 0.00197 /
coef 0.99989 (c0 0.00168 / 0.00193 / 0.99990), vs golden 0.00493 / [0.9972, 1.0006] / 0.00981 / 0.99955; norm share
34 / 0.00065 / [0.9982, 1.0014], flipped rows [0.9593, 1.0073].
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
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.975, 1.025)  # per-row ||got|| / ||want||, rows whose top-8 equals the golden's
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 96  # rows whose top-8 differs from the CPU block of the same `in` (bf16 noise; rel L2 gates)
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
    fn_flips = _flipped(seen["router"], nfshare["router"])
    n_fn = int(fn_flips.sum())
    if n_fn > FN_SHARE_MAX_FLIPS:
        failures.append(f"{n_fn} tokens route to another top-8 than the norm-share block (> {FN_SHARE_MAX_FLIPS})")
    nfshare_out = nfshare["out"].float().reshape(w.shape)
    failures += _row_checks(
        "fn_share", out, nfshare_out, fn_flips, FN_SHARE_RATIO, FN_SHARE_FLIPPED_RATIO, FN_SHARE_MAX_REL_L2
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
    assert not failures, "; ".join(failures)
