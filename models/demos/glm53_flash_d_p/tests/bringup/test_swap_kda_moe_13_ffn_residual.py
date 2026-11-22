# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 13: block type kda_moe (layer 4) with ffn_residual swapped in last (every step on the device).

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 4 (kda_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    attention
    attn_residual
    ffn_hc
    ffn_collapse
    ffn_norm
    router
    experts
    shared_expert
    moe_add
    ffn_residual

Reviewed (S.kda_moe.13.test.1): rewritten from the frozen kda_moe swap 12 test (every check and limit kept, its
docstring history is in git) plus dsa_moe swap 15's ffn_residual checks at the layer-4 limits. The gated metric is
pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual [S * 4, H], spec block threshold 0.98). Block out is now the
device ffn_residual, out = post * mlp_out + comb^T @ h_mid (post = ffn_hc[:, 4:8], comb = ffn_hc[:, 8:24]). At layer
4 the post term is 0.148 of the output and the comb term dominates (layer 3: post term 0.70).
Shares telescope: the new residual-share block (device outputs through moe_add fixed, CPU ffn_residual) is the base of
the residual's share. Its out is exactly the fp32 CPU residual of the device (h_mid, ffn_hc, mlp_out), so the
residual's share is the same-input check below. The add share is now that block vs the add-share block and
reproduces swap 12's add share (both end in the CPU residual). The earlier shares are unchanged.
Extra asserted checks (informational metrics, not in the runner's list; limits are written `not x <= lim`, so NaN
fails):
  - everything swap 12 asserts, limits unchanged. Block out vs golden rel L2 <= 0.01. Vs the all-CPU block,
    same-routing rel L2 <= 0.004: the device residual adds its bf16 output rounding (0.00166) in quadrature to swap
    12's 0.00296, giving 0.00339. A truncating bf16 output would add about 0.0028 and reach about 0.0041;
  - add share: the residual-share block vs the add-share block (limits as swap 12);
  - ffn_residual vs the fp32 CPU residual of the (h_mid, ffn_hc, mlp_out) it actually got: the layer-4 component
    test's same-input limits (rel L2 <= 0.0035, per-row ratio [0.997, 1.003], worst row <= 0.006, per-stream rel
    <= 0.005). Each term is also checked on its own, as the output minus the exact other term, vs the term: post
    coefficient [0.998, 1.002] and rel <= 0.025 (an RNE bf16 output alone gives 0.0112), comb coefficient
    [0.998, 1.002] and rel <= 0.004. Also on chunk 0. These were sized in test_c_kda_moe_ffn_residual.py on the
    layer-4 golden. Passes: RNE bf16 output, all-bf16 mix, 0.3% noise. Fails: post or comb x1.003, a truncating bf16
    output, each single stream x1.01, half the rows x1.003, the last row zeroed or x1.01, the last token x1.05, post on
    stream 0 x1.01. The residual is linear and has no weights, and the device inputs have the golden's scale, so these
    sensitivities carry over.
Device (first run): PCC 0.999989 (rel 0.00486). Every swap 12 number is unchanged (add share 0.00026 /
[1.0000, 1.0006], shared share 0.00009, experts share 0.00167).
  - Block out vs golden: 64 flips; same-routing 0.00411 / [0.9802, 1.0050].
  - Vs the all-CPU block: 62 flips / 0.00339 / [0.9804, 1.0054].
  - ffn_residual vs the CPU residual of the same inputs: 0.00166 / [0.9998, 1.0013] / row 0.00206 / per-stream
    0.00166; post coef 1.00000 rel 0.01119; comb coef 1.00000 rel 0.00169.
  - Chunk 0: 0.00165 / post rel 0.01350 / comb rel 0.00166.
  - That is exactly an RNE bf16 output of the exact step, as in the component test.
Reference: PCC 0.999997, every share and same-input check exact. Stub fails (PCC 0). About 105 s (real pass).
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
BLOCK_TYPE = "kda_moe"
SWAPPED = [
    "attn_hc",
    "attn_collapse",
    "attn_norm",
    "attention",
    "attn_residual",
    "ffn_hc",
    "ffn_collapse",
    "ffn_norm",
    "router",
    "experts",
    "shared_expert",
    "moe_add",
    "ffn_residual",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.975, 1.015)  # per-row ||got|| / ||want||, same top-8 as golden (ffn_hc comb bias: 0.980)
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 96  # rows whose top-8 differs from the CPU block of the same `in`
CPU_MAX_REL_L2 = 0.004  # block out vs CPU block, same-routing rows (swap 09 0.00242 + the experts share)
CPU_ROW_RATIO = (0.975, 1.015)  # swap 05 0.985; the device ffn_hc takes the minimum to 0.980
CPU_ROW_RATIO_FLIPPED = (0.9, 1.1)
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.02, "post": 6e-3, "comb": 0.02}
MAX_COL_REL_L2 = 0.05
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|
COL_GOLD = {"rel": 0.01, "ratio": (0.985, 1.015), "row": 0.02}  # attn_collapse vs golden attn_in (device attn_hc in)
COL_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.0065}  # vs the fp32 CPU collapse of the same inputs
SHARE_MAX_FLIPS = 24  # collapse share: vs the attention-share block, CPU collapse instead of the device one
SHARE_MAX_REL_L2 = 0.0015  # collapse share, same-routing rows
SHARE_RATIO = (0.995, 1.005)
SHARE_RATIO_FLIPPED = (0.9, 1.1)
NORM_GOLD = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.015, "coef": (0.996, 1.004)}  # vs golden (device attn_in in)
NORM_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.008, "coef": (0.999, 1.001)}  # vs fp32 CPU norm, same in
NORM_SHARE_MAX_FLIPS = 20  # norm share: vs the attention-share block, CPU norm instead of the device one
NORM_SHARE_MAX_REL_L2 = 0.0004
NORM_SHARE_RATIO = (0.998, 1.002)
NORM_SHARE_RATIO_FLIPPED = (0.9, 1.1)
ATTN_BLOCK_ROWS = 128  # state / SP-carry / conv-tail bugs hit the first rows of the chunk or of an SP segment
ATTN_GOLD = {"rel": 0.015, "ratio": (0.975, 1.006), "row": 0.06, "block": 0.015, "coef": (0.985, 1.004)}  # vs golden
ATTN_SAME = {"rel": 0.012, "ratio": (0.98, 1.004), "row": 0.05, "block": 0.012, "coef": (0.99, 1.002)}  # vs CPU KDA
C0_ATTN_SAME = ATTN_SAME  # chunk 0 (start 0: zero prefix state), same inputs
STATE_GOLD = {"kda_recurrent": 0.025, "kda_conv": 0.008, "head": 0.05}  # post-chunk state vs golden snapshot
STATE_SAME = {"kda_recurrent": 0.025, "kda_conv": 0.005, "head": 0.05}  # vs the CPU state of the same attn_norm
ATTN_SHARE_MAX_FLIPS = 96  # attention share: vs the CPU block with device hc, attn_in, attn_norm and CPU attention
ATTN_SHARE_MAX_REL_L2 = 0.002
ATTN_SHARE_RATIO = (0.99, 1.01)
ATTN_SHARE_RATIO_FLIPPED = (0.95, 1.06)
RES_GOLD = {"rel": 0.006, "ratio": (0.985, 1.015), "row": 0.03, "stream": 0.008}  # h_mid vs golden (device inputs)
RES_SAME = {"rel": 0.0035, "ratio": (0.997, 1.003), "row": 0.008}  # vs the fp32 CPU residual of the same inputs
COMB_COEF = (0.998, 1.002)  # <out - post term, comb term> / ||comb term||^2, terms from the step's own inputs
COMB_MAX_REL = 0.0035  # ||out - post term - comb term|| / ||comb term||
POST_COEF = (0.99, 1.01)  # <out - comb term, post term> / ||post term||^2
POST_MAX_REL = 0.10  # bf16 output rounding alone gives 0.063 (post term 2.6% of h_mid at layer 4)
RES_SHARE_MAX_FLIPS = 64  # residual share: block out vs the CPU block with device hc..attention, CPU residual
RES_SHARE_MAX_REL_L2 = 0.0025  # same-routing rows
RES_SHARE_RATIO = (0.997, 1.003)
FHC_SAME = {  # ffn_hc vs the fp32 CPU ffn_hc of the same h_mid: the component test's limits
    "rel": {"pre": 0.01, "post": 0.01, "comb": 0.01},
    "abs": {"pre": 0.02, "post": 0.01, "comb": 0.02},
    "col": 0.06,
    "coef": (0.995, 1.005),
}
FHC_GOLD = FHC_SAME  # vs golden (device h_mid in)
FHC_SHARE_MAX_FLIPS = 40  # ffn_hc share: rows whose top-8 differs from the CPU block with device hc..attn_residual
FHC_SHARE_MAX_REL_L2 = 0.0035  # same-routing rows
FHC_SHARE_RATIO = (0.975, 1.01)  # device [0.9825, 1.0046]: small comb entries a few % low
FHC_SHARE_RATIO_FLIPPED = (0.9, 1.1)  # a zeroed row flips its token and hides from the same-routing check
FC_SAME = {"rel": 0.008, "ratio": (0.99, 1.01), "row": 0.008, "coef": (0.996, 1.004)}  # vs the fp32 CPU collapse of
# the same (h_mid, ffn_hc): the component test's limits (also on chunk 0 and on layer 3's golden)
FC_GOLD = {"rel": 0.008, "ratio": (0.985, 1.01), "row": 0.03, "coef": (0.995, 1.005)}  # vs golden ffn_in: the
# upstream h_mid error (RES_GOLD); the CPU collapse of the device inputs scores 0.0057 / [0.9930, 1.0011] / 0.0165
FC_SHARE_MAX_FLIPS = 64  # collapse share: rows whose top-8 differs from the CPU block with device hc..ffn_hc
FC_SHARE_MAX_REL_L2 = 0.0008  # same-routing rows (bf16 products and sums 0.00038, 0.5% noise 0.00094)
FC_SHARE_RATIO = (0.996, 1.004)  # bf16 [0.9979, 1.0008]; x1.003 1.0047
FC_SHARE_RATIO_FLIPPED = (0.95, 1.05)  # bf16 [0.9866, 1.0040]; a zeroed / reversed / duplicated row 0.84 / 0.63 / 1.71
FN_SAME = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.015, "coef": (0.996, 1.004)}  # vs the fp32 CPU norm of the
# same ffn_in: the component test's limits
FN_GOLD = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.03, "coef": (0.995, 1.005)}  # vs golden (device ffn_in in)
FN_SHARE_MAX_FLIPS = 64  # norm share: rows whose top-8 differs from the CPU block with device hc..ffn_collapse
FN_SHARE_MAX_REL_L2 = 0.0009  # same-routing rows (bf16 output 0.00031, 0.3% noise 0.00065, x1.0025 0.00094)
FN_SHARE_RATIO = (0.996, 1.004)  # bf16 [0.9987, 1.0008]; rsqrt row noise 1e-3 [0.9950, 1.0052]
FN_SHARE_FLIPPED_RATIO = (0.93, 1.07)  # bf16 [0.995, 1.011], 0.2% noise 0.973; a zeroed row 0.84
TOP_K = 8
ROUTE_SCALE = 2.5  # routed_scaling_factor: every routing row sums to 2.5
RT_SAME = {"mean": 0.999, "row": 0.75, "mrel": 0.0025, "coef": (0.9995, 1.0005), "sum": 0.015}  # vs the fp32 CPU
# router of the same ffn_norm (bf16 output rounding alone: 1.0 / 1.0 / 0.00167 / 0.99999 / [2.4941, 2.5059])
RT_GOLD = {"mean": 0.99, "row": 0.5, "mrel": 0.005, "coef": (0.998, 1.002), "sum": 0.015}  # vs golden (device
# ffn_norm in): the component test's limits (the CPU router of the device ffn_norm: 0.99615 / 0.00281 / 0.99994)
RT_SHARE_MAX_FLIPS = 16  # router share: rows whose top-8 differs from the CPU block with device hc..ffn_norm
RT_SHARE_MAX_REL_L2 = 0.0005  # same-routing rows (bf16 weights alone 0.00023; dsa_moe 0.0015 passes 1% weight noise)
RT_SHARE_RATIO = (0.997, 1.003)  # bf16 weights alone [0.9978, 1.0020]; x1.004 1.0035, 0.3% weight noise 0.9951
RT_SHARE_FLIPPED_RATIO = (0.95, 1.05)  # a copied, rolled or zeroed row 0.939 / 0.916 / 0.857
EX_SAME = {"rel": 0.015, "ratio": (0.988, 1.012), "row": 0.02, "coef": (0.997, 1.003), "block": (0.996, 1.004)}
# experts vs the fp32 CPU experts of the same (ffn_norm, router): the layer-4 component test's limits
EX_GOLD = {"rel": 0.015, "ratio": (0.985, 1.015), "row": 0.035, "coef": (0.996, 1.004), "block": (0.995, 1.005)}
# experts vs golden on the rows whose top-8 equals the golden's (device ffn_norm and router in; the CPU experts of the
# same inputs score coef 0.99778 / blocks min 0.99701 / worst row 0.0255: the upstream ffn_collapse coef 0.99744)
EX_GOLD_FLIPPED_RATIO = (0.8, 1.25)  # rows with another top-8 (a whole expert changes)
BLOCK_ROWS = 128
EX_SHARE_MAX_REL_L2 = 0.003  # experts share: block out vs the CPU block with device hc..router (device-like 0.0015)
EX_SHARE_RATIO = (0.994, 1.006)  # x1.005 experts gives 1.0044, a zeroed row 0.857
SE_SAME = {"rel": 0.007, "ratio": (0.995, 1.005), "row": 0.012, "coef": (0.997, 1.003), "block": (0.997, 1.003)}
# shared_expert vs the fp32 CPU shared expert of the same ffn_norm: the layer-4 component test's limits
SE_GOLD = {"rel": 0.018, "ratio": (0.98, 1.03), "row": 0.06, "coef": (0.995, 1.005), "block": (0.994, 1.006)}
# shared_expert vs golden (device ffn_norm in: carries the upstream error; the CPU shared expert of the same inputs
# scores 0.0121 / [0.9903, 1.0201] / worst row 0.0430 / coef 1.00243)
SE_SHARE_MAX_REL_L2 = 0.00025  # shared share: block out vs the CPU block with device hc..experts (x1.01 0.00038)
SE_SHARE_RATIO = (0.9985, 1.0015)  # x1.005 shared gives max 1.0019, last row zeroed min 0.984
MA_SAME = {"rel": 0.0035, "ratio": (0.997, 1.003), "row": 0.006}  # moe_add vs the fp32 sum of the same device
# experts_out and shared_out: the layer-4 component test's limits
MA_EXPERTS = ((0.997, 1.003), 0.004)  # coef <out - shared, experts> / ||experts||^2, rel of out - shared vs experts
MA_SHARED = ((0.996, 1.004), 0.012)  # the same for the shared addend (layer 4: a truncating bf16 add gives 0.0091)
MA_GOLD = {"rel": 0.015, "ratio": (0.985, 1.015), "row": 0.035, "coef": (0.996, 1.004), "block": (0.995, 1.005)}
# moe_add vs golden on the rows whose top-8 equals the golden's (device experts and shared in: upstream error)
MA_GOLD_FLIPPED_RATIO = (0.8, 1.25)
MA_SHARE_MAX_REL_L2 = 0.0005  # add share: block out vs the CPU block with device hc..shared_expert (bf16 RNE 0.00025)
MA_SHARE_RATIO = (0.998, 1.002)  # truncating bf16 min 0.99765, x1.003 max 1.00264, shared x1.01 1.0037
FR_SAME = {"rel": 0.0035, "ratio": (0.997, 1.003), "row": 0.006, "stream": 0.005}  # ffn_residual vs the fp32 CPU
# residual of the same device inputs: the layer-4 component test's limits
FR_TERMS = ((0.998, 1.002), 0.025, (0.998, 1.002), 0.004)  # post coef, post rel, comb coef, comb rel (layer-4
# component test: an RNE bf16 output alone gives post rel 0.0112; comb x1.005 gives comb rel 0.0053)
SECOND_LAYER = 3  # dsa_moe: the same weightless collapse with pre col 3 live (dead at layer 4)


def _f32(t):
    return t.float() if torch.is_tensor(t) and t.is_floating_point() else t


def _flipped(router, want_router):
    """Per token: top-8 selection (nonzero routing weights) differs."""
    return ((router.float() != 0) != (want_router.float() != 0)).any(dim=-1)


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
            f" (<= {max_rel}) ratio=[{rmin:.4f}, {rmax:.4f}] (limit {same_lim})"
        )
        if not (same_lim[0] <= rmin and rmax <= same_lim[1]):
            fails.append(f"block out vs {tag}: same-routing per-row ratio [{rmin:.4f}, {rmax:.4f}] outside {same_lim}")
        if max_rel is not None and not rel <= max_rel:
            fails.append(f"block out vs {tag}: same-routing rel L2 {rel:.5f} > {max_rel}")
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
    msg = (
        f"{step} {tag}: rel_l2={rel:.5f} (<= {lim['rel']}) norm ratio [{lo:.4f}, {hi:.4f}] (in {lim['ratio']})"
        f" worst row {row:.5f} (<= {lim['row']})"
    )
    fails = []
    if "coef" in lim:  # <got, want> / <want, want>: a uniform scale error
        coef = ((got * w).sum() / (w * w).sum()).item()
        metrics.record(f"coef_swap_{step}_{tag}", coef)
        msg += f" coefficient {coef:.5f} (in {lim['coef']})"
        if not (lim["coef"][0] <= coef <= lim["coef"][1]):
            fails.append(f"{step} {tag} coefficient {coef:.5f} outside {lim['coef']}")
    if "block" in lim:  # every ATTN_BLOCK_ROWS-row block of the chunk
        br = ATTN_BLOCK_ROWS
        blocks = [
            ((got[i : i + br] - w[i : i + br]).norm() / w[i : i + br].norm().clamp_min(1e-30)).item()
            for i in range(0, w.shape[0], br)
        ]
        blk = max(blocks)
        at = blocks.index(blk) * br
        metrics.record(f"max_block_rel_l2_swap_{step}_{tag}", blk)
        msg += f" worst {br}-row block {blk:.5f} at rows {at}.. (<= {lim['block']})"
        print(f"{step} {tag} block rel_l2: " + " ".join(f"{b:.4f}" for b in blocks))
        if not blk <= lim["block"]:
            fails.append(f"{step} {tag} rows {at}..{at + br - 1} rel L2 {blk:.4f} > {lim['block']}")
    print(msg)
    if not rel <= lim["rel"]:  # NaN fails
        fails.append(f"{step} {tag} rel L2 {rel:.4f} > {lim['rel']}")
    if not (lim["ratio"][0] <= lo and hi <= lim["ratio"][1]):
        fails.append(f"{step} {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {lim['ratio']}")
    if not row <= lim["row"]:
        fails.append(f"{step} {tag} worst row rel L2 {row:.4f} > {lim['row']}")
    return fails


def _state_checks(tag, got_state, want_state, lim):
    if got_state is None:
        return [f"state {tag}: the device attention did not expose dctx.extra['state_out']"]
    fails = []
    for name in ("kda_recurrent", "kda_conv"):
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
        rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
        metrics.record(f"rel_l2_swap_state_{name}_{tag}", rel)
        msg = f"state {tag} {name}: rel_l2={rel:.5f} (<= {lim[name]})"
        if not rel <= lim[name]:
            fails.append(f"state {tag} {name} rel L2 {rel:.4f} > {lim[name]}")
        if name == "kda_recurrent":
            d = (got - want).flatten(1).norm(dim=1) / want.flatten(1).norm(dim=1).clamp_min(1e-12)
            head = d.max().item()
            metrics.record(f"max_head_rel_l2_swap_state_{name}_{tag}", head)
            msg += f" worst head {head:.5f} (head {d.argmax().item()}, <= {lim['head']})"
            if not head <= lim["head"]:
                fails.append(f"state {tag} {name} worst head rel L2 {head:.4f} > {lim['head']}")
        print(msg)
    return fails


def _res_checks(tag, got, want, lim, inputs=None, step="attn_residual", terms=None):
    """A residual's output vs want; with ``inputs`` (x, hc, y) also each term on its own. ``terms`` = (post coef, post
    rel, comb coef, comb rel), default attn_residual's."""
    fails = _out_checks(step, tag, got, want, lim)
    if any("elements" in m or "non-finite" in m for m in fails):
        return fails
    got, w = got.float().reshape(want.shape), want.float()
    h = w.shape[-1]
    if "stream" in lim:
        gs, ws = got.view(-1, N, h), w.view(-1, N, h)
        srel = [((gs[:, n] - ws[:, n]).norm() / ws[:, n].norm()).item() for n in range(N)]
        metrics.record(f"worst_stream_rel_l2_swap_{step}_{tag}", max(srel))
        print(f"{step} {tag}: per-stream rel L2 {[round(v, 5) for v in srel]} (<= {lim['stream']})")
        if not max(srel) <= lim["stream"]:
            fails.append(f"{step} {tag} per-stream rel L2 {max(srel):.4f} > {lim['stream']}")
    if inputs is not None:
        x, hc, y = (t.float() for t in inputs)
        post = hc[:, N : 2 * N]
        comb = hc[:, 2 * N :].reshape(-1, N, N)
        pterm = (post.unsqueeze(-1) * y.reshape(-1, h).unsqueeze(-2)).reshape(-1, h)
        cterm = torch.matmul(comb.transpose(-1, -2), x.reshape(-1, N, h)).reshape(-1, h)
        pc, pr, cc, cr = terms or (POST_COEF, POST_MAX_REL, COMB_COEF, COMB_MAX_REL)
        for name, term, other, (clo, chi), max_rel in (
            ("post_term", pterm, cterm, pc, pr),
            ("comb_term", cterm, pterm, cc, cr),
        ):
            d = got - other
            coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
            trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
            metrics.record(f"{name}_coef_swap_{step}_{tag}", coef)
            metrics.record(f"{name}_rel_l2_swap_{step}_{tag}", trel)
            print(f"{step} {tag}: {name} coef={coef:.5f} (in [{clo}, {chi}]) rel={trel:.5f} (<= {max_rel})")
            if not (clo <= coef <= chi):
                fails.append(f"{step} {tag} {name} coefficient {coef:.5f} outside [{clo}, {chi}]")
            if not trel <= max_rel:
                fails.append(f"{step} {tag} {name} rel L2 {trel:.4f} > {max_rel}")
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
    print(f"ffn_hc {tag}: comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
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


def _fc_checks(tag, got, want, lim):
    """ffn_in [S, H] vs want: rel L2, per-token norm ratio, worst row and the coefficient <got, want> / <want, want>."""
    return _out_checks("ffn_collapse", tag, got, want, lim)


def _fn_checks(tag, got, want, lim):
    """ffn_norm [S, H] vs want: rel L2, per-token ratio, worst row, coefficient <got, want> / <want, want>."""
    return _out_checks("ffn_norm", tag, got, want, lim)


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


def _moe_add_checks(tag, got, experts, shared):
    """mlp_out vs the fp32 sum of its own inputs (as the component test), and each addend on its own."""
    experts, shared = experts.float(), shared.float().reshape(experts.shape)
    fails = _out_checks("moe_add", tag, got, experts + shared, MA_SAME)
    if any("elements" in m or "non-finite" in m for m in fails):
        return fails
    got = got.float().reshape(experts.shape)
    for name, term, other, ((clo, chi), max_rel) in (
        ("experts", experts, shared, MA_EXPERTS),
        ("shared", shared, experts, MA_SHARED),
    ):
        d = got - other
        coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
        trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
        metrics.record(f"{name}_coef_swap_moe_add_{tag}", coef)
        metrics.record(f"{name}_rel_l2_swap_moe_add_{tag}", trel)
        print(f"moe_add {tag}: {name} coef={coef:.5f} (in [{clo}, {chi}]) rel={trel:.5f} (<= {max_rel})")
        if not (clo <= coef <= chi):
            fails.append(f"moe_add {tag} {name} coefficient {coef:.5f} outside [{clo}, {chi}]")
        if not trel <= max_rel:
            fails.append(f"moe_add {tag} {name} rel L2 {trel:.4f} > {max_rel}")
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


def _cpu_block(steps, ref, layer, g, c, x, overrides):
    """The CPU block of `x` (fresh state from the golden prefix) with some steps replaced; every boundary."""
    blk = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        x,
        rec=lambda n, t: blk.__setitem__(n, t),
        overrides=overrides,
    )
    return blk


def _share(tag, base, other, shape, max_flips, ratio, ratio_flipped, max_rel):
    """Block out of `base` vs a block that differs from it in a single step."""
    flips = _flipped(base["router"], other["router"])
    n = int(flips.sum())
    fails = []
    if not n <= max_flips:
        fails.append(f"{n} tokens route to another top-8 than the {tag} block (> {max_flips})")
    got, want = base["out"].float().reshape(shape), other["out"].float().reshape(shape)
    return fails + _row_checks(tag, got, want, flips, ratio, ratio_flipped, max_rel)


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
    # the KDA state the swapped attention left (read now: nothing below re-runs the device attention before chunk 0)
    mode = impl_mode()
    if mode == "device":
        got_state = dctx.extra.get("state_out")
    elif mode == "reference":
        got_state = ref.state_tensors(rctx.state, layer, g.seq)
    else:
        got_state = {k: torch.zeros_like(v) for k, v in g.state(layer, at=c * g.chunk + g.chunk).items()}
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
    hc_dev = seen["attn_hc"].float()
    in_dev = seen["attn_in"].float().reshape(gl["attn_in"].shape)
    norm_dev = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
    blk = (steps, ref, layer, g, c, gl["in"].float())
    shape = w.shape

    # Every device step's share: the all-CPU block of the same golden `in`.
    cpu = _cpu_block(*blk, {})
    failures += _share("cpu", seen, cpu, shape, CPU_MAX_FLIPS, CPU_ROW_RATIO, CPU_ROW_RATIO_FLIPPED, CPU_MAX_REL_L2)

    failures += _hc_checks(seen["attn_hc"], gl["attn_hc"])

    # attn_collapse vs golden (its attn_hc input is the device one), and vs the CPU collapse of the same inputs.
    failures += _out_checks("attn_collapse", "vs_golden", seen["attn_in"], gl["attn_in"], COL_GOLD)
    cpu_in = ref.component(layer, "attn_collapse")(rctx, gl["in"].float(), hc_dev)
    failures += _out_checks("attn_collapse", "vs_cpu_same_input", seen["attn_in"], cpu_in, COL_SAME)

    # attn_norm vs golden (its attn_in input is the device one), and vs the fp32 CPU norm of the same input.
    failures += _out_checks("attn_norm", "vs_golden", seen["attn_norm"], gl["attn_norm"], NORM_GOLD)
    cpu_norm = ref.component(layer, "attn_norm")(rctx, in_dev)
    failures += _out_checks("attn_norm", "vs_cpu_same_input", seen["attn_norm"], cpu_norm, NORM_SAME)

    # attention vs golden (its input is the device attn_norm), and vs the fp32 CPU KDA of the same input (fresh state
    # from the golden prefix); the post-chunk KDA state vs the golden snapshot and vs the CPU state of the same input.
    failures += _out_checks("attention", "vs_golden", seen["attn_out"], gl["attn_out"], ATTN_GOLD)
    actx = reference_ctx(ref, layer, g, c)
    cpu_attn = ref.component(layer, "attention")(actx, norm_dev)
    failures += _out_checks("attention", "vs_cpu_same_input", seen["attn_out"], cpu_attn, ATTN_SAME)
    _out_checks("attention", "cpu_vs_golden_info", cpu_attn, gl["attn_out"], ATTN_GOLD)  # upstream share, not asserted
    want_state = g.state(layer, at=c * g.chunk + g.chunk)
    failures += _state_checks("vs_golden", got_state, want_state, STATE_GOLD)
    failures += _state_checks("vs_cpu_same_input", got_state, ref.state_tensors(actx.state, layer, g.seq), STATE_SAME)

    # attn_residual vs golden h_mid (its attn_hc and attn_out inputs are the device ones), and vs the fp32 CPU
    # residual of the same inputs, with each term on its own.
    res_in = [gl["in"].float(), hc_dev, seen["attn_out"].float().reshape(gl["attn_out"].shape)]
    failures += _res_checks("vs_golden", seen["h_mid"], gl["h_mid"], RES_GOLD)
    cpu_res = ref.component(layer, "attn_residual")(reference_ctx(ref, layer, g, c), *res_in)
    failures += _res_checks("vs_cpu_same_input", seen["h_mid"], cpu_res, RES_SAME, res_in)
    _res_checks("cpu_vs_golden_info", cpu_res, gl["h_mid"], RES_GOLD)  # upstream share, not asserted

    # The attention share alone: the CPU block with the device attn_hc, attn_in and attn_norm and the CPU attention.
    # It is the base of the earlier steps' shares below, which keep the CPU attention downstream (a re-run device
    # attention would add its own error to every share).
    fixed = {
        "attn_hc": lambda ctx, x: hc_dev,
        "attn_collapse": lambda ctx, x, h: in_dev,
        "attn_norm": lambda ctx, x: norm_dev,
    }
    ashare = _cpu_block(*blk, fixed)
    # The residual-share block: the device outputs through attn_out fixed, the CPU residual. Block out vs it is the
    # residual's share; it vs the attention-share block is the attention's share (it reproduces swap 04's block out).
    ao_dev = seen["attn_out"].float().reshape(gl["attn_out"].shape)
    rshare = _cpu_block(*blk, {**fixed, "attention": lambda ctx, x: ao_dev})
    # The ffn_hc-share block: the device outputs through h_mid fixed, the CPU ffn_hc. Block out vs it is the ffn_hc's
    # share; it vs the residual-share block is the residual's share (it reproduces swap 05's block out).
    hm_dev = seen["h_mid"].float().reshape(gl["h_mid"].shape)
    hshare = _cpu_block(
        *blk, {**fixed, "attention": lambda ctx, x: ao_dev, "attn_residual": lambda ctx, x, h, y: hm_dev}
    )
    # The collapse-share block: the device outputs through ffn_hc fixed, the CPU ffn_collapse. Block out vs it is the
    # collapse's share; it vs the ffn_hc-share block is the ffn_hc's share (it reproduces swap 06's block out).
    fhc_dev = seen["ffn_hc"].float().reshape(gl["ffn_hc"].shape)
    fshare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
        },
    )
    # The norm-share block: the device outputs through ffn_in fixed, the CPU ffn_norm. Block out vs it is the norm's
    # share; it vs the collapse-share block is the collapse's share (it reproduces swap 07's block out).
    fin_dev = seen["ffn_in"].float().reshape(gl["ffn_in"].shape)
    nfshare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
        },
    )
    # The router-share block: the device outputs through ffn_norm fixed, the CPU router. Block out vs it is the
    # router's share; it vs the norm-share block is the norm's share (it reproduces swap 08's block out).
    fn_dev = seen["ffn_norm"].float().reshape(gl["ffn_norm"].shape)
    rtshare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
        },
    )
    # The experts-share block: the device outputs through router fixed, the CPU experts. Block out vs it is the
    # experts' share (the routing is the same by construction); it vs the router-share block is the router's share
    # (it reproduces swap 09's block out).
    rt_dev = seen["router"].float().reshape(gl["router"].shape)
    exshare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
            "router": lambda ctx, x: rt_dev,
        },
    )
    # The shared-share block: the device outputs through experts fixed, the CPU shared_expert. Block out vs it is the
    # shared expert's share (same routing by construction); it vs the experts-share block is the experts' share (it
    # reproduces swap 10's block out).
    ex_dev = seen["experts_out"].float().reshape(gl["experts_out"].shape)
    seshare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
            "router": lambda ctx, x: rt_dev,
            "experts": lambda ctx, x, r: ex_dev,
        },
    )
    # The add-share block: the device outputs through shared_expert fixed, the CPU moe_add. Block out vs it is the
    # add's share (same routing by construction); it vs the shared-share block is the shared expert's share (it
    # reproduces swap 11's block out).
    se_dev = seen["shared_out"].float().reshape(gl["shared_out"].shape)
    mashare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
            "router": lambda ctx, x: rt_dev,
            "experts": lambda ctx, x, r: ex_dev,
            "shared_expert": lambda ctx, x: se_dev,
        },
    )
    # The residual-share block: the device outputs through mlp_out fixed, the CPU ffn_residual. Its out is the fp32
    # CPU residual of the device (h_mid, ffn_hc, mlp_out): block out vs it is the residual's share (the same-input
    # check below); it vs the add-share block is the add's share (it reproduces swap 12's block out).
    ma_dev = seen["mlp_out"].float().reshape(gl["mlp_out"].shape)
    frshare = _cpu_block(
        *blk,
        {
            **fixed,
            "attention": lambda ctx, x: ao_dev,
            "attn_residual": lambda ctx, x, h, y: hm_dev,
            "ffn_hc": lambda ctx, x: fhc_dev,
            "ffn_collapse": lambda ctx, x, h: fin_dev,
            "ffn_norm": lambda ctx, x: fn_dev,
            "router": lambda ctx, x: rt_dev,
            "experts": lambda ctx, x, r: ex_dev,
            "shared_expert": lambda ctx, x: se_dev,
            "moe_add": lambda ctx, e, sh: ma_dev,
        },
    )
    fr_flips = _flipped(seen["router"], frshare["router"])
    ma_flips = _flipped(frshare["router"], mashare["router"])
    se_flips = _flipped(mashare["router"], seshare["router"])
    ex_flips = _flipped(seshare["router"], exshare["router"])
    if fr_flips.any() or ma_flips.any() or se_flips.any() or ex_flips.any():
        failures.append(
            f"share blocks route {int(fr_flips.sum())} / {int(ma_flips.sum())} / {int(se_flips.sum())} /"
            f" {int(ex_flips.sum())} tokens differently (harness bug)"
        )
    frshare_out = frshare["out"].float().reshape(shape)
    mashare_out = mashare["out"].float().reshape(shape)
    seshare_out = seshare["out"].float().reshape(shape)
    exshare_out = exshare["out"].float().reshape(shape)
    failures += _row_checks("ma_share", frshare_out, mashare_out, ma_flips, MA_SHARE_RATIO, None, MA_SHARE_MAX_REL_L2)

    # ffn_residual (block out) vs the fp32 CPU residual of the device inputs it got, with each term on its own.
    fr_in = [hm_dev, fhc_dev, ma_dev]
    fr = "ffn_residual"
    failures += _res_checks("vs_cpu_same_input", seen["out"], frshare_out, FR_SAME, fr_in, step=fr, terms=FR_TERMS)
    failures += _row_checks("se_share", mashare_out, seshare_out, se_flips, SE_SHARE_RATIO, None, SE_SHARE_MAX_REL_L2)
    failures += _row_checks("ex_share", seshare_out, exshare_out, ex_flips, EX_SHARE_RATIO, None, EX_SHARE_MAX_REL_L2)

    # shared_expert vs the fp32 CPU shared expert of the same ffn_norm, and vs golden (device ffn_norm in).
    se = "shared_expert"
    failures += _experts_checks("vs_cpu_same_input", seen["shared_out"], seshare["shared_out"], SE_SAME, step=se)
    failures += _experts_checks("vs_golden", seen["shared_out"], gl["shared_out"], SE_GOLD, step=se)
    _experts_checks("cpu_vs_golden_info", seshare["shared_out"], gl["shared_out"], SE_GOLD, step=se)  # not asserted

    # moe_add vs the fp32 sum of the same device experts_out and shared_out, and vs golden on the rows whose top-8 is
    # the golden's (a flipped row changes a whole expert: those rows get only a loose norm-ratio check).
    g_flips = _flipped(seen["router"], gl["router"])
    failures += _moe_add_checks("vs_cpu_same_input", seen["mlp_out"], ex_dev, se_dev)
    ma = "moe_add"
    failures += _experts_checks("vs_golden", seen["mlp_out"], gl["mlp_out"], MA_GOLD, ~g_flips, step=ma)
    _experts_checks("cpu_vs_golden_info", mashare["mlp_out"], gl["mlp_out"], MA_GOLD, ~g_flips, step=ma)  # not asserted
    if g_flips.any():
        mg = seen["mlp_out"].float().reshape(gl["mlp_out"].shape)[g_flips]
        mw = gl["mlp_out"].float()[g_flips]
        fr = mg.norm(dim=-1) / mw.norm(dim=-1).clamp_min(1e-12)
        lo, hi = fr.min().item(), fr.max().item()
        metrics.record("flipped_norm_ratio_min_swap_moe_add_vs_golden", lo)
        metrics.record("flipped_norm_ratio_max_swap_moe_add_vs_golden", hi)
        print(
            f"moe_add vs_golden: {int(g_flips.sum())} flipped rows ratio [{lo:.4f}, {hi:.4f}]"
            f" (in {MA_GOLD_FLIPPED_RATIO})"
        )
        if not (MA_GOLD_FLIPPED_RATIO[0] <= lo and hi <= MA_GOLD_FLIPPED_RATIO[1]):
            failures.append(f"moe_add vs_golden flipped-row ratio [{lo:.4f}, {hi:.4f}] outside {MA_GOLD_FLIPPED_RATIO}")

    # experts vs the fp32 CPU experts of the same (ffn_norm, router), and vs golden on the rows whose top-8 is the
    # golden's (the router's flips each move a whole expert: those rows get only a loose norm-ratio check).
    failures += _experts_checks("vs_cpu_same_input", seen["experts_out"], exshare["experts_out"], EX_SAME)
    failures += _experts_checks("vs_golden", seen["experts_out"], gl["experts_out"], EX_GOLD, ~g_flips)
    _experts_checks("cpu_vs_golden_info", exshare["experts_out"], gl["experts_out"], EX_GOLD, ~g_flips)  # not asserted
    if g_flips.any():
        eg = seen["experts_out"].float().reshape(gl["experts_out"].shape)[g_flips]
        ew = gl["experts_out"].float()[g_flips]
        fr = eg.norm(dim=-1) / ew.norm(dim=-1).clamp_min(1e-12)
        lo, hi = fr.min().item(), fr.max().item()
        metrics.record("flipped_norm_ratio_min_swap_experts_vs_golden", lo)
        metrics.record("flipped_norm_ratio_max_swap_experts_vs_golden", hi)
        print(
            f"experts vs_golden: {int(g_flips.sum())} flipped rows ratio [{lo:.4f}, {hi:.4f}] (in {EX_GOLD_FLIPPED_RATIO})"
        )
        if not (EX_GOLD_FLIPPED_RATIO[0] <= lo and hi <= EX_GOLD_FLIPPED_RATIO[1]):
            failures.append(f"experts vs_golden flipped-row ratio [{lo:.4f}, {hi:.4f}] outside {EX_GOLD_FLIPPED_RATIO}")

    failures += _share(
        "rt_share",
        exshare,
        rtshare,
        shape,
        RT_SHARE_MAX_FLIPS,
        RT_SHARE_RATIO,
        RT_SHARE_FLIPPED_RATIO,
        RT_SHARE_MAX_REL_L2,
    )
    failures += _share(
        "fn_share",
        rtshare,
        nfshare,
        shape,
        FN_SHARE_MAX_FLIPS,
        FN_SHARE_RATIO,
        FN_SHARE_FLIPPED_RATIO,
        FN_SHARE_MAX_REL_L2,
    )
    failures += _share(
        "fc_share",
        nfshare,
        fshare,
        shape,
        FC_SHARE_MAX_FLIPS,
        FC_SHARE_RATIO,
        FC_SHARE_RATIO_FLIPPED,
        FC_SHARE_MAX_REL_L2,
    )
    failures += _share(
        "fhc_share",
        fshare,
        hshare,
        shape,
        FHC_SHARE_MAX_FLIPS,
        FHC_SHARE_RATIO,
        FHC_SHARE_RATIO_FLIPPED,
        FHC_SHARE_MAX_REL_L2,
    )
    failures += _share(
        "res_share", hshare, rshare, shape, RES_SHARE_MAX_FLIPS, RES_SHARE_RATIO, None, RES_SHARE_MAX_REL_L2
    )

    # ffn_hc vs golden (its h_mid input is the device one), and vs the fp32 CPU ffn_hc of the same h_mid.
    failures += _fhc_checks("vs_golden", seen["ffn_hc"], gl["ffn_hc"], FHC_GOLD)
    cpu_fhc = ref.component(layer, "ffn_hc")(reference_ctx(ref, layer, g, c), hm_dev)
    failures += _fhc_checks("vs_cpu_same_input", seen["ffn_hc"], cpu_fhc, FHC_SAME)
    _fhc_checks("cpu_vs_golden_info", cpu_fhc, gl["ffn_hc"], FHC_GOLD)  # upstream share, not asserted

    # ffn_collapse vs golden (its h_mid and ffn_hc inputs are the device ones), and vs the fp32 CPU collapse of the
    # same inputs.
    failures += _fc_checks("vs_golden", seen["ffn_in"], gl["ffn_in"], FC_GOLD)
    cpu_fc = ref.component(layer, "ffn_collapse")(reference_ctx(ref, layer, g, c), hm_dev, fhc_dev)
    failures += _fc_checks("vs_cpu_same_input", seen["ffn_in"], cpu_fc, FC_SAME)
    _fc_checks("cpu_vs_golden_info", cpu_fc, gl["ffn_in"], FC_GOLD)  # upstream share, not asserted

    # ffn_norm vs golden (its ffn_in input is the device one), and vs the fp32 CPU norm of the same ffn_in.
    failures += _fn_checks("vs_golden", seen["ffn_norm"], gl["ffn_norm"], FN_GOLD)
    cpu_fn = ref.component(layer, "ffn_norm")(reference_ctx(ref, layer, g, c), fin_dev)
    failures += _fn_checks("vs_cpu_same_input", seen["ffn_norm"], cpu_fn, FN_SAME)
    _fn_checks("cpu_vs_golden_info", cpu_fn, gl["ffn_norm"], FN_GOLD)  # upstream share, not asserted

    # router vs golden (its ffn_norm input is the device one), and vs the fp32 CPU router of the same ffn_norm.
    failures += _router_checks("vs_golden", seen["router"], gl["router"], RT_GOLD)
    failures += _router_checks("vs_cpu_same_input", seen["router"], rtshare["router"], RT_SAME)
    _router_checks("cpu_vs_golden_info", rtshare["router"], gl["router"], RT_GOLD)  # upstream share, not asserted

    # The same weightless collapse on layer 3's golden, where pre col 3 carries weight (dead at layer 4).
    gl3 = g.layer(c, SECOND_LAYER)
    fc3 = muts["ffn_collapse"](
        reference_ctx(ref, layer, g, c), device_ctx(SECOND_LAYER, g, c), gl3["h_mid"].float(), gl3["ffn_hc"].float()
    )
    cpu_fc3 = ref.component(layer, "ffn_collapse")(
        reference_ctx(ref, layer, g, c), gl3["h_mid"].float(), gl3["ffn_hc"].float()
    )
    failures += _fc_checks(f"L{SECOND_LAYER:02d}_vs_cpu_same_input", fc3, cpu_fc3, FC_SAME)
    failures += _fc_checks(f"L{SECOND_LAYER:02d}_vs_golden", fc3, gl3["ffn_in"], FC_SAME)
    failures += _share(
        "attn_share",
        rshare,
        ashare,
        shape,
        ATTN_SHARE_MAX_FLIPS,
        ATTN_SHARE_RATIO,
        ATTN_SHARE_RATIO_FLIPPED,
        ATTN_SHARE_MAX_REL_L2,
    )

    # The attn_collapse share alone: the device attn_hc, the CPU collapse, the device norm, the CPU attention.
    dev_norm = lambda ctx, x: _f32(muts["attn_norm"](ctx, dctx, x))  # noqa: E731
    share = _cpu_block(*blk, {"attn_hc": fixed["attn_hc"], "attn_norm": dev_norm})
    failures += _share(
        "collapse_share", ashare, share, shape, SHARE_MAX_FLIPS, SHARE_RATIO, SHARE_RATIO_FLIPPED, SHARE_MAX_REL_L2
    )

    # The attn_norm share alone: the device attn_hc and attn_in, the CPU norm, the CPU attention.
    nshare = _cpu_block(*blk, {"attn_hc": fixed["attn_hc"], "attn_collapse": fixed["attn_collapse"]})
    failures += _share(
        "norm_share",
        ashare,
        nshare,
        shape,
        NORM_SHARE_MAX_FLIPS,
        NORM_SHARE_RATIO,
        NORM_SHARE_RATIO_FLIPPED,
        NORM_SHARE_MAX_REL_L2,
    )

    # Chunk 0 (start 0: the KDA must start from a zero state, not the state the chunk-1 run left): the device
    # attention vs the CPU KDA of the same device attn_norm.
    if c > 0 and 0 in g.dumped_chunks:
        gl0 = g.layer(0, layer)
        rctx0, dctx0 = reference_ctx(ref, layer, g, 0), device_ctx(layer, g, 0)
        x0 = gl0["in"].float()
        hc0 = _f32(muts["attn_hc"](rctx0, dctx0, x0))
        in0 = _f32(muts["attn_collapse"](rctx0, dctx0, x0, hc0)).reshape(gl0["attn_in"].shape)
        n0 = _f32(muts["attn_norm"](rctx0, dctx0, in0)).reshape(gl0["attn_norm"].shape)
        a0 = muts["attention"](rctx0, dctx0, n0)
        cpu_a0 = ref.component(layer, "attention")(reference_ctx(ref, layer, g, 0), n0)
        failures += _out_checks("attention", "c0_vs_cpu_same_input", a0, cpu_a0, C0_ATTN_SAME)
        failures += _out_checks("attention", "c0_vs_golden", a0, gl0["attn_out"], ATTN_GOLD)
        a0 = _f32(a0).reshape(gl0["attn_out"].shape)
        h0 = muts["attn_residual"](rctx0, dctx0, x0, hc0, a0)
        res_in0 = [x0, hc0.reshape(gl0["attn_hc"].shape), a0]
        cpu_h0 = ref.component(layer, "attn_residual")(reference_ctx(ref, layer, g, 0), *res_in0)
        failures += _res_checks("c0_vs_cpu_same_input", h0, cpu_h0, RES_SAME, res_in0)
        h0 = _f32(h0).reshape(gl0["h_mid"].shape)
        f0 = muts["ffn_hc"](rctx0, dctx0, h0)
        cpu_f0 = ref.component(layer, "ffn_hc")(reference_ctx(ref, layer, g, 0), h0)
        failures += _fhc_checks("c0_vs_cpu_same_input", f0, cpu_f0, FHC_SAME)
        f0 = _f32(f0).reshape(gl0["ffn_hc"].shape)
        fc0 = muts["ffn_collapse"](rctx0, dctx0, h0, f0)
        cpu_fc0 = ref.component(layer, "ffn_collapse")(reference_ctx(ref, layer, g, 0), h0, f0)
        failures += _fc_checks("c0_vs_cpu_same_input", fc0, cpu_fc0, FC_SAME)
        fc0 = _f32(fc0).reshape(gl0["ffn_in"].shape)
        fn0 = muts["ffn_norm"](rctx0, dctx0, fc0)
        cpu_fn0 = ref.component(layer, "ffn_norm")(reference_ctx(ref, layer, g, 0), fc0)
        failures += _fn_checks("c0_vs_cpu_same_input", fn0, cpu_fn0, FN_SAME)
        fn0 = _f32(fn0).reshape(gl0["ffn_norm"].shape)
        rt0 = muts["router"](rctx0, dctx0, fn0)
        cpu_rt0 = ref.component(layer, "router")(reference_ctx(ref, layer, g, 0), fn0)
        failures += _router_checks("c0_vs_cpu_same_input", rt0, cpu_rt0, RT_SAME)
        failures += _router_checks("c0_vs_golden", rt0, gl0["router"], RT_GOLD)
        rt0 = _f32(rt0).reshape(gl0["router"].shape)
        ex0 = muts["experts"](rctx0, dctx0, fn0, rt0)
        cpu_ex0 = ref.component(layer, "experts")(reference_ctx(ref, layer, g, 0), fn0, rt0)
        failures += _experts_checks("c0_vs_cpu_same_input", ex0, cpu_ex0, EX_SAME)
        se0 = muts["shared_expert"](rctx0, dctx0, fn0)
        cpu_se0 = ref.component(layer, "shared_expert")(reference_ctx(ref, layer, g, 0), fn0)
        failures += _experts_checks("c0_vs_cpu_same_input", se0, cpu_se0, SE_SAME, step="shared_expert")
        ex0 = _f32(ex0).reshape(gl0["experts_out"].shape)
        se0 = _f32(se0).reshape(gl0["shared_out"].shape)
        ma0 = muts["moe_add"](rctx0, dctx0, ex0, se0)
        failures += _moe_add_checks("c0_vs_cpu_same_input", ma0, ex0, se0)
        ma0 = _f32(ma0).reshape(gl0["mlp_out"].shape)
        fr0 = muts["ffn_residual"](rctx0, dctx0, h0, f0, ma0)
        fr_in0 = [h0, f0, ma0]
        cpu_fr0 = ref.component(layer, "ffn_residual")(reference_ctx(ref, layer, g, 0), *fr_in0)
        failures += _res_checks(
            "c0_vs_cpu_same_input", fr0, cpu_fr0, FR_SAME, fr_in0, step="ffn_residual", terms=FR_TERMS
        )
    assert not failures, "; ".join(failures)
