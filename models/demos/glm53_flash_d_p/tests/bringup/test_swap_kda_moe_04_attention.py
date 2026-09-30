# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 4: block type kda_moe (layer 4) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 4 (kda_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    attention

Reviewed (S.kda_moe.04.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). attention is KDA over the golden prefix state (kda_recurrent [64, 128, 128],
kda_conv [3, 24576]) on attn_norm [2048, 4096] -> attn_out; it enters block out as post * attn_out, then the MoE.
Block out hardly sees it: every KDA state bug in the attention component review passes PCC, and here (CPU runs,
golden s4096 chunk 1, the CPU attention output perturbed, device-free; host script /tmp/kmoe04/sens.py, not kept)
block out moves little. Columns: attn_out rel / per-token ratio / worst row / worst 128-row block / coefficient vs
the unperturbed CPU attention, then block out tokens with another top-8 / same-routing rel L2 / ratio / flipped rows:
  bf16 0.0017, 13 / 0.00024; 0.5% noise 0.0050, 41 / 0.00072 / [0.9989, 1.0012]; 1% 0.010, 69 / 0.0014 /
  [0.998, 1.003] / [0.992, 1.036]; x0.996 27 / 0.00049; x0.99 77 / 0.0012 / [0.9907, 1.0015]; x1.005 27 / 0.00062;
  x1.01 51 / 0.0012; x1.02 115 / 0.0024; per-row scale noise 0.5% ratio [0.985, 1.016], 19 / 0.00058 / [0.990,
  1.011]; rows 0..127 zeroed 0.25 / block 1.0, 128 flips; last row zeroed 0.0105 / row 1.0 / block 0.042, 1 flip,
  same-routing unchanged; row 100 x1.05 row 0.050, 0 flips / 0.00017; head 7 zeroed 0.125 / coef 0.984, 762 flips;
  recurrent prefix zeroed 0.064 / [0.52, 1.13] / row 0.98 / block 0.20, 341 / 0.0054; conv tail zeroed 0.012 /
  [0.934, 1.0005] / row 0.43 / block 0.047, 14 / 0.0011 / [0.977, 1.016] / [0.975, 1.136]; recurrent x0.95 0.0032 /
  [0.961, 1.0008] / row 0.049, 28 / 0.00058 / [0.985, 1.002]; recurrent transposed 0.073 / row 1.17, 378 / 0.0061.
So the test checks the KDA output and its post-chunk state directly, and re-bases the earlier shares on the
attention share (as dsa_moe swap 06), so their swap 03 limits still hold.
Extra asserted checks (informational metrics, not in the runner's list; every limit is written `not x <= lim`, so a
NaN metric fails):
  - block out vs golden (rel L2 <= 0.01; per-row ratio [0.985, 1.015] on rows with the golden's top-8, [0.9, 1.1] on
    flipped rows) and vs the all-CPU block of the same golden `in` (all four device steps' share: flips <= 96,
    same-routing rel L2 <= 0.003, ratio [0.985, 1.015], flipped rows [0.9, 1.1]);
  - attn_hc, attn_collapse, attn_norm vs golden and vs the fp32 CPU step of their own inputs: swap 03's limits;
  - attention vs the fp32 CPU KDA of the same device attn_norm (fresh state from the golden prefix): rel <= 0.012,
    ratio [0.98, 1.004], worst row <= 0.05, every 128-row block <= 0.012, coefficient [0.99, 1.002] (fails every state
    bug above, zeroed rows, row 100 x1.05, a dropped head, x0.99 and x1.005 on top of the device's own -0.4%);
  - attention vs golden (device inputs, upstream error included): rel <= 0.015, ratio [0.975, 1.006], worst row
    <= 0.06, block <= 0.015, coefficient [0.985, 1.004];
  - KDA state after the chunk (``dctx.extra["state_out"]`` on the device, the CPU state in reference mode, zeros for
    the stub) vs the golden snapshot at start + chunk (recurrent rel <= 0.025, worst head <= 0.05, conv <= 0.008) and
    vs the CPU state of the same attn_norm (recurrent <= 0.025, worst head <= 0.05, conv <= 0.005). The worst head is
    kept at the component test's 0.05: the device's head 20 (one slow-decay key row, see the component review) sits
    at 0.047..0.048;
  - the attention share: block out vs the CPU block with the device attn_hc, attn_in and attn_norm and the CPU
    attention: flips <= 96, same-routing rel L2 <= 0.002, ratio [0.99, 1.01], flipped rows [0.95, 1.06] (fails x1.02,
    recurrent zeroed or transposed, conv zeroed, recurrent x0.95, a dropped head);
  - the collapse and norm shares, now vs the attention-share block (CPU attention downstream): swap 03's limits;
  - chunk 0 (start 0: the KDA must start from a zero state, not the one the chunk-1 run left): the device
    hc/collapse/norm/attention on the chunk-0 golden `in`, attention vs the CPU KDA of the same attn_norm (limits as
    chunk 1) and vs golden (as chunk 1).
Device (first run): PCC 0.999993, rel 0.0039, 58 flips vs golden; vs the all-CPU block 52 flips / 0.00143 /
[0.9926, 1.0016]; attention vs CPU same input 0.0071 / [0.9883, 1.0001] / worst row 0.031 / block 0.0074 / coef
0.9962 (chunk 0 0.0069 / [0.9924, 1.0002] / 0.0135 / 0.0071 / 0.9962), vs golden 0.0085 / [0.9870, 0.9992] / 0.032
/ 0.0089 / 0.9947 (the CPU KDA of the device norm scores 0.0032 / coef 0.9985); state vs golden recurrent 0.0170 /
worst head 0.048 / conv 0.0030, vs CPU 0.0166 / 0.047 / 0.0017; attention share 58 flips / 0.00103 / [0.9951, 1.0017]
/ flipped [0.9887, 1.0280]; collapse share 0 flips / 0.0; norm share 8 / 0.00016 / [0.9995, 1.0004]; earlier steps as
swap 03. About 85 s. Reference: every same-input and share check exact, PCC 0.999997. Stub: fails PCC and every check.
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
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "attention"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.985, 1.015)  # per-row ||got|| / ||want||, rows whose top-8 equals the golden's
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 96  # rows whose top-8 differs from the CPU block of the same `in`
CPU_MAX_REL_L2 = 0.003  # block out vs CPU block, same-routing rows
CPU_ROW_RATIO = (0.985, 1.015)
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

    # All four device steps' share: the all-CPU block of the same golden `in`.
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

    # The attention share alone: the CPU block with the device attn_hc, attn_in and attn_norm and the CPU attention.
    # It is the base of the earlier steps' shares below, which keep the CPU attention downstream (a re-run device
    # attention would add its own error to every share).
    fixed = {
        "attn_hc": lambda ctx, x: hc_dev,
        "attn_collapse": lambda ctx, x, h: in_dev,
        "attn_norm": lambda ctx, x: norm_dev,
    }
    ashare = _cpu_block(*blk, fixed)
    failures += _share(
        "attn_share",
        seen,
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
    assert not failures, "; ".join(failures)
