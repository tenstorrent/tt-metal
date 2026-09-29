# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 5: block type moe_shared (layer 2) with topk_shared swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 2 (moe_shared) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    topk_shared

Reviewed (S.moe_shared.05.test.1). Built from test_swap_moe_shared_04_q_a.py (layer-2 setup: ctx.extra
["shared_topk"] = the golden's L{src}.topk, src = cfg.topk_source(2) = 1, int64 [2048, 2048], in both contexts;
every attn_hc / attn_hc_pre / attn_norm / q_a check and limit of swap 04 unchanged) plus the topk checks of
test_c_moe_shared_topk_shared.py. The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). topk_shared is an identity (components.yaml NATIVE, no op): it hands on the
source full layer's top-k for this chunk; its graph input attn_norm is not used. Golden L2.topk == L1.topk exactly.
Measured on the CPU (golden s4096 chunk 1, 2048 rows; topk_shared replaced by mutations of the shared top-k, every
other step the fp32 CPU reference; script /tmp/hy4_ssh5/study.py, 9 s per variant). "ov" = mean per-row set overlap
vs golden, "ao" = attn_out rel / worst row, "hm" = h_mid rel / max per-stream rel / worst (row, stream), "rt" =
router top-8 overlap, "out" = PCC / rel L2 / worst (row, stream):

    variant                   ov     | ao             | hm                       | rt     | out
    reference (identity)      1.0000 | 0.0019 / 0.002 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 1.000000 0.0030 0.024
    shuffled within rows      1.0000 | 0.0019 / 0.002 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 1.000000 0.0030 0.024 (harmless)
    sorted descending         1.0000 | 0.0019 / 0.002 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 1.000000 0.0030 0.024 (harmless)
    last tile row = row -33   0.9975 | 0.0044 / 0.065 | 0.0010 / 0.0038 / 0.062  | 0.9970 | 1.000000 0.0032 0.064 (passes)
    last entry dropped        0.9995 | 0.0084 / 0.074 | 0.0013 / 0.0055 / 0.059  | 0.9926 | 1.000000 0.0054 0.062 (passes)
    rows shifted by 1         0.9254 | 0.0107 / 0.074 | 0.0016 / 0.0076 / 0.076  | 0.9930 | 1.000000 0.0055 0.101 (passes)
    sliding window 2048       0.7706 | 0.0166 / 0.047 | 0.0027 / 0.0135 / 0.047  | 0.9842 | 0.999982 0.0119 0.132 (passes)
    first 2048 positions      0.5937 | 0.0629 / 0.230 | 0.0074 / 0.0384 / 0.212  | 0.9531 | 0.999926 0.0157 0.216 (passes)
    half the keys dropped     0.5000 | 0.0738 / 0.318 | 0.0091 / 0.0473 / 0.247  | 0.9438 | 0.999884 0.0185 0.248 (passes)
    SP row halves swapped     0.5459 | 0.0507 / 0.218 | 0.0063 / 0.0327 / 0.180  | 0.9620 | 0.999846 0.0200 0.414 (passes)
    chunk-0 top-k             0.2294 | 0.1429 / 0.870 | 0.0197 / 0.1017 / 0.904  | 0.9113 | 0.999512 0.0337 0.907 (passes)
    zero top-k (all 0)        0.0005 | 0.4781 / 0.617 | 0.0917 / 0.4806 / 0.614  | 0.6728 | 0.993458 0.1260 0.619 (passes)

"(passes)" = passes the 0.98 out gate: every topk mutation does, even an all-zero top-k (the residual dominates block
out; the BRINGUP_IMPL=stub run still fails the gate because it zeroes the iHC steps too). The swap 04 attn_out /
h_mid limits catch only some, through attn_out's worst row (last entry dropped 0.074, a copied last tile row
0.065). An identity has no precision loss, so the test asserts the topk exactly (informational metrics, asserted):
  - swap 04's checks, unchanged: attn_hc, attn_x (vs golden, vs the CPU hc_pre, rotated pre gates), attn_norm (vs
    golden, vs CPU, x 0.1 eps run), q_resid (vs golden, vs CPU, x 0.01 eps run), attn_out rel <= 0.01 / worst row
    <= 0.05, h_mid rel / per-stream <= 0.005, worst (row, stream) <= 0.02, router overlap >= 0.98, out rel <= 0.01;
  - topk (the swapped step): integer, S x 2048 elements, pads (-1 or 0xFFFFFFFF) normalized to -1; every row's set
    equals the golden L2.topk row's set and the shared input's row set (no repeats, no extra or missing positions,
    no position after its query row); pads a contiguous tail and >= 1 valid key per row (sparse_sdpa's
    preconditions). The order within a row is free (shuffled / reversed rows leave attn_out bit-identical);
  - the topk_shared module again on chunk 0 (start 0, golden chunk-0 attn_norm, shared_topk = the golden's chunk-0
    L1.topk, 2047 padded rows) must return exactly the golden chunk-0 sets, so a module that caches its first result
    or ignores ctx.extra["shared_topk"] fails.
The trail line pcc_swap_topk (positional match) is diagnostic only (known issue: the swap trail scores top-k indices
by positional match). The out worst (row, stream) rel L2 is recorded, not asserted (near-tie expert flips).
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
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
BLOCK_TYPE = "moe_shared"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "topk_shared"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
HC_MULT = 4
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs golden
GATES_MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
X_MAX_REL_L2 = 0.005  # attn_x vs golden, whole tensor
X_RATIO = (0.996, 1.004)  # attn_x per-row ||got|| / ||want|| vs golden
X_MAX_ROW_REL = 0.01  # attn_x vs golden, worst row
CPU_MAX_REL_L2 = 0.003  # attn_x vs the CPU hc_pre on the same inputs
CPU_MAX_ROW_REL = 0.006
ROT_MAX_REL_L2 = 0.004  # rotated pre gates, vs the CPU hc_pre on the same inputs
ROT_MAX_ROW_REL = 0.01
H_MID_MAX_REL = 0.005  # h_mid vs golden, whole tensor
H_MID_MAX_STREAM_REL = 0.005  # h_mid, rel L2 of each stream (stream 3 dominates the whole tensor at layer 2)
H_MID_MAX_ROW_REL = 0.02  # h_mid, worst (row, stream)
N_MAX_REL_L2 = 0.008  # attn_norm vs golden, and vs the CPU attn_norm on the same input
N_RATIO = (0.993, 1.007)  # attn_norm per-row norm ratio vs golden
N_MAX_ROW_REL = 0.015  # attn_norm worst row, vs golden and vs the CPU step on the same input
SYN_SCALE = 0.1  # eps check: the device attn_x x SYN_SCALE (bf16) through the module vs the CPU step
SYN_MAX_REL_L2 = 0.01
SYN_MAX_ROW_REL = 0.02
Q_MAX_REL_L2 = 0.008  # q_resid vs golden, and vs the CPU q_a on the same input
Q_RATIO = (0.994, 1.006)  # q_resid per-row norm ratio vs golden
Q_MAX_ROW_REL = 0.015  # q_resid worst row, vs golden and vs the CPU step on the same input
Q_SYN_SCALE = 0.01  # eps check: the device attn_norm x Q_SYN_SCALE (bf16) through the q_a module vs the CPU step
Q_SYN_MAX_REL_L2 = 0.01
Q_SYN_MAX_ROW_REL = 0.02
ATT_MAX_REL_L2 = 0.01  # attn_out, whole tensor
ATT_MAX_ROW_REL = 0.05  # attn_out, worst token row
ROUTER_MIN_OVERLAP = 0.98  # router top-8 selection overlap vs golden
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor
SENTINEL = 0xFFFFFFFF  # topk_large_indices' / sparse_sdpa's pad


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _worst_row_rel(got, want, streams=1):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)).max().item()


def _ratio(got, want):
    r = got.float().reshape(want.shape).norm(dim=-1) / want.float().norm(dim=-1).clamp_min(1e-12)
    return r.min().item(), r.max().item()


def _rotated(gates):
    """Each row's pre gates rotated by (row mod 4), so every stream meets the large gate on a quarter of the rows."""
    n = gates.shape[0]
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, :HC_MULT] = torch.gather(gates[:, :HC_MULT], 1, idx)
    return gs


def _stream_rel(got, want, streams):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=(0, 2)) / want.norm(dim=(0, 2)).clamp_min(1e-12)).tolist()


def _norm_topk(out, want):
    """int64 [S, k] with every pad (negative, or the uint32 sentinel) as -1; None if not integer / wrong size."""
    if out.is_floating_point() or out.numel() != want.numel():
        return None
    t = out.reshape(want.shape).to(torch.int64)
    return torch.where((t < 0) | (t == SENTINEL), torch.full_like(t, -1), t)


def _topk_check(got, want, start, tag):
    """Exact per-row set equality with want plus sparse_sdpa's pad layout; returns failure messages."""
    fails = []
    valid = got >= 0
    pos = torch.arange(start, start + got.shape[0])[:, None]
    noncausal = (valid & (got > pos)).sum().item()
    if noncausal:
        fails.append(f"{noncausal} selected positions are after their query row (non-causal)")
    srt, ws = got.sort(dim=-1).values, want.sort(dim=-1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item()
    if dup:
        fails.append(f"{dup} repeated positions within rows")
    bad = (srt != ws).any(-1).nonzero().flatten()
    if bad.numel():
        r = bad[0].item()
        fails.append(
            f"{bad.numel()} rows differ from the wanted set (first: row {r} at position {start + r}: "
            f"{valid[r].sum().item()} valid, want {(want[r] >= 0).sum().item()})"
        )
    tail_bad = (valid[:, 1:] & ~valid[:, :-1]).any(-1).sum().item()
    if tail_bad:
        fails.append(f"{tail_bad} rows have a valid position after a pad (sparse_sdpa needs a contiguous pad tail)")
    empty = (valid.sum(-1) == 0).sum().item()
    if empty:
        fails.append(f"{empty} rows have no valid key (sparse_sdpa needs >= 1)")
    rows = torch.tensor(
        [torch.isin(b[b >= 0], a[a >= 0]).float().mean().item() if (b >= 0).any() else 1.0 for a, b in zip(got, want)]
    )
    metrics.record(f"{tag}topk_overlap_swap", rows.mean().item())
    metrics.record(f"{tag}topk_worst_row_overlap_swap", rows.min().item())
    print(
        f"{tag}topk: mean set overlap={rows.mean().item():.6f} worst row={rows.min().item():.5f} (== 1) "
        f"differing rows={bad.numel()} pads={(~valid).sum().item()} (want {(want < 0).sum().item()}) "
        f"positional match={(got == want).float().mean().item():.4f} (not gated)"
    )
    return [f"{tag}topk: {m}" for m in fails]


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    # A shared layer reuses the latest full layer's top-k for this chunk; that layer does not run here, so take it
    # from the golden (the source layer's recorded topk, int64 [S, index_topk]).
    src = ref.cfg.topk_source(layer)
    shared_topk = g.layer(c, src)["topk"]
    assert src != layer and shared_topk.shape[0] == g.chunk, f"layer {src} topk {tuple(shared_topk.shape)}"
    rctx.extra["shared_topk"] = shared_topk
    dctx.extra["shared_topk"] = shared_topk
    overrides, muts = {}, {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
        assert not getattr(mut, "cpu_bridge", False), f"device_component returned a CPU bridge for {name}"
        muts[name] = mut
        overrides[name] = lambda ctx, *x, mut=mut: mut(ctx, dctx, *x)
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

    # Extra checks (see the module docstring). Recorded as informational metrics, asserted here.
    failures = [] if ok else [f"pcc_swap_out below {thr}"]

    def finite_shape(n):
        got, want = seen[n], gl[n]
        if got.numel() != want.numel() or got.shape[-1] != want.shape[-1]:
            failures.append(f"{n}: shape {tuple(got.shape)} vs golden {tuple(want.shape)}")
            return False
        if not torch.isfinite(got.float()).all():
            failures.append(f"{n}: non-finite")
            return False
        return True

    # attn_hc: the iHC gates [S, 8] (pre 0-3 | post 4-7), as swap 01.
    gates_ok = finite_shape("attn_hc")
    if gates_ok:
        want = gl["attn_hc"].float()
        got = seen["attn_hc"].float().reshape(want.shape)
        rel = _rel(got, want)
        col_rel = ((got - want).norm(dim=0) / want.norm(dim=0).clamp_min(1e-12)).tolist()
        post_row = _worst_row_rel(got[:, HC_MULT:], want[:, HC_MULT:])
        metrics.record("rel_l2_swap_attn_hc", rel)
        metrics.record("max_col_rel_l2_swap_attn_hc", max(col_rel))
        metrics.record("post_worst_row_rel_l2_swap_attn_hc", post_row)
        print(
            f"attn_hc: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) col rel (pre 0-3 | post 4-7)="
            f"{[round(v, 5) for v in col_rel]} (<= {GATES_MAX_COL_REL}) post worst row={post_row:.5f} "
            f"(<= {GATES_MAX_POST_ROW_REL})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"attn_hc: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_rel) if v > GATES_MAX_COL_REL]
        if bad:
            failures.append(f"attn_hc: rel L2 > {GATES_MAX_COL_REL} in columns {bad}")
        if post_row > GATES_MAX_POST_ROW_REL:
            failures.append(f"attn_hc: post worst row rel L2 {post_row:.5f} > {GATES_MAX_POST_ROW_REL}")

    # attn_x vs golden (swap 02).
    x_ok = finite_shape("attn_x")
    if x_ok:
        rel = _rel(seen["attn_x"], gl["attn_x"])
        rmin, rmax = _ratio(seen["attn_x"], gl["attn_x"])
        row = _worst_row_rel(seen["attn_x"], gl["attn_x"])
        metrics.record("rel_l2_swap_attn_x", rel)
        metrics.record("worst_row_rel_l2_swap_attn_x", row)
        print(
            f"attn_x vs golden: rel_l2={rel:.6f} (<= {X_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(X_RATIO)}) worst_row_rel_l2={row:.5f} (<= {X_MAX_ROW_REL})"
        )
        if rel > X_MAX_REL_L2:
            failures.append(f"attn_x: rel L2 {rel:.5f} > {X_MAX_REL_L2}")
        if not (X_RATIO[0] <= rmin and rmax <= X_RATIO[1]):
            failures.append(f"attn_x: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(X_RATIO)}")
        if row > X_MAX_ROW_REL:
            failures.append(f"attn_x: worst row rel L2 {row:.5f} > {X_MAX_ROW_REL}")

        if gates_ok:
            # vs the CPU hc_pre on the same inputs (block input + the gates the device produced).
            cpu = ref.component(layer, "attn_hc_pre")
            gates = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            same = cpu(rctx, gl["in"].float(), gates).float()
            crel, crow = _rel(seen["attn_x"], same), _worst_row_rel(seen["attn_x"], same)
            metrics.record("rel_l2_swap_attn_x_vs_cpu", crel)
            print(
                f"attn_x vs CPU hc_pre on the same inputs: rel_l2={crel:.6f} (<= {CPU_MAX_REL_L2}) "
                f"worst_row_rel_l2={crow:.5f} (<= {CPU_MAX_ROW_REL})"
            )
            if crel > CPU_MAX_REL_L2 or crow > CPU_MAX_ROW_REL:
                failures.append(f"attn_x vs CPU on the same inputs: rel {crel:.5f} / worst row {crow:.5f}")

            # Stream order: the module again with each row's pre gates rotated by (row mod 4).
            rot = (gl["in"].float(), _rotated(gates))
            rot_want = cpu(rctx, *rot).float()
            rot_out = muts["attn_hc_pre"](rctx, dctx, *rot)
            if rot_out.numel() != rot_want.numel() or not torch.isfinite(rot_out.float()).all():
                failures.append(f"attn_hc_pre rotated gates: shape {tuple(rot_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(rot_out, rot_want), _worst_row_rel(rot_out, rot_want)
                metrics.record("rot_rel_l2_swap_attn_x", yrel)
                print(
                    f"attn_hc_pre with rotated pre gates vs CPU: rel_l2={yrel:.6f} (<= {ROT_MAX_REL_L2}) "
                    f"worst_row_rel_l2={yrow:.5f} (<= {ROT_MAX_ROW_REL})"
                )
                if yrel > ROT_MAX_REL_L2 or yrow > ROT_MAX_ROW_REL:
                    failures.append(f"attn_hc_pre rotated gates: rel {yrel:.5f} / worst row {yrow:.5f} (stream order?)")

    def rel_row(tag, got, want, max_rel, max_row, ratio=None, what="vs golden"):
        rel, row = _rel(got, want), _worst_row_rel(got, want)
        metrics.record(f"rel_l2_swap_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_{tag}", row)
        msg = f"{tag} {what}: rel_l2={rel:.6f} (<= {max_rel})"
        if ratio is not None:
            rmin, rmax = _ratio(got, want)
            msg += f" row norm ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(ratio)})"
            if not (ratio[0] <= rmin and rmax <= ratio[1]):
                failures.append(f"{tag} {what}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if max_row is not None:
            msg += f" worst_row_rel_l2={row:.5f} (<= {max_row})"
            if row > max_row:
                failures.append(f"{tag} {what}: worst row rel L2 {row:.5f} > {max_row}")
        print(msg)
        if rel > max_rel:
            failures.append(f"{tag} {what}: rel L2 {rel:.5f} > {max_rel}")

    # attn_norm (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    n_ok = finite_shape("attn_norm")
    if n_ok:
        rel_row("attn_norm", seen["attn_norm"], gl["attn_norm"], N_MAX_REL_L2, N_MAX_ROW_REL, N_RATIO)
        if x_ok:
            cpu_n = ref.component(layer, "attn_norm")
            x = seen["attn_x"].float().reshape(gl["attn_x"].shape)
            rel_row(
                "attn_norm_vs_cpu",
                seen["attn_norm"],
                cpu_n(rctx, x).float(),
                N_MAX_REL_L2,
                N_MAX_ROW_REL,
                what="vs CPU attn_norm on the device attn_x",
            )
            xs = (x * SYN_SCALE).bfloat16().float()
            syn_want = cpu_n(rctx, xs).float()
            syn_out = muts["attn_norm"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"attn_norm scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(syn_out, syn_want), _worst_row_rel(syn_out, syn_want)
                ymin, ymax = _ratio(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_attn_norm", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_attn_norm", yrow)
                print(
                    f"attn_norm on attn_x x{SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {SYN_MAX_ROW_REL})"
                )
                if yrel > SYN_MAX_REL_L2 or yrow > SYN_MAX_ROW_REL:
                    failures.append(f"attn_norm scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # q_resid (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    if finite_shape("q_resid"):
        rel_row("q_resid", seen["q_resid"], gl["q_resid"], Q_MAX_REL_L2, Q_MAX_ROW_REL, Q_RATIO)
        if n_ok:
            cpu_q = ref.component(layer, "q_a")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            rel_row(
                "q_resid_vs_cpu",
                seen["q_resid"],
                cpu_q(rctx, xn).float(),
                Q_MAX_REL_L2,
                Q_MAX_ROW_REL,
                what="vs CPU q_a on the device attn_norm",
            )
            xs = (xn * Q_SYN_SCALE).bfloat16().float()
            syn_want = cpu_q(rctx, xs).float()
            syn_out = muts["q_a"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"q_a scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(syn_out, syn_want), _worst_row_rel(syn_out, syn_want)
                ymin, ymax = _ratio(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_q_resid", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_q_resid", yrow)
                print(
                    f"q_a on attn_norm x{Q_SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {Q_SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {Q_SYN_MAX_ROW_REL})"
                )
                if yrel > Q_SYN_MAX_REL_L2 or yrow > Q_SYN_MAX_ROW_REL:
                    failures.append(f"q_a scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # topk (the swapped step): exact per-row sets vs golden and vs the shared input, pad layout, then chunk 0.
    want_tk = gl["topk"].long()
    got_tk = _norm_topk(seen["topk"], want_tk)
    if got_tk is None:
        failures.append(f"topk: {seen['topk'].dtype} {tuple(seen['topk'].shape)}, want integer {tuple(want_tk.shape)}")
    else:
        start = c * g.chunk
        failures += _topk_check(got_tk, want_tk, start, "")
        failures += _topk_check(got_tk, shared_topk.long(), start, "vs_shared_input_")
        g0 = g.layer(0, layer)
        shared0 = g.layer(0, src)["topk"]
        rctx0, dctx0 = reference_ctx(ref, layer, g, 0), device_ctx(layer, g, 0)
        rctx0.extra["shared_topk"] = shared0.clone()
        dctx0.extra["shared_topk"] = shared0.clone()
        want0 = g0["topk"].long()
        got0 = _norm_topk(muts["topk_shared"](rctx0, dctx0, g0["attn_norm"].float()), want0)
        if got0 is None:
            failures.append("chunk0_topk: not integer or wrong size")
        else:
            failures += _topk_check(got0, want0, 0, "chunk0_")

    # Downstream of topk_shared: attn_out (the attention consumes the swapped top-k).
    if finite_shape("attn_out"):
        rel_row("attn_out", seen["attn_out"], gl["attn_out"], ATT_MAX_REL_L2, ATT_MAX_ROW_REL)

    # h_mid.
    if finite_shape("h_mid"):
        rel = _rel(seen["h_mid"], gl["h_mid"])
        wr = _worst_row_rel(seen["h_mid"], gl["h_mid"], HC_MULT)
        metrics.record("rel_l2_swap_h_mid", rel)
        metrics.record("worst_row_rel_l2_swap_h_mid", wr)
        print(f"h_mid: rel_l2={rel:.6f} (<= {H_MID_MAX_REL}) worst (row, stream) rel={wr:.6f} (<= {H_MID_MAX_ROW_REL})")
        if rel > H_MID_MAX_REL:
            failures.append(f"h_mid rel L2 {rel:.5f} > {H_MID_MAX_REL}")
        if wr > H_MID_MAX_ROW_REL:
            failures.append(f"h_mid worst row rel L2 {wr:.5f} > {H_MID_MAX_ROW_REL}")
        sr = _stream_rel(seen["h_mid"], gl["h_mid"], HC_MULT)
        metrics.record("max_stream_rel_l2_swap_h_mid", max(sr))
        print(f"h_mid: per-stream rel_l2={[round(v, 6) for v in sr]} (<= {H_MID_MAX_STREAM_REL})")
        bad = [j for j, v in enumerate(sr) if v > H_MID_MAX_STREAM_REL]
        if bad:
            failures.append(f"h_mid streams {bad} rel L2 above {H_MID_MAX_STREAM_REL}: {sr}")

    # Router selection (top-8 of 256; the dense routing weights are nonzero on the selected experts).
    want_sel = gl["router"] != 0
    got_sel = seen["router"].reshape(want_sel.shape) != 0
    overlap = ((got_sel & want_sel).sum(-1).float() / want_sel.sum(-1).clamp_min(1)).mean().item()
    metrics.record("router_overlap_swap", overlap)
    print(f"router top-8 overlap={overlap:.5f} (>= {ROUTER_MIN_OVERLAP})")
    if overlap < ROUTER_MIN_OVERLAP:
        failures.append(f"router selection overlap {overlap:.4f} < {ROUTER_MIN_OVERLAP}")

    # Block out.
    out_rel = _rel(seen["out"], gl["out"])
    out_row = _worst_row_rel(seen["out"], gl["out"], HC_MULT)
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("worst_row_rel_l2_swap_out", out_row)  # informational only (near-tie expert flips)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2}) worst (row, stream) rel={out_row:.5f} (not gated)")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
