# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 5: block type dense_full (layer 0) with indexer swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    indexer

Reviewed (S.dense_full.05.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). topk [S, 2048] = the indexer's top-2048 key positions per query row (absolute,
-1 padded; the device returns them unsorted) feeds only the sparse attention, which attends to exactly the selected
latent keys: a wrong selection can pull in future keys, and the sinks and the residual keep block out close to the
golden. Measured on the CPU (golden s4096 chunk 1, 2048 rows, 2048-key prefix; indexer replaced by mutations of the
fp32 step on the block's own inputs, every other step the fp32 CPU reference; topk = mean / worst row set overlap):

    variant                         topk ov / worst | attn_out rel / worst row | h_mid rel / worst | out PCC   rel
    fp32 reference                  0.9993 / 0.9976 | 0.0017 / 0.002           | 0.0017 / 0.0017   | 1.000000  0.0017
    bf16 in / q / k / cache         0.9990 / 0.9961 | 0.0017 / 0.002           | 0.0017 / 0.0017   | 1.000000  0.0017
    same + bf16 scores              0.9970 / 0.9893 | 0.0017 / 0.002           | 0.0017 / 0.0019   | 1.000000  0.0017
    unsorted (columns shuffled)     0.9993 / 0.9976 | 0.0017 / 0.002           | 0.0017 / 0.0017   | 1.000000  0.0017 (fine)
    top-2047 (+ one pad)            0.9990 / 0.9971 | 0.0017 / 0.002           | 0.0017 / 0.0017   | 1.000000  0.0017 (passes)
    RoPE positions + 1              0.9842 / 0.9561 | 0.0018 / 0.003           | 0.0018 / 0.0032   | 1.000000  0.0018 (passes)
    no k_norm                       0.9889 / 0.9565 | 0.0018 / 0.003           | 0.0018 / 0.0035   | 1.000000  0.0019 (passes)
    RoPE positions from 0           0.7723 / 0.6470 | 0.0054 / 0.011           | 0.0051 / 0.0114   | 1.000000  0.0069 (passes)
    causal mask t <= s + 1          0.9990 / 0.9971 | 0.0059 / 0.065           | 0.0054 / 0.0824   | 1.000000  0.0082 (passes)
    causal mask t < s (own dropped) 0.9990 / 0.9971 | 0.0115 / 0.063           | 0.0074 / 0.0798   | 0.999987  0.0114 (passes)
    rows shifted by 1               0.9269 / 0.7920 | 0.0131 / 0.064           | 0.0087 / 0.0804   | 0.999969  0.0129 (passes)
    last 2048 positions (window)    0.8259 / 0.6919 | 0.0116 / 0.032           | 0.0095 / 0.0413   | 0.999973  0.0131 (passes)
    non-causal                      0.9846 / 0.9692 | 0.0126 / 0.064           | 0.0109 / 0.0807   | 0.999932  0.0164 (passes)
    uniform head weights            0.7246 / 0.4409 | 0.0241 / 0.114           | 0.0186 / 0.1355   | 0.999650  0.0293 (passes)
    last row all pads               0.9988 / 0.0    | 0.0245 / 1.0             | 0.0311 / 0.8918   | 0.999363  0.0377 (passes)
    RoPE on the first 64 dims       0.7726 / 0.6353 | 0.0337 / 0.224           | 0.0280 / 0.2668   | 0.999167  0.0434 (passes)
    prefix keys zero                0.5729 / 0.2407 | 0.0559 / 0.126           | 0.0568 / 0.1121   | 0.996657  0.0844 (passes)
    no q RoPE                       0.6917 / 0.4365 | 0.0998 / 0.270           | 0.0833 / 0.3098   | 0.992536  0.1257 (passes)
    last tile row all pads          0.9836 / 0.0    | 0.1241 / 1.0             | 0.1148 / 1.3154   | 0.988826  0.1522 (passes)
    SP row halves swapped           0.5236 / 0.4653 | 0.1350 / 0.331           | 0.1064 / 0.3629   | 0.988685  0.1553 (passes)
    first 2048 positions            0.5699 / 0.2393 | 0.1918 / 0.350           | 0.1513 / 0.3947   | 0.977033  0.2190
    bottom-2048                     0.4844 / 0.0    | 0.1760 / 0.358           | 0.1467 / 0.4225   | 0.977126  0.2187
    half the columns = position 0   0.4993 / 0.4976 | 0.2285 / 0.352           | 0.1889 / 0.3860   | 0.963768  0.2741
    zero stub (every entry 0)       0.0001 / 0.0    | 0.5452 / 0.852           | 0.4919 / 0.9193   | 0.796307  0.7923
    all pads                        0.0    / 0.0    | 1.0    / 1.0             | 0.8697 / 1.3177   | 0.529727  1.1612

"(passes)" = passes the 0.98 out gate; 18 of 24 bugs do. So the test also asserts (informational metrics):
  - everything swap 04 asserts: attn_hc gates, attn_x, attn_norm (vs golden, vs the CPU step, x 0.1 eps check),
    q_resid (vs golden, vs the CPU step, x 0.01 eps check), attn_out rel L2 <= 0.01 / worst row <= 0.05, h_mid rel
    L2 <= 0.01 / worst row <= 0.05, block out finite and rel L2 <= 0.01;
  - topk (the swapped step) at the component limits (test_c_dense_full_indexer.py): integer output with S x 2048
    elements, pads -1 or 0xFFFFFFFF; mean per-row set overlap vs golden >= 0.99 (device component 0.9971) and worst
    row >= 0.97; no position after its query row; no repeats within a row; exactly min(pos + 1, 2048) valid
    positions per row; every row selects its own position. This catches every variant above except top-2047 +
    causal t <= s + 1 / t < s by overlap alone; the structure checks catch those (count, non-causal, self);
  - topk vs the CPU indexer on the same device inputs (device attn_norm and q_resid): overlap >= 0.99, worst row
    >= 0.97 (the step's own error, apart from the upstream device error);
  - the indexer module once more on golden chunk 0 (start 0, empty prefix): every row must hold exactly [0, pos]
    (the short-row rule, which chunk 1 cannot exercise), with the same structure checks.
The device returns unsorted indices; the sparse attention does not care (shuffled columns leave attn_out unchanged),
so order is not checked. The trail line pcc_swap_topk is positional match (template default) and is not meaningful.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import Ctx, run_block
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
BLOCK_TYPE = "dense_full"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "indexer"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_ABS_ERR = 0.015  # attn_hc, per column: max |got - want|
X_MAX_REL_L2 = 0.004  # attn_x vs golden
X_RATIO = (0.995, 1.005)  # attn_x per-row ||got|| / ||want||
X_MAX_ROW_REL_L2 = 0.01  # attn_x worst row
N_MAX_REL_L2 = 0.008  # attn_norm vs golden, and vs the CPU attn_norm on the same input
N_RATIO = (0.993, 1.007)  # attn_norm per-row norm ratio vs golden
N_MAX_ROW_REL_L2 = 0.015  # attn_norm worst row, vs golden and vs the CPU step on the same input
SYN_SCALE = 0.1  # eps check: the device attn_x x SYN_SCALE (bf16) through the module vs the CPU step
SYN_MAX_REL_L2 = 0.01
SYN_MAX_ROW_REL_L2 = 0.02
Q_MAX_REL_L2 = 0.008  # q_resid vs golden, and vs the CPU q_a on the same input
Q_RATIO = (0.994, 1.006)  # q_resid per-row norm ratio vs golden
Q_MAX_ROW_REL_L2 = 0.015  # q_resid worst row, vs golden and vs the CPU step on the same input
Q_SYN_SCALE = 0.01  # eps check: the device attn_norm x Q_SYN_SCALE (bf16) through the q_a module vs the CPU step
Q_SYN_MAX_REL_L2 = 0.01
Q_SYN_MAX_ROW_REL_L2 = 0.02
TOPK_MIN_OVERLAP = 0.99  # topk vs golden and vs the CPU indexer on the same input: mean per-row set overlap
TOPK_MIN_ROW_OVERLAP = 0.97  # topk worst row overlap (vs golden and vs the CPU step)
SENTINEL = 0xFFFFFFFF  # topk_large_indices' pad for rows with fewer valid keys than k
ATT_MAX_REL_L2 = 0.01  # attn_out, whole tensor
ATT_MAX_ROW_REL_L2 = 0.05  # attn_out, worst token row
MID_MAX_REL_L2 = 0.01  # h_mid, whole tensor
MID_MAX_ROW_REL_L2 = 0.05  # h_mid, worst token row
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor


def _errors(got, want):
    """rel L2, per-row norm ratio (min, max), worst row rel L2."""
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    wn = want.norm(dim=-1).clamp_min(1e-12)
    ratio = got.norm(dim=-1) / wn
    return rel, ratio.min().item(), ratio.max().item(), ((got - want).norm(dim=-1) / wn).max().item()


def _normalize(out, want, failures):
    """int64 [S, k] with every pad (negative, or the uint32 sentinel) as -1; None (and a failure) if unusable."""
    if out.is_floating_point():
        failures.append(f"topk: output must be integer positions, got {out.dtype}")
        return None
    if out.numel() != want.numel():
        failures.append(f"topk: output has {out.numel()} elements, want {tuple(want.shape)}")
        return None
    t = out.reshape(want.shape).to(torch.int64)
    return torch.where((t < 0) | (t == SENTINEL), torch.full_like(t, -1), t)


def _row_overlap(got, want):
    """Per-row |got & want| / |want| over the valid (non -1) positions; 1.0 for a row with none wanted."""
    res = []
    for a, b in zip(got, want):
        b = b[b >= 0]
        res.append(torch.isin(b, a[a >= 0]).float().mean().item() if b.numel() else 1.0)
    return torch.tensor(res)


def _structure(got, start, k):
    """Causality, uniqueness and per-row valid count of a normalized output at absolute rows [start, start + S)."""
    fails = []
    pos = torch.arange(start, start + got.shape[0])[:, None]
    valid = got >= 0
    noncausal = (valid & (got > pos)).sum().item()
    if noncausal:
        fails.append(f"{noncausal} selected positions are after their query row (non-causal)")
    srt = got.sort(dim=-1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item()
    if dup:
        fails.append(f"{dup} repeated positions within rows")
    n = valid.sum(-1)
    exp = torch.clamp(pos[:, 0] + 1, max=k)
    bad = (n != exp).nonzero().flatten()
    if bad.numel():
        r = bad[0].item()
        fails.append(
            f"{bad.numel()} rows hold the wrong number of valid positions "
            f"(first: row {r} at position {start + r} has {n[r].item()}, want {exp[r].item()})"
        )
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

    def rel_row(n, max_rel, max_row, ratio=None, what="vs golden", want=None):
        rel, rmin, rmax, row = _errors(seen[n], gl[n] if want is None else want)
        tag = n if what == "vs golden" else f"{n}_{what.replace(' ', '_')}"
        metrics.record(f"rel_l2_swap_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_{tag}", row)
        msg = f"{n} {what}: rel_l2={rel:.6f} (<= {max_rel})"
        if ratio is not None:
            msg += f" row norm ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(ratio)})"
        if max_row is not None:
            msg += f" worst_row_rel_l2={row:.5f} (<= {max_row})"
        print(msg)
        if rel > max_rel:
            failures.append(f"{n} {what}: rel L2 {rel:.5f} > {max_rel}")
        if ratio is not None and not (ratio[0] <= rmin and rmax <= ratio[1]):
            failures.append(f"{n} {what}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if max_row is not None and row > max_row:
            failures.append(f"{n} {what}: worst row rel L2 {row:.5f} > {max_row}")

    # attn_hc: the iHC gates [S, 8] (pre 0-3 | post 4-7).
    if finite_shape("attn_hc"):
        want = gl["attn_hc"].float()
        got = seen["attn_hc"].float().reshape(want.shape)
        rel = _errors(got, want)[0]
        col_err = (got - want).abs().amax(dim=0)
        metrics.record("rel_l2_swap_attn_hc", rel)
        metrics.record("max_abs_err_swap_attn_hc", col_err.max().item())
        print(
            f"attn_hc: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) max_abs_err per column="
            f"{[round(v, 5) for v in col_err.tolist()]} (<= {GATES_MAX_ABS_ERR})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"attn_hc: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_err.tolist()) if v > GATES_MAX_ABS_ERR]
        if bad:
            failures.append(f"attn_hc: max abs error > {GATES_MAX_ABS_ERR} in columns {bad}")

    # attn_x vs golden.
    x_ok = finite_shape("attn_x")
    if x_ok:
        rel_row("attn_x", X_MAX_REL_L2, X_MAX_ROW_REL_L2, X_RATIO)

    # attn_norm (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    n_ok = finite_shape("attn_norm")
    if n_ok:
        rel_row("attn_norm", N_MAX_REL_L2, N_MAX_ROW_REL_L2, N_RATIO)
        if x_ok:
            cpu = ref.component(layer, "attn_norm")
            x = seen["attn_x"].float().reshape(gl["attn_x"].shape)
            rel_row("attn_norm", N_MAX_REL_L2, N_MAX_ROW_REL_L2, what="vs cpu", want=cpu(rctx, x).float())

            xs = (x * SYN_SCALE).bfloat16().float()
            syn_want = cpu(rctx, xs).float()
            syn_out = muts["attn_norm"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"attn_norm scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, ymin, ymax, yrow = _errors(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_attn_norm", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_attn_norm", yrow)
                print(
                    f"attn_norm on attn_x x{SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {SYN_MAX_ROW_REL_L2})"
                )
                if yrel > SYN_MAX_REL_L2 or yrow > SYN_MAX_ROW_REL_L2:
                    failures.append(f"attn_norm scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # q_resid (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    if finite_shape("q_resid"):
        rel_row("q_resid", Q_MAX_REL_L2, Q_MAX_ROW_REL_L2, Q_RATIO)
        if n_ok:
            cpu = ref.component(layer, "q_a")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            rel_row("q_resid", Q_MAX_REL_L2, Q_MAX_ROW_REL_L2, what="vs cpu", want=cpu(rctx, xn).float())

            xs = (xn * Q_SYN_SCALE).bfloat16().float()
            syn_want = cpu(rctx, xs).float()
            syn_out = muts["q_a"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"q_a scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, ymin, ymax, yrow = _errors(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_q_resid", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_q_resid", yrow)
                print(
                    f"q_a on attn_norm x{Q_SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {Q_SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {Q_SYN_MAX_ROW_REL_L2})"
                )
                if yrel > Q_SYN_MAX_REL_L2 or yrow > Q_SYN_MAX_ROW_REL_L2:
                    failures.append(f"q_a scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # topk (the swapped step): vs golden, structure, vs the CPU indexer on the same input, and chunk 0.
    def topk_checks(tag, got, want, start):
        """Overlap (mean, worst row) of normalized got vs want and the structure checks; returns nothing."""
        rows = _row_overlap(got, want)
        pos = torch.arange(start, start + got.shape[0])[:, None]
        self_frac = (got == pos).any(-1).float().mean().item()
        metrics.record(f"topk_overlap_swap_{tag}", rows.mean().item())
        metrics.record(f"topk_worst_row_overlap_swap_{tag}", rows.min().item())
        print(
            f"topk {tag}: overlap={rows.mean().item():.5f} (>= {TOPK_MIN_OVERLAP}) worst row={rows.min().item():.5f} "
            f"(>= {TOPK_MIN_ROW_OVERLAP}) self selected={self_frac:.5f} (== 1)"
        )
        if rows.mean().item() < TOPK_MIN_OVERLAP:
            failures.append(f"topk {tag}: overlap {rows.mean().item():.5f} < {TOPK_MIN_OVERLAP}")
        if rows.min().item() < TOPK_MIN_ROW_OVERLAP:
            failures.append(f"topk {tag}: worst row overlap {rows.min().item():.5f} < {TOPK_MIN_ROW_OVERLAP}")
        if self_frac < 1.0:
            failures.append(f"topk {tag}: only {self_frac:.5f} of rows select their own position")
        failures.extend(f"topk {tag}: {f}" for f in _structure(got, start, want.shape[-1]))

    want_tk = gl["topk"].long()
    got_tk = _normalize(seen["topk"], want_tk, failures)
    if got_tk is not None:
        start = c * g.chunk
        topk_checks("vs_golden", got_tk, want_tk, start)
        if n_ok and "q_resid" in seen:
            cpu = ref.component(layer, "indexer")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            qr = seen["q_resid"].float().reshape(gl["q_resid"].shape)
            cpu_tk = cpu(reference_ctx(ref, layer, g, c), xn, qr).long()
            topk_checks("vs_cpu", got_tk, cpu_tk, start)

        # Chunk 0: every row sees <= 2048 keys and must keep exactly [0, position], the rest padded.
        g0 = g.layer(0, layer)
        want0 = g0["topk"].long()
        dctx0 = Ctx(layer, 0, g.chunk, None, {"state_prefix": g.state(layer), "prefix_len": 0, "max_seq": g.seq})
        out0 = muts["indexer"](reference_ctx(ref, layer, g, 0), dctx0, g0["attn_norm"].float(), g0["q_resid"].float())
        got0 = _normalize(out0, want0, failures)
        if got0 is not None:
            rows0 = _row_overlap(got0, want0)
            metrics.record("topk_chunk0_worst_row_overlap_swap", rows0.min().item())
            print(
                f"topk chunk 0 (start 0): mean overlap={rows0.mean().item():.6f} worst row={rows0.min().item():.5f} (== 1)"
            )
            if rows0.min().item() < 1.0:
                failures.append(f"topk chunk 0: worst row overlap {rows0.min().item():.5f} < 1")
            failures.extend(f"topk chunk 0: {f}" for f in _structure(got0, 0, want0.shape[-1]))

    if finite_shape("attn_out"):
        rel_row("attn_out", ATT_MAX_REL_L2, ATT_MAX_ROW_REL_L2)

    # h_mid.
    if finite_shape("h_mid"):
        rel_row("h_mid", MID_MAX_REL_L2, MID_MAX_ROW_REL_L2)

    # Block out.
    out_rel = _errors(seen["out"], gl["out"])[0]
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not (torch.isfinite(seen["out"].float()).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
