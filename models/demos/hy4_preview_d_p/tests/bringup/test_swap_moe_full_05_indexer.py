# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 5: block type moe_full (layer 1) with indexer swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (moe_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    indexer

Reviewed (S.moe_full.05.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). topk [S, 2048] = the indexer's top-2048 key positions per query row (absolute,
-1 padded; the device returns them unsorted) feeds only the sparse attention; then come attn_residual and the MoE half
(ffn_hc, router top-8 of 256, experts, shared expert). Layer 1 has its own indexer weights (moe_shared layers 2-4
reuse its selection). Measured on the CPU (golden s4096 chunk 1, 2048 rows, 2048-key prefix; the indexer replaced by
mutations of the fp32 step on the block's own inputs, every other step the fp32 CPU reference). "tk" = topk mean /
worst row set overlap vs golden, "ao" = attn_out rel / worst row, "hm" = h_mid rel / worst (row, stream), "rt" =
router top-8 overlap, "out" = PCC / rel L2 / worst (row, stream):

    variant                         tk              | ao             | hm              | rt     | out
    fp32 reference                  0.9991 / 0.9966 | 0.0018 / 0.004 | 0.0021 / 0.0029 | 0.9987 | 0.999956 0.0024 0.056
    bf16 in / q / k / cache         0.9987 / 0.9961 | 0.0018 / 0.003 | 0.0021 / 0.0025 | 0.9987 | 0.999957 0.0023 0.031
    same + bf16 scores              0.9957 / 0.9863 | 0.0019 / 0.005 | 0.0021 / 0.0038 | 0.9986 | 0.999958 0.0025 0.056
    unsorted (columns shuffled)     0.9991 / 0.9966 | 0.0018 / 0.004 | 0.0021 / 0.0029 | 0.9987 | 0.999956 0.0024 (fine)
    top-2047 (+ one pad)            0.9988 / 0.9966 | 0.0018 / 0.004 | 0.0021 / 0.0029 | 0.9985 | 0.999956 0.0024 (passes)
    RoPE positions + 1              0.9851 / 0.9697 | 0.0026 / 0.012 | 0.0023 / 0.0089 | 0.9979 | 0.999958 0.0026 (passes)
    causal mask t <= s + 1          0.9988 / 0.9966 | 0.0248 / 0.239 | 0.0136 / 0.1521 | 0.9873 | 0.999888 0.0116 (passes)
    no k_norm                       0.9092 / 0.6865 | 0.0116 / 0.050 | 0.0068 / 0.0340 | 0.9932 | 0.999913 0.0078 (passes)
    RoPE positions from 0           0.7597 / 0.6196 | 0.0149 / 0.055 | 0.0076 / 0.0410 | 0.9870 | 0.999912 0.0087 (passes)
    last row all pads               0.9986 / 0.0    | 0.0207 / 1.0   | 0.0173 / 0.6403 | 0.9984 | 0.999838 0.0156 (passes)
    non-causal                      0.9846 / 0.9692 | 0.0486 / 0.242 | 0.0270 / 0.1541 | 0.9688 | 0.999682 0.0237 (passes)
    last 2048 positions (window)    0.7706 / 0.6357 | 0.0475 / 0.198 | 0.0217 / 0.1349 | 0.9645 | 0.999735 0.0213 (passes)
    causal mask t < s (own dropped) 0.9988 / 0.9966 | 0.0373 / 0.226 | 0.0237 / 0.1636 | 0.9739 | 0.999597 0.0278 (passes)
    rows shifted by 1               0.9257 / 0.7896 | 0.0410 / 0.226 | 0.0249 / 0.1639 | 0.9719 | 0.999585 0.0280 (passes)
    uniform head weights            0.6664 / 0.3965 | 0.0810 / 0.287 | 0.0423 / 0.2138 | 0.9431 | 0.999218 0.0410 (passes)
    last tile row all pads          0.9835 / 0.0    | 0.1217 / 1.0   | 0.0711 / 0.8156 | 0.9891 | 0.997329 0.0729 (passes)
    RoPE on the first 64 dims       0.7726 / 0.5684 | 0.2652 / 0.798 | 0.1361 / 0.6386 | 0.8989 | 0.988488 0.1525 (passes)
    prefix keys zero                0.5940 / 0.3091 | 0.3306 / 0.820 | 0.1723 / 0.6563 | 0.8094 | 0.984979 0.1757 (passes)
    no q RoPE                       0.7357 / 0.4785 | 0.4307 / 0.850 | 0.2272 / 0.6744 | 0.7719 | 0.974975 0.2247
    SP row halves swapped           0.5459 / 0.4941 | 0.4585 / 0.829 | 0.2353 / 0.6798 | 0.7607 | 0.972690 0.2344
    bottom-2048                     0.4844 / 0.0005 | 0.6286 / 0.875 | 0.3267 / 0.7146 | 0.6041 | 0.945459 0.3284
    first 2048 positions            0.5937 / 0.3086 | 0.6327 / 0.834 | 0.3282 / 0.6822 | 0.5981 | 0.944460 0.3311
    half the columns = position 0   0.4992 / 0.4971 | 0.8599 / 1.084 | 0.4650 / 0.7988 | 0.4901 | 0.916360 0.4072
    zero stub (every entry 0)       0.0005 / 0.0005 | 1.0438 / 1.242 | 0.5717 / 0.9168 | 0.4250 | 0.892140 0.4782
    all pads                        0.0    / 0.0    | 1.0    / 1.0   | 0.5446 / 0.8347 | 0.3696 | 0.861220 0.5328

"(passes)" = passes the 0.98 out gate; 14 of the 21 bugs do (the residual dominates block out). The out worst (row,
stream) rel is 0.056 even for the fp32 reference (near-tie expert flips), so it is recorded, not asserted. So the test
also asserts (informational metrics):
  - everything swap 04 (test_swap_moe_full_04_q_a.py) asserts, at its limits: attn_hc gates, attn_x (vs golden, vs
    the CPU hc_pre, rotated pre gates), attn_norm (vs golden, vs the CPU step, x 0.1 eps check), q_resid (vs golden,
    vs the CPU step, x 0.01 eps check), attn_out rel L2 <= 0.01 / worst row <= 0.05, h_mid rel <= 0.005 / worst
    (row, stream) <= 0.02, router top-8 overlap >= 0.99, block out finite and rel L2 <= 0.01;
  - topk (the swapped step) at the component limits (test_c_moe_full_indexer.py), replacing swap 04's topk overlap
    >= 0.995 (the layer-1 device indexer scores 0.9959): integer output with S x 2048 elements, pads -1 or
    0xFFFFFFFF; mean per-row set overlap vs golden >= 0.99 and worst row >= 0.97; no position after its query row;
    no repeats within a row; exactly min(pos + 1, 2048) valid positions per row; every row selects its own position.
    Mean overlap catches every bug above except top-2047, causal t <= s + 1, t < s and the padded last row; the count,
    causality, self and worst-row checks catch those;
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
BLOCK_TYPE = "moe_full"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "indexer"]
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
TOPK_MIN_OVERLAP = 0.99  # topk vs golden and vs the CPU indexer on the same input: mean per-row set overlap
TOPK_MIN_ROW_OVERLAP = 0.97  # topk worst row overlap (vs golden and vs the CPU step)
SENTINEL = 0xFFFFFFFF  # topk_large_indices' pad for rows with fewer valid keys than k
ATT_MAX_REL_L2 = 0.01  # attn_out, whole tensor
ATT_MAX_ROW_REL = 0.05  # attn_out, worst token row
ROUTER_MIN_OVERLAP = 0.99  # router top-8 selection overlap vs golden
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor


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

    # attn_norm: vs golden, vs the CPU step on the same input, and on a scaled input (eps).
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
