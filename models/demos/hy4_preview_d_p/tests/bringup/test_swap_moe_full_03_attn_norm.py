# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 3: block type moe_full (layer 1) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (moe_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm

Reviewed (S.moe_full.03.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). attn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) feeds q_a (then
q_a_layernorm, which removes a per-row scale), the indexer (top-2048) and the attention (kv_a, output gate), then
attn_residual and the MoE half (ffn_hc, router top-8 of 256, experts, shared expert). At layer 1 the row rms of attn_x
starts at 0.0045, so eps is visible on the golden (the smallest mean(x^2) is 2.1x eps). Measured on the CPU (golden
s4096 chunk 1, 2048 rows; attn_hc and attn_hc_pre the fp32 CPU steps, attn_norm replaced by mutations of the fp32
reference). "an" = attn_norm vs golden rel L2 / row norm ratio / worst row, "syn" = the step on attn_x x 0.1 (bf16)
vs the CPU step (rel / worst row), "q" = q_resid rel, "tk" = indexer top-k set overlap, "ao" = attn_out rel / worst
row, "hm" = h_mid rel / worst (row, stream), "rt" = router top-8 overlap:

    variant                      an                           | syn           | q      | tk     | ao           | hm            | rt     | out PCC   rel
    fp32 reference               0.0024 [0.9999, 1.0001] 0.0027 | 0      / 0      | 0.0018 | 0.9991 | 0.0018 / 0.004 | 0.0021 / 0.0029 | 0.9987 | 0.999997  0.0024
    bf16 everywhere (device)     0.0042 [0.9952, 1.0045] 0.0062 | 0.0029 / 0.0052 | 0.0022 | 0.9991 | 0.0023 / 0.004 | 0.0022 / 0.0031 | 0.9981 | 0.999995  0.0031
    eps 1e-6                     0.068  [1.0008, 1.1896] 0.19   | 0.61   / 1.9    | 0.0018 | 0.9991 | 0.046  / 0.11  | 0.015  / 0.076  | 0.9668 | 0.999880  0.016  (passes)
    eps 2e-5                     0.055  [0.8683, 0.9993] 0.13   | 0.18   / 0.29   | 0.0018 | 0.9991 | 0.041  / 0.090 | 0.015  / 0.061  | 0.9695 | 0.999902  0.015  (passes)
    eps 0                        0.078  [1.0009, 1.2180] 0.22   | 1.16   / 6.0    | 0.0018 | 0.9991 | 0.052  / 0.13  | 0.017  / 0.086  | 0.9634 | 0.999858  0.018  (passes)
    x 1.01                       0.0103 [1.0099, 1.0101] 0.0105 | 0.010  / 0.010  | 0.0018 | 0.9991 | 0.0062 / 0.008 | 0.0039 / 0.0058 | 0.9940 | 0.999981  0.0069 (passes)
    x 1.02                       0.020  [1.0199, 1.0201] 0.020  | 0.020  / 0.020  | 0.0018 | 0.9991 | 0.012  / 0.014 | 0.0068 / 0.011  | 0.9898 | 0.999945  0.012  (passes)
    RMS over half the columns    0.0107 [0.9834, 1.0289] 0.029  | 0.0063 / 0.022  | 0.0018 | 0.9991 | 0.0068 / 0.016 | 0.0043 / 0.012  | 0.9944 | 0.999980  0.0067 (passes)
    RMS over a quarter           0.021  [0.9734, 1.0709] 0.071  | 0.013  / 0.052  | 0.0018 | 0.9991 | 0.013  / 0.040 | 0.0076 / 0.026  | 0.9900 | 0.999946  0.012  (passes)
    LayerNorm instead of RMS     0.0101 [0.9998, 1.0002] 0.035  | 0.010  / 0.035  | 0.0045 | 0.9990 | 0.0030 / 0.007 | 0.0025 / 0.0059 | 0.9981 | 0.999995  0.0032 (passes)
    last row zeroed              0.024  [0.0,    1.0001] 1.0    | 0.034  / 1.0    | 0.022  | 0.9989 | 0.026  / 1.25  | 0.022  / 0.80   | 0.9985 | 0.999798  0.020  (passes)
    w halves swapped (TP)        0.19   [0.9624, 1.0167] 0.21   | 0.19   / 0.21   | 0.10   | 0.9772 | 0.11   / 0.13  | 0.057  / 0.100  | 0.9187 | 0.998853  0.052  (passes)
    sum instead of mean          0.986  [0.0128, 0.0155] 0.987  | 0.979  / 0.986  | 0.014  | 0.9742 | 1.18   / 1.38  | 0.65   / 1.05   | 0.3324 | 0.807     0.59
    1 + w                        7.7    [8.5422, 8.9065] 7.9    | 7.7    / 7.9    | 0.077  | 0.9812 | 0.80   / 1.05  | 0.43   / 0.79   | 0.5953 | 0.937     0.60
    output column halves swapped 1.41   [0.9999, 1.0001] 1.45   | 1.41   / 1.45   | 1.37   | 0.7712 | 1.12   / 1.59  | 0.61   / 1.13   | 0.3346 | 0.840     0.54
    SP row halves swapped        1.07   [0.8108, 1.2334] 1.30   | 1.19   / 5.9    | 0.85   | 0.8967 | 0.99   / 1.35  | 0.55   / 1.00   | 0.4125 | 0.857     0.52
    zero stub                    1.0    [0.0,    0.0]    1.0    | 1.0    / 1.0    | 1.0    | 0.7554 | 0.95   / 1.30  | 0.51   / 0.94   | 0.3967 | 0.871     0.49

"(passes)" = passes the 0.98 out gate: every eps, scale, subset, centring, zeroed-row and TP-weight bug passes it.
The residual dominates block out and q_a_layernorm removes a per-row scale from q_resid. So the test also asserts
(informational metrics):
  - attn_hc (the gates) and attn_x at the swap 02 limits: gates rel L2 <= 0.01, per column <= 0.01, post worst row
    <= 0.015; attn_x vs golden rel <= 0.005, row norm ratio in [0.996, 1.004], worst row <= 0.01; attn_x vs the CPU
    hc_pre on the same inputs 0.003 / 0.006; the attn_hc_pre module again with rotated pre gates 0.004 / 0.01;
  - attn_norm (the swapped step) vs golden at the component limits (test_c_moe_full_attn_norm.py): finite, shape,
    rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row <= 0.015 (catches every mutation above);
  - attn_norm vs the CPU attn_norm on the same input (the device attn_x): same limits (the step's own error);
  - the attn_norm module again on the device attn_x x 0.1 (bf16) vs the CPU step on the same input: rel L2 <= 0.01,
    worst row <= 0.02 (every row eps-sensitive; eps 2e-5 0.18);
  - q_resid rel L2 <= 0.01; indexer top-k overlap >= 0.995 (reference 0.9991: near-tie flips);
  - attn_out rel <= 0.01, worst row <= 0.05; h_mid rel <= 0.005, worst (row, stream) <= 0.02 (as swap 02);
  - router top-8 overlap >= 0.99 (reference 0.9987, bf16 0.9981; x 1.02 0.9898); block out finite, rel <= 0.01.
The out worst (row, stream) rel L2 is recorded, not asserted (near-tie expert flips, see swap 01).
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
BLOCK_TYPE = "moe_full"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm"]
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
Q_MAX_REL_L2 = 0.01  # q_resid
TOPK_MIN_OVERLAP = 0.995  # indexer top-k selection vs golden (mean per-row set overlap, -1 pads ignored)
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


def _topk_overlap(got, want):
    """Mean per-row |got & want| / |want| over the valid (non -1) indices."""
    got, want = got.reshape(want.shape).long(), want.long()
    tot = 0.0
    for a, b in zip(got, want):
        b = b[b >= 0]
        if b.numel():
            tot += torch.isin(b, a[a >= 0]).float().mean().item()
        else:
            tot += 1.0
    return tot / want.shape[0]


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

    # attn_norm (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    if finite_shape("attn_norm"):
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

    # Downstream of attn_norm: q_resid, the indexer's top-k, attn_out.
    if finite_shape("q_resid"):
        rel_row("q_resid", seen["q_resid"], gl["q_resid"], Q_MAX_REL_L2, None)
    if seen["topk"].numel() != gl["topk"].numel():
        failures.append(f"topk: shape {tuple(seen['topk'].shape)} vs golden {tuple(gl['topk'].shape)}")
    else:
        ov = _topk_overlap(seen["topk"], gl["topk"])
        metrics.record("topk_overlap_swap", ov)
        print(f"topk overlap vs golden: {ov:.5f} (>= {TOPK_MIN_OVERLAP})")
        if ov < TOPK_MIN_OVERLAP:
            failures.append(f"topk overlap {ov:.5f} < {TOPK_MIN_OVERLAP}")
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
