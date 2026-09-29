# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 3: block type moe_shared (layer 2) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 2 (moe_shared) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm

Reviewed (S.moe_shared.03.test.1). Built from the layer-1 swap test (test_swap_moe_full_03_attn_norm.py) with the
layer-2 changes of swaps 01 / 02 (test_swap_moe_shared_02_attn_hc_pre.py): ctx.extra["shared_topk"] = the golden's
L{src}.topk (src = cfg.topk_source(2) = 1) in both contexts (a shared layer has no indexer; its topk_shared step
otherwise raises a KeyError, since layer 1 does not run here; the golden's L1 and L2 topk are identical), no indexer
top-k check, a per-stream h_mid limit (stream 3 is about 5x the others) and router overlap >= 0.98. The gated metric
is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC streams, spec block threshold 0.98). attn_norm [S, H] =
w * x * rsqrt(mean(x^2) + 1e-5) feeds q_a (then q_a_layernorm, which removes a per-row scale) and the attention
(kv_a, output gate), then attn_residual and the MoE half. At layer 2, 11 rows of attn_x have mean(x^2) below eps and
w is flat ([0.084, 0.159]); see test_c_moe_shared_attn_norm.py. Measured on the CPU (golden s4096 chunk 1, 2048 rows;
attn_hc and attn_hc_pre the fp32 CPU steps, attn_norm replaced by mutations of the fp32 reference; script
/tmp/hy4_ssh3/study.py). "an" = attn_norm vs golden rel L2 / row norm ratio / worst row, "syn" = the step on attn_x
x 0.1 (bf16) vs the CPU step (rel / worst row), "q" = q_resid rel, "ao" = attn_out rel / worst row, "hm" = h_mid rel /
max per-stream rel / worst (row, stream), "rt" = router top-8 overlap:

    variant                     an                            | syn             | q      | ao            | hm                       | rt     | out PCC   rel
    fp32 reference              0.0022 [0.9999, 1.0001] 0.0024 | 0      / 0      | 0.0018 | 0.0019 / 0.002 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995  0.0030
    bf16 everywhere             0.0041 [0.9939, 1.0058] 0.0071 | 0.0031 / 0.0062 | 0.0022 | 0.0031 / 0.005 | 0.0010 / 0.0037 / 0.0062 | 0.9964 | 0.999990  0.0044
    eps 1e-6                    0.133  [1.0011, 1.3649] 0.36   | 0.74   / 2.0    | 0.0018 | 0.093  / 0.24  | 0.0055 / 0.028  / 0.21   | 0.9374 | 0.999821  0.019  (passes)
    eps 2e-5                    0.083  [0.8125, 0.9987] 0.19   | 0.21   / 0.29   | 0.0018 | 0.064  / 0.15  | 0.0036 / 0.018  / 0.13   | 0.9593 | 0.999956  0.0094 (passes)
    eps 0                       0.156  [1.0013, 1.4355] 0.44   | 1.61   / 9.3    | 0.0018 | 0.107  / 0.28  | 0.0064 / 0.033  / 0.24   | 0.9295 | 0.999809  0.020  (passes)
    x 1.01                      0.0102 [1.0099, 1.0101] 0.0104 | 0.010  / 0.010  | 0.0018 | 0.0076 / 0.008 | 0.0018 / 0.0085 / 0.011  | 0.9928 | 0.999979  0.0065 (passes)
    x 1.02                      0.020  [1.0199, 1.0201] 0.020  | 0.020  / 0.020  | 0.0018 | 0.015  / 0.016 | 0.0032 / 0.016  / 0.021  | 0.9865 | 0.999954  0.0098 (passes)
    RMS over half the columns   0.0101 [0.9997, 1.0003] 0.029  | 0.0046 / 0.019  | 0.0048 | 0.0068 / 0.019 | 0.0017 / 0.0078 / 0.023  | 0.9943 | 0.999981  0.0062 (passes)
    RMS over a quarter          0.0151 [0.9995, 1.0002] 0.035  | 0.0076 / 0.025  | 0.0070 | 0.010  / 0.026 | 0.0024 / 0.012  / 0.026  | 0.9930 | 0.999978  0.0067 (passes)
    LayerNorm instead of RMS    0.0113 [0.9998, 1.0001] 0.036  | 0.012  / 0.035  | 0.0052 | 0.0076 / 0.024 | 0.0018 / 0.0085 / 0.027  | 0.9947 | 0.999983  0.0058 (passes)
    last row zeroed             0.024  [0.0,    1.0001] 1.0    | 0.033  / 1.0    | 0.021  | 0.024  / 0.98  | 0.0068 / 0.036  / 1.15   | 0.9969 | 0.999972  0.0075 (passes)
    w halves swapped (TP)       0.087  [0.9998, 1.0151] 0.092  | 0.087  / 0.092  | 0.042  | 0.058  / 0.067 | 0.012  / 0.062  / 0.082  | 0.9592 | 0.999813  0.019  (passes)
    sum instead of mean         0.986  [0.0128, 0.0183] 0.987  | 0.975  / 0.986  | 0.014  | 0.93   / 1.0   | 0.19   / 1.02   / 1.30   | 0.5637 | 0.979803  0.22
    no weight                   6.5    [7.4569, 7.5469] 6.5    | 6.5    / 6.5    | 0.032  | 1.05   / 1.46  | 0.24   / 1.25   / 1.67   | 0.4109 | 0.942398  0.35
    1 + w                       7.5    [8.4552, 8.5451] 7.5    | 7.5    / 7.5    | 0.028  | 1.07   / 1.50  | 0.24   / 1.28   / 1.71   | 0.4067 | 0.940311  0.35
    output column halves swapped 1.41  [0.9999, 1.0001] 1.46   | 1.41   / 1.46   | 1.33   | 1.08   / 1.30  | 0.22   / 1.16   / 1.54   | 0.5402 | 0.941364  0.50
    SP row halves swapped       1.20   [0.7051, 1.4183] 1.58   | 1.30   / 8.4    | 0.79   | 1.08   / 1.42  | 0.22   / 1.16   / 1.51   | 0.5439 | 0.969227  0.25
    zero stub                   1.0    [0.0,    0.0]    1.0    | 1.0    / 1.0    | 1.0    | 0.91   / 1.0   | 0.19   / 1.00   / 1.29   | 0.5915 | 0.942568  0.53

"(passes)" = passes the 0.98 out gate: every eps, scale, subset, centring, zeroed-row and TP-weight bug passes it.
So the test also asserts (informational metrics):
  - attn_hc (the gates) and attn_x at the swap 02 limits: gates rel L2 <= 0.01, per column <= 0.01, post worst row
    <= 0.015; attn_x vs golden rel <= 0.005, row norm ratio in [0.996, 1.004], worst row <= 0.01; attn_x vs the CPU
    hc_pre on the same inputs 0.003 / 0.006; the attn_hc_pre module again with rotated pre gates 0.004 / 0.01;
  - attn_norm (the swapped step) vs golden at the component limits (test_c_moe_shared_attn_norm.py): finite, shape,
    rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row <= 0.015 (catches every mutation above;
    bf16 everywhere fits: 0.0041 / [0.9939, 1.0058] / 0.0071);
  - attn_norm vs the CPU attn_norm on the same input (the device attn_x): same limits (the step's own error);
  - the attn_norm module again on the device attn_x x 0.1 (bf16) vs the CPU step on the same input: rel L2 <= 0.01,
    worst row <= 0.02 (every row eps-sensitive; eps 2e-5 0.21);
  - q_resid rel L2 <= 0.01; attn_out rel <= 0.01, worst row <= 0.05;
  - h_mid rel <= 0.005, max per-stream rel <= 0.005, worst (row, stream) <= 0.02 (as swap 02);
  - router top-8 overlap >= 0.98 (a gross check as swaps 01 / 02; reference 0.9973, bf16 0.9964); block out finite,
    rel <= 0.01.
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
BLOCK_TYPE = "moe_shared"
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
H_MID_MAX_STREAM_REL = 0.005  # h_mid, rel L2 of each stream (stream 3 dominates the whole tensor at layer 2)
H_MID_MAX_ROW_REL = 0.02  # h_mid, worst (row, stream)
N_MAX_REL_L2 = 0.008  # attn_norm vs golden, and vs the CPU attn_norm on the same input
N_RATIO = (0.993, 1.007)  # attn_norm per-row norm ratio vs golden
N_MAX_ROW_REL = 0.015  # attn_norm worst row, vs golden and vs the CPU step on the same input
SYN_SCALE = 0.1  # eps check: the device attn_x x SYN_SCALE (bf16) through the module vs the CPU step
SYN_MAX_REL_L2 = 0.01
SYN_MAX_ROW_REL = 0.02
Q_MAX_REL_L2 = 0.01  # q_resid
ATT_MAX_REL_L2 = 0.01  # attn_out, whole tensor
ATT_MAX_ROW_REL = 0.05  # attn_out, worst token row
ROUTER_MIN_OVERLAP = 0.98  # router top-8 selection overlap vs golden
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


def _stream_rel(got, want, streams):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=(0, 2)) / want.norm(dim=(0, 2)).clamp_min(1e-12)).tolist()


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

    # Downstream of attn_norm: q_resid, attn_out (the top-k is layer 1's, from the golden).
    if finite_shape("q_resid"):
        rel_row("q_resid", seen["q_resid"], gl["q_resid"], Q_MAX_REL_L2, None)
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
