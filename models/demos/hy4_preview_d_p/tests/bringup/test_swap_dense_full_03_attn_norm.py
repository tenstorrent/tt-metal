# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 3: block type dense_full (layer 0) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm

Reviewed (S.dense_full.03.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). attn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) feeds q_a (then
q_a_layernorm, which removes a per-row scale), the indexer (top-2048) and the attention (kv_a, then kv_a_layernorm;
the output gate linear_gate reads attn_norm directly), then attn_residual and the FFN. Measured on the CPU (golden
s4096 chunk 1, 2048 rows, 2048-key prefix; attn_norm replaced by mutations of the fp32 reference, attn_hc and
attn_hc_pre the fp32 CPU steps):

    variant                     attn_norm rel / row ratio / worst row | q_resid | topk ov | attn_out rel / row | h_mid rel / row | out PCC   rel
    fp32 reference              0.0017 / [0.9998, 1.0001] / 0.0017    | 0.0017  | 0.9993  | 0.0017 / 0.002     | 0.0017 / 0.0017 | 0.999999  0.0017
    bf16 everywhere (device)    0.0034 / [0.9945, 1.0048] / 0.0062    | 0.0020  | 0.9992  | 0.0022 / 0.004     | 0.0021 / 0.0046 | 0.999997  0.0023
    eps 1e-6                    0.0030 / [0.9999, 1.0062] / 0.0065    | 0.0017  | 0.9993  | 0.0023 / 0.004     | 0.0017 / 0.0057 | 0.999998  0.0019 (passes)
    eps 0                       0.0033 / [0.9999, 1.0070] / 0.0071    | 0.0017  | 0.9993  | 0.0024 / 0.005     | 0.0018 / 0.0063 | 0.999998  0.0019 (passes)
    x 1.01                      0.0101 / [1.0098, 1.0101] / 0.0103    | 0.0017  | 0.9993  | 0.0060 / 0.007     | 0.0049 / 0.0089 | 0.999979  0.0066 (passes)
    x 1.02 (x 0.98 the same)    0.0201 / [1.0198, 1.0201] / 0.0202    | 0.0017  | 0.9993  | 0.0117 / 0.013     | 0.0092 / 0.0175 | 0.999919  0.0128 (passes)
    rows x U[0.979, 0.989]      0.0163 / [0.9790, 0.9891] / 0.0211    | 0.0017  | 0.9993  | 0.0097 / 0.014     | 0.0077 / 0.0184 | 0.999945  0.0106 (passes)
    RMS over half the columns   0.0087 / [0.9995, 1.0005] / 0.0297    | 0.0042  | 0.9992  | 0.0037 / 0.012     | 0.0035 / 0.0107 | 0.999992  0.0040 (passes)
    RMS over a quarter          0.0170 / [0.9995, 1.0010] / 0.0386    | 0.0076  | 0.9991  | 0.0066 / 0.016     | 0.0060 / 0.0160 | 0.999975  0.0070 (passes)
    LayerNorm instead of RMS    0.0115 / [0.9997, 1.0004] / 0.0348    | 0.0056  | 0.9992  | 0.0049 / 0.015     | 0.0047 / 0.0144 | 0.999985  0.0055 (passes)
    last row zeroed             0.0228 / [0.0,    1.0001] / 1.0       | 0.0218  | 0.9991  | 0.0169 / 0.69      | 0.0215 / 0.615  | 0.99961   0.0283 (passes)
    w halves swapped (TP)       0.283  / [0.9808, 1.0751] / 0.333     | 0.195   | 0.9373  | 0.165  / 0.221     | 0.129  / 0.277  | 0.98673   0.164  (passes)
    sum instead of mean         0.987  / [0.0128, 0.0128] / 0.987     | 0.0166  | 0.9926  | 0.966  / 1.29      | 0.677  / 1.70   | 0.781     0.867
    no weight / 1 + w           6.8 / 7.8 (~8x norms)                 | 0.18    | 0.94    | 0.74               | 0.67            | 0.74      0.85
    output column halves swapped 1.41 / [0.9998, 1.0001] / 1.45       | 1.33    | 0.8101  | 0.930  / 1.34      | 0.728  / 1.76   | 0.713     0.927
    SP row halves swapped       1.22   / [0.9121, 1.0963] / 1.44      | 0.929   | 0.9056  | 0.989  / 1.43      | 0.843  / 1.89   | 0.640     1.09
    zero stub                   1.0    / [0.0,    0.0]    / 1.0       | 1.0     | 0.7754  | 0.887  / 1.30      | 0.692  / 1.71   | 0.718     0.877

"(passes)" = passes the 0.98 out gate. The residual dominates block out and q_a_layernorm removes any per-row scale
of attn_norm from q_resid, so a scale bug moves out by only about 0.65x its own size; the eps bugs are invisible on
this golden (row rms >= 0.0267, mean(x^2) >= 70x eps). So the test also asserts (informational metrics):
  - attn_hc (gates) and attn_x vs golden at the swap 01 / 02 limits (gates rel <= 0.01, per-column max abs
    <= 0.015; attn_x rel <= 0.004, row ratio [0.995, 1.005], worst row <= 0.01), so upstream regressions still fail;
  - attn_norm (the swapped step) vs golden at the component limits (test_c_dense_full_attn_norm.py): finite, shape,
    rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row rel L2 <= 0.015 (catches x 1.01, the
    row-scale, RMS-subset, LayerNorm and zeroed-row bugs and everything worse);
  - attn_norm vs the CPU attn_norm on the same input (the device attn_x): rel L2 <= 0.008, worst row <= 0.015
    (the step's own error, apart from the device attn_x error);
  - the attn_norm module once more on the device attn_x x 0.1 (bf16) vs the CPU step on the same input: rel L2
    <= 0.01, worst row <= 0.02 (as the component test: eps 1e-6 scores 0.158, eps 2e-5 0.090, bf16 noise 0.003);
  - q_resid rel L2 <= 0.01; indexer top-k overlap vs golden >= 0.995 (reference 0.9993: near-tie flips);
  - attn_out rel L2 <= 0.01, worst row <= 0.05; h_mid rel L2 <= 0.01, worst row <= 0.05; block out finite, rel L2
    <= 0.01 (as swaps 01 / 02).
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
BLOCK_TYPE = "dense_full"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm"]
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
Q_MAX_REL_L2 = 0.01  # q_resid
TOPK_MIN_OVERLAP = 0.995  # indexer top-k selection vs golden (mean per-row set overlap, -1 pads ignored)
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
    if finite_shape("attn_norm"):
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

    # Downstream of attn_norm: q_resid, the indexer's top-k, attn_out.
    if finite_shape("q_resid"):
        rel_row("q_resid", Q_MAX_REL_L2, None)
    if "topk" in gl:
        if seen["topk"].numel() != gl["topk"].numel():
            failures.append(f"topk: shape {tuple(seen['topk'].shape)} vs golden {tuple(gl['topk'].shape)}")
        else:
            ov = _topk_overlap(seen["topk"], gl["topk"])
            metrics.record("topk_overlap_swap", ov)
            print(f"topk overlap vs golden: {ov:.5f} (>= {TOPK_MIN_OVERLAP})")
            if ov < TOPK_MIN_OVERLAP:
                failures.append(f"topk overlap {ov:.5f} < {TOPK_MIN_OVERLAP}")
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
