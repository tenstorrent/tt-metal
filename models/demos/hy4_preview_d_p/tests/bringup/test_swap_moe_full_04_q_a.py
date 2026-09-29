"""Swap test 4: block type moe_full (layer 1) with q_a swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (moe_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a

Reviewed (S.moe_full.04.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). q_resid [S, 2048] = q_a_layernorm(q_a_proj(attn_norm)) (eps 1e-6, K = 6144
split over the mesh columns) feeds only the indexer queries (top-2048) and the attention queries (q_b); then come
attn_residual and the MoE half (ffn_hc, router top-8 of 256, experts, shared expert). Measured on the CPU (golden
s4096 chunk 1, 2048 rows; q_a replaced by mutations of the fp32 step, every other step the fp32 CPU reference).
"q" = q_resid vs golden rel L2 / row norm ratio / worst row, "tk" = indexer top-k set overlap, "ao" = attn_out rel /
worst row, "syn" = the step on golden attn_norm x 0.01 (bf16) vs the CPU step (rel / worst row), "hm" = h_mid rel /
worst (row, stream), "rt" = router top-8 overlap, "out" = PCC / rel L2 / worst (row, stream):

    variant                       q                            | tk     | ao           | syn           | hm            | rt     | out
    fp32 reference                0.0018 [0.9997, 1.0003] 0.0020 | 0.9991 | 0.0018 / 0.004 | 0      / 0      | 0.0021 / 0.0029 | 0.9987 | 0.99996 0.0024 0.056
    bf16 x / W, bf16 pre and out  0.0030 [0.9995, 1.0004] 0.0036 | 0.9991 | 0.0018 / 0.004 | 0.0023 / 0.0027 | 0.0021 / 0.0033 | 0.9984 | 0.99996 0.0025 0.056
    eps 1e-5                      0.0018 [0.9997, 1.0003] 0.0020 | 0.9991 | 0.0018 / 0.004 | 0.168  / 0.222  | 0.0021 / 0.0029 | 0.9987 | 0.99996 0.0024 (passes)
    eps 0                         0.0018 [0.9997, 1.0003] 0.0020 | 0.9991 | 0.0018 / 0.004 | 0.026  / 0.038  | 0.0021 / 0.0029 | 0.9987 | 0.99996 0.0024 (passes)
    eps 2e-6                      0.0018 [0.9997, 1.0003] 0.0020 | 0.9991 | 0.0018 / 0.004 | 0.024  / 0.034  | 0.0021 / 0.0029 | 0.9987 | 0.99996 0.0024 (passes)
    x 1.01                        0.0102 [1.0097, 1.0103] 0.0105 | 0.9991 | 0.0050 / 0.008 | 0.010  / 0.010  | 0.0033 / 0.0057 | 0.9961 | 0.99994 0.0045 (passes)
    x 1.02                        0.0201 [1.0197, 1.0203] 0.0204 | 0.9991 | 0.0094 / 0.015 | 0.020  / 0.020  | 0.0054 / 0.0104 | 0.9921 | 0.99992 0.0074 (passes)
    RMS over half the columns     0.0179 [0.9707, 1.0542] 0.054  | 0.9991 | 0.0081 / 0.029 | 0.017  / 0.051  | 0.0046 / 0.0186 | 0.9944 | 0.99992 0.0081 (passes)
    last row zeroed               0.0220 [0.0,    1.0003] 1.0    | 0.9989 | 0.0137 / 0.657 | 0.022  / 1.0    | 0.0115 / 0.42   | 0.9985 | 0.99992 0.0085 (passes)
    last tile row (32) zeroed     0.125  [0.0,    1.0003] 1.0    | 0.9940 | 0.086  / 0.839 | 0.125  / 1.0    | 0.050  / 0.68   | 0.9911 | 0.99866 0.051  (passes)
    only one K half (no reduce)   0.403  [0.9714, 0.9965] 0.525  | 0.9652 | 0.074  / 0.173 | 0.397  / 0.513  | 0.045  / 0.106  | 0.9454 | 0.99909 0.044  (passes)
    norm w halves swapped         0.177  [0.9009, 0.9563] 0.201  | 0.9820 | 0.057  / 0.091 | 0.177  / 0.201  | 0.032  / 0.068  | 0.9579 | 0.99944 0.035  (passes)
    norm per K partial, then sum  0.826  [1.7157, 1.8710] 0.871  | 0.9988 | 0.265  / 0.474 | 0.730  / 0.779  | 0.133  / 0.316  | 0.8394 | 0.99517 0.111  (passes)
    no norm weight                4.58   [5.4334, 5.7297] 4.74   | 0.9836 | 0.575  / 0.861 | 4.58   / 4.74   | 0.307  / 0.622  | 0.6978 | 0.97850 0.226
    1 + w                         5.57   [6.4246, 6.7196] 5.73   | 0.9861 | 0.597  / 0.884 | 5.57   / 5.73   | 0.320  / 0.639  | 0.6889 | 0.97700 0.235
    row halves swapped (SP order) 0.848  [0.9694, 1.0316] 1.10   | 0.9065 | 0.437  / 0.992 | 0.849  / 1.10   | 0.243  / 0.669  | 0.7368 | 0.96976 0.246
    zero stub                     1.0    [0.0,    0.0]    1.0    | 0.7554 | 0.701  / 0.861 | 1.0    / 1.0    | 0.381  / 0.698  | 0.5305 | 0.93151 0.372

"(passes)" = passes the 0.98 out gate: every eps, scale, subset, zeroed-row, missing-reduce, per-partial-norm and
TP-weight bug passes it (the residual dominates block out). The out worst (row, stream) rel is 0.056 even for the fp32
reference (near-tie expert flips), so it is recorded, not asserted. The test also asserts (informational metrics):
  - attn_hc (the gates) and attn_x at the swap 02 / 03 limits: gates rel L2 <= 0.01, per column <= 0.01, post worst
    row <= 0.015; attn_x vs golden rel <= 0.005, row norm ratio in [0.996, 1.004], worst row <= 0.01; attn_x vs the
    CPU hc_pre on the same inputs 0.003 / 0.006; the attn_hc_pre module again with rotated pre gates 0.004 / 0.01;
  - attn_norm at the swap 03 limits: vs golden rel <= 0.008, row ratio in [0.993, 1.007], worst row <= 0.015; vs the
    CPU attn_norm on the device attn_x (same limits); the module on attn_x x 0.1 vs CPU 0.01 / 0.02 (eps);
  - q_resid (the swapped step) vs golden at the component limits (test_c_moe_full_q_a.py): finite, shape, rel L2
    <= 0.008, every row's norm ratio in [0.994, 1.006], worst row <= 0.015 (catches every q mutation above except
    eps); vs the CPU q_a on the device attn_norm (same limits, the step's own error);
  - the q_a module again on the device attn_norm x 0.01 (bf16) vs the CPU step on the same input: rel L2 <= 0.01,
    worst row <= 0.02 (the golden cannot see eps: eps 1e-5 0.168, eps 0 0.026, eps 2e-6 0.024 vs bf16 0.0023);
  - indexer top-k overlap >= 0.995 (reference 0.9991); attn_out rel <= 0.01, worst row <= 0.05; h_mid rel <= 0.005,
    worst (row, stream) <= 0.02; router top-8 overlap >= 0.99 (reference 0.9987, x 1.02 0.9921); block out finite,
    rel <= 0.01.
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
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a"]
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

    # Downstream of q_a: the indexer's top-k, attn_out.
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
