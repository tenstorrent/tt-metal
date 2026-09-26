# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 14: block type sliding (layer 0) with ffn_residual swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (sliding) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
    post_attn_norm
    attn_residual
    ffn_norm
    mlp
    post_mlp_norm
    router
    moe_norm
    experts
    post_moe_norm
    ffn_combine
    post_ffn_norm
    ffn_residual

Reviewed (S.sliding.14.test.1): swap 13's checks, SWAPPED extended with ffn_residual (out = (h_mid + ffn_out) *
layer_scalar, kind `residual`). Every step of the block is now on the device; ffn_residual's output is block out. It gets
the existing residual checks: its own output vs golden (PCC >= component threshold, rel L2 <= 0.03; block out also
<= 0.02) and vs the CPU residual on the same device h_mid and ffn_out (rel L2 <= 0.03, per-token norm ratio in
[0.97, 1.03]). That catches a dropped layer_scalar (rel 13.2, ratio 14.2), 2x and zeroed rows, which PCC misses
(test_c_sliding_ffn_residual.py). No new limits.
Measured (S.sliding.14.test.1): CPU reference pcc_swap_out 0.999996, block out rel 0.0027, ffn_residual iso 0.0; zero stub
fails (every check). Device (all 14 steps): pcc_swap_out 0.999948, block out rel 0.0102 / 0.0092; ffn_residual iso rel
0.0024, ratio [0.9972, 1.0042].
Earlier notes (S.sliding.13.test.1): swap 12's checks, SWAPPED extended with post_ffn_norm (ffn_out = rms(ffn_sum) *
post_feedforward_layernorm.weight, kind `norm`). Only ffn_residual (CPU, (h_mid + ffn_out) * layer_scalar) follows, so block
out sees a post_ffn_norm error only diluted by the h_mid residual. post_ffn_norm gets the existing norm checks: its own
output vs golden (PCC >= component threshold, rel L2 <= 0.03) and vs the CPU norm on the same device ffn_sum (rel L2 <= 0.03,
per-token norm ratio in [0.97, 1.03]). That catches `1 + w` (rel 0.047, ratio >= 1.042), sum instead of mean, 2x,
layer_scalar folded in and zeroed rows, which PCC misses (test_c_sliding_post_ffn_norm.py). No new limits.
Measured (S.sliding.13.test.1): CPU reference pcc_swap_out 0.999996, post_ffn_norm rel 0.0029 / iso 0.0; zero stub fails
(every check). Device: pcc_swap_out 0.999951, block out rel 0.0099 / 0.0090; post_ffn_norm pcc 0.99994, rel 0.0113 vs
golden, iso rel 0.0019, ratio [0.9974, 1.0013].
Earlier notes (S.sliding.12.test.1): swap 11's checks, SWAPPED extended with ffn_combine (ffn_sum = mlp_post_norm +
moe_post_norm, kind `residual`). post_ffn_norm (CPU) normalizes every row right after it, so a per-row scale error or
a zeroed row in ffn_combine barely moves block out. It gets the existing residual checks: its own output vs golden
(PCC >= component threshold, rel L2 <= 0.03) and vs the CPU add on the same device mlp_post_norm and moe_post_norm
(rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]). That catches 2x / 0.5x and zeroed rows, which PCC misses
(test_c_sliding_ffn_combine.py). A dropped operand already fails PCC. No new limits.
Earlier notes (S.sliding.11.test.1): swap 10's checks (test_swap_sliding_10_experts.py), SWAPPED extended with
post_moe_norm. post_moe_norm is a `norm` step (experts_out -> post_feedforward_layernorm_2), so it gets the existing norm
checks: its own output vs golden (PCC >= component threshold, rel L2 <= 0.03; device 0.0195, most of it the experts error
carried through) and vs the CPU norm on the same device experts_out (rel L2 <= 0.03, per-token norm ratio in
[0.97, 1.03]; device 0.0019, [0.9975, 1.0009]). The iso check catches `1 + w` (rel 0.099), sum instead of mean and
zeroed rows, which PCC misses (test_c_sliding_post_moe_norm.py) and which ffn_combine + post_ffn_norm partly hide at
block out. Earlier notes (S.sliding.10.test.1): experts (kind `moe`, inputs moe_norm and the dense router) is followed by
post_moe_norm, which normalizes every
row and hides a per-row scale error at block out, and upstream router selection flips move whole experts_out rows vs the
golden. So experts gets its own checks: vs golden PCC >= component threshold and rel L2 <= 0.05 (looser than 0.03: device
0.0266, of which the flips from upstream device error are a large share; a dropped expert scores >= 0.07), and vs the CPU
experts on the exact device moe_norm and router the component limits: rel L2 <= 0.03, per-token norm ratio in
[0.97, 1.03], worst per-token rel L2 <= 0.1 (dropped (token, expert) pair; test_c_sliding_experts.py).
Gated pcc_swap_out (PCC, float [2048, 2816], spec block threshold 0.98) plus asserted extras:
  - each swapped float step's own output vs golden: PCC >= spec component threshold, and rel L2 <= 0.03 except the router
    and experts (experts <= 0.05);
  - attention output over the first ``sliding_window`` rows: rel L2 <= 0.03 (stateful bugs);
  - block out rel L2 <= 0.02, whole chunk and first ``sliding_window`` rows;
  - every swapped norm, residual or mlp step vs the CPU step applied to the exact inputs the device step received:
    rel L2 <= 0.03 and per-token norm ratio in [0.97, 1.03];
  - router: exactly 8 nonzeros per row, no negative weights, and vs golden / vs the CPU router on the same device h_mid:
    selection overlap >= 0.99 / 0.995, matched-row weight rel L2 <= 0.01 / 0.005, per-row sum ratio in [0.99, 1.01];
  - experts vs the CPU experts on the same device inputs: rel L2 <= 0.03, norm ratio in [0.97, 1.03], worst row <= 0.1;
  - finite outputs.
CPU reference: pcc_swap_out 0.999996, block out rel 0.0027, experts rel 0.0031, post_moe_norm rel 0.0043 (golden) / 0.0
(iso); zero stub fails (every check).
Device (first run, S.sliding.11.test.1): pcc_swap_out 0.999953, block out rel 0.0096 / 0.0087; experts pcc 0.99969,
rel 0.0266 vs golden, iso rel 0.0183, row-norm ratio [0.9905, 1.0203], worst row 0.0273; router overlap 0.99725 vs
golden; post_moe_norm pcc 0.99981, rel 0.0195 vs golden, iso rel 0.0019, ratio [0.9975, 1.0009].
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
BLOCK_TYPE = "sliding"
SWAPPED = [
    "attn_norm",
    "attention",
    "post_attn_norm",
    "attn_residual",
    "ffn_norm",
    "mlp",
    "post_mlp_norm",
    "router",
    "moe_norm",
    "experts",
    "post_moe_norm",
    "ffn_combine",
    "post_ffn_norm",
    "ffn_residual",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = 0.03  # swapped step output: ||got - want|| / ||want||, whole chunk
STEP_MAX_REL_L2_PREFIX_ROWS = 0.03  # attention output, first sliding_window rows (the rows that read the KV prefix)
OUT_MAX_REL_L2 = 0.02  # block output, whole chunk and first sliding_window rows (device 0.0074: MoE routing amplifies)
PREFIX_ROW_STEPS = {"attention"}  # stateful steps whose first sliding_window rows are checked separately
ISO_STEP_KINDS = ("norm", "residual", "mlp")  # steps checked against the CPU step on the same (device) inputs
ISO_MAX_REL_L2 = 0.03  # norm/residual/mlp step vs the CPU step applied to the same (device) input
ROW_NORM_RATIO = (0.97, 1.03)  # norm/residual/mlp step: per-token ||got|| / ||cpu step(same input)||
ROUTER_STEP_KINDS = ("router",)  # dense [S, 128] routing: selection / weight checks instead of whole-matrix rel L2
TOP_K = int(S.get("checkpoint.config.top_k_experts", 8))
ROUTER_MIN_OVERLAP_GOLDEN = 0.99  # mean top-k selection overlap vs golden (upstream device error flips near ties)
ROUTER_MIN_OVERLAP_ISO = 0.995  # vs the CPU router on the same (device) h_mid; component test limit
ROUTER_MAX_MATCHED_REL_L2_GOLDEN = 0.01  # weights on matched rows vs golden (device 0.0044: upstream h_mid error)
ROUTER_MAX_MATCHED_REL_L2_ISO = 0.005  # vs the CPU router on the same h_mid (device 0.0022; no per_expert_scale 0.0105)
ROUTER_ROW_SUM_RATIO = (0.99, 1.01)  # per-token sum(got) / sum(want), golden and iso
MOE_STEP_KINDS = ("moe",)  # routed experts: iso checks incl. worst row (post_moe_norm hides per-row scale errors)
MOE_MAX_REL_L2_GOLDEN = 0.05  # experts_out vs golden, whole chunk (device 0.0266: upstream router flips + bfp8 kernel)
MOE_ISO_MAX_REL_L2 = 0.03  # vs the CPU experts on the same (device) moe_norm and router; component test limit
MOE_ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||cpu experts(same inputs)||; component test limit
MOE_ISO_MAX_ROW_REL_L2 = (
    0.1  # worst per-token rel L2 vs the CPU experts on the same inputs (dropped (token, expert) pair)
)


def _routing_stats(got, want):
    """(nnz per row, mean selection overlap, matched rows mask, matched-row rel L2, row-sum ratio min, max)."""
    gs, ws = got != 0, want != 0
    overlap = ((gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)).mean().item()
    m = (gs == ws).all(-1)
    mrel = ((got[m] - want[m]).norm() / want[m].norm().clamp_min(1e-12)).item() if m.any() else float("inf")
    ratio = got.sum(-1) / want.sum(-1).clamp_min(1e-12)
    return gs.sum(-1), overlap, m, mrel, ratio.min().item(), ratio.max().item()


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "swap chunk must start after 0 so the attention reads a real KV prefix"
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    overrides = {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
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
    window = int(S.get("checkpoint.config.sliding_window", 1024))
    want_out = gl["out"].float()
    got_out = seen["out"].float().reshape(want_out.shape)
    rows = min(window, want_out.shape[0])
    out_rel = _rel(got_out, want_out)
    out_rel_head = _rel(got_out[:rows], want_out[:rows])
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("rel_l2_prefix_rows_swap_out", out_rel_head)
    print(f"rel_l2_swap_out={out_rel:.6f} first_{rows}_rows={out_rel_head:.6f} (<= {OUT_MAX_REL_L2})")
    if not torch.isfinite(got_out).all():
        failures.append("block out non-finite")
    if out_rel > OUT_MAX_REL_L2 or out_rel_head > OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} / first {rows} rows {out_rel_head:.4f} > {OUT_MAX_REL_L2}")
    comp_thr = threshold(S, "component")
    for name in SWAPPED:
        o = _step(ref, layer, name).output
        want = gl[o]
        if not want.is_floating_point():
            continue
        if seen[o].numel() != want.numel():
            failures.append(f"swapped step {name}: shape {tuple(seen[o].shape)} vs golden {tuple(want.shape)}")
            continue
        got = seen[o].float().reshape(want.shape)
        v = metrics.pcc(got, want.float())
        rel = _rel(got, want)
        metrics.record(f"rel_l2_swap_{o}", rel)
        kind = _step(ref, layer, name).kind
        msg = f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={rel:.6f}"
        if not torch.isfinite(got).all():
            failures.append(f"swapped step {name}: non-finite output")
        if kind in ROUTER_STEP_KINDS:
            # Whole-matrix rel L2 is dominated by selection flips from upstream device error; check selection and weights.
            if v < comp_thr:
                failures.append(f"swapped step {name}: pcc {v:.4f} < {comp_thr}")
            st = _step(ref, layer, name)
            ins = [seen[i].float().reshape(gl[i].shape) if i in gl else seen[i].float() for i in st.inputs]
            iso = ref.component(layer, name)(rctx, *ins).float().reshape(want.shape)
            for tag, w, min_ov, max_mrel in (
                ("golden", want.float(), ROUTER_MIN_OVERLAP_GOLDEN, ROUTER_MAX_MATCHED_REL_L2_GOLDEN),
                ("iso", iso, ROUTER_MIN_OVERLAP_ISO, ROUTER_MAX_MATCHED_REL_L2_ISO),
            ):
                nnz, ov, m, mrel, rmin, rmax = _routing_stats(got, w)
                metrics.record(f"selection_overlap_{tag}_swap_{o}", ov)
                metrics.record(f"matched_rel_l2_{tag}_swap_{o}", mrel)
                metrics.record(f"row_sum_ratio_min_{tag}_swap_{o}", rmin)
                metrics.record(f"row_sum_ratio_max_{tag}_swap_{o}", rmax)
                msg += (
                    f"\n  vs {tag}: selection_overlap={ov:.5f} (>= {min_ov}) matched_rows={m.sum().item()}/{m.numel()} "
                    f"matched_rel_l2={mrel:.5f} (<= {max_mrel}) row_sum_ratio=[{rmin:.4f}, {rmax:.4f}] "
                    f"(in {ROUTER_ROW_SUM_RATIO})"
                )
                if ov < min_ov:
                    failures.append(f"swapped step {name}: top-{TOP_K} selection overlap vs {tag} {ov:.5f} < {min_ov}")
                if mrel > max_mrel:
                    failures.append(
                        f"swapped step {name}: matched-row weight rel L2 vs {tag} {mrel:.5f} > {max_mrel} "
                        "(per_expert_scale or renorm bug)"
                    )
                if not (ROUTER_ROW_SUM_RATIO[0] <= rmin and rmax <= ROUTER_ROW_SUM_RATIO[1]):
                    failures.append(
                        f"swapped step {name}: row-sum ratio vs {tag} [{rmin:.4f}, {rmax:.4f}] outside {ROUTER_ROW_SUM_RATIO}"
                    )
            msg += f"\n  nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K})"
            if not (nnz == TOP_K).all():
                failures.append(
                    f"swapped step {name}: nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want exactly {TOP_K}"
                )
            if (got < 0).any():
                failures.append(f"swapped step {name}: negative routing weight")
        elif kind in MOE_STEP_KINDS:
            if v < comp_thr or rel > MOE_MAX_REL_L2_GOLDEN:
                failures.append(
                    f"swapped step {name}: pcc {v:.4f} / rel L2 vs golden {rel:.4f} > {MOE_MAX_REL_L2_GOLDEN}"
                )
            # Isolate from upstream error (router flips move whole rows): the CPU experts on the device moe_norm and router.
            st = _step(ref, layer, name)
            ins = [seen[i].float().reshape(gl[i].shape) if i in gl else seen[i].float() for i in st.inputs]
            iso = ref.component(layer, name)(rctx, *ins).float().reshape(want.shape)
            iso_rel = _rel(got, iso)
            isn = iso.norm(dim=-1).clamp_min(1e-12)
            ratio = got.norm(dim=-1) / isn
            rmin, rmax = ratio.min().item(), ratio.max().item()
            row_rel = (got - iso).norm(dim=-1) / isn
            worst = row_rel.max().item()
            metrics.record(f"rel_l2_iso_swap_{o}", iso_rel)
            metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
            metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
            metrics.record(f"max_row_rel_l2_iso_swap_{o}", worst)
            msg += (
                f" (<= {MOE_MAX_REL_L2_GOLDEN})\n  vs CPU experts on the same inputs: rel_l2={iso_rel:.6f} (<= {MOE_ISO_MAX_REL_L2}) "
                f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {MOE_ROW_NORM_RATIO}) max_row_rel_l2={worst:.5f} "
                f"(<= {MOE_ISO_MAX_ROW_REL_L2}, worst row {row_rel.argmax().item()})"
            )
            if iso_rel > MOE_ISO_MAX_REL_L2:
                failures.append(
                    f"swapped step {name}: vs CPU experts on the same inputs rel L2 {iso_rel:.4f} > {MOE_ISO_MAX_REL_L2}"
                )
            if not (MOE_ROW_NORM_RATIO[0] <= rmin and rmax <= MOE_ROW_NORM_RATIO[1]):
                failures.append(
                    f"swapped step {name}: per-token norm ratio vs CPU experts [{rmin:.4f}, {rmax:.4f}] outside {MOE_ROW_NORM_RATIO}"
                )
            if worst > MOE_ISO_MAX_ROW_REL_L2:
                failures.append(
                    f"swapped step {name}: worst per-token rel L2 vs CPU experts {worst:.4f} > {MOE_ISO_MAX_ROW_REL_L2} "
                    f"at row {row_rel.argmax().item()} (dropped (token, expert) pair?)"
                )
        elif v < comp_thr or rel > STEP_MAX_REL_L2:
            failures.append(f"swapped step {name}: pcc {v:.4f} / rel L2 {rel:.4f} (scale, weight or state bug)")
        else:
            msg += f" (<= {STEP_MAX_REL_L2})"
        if name in PREFIX_ROW_STEPS:
            r = min(window, want.shape[0])
            rel_head = _rel(got[:r], want[:r])
            metrics.record(f"rel_l2_prefix_rows_swap_{o}", rel_head)
            msg += f" first_{r}_rows={rel_head:.6f} (<= {STEP_MAX_REL_L2_PREFIX_ROWS})"
            if rel_head > STEP_MAX_REL_L2_PREFIX_ROWS:
                failures.append(
                    f"swapped step {name}: rel L2 on the first {r} rows {rel_head:.4f} > {STEP_MAX_REL_L2_PREFIX_ROWS} "
                    "(KV prefix ignored or RoPE positions not offset by the chunk start?)"
                )
        if kind in ISO_STEP_KINDS:
            # Isolate the step from the upstream device error: the CPU step on the exact inputs the device step saw.
            st = _step(ref, layer, name)
            ins = [seen[i].float().reshape(gl[i].shape) if i in gl else seen[i].float() for i in st.inputs]
            iso = ref.component(layer, name)(rctx, *ins).float().reshape(want.shape)
            iso_rel = _rel(got, iso)
            ratio = got.norm(dim=-1) / iso.norm(dim=-1).clamp_min(1e-12)
            rmin, rmax = ratio.min().item(), ratio.max().item()
            metrics.record(f"rel_l2_iso_swap_{o}", iso_rel)
            metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
            metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
            msg += f" iso_rel_l2={iso_rel:.6f} (<= {ISO_MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO})"
            if iso_rel > ISO_MAX_REL_L2 or not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
                failures.append(
                    f"swapped step {name}: vs CPU step on the same input rel L2 {iso_rel:.4f}, "
                    f"row-norm ratio [{rmin:.4f}, {rmax:.4f}] (weight or scale bug)"
                )
        print(msg)
    assert not failures, "; ".join(failures)
