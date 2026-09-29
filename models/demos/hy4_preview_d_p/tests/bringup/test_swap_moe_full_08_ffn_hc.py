# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 8: block type moe_full (layer 1) with ffn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (moe_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    indexer
    attention
    attn_residual
    ffn_hc

Reviewed (S.moe_full.08.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). The swapped step is ffn_hc, the iHC gates [S, 8] fp32 (pre 0-3 | post 4-7) from
h_mid with the hc_mlp_layer weights of layer 1. The gate columns span six orders of magnitude (means pre 1e-5, 1.5e-6,
0.93, 0.33, post 1e-4, 8e-5, 1.6e-6, 0.023; columns 0 / 1 / 6 within ~10x of hc_eps), and the h_mid row RMS is small
(0.006 .. 0.08), so rms_norm_eps 1e-5 matters at the block's own scale. The pre gates make ffn_x (then ffn_norm, the
MoE); the post gates scale mlp_out into each stream (ffn_residual -> out; only post 7 moves out much). Measured on the
CPU (golden s4096 chunk 1, 2048 rows; ffn_hc replaced by mutations of the fp32 step, every other step the fp32 CPU
reference; study /tmp/hy4_sm8/study.py, outside the repo). "gates c" = vs the CPU ffn_hc on the same h_mid: rel /
worst of columns 2, 3, 4, 5, 7 / worst of columns 0, 1, 6 / post worst row; "ffn_x c" = rel / row; "post" = out vs
ffn_residual with the CPU gates and the same mlp_out, worst stream / row; "tail" = out vs the whole CPU tail, rel:

    variant                      gates c                    | ffn_x c        | post           | tail    | out PCC
    fp32 reference               0 / 0 / 0 / 0              | 0 / 0          | 0 / 0          | 0       | 0.999997
    bf16 output                  0.0013 / 0.0017 / 0.0023 / 0.0039 | 0.0013 / 0.0046 | 0.0013 / 0.0032 | 0.0022 | 0.999995 (ok)
    bf16 input and fn            0.0001 / 0.0005 / 0.0010 / 0.0007 | 0.0001 / 0.0005 | 0.0002 / 0.0005 | 0.0002 | 0.999997 (ok)
    sigmoid abs err 1e-4         0.0005 / 1.42 / 66 / 0.085 | 0.0002 / 0.0005 | 0.025 / 0.27   | 0.0091  | 0.999955 (p)
    pre x 1.01                   0.010 / 0.010 / 0.010 / 0  | 0.010 / 0.010  | 0 / 0          | 0.0006  | 0.999997 (p)
    post x 1.01                  0.0003 / 0.010 / 0.010 / 0.010 | 0 / 0      | 0.0079 / 0.0095 | 0.0075 | 0.999989 (p)
    post x 1.1                   0.0028 / 0.10 / 0.10 / 0.10 | 0 / 0         | 0.079 / 0.095  | 0.075   | 0.999296 (p)
    gate 0 / 1 / 6 zeroed        0 / 0 / 1.0 / <= 0.002     | 0 / 0          | <= 0.0004 / 0.003 | 1e-4 | 0.999997 (p)
    gate 1 = gate 0              0 / 0 / 7.9 / 0            | 0 / 0          | 0 / 0          | 0       | 0.999997 (p)
    gate 4 / 5 zeroed            0.0002 / 1.0 / 0 / 0.086   | 0 / 0          | 0.028 / 0.22   | 0.0054  | 0.999982 (p)
    base 0 / 1 swapped           0 / 0 / 6.8 / 0            | 0 / 0          | 0 / 0          | 0       | 0.999997 (p)
    base 2 / 3 swapped           0.11 / 0.28 / 0 / 0        | 0.16 / 0.42    | 0 / 0          | 0.13    | 0.993984 (p)
    base 5 / 6 swapped           0 / 0.18 / 0.21 / 0.016    | 0 / 0          | 0.0018 / 0.040 | 0.0003  | 0.999997 (p)
    fn rows 4 / 5 swapped        0.0005 / 3.4 / 0 / 0.093   | 0 / 0          | 0.067 / 0.36   | 0.013   | 0.999911 (p)
    fn streams 0 / 1 swapped     0.010 / 0.55 / 0.28 / 0.11 | 0.016 / 0.040  | 0.029 / 0.14   | 0.035   | 0.999777 (p)
    last row = previous row      0.0012 / 0.018 / 0.002 / 0.61 | 0.0030 / 0.10 | 0.017 / 0.49 | 0.016   | 0.999864 (p)
    last row zeroed              0.021 / 0.029 / 0.018 / 1.0 | 0.029 / 1.0   | 0.028 / 0.80   | 0.026   | 0.999665 (p)
    rms_norm_eps 1e-6 / 1e-4     0.017 / 0.40 / 0.25 / 0.48; 0.089 / 15 / 12 / 8.7 | 0.007; 0.034 | ... | 0.020; 0.23 | 0.99985; 0.98351 (p)
    hc_eps dropped               0 / 0.0071 / 0.66 / 0.0004 | 0 / 0          | 0.0001 / 0.0014 | 5e-5   | 0.999997 (p)
    post = 1 x sigmoid, gate 3 = gate 2, gate 7 zeroed, fn rows 1 / 2 or 6 / 7, fn streams 2 / 3, one chip's partial
    sumsq, RMS over one stream, chip-major columns, attn_hc weights, pre | post halves swapped, zero stub: out PCC
    <= 0.961 (fail)

"(p)" = passes the 0.98 out gate: 23 of 35 mutations do, 21 of them real bugs (sigmoid abs err 1e-3, not listed,
passes at 0.995854 and fails every column check). Several (gates 0 / 1 / 6, base 0 / 1, hc_eps) change out by
<= 1e-4 and are visible only per gate column. So the test also asserts (informational metrics):
  - everything swap 07 (test_swap_moe_full_07_attn_residual.py) asserts, at its limits: attn_hc, attn_x (+ vs CPU,
    rotated pre gates), attn_norm and q_resid (vs golden, vs the CPU step, eps checks), topk (overlap vs golden and
    vs the CPU indexer, structure, chunk 0 exact), attn_out five ways, h_mid (vs golden, stream ratio, vs the CPU
    attn_residual, addend, rotated post gates), router top-8 overlap >= 0.99, block out finite and rel L2 <= 0.01;
  - ffn_hc vs golden: rel L2 <= 0.01, per column <= 0.015 (columns 0 / 1 / 6: <= 0.03), post worst row <= 0.02 (the
    component's limits widened for the device h_mid, rel 0.004 off the golden); ffn_x vs golden rel <= 0.01, worst
    row <= 0.05;
  - ffn_hc vs the CPU ffn_hc on the same device h_mid: rel L2 <= 0.005, per column <= 0.015 (columns 0 / 1 / 6:
    <= 0.02), post worst row <= 0.015; the pre gates through ffn_x (CPU ffn_hc_pre from the device gates vs from the
    CPU gates): rel <= 0.005, worst row <= 0.02;
  - the post gates through out: block out vs ffn_residual(h_mid, CPU gates, the block's own mlp_out), per stream
    rel <= 0.005, worst (row, stream) <= 0.02 (same mlp_out, so no routing flips in between);
  - block out vs the whole CPU tail (ffn_hc .. ffn_residual) from the device h_mid: rel <= 0.01 (worst row not gated:
    near-tie expert flips give 0.055 even for a bf16 output).
Every "(p)" bug above fails one of these; bf16 rounding of the input, fn or output passes them. No scaled-input eps
probe (layer 0's swap 08 has one): here rms_norm_eps 1e-6 already fails the per-column check at the block's scale.
The trail line pcc_swap_topk is positional match (template default) and is not meaningful for the device's unsorted
indices.
Device run (this gate): pcc_swap_out 0.999983; ffn_hc vs golden 0.00155, columns [0.0146, 0.0035, 0.0013, 0.0028,
0.0054, 0.0059, 0.0095, 0.0032], post row 0.0139; vs CPU 0.00036, columns [0.0099, 0.0042, 0.0003, 0.0007, 0.0074,
0.0089, 0.0105, 0.0040], post row 0.0095; ffn_x vs golden 0.0043 / 0.0128, vs CPU 0.00048 / 0.0012; post through out
stream <= 0.0032 / row 0.0054; tail 0.0033; router 0.99377; out rel 0.00608.
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
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "indexer", "attention", "attn_residual", "ffn_hc"]
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
H_MID_MAX_REL = 0.007  # h_mid vs golden, whole tensor (as swap 06)
H_MID_MAX_ROW_REL = 0.025  # h_mid, worst (row, stream) (as swap 06)
MID_STREAM_RATIO = (0.98, 1.02)  # h_mid per token and stream ||got|| / ||want|| vs golden
RES_MAX_REL_L2 = 5e-4  # h_mid vs the CPU attn_residual on the same inputs (device fp32: bit-identical), and rotated
RES_MAX_ROW_REL = 1e-3  # the same, worst (row, stream)
ADD_COEF_TOL = 0.01  # per stream |coef_j - 1| <= this + ROUND_MULT * r_j / ||t_j|| (test_c_moe_full_attn_residual.py)
MAX_ADD_REL = 0.01  # per stream ||delta_j - t_j|| <= this * ||t_j|| + ROUND_MULT * r_j
MAX_ADD_ROW_REL = 0.05  # per token ||delta - t|| <= this * ||t|| + ROUND_MULT * r + ROW_FLOOR
ROUND_MULT = 2.0  # allowance for the bf16 rounding of the output
ROW_FLOOR = 1e-6
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
ATT_MAX_REL_L2 = 0.015  # attn_out (the swapped step), whole tensor: vs golden, vs CPU, chunk 0, probe, scaled
ATT_RATIO = (0.985, 1.015)  # attn_out per-row ||got|| / ||want|| (test_c_moe_full_attention.py limits)
ATT_MAX_ROW_REL = 0.04  # attn_out, worst token row
PROBE_KEYS = 64  # probe topk: this many random causal positions per row, unsorted, the rest -1
PROBE_SEED = 0
ATT_SYN_SCALE = 1e-3  # eps check: the device attn_norm x ATT_SYN_SCALE (bf16), where kv_a_layernorm's eps matters
ROUTER_MIN_OVERLAP = 0.99  # router top-8 selection overlap vs golden
FHC_SMALL_COLS = (0, 1, 6)  # ffn_hc gates within ~10x of hc_eps (column means 1e-5, 1.5e-6, 1.6e-6)
FHC_MAX_REL_L2 = 0.01  # ffn_hc [S, 8] vs golden, whole tensor
FHC_MAX_COL_REL = 0.015  # ffn_hc vs golden, per column rel L2 (columns 2, 3, 4, 5, 7)
FHC_MAX_SMALL_COL_REL = 0.03  # the same on FHC_SMALL_COLS
FHC_MAX_POST_ROW_REL = 0.02  # ffn_hc vs golden, worst row over the post columns
FHC_CPU_MAX_REL_L2 = 0.005  # ffn_hc vs the CPU ffn_hc on the same device h_mid
FHC_CPU_MAX_COL_REL = 0.015  # the same, per column (columns 2, 3, 4, 5, 7)
FHC_CPU_MAX_SMALL_COL_REL = 0.02  # the same on FHC_SMALL_COLS
FHC_CPU_MAX_POST_ROW_REL = 0.015  # the same, worst row over the post columns
FX_MAX_REL_L2 = 0.01  # ffn_x (CPU ffn_hc_pre from the device gates) vs golden
FX_MAX_ROW_REL = 0.05  # the same, worst row
FX_CPU_MAX_REL_L2 = 0.005  # ffn_x from the device gates vs from the CPU gates on the same h_mid
FX_CPU_MAX_ROW_REL = 0.02  # the same, worst row
POST_MAX_STREAM_REL = 0.005  # out vs ffn_residual(h_mid, CPU gates, the block's mlp_out): per stream rel L2
POST_MAX_ROW_REL = 0.02  # the same, worst (row, stream)
TAIL_MAX_REL_L2 = 0.01  # block out vs the CPU tail (ffn_hc .. ffn_residual) from the same device h_mid
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


def _rotated_post(gates):
    """Each row's post gates (columns 4-7) rotated by (row mod 4), so every stream meets the large post gates."""
    n = gates.shape[0]
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, HC_MULT:] = torch.gather(gates[:, HC_MULT:], 1, idx)
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


def _probe_topk(start, rows, k, n, seed):
    """[rows, k] int64: per row min(n, pos + 1) distinct random positions in [0, pos], unsorted, then -1 pads."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.full((rows, k), -1, dtype=torch.int64)
    for i in range(rows):
        p = start + i
        m = min(n, p + 1)
        out[i, :m] = torch.randperm(p + 1, generator=gen)[:m]
    return out


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

    # attn_out (the swapped step): vs golden, vs the CPU step on the same device inputs, then the module on chunk 0,
    # on a probe topk and on a scaled attn_norm, each vs the CPU step on the same inputs.
    def att_check(tag, got, want):
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"attn_out {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        rel, row = _rel(got, want), _worst_row_rel(got, want)
        rmin, rmax = _ratio(got, want)
        metrics.record(f"rel_l2_swap_attn_out_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_attn_out_{tag}", row)
        print(
            f"attn_out {tag}: rel_l2={rel:.6f} (<= {ATT_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(ATT_RATIO)}) worst_row_rel_l2={row:.5f} (<= {ATT_MAX_ROW_REL})"
        )
        if rel > ATT_MAX_REL_L2:
            failures.append(f"attn_out {tag}: rel L2 {rel:.5f} > {ATT_MAX_REL_L2}")
        if not (ATT_RATIO[0] <= rmin and rmax <= ATT_RATIO[1]):
            failures.append(f"attn_out {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ATT_RATIO)}")
        if row > ATT_MAX_ROW_REL:
            failures.append(f"attn_out {tag}: worst row rel L2 {row:.5f} > {ATT_MAX_ROW_REL}")

    if finite_shape("attn_out"):
        att_check("vs_golden", seen["attn_out"], gl["attn_out"])
        if n_ok and "q_resid" in seen and got_tk is not None:
            cpu = ref.component(layer, "attention")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            qr = seen["q_resid"].float().reshape(gl["q_resid"].shape)
            att_check("vs_cpu", seen["attn_out"], cpu(reference_ctx(ref, layer, g, c), xn, qr, got_tk).float())

            # (c) golden chunk 0: start 0, empty prefix, -1 pads in every row.
            g0 = g.layer(0, layer)
            in0 = (g0["attn_norm"].float(), g0["q_resid"].float(), g0["topk"].long())
            dctx0 = Ctx(layer, 0, g.chunk, None, {"state_prefix": g.state(layer), "prefix_len": 0, "max_seq": g.seq})
            want0 = cpu(reference_ctx(ref, layer, g, 0), *in0).float()
            att_check("chunk0", muts["attention"](reference_ctx(ref, layer, g, 0), dctx0, *in0), want0)

            # (d) probe topk: 64 random causal positions per row, unsorted, the rest -1.
            start = c * g.chunk
            ptk = _probe_topk(start, xn.shape[0], gl["topk"].shape[-1], PROBE_KEYS, PROBE_SEED)
            pwant = cpu(reference_ctx(ref, layer, g, c), xn, qr, ptk).float()
            att_check("probe", muts["attention"](reference_ctx(ref, layer, g, c), dctx, xn, qr, ptk), pwant)

            # (e) eps: attn_norm scaled so kv_a_layernorm's eps 1e-6 is visible.
            xs = (xn * ATT_SYN_SCALE).bfloat16().float()
            swant = cpu(reference_ctx(ref, layer, g, c), xs, qr, got_tk).float()
            att_check("scaled", muts["attention"](reference_ctx(ref, layer, g, c), dctx, xs, qr, got_tk), swant)
        else:
            failures.append("attn_out: vs-CPU, chunk 0, probe and scaled checks skipped (unusable upstream output)")

    # h_mid (the swapped step): vs golden (+ per-stream norm ratio), vs the CPU step on the same device inputs (+ the
    # addend per stream), and the module on rotated post gates vs the CPU step.
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
        want = gl["h_mid"].float()
        n = want.shape[0]
        hm = seen["h_mid"].float().reshape(n, HC_MULT, -1)
        sr = hm.norm(dim=-1) / want.view(n, HC_MULT, -1).norm(dim=-1).clamp_min(1e-12)
        smin, smax = sr.min().item(), sr.max().item()
        metrics.record("stream_norm_ratio_min_swap_h_mid", smin)
        metrics.record("stream_norm_ratio_max_swap_h_mid", smax)
        print(f"h_mid per-token stream norm ratio=[{smin:.5f}, {smax:.5f}] (in {list(MID_STREAM_RATIO)})")
        if not (MID_STREAM_RATIO[0] <= smin and smax <= MID_STREAM_RATIO[1]):
            failures.append(f"h_mid: stream norm ratio [{smin:.5f}, {smax:.5f}] outside {list(MID_STREAM_RATIO)}")

        a_ok = seen["attn_out"].numel() == gl["attn_out"].numel() and torch.isfinite(seen["attn_out"].float()).all()
        g_ok = seen["attn_hc"].numel() == gl["attn_hc"].numel() and torch.isfinite(seen["attn_hc"].float()).all()
        if a_ok and g_ok:
            cpu_r = ref.component(layer, "attn_residual")
            xs_in = gl["in"].float()
            gt = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            y = seen["attn_out"].float().reshape(gl["attn_out"].shape)
            same = cpu_r(rctx, xs_in, gt, y).float()
            crel, crow = _rel(seen["h_mid"], same), _worst_row_rel(seen["h_mid"], same, HC_MULT)
            metrics.record("rel_l2_swap_h_mid_vs_cpu", crel)
            metrics.record("worst_row_rel_l2_swap_h_mid_vs_cpu", crow)
            print(
                f"h_mid vs CPU attn_residual on the same inputs: rel_l2={crel:.7f} (<= {RES_MAX_REL_L2}) "
                f"worst (row, stream) rel={crow:.7f} (<= {RES_MAX_ROW_REL})"
            )
            if crel > RES_MAX_REL_L2 or crow > RES_MAX_ROW_REL:
                failures.append(f"h_mid vs CPU on the same inputs: rel {crel:.6f} / worst row {crow:.6f}")

            # The addend on each stream: delta_j = h_mid_j - in_j vs t_j = post_j * attn_out, rounding-aware limits
            # (the component's; r = bf16 rounding error of the exact fp32 result), float64 statistics.
            xs = xs_in.view(n, HC_MULT, -1)
            tgt32 = gt[:, HC_MULT:].unsqueeze(-1) * y.view(n, 1, -1)
            exact = xs + tgt32
            rnd = (exact.bfloat16().float() - exact).double()
            tgt = tgt32.double()
            delta = (hm - xs).double()
            err = delta - tgt
            tn = tgt.norm(dim=(0, 2)).clamp_min(1e-30)
            rs = rnd.norm(dim=(0, 2))
            coef = (delta * tgt).sum(dim=(0, 2)) / (tn * tn)
            coef_x = ((coef - 1).abs() / (ADD_COEF_TOL + ROUND_MULT * rs / tn)).tolist()  # > 1 fails
            excess = (err.norm(dim=(0, 2)) / (MAX_ADD_REL * tn + ROUND_MULT * rs)).tolist()  # > 1 fails
            row_x = err.norm(dim=-1) / (MAX_ADD_ROW_REL * tgt.norm(dim=-1) + ROUND_MULT * rnd.norm(dim=-1) + ROW_FLOOR)
            worst = row_x.max().item()
            worst_at = divmod(int(row_x.argmax().item()), HC_MULT)
            metrics.record("add_coef_min_swap_h_mid", coef.min().item())
            metrics.record("add_coef_max_swap_h_mid", coef.max().item())
            metrics.record("add_excess_swap_h_mid", max(excess))
            metrics.record("add_worst_row_excess_swap_h_mid", worst)
            print(
                f"h_mid addend per stream: coef={[round(v, 5) for v in coef.tolist()]} (|coef-1|/tol="
                f"{[round(v, 3) for v in coef_x]} <= 1) excess={[round(v, 3) for v in excess]} (<= 1) "
                f"worst row excess={worst:.3f} (<= 1) at (row, stream)={worst_at}"
            )
            bad = [j for j, v in enumerate(coef_x) if v > 1]
            if bad:
                failures.append(f"h_mid addend: coefficient off on streams {bad}: {coef.tolist()}")
            bad = [j for j, v in enumerate(excess) if v > 1]
            if bad:
                failures.append(f"h_mid addend: error above the limit on streams {bad}: excess {excess}")
            if worst > 1:
                failures.append(f"h_mid addend: worst row excess {worst:.3f} at (row, stream) {worst_at}")

            # Rotated post gates: every stream meets the large gates on a quarter of the rows.
            gr = _rotated_post(gt)
            rwant = cpu_r(rctx, xs_in, gr, y).float()
            rout = muts["attn_residual"](rctx, dctx, xs_in, gr, y)
            if rout.numel() != rwant.numel() or not torch.isfinite(rout.float()).all():
                failures.append(f"attn_residual rotated post gates: shape {tuple(rout.shape)} or non-finite")
            else:
                prel, prow = _rel(rout, rwant), _worst_row_rel(rout, rwant, HC_MULT)
                metrics.record("rot_rel_l2_swap_h_mid", prel)
                metrics.record("rot_worst_row_rel_l2_swap_h_mid", prow)
                print(
                    f"attn_residual with rotated post gates vs CPU: rel_l2={prel:.7f} (<= {RES_MAX_REL_L2}) "
                    f"worst (row, stream) rel={prow:.7f} (<= {RES_MAX_ROW_REL})"
                )
                if prel > RES_MAX_REL_L2 or prow > RES_MAX_ROW_REL:
                    failures.append(f"attn_residual rotated post gates: rel {prel:.6f} / worst row {prow:.6f}")
        else:
            failures.append("h_mid: attn_hc / attn_out unusable, cannot check attn_residual against the CPU step")

    # ffn_hc (the swapped step): the iHC gates [S, 8] from h_mid, vs golden and vs the CPU ffn_hc on the same device
    # h_mid; per column (the small columns 0 / 1 / 6 carry bugs nothing downstream sees), the pre gates through ffn_x,
    # the post gates through out with the block's own mlp_out (no routing in between).
    def gate_check(tag, got, want, max_rel, max_col, max_small, max_row):
        err = got - want
        rel = (err.norm() / want.norm().clamp_min(1e-12)).item()
        col = (err.norm(dim=0) / want.norm(dim=0).clamp_min(1e-12)).tolist()
        prow = (err[:, HC_MULT:].norm(dim=-1) / want[:, HC_MULT:].norm(dim=-1).clamp_min(1e-12)).max().item()
        big = max(v for j, v in enumerate(col) if j not in FHC_SMALL_COLS)
        small = max(col[j] for j in FHC_SMALL_COLS)
        metrics.record(f"rel_l2_swap_ffn_hc_{tag}", rel)
        metrics.record(f"max_col_rel_l2_swap_ffn_hc_{tag}", big)
        metrics.record(f"max_small_col_rel_l2_swap_ffn_hc_{tag}", small)
        metrics.record(f"post_worst_row_rel_l2_swap_ffn_hc_{tag}", prow)
        print(
            f"ffn_hc {tag}: rel_l2={rel:.6f} (<= {max_rel}) col rel={[round(v, 5) for v in col]} (<= {max_col}, "
            f"columns {list(FHC_SMALL_COLS)} <= {max_small}) post worst row={prow:.5f} (<= {max_row})"
        )
        if rel > max_rel:
            failures.append(f"ffn_hc {tag}: rel L2 {rel:.5f} > {max_rel}")
        bad = [j for j, v in enumerate(col) if v > (max_small if j in FHC_SMALL_COLS else max_col)]
        if bad:
            failures.append(f"ffn_hc {tag}: column rel L2 above the limit in columns {bad}")
        if prow > max_row:
            failures.append(f"ffn_hc {tag}: post worst row rel L2 {prow:.5f} > {max_row}")

    mid_ok = seen["h_mid"].numel() == gl["h_mid"].numel() and torch.isfinite(seen["h_mid"].float()).all()
    if finite_shape("ffn_hc"):
        want_hc = gl["ffn_hc"].float()
        gt = seen["ffn_hc"].float().reshape(want_hc.shape)
        gate_check(
            "vs_golden", gt, want_hc, FHC_MAX_REL_L2, FHC_MAX_COL_REL, FHC_MAX_SMALL_COL_REL, FHC_MAX_POST_ROW_REL
        )
        if finite_shape("ffn_x"):
            rel_row("ffn_x", seen["ffn_x"], gl["ffn_x"], FX_MAX_REL_L2, FX_MAX_ROW_REL)
        if mid_ok:
            hm = seen["h_mid"].float().reshape(gl["h_mid"].shape)
            cpu_hc, cpu_pre = ref.component(layer, "ffn_hc"), ref.component(layer, "ffn_hc_pre")
            cpu_g = cpu_hc(rctx, hm).float()
            gate_check(
                "vs_cpu",
                gt,
                cpu_g,
                FHC_CPU_MAX_REL_L2,
                FHC_CPU_MAX_COL_REL,
                FHC_CPU_MAX_SMALL_COL_REL,
                FHC_CPU_MAX_POST_ROW_REL,
            )
            cpu_fx = cpu_pre(rctx, hm, cpu_g).float()
            rel_row(
                "ffn_x_vs_cpu",
                cpu_pre(rctx, hm, gt).float(),
                cpu_fx,
                FX_CPU_MAX_REL_L2,
                FX_CPU_MAX_ROW_REL,
                what="from the device gates vs from the CPU gates on the device h_mid",
            )

            # Post gates through out: the block's out vs ffn_residual with the CPU gates and the block's own mlp_out.
            res = ref.component(layer, "ffn_residual")
            if "mlp_out" in seen and seen["mlp_out"].numel() * HC_MULT == hm.numel():
                mo = seen["mlp_out"].float().reshape(hm.shape[0], -1)
                pw = res(rctx, hm, cpu_g, mo).float()
                n = pw.shape[0]
                d = seen["out"].float().reshape(n, HC_MULT, -1) - pw.view(n, HC_MULT, -1)
                srel = (d.norm(dim=(0, 2)) / pw.view(n, HC_MULT, -1).norm(dim=(0, 2)).clamp_min(1e-12)).tolist()
                prow = _worst_row_rel(seen["out"], pw, HC_MULT)
                metrics.record("post_stream_rel_l2_swap_out_vs_cpu_gates", max(srel))
                metrics.record("post_worst_row_rel_l2_swap_out_vs_cpu_gates", prow)
                print(
                    f"out vs ffn_residual(h_mid, CPU gates, block mlp_out): stream rel={[round(v, 5) for v in srel]} "
                    f"(<= {POST_MAX_STREAM_REL}) worst (row, stream) rel={prow:.5f} (<= {POST_MAX_ROW_REL})"
                )
                if max(srel) > POST_MAX_STREAM_REL or prow > POST_MAX_ROW_REL:
                    failures.append(f"out vs CPU post gates: stream rel {max(srel):.5f} / worst row {prow:.5f}")
            else:
                failures.append("mlp_out missing or misshapen, cannot check the post gates through out")

            # Block out vs the whole CPU tail from the same device h_mid (routing flips make rows noisy: rel only).
            fnm = ref.component(layer, "ffn_norm")(rctx, cpu_fx)
            rt = ref.component(layer, "router")(rctx, fnm)
            mo_c = ref.component(layer, "moe_combine")(
                rctx, ref.component(layer, "experts")(rctx, fnm, rt), ref.component(layer, "shared_expert")(rctx, fnm)
            )
            tail = res(rctx, hm, cpu_g, mo_c).float()
            trel, trow = _rel(seen["out"], tail), _worst_row_rel(seen["out"], tail, HC_MULT)
            metrics.record("rel_l2_swap_out_vs_cpu_tail", trel)
            metrics.record("worst_row_rel_l2_swap_out_vs_cpu_tail", trow)  # informational (near-tie expert flips)
            print(
                f"out vs CPU tail from the device h_mid: rel_l2={trel:.6f} (<= {TAIL_MAX_REL_L2}) "
                f"worst (row, stream) rel={trow:.5f} (not gated)"
            )
            if trel > TAIL_MAX_REL_L2:
                failures.append(f"out vs CPU tail: rel {trel:.5f} > {TAIL_MAX_REL_L2}")
        else:
            failures.append("ffn_hc: h_mid unusable, cannot check ffn_hc against the CPU step")

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
