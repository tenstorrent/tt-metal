# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 8: block type dense_full (layer 0) with ffn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    indexer
    attention
    attn_residual
    ffn_hc

Reviewed (S.dense_full.08.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). The swapped step is ffn_hc, the iHC gates [S, 8] fp32 (pre 0-3 | post 4-7)
from h_mid with the hc_mlp_layer weights. The pre gates make ffn_x = sum_j pre_j x stream_j (then ffn_norm, the
MLP); the post gates scale the MLP output into each stream (ffn_residual -> out). At h_mid the streams are distinct,
and the pre gates are small and unsaturated (col means 0.094, ~1e-5, 0.0009, 0.018), so their absolute error is
what reaches ffn_x. Measured on the CPU (golden s4096 chunk 1, 2048 rows; ffn_hc replaced by mutations of the fp32
step, every other step the fp32 CPU reference; "c" = vs the CPU step / CPU tail on the same h_mid):

    variant                     gates c rel/col/row       | ffn_x c rel/row  | out c rel/row   | out PCC
    fp32 reference              0 / 0 / 0                 | 0 / 0            | 0 / 0           | 0.999999
    bf16 output                 0.0017 / 0.0017 / 0.0031  | 0.0012 / 0.0035  | 0.0024 / 0.0071 | 0.999996 (ok)
    hc_eps dropped              0 / 0 / 0                 | 5e-5 / 1e-4      | 5e-5 / 1e-4     | 0.999999 (ok)
    sigmoid abs err 3e-4        0.0011 / 0.0012 / 0.0016  | 0.0087 / 0.019   | 0.0088 / 0.029  | 0.999960 (pass)
    sigmoid abs err 1e-3        0.0038 / 0.0041 / 0.0053  | 0.029 / 0.065    | 0.029 / 0.097   | 0.999565 (pass)
    pre x 1.01                  0.0009 / 0 / 0            | 0.0100 / 0.0100  | 0.0091 / 0.019  | 0.999958 (pass)
    post x 1.01 / x 1.1         0.0100 / 0.0100 / 0.0100  | 0 / 0            | 0.012 / 0.016   | 0.99993 / 0.9929 (pass)
    pre gate 2 = pre gate 1     0.0008 / 0 / 0            | 0.018 / 0.072    | 0.011 / 0.026   | 0.999943 (pass)
    fn rows 1 / 2 swapped (pre) 0.0009 / 0 / 0            | 0.013 / 0.051    | 0.0077 / 0.020  | 0.999969 (pass)
    fn rows 4 / 5 swapped       0.092 / 0.156 / 0.197     | 0 / 0            | 0.138 / 0.214   | 0.990601 (pass)
    base 4 / 6 (4 / 5) swapped  0.015 / 0.026 / 0.020     | 0 / 0            | 0.016 / 0.023   | 0.999868 (pass)
    fn streams 0 / 2 (0 / 1)    0.013 / 0.020 / 0.029     | 0.019 / 0.073    | 0.018 / 0.061   | 0.999834 (pass)
    rows 1023 / 1024 swapped    0.0053 / - / 0.19         | 0.0039 / 0.21    | 0.0027 / 0.14   | 0.999995 (pass)
    last row = previous row     0.0074 / 0.0095 / 0.25    | 0.0085 / 0.29    | 0.0086 / 0.26   | 0.999962 (pass)
    last row zeroed             0.029 / 0.033 / 1.0       | 0.029 / 1.0      | 0.044 / 1.36    | 0.999037 (pass)
    post = 1 x sigmoid, pre gate 3 = pre gate 2, fn rows 0 / 1, fn streams 2 / 3, one chip's partial sumsq,
    chip-major columns, attn_hc weights, eps 1e-2, pre | post halves swapped, zero stub: out PCC <= 0.903 (fail)
    rms_norm_eps 1e-6 / 2e-5: on h_mid x 0.1 vs the CPU step, gates rel 0.19 / 0.054, ffn_x rel 0.30 / 0.20

17 of 27 mutations pass the 0.98 out gate (15 of them real bugs). So the test also asserts (informational metrics):
  - everything swap 07 asserts: attn_hc, attn_x, attn_norm, q_resid (vs golden, vs the CPU step, eps checks), topk
    (overlap vs golden and the CPU indexer, structure, chunk 0 exact), attn_out (vs golden, vs the CPU step,
    chunk 0, probe topk, scaled input), h_mid (vs golden, per-stream ratio, vs the CPU attn_residual, addend,
    distinct-stream probe), block out finite and rel L2 <= 0.01;
  - ffn_hc (the swapped step) vs golden at the component's limits: rel L2 <= 0.01, per post column rel L2 <= 0.006,
    worst row over the post columns <= 0.015 (device 0.0023 / 0.0028 / 0.0067: the upstream device h_mid error);
    ffn_x (the CPU ffn_hc_pre from the device gates) vs golden: rel L2 <= 0.01, worst row <= 0.05 (as h_mid);
  - ffn_hc vs the CPU ffn_hc on the same device h_mid: rel L2 <= 0.005, post column <= 0.004, post worst row <= 0.01
    (device 0.00062 / 0.00076 / 0.0015); the pre gates through ffn_x, device gates vs CPU gates on that h_mid:
    rel L2 <= 0.005, worst row <= 0.02 (device 0.0022 / 0.0030, the component's limits);
  - block out vs the CPU tail (ffn_hc .. ffn_residual) from the same device h_mid: rel L2 <= 0.005, worst row
    <= 0.02 (device 0.0023 / 0.0047): what the device gates do to out, with the upstream error removed;
  - the ffn_hc module once more on h_mid x 0.1 (bf16) vs the CPU step, the same gate and ffn_x limits, where the
    gate mix's rms_norm_eps 1e-5 matters (device 0.00024 / ffn_x 0.0013).
Every "(passes)" variant above fails one of these; bf16 output and a dropped hc_eps pass them and change out by
<= 0.0024. The trail line pcc_swap_topk is positional match (template default) and is not meaningful for the
device's unsorted indices.
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
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "indexer", "attention", "attn_residual", "ffn_hc"]
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
ATT_MAX_REL_L2 = 0.01  # attn_out, whole tensor (vs golden, vs the CPU step, chunk 0, probe, scaled)
ATT_RATIO = (0.99, 1.01)  # attn_out per-row ||got|| / ||want||
ATT_MAX_ROW_REL_L2 = 0.02  # attn_out, worst token row
PROBE_KEYS = 64  # probe topk: this many random causal positions per row, unsorted, the rest -1
PROBE_SEED = 0
ATT_SYN_SCALE = 1e-3  # eps check: the device attn_norm x ATT_SYN_SCALE (bf16), where kv_a_layernorm's eps matters
MID_MAX_REL_L2 = 0.01  # h_mid, whole tensor
MID_MAX_ROW_REL_L2 = 0.05  # h_mid, worst token row
HC = 4  # iHC streams
MID_STREAM_RATIO = (0.98, 1.02)  # h_mid per token and stream ||got|| / ||want|| vs golden
RES_MAX_REL_L2 = 5e-4  # h_mid vs the CPU attn_residual on the same inputs (device fp32: bit-identical), and the probe
RES_MAX_ROW_REL_L2 = 1e-3  # the same, worst token row
ADD_COEF = (0.97, 1.03)  # per stream <h_mid_j - in_j, post_j * attn_out> / ||post_j * attn_out||^2
MAX_ADD_REL = 0.03  # per stream ||(h_mid_j - in_j) - post_j * attn_out|| / ||post_j * attn_out||
MAX_ADD_ROW_REL = 0.1  # the same per token, worst row
FHC_MAX_REL_L2 = 0.01  # ffn_hc gates [S, 8] vs golden, whole tensor
FHC_POST_COL_REL = 0.006  # ffn_hc vs golden, per post column (4-7) rel L2
FHC_POST_ROW_REL = 0.015  # ffn_hc vs golden, worst row rel L2 over the post columns
FHC_CPU_MAX_REL_L2 = 0.005  # ffn_hc vs the CPU ffn_hc on the same device h_mid (and on the scaled probe)
FHC_CPU_POST_COL_REL = 0.004  # the same, per post column
FHC_CPU_POST_ROW_REL = 0.01  # the same, worst row over the post columns
FX_MAX_REL_L2 = 0.01  # ffn_x (CPU ffn_hc_pre from the device gates) vs golden
FX_MAX_ROW_REL_L2 = 0.05  # the same, worst row
FX_CPU_MAX_REL_L2 = 0.005  # ffn_x from the device gates vs from the CPU gates on the same h_mid (and the probe)
FX_CPU_MAX_ROW_REL_L2 = 0.02  # the same, worst row
FHC_SYN_SCALE = 0.1  # eps check: the device h_mid x FHC_SYN_SCALE (bf16) through the module vs the CPU step
OUT_CPU_MAX_REL_L2 = 0.005  # block out vs the CPU tail (ffn_hc .. ffn_residual) from the same device h_mid
OUT_CPU_MAX_ROW_REL_L2 = 0.02  # the same, worst row
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


def _probe_topk(start, rows, k, n, seed):
    """[rows, k] int64: per row min(n, pos + 1) distinct random positions in [0, pos], unsorted, then -1 pads."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.full((rows, k), -1, dtype=torch.int64)
    for i in range(rows):
        p = start + i
        m = min(n, p + 1)
        out[i, :m] = torch.randperm(p + 1, generator=gen)[:m]
    return out


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

    # attn_out (the swapped step): vs golden, vs the CPU step on the same device inputs, then the module on chunk 0,
    # on a probe topk and on a scaled attn_norm, each vs the CPU step on the same inputs.
    def att_check(tag, got, want):
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"attn_out {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        rel, rmin, rmax, row = _errors(got, want)
        metrics.record(f"rel_l2_swap_attn_out_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_attn_out_{tag}", row)
        print(
            f"attn_out {tag}: rel_l2={rel:.6f} (<= {ATT_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(ATT_RATIO)}) worst_row_rel_l2={row:.5f} (<= {ATT_MAX_ROW_REL_L2})"
        )
        if rel > ATT_MAX_REL_L2:
            failures.append(f"attn_out {tag}: rel L2 {rel:.5f} > {ATT_MAX_REL_L2}")
        if not (ATT_RATIO[0] <= rmin and rmax <= ATT_RATIO[1]):
            failures.append(f"attn_out {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ATT_RATIO)}")
        if row > ATT_MAX_ROW_REL_L2:
            failures.append(f"attn_out {tag}: worst row rel L2 {row:.5f} > {ATT_MAX_ROW_REL_L2}")

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

    # h_mid (the swapped step): vs golden (+ per-stream norm ratio), vs the CPU step on the same device inputs (+ the
    # addend per stream), and the module on distinct input streams vs the CPU step.
    if finite_shape("h_mid"):
        rel_row("h_mid", MID_MAX_REL_L2, MID_MAX_ROW_REL_L2)
        want = gl["h_mid"].float()
        n = want.shape[0]
        sr = seen["h_mid"].float().reshape(n, HC, -1).norm(dim=-1) / want.view(n, HC, -1).norm(dim=-1).clamp_min(1e-12)
        smin, smax = sr.min().item(), sr.max().item()
        metrics.record("stream_norm_ratio_min_swap_h_mid", smin)
        metrics.record("stream_norm_ratio_max_swap_h_mid", smax)
        print(f"h_mid per-token stream norm ratio=[{smin:.5f}, {smax:.5f}] (in {list(MID_STREAM_RATIO)})")
        if not (MID_STREAM_RATIO[0] <= smin and smax <= MID_STREAM_RATIO[1]):
            failures.append(f"h_mid: stream norm ratio [{smin:.5f}, {smax:.5f}] outside {list(MID_STREAM_RATIO)}")

        a_ok = "attn_out" in seen and seen["attn_out"].numel() == gl["attn_out"].numel()
        g_ok = "attn_hc" in seen and seen["attn_hc"].numel() == gl["attn_hc"].numel()
        if a_ok and g_ok and torch.isfinite(seen["attn_out"].float()).all():
            cpu = ref.component(layer, "attn_residual")
            xs_in = gl["in"].float()
            gt = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            y = seen["attn_out"].float().reshape(gl["attn_out"].shape)
            rel_row("h_mid", RES_MAX_REL_L2, RES_MAX_ROW_REL_L2, what="vs cpu", want=cpu(rctx, xs_in, gt, y).float())

            # The addend on each stream: delta_j = h_mid_j - in_j vs post_j * attn_out.
            post = gt[:, HC : 2 * HC]
            delta = seen["h_mid"].float().reshape(n, HC, -1) - xs_in.view(n, HC, -1)
            tgt = post.unsqueeze(-1) * y.view(n, 1, -1)
            err = delta - tgt
            coef = ((delta * tgt).sum(dim=(0, 2)) / (tgt * tgt).sum(dim=(0, 2)).clamp_min(1e-30)).tolist()
            arel = (err.norm(dim=(0, 2)) / tgt.norm(dim=(0, 2)).clamp_min(1e-30)).tolist()
            worst = (err.norm(dim=-1) / tgt.norm(dim=-1).clamp_min(1e-30)).max().item()
            metrics.record("add_coef_min_swap_h_mid", min(coef))
            metrics.record("add_coef_max_swap_h_mid", max(coef))
            metrics.record("add_rel_l2_swap_h_mid", max(arel))
            metrics.record("add_worst_row_rel_swap_h_mid", worst)
            print(
                f"h_mid addend per stream: coef={[round(v, 5) for v in coef]} (in {list(ADD_COEF)}) "
                f"rel={[round(v, 5) for v in arel]} (<= {MAX_ADD_REL}) worst row={worst:.5f} (<= {MAX_ADD_ROW_REL})"
            )
            bad = [j for j, v in enumerate(coef) if not ADD_COEF[0] <= v <= ADD_COEF[1]]
            if bad:
                failures.append(f"h_mid addend: coefficient outside {list(ADD_COEF)} on streams {bad}: {coef}")
            bad = [j for j, v in enumerate(arel) if v > MAX_ADD_REL]
            if bad:
                failures.append(f"h_mid addend: rel L2 above {MAX_ADD_REL} on streams {bad}: {arel}")
            if worst > MAX_ADD_ROW_REL:
                failures.append(f"h_mid addend: worst row rel L2 {worst:.5f} > {MAX_ADD_ROW_REL}")

            # Distinct input streams (layer 0's are identical): the golden h_mid as streams.
            ps = gl["h_mid"].float()
            pwant = cpu(rctx, ps, gt, y).float()
            pout = muts["attn_residual"](rctx, dctx, ps, gt, y)
            if pout.numel() != pwant.numel() or not torch.isfinite(pout.float()).all():
                failures.append(f"attn_residual probe: shape {tuple(pout.shape)} or non-finite")
            else:
                prel, pmin, pmax, prow = _errors(pout, pwant)
                metrics.record("probe_rel_l2_swap_h_mid", prel)
                metrics.record("probe_worst_row_rel_l2_swap_h_mid", prow)
                print(
                    f"attn_residual on distinct streams vs CPU: rel_l2={prel:.7f} (<= {RES_MAX_REL_L2}) row norm "
                    f"ratio=[{pmin:.6f}, {pmax:.6f}] worst_row_rel_l2={prow:.7f} (<= {RES_MAX_ROW_REL_L2})"
                )
                if prel > RES_MAX_REL_L2 or prow > RES_MAX_ROW_REL_L2:
                    failures.append(f"attn_residual probe: rel {prel:.6f} / worst row {prow:.6f} (stream order?)")
        else:
            failures.append("h_mid: attn_hc / attn_out unusable, cannot check attn_residual against the CPU step")

    # ffn_hc (the swapped step): the iHC gates [S, 8] from h_mid. The pre gates are small and unsaturated at this
    # layer, so they are judged through ffn_x = sum_j pre_j x stream_j; the post columns per column and per row.
    def gate_check(tag, got, want, max_rel, post_col_max, post_row_max):
        got = got.float().reshape(want.shape)
        err = got - want
        rel = (err.norm() / want.norm().clamp_min(1e-12)).item()
        pcol = max((err[:, HC:].norm(dim=0) / want[:, HC:].norm(dim=0).clamp_min(1e-12)).tolist())
        prow = (err[:, HC:].norm(dim=-1) / want[:, HC:].norm(dim=-1).clamp_min(1e-12)).max().item()
        metrics.record(f"rel_l2_swap_ffn_hc_{tag}", rel)
        metrics.record(f"post_col_rel_l2_swap_ffn_hc_{tag}", pcol)
        metrics.record(f"post_worst_row_rel_l2_swap_ffn_hc_{tag}", prow)
        print(
            f"ffn_hc {tag}: rel_l2={rel:.6f} (<= {max_rel}) post max col rel={pcol:.6f} (<= {post_col_max}) "
            f"post worst row={prow:.5f} (<= {post_row_max}) max abs per column="
            f"{[f'{v:.2e}' for v in err.abs().amax(dim=0).tolist()]}"
        )
        if rel > max_rel:
            failures.append(f"ffn_hc {tag}: rel L2 {rel:.5f} > {max_rel}")
        if pcol > post_col_max:
            failures.append(f"ffn_hc {tag}: post column rel L2 {pcol:.5f} > {post_col_max}")
        if prow > post_row_max:
            failures.append(f"ffn_hc {tag}: post worst row rel L2 {prow:.5f} > {post_row_max}")

    def fx_check(tag, got, want):
        rel, rmin, rmax, row = _errors(got, want)
        metrics.record(f"rel_l2_swap_ffn_x_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_ffn_x_{tag}", row)
        print(
            f"ffn_x {tag}: rel_l2={rel:.6f} (<= {FX_CPU_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"worst_row_rel_l2={row:.5f} (<= {FX_CPU_MAX_ROW_REL_L2})"
        )
        if rel > FX_CPU_MAX_REL_L2 or row > FX_CPU_MAX_ROW_REL_L2:
            failures.append(f"ffn_x {tag}: rel {rel:.5f} / worst row {row:.5f} (pre gate error?)")

    mid_ok = "h_mid" in seen and seen["h_mid"].numel() == gl["h_mid"].numel() and torch.isfinite(seen["h_mid"]).all()
    if finite_shape("ffn_hc"):
        want_hc = gl["ffn_hc"].float()
        gate_check("vs_golden", seen["ffn_hc"], want_hc, FHC_MAX_REL_L2, FHC_POST_COL_REL, FHC_POST_ROW_REL)
        if finite_shape("ffn_x"):
            rel_row("ffn_x", FX_MAX_REL_L2, FX_MAX_ROW_REL_L2)
        if mid_ok:
            hm = seen["h_mid"].float().reshape(gl["h_mid"].shape)
            gt = seen["ffn_hc"].float().reshape(gl["ffn_hc"].shape)
            cpu_hc, cpu_pre = ref.component(layer, "ffn_hc"), ref.component(layer, "ffn_hc_pre")
            cpu_g = cpu_hc(rctx, hm).float()
            gate_check("vs_cpu", gt, cpu_g, FHC_CPU_MAX_REL_L2, FHC_CPU_POST_COL_REL, FHC_CPU_POST_ROW_REL)
            cpu_fx = cpu_pre(rctx, hm, cpu_g).float()
            fx_check("vs_cpu", cpu_pre(rctx, hm, gt).float(), cpu_fx)

            # Block out vs the CPU tail from the same device h_mid: how the device gates reach out (post gates).
            fnorm = ref.component(layer, "ffn_norm")(rctx, cpu_fx)
            mo = ref.component(layer, "mlp")(rctx, fnorm)
            cpu_out = ref.component(layer, "ffn_residual")(rctx, hm, cpu_g, mo).float()
            rel, _, _, row = _errors(seen["out"], cpu_out)
            metrics.record("rel_l2_swap_out_vs_cpu_tail", rel)
            metrics.record("worst_row_rel_l2_swap_out_vs_cpu_tail", row)
            print(
                f"out vs CPU tail from the device h_mid: rel_l2={rel:.6f} (<= {OUT_CPU_MAX_REL_L2}) "
                f"worst_row_rel_l2={row:.5f} (<= {OUT_CPU_MAX_ROW_REL_L2})"
            )
            if rel > OUT_CPU_MAX_REL_L2 or row > OUT_CPU_MAX_ROW_REL_L2:
                failures.append(f"out vs CPU tail: rel {rel:.5f} / worst row {row:.5f}")

            # eps: h_mid scaled so rms_norm_eps (1e-5) inside the gate mix is visible.
            hs = (hm * FHC_SYN_SCALE).bfloat16().float()
            syn_want = cpu_hc(rctx, hs).float()
            syn_out = muts["ffn_hc"](rctx, dctx, hs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"ffn_hc scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                syn_out = syn_out.float().reshape(syn_want.shape)
                gate_check("scaled", syn_out, syn_want, FHC_CPU_MAX_REL_L2, FHC_CPU_POST_COL_REL, FHC_CPU_POST_ROW_REL)
                fx_check("scaled", cpu_pre(rctx, hs, syn_out).float(), cpu_pre(rctx, hs, syn_want).float())
        else:
            failures.append("ffn_hc: h_mid unusable, cannot check ffn_hc against the CPU step")

    # Block out.
    out_rel = _errors(seen["out"], gl["out"])[0]
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not (torch.isfinite(seen["out"].float()).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
