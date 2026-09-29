# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 6: block type dense_full (layer 0) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    indexer
    attention

Reviewed (S.dense_full.06.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). The swapped step is attn_out [S, 6144] = gated sparse MLA (64 heads, absorbed
576 / 512 latent, scale 1/16, per-head sinks, sigmoid output gate, o_proj) over the indexer's topk, stateful (it
writes this chunk's kv_latent rows and reads the golden prefix [0, 2048)). attn_out enters the residual through
post * attn_out, so the four streams damp most attention bugs. Measured on the CPU (golden s4096 chunk 1, 2048 rows,
2048-key prefix; attention replaced by mutations of the fp32 step on the block's own inputs, every other step the
fp32 CPU reference; rel = rel L2 vs golden / worst token row):

    variant                          attn_out rel / worst row | h_mid rel / worst | out PCC   rel
    fp32 reference                   0.0017 / 0.002           | 0.0017 / 0.0017   | 0.999999  0.0017
    bf16 W / act / P (device est.)   0.0032 / 0.004           | 0.0031 / 0.0050   | 0.999994  0.0035
    kv_a_layernorm eps 1e-5          0.0017 / 0.002           | 0.0017 / 0.0017   | 0.999999  0.0017 (passes)
    pads read as key 0               0.0017 / 0.002           | 0.0017 / 0.0017   | 0.999999  0.0017 (passes)
    RoPE positions + 1               0.0042 / 0.134           | 0.0038 / 0.131    | 0.999985  0.0054 (passes)
    dense causal (topk ignored)      0.0064 / 0.017           | 0.0060 / 0.017    | 0.999967  0.0081 (passes)
    prefix row halves swapped        0.0076 / 0.025           | 0.0068 / 0.032    | 0.999957  0.0093 (passes)
    x 1.01 / x 1.02 / x 1.05         0.010 / 0.020 / 0.050    | 0.0088 / 0.0175   | 0.99996 / 0.99984 / 0.99903 (pass)
    own key dropped                  0.0115 / 0.063           | 0.0074 / 0.080    | 0.999935  0.0114 (passes)
    sliding window of the last 2048  0.0116 / 0.032           | 0.0095 / 0.041    | 0.999915  0.0131 (passes)
    topk - 1                         0.0146 / 0.063           | 0.0113 / 0.080    | 0.999864  0.0166 (passes)
    last row zeroed                  0.0245 / 1.0             | 0.0311 / 0.89     | 0.999297  0.0377 (passes)
    scale 192^-0.5 / 576^-0.5        0.025 / 0.077            | 0.023 / 0.074     | 0.99941 / 0.99416 (pass)
    dense non-causal                 0.0309 / 0.066           | 0.0287 / 0.084    | 0.999100  0.0424 (passes)
    sink raw to sparse_sdpa (/ 16)   0.0352 / 0.053           | 0.0329 / 0.054    | 0.998850  0.0481 (passes)
    RoPE from position 0             0.0555 / 0.199           | 0.0472 / 0.234    | 0.997374  0.0723 (passes)
    rotate-half RoPE / no q RoPE     0.103 / 0.127            | 0.088 / 0.103     | 0.99168 / 0.98712 (pass)
    no sink / sink mass kept         0.1263 / 0.204           | 0.1299 / 0.222    | 0.985374  0.1753 (passes)
    sink sign                        0.1450 / 0.310           | 0.1061 / 0.402    | 0.985085  0.1720 (passes)
    last tile row zeroed             0.1241 / 1.0             | 0.1148 / 1.32     | 0.988651  0.1522 (passes)
    no k RoPE                        0.1832 / 0.346           | 0.1434 / 0.395    | 0.978590  0.2060
    zero prefix, no gate, gate 0.5, gate / o head halves, sink x 16, no kv norm (weight), SP row halves swapped
                                     >= 0.71                  | >= 0.63           | < 0.80

"(passes)" = passes the 0.98 out gate; 20 of 29 bugs do. So the test also asserts (informational metrics):
  - everything swap 05 asserts: attn_hc gates, attn_x, attn_norm and q_resid (vs golden, vs the CPU step, eps
    checks), topk (overlap vs golden and vs the CPU indexer, structure, chunk 0 exact), h_mid rel L2 <= 0.01 /
    worst row <= 0.05, block out finite and rel L2 <= 0.01;
  - attn_out (the swapped step) at the component limits (test_c_dense_full_attention.py): rel L2 <= 0.01, every
    row's norm ratio in [0.99, 1.01], worst row rel L2 <= 0.02, (a) vs the golden and (b) vs the CPU attention on
    the same device inputs (device attn_norm, q_resid, topk: the step's own error);
  - the attention module once more, vs the CPU step on the same inputs, same limits: (c) golden chunk 0 (start 0,
    empty prefix, -1 pads in every row: pads read as key 0 score worst row 0.83); (d) a probe topk (per row 64
    random causal positions, unsorted, the rest -1) on the device attn_norm / q_resid: dense causal scores rel 0.48,
    prefix row halves swapped 0.049 / worst row 0.19, topk - 1 0.14; (e) the device attn_norm x 1e-3 (bf16), where
    kv_a_layernorm's eps 1e-6 matters (eps 1e-5 scores rel 0.43).
Every variant above except x 1.01 fails one of these (RoPE + 1 and the rest on worst row; x 1.01 sits at the
tolerance). The chunk's kv_latent rows are not read back: the harness's module_under_test exposes only the forward,
and the ladder's state gate compares the cache. The trail line pcc_swap_topk is positional match (template
default) and is not meaningful for the device's unsorted indices.
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
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "indexer", "attention"]
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
