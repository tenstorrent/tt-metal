# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 6: block type moe_shared (layer 2) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 2 (moe_shared) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    topk_shared
    attention

Reviewed (S.moe_shared.06.test.1). Built from test_swap_moe_shared_05_topk_shared.py (layer-2 setup: ctx.extra
["shared_topk"] = the golden's L{src}.topk, src = cfg.topk_source(2) = 1, in both contexts; every attn_hc / attn_hc_pre
/ attn_norm / q_a / topk check and limit of swap 05) plus the attention checks of test_c_moe_shared_attention.py at
its layer-2 limits. The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC streams, spec block
threshold 0.98). The swapped step is attn_out [S, 6144] = gated sparse MLA (64 heads, absorbed 576 / 512 latent,
scale 1/16, per-head sinks, sigmoid output gate, o_proj) over the shared top-k. It is stateful: it writes this chunk's
kv_latent rows and reads the golden layer-2 prefix [0, 2048). Measured on the CPU (golden s4096 chunk 1, 2048 rows;
attention replaced by mutations of the fp32 step on the block's own inputs, every other step the fp32 CPU reference;
study /tmp/hy4_ssh6/study.py on /tmp/hy4_c_attn2/mut.py, outside the repo, ~13 s per variant). "ao" = attn_out rel
L2 / worst row vs golden, "hm" = h_mid rel / max per-stream rel / worst (row, stream), "rt" = router top-8 overlap,
"out" = PCC / rel L2:

    variant                          ao             | hm                       | rt     | out
    fp32 reference                   0.0019 / 0.002 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030
    bf16 W / act / P (device est.)   0.0045 / 0.005 | 0.0012 / 0.0049 / 0.0060 | 0.9958 | 0.999993 0.0038
    device (this gate)               0.0063 / 0.009 | 0.0015 / 0.0067 / 0.0084 | 0.9935 | 0.999977 0.0068
    kv_a_layernorm eps 1e-5          0.0019 / 0.002 | 0.0009 / 0.0029 / 0.0033 | 0.9973 | 0.999995 0.0030 (passes)
    RoPE positions + 1               0.0026 / 0.064 | 0.0009 / 0.0030 / 0.0496 | 0.9971 | 0.999995 0.0031 (passes)
    sink raw to sparse_sdpa (/ 16)   0.0075 / 0.037 | 0.0012 / 0.0051 / 0.0445 | 0.9948 | 0.999990 0.0045 (passes)
    prefix row halves swapped        0.0080 / 0.025 | 0.0015 / 0.0070 / 0.0223 | 0.9923 | 0.999981 0.0062 (passes)
    own key dropped                  0.0084 / 0.074 | 0.0013 / 0.0055 / 0.0591 | 0.9926 | 0.999986 0.0054 (passes)
    dense causal (topk ignored)      0.0094 / 0.033 | 0.0016 / 0.0077 / 0.0278 | 0.9939 | 0.999985 0.0055 (passes)
    x 1.01                           0.0102 / 0.010 | 0.0022 / 0.0111 / 0.0141 | 0.9888 | 0.999909 0.0145 (passes)
    topk - 1                         0.0111 / 0.074 | 0.0019 / 0.0091 / 0.0591 | 0.9910 | 0.999969 0.0079 (passes)
    sink sign                        0.0136 / 0.067 | 0.0018 / 0.0086 / 0.0812 | 0.9915 | 0.999983 0.0059 (passes)
    window of the last 2048          0.0166 / 0.047 | 0.0027 / 0.0135 / 0.0468 | 0.9842 | 0.999930 0.0119 (passes)
    x 1.02 / x 1.05                  ao 0.020 / 0.050 (every row alike), hm stream 0.022 / 0.054, rt 0.9789 / 0.9503,
                                     out 0.99983 / 0.99935 (pass)
    last row zeroed                  0.0244 / 1.0   | 0.0070 / 0.0364 / 1.17   | 0.9969 | 0.999767 0.0217 (passes)
    no sink / sink mass kept         0.0273 / 0.140 | 0.0030 / 0.0151 / 0.1647 | 0.9833 | 0.999974 0.0073 (passes)
    dense non-causal                 0.0281 / 0.126 | 0.0042 / 0.0215 / 0.1067 | 0.9769 | 0.999900 0.0142 (passes)
    scale 192^-0.5 / 576^-0.5        ao 0.029 / 0.107, 0.044 / 0.193; rt 0.9781 / 0.9667; out 0.99986 / 0.99989 (pass)
    RoPE from position 0             0.0439 / 0.216 | 0.0056 / 0.0287 / 0.1642 | 0.9655 | 0.999896 0.0146 (passes)
    rotate-half RoPE / no k / no q   ao 0.060 / 0.064 / 0.081, worst row >= 0.21; out >= 0.99975 (pass)
    last tile row zeroed             0.1236 / 1.0   | 0.0271 / 0.1420 / 1.21   | 0.9893 | 0.988447 0.1563 (passes)
    sink x 16, zero prefix, no gate, gate 0.5, gate or o head halves, no kv norm (weight), SP row halves swapped,
    attn_out zeroed (0.852)          ao >= 0.83                                            | out < 0.98

"(passes)" = passes the 0.98 out gate: 23 of 32 bugs do (every bug with ao < 0.13), and eps 1e-5 is invisible on
the golden. The residual streams dominate block out at layer 2. So the test also asserts (informational metrics):
  - everything swap 05 asserts, at its limits: attn_hc gates, attn_x (vs golden, vs the CPU hc_pre, rotated pre
    gates), attn_norm and q_resid (vs golden, vs the CPU step, eps runs), topk (exact per-row sets vs golden L2.topk
    and vs the shared input, pad layout, chunk 0 exact), h_mid rel <= 0.005 / worst (row, stream) <= 0.02, router
    overlap >= 0.98, block out finite and rel L2 <= 0.01. One change: the h_mid per-stream limit is 0.01 (swap 05:
    0.005). The device attention's own error (0.0058 vs the CPU step) now reaches streams 0-2, where attn_out is a
    large part of h_mid: device [0.0051, 0.0067, 0.0050, 0.0006], bf16 estimate 0.0049. Swap 05's attn_out check
    (0.01 / 0.05) is replaced by the one below;
  - attn_out (the swapped step) at the layer-2 component limits (test_c_moe_shared_attention.py): rel L2 <= 0.012,
    every row's norm ratio in [0.99, 1.01], worst row <= 0.025, float64 coefficient <got, want> / <want, want> within
    0.004 of 1, (a) vs the golden and (b) vs the CPU attention on the same device inputs (device attn_norm, q_resid,
    topk: the step's own error);
  - the attention module once more, vs the CPU step on the same inputs, same limits: (c) golden chunk 0 (start 0,
    empty prefix, golden chunk-0 L2.topk with -1 pads in every row: pads read as key 0 score worst row 0.90); (d) a
    probe topk (per row 64 random causal positions, unsorted, the rest -1) on the device attn_norm / q_resid: dense
    causal scores rel 0.41, topk - 1 0.117, prefix row halves swapped 0.064, sink / 16 0.086; (e) the device
    attn_norm x 1e-3 (bf16), where kv_a_layernorm's eps 1e-6 matters, at component check 4's limits (rel 0.03, ratio
    [0.99, 1.01], worst row 0.05, coef 0.004; eps 1e-5 scores rel 0.47).
Every variant above fails at least one of (a)-(e) (the component study's numbers): RoPE + 1, sink / 16, own key
dropped, topk - 1, dense causal and the zeroed rows on the worst row, x 1.01 on the coefficient, prefix row halves
swapped on the probe, eps 1e-5 on (e). The chunk's kv_latent rows are not read back, because the harness's
module_under_test exposes only the forward; the ladder's state gate compares the cache. The trail line pcc_swap_topk
is positional match (template default), diagnostic only. The out worst (row, stream) rel L2 is recorded, not asserted.
Device run (this gate): pcc_swap_out 0.999977; attn_out vs golden 0.00628 / [0.99818, 1.00155] / 0.0088 / coef
0.99936, vs CPU 0.00582 / 0.0083, chunk 0 0.00558 / 0.0077, probe 0.00539 / 0.0077, scaled 0.0119 / 0.0212; h_mid
0.00146 / stream max 0.0067 / 0.0084; router 0.99353; out rel 0.00683.
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
BLOCK_TYPE = "moe_shared"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a", "topk_shared", "attention"]
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
H_MID_MAX_STREAM_REL = 0.01  # h_mid, rel L2 of each stream (stream 3 dominates the whole tensor at layer 2;
# swap 05: 0.005. The device attention's own error now reaches streams 0-2: device 0.0067, bf16 estimate 0.0049)
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
ATT_LIMITS = (0.012, (0.99, 1.01), 0.025, 0.004)  # attn_out (the swapped step): max rel L2, per-row norm ratio,
# worst row rel L2, max |<got, want> / <want, want> - 1| (test_c_moe_shared_attention.py checks 1-3): vs golden, vs CPU,
# chunk 0, probe
ATT_SCALED_LIMITS = (0.03, (0.99, 1.01), 0.05, 0.004)  # the attn_norm x 1e-3 run (component check 4)
PROBE_KEYS = 64  # probe topk: this many random causal positions per row, unsorted, the rest -1
PROBE_SEED = 0
ATT_SYN_SCALE = 1e-3  # eps check: the device attn_norm x ATT_SYN_SCALE (bf16), where kv_a_layernorm's eps matters
ROUTER_MIN_OVERLAP = 0.98  # router top-8 selection overlap vs golden
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor
SENTINEL = 0xFFFFFFFF  # topk_large_indices' / sparse_sdpa's pad


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


def _norm_topk(out, want):
    """int64 [S, k] with every pad (negative, or the uint32 sentinel) as -1; None if not integer / wrong size."""
    if out.is_floating_point() or out.numel() != want.numel():
        return None
    t = out.reshape(want.shape).to(torch.int64)
    return torch.where((t < 0) | (t == SENTINEL), torch.full_like(t, -1), t)


def _topk_check(got, want, start, tag):
    """Exact per-row set equality with want plus sparse_sdpa's pad layout; returns failure messages."""
    fails = []
    valid = got >= 0
    pos = torch.arange(start, start + got.shape[0])[:, None]
    noncausal = (valid & (got > pos)).sum().item()
    if noncausal:
        fails.append(f"{noncausal} selected positions are after their query row (non-causal)")
    srt, ws = got.sort(dim=-1).values, want.sort(dim=-1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item()
    if dup:
        fails.append(f"{dup} repeated positions within rows")
    bad = (srt != ws).any(-1).nonzero().flatten()
    if bad.numel():
        r = bad[0].item()
        fails.append(
            f"{bad.numel()} rows differ from the wanted set (first: row {r} at position {start + r}: "
            f"{valid[r].sum().item()} valid, want {(want[r] >= 0).sum().item()})"
        )
    tail_bad = (valid[:, 1:] & ~valid[:, :-1]).any(-1).sum().item()
    if tail_bad:
        fails.append(f"{tail_bad} rows have a valid position after a pad (sparse_sdpa needs a contiguous pad tail)")
    empty = (valid.sum(-1) == 0).sum().item()
    if empty:
        fails.append(f"{empty} rows have no valid key (sparse_sdpa needs >= 1)")
    rows = torch.tensor(
        [torch.isin(b[b >= 0], a[a >= 0]).float().mean().item() if (b >= 0).any() else 1.0 for a, b in zip(got, want)]
    )
    metrics.record(f"{tag}topk_overlap_swap", rows.mean().item())
    metrics.record(f"{tag}topk_worst_row_overlap_swap", rows.min().item())
    print(
        f"{tag}topk: mean set overlap={rows.mean().item():.6f} worst row={rows.min().item():.5f} (== 1) "
        f"differing rows={bad.numel()} pads={(~valid).sum().item()} (want {(want < 0).sum().item()}) "
        f"positional match={(got == want).float().mean().item():.4f} (not gated)"
    )
    return [f"{tag}topk: {m}" for m in fails]


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

    # topk (the swapped step): exact per-row sets vs golden and vs the shared input, pad layout, then chunk 0.
    want_tk = gl["topk"].long()
    got_tk = _norm_topk(seen["topk"], want_tk)
    if got_tk is None:
        failures.append(f"topk: {seen['topk'].dtype} {tuple(seen['topk'].shape)}, want integer {tuple(want_tk.shape)}")
    else:
        start = c * g.chunk
        failures += _topk_check(got_tk, want_tk, start, "")
        failures += _topk_check(got_tk, shared_topk.long(), start, "vs_shared_input_")
        g0 = g.layer(0, layer)
        shared0 = g.layer(0, src)["topk"]
        rctx0, dctx0 = reference_ctx(ref, layer, g, 0), device_ctx(layer, g, 0)
        rctx0.extra["shared_topk"] = shared0.clone()
        dctx0.extra["shared_topk"] = shared0.clone()
        want0 = g0["topk"].long()
        got0 = _norm_topk(muts["topk_shared"](rctx0, dctx0, g0["attn_norm"].float()), want0)
        if got0 is None:
            failures.append("chunk0_topk: not integer or wrong size")
        else:
            failures += _topk_check(got0, want0, 0, "chunk0_")

    # attn_out (the swapped step): vs golden, vs the CPU step on the same device inputs, then the module on chunk 0,
    # on a probe topk and on a scaled attn_norm, each vs the CPU step on the same inputs.
    def att_check(tag, got, want, limits=ATT_LIMITS):
        max_rel, ratio_lim, max_row, max_coef = limits
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"attn_out {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        rel, row = _rel(got, want), _worst_row_rel(got, want)
        rmin, rmax = _ratio(got, want)
        gd, wd = got.double().reshape(want.shape), want.double()
        coef = ((gd * wd).sum() / (wd * wd).sum()).item()
        metrics.record(f"rel_l2_swap_attn_out_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_attn_out_{tag}", row)
        metrics.record(f"scale_coef_swap_attn_out_{tag}", coef)
        print(
            f"attn_out {tag}: rel_l2={rel:.6f} (<= {max_rel}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(ratio_lim)}) worst_row_rel_l2={row:.5f} (<= {max_row}) coef={coef:.6f} (within {max_coef})"
        )
        if rel > max_rel:
            failures.append(f"attn_out {tag}: rel L2 {rel:.5f} > {max_rel}")
        if not (ratio_lim[0] <= rmin and rmax <= ratio_lim[1]):
            failures.append(f"attn_out {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio_lim)}")
        if row > max_row:
            failures.append(f"attn_out {tag}: worst row rel L2 {row:.5f} > {max_row}")
        if abs(coef - 1) > max_coef:
            failures.append(f"attn_out {tag}: global scale coefficient {coef:.5f} not within {max_coef} of 1")

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
            sout = muts["attention"](reference_ctx(ref, layer, g, c), dctx, xs, qr, got_tk)
            att_check("scaled", sout, swant, ATT_SCALED_LIMITS)
        else:
            failures.append("attn_out: vs-CPU, chunk 0, probe and scaled checks skipped (unusable upstream output)")

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
