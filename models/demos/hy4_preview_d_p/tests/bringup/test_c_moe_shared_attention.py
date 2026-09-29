# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.attention.test.1). Same step and the same four checks as test_c_moe_full_attention.py
(layer 1), at LAYER = 2 with layer 2's own weights and sinks (sink in [-2.32, 0.74]). attn_out [S, 6144] comes from
attn_norm [S, 6144], q_resid [S, 2048] and topk [S, 2048]. A shared layer has no indexer, so topk is the output of
topk_shared: layer 1's selection (cfg.topk_source(2) = 1), a graph input here, so the test needs no
ctx.extra["shared_topk"]. The step is gated sparse MLA, 64 heads, absorbed into the 576-wide latent (512
kv_a_layernorm(latent), eps 1e-6 | 64 RoPE'd k_rope), scale 1/16, a per-head sink, the sigmoid output gate, then o_proj.
It reads the prefix [0, start) from the golden layer-2 kv_latent state (bf16). Golden: s4096 chunk 1 (start 2048).
Measured on the layer-2 golden (CPU mutations of the reference; study /tmp/hy4_c_attn2/mut.py, outside the repo):

    variant (golden chunk 1)                   PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                             0.999998   0.00200   [0.9998, 1.0002]     0.0022
    bf16 W / q / kv (+ bf16 scores / P)        0.999990   0.00450   [0.9995, 1.0006]     0.0057
    device (TtHy4Attention, this gate)         0.999981   0.00613   [0.9980, 1.0012]     0.0081  (coef 0.99970)
    kv_a_layernorm eps 1e-5                    0.999998   0.00200   [0.9998, 1.0002]     0.0022
    RoPE positions + 1                         0.999996   0.00268   [0.9980, 1.0002]     0.064
    sink raw to sparse_sdpa (sink / 16)        0.999973   0.00752   [1.0003, 1.0104]     0.037
    prefix row halves swapped (cache layout)   0.999968   0.00804   [0.9952, 1.0011]     0.026
    own key dropped / topk - 1                 0.99997 / 0.99994   0.0084 / 0.0111      0.074 / 0.074
    dense causal attention (topk ignored)      0.999956   0.00945   [0.9997, 1.0083]     0.033
    sink sign                                  0.999912   0.0136    [1.0002, 1.0200]     0.067
    sliding window of the last 2048            0.999863   0.0166    [0.9892, 1.0013]     0.047
    x 1.02                                     0.999998   0.0201    [1.0198, 1.0202]     0.020
    last row zeroed / last tile row zeroed     0.99970 / 0.99234   0.024 / 0.124        1.0
    no sink / sink mass kept                   0.999653   0.0273    [1.0011, 1.0548]     0.140
    dense non-causal                           0.999607   0.0281    [0.9953, 1.0190]     0.126
    scale 192^-0.5 / 576^-0.5 (op default)     0.99959 / 0.99903   0.029 / 0.044        0.107 / 0.193
    RoPE from position 0                       0.999037   0.0439    [0.9955, 1.0225]     0.216
    rotate-half RoPE, no q / k RoPE            0.9967 .. 0.9982    0.060 .. 0.081       0.22 .. 0.26
    sink x 16, zero prefix, no gate, gate 0.5, gate or o head halves, no kv norm (weight), SP row halves  < 0.99

Everything from "eps 1e-5" through "rotate-half RoPE" passes the 0.99 PCC gate. At layer 2 the device is quieter
on the golden than at layer 1 (rel 0.0061 / worst row 0.0081 vs 0.0066 / 0.0165), but the bugs are closer to it.
Passing the sink raw scores rel 0.0075 / ratio max 1.0104 / worst row 0.037, which is inside the layer-1 limits
(0.015, [0.985, 1.015], 0.04). The limits here are therefore rel L2 <= 0.012, row norm ratio in [0.99, 1.01] and
worst row <= 0.025: 2x / 5x / 3x the device. A float64 global coefficient <got, want> / <want, want> must also sit
within 0.004 of 1 (device 0.9997 .. 0.9999), so a uniform x 1.005 scale fails. These limits apply to checks 1-3:

1. Golden chunk 1, vs the golden (finite, S x 6144 elements).
2. Golden chunk 0 (start 0, empty prefix, row r holds only [0, r] and -1 pads), vs the golden. Here pads read as key 0
   score worst row 0.90, own key dropped 1.0, sink / 16 worst row 0.091, topk - 1 0.23, no sink 0.91 and non-causal
   0.94 (device 0.0059 / [0.9979, 1.0010] / 0.0080).
3. Probe: golden chunk 1 inputs with a synthetic topk (per row 64 random causal positions, unsorted, the rest -1),
   vs the CPU step on the same inputs:

       variant (probe, vs CPU)                  rel L2    per-row norm ratio   worst row rel L2
       bf16 device estimate                     0.00398   [0.9993, 1.0010]     0.0052
       device (this gate)                       0.00538   [0.9973, 1.0015]     0.0079
       dense causal (topk ignored)              0.410     [0.9537, 1.2385]     0.74
       topk - 1 / pads read as key 0            0.117 / 0.594                  0.42 / 0.86
       prefix row halves swapped                0.064     [0.9557, 1.0418]     0.29
       own key dropped / RoPE + 1               0.023 / 0.010                  0.38 / 0.052
       sink / 16 / scale 192^-0.5 / no sink     0.086 / 0.095 / 0.423          0.14 / 0.14 / 0.80

4. eps: the golden cannot see the kv_a_layernorm eps (latent mean square 0.67 .. 2.94). The module runs again on
   the golden chunk 1 attn_norm scaled by 1e-3 (bf16) and is compared with the CPU step on the same input. The
   device is noisier on this input (rel 0.0119 vs a bf16 estimate of 0.0049), so this check has its own limits:
   rel L2 <= 0.03, row norm ratio in [0.99, 1.01], worst row <= 0.05, coef within 0.004.

       variant (attn_norm x 1e-3, vs CPU)       rel L2    per-row norm ratio   worst row rel L2
       bf16 device estimate                     0.00492   [0.9999, 1.0009]     0.0055
       device (this gate)                       0.0119    [0.9963, 1.0034]     0.0206  (coef 0.99992)
       eps 1e-5 (rms_norm_eps, ttMLA's default) 0.467     [0.709, 1.080]       0.62
       eps 0 / eps 2e-6                         0.228 / 0.127                  0.30 / 0.17

Every mutation in the tables fails at least one of the four checks, checked against the study's numbers. The chunk's
kv_latent rows are not read back here: chunk 0 attends only to the chunk's own writes, and the ladder's state gate
compares the cache.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import Ctx
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
STEP = "attention"
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.012  # ||got - want|| / ||want|| over [S, 6144] (layer-2 device 0.0061; layer 1: 0.015)
RATIO = (0.99, 1.01)  # per-row ||got|| / ||want|| (layer-2 device [0.9973, 1.0015]; layer 1: [0.985, 1.015])
MAX_ROW_REL = 0.025  # worst per-row rel L2 (layer-2 device 0.0081; layer 1: 0.04)
MAX_COEF = 0.004  # |<got, want> / <want, want> - 1| in float64: a uniform scale error (x 1.01 gives 0.01)
SCALED_LIMITS = (0.03, (0.99, 1.01), 0.05, MAX_COEF)  # the attn_norm x 1e-3 check (layer-2 device 0.0119 / 0.0206)
PROBE_KEYS = 64  # probe topk: this many random causal positions per row, unsorted, the rest -1
PROBE_SEED = 0
SYN_SCALE = 1e-3  # eps check: attn_norm * SYN_SCALE (bf16), where kv_a_layernorm's eps 1e-6 matters


def _errors(got: torch.Tensor, want: torch.Tensor):
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm()).item()
    wn = want.norm(dim=-1).clamp_min(1e-30)
    ratio = got.norm(dim=-1) / wn
    row = (got - want).norm(dim=-1) / wn
    return rel, ratio.min().item(), ratio.max().item(), row.max().item(), row.argmax().item()


def _check(tag: str, got: torch.Tensor, want: torch.Tensor, limits=None) -> list[str]:
    """Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list.
    limits = (max rel L2, (ratio min, ratio max), max worst-row rel L2, max |coef - 1|)."""
    max_rel, ratio_lim, max_row, max_coef = limits or (MAX_REL_L2, RATIO, MAX_ROW_REL, MAX_COEF)
    if got.numel() != want.numel():
        return [f"{tag}: output has {got.numel()} elements, want {tuple(want.shape)}"]
    if not torch.isfinite(got.float()).all():
        return [f"{tag}: non-finite output"]
    rel, rmin, rmax, row, arg = _errors(got, want)
    gd, wd = got.double().reshape(want.shape), want.double()
    coef = ((gd * wd).sum() / (wd * wd).sum()).item()
    p = metrics.pcc(got.float().reshape(want.shape), want.float())
    metrics.record(f"{tag}_rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"{tag}_row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"{tag}_row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"{tag}_worst_row_rel_l2_{STEP}_L{LAYER:02d}", row)
    metrics.record(f"{tag}_scale_coef_{STEP}_L{LAYER:02d}", coef)
    print(
        f"{tag}: pcc={p:.6f} rel_l2={rel:.6f} (<= {max_rel}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
        f"(in {list(ratio_lim)}) worst_row_rel_l2={row:.5f} (<= {max_row}, row {arg}) "
        f"coef={coef:.6f} (within {max_coef} of 1)"
    )
    fails = []
    if rel > max_rel:
        fails.append(f"{tag}: relative L2 error {rel:.5f} > {max_rel}")
    if not (ratio_lim[0] <= rmin and rmax <= ratio_lim[1]):
        fails.append(f"{tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio_lim)}")
    if row > max_row:
        fails.append(f"{tag}: worst row rel L2 {row:.5f} > {max_row} (row {arg})")
    if abs(coef - 1) > max_coef:
        fails.append(f"{tag}: global scale coefficient {coef:.5f} not within {max_coef} of 1")
    return fails


def _probe_topk(start: int, rows: int, k: int, n: int, seed: int) -> torch.Tensor:
    """[rows, k] int64: per row min(n, pos + 1) distinct random positions in [0, pos], unsorted, then -1 pads."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.full((rows, k), -1, dtype=torch.int64)
    for i in range(rows):
        p = start + i
        m = min(n, p + 1)
        out[i, :m] = torch.randperm(p + 1, generator=gen)[:m]
    return out


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    start = c * g.chunk
    assert start > 0, "component chunk must start after 0 so the attention reads a real KV prefix"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(
        fn, "cpu_bridge", False
    ), "device_component returned a CPU bridge; attention is not on the device"
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # 1. Golden chunk 1: scale, per-row norm, worst row.
    fails = _check("golden", out, want)

    # 2. Golden chunk 0 (start 0, empty prefix, -1 pads in every row).
    g0 = g.layer(0, LAYER)
    in0 = [g0[i].float() if g0[i].is_floating_point() else g0[i] for i in st.inputs]
    dctx0 = Ctx(LAYER, 0, g.chunk, None, {"state_prefix": g.state(LAYER), "prefix_len": 0, "max_seq": g.seq})
    out0 = fn(reference_ctx(ref, LAYER, g, 0), dctx0, *in0)
    fails += _check("chunk0", out0, g0[st.output])

    # 3. Probe: synthetic unsorted topk (64 random causal positions per row), vs the CPU step on the same inputs.
    x, qr, tk = inputs
    ptk = _probe_topk(start, tk.shape[0], tk.shape[-1], PROBE_KEYS, PROBE_SEED)
    cpu = ref.component(LAYER, STEP)
    pwant = cpu(rctx, x, qr, ptk).float()
    pout = fn(rctx, dctx, x, qr, ptk)
    fails += _check("probe", pout, pwant)

    # 4. eps: attn_norm scaled so kv_a_layernorm's eps is visible, vs the CPU step on the same input.
    xs = (x * SYN_SCALE).bfloat16().float()
    swant = cpu(rctx, xs, qr, tk).float()
    sout = fn(rctx, dctx, xs, qr, tk)
    fails += _check("scaled", sout, swant, SCALED_LIMITS)

    assert not fails, "; ".join(fails)
