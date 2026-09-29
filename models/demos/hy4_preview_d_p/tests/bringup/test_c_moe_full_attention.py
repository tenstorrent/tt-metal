# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.attention.test.1). Same step and checks as test_c_dense_full_attention.py (layer 0), at
LAYER = 1 with layer 1's own weights and sinks (sink in [-4.96, 1.01]): attn_out [S, 6144] from attn_norm [S, 6144],
q_resid [S, 2048] and topk [S, 2048] (the layer-1 indexer's key positions, -1 padded): gated sparse MLA, 64 heads,
absorbed into the 576-wide latent (512 kv_a_layernorm(latent), eps 1e-6 | 64 RoPE'd k_rope, interleaved RoPE,
theta 1e7), scale 1/16, learnable per-head sink logit (its softmax mass dropped), wkv_b2 to 256 per head,
sigmoid(linear_gate(attn_norm)) per (head, v-dim), then o_proj. Stateful: it writes this chunk's kv_latent rows and
reads the prefix [0, start) from the golden layer-1 state (bf16). Golden: s4096 chunk 1 (start 2048, 2048 rows).
Measured on the layer-1 golden (CPU; mutations of the reference; study /tmp/hy4_c_attn1/mut.py, outside the repo):

    variant (golden chunk 1)                   PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                             0.999999   0.00173   [0.9998, 1.0008]     0.0027
    bf16 W / q / kv, fp32 acc (device est)     0.999990   0.00452   [0.9959, 1.0036]     0.0132
    same + bf16 scores / P                     0.999989   0.00464   [0.9960, 1.0037]     0.0132
    device (TtHy4Attention, this gate)         0.999978   0.00663   [0.9942, 1.0062]     0.0155
    kv_a_layernorm eps 1e-5                    0.999999   0.00173   [0.9998, 1.0008]     0.0027
    RoPE positions + 1                         0.999947   0.0103    [0.9151, 1.0034]     0.313
    x 1.02                                     0.999999   0.0201    [1.0198, 1.0208]     0.021
    last row zeroed / last tile row zeroed     0.99979 / 0.99257   0.021 / 0.122        1.0
    dense causal attention (topk ignored)      0.999767   0.0216    [0.9949, 1.0051]     0.083
    prefix row halves swapped (cache layout)   0.999694   0.0247    [0.9868, 1.0059]     0.107
    sink raw to sparse_sdpa (sink / 16)        0.999493   0.0366    [1.0057, 1.0339]     0.060
    own key dropped / topk - 1                 0.99931 / 0.99904   0.037 / 0.044        0.23 / 0.23
    sliding window of the last 2048            0.998878   0.0475    [0.9862, 1.0149]     0.198
    scale 192^-0.5 / 576^-0.5 (op default)     0.99810 / 0.97721   0.066 / 0.218        0.11 / 0.29
    sink sign                                  0.997665   0.0773    [1.0088, 1.0702]     0.125
    dense non-causal                           0.996036   0.0892    [0.9566, 1.0136]     0.254
    RoPE from position 0                       0.948675   0.318     (fails PCC; at layer 0 it passed, 0.9985)
    no sink / sink mass kept, rotate-half RoPE, no q / k RoPE, zero prefix, no gate, gate 0.5, gate or o head
    halves, sink x 16, no kv norm (weight), SP row halves swapped                                 < 0.99

"eps 1e-5" through "dense non-causal" pass the 0.99 PCC gate. Layer 1 separates the bugs better than layer 0 (dense
causal rel 0.0216 vs 0.0064), but its device noise is higher (bf16 estimate 0.0045 / worst row 0.013 vs 0.0033 /
0.0042 at layer 0; the device module measured 0.0066 / 0.0165). The layer-0 limits (rel 0.01, ratio [0.99, 1.01],
worst row 0.02) would leave the device a 1.2x margin on the worst row, so the limits here are rel L2 <= 0.015, row
norm ratio in [0.985, 1.015], worst row <= 0.04 (>= 2x the device), still below every bug above (RoPE + 1 and x 1.02
fail the worst row / the ratio). The same limits apply to all four checks:

1. Golden chunk 1, vs the golden (finite, S x 6144 elements).
2. Golden chunk 0 (start 0, empty prefix, row r holds only [0, r] and -1 pads), vs the golden: pads read as key 0
   score worst row 1.41, own key dropped 1.0, sink / 16 rel 0.038 / worst row 0.079, topk - 1 0.27, non-causal 0.83
   (device 0.0067 / [0.9927, 1.0066] / 0.0158; bf16 estimate 0.0045 / 0.0092).
3. Probe: golden chunk 1 inputs with a synthetic topk (per row 64 random causal positions, unsorted, the rest -1),
   vs the CPU step on the same inputs:

       variant (probe, vs CPU)                  rel L2    per-row norm ratio   worst row rel L2
       bf16 device estimate                     0.00355   [0.9950, 1.0029]     0.0097
       device (this gate)                       0.00670   [0.9944, 1.0045]     0.0165
       dense causal (topk ignored)              1.035     [0.8986, 2.4849]     2.20
       topk - 1 / pads read as key 0            0.587 / 1.214                  1.44 / 2.04
       prefix row halves swapped                0.198     [0.8567, 1.2112]     0.65
       own key dropped / RoPE + 1               0.075 / 0.051                  0.66 / 0.48
       sink / 16 / scale 192^-0.5 / no sink     0.068 / 0.067 / 1.21           0.18 / 0.11 / 2.11

4. eps: the golden cannot see the kv_a_layernorm eps (latent mean square 0.38 .. 0.92). The module runs on the
   golden chunk 1 attn_norm scaled by 1e-3 (bf16), vs the CPU step on the same input:

       variant (attn_norm x 1e-3, vs CPU)       rel L2    per-row norm ratio   worst row rel L2
       bf16 device estimate                     0.00233   [1.0002, 1.0011]     0.0027
       device (this gate)                       0.00385   [0.9984, 1.0016]     0.0061
       eps 1e-5 (rms_norm_eps, ttMLA's default) 0.490     [0.422, 0.960]       0.59
       eps 0 / eps 2e-6                         0.522 / 0.174                  0.62 / 0.21

Every mutation in the tables fails at least one of the four checks (checked against the study's numbers). The chunk's
kv_latent rows are not read back here; chunk 0 attends only to the chunk's own writes, and the ladder's state gate
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.015  # ||got - want|| / ||want|| over [S, 6144] (layer 0: 0.01; layer-1 device 0.0067)
RATIO = (0.985, 1.015)  # per-row ||got|| / ||want|| (layer 0: [0.99, 1.01]; layer-1 device [0.9927, 1.0066])
MAX_ROW_REL = 0.04  # worst per-row rel L2 (layer 0: 0.02; layer-1 device 0.0165)
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


def _check(tag: str, got: torch.Tensor, want: torch.Tensor) -> list[str]:
    """Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list."""
    if got.numel() != want.numel():
        return [f"{tag}: output has {got.numel()} elements, want {tuple(want.shape)}"]
    if not torch.isfinite(got.float()).all():
        return [f"{tag}: non-finite output"]
    rel, rmin, rmax, row, arg = _errors(got, want)
    p = metrics.pcc(got.float().reshape(want.shape), want.float())
    metrics.record(f"{tag}_rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"{tag}_row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"{tag}_row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"{tag}_worst_row_rel_l2_{STEP}_L{LAYER:02d}", row)
    print(
        f"{tag}: pcc={p:.6f} rel_l2={rel:.6f} (<= {MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
        f"(in {list(RATIO)}) worst_row_rel_l2={row:.5f} (<= {MAX_ROW_REL}, row {arg})"
    )
    fails = []
    if rel > MAX_REL_L2:
        fails.append(f"{tag}: relative L2 error {rel:.5f} > {MAX_REL_L2}")
    if not (RATIO[0] <= rmin and rmax <= RATIO[1]):
        fails.append(f"{tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(RATIO)}")
    if row > MAX_ROW_REL:
        fails.append(f"{tag}: worst row rel L2 {row:.5f} > {MAX_ROW_REL} (row {arg})")
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
    fails += _check("scaled", sout, swant)

    assert not fails, "; ".join(fails)
