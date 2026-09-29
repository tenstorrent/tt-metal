# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.attention.test.1). The step is attn_out [S, 6144] from attn_norm [S, 6144], q_resid [S, 2048]
and topk [S, 2048] (the indexer's key positions, -1 padded): gated sparse MLA with 64 heads, absorbed into the
576-wide latent (512 kv_a_layernorm(latent), eps 1e-6 | 64 RoPE'd k_rope, interleaved RoPE, theta 1e7), scale 1/16,
a learnable per-head sink logit (its softmax mass dropped), wkv_b2 to 256 per head, sigmoid(linear_gate(attn_norm))
per (head, v-dim), then o_proj. Stateful: it writes this chunk's kv_latent rows and reads the prefix [0, start) from
the golden state (bf16). Golden: s4096 chunk 1 (start 2048, 2048 rows, every row holds 2048 valid positions).
Measured (CPU; mutations of the reference; study script outside the repo):

    variant (golden chunk 1)                   PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                             0.999998   0.00178   [0.9998, 1.0002]     0.0019
    bf16 W / q / kv / P, fp32 acc (device est) 0.999995   0.00329   [0.9988, 1.0010]     0.0042
    kv_a_layernorm eps 1e-5                    0.999998   0.00178   [0.9998, 1.0002]     0.0019
    RoPE positions + 1                         0.999991   0.00424   [0.9599, 1.0008]     0.134
    dense causal attention (topk ignored)      0.999980   0.00639   [0.9987, 1.0028]     0.017
    prefix row halves swapped (cache layout)   0.999971   0.00767   [0.9967, 1.0031]     0.025
    sliding window of the last 2048            0.999933   0.0116    [0.9952, 1.0046]     0.032
    own key dropped / topk - 1                 0.99993 / 0.99989   0.012 / 0.015        0.063
    x 1.02                                     0.999998   0.0201    [1.0198, 1.0202]     0.020
    scale 192^-0.5 / 576^-0.5 (op default)     0.99969 / 0.99702   0.025 / 0.077        0.045 / 0.139
    dense non-causal                           0.999538   0.0309    [0.9927, 1.0173]     0.066
    sink raw to sparse_sdpa (sink / 16)        0.999459   0.0352    [1.0056, 1.0218]     0.053
    RoPE from position 0                       0.998475   0.0555    [0.9579, 1.0283]     0.198
    rotate-half RoPE / no q RoPE               0.9950 / 0.9926   0.103 / 0.127        0.28 / 0.27
    no sink / sink mass kept (renormalized)    0.992780   0.126     [1.0115, 1.0672]     0.204
    last row zeroed / last tile row zeroed     0.99970 / 0.99228   0.025 / 0.124        1.0
    sink sign, no k RoPE, zero prefix, no gate, gate 0.5, gate or o head halves, sink x 16, no kv norm (weight),
    SP row halves swapped                      < 0.99

Everything from "eps 1e-5" to "last tile row zeroed" passes the 0.99 PCC gate. So the test also checks:

1. Golden chunk 1, vs the golden: output finite, S x 6144 elements, rel L2 <= 0.01, every row's norm ratio in
   [0.99, 1.01], worst row rel L2 <= 0.02 (device estimate 0.0033 / [0.9988, 1.0010] / 0.0042).
2. Golden chunk 0 (start 0, empty prefix, row r holds only [0, r] and -1 pads), same limits vs the golden: the pad
   handling is not exercised by chunk 1. There pads read as key 0 score worst row 0.83, own key dropped 1.0, sink / 16
   0.27, topk - 1 0.18, non-causal 0.74 (device estimate rel 0.0033, worst row 0.0048).
3. Probe: golden chunk 1 inputs with a synthetic topk (per row 64 random causal positions, unsorted, the rest -1),
   vs the CPU step on the same inputs, same limits. It proves the module attends to exactly the given positions, in
   any order, with the prefix cache mapped right:

       variant (probe, vs CPU)                  rel L2    per-row norm ratio   worst row rel L2
       bf16 device estimate                     0.00304   [0.9985, 1.0016]     0.0051
       dense causal (topk ignored)              0.477     [1.0445, 1.4537]     0.85
       topk - 1 / pads read as key 0            0.140 / 0.254                  0.36 / 0.57
       prefix row halves swapped                0.0485    [0.9431, 1.0693]     0.19
       own key dropped / RoPE + 1               0.027 / 0.0104                 0.26 / 0.064
       sink / 16 / scale 192^-0.5 / no sink     0.157 / 0.042 / 0.532          0.24 / 0.080 / 0.91

4. eps: the golden cannot see the kv_a_layernorm eps (latent mean square >= 0.37, 3.7e5 x eps). The module runs on the
   golden chunk 1 attn_norm scaled by 1e-3 (bf16), vs the CPU step on the same input (pre-norm mean square
   3.7e-7 .. 2.3e-6); limits rel L2 <= 0.01, worst row <= 0.02:

       variant (attn_norm x 1e-3, vs CPU)       rel L2    per-row norm ratio   worst row rel L2
       bf16 device estimate                     0.00297   [0.9999, 1.0012]     0.0034
       eps 1e-5 (rms_norm_eps, ttMLA's default) 0.432     [0.505, 0.997]       0.52
       eps 0 / eps 2e-6                         0.306 / 0.134                  0.38 / 0.16

   (At x 1e-2, eps 1e-5 scores only rel 0.036.)

The chunk's kv_latent rows are not read back here (the module's state API is the implementer's); chunk 0 attends
only to the chunk's own writes, and the ladder's state gate compares the cache.
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 6144]
RATIO = (0.99, 1.01)  # per-row ||got|| / ||want||
MAX_ROW_REL = 0.02  # worst per-row rel L2
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
