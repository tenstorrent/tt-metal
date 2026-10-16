# Known issues detail: deepseek_v41_flash_d_p

## (RESOLVED by the overseer) The canonical prompt was shorter than the scaled rung

Resolved 2026-10-15 by the supervisor: `scaled.ladder` is now 55296 tokens in 2048-token chunks, within the existing
prompt (56320 tokens); no prompt rebuild.

The alternatives that were open before the resolution (both outside SG.1's paths): set spec `text.length: 65536` and
rebuild the prompt (`prompt.build(force=True)`; ids stay prefix-identical, new sha), or make `prompt.default_length`
count `scaled.ladder` as well as the real ladder rungs.

(Moved here from the bullet in C.swa.kv test attempt 2: the RESOLVED note pushed the bullet past 600 characters.)

## Scaled router vs golden sits at the device noise floor; judge it by the noise-floor limit
FIX.2 (2026-10-16): the fixed PCC 0.99 is gone (F76, every limit 2 x the noise floor of the reference's dtypes). At the reference kernel's attention dtypes (bf16 q / P / output, fp32 scores / max / sum) S.swa.11 records router vs golden 0.9862 (limit 0.254), router vs CPU 0.99987, pcc_swap_out 0.99982 (limit 0.9923). The history below is kept for the numbers.

Gate S.swa.11 (scaled 2x2 checkpoint, 96 experts, top 6) fails only `swap_router_vs_golden` (the router component
test's default PCC 0.99 against the golden dense routing weights). Every other check passes, pcc_swap_out 0.999848.

| run | attn_out vs golden | ffn_norm vs golden | router vs golden | rows reselected vs golden |
|---|---|---|---|---|
| CPU reference, golden input | 0.999999 | 0.999999 | 0.999999 | 0 |
| CPU, attention = its bf16 precision model (CC.precision_model) | 0.999631 | 0.999831 | 0.984229 | 0.131 |
| CPU, q_a + kv + attention bf16 models | 0.999206 | 0.999639 | 0.978566 | 0.178 |
| CPU, attn_hc .. ffn_norm bf16 models | 0.999187 | 0.999611 | 0.976934 | 0.187 |
| device, defaults (WINDOW_SDPA=high_precision) | 0.999811 | 0.999911 | 0.988729 | - |
| device, S.swa.10 (router on the CPU) | 0.999811 | 0.999911 | 0.988758 | - |
| device, DS41_WINDOW_SDPA=exact (comparison only) | 0.999930 | 0.999964 | 0.992686 | - |

The router loss is about linear in the attention's rel error (0.58 x rel). The router itself matches the CPU step on
the same inputs (pcc 0.99985, rel 9e-5, reselected 0.0015). The real-size run passed this check at 0.9959 (384
experts). The CPU numbers come from a host-only script (run_block with overrides = CC.precision_model of the listed
steps on golden layer 0, chunk of the component rung). Levers: owner decision on WINDOW_SDPA=exact (OV-3), or a
framework vs-golden check for selection outputs that splits rows by routing (K-39), as step_errors does for vs CPU.

S.swa.11 attempt 2 (host-only, `/tmp/s11b/attn_variants.py`): the reference block with only the window attention's
operands rounded as a device variant would (scale folded into q, fp32 output, no in-tile partial-dot rounding, every
other step the CPU reference). TF32 = fp32 read by the FPU (K-30).

| q | probabilities | attention rel vs golden | ffn_norm pcc | router vs golden | rows reselected |
|---|---|---|---|---|---|
| fp32 | fp32 | 0.0017 | 0.999999 | 0.999956 | 0.000 |
| bf16 | bf16 (today's sparse_sdpa high_precision) | 0.0138 | 0.999956 | 0.992121 | 0.066 |
| tf32 | bf16 | 0.0114 | 0.999970 | 0.994163 | 0.049 |
| bf16 | tf32 | 0.0127 | 0.999963 | 0.993652 | 0.053 |
| tf32 | tf32 | 0.0069 | 0.999988 | 0.995644 | 0.037 |

The device adds the in-tile rounding (DS41-16) and the other steps' noise: 0.9887 on device vs 0.9921 here. An
fp32 q + fp32 probability sparse_sdpa (fork extension; ~+0.0035 in emulation, estimated ~0.991 on device, between
today's 0.9887 and exact's 0.9927) would be a slower path than the default, chosen only to clear a fixed PCC: left to
the owner / overseer.

S.swa_engram.14.test.1 (layer 1, scaled): the fp32 CPU reference from the golden block input already fails the
0.99 PCC (0.984586; reselected 0.1245 vs the golden). Golden boundaries are bf16, and layer 1's input is a rounded
fp32 stream, so the chain's fp8 q-dq steps flip (DS41-1): q_resid 0.999985, kv 0.99989, attn_out 0.99944, ffn_norm
0.99985 on the CPU. The swap test now splits the router vs golden by routing (DS41-32): device 0.9845 / 0.1284
reselected / rel 0.0053 on the same-expert rows, the CPU 0.9846 / 0.1245 / 0.0051.

## The FPU truncates fp32 operands to TF32: round them to nearest first
S.swa.11 attempt 3. The window attention's sparse_sdpa (high_precision, fp32_output) still packed the scores and the
softmax probabilities to a bf16 CB (`cb_qk_im`), and every fp32 value an FPU op reads (srcA / srcB) loses its low 13
mantissa bits by truncation (K-30): a bias toward zero that adds coherently (scores: a flatter softmax; the normalize
`out x 1/sum`: a smaller output). CPU emulation of the attention alone (`/tmp/s11c/variants*.py`, every other step
the CPU reference, no in-tile rounding), router vs golden:

| scores | q | probabilities | attention rel vs golden | router vs golden |
|---|---|---|---|---|
| bf16 | bf16 | bf16 | 0.0170 | 0.9910 |
| fp32 | bf16 | bf16 | 0.0138 | 0.9921 |
| bf16 | TF32 nearest | TF32 nearest | 0.0153 | 0.9906 |
| TF32 truncated | TF32 nearest | TF32 truncated | 0.0122 | 0.9933 |
| TF32 nearest | TF32 nearest | TF32 nearest | 0.0087 | 0.9953 |

Device, S.swa.11 gate (layer 0, scaled 2x2, chunk 2048):

| change | attention vs CPU rel / bias | router vs golden |
|---|---|---|
| before (bf16 q, bf16 scores) | 0.0189 / -0.00075 | 0.98873 |
| + fp32_scores, fp32 q (FPU truncates) | 0.0171 / -0.00106 | 0.98957 |
| + hc_fn rounded at load (tt/common.fpu_weight) | same | 0.98959 |
| + q rounded to TF32 nearest (tt/quant.round_tf32) | 0.0163 / -0.00080 | 0.98987 |
| + scores / probabilities rounded in DEST (SFPU) | 0.0159 / -0.00062 | 0.98994 |
| + running output, row-sum, 1/sum rounded in DEST | 0.0143 / -0.00030 | 0.99169-0.99182 (PASS) |

Tried and dropped: rounding the fp32 mHC streams and the router input on the device before their matmuls
(round_tf32, 2.0 + 1.0 ms per layer): router vs CPU rel 9e-5 -> 5e-5, router vs golden unchanged within flip noise
(0.9918 without it). attn_hc's remaining error (6.1e-4, bias -3.4e-4) is the FPU's in-tile partial-dot rounding
(DS41-16), not the weight. Kernel gotcha: the SFPU is shared by the threads; with half-sync DEST the math thread's
next tiles overlap the pack thread's SFPU work on the other half, so a math-thread `add_binary_tile` beside the pack
thread's rounding corrupted every multi-group case (rel 2-400). All of the option's SFPU work runs on the pack thread
(`round_tf32_packthread_tile`, `sum_round_tf32_packthread_tile0` in sparse_sdpa_compute.cpp).

Layer 1 (swa_engram, S.swa_engram.14 on the scaled model, a later task): router vs golden 0.9847 with all of this
(0.9839 without), so the fixed 0.99 vs-golden PCC still fails there; the levers left are the owner's (ATTN_IN_MM /
WINDOW_SDPA exact) or a routing-aware vs-golden check (K-39).

### FIX.2: back to the reference kernel's precision
The overseer's FIX.2: the attention matches inference/kernel.py sparse_attn stage by stage (bf16 q / kv; fp32 scores,
running max and sum; P cast to bf16 before P.V; fp32 accumulation; bf16 output; fp32 sink), the default and the
max-precision value. The fork option `bf16_probs` (with fp32_scores) does it: Float32 max, row-sum from the TF32
probabilities, then P rounded to bf16 in DEST; bf16 output.

| attention (layer 0, scaled 2x2, chunk 2048) | vs CPU rel / bias | router vs golden | ms per layer |
|---|---|---|---|
| DS41-31 (TF32 q / P, fp32 out), before FIX.2 | 0.0143 / -3.0e-4 | 0.9918 | 28.76 |
| FIX.2 default (reference dtypes) | 0.0221 / -5.0e-4 (limit 0.0526) | 0.9862 (limit 0.254) | 26.13 |
| bf16 scores, bf16 out (perf pick) | - | - | 26.17 |

Op limits left: the score - max subtraction on the FPU is TF32-precise (probe: with bf16 scores and a max off 0 the
probabilities' error grows with |s - m|, mean -2.6e-4, max 3.3e-3 at |s - m| ~ 5; exact when the max is 0); the flash
correction stays bf16 (more than one k_chunk only; the window is one). Found: the fp32_scores TF32 P reaches P@V with
~9 bits (one-hot keys: 0.495 of P equal rne_tf32(exp), the rest truncated).

## (Cause corrected by DS41-42) What remains in the hc mixes is the linear's in-tile truncation: a column bias no setting removes

FIX.4 attempt 5: the -3.4e-4 mixes bias below is not the in-tile partial-dot truncation; it is the 10th mantissa bit of the TF32 weight, which HiFi4 does not multiply in srcA (DS41-42). The column biases that remain after DS41-42 are the weight's fixed rounding.
DS41-41, FIX.4 (2026-10-16). Golden inputs of the 5 hc gates (component chunk 1, 2x2, 1024 rows per chip), after
DS41-40 (HC_SINKHORN=sfpu_fp32). Probes: tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/probe_001..003.py.

| step | device mixes vs fp64 | multiplicative bias | step vs CPU (rel limit) | worst col bias of rms: device / CPU Sinkhorn on device mixes |
|---|---|---|---|---|
| L0 attn_hc | 3.39e-4 | -3.34e-4 | 6.07e-5 (3.52e-5) | col 20 -1.74e-3 / -1.74e-3 |
| L0 ffn_hc | 3.67e-4 | | 1.08e-4 (1.47e-4) | col 2 +9.19e-4 / +9.18e-4 |
| L1 attn_hc | 3.46e-4 | -3.36e-4 | 1.36e-4 (3.94e-4) | col 7 +1.65e-3 / +1.65e-3 |
| L1 ffn_hc | 3.87e-4 | -3.75e-4 | 1.50e-4 (2.84e-4) | col 14 -1.01e-3 / -1.01e-3 |
| L2 attn_hc | 3.47e-4 | | 1.00e-4 (4.05e-4) | col 10 -2.15e-3 / -2.15e-3 |

The sum of x^2 is exact (rel 4-6e-7), the CPU fp32 mixes are 1.9-3.6e-7 from fp64, and the SFPU Sinkhorn adds
6.5e-8..7.6e-8: every column's bias equals, to 2-3 digits, the one the CPU fp32 Sinkhorn makes from the device's own
mixes. The golden streams are 100 % bf16-valued and hc_fn is TF32-rounded at load (FP32_WEIGHT_ROUND=rne), so both
FPU operands are exact; rounding x to TF32-nearest first changes nothing (same numbers to 4 digits). What is left is
the in-tile partial-dot truncation (DS41-16; the floor probe's "bias -3.3e-4 even with TF32-exact operands",
device_floor_probes/REPORT.md): every mix shrinks by ~3.4e-4, which the sigmoid / Sinkhorn turn into a steady bias in
the columns whose mix keeps one sign. Only a multi-pass matmul would remove it (owner: not allowed).
In the swap / assembled model the hc inputs are fp32 (h_mid, block outputs), not bf16-valued: the FPU then also
truncates x (K-30), and the mixes shrink by 6.9-7.2e-4 (step rel 2.5-3.1e-4); tt/quant.round_tf32(x) before the
matmul brings that back to 3.4-3.75e-4 (step 1.1-1.5e-4) but costs 1.94 ms per call (5 bit ops on [1024, 10240] fp32)
against a 0.41 ms step: an owner pick, not taken. S.swa_engram.18 (fp32 inputs): attn_hc rel 2.72e-4 (limit 3.9e-4),
ffn_hc 3.14e-4 (limit 2.8e-4), column bias 83 / 49 x.
Re-measured FIX.4 attempt 4 (same tree, bit-identical): S.swa.15 (fp32 inputs) attn_hc vs CPU rel 6.1e-5 (limit
3.5e-5), ffn_hc 2.79e-4 (limit 1.5e-4), column bias 108 / 80 x; err_swap_out 0.0187 (limit 0.083) and 0.0218 (0.083).
The vs-CPU own-floor row / column checks (2.1-3.2 x on the components) fail by the same mechanism.

## At HiFi4 the FPU multiplies only 9 mantissa bits of matmul in1 (srcA)

FIX.4 attempt 5 (probes in `tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/`).

Operand probe (probe_008.py): chip 0 of the 2x2, L0 attn_hc golden streams x [1024, 10240] (bf16-valued) @ the
chip's hc_fn^T block [10240, 32], HiFi4, fp32 DEST, against fp64 of the same stored operands:

| operands | x @ W (W = in1 = srcA) | W^T @ x^T (W = in0 = srcB) |
|---|---|---|
| x bf16-valued, W TF32 (RNE at load) | rel 3.57e-4, bias -3.46e-4 | 7.85e-5, +3.05e-5 |
| x bf16-valued, W 9 mantissa bits | 8.07e-5, +3.00e-5 | 7.86e-5, +3.02e-5 |
| x bf16-valued, W bf16 | 8.04e-5, +2.91e-5 | 7.90e-5, +2.87e-5 |
| x fp32 mantissas, W TF32 | 6.79e-4, -6.74e-4 | 5.93e-4, -5.88e-4 |
| x fp32 mantissas, W 9 bits | 3.09e-4, -2.98e-4 | 5.93e-4, -5.89e-4 |
| HiFi3, x bf16, W TF32 | 4.17e-4, -4.08e-4 | 1.12e-4, -7.4e-5 |
| HiFi2, x bf16, W TF32 | 3.24e-3, -3.18e-3 | 5.44e-3, -5.27e-3 |

So in1 (srcA) keeps 9 explicit mantissa bits at HiFi4 (5 bits per fidelity phase, 2 phases, hidden bit included),
in0 (srcB) all 10 of TF32 (7 bits per phase). An fp32 activation in in0 is still truncated to TF32 on unpack (K-30:
-3e-4); in in1 it is truncated to 9 bits (-5.9e-4). For an fp32 weight in in1 and an fp32 / bf16 activation in in0,
round the weight to 9 bits RNE at load: what the FPU multiplies is then exact and the error unbiased.

hc step stage by stage (probe_009.py, golden inputs, layers 0..2, Sinkhorn = mhc_pre_xing(deepseek=True)), rne
(TF32 weight, attempts 3/4) -> srca (9 bits):

| step | mixes vs fp64 | step vs CPU (limit) | worst column bias of rms |
|---|---|---|---|
| L0 attn | 3.39e-4 (-3.34e-4) -> 7.46e-5 (+1.93e-5) | 6.07e-5 -> 1.07e-5 (3.52e-5) | col 20 -1.74e-3 -> +2.16e-4 |
| L0 ffn | 3.67e-4 -> 1.49e-4 | 1.08e-4 -> 4.44e-5 (1.47e-4) | 9.2e-4 -> 5.2e-4 |
| L1 attn | 3.46e-4 -> 7.50e-5 | 1.36e-4 -> 6.02e-5 (3.94e-4) | 1.65e-3 -> 5.6e-4 |
| L1 ffn | 3.87e-4 -> 1.20e-4 | 1.50e-4 -> 6.58e-5 (2.84e-4) | 1.01e-3 -> 4.6e-4 |
| L2 attn | 3.47e-4 -> 1.22e-4 | 1.00e-4 -> 5.88e-5 (4.05e-4) | 2.15e-3 -> 1.72e-3 |

With fp32-mantissa streams (the swap / assembled case) the step goes 1.19e-4..3.04e-4 -> 5.9e-5..1.56e-4.
The Sinkhorn alone stays 6.5e-8..7.6e-8; the sum of x^2 4.7e-7 (probe_007.py).

What remains is the weight's own rounding, fixed per column (host-only, the CPU step with the rounded weight vs the
fp32 one, worst column bias of rms): TF32 9.3e-5 / 2.3e-4 / 1.6e-4 / 2.4e-4 / 1.7e-3, 9 bits 1.9e-4 / 3.8e-4 /
5.0e-4 / 5.2e-4 / 1.8e-3; L0 attn column 22: host 9-bit +1.6e-4, device +1.38e-4 (the gate's 23 x). The own-floor
column checks need the exact fp32 weight: a split matmul, which the owner rules out: waiver.

## The hc steps' own-floor column checks need at least 12 mantissa bits of hc_fn: no single-pass FPU weight passes them

DS41-43. FIX.4 attempt 101 (2026-10-16). CPU, no device: weight_floor_cpu.py (out of tree: runs/run1/evidence/FIX.4/; code below)
builds the gate's own `component_checks.Expect` (the same floor: input nudges, weights fixed at fp32) and evaluates a
"perfect device": the fp32 CPU step on the golden inputs with only hc_fn rounded, everything else exact. Worst
vs-CPU column-bias ratio (limit 2; `*` = vs-golden check fails too):

| step | TF32 rne | TF32 truncate | 9 bits rne (srcA, today) | 11 bits | 12 bits |
|---|---|---|---|---|---|
| L0 attn_hc | 5.30 | rel 6.3e-5 > 3.5e-5 * | 26.5 (col 22) * (col 10 2.8) | 4.43 | 1.25 |
| L0 ffn_hc | 2.87 | rel 1.7e-4 > 1.5e-4 * | 11.3 | 2.03 | 1.01 |
| L1 attn_hc | 1.65 | 40.0 * | 5.90 | 1.22 | 0.76 |
| L1 ffn_hc | 4.07 | 22.8 | 5.18 | 1.46 | 0.96 |
| L2 attn_hc | 7.41 | 40.1 * | 3.61 | 1.31 | 1.65 |

FPU sources hold at most TF32 (10 bits; srcA 9 at HiFi4, DS41-42), so no single-pass weight format passes. Only a split
or exact matmul would, and the owner ruled those out. A weight in in0 / srcB at full TF32 would also fail (4 of 5).
It is better than srcA's 9 bits on 4 of 5 steps, but x would then sit in srcA at 9 bits truncated, which is worse for
the assembled model's fp32 streams (K-30). Not taken; an option for the owner.

Device attribution, L0 attn_hc chunk 1, today's tree (FP32_WEIGHT_ROUND=srca, HC_SINKHORN=sfpu_fp32; probe_010.py):
- step rel: device vs CPU 1.07e-5; host emulation of the 9-bit weight alone (fp64 matmul, fp32 Sinkhorn) vs CPU
  7.85e-6; device vs that emulation 1.04e-5. Mixes vs fp64: device 7.46e-5, emulation 4.99e-5, device vs emulation
  5.63e-5 (column means -1.2e-5..-4.0e-5 of rms: the FPU's accumulation, DS41-16).
- column 22 bias of rms: device +1.38e-4, emulation +1.57e-4, device vs emulation -1.8e-5. Column 10 (vs golden):
  +1.66e-4 / +1.90e-4. So the weight's rounding causes them.
- row 900 (worst, 2.5 x): |dev - cpu| 1.34e-4 vs median row 1.6e-5. The emulation alone gives 3.96e-5, device vs
  emulation 9.47e-5, the Sinkhorn 1.25e-7 (exact). That row's mixes error 1.47e-2 is below the median row's 2.88e-2,
  so the FPU's accumulation error is ordinary there; row 900's Sinkhorn just amplifies it.
- Every SFPU stage is exact (REPORT.md: 2.5e-8..5e-8; Sinkhorn alone 6.5e-8..7.6e-8, DS41-40), and the sum of x^2 is
  exact (4.7e-7). Nothing on the device path is a bug or a lower-precision choice at HiFi4 / fp32 acc / approx off.

Evidence script (`PYTHONPATH=$PWD python /localdev/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/run1/evidence/FIX.4/weight_floor_cpu.py 0 5`, CPU, ~25 min):

```python
# FIX.4 attempt 101 (CPU only): can ANY weight the FPU holds pass the step's own-floor checks?
# The gate's own Expect (floor = input nudges, weights fixed at fp32) evaluates a "perfect device": the fp32 CPU step
# with only hc_fn rounded (TF32 rne / truncate, 9-bit rne = srcA at HiFi4), everything else fp32 exact.
import sys, torch
from models.demos.common.bringup.testing.harness import spec, component_golden
from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import model_precision

def rnd(w, bits, trunc=False):
    w = w.float().contiguous(); sh = 23 - bits; i = w.view(torch.int32)
    if trunc:
        r = (i >> sh) << sh
    else:
        lsb = (i >> sh) & 1; r = ((i + ((1 << (sh - 1)) - 1) + lsb) >> sh) << sh
    return r.view(torch.float32)

s = spec(); model_precision.apply(s); g, c = component_golden(s)
cases = [(0, "attn_hc"), (0, "ffn_hc"), (1, "attn_hc"), (1, "ffn_hc"), (2, "attn_hc")]
if len(sys.argv) > 1:
    cases = cases[int(sys.argv[1]) : int(sys.argv[2])]
for layer, step in cases:
    ref = s.hooks().reference(s, layers=[*s.state_sources(layer), layer], dtype=torch.float32)
    st = _step(ref, layer, step)
    want = g.layer(c, layer)[st.output].float()
    ex = CC.Expect(s, ref, layer, st, g, c, want, None, None)
    key = {"attn_hc": "hc_attn_fn", "ffn_hc": "hc_ffn_fn"}[step]
    w = ref.w[layer]; fn0 = w[key].clone()
    print(f"\n==== L{layer} {step}", flush=True)
    for name, fw in (("fp32", fn0), ("tf32_rne", rnd(fn0, 10)), ("tf32_trunc", rnd(fn0, 10, True)),
                     ("9bit_rne", rnd(fn0, 9)), ("11bit_rne", rnd(fn0, 11)), ("12bit_rne", rnd(fn0, 12))):
        w[key] = fw
        outs = {x.name: CC.run_cpu(ex.cpu_step, x) for x in ex.cases}
        w[key] = fn0
        print(f"-- weight {name}", flush=True)
        bad = ex.evaluate(outs)
        print(f"   -> {len(bad)} failing checks", flush=True)
```

## ttnn.bringup.rms_norm's scale at fp32 is TF32 and biased: not a substitute for the hc RMS statistic

DS41-44. FIX.4 attempt 103 (2026-10-16), probe_013.py (`tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/`).
Chip 0's L0 attn_hc golden streams [1024, 10240] fp32 (bf16-valued) and random bf16 rows of the same shape,
HiFi4, fp32 DEST, eps = cfg.eps. For bf16-valued x the fork's y = x * r is exact (8 x 11 mantissa bits), so
y / x is the r the FPU multiplied (y == x * r_row on every element):

| statistic | golden streams | random bf16 rows |
|---|---|---|
| mhc_pre_xing_pack column 24 (today), sum x^2 vs fp64 | bias -4.5e-7, rms 4.7e-7 | -4.4e-7, 4.6e-7 |
| ttnn.bringup.rms_norm r vs fp64 r | bias +2.91e-4, rms 3.69e-4 | +2.79e-4, 3.73e-4 |
| rows where fork r == TF32-RNE(exact r) | 62 % (off by up to 2 ulps) | 58 % |
| rows where fork r == TF32-truncate(exact r) | 8.6 % (mean +0.96 ulp) | 15 % |

So the fork's r is on the TF32 grid (srcB holds it at 10 bits), and its fp32 r before that step is already off by a
large fraction of a TF32 ulp: x^2 is summed by an FPU reduce (x^2 read at TF32, in-tile partial sums at ~TF32,
DS41-16; the cross-chunk carry reloaded through srcA, its CHANGELOG entry 3). A `return_sum_sq` option would remove the
r rounding but not the FPU sum, so it is not "at least as accurate" as today's SFPU fp32 statistic (4.7e-7, and the
rsqrt inside mhc_pre_xing 5e-7, REPORT.md). The overseer's OV-10 (closed) reached the same decision from REPORT.md.
