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
