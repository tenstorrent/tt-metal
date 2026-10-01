# SDPA Precision Recipes

`ttnn.transformer.scaled_dot_product_attention` and `joint_scaled_dot_product_attention` take an optional
`precision=ttnn.SDPAPrecision.<RECIPE>`. A recipe fixes every numerical choice in the attention kernel, so
callers pick an accuracy/throughput point instead of tuning compute-kernel fields. Omitting `precision`
keeps the legacy kernel and its `compute_kernel_config` / `exp_approx_mode` controls.

```python
out = ttnn.transformer.scaled_dot_product_attention(
    q, k, v, is_causal=False, attn_mask=mask, precision=ttnn.SDPAPrecision.BALANCED
)
```

## Recipes

All recipes use the flash-attention online softmax: for each K chunk, S = QKᵀ, m = rowmax, P = exp(scale·(S − m)),
then the running numerator O and denominator l are rescaled by c = exp(scale·(m_old − m_new)) and the chunk's
PV and row sums are added. The recipes differ in the arithmetic of each step.

| | FAST | STANDARD | BALANCED | ACCURATE | LOW_PRECISION |
|---|---|---|---|---|---|
| QK matmul | HiFi2 | HiFi2 | HiFi4 | HiFi4 | LoFi |
| Scores S, P | BF16 | BF16 | FP32 | FP32 | BF16 |
| S − m | FPU, BF16 | FPU, BF16 | FPU, TF32 operands, FP32 dest | packer FP32 L1 add | FPU, BF16 |
| exp(P) | approximate exp | approximate exp | bit-trick exp + cubic | bit-trick exp + cubic (tighter fit) | approximate exp |
| Correction c | approximate exp | approximate exp | accurate FP32 exp | accurate FP32 exp | approximate exp |
| PV matmul | HiFi2 | HiFi2 | HiFi2 | HiFi4 | LoFi |
| Row sum l | BF16 | BF16 per chunk, FP32 L1 add | P·1 matmul, LoFi | P·1 matmul, HiFi2 | BF16 per chunk, FP32 L1 add |
| Running O, l | BF16 | FP32 in L1, reference max | FP32 | FP32 | FP32 in L1, reference max |
| O / l | reciprocal, multiply | reciprocal, multiply | reciprocal (2 Newton steps), multiply | same as BALANCED | reciprocal, HiFi2 multiply |
| K/V storage | BF16 | BF16 | BF16 | BF16 | BF16, BFP8 or BFP4 |
| Input rounding (caller) | none | none | none | none | `prepare_sdpa_input` |

- **FAST** is the legacy streaming kernel (`compute_streaming.hpp`), bit-identical to omitting `precision`
  with the same chunk sizes.
- **STANDARD** keeps O and l in FP32 in L1 with a *reference* row maximum m_ref. P = exp(scale·(S − m_ref))
  is computed in BF16 as in FAST, but the packer adds each chunk's PV and row sums onto the FP32 state in L1
  (the add is exact in FP32), so long-K accumulation does not swamp and needs no per-chunk fold. m_ref changes
  only when a row's maximum exceeds it by θ = 16·ln 2; that chunk rescales O and l once by c =
  exp(scale·(m_ref,old − m_ref,new)), rounding the state to BF16 once. To keep P in range while S exceeds
  m_ref, the approximate exp's input offset is lowered by τ = 28·ln 2 (exactly 28 octaves, so P is 2⁻²⁸ times
  FAST's P bit for bit and cancels in O / l). Saturation then starts τ + 0.72 above m_ref, past θ. See
  `streaming/recipe_streaming.hpp`.
- **BALANCED / ACCURATE** hold scores and state in FP32. The score exp is Schraudolph's bit trick on a 2⁻¹⁰
  grid in log2, refined by a cubic, and carries a constant factor of about 0.970 that cancels in O / l.
  Across the mantissa, the refined exp ripples ±0.24% (BALANCED) or ±0.10% (ACCURATE). See
  `streaming/recipe_sfpu.hpp`.
- **LOW_PRECISION** runs both matmuls at LoFi, which truncates SrcA to 5 and SrcB to 7 significant bits.
  `prepare_sdpa_input` rounds Q to 7 bits and K/V to 5 bits (or onto the BFP4 grid) with round-to-nearest-even
  beforehand, so the truncation loses nothing further. Its state is kept like STANDARD's, and the final
  O / l multiply runs at HiFi2 so O is not truncated.

## Accuracy and throughput

Relative L2 error (%) against FP64 attention on the same BF16 inputs. Q is 256 rows, D128, one head, with
normally distributed inputs. "Outliers" multiplies each Q, K and V element by 10 with probability 0.1%.

Throughput is TFLOP/s per Tensix core. It was measured on one Blackhole core with K/V resident in L1 (no DRAM
traffic), D128 and a K sequence of 8192. Legacy FAST is 1.99 TFLOP/s at Q256/K512. "vs FAST" is the geometric
mean ratio over Q chunks 128–320 and K chunks 128–512. STANDARD and LOW_PRECISION are furthest behind FAST at K128
(STANDARD 0.84, LOW_PRECISION 0.91–0.98, BFP8/BFP4 K/V lowest); BALANCED and ACCURATE at K512 (0.60 and 0.44).

| Recipe | K 4096 | K 32768 | K 262144 | K 262144 outliers | TFLOP/s/core (Q256/K512) | vs FAST |
|---|---:|---:|---:|---:|---:|---:|
| FAST | 2.54 | 2.75 | 18.2 | 5.27 | 1.99 | 1.00 |
| STANDARD | 2.46 | 2.55 | 3.19 | 4.40 | 1.81 | 0.88 |
| BALANCED | 0.39 | 0.39 | 0.38 | 0.59 | 1.21 | 0.63 |
| ACCURATE | 0.18 | 0.18 | 0.18 | 0.41 | 0.88 | 0.46 |
| LOW_PRECISION, BF16 K/V | 2.96 | 2.98 | 3.53 | 13.6 | 2.26 | 1.06 |
| LOW_PRECISION, BFP8 K/V | 3.05 | 3.05 | 3.61 | 13.5 | 2.20 | 1.02 |
| LOW_PRECISION, BFP4 K/V | 16.9 | 16.9 | 16.8 | 42.7 | 2.18 | 1.02 |

FAST's BF16 running state swamps at long K. With small logits (Q and K scaled by 0.25), FAST's error is 3.7% at
K 32768 and 53% at K 262144, against 1.6% and 2.5% for STANDARD. BFP8/BFP4 K/V gain nothing over BF16 at the
compute level; they cut K/V bandwidth and L1 to about a half or a quarter.

## Support

- Blackhole. Noncausal attention with an optional additive `attn_mask` of shape [1|B, 1|H, Sq, Sk] (BF16, BFP8,
  BFP4, or FP32 for BALANCED/ACCURATE). Joint attention supports the `"rear"` strategy without a mask.
- Batch and GQA are supported. Q/K lengths need not be tile or chunk multiples. Head dim and chunk sizes must be
  tile multiples, and the chunks must fit in L1.
- Tiled, interleaved DRAM inputs and output. Q and the output are BF16.
- A recipe cannot be combined with `compute_kernel_config` or `exp_approx_mode=False`, and `scale` must be the
  default 1/√D. Unsupported arguments raise before dispatch; there is no fallback.

## LOW_PRECISION inputs

SDPA never rounds, checks or converts its inputs. For LOW_PRECISION, the caller calls:

```python
q = ttnn.transformer.prepare_sdpa_input(q, is_query=True)                        # RNE to 7 bits, BF16
k = ttnn.transformer.prepare_sdpa_input(k, is_query=False, dtype=ttnn.bfloat8_b)  # RNE to 5 bits
v = ttnn.transformer.prepare_sdpa_input(v, is_query=False, dtype=ttnn.bfloat8_b)
```

Call it after any Q transforms (RoPE, norms) and before K/V are cached or communicated. A plain cast to
BFP8/BFP4 is not equivalent: it skips the 5-bit rounding that LoFi relies on, and the packer's shared-exponent
rounding differs from the round-to-nearest-even with saturation in `prepare_bfp4.cpp`.
