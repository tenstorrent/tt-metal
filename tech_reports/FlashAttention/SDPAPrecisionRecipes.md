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

| | FAST | COMPENSATED | BALANCED | ACCURATE | LOW_PRECISION |
|---|---|---|---|---|---|
| QK matmul | HiFi2 | HiFi2 | HiFi4 | HiFi4 | LoFi |
| Scores S, P | BF16 | BF16 | FP32 | FP32 | BF16 |
| S − m | FPU, BF16 | FPU, BF16 | FPU, TF32 operands, FP32 dest | packer FP32 L1 add | FPU, BF16 |
| exp(P) | approximate exp | approximate exp | bit-trick exp + cubic | bit-trick exp + cubic (tighter fit) | approximate exp |
| Correction c | approximate exp | approximate exp | accurate FP32 exp | accurate FP32 exp | approximate exp |
| PV matmul | HiFi2 | HiFi2 | HiFi2 | HiFi4 | LoFi |
| Row sum l | BF16 | hi/lo BF16 pair | P·1 matmul, LoFi | P·1 matmul, HiFi2 | hi/lo BF16 pair |
| Running O, l | BF16 | hi/lo BF16 pair | FP32 | FP32 | hi/lo BF16 pair |
| O / l | reciprocal, multiply | reciprocal, multiply | reciprocal (2 Newton steps), multiply | same as BALANCED | reciprocal, HiFi2 multiply |
| K/V storage | BF16 | BF16 | BF16 | BF16 | BF16, BFP8 or BFP4 |
| Input rounding (caller) | none | none | none | none | `prepare_sdpa_input` |

- **FAST** is the legacy streaming kernel (`compute_streaming.hpp`), bit-identical to omitting `precision`
  with the same chunk sizes.
- **COMPENSATED** keeps O and l as BF16 pairs, value = hi + lo. Each K chunk folds in as
  `total = (hi + lo)·c + chunk` in FP32, then `hi = RNE(total)` and `lo = RNE(total − hi)`. That keeps about 16
  significant bits across chunks instead of 8, so long-K accumulation does not swamp. When the row maximum is
  unchanged, two chunks fold at once. The math is in `streaming/compensated_sfpu.hpp`.
- **BALANCED / ACCURATE** hold scores and state in FP32. The score exp is Schraudolph's bit trick on a 2⁻¹⁰
  grid in log2, refined by a cubic, and carries a constant factor of about 0.970 that cancels in O / l.
  Across the mantissa, the refined exp ripples ±0.24% (BALANCED) or ±0.10% (ACCURATE). See
  `streaming/recipe_sfpu.hpp`.
- **LOW_PRECISION** runs both matmuls at LoFi, which truncates SrcA to 5 and SrcB to 7 significant bits.
  `prepare_sdpa_input` rounds Q to 7 bits and K/V to 5 bits (or onto the BFP4 grid) with round-to-nearest-even
  beforehand, so the truncation loses nothing further. Its state is compensated like COMPENSATED, and the final
  O / l multiply runs at HiFi2 so O is not truncated.

## Accuracy and throughput

Relative L2 error (%) against FP64 attention on the same BF16 inputs. Q is 256 rows, D128, one head, with
normally distributed inputs. "Outliers" adds a 0.1% chance of 10× spikes.

Throughput is TFLOP/s per Tensix core. It was measured on one Blackhole core with K/V resident in L1 (no DRAM
traffic), D128 and a K sequence of 8192. Legacy FAST is 1.985 TFLOP/s at Q256/K512. "vs FAST" is the geometric
mean ratio over Q chunks 128–320 and K chunks 128–512. Every recipe is slowest relative to FAST at K128.

| Recipe | K 4096 | K 32768 | K 262144 | K 262144 outliers | TFLOP/s/core (Q256/K512) | vs FAST |
|---|---:|---:|---:|---:|---:|---:|
| FAST | 2.48 | 2.74 | 18.7 | 3.42 | 1.99 | 1.00 |
| COMPENSATED | 2.46 | 2.56 | 3.20 | 1.94 | 1.60 | 0.76 |
| BALANCED | 0.38 | 0.39 | 0.38 | 0.41 | 1.21 | 0.62 |
| ACCURATE | 0.18 | 0.18 | 0.18 | 0.29 | 0.87 | 0.45 |
| LOW_PRECISION, BF16 K/V | 2.97 | 3.07 | 3.59 | 7.84 | 1.89 | 0.86 |
| LOW_PRECISION, BFP8 K/V | 3.03 | 3.14 | 3.65 | 7.98 | 1.85 | 0.85 |
| LOW_PRECISION, BFP4 K/V | 16.7 | 16.8 | 16.4 | 41.1 | 1.85 | 0.85 |

FAST's BF16 running state swamps at long K. With small logits (Q and K scaled by 0.25), FAST's error is 3.9% at
K 32768 and 54% at K 262144, against 1.7% and 2.5% for COMPENSATED. BFP8/BFP4 K/V gain nothing over BF16 at the
compute level; they cut K/V bandwidth and L1 to about a half or a quarter.

## Support

- Blackhole. Noncausal attention with an optional additive `attn_mask` of shape [1|B, 1|H, Sq, Sk] (BF16, BFP8,
  BFP4, or FP32 for BALANCED/ACCURATE). Joint attention supports the `"rear"` strategy without a mask.
- Batch and GQA are supported. Q/K lengths need not be tile or chunk multiples. Head dim and chunk sizes must be
  tile multiples, and the chunks must fit in L1.
- Tiled, interleaved DRAM inputs and output. Q and the output are BF16.
- A recipe cannot be combined with `compute_kernel_config` or `exp_approx_mode=False`, and `scale` must be the
  default 1/√D. Unsupported arguments raise before dispatch; there is no fallback.

## Blocking

With a recipe, the op chooses Q and K chunk sizes when the caller leaves them unset: no `program_config`,
or a chunk size of 0 in `SDPAProgramConfig`. For exp ring it also chooses the SDPA grid width. The chooser
(`sdpa_recipe_blocking.cpp`) scores every supported chunk pair that fits L1, counting the attn_mask buffer.
It uses a roofline cost model fitted to Blackhole timings, plus pipeline fill/drain terms that dominate
short-K cross attention. Explicit chunk sizes are always honored. Blocking never changes a recipe's
arithmetic, only its rounding order. Without a recipe, chunk sizes must be explicit.

## Ring and exp ring attention

`ring_joint_scaled_dot_product_attention` and `exp_ring_joint_scaled_dot_product_attention` take the same
`precision`. FAST runs the legacy ring kernels and keeps their chunk limits (ring: Q 128-320, K 256/384/512,
D 64/128/256; exp ring: K512, D128). COMPENSATED through LOW_PRECISION keep one online-softmax state per Q
chunk in L1 across all ring steps. They mask key tails (shard padding, `logical_n`, the joint tail) and
normalize once, on the last active step. Exp ring rows with several head segments (up to three passes) run
pass-outer and ring-inner.

- Noncausal only; cache, window and sink features are rejected.
- `logical_n` (and ring's `logical_l`) may be a host scalar or a single-value device tensor, so a captured
  trace can replay with new lengths.
- For LOW_PRECISION, prepare K/V before they are communicated.
- Ring's third output is internal scratch, not an LSE.

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
