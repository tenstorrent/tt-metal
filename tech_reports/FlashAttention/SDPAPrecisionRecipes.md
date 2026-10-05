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
  `streaming/recipe_streaming.hpp`, and [Fused K chunks](#fused-k-chunks-standard-low_precision) for how K chunks
  after the first skip the row maximum.
- **BALANCED / ACCURATE** hold scores and state in FP32. The score exp is Schraudolph's bit trick on a 2⁻¹⁰
  grid in log2, refined by a cubic, and carries a constant factor of about 0.970 that cancels in O / l.
  Across the mantissa, the refined exp ripples ±0.24% (BALANCED) or ±0.10% (ACCURATE). See
  `streaming/recipe_sfpu.hpp`.
- **LOW_PRECISION** runs both matmuls at LoFi, which truncates SrcA to 5 and SrcB to 7 significant bits.
  `prepare_sdpa_input` rounds Q to 7 bits and K/V to 5 bits (or onto the BFP4 grid) with round-to-nearest-even
  beforehand, so the truncation loses nothing further. Its state is kept like STANDARD's, and the final
  O / l multiply runs at HiFi2 so O is not truncated.

## Fused K chunks (STANDARD, LOW_PRECISION)

With the reference maximum fixed, a K chunk does not need its row maximum before the exp. Every K chunk after a
Q chunk's first runs a fused chunk (`streaming/recipe_fused_chunk.hpp`) that streams row groups through
QK → exp → PV without the max reduction:

- **m_ref folded into QK.** The QK matmul gains one inner step, [Q | M] × [Kᵀ ; −e₀], where M holds m_ref in
  column 0 and −e₀ is −1 in row 0, so DEST receives S − m_ref directly. Both operands of that step are exact in
  one fidelity phase (m_ref is truncated to 7 bits, −1 has one), so STANDARD replays its HiFi2 image once and
  only its inner 0–15 half (`INNER_HALF`). The extra step costs 1/(2·D/32) of the QK work.
- **Exp on the pack thread.** The packer's SFPU takes the exp in place and packs P once, L1-accumulating it
  onto per-row partial sums.
- **Saturation check and redo.** A row whose chunk sums reach the redo threshold may have saturated the exp
  (S − m_ref beyond τ + 0.72). Its row group is redone on the reduce path: the real maximum with the θ select,
  P, PV, and one rescale of its O and l rows. Groups are checked in units (two groups for STANDARD, one for
  LOW_PRECISION); a unit that fires is re-checked group by group, so only groups that need it are redone and
  every other group keeps m_ref. The output is the same as checking each group alone.
- **Software pipeline.** QK of group g, the check of the unit ending at g − 1 and the PV of an older group are
  interleaved in K pieces, so the FPU's PV overlaps the pack thread's exp. Dense STANDARD runs Q chunks of up
  to six tiles in one-row groups so the pipeline has enough groups; the ring kernels and LOW_PRECISION keep
  two-row groups.

Fused chunks need a QK subblock of at least two tiles and no attn_mask; the factories drop their CBs
(29–31) and run the reduce path when they do not fit L1. Errors are unchanged against the unfused chunks
(19 input distributions: fused/unfused rel-L2 ratio geomean 0.94, at most 1.04), and every later optimization in this path is
bit-identical.

## Accuracy and throughput

Relative L2 error (%) against FP64 attention on the same BF16 inputs. Q is 256 rows, D128, one head, with
normally distributed inputs. "Outliers" multiplies each Q, K and V element by 10 with probability 0.1%.

Throughput is TFLOP/s per Tensix core. It was measured on one Blackhole core with K/V resident in L1 (no DRAM
traffic), D128 and a K sequence of 8192. Legacy FAST is 1.99 TFLOP/s at Q256/K512. "vs FAST" is the geometric
mean ratio over Q chunks 128–320 and K chunks 128–512. STANDARD is at or above FAST at every chunk pair (lowest
1.01, at Q128/K384) and LOW_PRECISION at least 1.21× (Q128/K128); BALANCED and ACCURATE are furthest behind at
K512 (0.61 and 0.45).

| Recipe | K 4096 | K 32768 | K 262144 | K 262144 outliers | TFLOP/s/core (Q256/K512) | vs FAST |
|---|---:|---:|---:|---:|---:|---:|
| FAST | 2.54 | 2.75 | 18.2 | 5.27 | 1.99 | 1.00 |
| STANDARD | 2.37 | 2.42 | 3.04 | 4.39 | 2.09 | 1.07 |
| BALANCED | 0.39 | 0.39 | 0.38 | 0.59 | 1.21 | 0.64 |
| ACCURATE | 0.18 | 0.18 | 0.18 | 0.41 | 0.88 | 0.47 |
| LOW_PRECISION, BF16 K/V | 2.92 | 2.94 | 3.44 | 13.5 | 2.86 | 1.37 |
| LOW_PRECISION, BFP8 K/V | 3.00 | 3.00 | 3.51 | 13.4 | 2.86 | 1.37 |
| LOW_PRECISION, BFP4 K/V | 16.8 | 16.9 | 16.7 | 42.9 | 2.85 | 1.37 |

FAST's BF16 running state swamps at long K. With small logits (Q and K scaled by 0.25), FAST's error is 3.7% at
K 32768 and 53% at K 262144, against 1.5% and 2.3% for STANDARD. BFP8/BFP4 K/V gain nothing over BF16 at the
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
D 64/128/256; exp ring: K512, D128). STANDARD through LOW_PRECISION keep one online-softmax state per Q
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

## Kernel map

| File | Role |
|---|---|
| `kernels/compute/sdpa_recipe.cpp` | Dense / joint compute kernel: Q-chunk loop over `sdpa_segment_v2` |
| `kernels/compute/ring_joint_sdpa_recipe.cpp`, `exp_ring_joint_sdpa_recipe.cpp` | Ring and exp ring compute: one recurrent state per Q chunk across ring steps (`streaming/recipe_ring.hpp`) |
| `streaming/recipe_streaming.hpp` | Shared K-chunk step: reduce path (first chunk, redo), FP32 / reference-max state, normalization |
| `streaming/recipe_fused_chunk.hpp` | Fused K chunk (above) |
| `streaming/recipe_sfpu.hpp`, `recipe_tail.hpp` | Exp variants, key-tail masking |
| `streaming/recipe_checkpoint.hpp`, `dataflow/recipe_state_transfer.hpp` | Ring multi-Q state checkpoints (compute side, writer side) |
| `dataflow/reader_recipe.cpp`, `ring_joint_*_impl.hpp`, `exp_ring_joint_*_impl.hpp` | Readers / writers; the ring and exp ring bodies are shared with the legacy kernels through a `Policy` struct (legacy FAST kernels compile unchanged) |
| `sdpa_recipe.cpp`, `sdpa_recipe_blocking.cpp` | Host: recipe → CB layout and defines; chunk chooser |

Compile-time defines set by the host:

| Define | Set for | Meaning |
|---|---|---|
| `SDPA_RECIPE_FP32` | BALANCED, ACCURATE | FP32 scores and state (DEST in FP32) |
| `SDPA_RECIPE_ACCURATE` | ACCURATE | HiFi4 PV and the tighter exp fit |
| `SDPA_RECIPE_LOFI` | LOW_PRECISION | LoFi matmuls (prepared inputs) |
| `SDPA_RECIPE_FUSED` | STANDARD, LOW_PRECISION with QK width ≥ 2 | fused K chunks (inactive with `SDPA_RECIPE_MASK`) |
| `SDPA_RECIPE_MASK` | an attn_mask | additive mask on the reduce path |
| `SDPA_RECIPE_QK_W`, `SDPA_RECIPE_PV_W` | all | matmul subblock widths |
| `SDPA_RECIPE_RING` (in the ring kernels) | ring, exp ring | key-tail masking, resident state |

**Code size.** Each program must fit the 70656 B kernel config buffer, and the ring and exp ring LOW_PRECISION
kernels sit within a few hundred bytes of it. The reduce path (`SDPA_RECIPE_COLD`), normalization and the ring
unpack copy of the fused chunk are size-optimized and out of line; the ring kernels build unpack/pack at -O2
with fused chunks. New code in these paths should be checked against the ring/exp ring LOW_PRECISION BFP8
tests (Q224 two-pass, Wan Q320).
