# scaled_dot_product_attention — precision matrix results

Last run: 2026-06-16 (Refinement 1).

Source: `test_scaled_dot_product_attention_precision_matrix.py`
(107 passed, 21 skipped). Gate = PCC floor (fidelity/context-aware);
abs/rms printed for observability. randn(seed=0) Q/K/V, mask_mode=none.

## Key data points (rel_rms = abs_rms / ref.std())

| shape (B,H,S,D) | dtype | fp32_acc | fidelity | PCC | rel_rms |
|---|---|---|---|---|---|
| 1,1,2048,64 | bf16 | True | HiFi2 | 0.99976 | 0.146 |
| 1,1,2048,64 | bf16 | True | **HiFi4** | 0.99997 | **0.0145** |
| 1,1,2048,64 | bf16 | False | HiFi2 | 0.99970 | 0.0956 |
| 1,1,4096,64 | bf16 | True | HiFi4 | 0.99997 | ~0.014 |
| 1,1,4096,64 | bf16 | False | HiFi2 | 0.99850 | 0.178 |
| 1,1,4096,64 | fp32 | True | HiFi4 | 0.99990 | 0.0289 |
| 1,1,8192,64 | bf16 | False | HiFi2 | 0.99486 | 0.756 |
| 1,1,8192,64 | fp32 | True | HiFi4 | 0.99961 | 0.0529 |

## Headline finding

For SDPA (two chained matmuls whose operands unpack to TF32), the
**dominant precision lever at long context is `math_fidelity`, not the
accumulator format**: at S=2048 bf16, HiFi2→HiFi4 drops rel_rms 0.146→0.0145
(~10×). The fp32 accumulator (fp32 intermediate CBs + fp32 DEST) compounds it
(at HiFi4, acc=True 0.0145 < acc=False 0.0203). Hence the R1 default config
(compute_kernel_config=None) now uses **HiFi4 + fp32_dest_acc_en=True**.

## Frontier (resolved — see changelog R4 / R5)

- **fp32 @ D=1024**: L1 OOM — **resolved (R4)**: host single-buffers the input/
  output CBs when the double-buffered layout would exceed the L1 budget.
- **fp32 acc=True @ S≥4096**: was rel_rms 0.028–0.053 vs golden 0.02 — **resolved
  (R5)**: NOT a hard TF32 floor. `Bkv_t` KV-chunk blocking (fewer running-output
  accumulation roundings) drops it to 0.0063 (S=4096) / 0.0093 (S=8192), under
  the 0.02 gate. The earlier "TF32 matmul floor" reading was wrong — the
  dominant term was the per-KV-chunk accumulation of the near-zero mask=none
  output, not per-element TF32 truncation.
- **bf16 / bf8b acc=False @ S≥4096**: was rel_rms 0.15–0.77 vs golden 0.12 —
  **resolved (R5)**: the 16-bit-DEST error is *accumulation* error across the
  per-KV-chunk rescale rounds, not a per-op floor. `Bkv_t` blocking cuts the
  round count by `Bkv_t` and drops mask=none S=8192 from 0.756 to 0.048. CB
  format / fidelity remain irrelevant (the R1 probe stands); the lever is
  algorithmic, exactly as the verifier predicted.

## Skips

- (float32, fp32_dest_acc_en=False): op-side EXCLUSION (covered by
  `test_exclusion_fp32_no_acc`).
