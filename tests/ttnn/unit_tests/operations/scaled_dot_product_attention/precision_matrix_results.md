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

## Known frontier (left failing — see changelog R1)

- **fp32 @ D=1024**: L1 OOM (fp32 CBs scale with d_t). Follow-up: memory-budget.
- **fp32 acc=True @ S≥4096**: rel_rms 0.028–0.053 vs golden 0.02 — TF32 matmul
  floor (fp32 unpacks to TF32 through srcA/srcB; HiFi4 is already max fidelity).
- **bf16 / bf8b acc=False @ S≥4096**: rel_rms 0.15–0.77 vs golden 0.12 — 16-bit
  DEST register floor (golden pins fp32_dest_acc_en=False + HiFi2; probe with
  fp32-CB-always gave identical rms, proving CB format is irrelevant here).

## Skips

- (float32, fp32_dest_acc_en=False): op-side EXCLUSION (covered by
  `test_exclusion_fp32_no_acc`).
