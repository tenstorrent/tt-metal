# SDPA precision matrix results

Last run: 2026-07-10 (Refinement 1 — dtype expansion).
Source: `test_scaled_dot_product_attention_precision_matrix.py`
(dtype × math_fidelity × fp32_dest_acc_en × input-distribution, over 8 tile-aligned shapes).
Result: **240 passed, 48 skipped** (skips = the `{float32, fp32_dest_acc_en=False}` op EXCLUSION).

## Gating

Asserts on PCC only (all other metrics printed for observability). Gate is
distribution-aware:
- `normal` (randn) — realistic slice, matches the golden suite / real models: **0.99**
  across all dtypes and fidelities (all pass).
- `uniform` [0,1) — all-positive pathological case for attention (near-uniform softmax +
  reduced-fidelity matmul over positive-only operands), dtype-independent: HiFi4 0.96,
  HiFi2 0.95, LoFi 0.75. This is documented hardware precision loss, not a correctness bug.

## Representative numbers — shape (1,1,128,64), normal (randn) distribution

| dtype | fidelity | fp32_acc | PCC | rel_rms |
|-------|----------|----------|------|---------|
| bf16  | HiFi4    | True     | 0.99999 | 0.0041 |
| bf16  | HiFi4    | False    | 0.99994 | 0.0117 |
| bf16  | HiFi2    | True     | 0.99998 | 0.0101 |
| bf16  | LoFi     | True     | 0.99950 | 0.0722 |
| bf8b  | HiFi4    | True     | 0.99988 | 0.0157 |
| bf8b  | HiFi4    | False    | 0.99982 | 0.0217 |
| bf8b  | HiFi2    | True     | 0.99987 | 0.0165 |
| fp32  | HiFi4    | True     | 0.999999 | 0.0009 |

## Notes

- fp32 requires HiFi4 + `fp32_dest_acc_en=True`; `{fp32, fp32_dest_acc_en=False}` is an op
  EXCLUSION (skipped, not run).
- bf8b + custom (-inf) mask is correct here because the softmax score intermediates
  (`cb_scores`/`cb_masked`/`cb_probs`) are promoted to bf16 for bf8b input (see changelog
  Refinement 1). Without that, the bf8b block-exponent contaminates masked blocks (PCC ~0.9).
- Larger-S D=128 fp32 OOMs (L1 budget) — Refinement 4; this matrix uses a small-S D=128
  shape to exercise the wide-head path within fp32's doubled CB footprint.
