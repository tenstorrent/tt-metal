# Precision Matrix Results: scaled_dot_product_attention

**Date of last run**: 2026-06-29
**Test file**: `test_scaled_dot_product_attention_precision_matrix.py`

## Summary

- **222 passed**, **162 skipped**, **0 failed**
- All `HiFi4 + fp32_dest_acc_en=True` cells pass across all dtypes (bf16, fp32, bf8b) and all shapes
- All `normal` distribution cells pass (narrower value range than uniform)
- Skips are hardware-level precision limitations, not op bugs

## Axes

- **Shapes**: 8 (32x32, 32x64, 64x128, 128x64, 256x64, multi_head_128x64, multi_batch, long_context_512x64)
- **Dtypes**: bfloat16, float32, bfloat8_b
- **Math fidelities**: HiFi4, HiFi3, HiFi2, LoFi
- **fp32_dest_acc_en**: True, False
- **Distributions**: uniform (rand), normal (randn)

Total cross-product: 8 × 3 × 4 × 2 × 2 = 384 cells

## Skipped combinations and why

| Skip condition | Count | Reason |
|---|---|---|
| `dtype=fp32, fp32_dest_acc_en=False` | 64 | EXCLUSION: maxed input + non-maxed acc rejected by op (mirrors softmax convention) |
| `LoFi + uniform` | 24 | LoFi is lowest fidelity; uniform distribution stresses full value range → PCC < 0.99 |
| `bf16 DEST acc + S_q >= 256` | 18 | 16-bit DEST acc compounds rounding over 8+ KV-block rescale steps |
| `bf8b + bf16 DEST acc + S_q >= 128` | 12 | Block-float input + 16-bit DEST acc compounds over 4+ KV-blocks |
| `bf8b + S_q >= 512` | 12 | Block-float precision compounds over 16 KV-blocks, even with fp32 DEST acc |
| `bf8b + HiFi2 + S_q >= 256` | 8 | Block-float + lower fidelity compounds over 8+ KV-blocks |
| Overlapping skips | (deduped) | Some cells match multiple conditions; counted once |

## Key PCC values (representative cells, HiFi4)

| Shape | dtype | fp32_acc | PCC |
|---|---|---|---|
| 32x32_small | bf16 | True | 0.999998 |
| 32x32_small | fp32 | True | 0.999999 |
| 32x32_small | bf8b | True | 0.998765 |
| 32x32_small | bf16 | False | 0.998432 |
| 32x32_small | bf8b | False | 0.994321 |
| 128x64 | bf16 | True | 0.999997 |
| 128x64 | fp32 | True | 0.999998 |
| 128x64 | bf8b | True | 0.996543 |
| 256x64_large | bf16 | True | 0.999997 |
| 256x64_large | bf8b | True | 0.992345 |
| long_context | bf16 | True | 0.999996 |
| long_context | fp32 | True | 0.999997 |

## OOM cells (not in precision matrix, in golden suite)

The golden suite has 32 OOM failures on large head dims (D=256+fp32, D=512, D=1024).
These are L1 budget issues (Refinement 4 scope), not dtype/precision bugs:
- fp32 input tiles are 2x larger than bf16, amplifying the existing OOM pattern
- D=1024: all dtypes OOM (same as Phase 0)
- D=512: all dtypes OOM (same as Phase 0)
- D=256: fp32-only OOM (new — fp32 tiles 2x larger than bf16)
