# SDPA Precision Matrix Results

Authoritative precision characterization for `scaled_dot_product_attention`,
produced by `test_scaled_dot_product_attention_precision_matrix.py`.

- **Last run**: 2026-06-19 (Wormhole, branch `llk_helper_library`)
- **Result**: **384 passed, 0 skipped**
- **Axes**: 8 shapes × dtype {bf16, fp32, bf8b} × math_fidelity {HiFi4, HiFi3,
  HiFi2, LoFi} × fp32_dest_acc_en {True, False} × distribution {normal, uniform}
- **Reference**: fp32 PyTorch SDPA (`softmax(Q·Kᵀ·scale)·V`, auto scale, no mask)

## Assertion policy

- **normal (`randn`)** inputs are well-conditioned → assert **PCC** (floor
  0.99 for HiFi2/3/4, 0.98 for bf8b; 0.95 / 0.93 for LoFi).
- **uniform (`rand`)** inputs make softmax average an all-positive V into a
  **near-constant** output (std ≈ 0); PCC / rel-RMS are then ill-conditioned
  (tiny abs error ÷ ≈0). Gate on **max abs error** (≤ 0.08 bf16/fp32, ≤ 0.20
  bf8b); PCC printed for observability only. This is a metric artifact, not an
  op error (Phase-0 verification report documents it).

## Default-config precision (HiFi4 + fp32_dest_acc_en=True, normal inputs)

This is what callers get with `compute_kernel_config=None`.

| shape (B,H,S,D) | dtype | PCC | max_abs | median_abs | rel_rms |
|---|---|---|---|---|---|
| (1,1,32,32)   | bf16 | 1.00000 | 0.00278 | 0.00022 | 0.00229 |
| (1,1,32,32)   | bf8b | 0.99988 | 0.01526 | 0.00221 | 0.01554 |
| (1,1,128,64)  | bf16 | 1.00000 | 0.00192 | 0.00014 | 0.00223 |
| (1,1,128,64)  | fp32 | 1.00000 | 0.00447 | 0.00027 | 0.00307 |
| (1,1,128,64)  | bf8b | 0.99987 | 0.00874 | 0.00135 | 0.01602 |
| (1,1,128,256) | bf16 | 1.00000 | 0.00254 | 0.00015 | 0.00225 |
| (1,1,128,256) | fp32 | 1.00000 | 0.00334 | 0.00025 | 0.00315 |
| (1,1,128,256) | bf8b | 0.99989 | 0.01045 | 0.00135 | 0.01516 |

All dtypes hit PCC ≥ 0.9999 and rel-RMS ≤ 0.016 at the default config.

## Fidelity sweep (shape (1,2,128,64), fp32_dest_acc_en=True, normal inputs)

| dtype | fidelity | PCC | max_abs | rel_rms |
|---|---|---|---|---|
| bf16 | HiFi4 | 1.00000 | 0.00256 | 0.00217 |
| bf16 | HiFi3 | 1.00000 | 0.00282 | 0.00228 |
| bf16 | HiFi2 | 0.99998 | 0.01246 | 0.01561 |
| bf16 | LoFi  | 0.99951 | 0.07628 | 0.07356 |
| fp32 | HiFi4 | 1.00000 | 0.00826 | 0.00346 |
| fp32 | HiFi3 | 1.00000 | 0.00833 | 0.00372 |
| fp32 | HiFi2 | 0.99995 | 0.04141 | 0.02073 |
| fp32 | LoFi  | 0.99940 | 0.13359 | 0.08637 |
| bf8b | HiFi4 | 0.99989 | 0.01106 | 0.01476 |
| bf8b | HiFi3 | 0.99989 | 0.01106 | 0.01475 |
| bf8b | HiFi2 | 0.99988 | 0.01497 | 0.01820 |
| bf8b | LoFi  | 0.99958 | 0.05479 | 0.05565 |

**Reading the sweep**: fidelity directly drives matmul precision (QK and PV).
HiFi4/HiFi3 are excellent; HiFi2 is good (fp32 rel-RMS 0.0207 is right at the
fp32 golden 0.02 band — the reason HiFi4 is the default); LoFi is noticeably
looser (expected hardware behavior). PCC stays ≥ 0.999 even at LoFi for these
well-conditioned inputs.

## Characterized boundaries

- **bf8b + fp32_dest_acc_en=False → op forces fp32 acc.** The block-float
  matmul datapath produces uncorrelated output with bf16 dest (PCC ~0.06,
  rel-RMS >5). There is no valid bf16-dest mode for bf8b, so the program
  descriptor forces `fp32_dest_acc_en=True` for bf8b inputs regardless of the
  caller's flag. The matrix's bf8b `fp32_acc=False` cells therefore pass
  (the flag is a no-op for bf8b).
- **fp32 + fp32_dest_acc_en=False is honored** and works: the intermediate CBs
  are forced to Float32 whenever the input dtype is not bf16, so the fp32 input
  round-trips correctly through the helper chain (≈ bf16-precision result, the
  legitimate throughput mode). Without the fp32 intermediate this combo yielded
  PCC 0.37 (fixed in R1).
- **uniform-distribution PCC is not a defect.** See the assertion policy above.

## Notes

- The bf16 `Q1x1x128x1024` explicit-scale Phase-0 registry red cell now passes
  (rel-RMS 0.0138 → 0.0033 under the HiFi4 default).
- Non-tile-aligned shapes are out of scope for R1 (Refinement 2) and are not in
  this matrix.
