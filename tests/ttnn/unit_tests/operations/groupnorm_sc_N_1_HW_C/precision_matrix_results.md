# groupnorm_sc_N_1_HW_C — precision matrix results

Last run: 2026-09-17 (Refinement 4), Blackhole, `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_numerics.py`.
Reference: torch `group_norm` in fp32 on the same (quantized) input; `rel_rms = ||y − ŷ||₂ / ||ŷ||₂`.
Weights: gamma + beta in the activation dtype (ROW_MAJOR; TILE for bf8b — a block format has no ROW_MAJOR form).
Gates: PCC ≥ 0.999 / rel RMS ≤ 0.01 (fp32), 0.995 / 0.02 (bf16), 0.99 / 0.10 (bf8b) — the golden `TOLERANCES`.

## Default compute config (HiFi4, fp32 DEST, exact SFPU) — `test_groupnorm_sc_N_1_HW_C_precision_matrix`

80 passed / 16 skipped (bf8b × ROW_MAJOR: block formats have no ROW_MAJOR representation). TILE and ROW_MAJOR
give the same numbers to the printed precision (the RM leg tilizes the same values), so one row per shape × dtype.

| shape (N,1,HW,C), G | dtype | dist | PCC | rel RMS | max abs | median abs | p99 abs |
|---|---|---|---|---|---|---|---|
| (1,1,32,32) 1 | bf16 | uniform / normal | 0.999998 / 0.999998 | 0.0018 / 0.0017 | 0.013 / 0.018 | 0.0015 / 0.0011 | 0.008 / 0.008 |
| (1,1,32,32) 1 | fp32 | uniform / normal | 0.999999 / 1.000000 | 0.0018 / 0.0006 | 0.011 / 0.004 | 0.0016 / 0.0005 | 0.009 / 0.003 |
| (1,1,32,32) 1 | bf8b | uniform / normal | 0.999924 / 0.999927 | 0.0124 / 0.0121 | 0.073 / 0.097 | 0.0127 / 0.0124 | 0.051 / 0.051 |
| (1,1,64,128) 4 | bf16 | uniform / normal | 0.999996 / 0.999999 | 0.0030 / 0.0017 | 0.025 / 0.017 | 0.0025 / 0.0010 | 0.013 / 0.008 |
| (1,1,64,128) 4 | fp32 | uniform / normal | 0.999998 / 1.000000 | 0.0023 / 0.0013 | 0.018 / 0.017 | 0.0017 / 0.0007 | 0.011 / 0.007 |
| (1,1,64,128) 4 | bf8b | uniform / normal | 0.999916 / 0.999918 | 0.0134 / 0.0132 | 0.090 / 0.087 | 0.0126 / 0.0121 | 0.052 / 0.050 |
| (1,1,64,320) 32 (straddling) | bf16 | uniform / normal | 0.999994 / 0.999998 | 0.0034 / 0.0017 | 0.031 / 0.028 | 0.0025 / 0.0010 | 0.015 / 0.008 |
| (1,1,64,320) 32 | fp32 | uniform / normal | 0.999997 / 1.000000 | 0.0026 / 0.0010 | 0.017 / 0.020 | 0.0017 / 0.0005 | 0.011 / 0.006 |
| (1,1,64,320) 32 | bf8b | uniform / normal | 0.999917 / 0.999923 | 0.0132 / 0.0127 | 0.105 / 0.097 | 0.0117 / 0.0109 | 0.052 / 0.050 |
| (2,1,256,256) 8 | bf16 | uniform / normal | 0.999996 / 0.999999 | 0.0029 / 0.0017 | 0.029 / 0.037 | 0.0023 / 0.0010 | 0.013 / 0.008 |
| (2,1,256,256) 8 | fp32 | uniform / normal | 0.999998 / 1.000000 | 0.0021 / 0.0011 | 0.017 / 0.021 | 0.0013 / 0.0005 | 0.009 / 0.006 |
| (2,1,256,256) 8 | bf8b | uniform / normal | 0.999918 / 0.999922 | 0.0132 / 0.0128 | 0.113 / 0.120 | 0.0122 / 0.0118 | 0.052 / 0.051 |
| (1,1,4096,640) 32 (SDXL) | bf16 | uniform / normal | 0.999996 / 0.999998 | 0.0028 / 0.0017 | 0.026 / 0.037 | 0.0021 / 0.0010 | 0.012 / 0.008 |
| (1,1,4096,640) 32 | fp32 | uniform / normal | 0.999998 / 1.000000 | 0.0020 / 0.0010 | 0.018 / 0.024 | 0.0013 / 0.0005 | 0.009 / 0.005 |
| (1,1,4096,640) 32 | bf8b | uniform / normal | 0.999918 / 0.999924 | 0.0130 / 0.0126 | 0.122 / 0.148 | 0.0115 / 0.0110 | 0.050 / 0.048 |
| (1,1,64,50) 1 (C non-aligned) | bf16 | uniform / normal | 0.999994 / 0.999999 | 0.0036 / 0.0018 | 0.018 / 0.020 | 0.0032 / 0.0008 | 0.013 / 0.008 |
| (1,1,64,50) 1 | fp32 | uniform / normal | 0.999997 / 1.000000 | 0.0025 / 0.0008 | 0.011 / 0.008 | 0.0019 / 0.0004 | 0.009 / 0.004 |
| (1,1,64,50) 1 | bf8b | uniform / normal | 0.999931 / 0.999937 | 0.0119 / 0.0113 | 0.081 / 0.064 | 0.0103 / 0.0095 | 0.043 / 0.039 |
| (1,1,50,128) 1 (HW non-aligned) | bf16 | uniform / normal | 0.999993 / 0.999999 | 0.0038 / 0.0017 | 0.033 / 0.018 | 0.0030 / 0.0010 | 0.018 / 0.008 |
| (1,1,50,128) 1 | fp32 | uniform / normal | 0.999996 / 1.000000 | 0.0028 / 0.0006 | 0.019 / 0.008 | 0.0022 / 0.0003 | 0.013 / 0.003 |
| (1,1,50,128) 1 | bf8b | uniform / normal | 0.999911 / 0.999921 | 0.0136 / 0.0129 | 0.077 / 0.096 | 0.0121 / 0.0118 | 0.053 / 0.051 |
| (1,1,64,200) 8 (both) | bf16 | uniform / normal | 0.999995 / 0.999999 | 0.0032 / 0.0017 | 0.026 / 0.024 | 0.0026 / 0.0010 | 0.013 / 0.008 |
| (1,1,64,200) 8 | fp32 | uniform / normal | 0.999998 / 1.000000 | 0.0023 / 0.0008 | 0.016 / 0.017 | 0.0017 / 0.0004 | 0.010 / 0.004 |
| (1,1,64,200) 8 | bf8b | uniform / normal | 0.999920 / 0.999924 | 0.0129 / 0.0127 | 0.086 / 0.109 | 0.0117 / 0.0112 | 0.050 / 0.050 |

Reading: bf16 and fp32 outputs are within ~1 output ulp of the reference (bf16: 0.03 max abs = 1 ulp near |y| ≈ 4;
fp32 rel RMS 0.0006–0.0028 is the tf32 operand rounding of `x` and of the statistics on the FPU — the documented
`exact_sfpu_stats` regime headroom, not needed for the 0.999 / 0.01 gate). bf8b carries its own input + output
quantization (7-bit mantissa shared over 16 lanes): rel RMS ≈ 0.013, PCC ≈ 0.99992, well inside the 0.99 / 0.10 gate.
The uniform distribution shows a slightly larger bf16 / fp32 rel RMS than the normal one because `y` has a smaller
norm relative to the per-element rounding of `x` (variance of U(0,1) is 1/12).

## `compute_kernel_config` surface — `test_groupnorm_sc_N_1_HW_C_precision_matrix_fidelity` (48 passed)

Shapes (1,1,64,320) G=32 and (1,1,64,200) G=8, normal input, `fp32_dest_acc_en=True` (mandatory),
`math_approx_mode` = `dst_full_sync_en` = exact/False or approx/True.

| fidelity | mode | bf16 PCC / rel RMS | fp32 PCC / rel RMS | bf8b PCC / rel RMS |
|---|---|---|---|---|
| HiFi4 | exact | 0.999998–0.999999 / 0.0017 | 1.000000 / 0.0008–0.0010 | 0.999923–0.999924 / 0.0127 |
| HiFi4 | approx (+ full sync) | 0.999998–0.999999 / 0.0018 | 1.000000 / 0.0007 | 0.999923–0.999924 / 0.0128 |
| HiFi3 | exact / approx | 0.999998–0.999999 / 0.0017–0.0018 | 1.000000 / 0.0008–0.0012 | 0.999923–0.999924 / 0.0127–0.0128 |
| HiFi2 | exact / approx | 0.999983–0.999991 / 0.0043–0.0070 | 0.999973–0.999989 / 0.0056–0.0096 | 0.999913–0.999920 / 0.0130–0.0132 |
| LoFi | exact / approx | 0.999787–0.999829 / 0.022–0.026 | 0.999740–0.999795 / 0.025–0.029 | 0.999786–0.999814 / 0.020–0.022 |

Reading: HiFi3 is indistinguishable from HiFi4 (the fp32 statistics operands are already tf32 on srcA/srcB). HiFi2
drops the low mantissa bits of the `(1/n)`-scaled statistics and of `x·scale` (fp32 rel RMS 0.001 → 0.009); LoFi
costs ~2.5 % rel RMS on every dtype — expected hardware behaviour, PCC still ≥ 0.9997. `math_approx_mode` only
touches the `rsqrt` on the `Ng` group tiles and is invisible at this precision. `fp32_dest_acc_en=False` and
`packer_l1_acc=True` are refused with `ValueError` (`test_compute_kernel_config_refusals`): the statistics
path accumulates into fp32 CBs (op_design.md "Never store stat CBs as bf16"; matmul_block fidelity rule #38306).

## TILE-layout affine weights — `test_affine_layout_matrix` (27 passed) + `test_affine_tile_gamma_only_fp32` (2)

Every geometry × weight dtype ∈ {bf16, fp32, bf8b} in TILE layout, normal input: interleaved TILE bf16, interleaved
RM fp32, RM C=47, RM model shard [128,40] direct view (`c_period = 40`, in_place), RM [128,80] fp32, RM staged
[103,64] (`hw_mask`), TILE bf8b [128,160] in_place, TILE fp32 ragged last column [32,384] on 11×2, RM N=2 [52,128]
straddle. bf16 / fp32 TILE weights reproduce the ROW_MAJOR-weight numbers exactly (PCC ≥ 0.999998, rel RMS
0.0007–0.0018); bf8b TILE weights (decoded in the reader) add the weights' own quantization: PCC 0.99996–0.99997,
rel RMS 0.0075–0.0084 on bf16 / fp32 activations, 0.0126 on a bf8b activation.

## Skipped combinations

- bf8b × ROW_MAJOR (activation or weight): block formats have no ROW_MAJOR representation (INVALID in
  `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py`).
- `fp32_dest_acc_en=False`: refused by the op (not a precision cell — see above).
