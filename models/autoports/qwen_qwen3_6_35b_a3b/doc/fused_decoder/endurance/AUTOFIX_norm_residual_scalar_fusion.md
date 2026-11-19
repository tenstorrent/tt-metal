# Minimal exact-norm residual fusion

Source-only; no F edits or device execution. `norm_residual_rsub_fusion.patch` modifies one subexpression in the reciprocal correction. Base source SHA256 `0ef01cc509c617499722348c04cd04470e96fca1caa961a98a37174dffb8857f`. AST parsing and patch applicability are checked, not device performance.

## Proposed change

```python
# Before:
ttnn.subtract(ttnn.ones_like(shifted), ttnn.multiply(rh, ih))
# Candidate:
ttnn.multiply(rh, ih, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSUB, 1.0)])
```

The result is still `1 - (rh * ih)` with an FP32 product boundary. Remove the unnecessary constant-one tensor and standalone subtraction dispatch. The following three residual subtractions, reciprocal multiplication and inverse addition are unchanged. This is neither algebraic reassociation nor an FMA substitution. It preserves DRAM placement, so it does not reintroduce the rejected norm-L1 control.

## Source precision proof

1. Both `rh` and `ih` are positive, BF16-rounded values represented in FP32 in the existing norm correction. Their product near1 is exactly representable in FP32 (at most16 significand bits); no BF16 packing occurs. The shift includes positive epsilon, so this path has no arbitrary signed-zero or NaN input claim.
2. `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` calls `BINARY_SFPU_OP`, then `PROCESS_POST_ACTIVATIONS`, and finally packs the result. The same-shaped root/inverse split tensors take this nonbroadcast path. FP32 operand/output formats enable FP32 destination and direct FP32 unpack in `binary_ng_program_factory.cpp`.
3. Blackhole `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:221` computes the product and stores its FP32 result to destination. The following postactivation therefore reads the same FP32 product that the standalone subtract previously read after an external FP32 pack/unpack. It does not retain an unrounded hidden product or invoke the inaccurate MAD residual previously rejected.
4. `eltwise/unary/common/unary_op_utils.cpp:410` lowers `UnaryOpType.RSUB` with scalar1.0 to `rsub_tile` and the exact scalar bit pattern. `tt_metal/hw/inc/api/compute/eltwise_unary/rsub.h` dispatches `RSUB_UNARY`.
5. Blackhole `ckernel_sfpu_binop_with_unary.h:28` computes `parameter - val`; its additional BF16-RNE conversion is guarded by `!is_fp32_dest_acc_en`, so it does not run here. The old binary subtraction computes `in0 - in1` in `ckernel_sfpu_binary.h:128`. Both are the same FP32 subtraction of the same exactly representable1.0 and product.

Current source supports this operation, but installed native binaries can differ; root device execution remains the authority. Use the actual norm probe on sensitive rows first, requiring unchanged final BF16 norm codes and mean codes, then three-seed decoder correctness and paired warmed traced latency before keeping the candidate. Do not infer a speedup from removal of a dispatch. The current profile should quantify whether ones_like produces a separate kernel, a host allocation, or another trace action; the candidate removes its use regardless.

## Other scalar folds assessed, left for isolated controls

- `(high * 2) * low` can plausibly use a `MUL_UNARY_SFPU(2.0)` input-A activation on the final multiply. Scaling the BF16-high by2 is exact for this bounded norm range, and operand-A activation c3 retains FP32. This would preserve the two product sequence, but adds a preprocessing CB pass and should be measured independently.
- `reciprocal(root) * 0.5` can plausibly be a unary chain RECIP→MUL_UNARY_SFPU(0.5), or an equivalent binary multiply input activation. The reciprocal implementation and approximation-mode default must match the existing `ttnn.reciprocal` binding exactly. Do not substitute a different rsqrt or alter the two-rounding HF inverse target.
- Folding either correction addition into addcmul/MAD is rejected by earlier measured residual-precision evidence: the SFPU MAD does not retain the low product bits required by the split-product repair.

Only the first minimal residual change is included in this patch, to make any device failure attributable and avoid combining unmeasured tweaks. No state, precision-policy, cache-capacity or context-contract changes.
