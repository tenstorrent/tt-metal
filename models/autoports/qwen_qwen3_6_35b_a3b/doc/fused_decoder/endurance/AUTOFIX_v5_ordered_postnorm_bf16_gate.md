# Ordered postnorm plus canonical gated-norm dtype boundaries

Isolated candidate: `v5_ordered_postnorm_bf16_gate.py.snapshot`, SHA256 `264a46d3d6f60a12336201fcd63798afc6c5585474b1dd07d0788cc768ad8160`.
Base: `v5_ordered_postnorm.py.snapshot`, SHA256 `ff8227f036acd9ecdacfb83824cac07b1576e6b68594e459327b1f783316d83c`.
Adjacent patch/provenance records the precise scope. Source parsed successfully; only `_linear_attention` changes in the AST, and only its decode gated-norm tail changes. Setup, ordered postnorm, inputnorm, projection producers, recurrence and MoE remain unchanged. The entire prefill branch is byte-identical. No hardware, production or core-harness change was made.

The candidate replaces decode's fused sigmoid-gated norm plus external Z multiply with the previously used V4 boundary structure:

1. Cast recurrent core toBF16 and arrange32 contiguous head rows `[1,32,128]`.
2. Run **unweighted** `ttnn.rms_norm` with existing explicit compute config and epsilon, returningBF16. This is the HF unweighted-normalization rounding boundary.
3. Multiply by `self.w["linear_attn.norm.weight"]`, the exact FP32 representation of the actual learned BF16 gamma (no +1). Explicit `fast_and_approximate_mode=False` selects SFPU even though normalized input isBF16 and gamma isFP32. Output isFP32 with the existing BF16 round→FP32 widen activation chain. This restores the separate BF16 gamma-product boundary.
4. Explicitly widen Z toFP32, apply FP32 SiLU as the multiply input activation, multiply the rounded/widened gamma product, and returnBF16. Both operands areFP32; explicit accurate multiply is retained. This avoids the previous BF16 gate activation boundary trap.
5. Use the established V4 one-core head-concat adapter and explicit logicalH1/physicalH32 view before the unchanged output projection. All adapter costs remain in the actual method.

Source support: `normalization/rmsnorm/rmsnorm.cpp` forwards a nonempty tiled input without gamma to native layer_norm; `layernorm_device_operation.cpp:534–544` retains the input dtype when no output dtype override exists. Input isBF16, so unweighted output isBF16. `binary_ng_device_operation.cpp:51–65` selects SFPU multiply when the explicit accurate flag isfalse. `binary_ng_program_factory.cpp:1371–1400` uses direct-to-Dest unpack for the SFPU path. The existing round/widen chain is created in setup from TYPECAST(float32,bfloat16) followed by TYPECAST(bfloat16,float32). It materializes the required BF16-rounded value even though its stored carrier isFP32.

Compared with V4, gamma multiplication now explicitly selects SFPU for mixed input types; the V4 default flag could select an FPU route for mixedBF16/FP32 operands. The functional decoder's simpler `self.norm(...)*silu(z)` path does not by itself document both HF gated-norm boundaries, so it was not copied blindly.

This restores HF dtype boundaries and multiplication association. It does **not** claim native RMSNorm's reduction/rsqrt or native SiLU are bitwise identical to canonical HF on layer34. That distinction matters: CPU counterfactuals show canonical gated norm on actual TT core/Z improves the gated tensor and can restore137's route with canonical downstream arithmetic, but does not recover181. The upcoming unchanged-threshold diagnostic must decide whether this bounded repair helps with the actual TT downstream path. No accuracy waiver, host runtime fallback or quality/performance result is claimed.

The candidate is separate from the forthcoming precise-QZ projection snapshot. Parent may test it alone against its ordered-postnorm base and then compose with QZ after verifying the diff scopes. This snapshot does not itself alter Q/Z or merge another agent's changes.

## Parent-run long-stream outcome

The completed same-oracle layer34/seed1548192+33 diagnostics still fail unchanged quality thresholds: QZ alone282 unique failing positions; QZ+ordered postnorm283; ordered postnorm+BF16 gated boundaries281; all three combined268, minimum contribution PCC0.91271122765. Combined first failure remains181. This modest failure-count reduction is not a correctness pass; the severe floor remains. Each run uses one repeat, so its unconditional repeated-bit-equal field does not establish multi-repeat determinism. No production selection is claimed.

`v5_ordered_precision_layer34_comparison.json` contains exact artifact/provenance/source hashes, recomputed unique failure counts, first failures and minima. New immediate-source captures are the next discriminator.
