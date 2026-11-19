# Combined QZ, postnorm and gated-boundary diagnostic

Candidate `v5_ordered_qz_postnorm_bf16_gate.py.snapshot`, SHA256 `16fdf3f632f685af1b8e279f933fb28137308d8dc1f3965dafbb50c797401870`. This combines the frozen ordered QZ+postnorm candidate (SHA256 `2c66df4dbb29378764175edf6bb7591e96807f4d806c1de2207538cdbcc56fc2`) with the exact decode gated-norm tail from `v5_ordered_postnorm_bf16_gate.py.snapshot` (SHA256 `264a46d3d6f60a12336201fcd63798afc6c5585474b1dd07d0788cc768ad8160`). All donor files remain unchanged. Coordination confirmed the QZ+postnorm donor is frozen.

Source proof: the entire module AST equals the selected v5 module with exactly four donor methods substituted: `from_state_dict` and `linear` from ordered QZ, `norm` from ordered postnorm, and `_linear_attention` from the BF16 gated-boundary donor. Every other method and module-level node is identical. Replacing only the gated method with its original restores the complete QZ+postnorm module AST. The patch contains only that method change relative to QZ+postnorm; provenance binds all five source files. Syntax parsing passed without TTNN import.

QZ and postnorm changes are unchanged from their owners' candidates. Decode gated norm restores BF16 core/unweighted norm/gamma-product boundaries, followed by FP32 SiLU gating and final BF16 output, with explicit accurate SFPU gamma multiplication. Its head concat adapter and actual costs are retained. Prefill `_linear_attention` arithmetic remains unchanged, while QZ projection follows its donor's precise singleton/serial path in both modes.

This is an unselected diagnostic to test whether the combined changes suffice on actual layer34. Parent reports QZ alone still fails282 rows; that motivates composition, not a claim of recovery. No device execution, production/native/core-test change, accuracy waiver or runtime host fallback occurred here. Native RMSNorm/SiLU differences and upstream recurrence error may remain. Use unchanged complete-stream quality gates and compare actual captured IDs/intermediates before attributing any recovery.

## Parent-run long-stream outcome

The completed same-oracle layer34/seed1548192+33 diagnostics still fail unchanged quality thresholds: QZ alone282 unique failing positions; QZ+ordered postnorm283; ordered postnorm+BF16 gated boundaries281; all three combined268, minimum contribution PCC0.91271122765. Combined first failure remains181. This modest failure-count reduction is not a correctness pass; the severe floor remains. Each run uses one repeat, so its unconditional repeated-bit-equal field does not establish multi-repeat determinism. No production selection is claimed.

`v5_ordered_precision_layer34_comparison.json` contains exact artifact/provenance/source hashes, recomputed unique failure counts, first failures and minima. New immediate-source captures are the next discriminator.
