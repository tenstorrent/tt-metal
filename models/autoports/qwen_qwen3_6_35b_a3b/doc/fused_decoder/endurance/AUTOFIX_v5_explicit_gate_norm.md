# Combined accurate-residual plus explicit gated normalization

Candidate `v5_combined_accurate_residual_explicit_gate_norm.py.snapshot`, SHA256 `de82dc83d2b6dddc94f346716ac767b09bbf653de58fe10910149402277c6d63`, is based on frozen accurate-residual combined snapshot `e3aea1c50aaf54b0ffa69949f8fdcc6bbe7232a64fac3a9fed8d3eaf052f2e6c`. It replaces only decode's unweighted native RMSNorm call with the explicit FP32 sequence from the successful same-operand probe: square, sum128, multiply1/128, epsilon add, rsqrt, normalize product, then BF16 cast.

Core BF16 input boundary, gamma SFPU round/widen boundary, FP32 SiLU multiplication, BF16 output, head concat adapter and output projection are unchanged. Ordered Q/Z, postnorm and accurate residual changes are inherited unchanged. Prefill arithmetic, recurrence, state and MoE are unchanged. Restoring the old one-call block restores the base byte-for-byte. Only `_linear_attention` differs in the method AST; provenance records both hashes. Source parsed without TTNN import.

The parent-run local probe confirms bit-exact unweighted normalized and gamma-weighted BF16 values for all five actual layer34 operands. Final gated output is exact at four positions and differs one code at128; no full quality or timing conclusion follows yet. Use the pending real decoder stream with unchanged thresholds. Candidate device execution is pending; no hardware or production/core-harness edits were performed by this preparation.

This is an arithmetic-local repair discriminator. It does not claim that native normalization alone causes the severe remaining MoE failures, nor that prefix state or projection differences are solved.

## Parent-run full-stream outcome

The actual layer34 seed1548192+33 diagnostic `v5_combined_accurate_residual_explicit_gate_norm_layer34_seed154.json` completed but fails219 positions under unchanged thresholds, with minimum contribution PCC0.9126377645963. Position181 still fails. This is not a qualified repair. Artifact SHA256 `008024dbc66aedb4e32045f5d06981d932a9e8ae0a9e2995cab2a943c1404e7f`, provenance SHA256 `d371e50b1fb01c4e466aa783f2937a829102759f1890d37fa3e47863dc1063c4`. A subsequent181 capture shows local gated and postnorm arithmetic exact on actual TT inputs; the separate output-projection and router diagnostics address remaining measured discrepancies. No claim that the original native norm issue was the sole cause is made.
