# AutoFix: fused projection replica contract

Failing contract: rank pairs0/1 and2/3 must have exactly equal K/V caches because each pair owns a duplicated GQA head. The combined AGMM/MMRS candidate failed in two K scalar cells, with passing whole-output PCC; see AUTODEBUG.md, combined_fused_l3.pt and logs/combined_fused_l3.log.

Verified current source uses the fused Linear rank-relative K traversal identified in the diagnosis. This differs from canonical separate matmul order despite FP32 accumulation. Distributed norm stats are gathered in common rank order.

Prioritized repairs and discriminating checks:

1. Use canonical separate projection after normalized-input gather, while keeping hidden-sharded residual, distributed norm, batched projections and fused output MMRS. If paired KV equality passes on exactly the same natural input/state signature, the accumulation boundary is isolated.
2. A canonical-order fused kernel or single-owner/pair-broadcast KV could preserve fusion, but would be a larger algorithm change. Not needed to restore the contract; listed as future optimization, without an unmeasured claim.

The minimal repair is the existing explicit constructor policy `gather_projection="separate"`. No source kernel, tolerance or state comparison was modified. `autofix_fused_rs_l3.json` passes prefill PCC0.9999583845, minimum decode0.9998500053, all paired cache checks, deterministic changing-input replay and live-trace prefill. Traced B32 decode5.8921285ms is slower than replicated/batched directAR4.937193ms. Thus the repaired sharded fused-output family is valid but not selected; it does not immediately gather its residual back to the old interface. Its next norm consumes the hidden shard.

Exact hardware commands and source identities are in the JSON sidecars. The original combined candidate's exact rounding instruction remains unobserved; the source-supported order mismatch and matched repaired control establish a sufficient repair. The faster selected default already uses canonical separate projection, so it has no such failure.

Other adapted failures: the batched AR consumer now preserves L1 norm output; the list-return splitter explicitly handles the batched output layout. Initial and adapted attempts are in batch_ar_l3{,_v2,_v3} logs. K64 QKV geometry exceeded physical L1; both input8/output2 and adapted input16/output1 layouts were tried, with exact allocation failures retained. No correctness gate was weakened.
