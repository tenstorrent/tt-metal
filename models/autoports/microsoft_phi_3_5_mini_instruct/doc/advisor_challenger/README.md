# Phi-3.5 Mini Advisor Contribution at Decode Batch 32

## Full-model estimate: 32,515.68 us before, 32,515.68 us after, +/-34.94 us

The measured `$shard-advise` contribution is **zero**. No advisor change shipped. Phi has one dense layer kind and 32 hidden layers; `layer_counts = {"dense": 32}` is recorded in the frozen control. The headline is the reconciliation window (`1,016.115 us/layer`) scaled by 32, and its conservative band is the incumbent repeat spread (`1.092 us/layer`) scaled by 32.

The supplied decoder was batch-1-only. Before freezing the control, its existing layouts were mechanically generalized to a legal batch-32 Q/K/V grid. This prerequisite is not credited to the advisor. All `decode_batch`, `requested_decode_batch`, and `capture_batch` values are 32.

The incumbent was frozen at `2026-07-31T17:43:05+00:00`, before the capture at `2026-07-31T17:45:20+00:00`. The mandated harness produced five means of 50 traced replays after 10 untimed warmups: `[1.100939, 1.101220, 1.100495, 1.101046, 1.100128] ms`; median `1.100939 ms`, spread `1.092 us`.

The pinned advisor (`618cd4e75d`) captured the shipped BF8 attention, BF4 MLP, and BF16 norm policy at batch 32. It emitted 39 ops, 36 choices, one spill, and the known `nlp_concat_heads_decode` unfixable condition. `reconcile.py --incumbent` closed 100% without `DEGRADED`: its measurable ceiling was `83.551 us/layer` (`2,673.632 us/model`).

The ranking was dominated by the Phi split-half RoPE chains. The full L1 family measured median `1.101395 ms` and was slower. The legal L1-tail isolate measured median `1.100683 ms`, but its `[1.099390, 1.101032] ms` range overlaps the incumbent `[1.100128, 1.101220] ms` range, so it fails the rule that every candidate repeat beat every incumbent repeat. Direct sharded SDPA output was also measured as a doubtful direction and failed with `TT_FATAL: Sharded output not supported for GQA`. The remaining `0.874 us` boundary was below the incumbent floor. Every losing knob remains default-off in `optimized_decoder.py`.

The unresolved `CopyDeviceOperation` costs `2.678 us/layer`. IR lines `%57 -> %58 -> %60` show an L1 width-sharded `1x86` multiply output converted to the shipped/advised `1x8` down-projection input, so it is a real in-chain regrid that the advisor also retains; it is not booked as advisor contribution. `advised_boundaries.us_advisor_agrees` is `0.000 us` as parsed, while this separately resolved copy contributes `2.678 us` of additional agreement.

Reported but not screened: `model_estimate.layer_handoff` is `1.091 us/layer`, or `33.821 us/model`, because the profiled layer enters from DRAM and exits in L1. This is outside advisor attribution. The parsed `advised_boundaries.us_advisor_agrees` is also reported and not screened.

The shipped batch-32 decoder passes a real-weight HF-vs-TTNN oracle using layer-0 weights from the pinned Phi snapshot: decode PCC `0.999983418` at a `0.995` bar across 32 users. The evidence is recorded in `oracle_real_batch32.json` with `oracle_weights: real`.
