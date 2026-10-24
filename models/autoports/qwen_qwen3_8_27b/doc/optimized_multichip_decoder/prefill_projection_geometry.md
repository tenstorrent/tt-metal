# BF8 prefill projection geometry

All rows use the real TP4 4096-input/128-decode/B1 workload with25 warmed prefill replays. Times below are whole-layer **host trace latency**, not device time. Final-default device profiles are authoritative. Native prefill row fusion, hidden-sharded residual, BF8 columns/CCL, BFP4/LoFi weights, BF16 row inputs and unchanged decode are fixed.

Separate gate/up share one gather, gate fuses SiLU, and the independent BF16 product releases inputs before down. Packed decode and the one-row specialization are retained. The native reader requires one N block/core; padded interleaved endpoint copies have exact original quantized columns and checked zero tails. See AUTODEBUG_column.md and AUTOFIX_column.md.

|Geometry|Linear host us|Full host us|Result|
|---|---:|---:|---|
|control|6076.83|6382.81|PCC and exact replay pass|
|k20|6051.49|6343.77|PCC and exact replay pass|
|k32|6114.48|6523.35|PCC and exact replay pass|
|k40|6077.87|6406.39|PCC and exact replay pass|
|k20_sub8|6089.23|6381.72|PCC and exact replay pass|
|k20_grid11|5965.92|6244.54|PCC and exact replay pass|
|k80_padded|9322.95|9654.32|PCC and exact replay pass|
|k160_padded|9555.11|9869.75|PCC and exact replay pass|
|k16_l1|6113.51|6380.98|PCC and exact replay pass|
|k20_l1|6065.20|6392.67|PCC and exact replay pass|

Policies/commands and source hashes: prefill_projection_geometry{0,3}_{candidates,results}.json and each per-case JSON. control is10x8,K16,H8,sub4x2; k20_grid11 is11x8,K20,H8,sub8x1. Padded endpoints use10x8,H1,W7,sub1x7, physicalN4480. L1 rows place the one shared gather directly in L1 and configure both consumers for that same input. No immediate restore to a DRAM input is inserted.

Selection complete in prefill_projection_selection.json: native separate gate/up11x8,K20,H8,sub8x1 wins the forward/reverse comparisons for both kinds. Direct shared L1 and alternative mix configurations do not improve the paired means. Production v4 reproduces the selection in selected_default_receipt.json. No production source changes were made during the candidate sweeps.

Shared fused AG+gate/SiLU also passes at K4/K8 for both kinds: linear6523.74/6481.60us and full6818.61/6775.19us. Commands/source hashes: prefill_shared_gate{0,3}_results.json and per-case JSON. One BF8 AG output feeds both gate and up, with gate SiLU fused exactly once. These lose to native shared-gather separate projections (~5966/6245us); no watcher assertion is disabled. The previously diagnosed fused-AG watcher metadata issue is therefore not part of the selected path.
