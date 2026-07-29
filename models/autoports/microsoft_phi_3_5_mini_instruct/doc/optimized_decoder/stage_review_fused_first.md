# Independent fused-checkout stage review

Verdict: **more-work-needed**

## Required work

The optimized decoder had dropped the fused predecessor's measured batch-1
`paged_fused_update_cache` topology and issued two cache writes without an
optimized-path A/B or rejection. The reviewer required an adapted whole-layer
correctness/performance comparison and a structural regression test.

## Resolution

The direct optimized-path port reproduced the fused kernel's non-overlap
assert because K and V shared one L1 core. The adapted implementation assigns
disjoint height-sharded K/V grids. It passes real-weight decode PCC 0.999790
and measures 493.677 us over 200 traced replays, versus 521.679 us for the
prior optimized default. The fused update is retained for batch 1; batch 32
and modulo-cache updates retain the separate-write contract. The optimized
test suite now structurally asserts the selected fused operation and guard.

The reviewer otherwise rederived the advisor gate, runtime BFP4/LoFi policy,
BFP8 cache, batch-1/batch-32 performance, PCC, context, and watcher evidence.
