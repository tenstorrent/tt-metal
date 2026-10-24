# Native fused decode RS geometry closure

The existing full-width native reader requires output blockW=per_core_N; it does not require N20. The grid screen compares8x1/N20/sub1x5,10x1/N16/sub1x4 or1x8,11x1/N15/sub1x5 under identical BF16A/BFP4W/LoFi/BF16out, persistent out/down buffers, sharded inter-layer residual and CCL offset(0,6). Physical M32 makes perM1; extra Y cores cannot supply useful rows. Candidate source and per-run hashes are retained. Subsequent role-isolated K comparisons use shared packed RoPE on the full-attention path so the comparison retains the best correct sibling optimization.

Capacity endpoints derive from `ccl_fusion_mcast_2d.cpp` input/output CB formulas (lines119–189), with double-buffer depth2 from `matmul_utilities.hpp:23`. Native decode reshapes to B1; down totalK136. K68 uses two blocks/depth2; K136 uses one block/depth1. Both have identical minimum A+B+BF16output allocation:

`278528 + 136*N*576 + N*2048` bytes.

|Native N|Minimum bytes K68 orK136|K34 minimum bytes|
|---|---:|---:|
|20|1,886,208|963,584|
|16|1,564,672|798,720|
|15|1,484,288|757,504|

Blackhole physical L1 is1,572,864bytes; allocatable L1/bank is1,461,120bytes. Even N15 exceeds allocatable capacity before persistent buffers; current selected setup reserves278,912bytes/bank, leaving1,182,208bytes before remaining live allocations/CBs. This is a native-decode-specific source/capacity bound, not reuse of a prefill collision address. K34 is feasible and measured rather than rejected by the endpoint bound. The payload has576bytes/BFP4 tile (including exponent headers), BF16 tiles2048bytes. The bound already excludes additional runtime/CCL requirements, so these cannot make an impossible endpoint fit.

Selection is complete:10x1/N16/sub1x4,outK6linear/K24full,downK17. See decode_rs_grid_selection.json. K24 full-attention repeated994.99/994.24us versus997.75/999.61us on the shared packedRoPE family; final production batch32_l3 reproduces989.93us with selected prefill packing also enabled. Final device/runtime profiles pass:22-core nativeRS per projection,10x1/N16/sub1x4 and selected roleK. Batched timings are reported separately in README and batched_runtime_contract.json. `decode_rs_grid_b32_results.json` records the initial screen. Small grid-only gains must be judged with paired whole-layer measurements, not per-op throughput estimates.
