# Front A-routing-num-slices (design/recon swarm, 2026-08-17)

## Verdict

Routing-layer num_slices is implementable as a ~10-line change inside run_topk_large_indices_route (topk.cpp:401-440) by calling the already-linkable compute_column_split_config(allow_multi_row=true) — no predicate duplication needed — but the headline 2.8-2.9x is an OP-STAGE model only. Passing num_slices on the rows=32/k=32 shapes forces tile_output off (multi-rect + tile_output is a hard TT_FATAL, factory:915-919), re-adding the 2x single-core tilize (+typecast) tail the tile-output opt-in deleted (measured 20-78 us per tilize at <=32 output tiles). Net call-level expectation: 32x65536 ~193.6->~118 us (1.6x, save ~75), 32x64128 ~187.9->~118 us (1.6x, save ~70/call), IF the tilize pair for a [32,32] output lands at the ~20 us floor — worst case (78 us each) nets NEGATIVE. Also: 32x32768 k=32 does NOT route today (pow2 <65535 stays on the stock bitonic, 171.3 us; GATE_CELLS pins this) — it is out of scope for this change. GO for the k_rounded=2048 multi-row arm (RM tail already kept — pure op-stage win, cf. measured 3.6x at 30x65536 explicit P=4); CONDITIONAL-GO for the k_rounded<=1024 scenario shapes, gated on one measurement: the to_layout(TILE)+typecast tail cost for [rows<=32, k_rounded<=1024] outputs.

## Plan

## (1) Where the route calls the op, and with what

Route decision: ttnn::topk calls should_route_to_topk_large_indices at topk.cpp:602-610, then run_topk_large_indices_route(transformed_tensor, k_rounded) at topk.cpp:614-615, then the shared post_topk_transform_tensor at topk.cpp:616-617.

The op call is topk.cpp:419-425: `ttnn::experimental::topk_large_indices_with_values(op_input, k_rounded, /*valid_length=*/nullopt, /*num_slices=*/nullopt, /*tile_output=*/tile_native_output, index_dtype)` where:
- op_input = untilized RM tensor for multi-row (tile_native_input requires flattened_rows==1 && padded_width>=32768, topk.cpp:406-408) — so the rows=32 shapes always feed RM input;
- tile_native_output = k_rounded <= 1024 (large_k_route_tile_output_max_k, topk.cpp:370, 413) — TRUE for k=32 (k_rounded=32);
- index_dtype = UINT16 iff tile_native_output && padded_width<=65535, else nullopt with a to_layout(TILE)+typecast tail at topk.cpp:427-433.

Inside the op wrapper (device_operation.cpp:332-364), hybrid_row_split returns nullopt for these shapes (rows=32 <= cores=120/130, device_operation.cpp:311-312) AND whenever num_slices.has_value() (device_operation.cpp:294-296) — so a routing-passed num_slices produces exactly one single launch. The device op's AUTO model never picks multi-row rects (column_split_config_for passes allow_multi_row=false, device_operation.cpp:157-158), so today these shapes run row-parallel.

## (2) How routing should compute num_slices — direct call, no duplication

compute_column_split_config is declared in device/topk_large_indices_program_factory.hpp:65-71, which topk.cpp ALREADY transitively includes (topk.cpp:20 -> topk_large_indices.hpp:7 -> device_operation.hpp:13 -> program_factory.hpp). Linkage: it is defined in topk_large_indices_program_factory.cpp, compiled into ttnn_op_experimental_topk_large_indices whose objects are spliced into the final link via $<TARGET_OBJECTS:TTNN::Ops::Experimental::TopkLargeIndices> (experimental/CMakeLists.txt:93,143) — the same mechanism that already resolves topk.cpp's call to topk_large_indices_with_values. So: call it directly; do NOT duplicate the predicate.

Sketch (inside run_topk_large_indices_route, after line 405):
```cpp
const auto grid = transformed_tensor.device()->compute_with_storage_grid_size();
const auto rect_cfg = operations::experimental::topk_large_indices::program::compute_column_split_config(
    k_rounded, lshape[-1], flattened_rows, grid, std::nullopt, /*allow_multi_row=*/true);
const bool use_rects = flattened_rows > 1 && rect_cfg.enabled /* && measured net-win gate, see (3) */;
const bool tile_native_output = !use_rects && k_rounded <= large_k_route_tile_output_max_k;
// pass use_rects ? std::optional<uint32_t>(rect_cfg.num_slices) : std::nullopt as num_slices
```
The existing !tile_native_output tail (topk.cpp:427-433) already handles RM outputs (to_layout TILE x2 + typecast when emit_u16), so no new tail code.

Why pass cfg.num_slices explicitly rather than change the op: the device op deliberately never auto-selects multi-row rects — engine choice must not depend on layout opt-ins and non-stable tie order differs across engines (device_operation.cpp:151-158; RESULTS.md "Design decisions"). The override path in compute_column_split_config (factory:576-671) force-enables the rect form regardless of allow_multi_row, re-derives the rectangle shape (largest p<=requested fit, preferring tiling capacity, factory:628-643), and TT_FATALs on P outside [2,128] (factory:600-604) or P > chunk count (factory:607-614) — the model-derived P always satisfies both.

Model check for the scenario shapes (512-elem window since k_rounded=32 <= 512, factory:40-47; chunks = ceil(n/512), factory:497-501): rows=32 requires rect-tiling capacity >= 32 (factory:528-530). On both 13x10 (p150a) and 12x10 (Galaxy chip) the max capacity at P=4 is 30 < 32, so the model picks P=3 (3x1 or 1x3, capacity 36-40). Multi-row acceptance margin (factory:555): cost(3)+max(2,cost_row/8) <= cost_row holds comfortably at chunks=128/126/64.

## (3) Interactions — where num_slices is correct, redundant, or WRONG

HARD CONFLICT — tile_output: multi-rect + tile_output is a TT_FATAL ("drop tile_output or num_slices", factory:915-919; reason: the tile-scatter writer writes whole 32-row tiles and rect row ranges are not tile-aligned, factory:913-914). All k<=64 routed shapes have k_rounded<=64 <= 1024, so today they take tile_output=true + native-u16. Passing num_slices forces the RM tail back: +2 single-core TilizeWithValPadding launches (measured 20-78 us each for <=32-tile outputs, topk.cpp:363-369; i2-tile-native-report.md stage table: tilize_v 20.3 / tilize_i 39.3 at k=512) + typecast ~2.1 us (only when padded<=65535). This is the decisive cost against the 2.9x op-stage model — see expected wins below.

Arm-by-arm:
- Small-k WIDE arm (k<=64, padded>=4096, bitonic-structurally-ineligible; topk.cpp:316-341): NO row cap — the rows=32 sampling shapes 32x65536 and 32x64128 live here and benefit (conditional on the tail). 32x32768 does NOT enter this arm (32768 is pow2 and <65535 -> multicore_structurally_ineligible=false, topk.cpp:324-326); it stays on the stock bitonic (topk_device_operation.cpp:66-75; measured 171.3 us). Routing it would be a predicate widening with GATE_CELLS churn ("pow2-multi", FULL "w32768-multi" pin pow2 cells to multi_core) — separate change, not this one.
- MoE-GATE arm (k<=16, padded 128-512, <=32 rows): chunks = ceil(W/512) = 1 < 2 -> compute_model_column_split_config returns disabled (factory:501-504) -> cfg.enabled=false -> num_slices never passed. Automatically safe.
- Large-k arm (64<k<=2048): k_rounded in (1024, 2048] (i.e. k>1024) keeps the RM+tilize tail ANYWAY (topk.cpp:413, 427-433) -> passing num_slices there is a PURE op-stage win with zero tail regression (measured precedent: 30x65536 k=2048 explicit num_slices=4 = 98.8 us vs 356.5 RP = 3.6x, RESULTS.md:14). k in (64,1024] has the same tile_output conflict as the small-k arm. Recommend shipping the k_rounded==2048 arm unconditionally and the tile-output-displacing arm behind the measured gate.
- rows > rect capacity (e.g. GLM 160/185-row shapes): the model disables (no candidate passes factory:528-530) -> num_slices not passed -> hybrid_row_split still fires as today. Do NOT force num_slices there: measured 160-row explicit P=2 = 542.4 us vs hybrid 467.0 us (RESULTS.md:14-15). The cfg.enabled gate handles this with no extra code.
- Single row: keep nullopt — the model auto-selects the classic tree already (column_split_config_for enables single-row rects without override); passing it is redundant and would needlessly disable tile_output via the use_rects flag if not special-cased. Gate on flattened_rows > 1.
- Disqualifiers stable/indices_tensor/preallocated/sub_core_grids/largest=false: already rejected BEFORE the route (topk.cpp:299-312) — unchanged; sub-grid-pinned sampling callsites never reach this code (glm-callsite-map.md row 6).
- rows=64 portability note: on 13x10 (p150a) P=2 (1x2, capacity 65) enables; on Galaxy 12x10 capacity maxes at 60 < 64 -> disabled. Same shape, different engine per SKU — legal (grid is hashed) but means tie order differs across SKUs.

Program-cache: num_slices is hashed directly PLUS the derived split fields (device_operation.cpp:193, 200-206), and the nightly suite pins distinct-entry/cache-hit behavior (test_topk_large_indices_num_slices_program_cache_distinct_entries, nightly test:954-978). Width is NOT hashed — 32x65536 and 32x64128 (both P=3, 3x1, num_rects=40) share ONE cached program with chunks patched via runtime args (types.hpp comments; factory:661-669). P<=chunks is re-validated on every invocation because compute_column_split_config runs in both the hash path and create (device_operation.cpp:144-159, factory:899-900).

## (4) Tie-order story — confirmed safe

- Engines break bf16 ties differently and the device op comment + RESULTS.md design note say exactly this (device_operation.cpp:151-158; RESULTS.md:18-23). Routed-vs-stock divergence was already accepted (I5, topk.cpp routing predicate rejects stable=true at :302-304).
- No routed-internal test asserts row-parallel-vs-rect equality on tie-capable input. The only rect==RP equality assert is OP-level: test_topk_large_indices_num_slices_multirow_matches_row_parallel (nightly test:927-937), which uses _make_bf16_exact_input that "rules out [ties] by construction" — unaffected.
- The contract battery is tie-order-agnostic by design: verify_topk_cell asserts the value MULTISET (test_topk_contract.py:579-585), the sorted value BIT-SEQUENCE (unique under the total order, :587-594 — equal values have equal bits, so tie choice is invisible), gather validity input[idx]==value (:623-627), index uniqueness (:660-666), and the routed sentinel pairing (:644-658). test_contract_determinism asserts run-to-run identity only (:1015-1044). test_topk.py (reduce) asserts values via assert_equal + indices via gather/cosine only (:82-96). The sweep suite uses comp_topk_similarity on gathered values (sweeps/reduction/topk/topk.py:192-197). All survive an internal RP->rect engine flip; cross-VERSION indices on tie-heavy inputs will change (same class as I5).

## (5) Gating tests

- tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py — the routed GATE_CELLS whose execution flips to the rect chain (all built at rows=32 via gaussian_input default, :441-443): k65-routed (8192,65), k2048-routed (8192,2048 — note: model barely accepts P=3 at chunks=4; margin 6+2=8 is NOT > 8, factory:555), nonpow2-routed-smallk (8224,32), smallk-floor-over-nonpow2 (4128,32), w65504-u16-routed, w65534-u32-padded-routed, w65536-routed-smallk, w65536-k8-routed, k97-routed-round112; FULL: w524288-routed-ceiling, w131072-routed, w100000-routed-nonpow2, k95-routed-round96 (GATE_CELLS list :1052-1110, test :1113-1122). Gate-arm cells (gate-gptoss-w128-k4 etc.) stay row-parallel (chunks=1). Pinning AGAINST scope creep: pow2-multi (8192,32) and FULL w32768-multi must stay multi_core.
- Same file: test_contract_determinism (DET_CELLS routed-10000-k32, routed-8192-k96, :1011-1044); test_contract_nan_bf16/fp32, zeros, subnormal, ties_boundary on ENGINE_CELLS routed-10000-k32 + ROUTED_CELL 8192-k96 (:708-719); test_contract_ties_all_equal_row (:942-949); test_contract_infleak_routed_sentinel (routed-u16 8192/96, routed-u32 65536/96, :988-1004); test_contract_gates_sorted_flag_noop routed cell (:1136-1164); test_contract_gates_gate_arm_rows_cap (:1125-1133).
- tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py — num_slices section (:883-978 incl. override-correctness, non-model values, multirow_matches_row_parallel, out-of-range rejection, program-cache entries), tree tests parametrized on num_slices (:983-1120 incl. random ties, valid_length, return_values), the flex/matches_default bit-identity section, and the composite hybrid tests. (Known env-gated failures: 2 production_perf_check IOMMU cells, pre-existing.)
- tests/ttnn/unit_tests/operations/reduce/test_topk.py — routed multi-row cells (1,1,32,{10000,18992,64128},k=32) (:112-135).
- tests/sweep_framework/sweeps/reduction/topk/topk.py "large_k" suite (:65-81) — 32/64-row x 8192..65536 x k in {96..2048} vectors exercise the new arm in CI; 64-row vectors are the grid-portability cells.

## Expected wins (rect model: 512-window, chunks=128/126/64; P=3 on both 13x10 and 12x10)

Op-stage model (units: RP = 2*chunks, rect = 2*ceil(chunks/3)+2; factory:506-532):
- 32x65536: 256 -> 88 = 2.91x
- 32x64128: 252 -> 86 = 2.93x
- 32x32768: 128 -> 46 = 2.78x

Net at the routed-call level (today's measured totals: 193.6 / 187.9 us, implementation.md:65-67; untilize ~18 us at 4 MB read, i2 report):
- 32x65536 (u32): 18 + 175.6*(88/256)=60.4 + tilize pair (~40 floor, up to ~156) = ~118-234 us vs 193.6 -> best case 1.6x (save ~75 us), worst case NEGATIVE.
- 32x64128 (u16 via typecast): 17.6 + 170.3*(86/252)=58.1 + ~40-156 + 2.1 = ~118-234 us vs 187.9 -> best case 1.6x (save ~70 us/call, x2 calls/token for the llama split path).
- 32x32768: NOT routed (stock bitonic 171.3 us). Hypothetically, with a predicate widening: ~9 + 111.7*(46/128)=40.1 + ~40 = ~89 us -> ~1.9x vs bitonic — but that is a separate, GATE_CELLS-breaking change requiring new measurements.

The one measurement that decides the k_rounded<=1024 arm: Tracy stage profile of the to_layout(TILE) x2 (+typecast) tail on [1,1,32,32] outputs. If it lands near the 20 us/tilize floor, ship; if near 78 us, the arm loses and only the k_rounded==2048 arm should ship.

## Evidence

- ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:401-440 — run_topk_large_indices_route; op call with num_slices=nullopt at :419-425; RM tail at :427-433
- ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:602-618 — route decision and call site in ttnn::topk
- ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:299-341 — disqualifiers (stable/indices_tensor/prealloc/sub_core_grids) and wide/gate arms; :324-326 pow2 exclusion that keeps 32x32768 on stock
- ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:363-380 — tile_output_max_k=1024 policy + 20-78us single-core tilize measurement note
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.hpp:65-71 — compute_column_split_config public declaration (in installed API header set)
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:40-47 — snap_to_llk_target_k: k<=512 -> 512 window
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:473-572 — cost model: cost(P)=2*ceil(chunks/P)+ceil(log2 P) vs 2*chunks; capacity>=rows constraint :528-530; 12.5% multi-row margin :555
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:576-671 — num_slices override: force-enable, [2,128] and P<=chunks TT_FATALs, capacity-preferring rectangle fit
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:899-919 — multicore create: allow_multi_row=!tile_output and the num_rects>1 + tile_output TT_FATAL
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:144-159 — column_split_config_for(allow_multi_row=false): auto model never picks multi-row rects; tie-order rationale comment
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:171-207 — compute_program_hash includes num_slices and all derived split fields + grid
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:293-328 — hybrid_row_split: bails on num_slices.has_value(), tile_output, rows<=cores
- ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt:93,143 — TopkLargeIndices objects spliced into the umbrella link (proves compute_column_split_config is linkable from topk.cpp)
- tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py:1052-1122 — GATE_CELLS + test_contract_gates
- tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py:579-666 — tie-order-agnostic contract assertions (multiset, bit-sequence, gather, uniqueness, sentinel)
- tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py:1011-1044 — determinism cells (run-to-run only)
- tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py:927-937 — the ONLY rect==row-parallel equality assert, on tie-free-by-construction input
- tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py:954-978 — num_slices program-cache distinct-entries pin
- tests/sweep_framework/sweeps/reduction/topk/topk.py:65-81,192-197 — large_k suite shapes and tie-tolerant comp_topk_similarity
- tests/ttnn/unit_tests/operations/reduce/test_topk.py:74-96,112-135 — values assert_equal + gather/cosine indices; routed multi-row cells 32x{10000,18992,64128}
- paper-topk/evidence/scenarios/implementation.md:65-67 — measured routed totals: 32x65536=193.6us, 32x64128=187.9us, 32x32768 stock bitonic=171.3us / routed-direct 120.7us
- paper-topk/evidence/i2-i3-i4-landings/i2-tile-native-report.md:43-61 — stage profile: untilize ~18us at W=65536, tilize_v 20.3 / tilize_i 39.3 at k=512, typecast 2.1
- paper-topk/evidence/glm-hybrid-composite/RESULTS.md:14-15,18-32 — 30x65536 k=2048 P=4 = 98.8us (3.6x); 160-row explicit P=2 (542.4us) loses to hybrid (467.0us); design rationale

## Risks

- Tail cost is the go/no-go unknown: the 20-78us/tilize band was measured on [1,k_rounded] outputs; the scenario outputs are [32,32] (1 tile). If each tilize lands high in the band, the num_slices arm nets NEGATIVE on all three shapes despite the 2.9x op-stage model. One Tracy stage measurement (owner: implementer, before enabling the k_rounded<=1024 arm) resolves it.
- 32x32768 k=32 does not route today (pow2, <65535 -> stock bitonic); the task premise that all three shapes route is wrong for this one. Widening the predicate would break GATE_CELLS pow2-multi / w32768-multi pins and needs a measured case vs the 171.3us bitonic — treat as a separate change.
- Passing num_slices disables hybrid_row_split for that call (device_operation.cpp:294-296); gating on cfg.enabled (which is false when rows > rect capacity) is load-bearing — an unconditional pass would regress rows>grid shapes (542.4us vs 467.0us measured).
- k2048-routed GATE_CELL (8192, k=2048, rows=32): the model accepts P=3 with zero slack (margin 6+2=8 vs cost_row=8, factory:555 uses strict >). Any margin-constant tweak or off-by-one flips this cell between engines; it is the most fragile cell in the suite for this change.
- Grid-dependent engine choice: rows=64 shapes get rects on 13x10 (capacity 65) but not on 12x10 Galaxy (capacity 60) — same call, different tie winners across SKUs. Legal (grid is hashed) but worth a line in the change description; the sweep large_k 64-row vectors cover it.
- Cross-version index churn on tie-heavy inputs for every routed multi-row cell that flips RP->rect (same acceptance class as I5); no test asserts against it, but downstream consumers doing exact-index comparisons across software versions will see diffs.
- Rect-path special-value coverage (NaN/Inf canonicalization) under the contract battery has not previously run through the multi-rect engine at these cells; nightly tree tests cover ties/valid_length/values but the NaN cells will exercise rect+RM-tail for the first time on hardware.
