# Compact QKV grid ordering diagnosis

Source-only, 2026-11-12. No TT execution and no main-code edits. This corrects the initial compact-attention prototype's singleton-set grid assumption.

## Verdict

**Confirmed causal mismatch:** a set of singleton core ranges preserves x-major range ordering, while the native QKV split factory assigns users globally row-major. `ShardOrientation.ROW_MAJOR` orders cores **within each range**, not across the range list. This predicts exactly the observed two correct users (0 and31) on an8x4 grid.

## Source chain

1. `tt-umd/types/xy_pair.hpp:32-34` orders coordinates by x, then y. `tt_metal/api/tt-metalium/core_coord.hpp:104-108` orders CoreRanges first by start coordinate, then end coordinate. Thus `std::set<CoreRange>` singleton grids are x-major.
2. Python `CoreRangeSet` provides separate vector and set constructors (`ttnn/cpp/ttnn-nanobind/tensor.cpp:616-626`). The set constructor preserves set iteration order; it does not coalesce rectangles (`tt_metal/common/core_coord.cpp:156-165`).
3. `nlp_create_qkv_heads_decode_device_operation.cpp:140-146` derives the output Q grid with `num_cores_to_corerangeset_in_subcoregrids`. That function traverses `sub_core_grids.ranges()` in stored order (`work_split.cpp:230-254`) and consumes each range row-wise. It merges consecutive rectangles only when their x spans match (`work_split.cpp:262-308`). Starting from singleton x-major ranges therefore produces x-column rectangles; it does not restore global row order.
4. Interleaved native QKV factory lines263-278 instead discards range ordering: it gets the bounding-box dimensions, invokes `grid_to_cores(...,true)`, and assigns input batch i to coordinate `(i % grid_width, i / grid_width)`. This explicitly reads user i on global row-major core i. The sharded native factory has the same `grid_to_cores` assignment for Q/V at lines125-128, explaining the same failure under the different input layouts.
5. Generic tensor page mapping (`tt_metal/impl/buffers/buffer.cpp:438-440`) invokes `corerange_to_cores(grid,...,row_major)`. That helper loops stored ranges, then enumerates **each** range row-major (`core_coord.cpp:604-623`). The sharded-to-interleaved operation independently uses the same enumeration (`sharded_to_interleaved_program_factory.cpp:87`). Therefore the native-written physical core contents are read back with the wrong logical-user association.

This is an ordering defect, not matmul precision, projection packing, cos/sin, or an interleaved-vs-sharded numerical issue.

## Predicted permutation and artifact agreement

For the8x4 B32 singleton set, logical output user i enumerates physical core `(floor(i/4), i mod4)`. That core holds native-written source user:

`source(i) = 8 * (i mod4) + floor(i/4)`.

Predicted output-to-source map:

`[0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31]`.

The only fixed points are0 and31. `compact_probe.json` reports Q PCC0.057522256, K0.072216577, V0.059778111 for DRAM input, with users0 and31 exact and other same-index comparisons near zero; L1 results match. This strongly corroborates the exact source permutation. A raw tensor cross-user comparison is still the best device confirmation; the current JSON contains same-user PCC only. The reported full layer PCC~0.296 and cache PCC~0.972 are downstream consequences consistent with wrong-user QKV; they are not acceptable correctness evidence.

## Decoder-local repair and scope

For offset-zero splitter output, construct a full rectangle of complete rows plus an optional final row tail. At B32 use the single range `(0,0)..(7,3)`. At B13 use `(0,0)..(7,0)` followed by `(0,1)..(4,1)`. Native row-major assignment and tensor range enumeration then agree. This is the repair currently being measured by the stage owner.

Prefer `CoreRangeSet(ranges)` with an explicitly ordered **list**, not `set(ranges)`, in the general helper. The current rectangle-plus-tail offset-zero form happens to sort correctly because both ranges start at x=0. A nonzero offset with an initial partial row does not: e.g. offset13 starts at x5, followed by later full rows at x0; a set sorts the later full row first. For K cache input this may remain internally consistent because both generic reshard and fused update enumerate `corerange_to_cores`, but it violates the helper's advertised row-order intent and makes later consumers hazardous. Preserve the intended list order explicitly.

Fused cache uses `corerange_to_cores` for each K/V grid (`paged_tiled_fused_update_cache_program_factory.cpp:115-118`) and assigns `my_batch_idx=i` in that ordering. Native concat's subcoregrid factory also uses `corerange_to_cores` for input users. Correcting the initial native split mapping therefore addresses the origin; no downstream inverse permutation should be introduced. Rectangular rotary grids are already single ranges and unaffected.

The initial `candidates/compact_attention_decode.py` remains an unselected historical prototype containing the faulty singleton-set helper. Production main code is being fixed by the stage owner; this report does not claim that helper is safe.

## Focused verification

Rerun the split probe with the repaired grid and require exact equality for all users and Q/K/V. Then rerun the integrated full-attention decoder, distinct positions, and cache ownership checks. Include B13/B26 or a similarly awkward partial-row case as well as B32; prefix-grid kernels assume origin0 and row-major coverage, so do not generalize native splitter output to an offset grid without inspecting its runtime assignment. No library changes are required for the decoder-local fix.
