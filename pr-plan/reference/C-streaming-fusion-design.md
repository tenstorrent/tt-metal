# Front C-streaming-fusion-design (design/recon swarm, 2026-08-17)

## Verdict

Front C is buildable as four independently-landable increments on the existing multi-rect tree factory, in strict dependency order: (1) a `core_grid` CoreRangeSet placement param (~200 LOC, 2-3 days) that offsets the rect-tiling and row-parallel work-split into a caller-chosen rectangle; (2) HEIGHT_SHARDED-L1 input support (~300 LOC, 4-5 days) where the shard spec BECOMES the placement (the factory's self-placement inverts: cores are given, the factory derives num_slices/rect layout from the shard grid); (3) the streaming handshake via a global-semaphore credit counter + concurrent sub-device programs (~300 LOC total, 4-6 days) — achievable today WITHOUT FusedProgram, with the dram_prefetcher GlobalCircularBuffer/SubDevice stack as the proven in-tree precedent and blaze's stream_sem design as the exact semantics to copy; (4) indexer_score output_memory_config + sharded/streaming writer (~400 LOC, 5-8 days). Increments 1-2 alone close blaze requirement (1) and the resident-input half of (2); 1+2+3+4 bound GLM prefill per-chunk indexer cost at max(t_score, t_topk)+one-chunk drain instead of t_score+467us, and replicate the blaze decode topology (32 topk cores below 8 SDPA banks, 2048-position credits) minus the bank/device bit-stamp and the single-program co-residency, both of which have cheap approximations (per-slice index_base attribute; sub-device co-residency under trace).

## Plan

## Ground truth: what blaze actually requires (verified in /home/nachiket/tt-blaze)

Blaze's decode indexer is ONE FusedProgram chaining DsaIndexerSdpa -> IndexerLocalTopK -> CrossDeviceAllgatherTreeMerge (blaze/ops/distributed_indexer/op.py:161-202, 341-387). The four requirements, with the exact mechanism each uses:

1. **Explicit core placement**: 8 SDPA groups of 1x4 cores; the topk cores are the 1x4 row directly below each group, `topk_y = (sdpa_y + 1) % grid_h` (config.py:31-34), disjoint by construction and validated (`config.validate`, config.py:120-124: overlap check against sdpa_group_cores). The topk core LIST (block-major, core_id = block*4+slot) drives the reduction-tree hierarchy (indexer_local_topk/op.py:165-181).
2. **L1-streamed HEIGHT_SHARDED input**: `topk_input` is a zero-filled HEIGHT_SHARDED L1 TILE tensor on `config.topk_grid`, shard = [32, chunks_per_core * 2 tiles * 32] bf16 (op.py:84-106, 122-125, 309-320). SDPA raw-NOC-writes 2048-position chunks (STREAM_CHUNK=2048 = 2 [32,32] tiles, config.py:20-21) into each topk core's shard and increments a shared `stream_sem`; the topk DM1 kernel waits the semaphore, then pushes pages for compute (indexer_local_topk/op.py:86-87, 255-263 — `stream_input`, `stream_sem` are DM1 CT args). The coupling is so tight that `IndexerLocalTopK.compose` raises NotImplementedError: "input L1 is raw-NOC-written by DsaIndexerSdpa, which also shares stream_sem with it" (op.py:57-63). The producer side takes `topk_input_tensor`, `topk_stream_groups`, `stream_chunk_tiles`, `external_stream_sem` (dsa_indexer_sdpa/op.py:237-240).
3. **Index bit-stamping**: bits 0-13 within-bank, 14-16 local bank, 17-19 device (config.py:16-18); stamped in two halves so early cross-bank tree stages stay fused (indexer_local_topk/op.py:110-159).
4. **Co-residency**: everything above is emitted into one FusedProgram `f` (unified CT args, shared semaphores, cb_scratch arenas — op.py:185-272).

Per-bank validity is derived from cur_pos metadata (`pos_addr` CT arg, op.py:208, 234) with a physical-order `bank_validity_remap` (distributed_indexer/op.py:72-81).

## Our op today (the deltas to close)

- Signature: `(input, k, valid_length, return_values, num_slices, tile_output, index_dtype)` + internal row_start/row_count (device_operation_types.hpp:30-73). **No placement param**; input must be interleaved (`TT_FATAL(!input.is_sharded(), ...)`, device_operation.cpp:38); reader pulls from DRAM/L1-interleaved via TensorAccessor (reader_local.cpp:31-33, factory `interleaved_accessor_args` at program_factory.cpp:69-72); own program, own dispatch.
- The multi-rect factory self-places: `compute_column_split_config` searches a×b rectangles against the FULL worker grid (program_factory.cpp:520-540 model path, 629-643 override path), and `create()` tiles rect origins from (0,0): `ox = (r % rects_x) * local_grid_x; oy = (r / rects_x) * local_grid_y` (program_factory.cpp:925-928). Row-parallel factory creates kernels/CBs on the full grid and splits rows with `split_work_to_cores(grid, ...)` (program_factory.cpp:153, 246-248). The hybrid wrapper's wave math also uses the full grid (device_operation.cpp:309-314).
- Program hash includes k/opt-ins/dtype/layout/mem-layout/buffer-type/grid.x/grid.y + all split-config fields (device_operation.cpp:183-206). Tree flow control is already placement-agnostic: partner/winner coords go through `device->worker_core_from_logical_core` runtime args (program_factory.cpp:818, 835) and two program-local semaphores (1053-1054); recv CB spans the whole rectangle so every core sees one address (986-989).

## Increment 1 — `core_grid: std::optional<CoreRangeSet>` placement param

**Scope (v1)**: accept a single rectangular CoreRange (reject multi-range sets loudly; blaze's 1x4-rows-below-SDPA placement is a union of rectangles — defer to v2 or accept a CoreRangeSet whose bounding box is fully covered). Semantics: all placement math runs inside this rectangle instead of the full worker grid.

**Changes**:
- `operation_attributes_t`: add `std::optional<CoreRangeSet> core_grid` (device_operation_types.hpp:30); plumb through `invoke` (device_operation.cpp:250-272) and nanobind (topk_large_indices_nanobind.cpp:85-100 — mirror ttnn.topk's `sub_core_grids` naming, reduction/topk/topk.cpp:456).
- `compute_column_split_config` already takes `const CoreCoord& grid` (program_factory.hpp:65-71): pass the region's extent instead of `compute_with_storage_grid_size()` at the three call sites (program_factory.cpp:898, device_operation.cpp:150-158, hybrid wrapper device_operation.cpp:309). Factory `create()`: offset rect origins by the region origin (program_factory.cpp:925-928) and build `all_cores`/CB/kernel core sets from the region (row-parallel path: replace the full-grid `all_cores` at program_factory.cpp:246-248 and feed `split_work_to_cores` the region extent, offsetting each core by region origin when setting runtime args).
- Hash: add `attrs.core_grid` to `compute_program_hash` (device_operation.cpp:183-206; CoreRangeSet is hashable — matmul program configs containing CoreRangeSet are already hashed) and replace the grid.x/grid.y terms with the effective region extent.
- Hybrid wrapper: when `core_grid` is set, compute waves against the region's core count (device_operation.cpp:310-314); everything else falls out.
- Validation: region within `compute_with_storage_grid_size()`; region core count >= 2 for the tree path.

**What the factory's self-placement must invert**: nothing yet — increment 1 only *translates* the search/tiling domain. The true inversion (cores given, layout derived) arrives with increment 2.

**Size**: ~200 LOC (attrs+nanobind+hash ~60, factory offsets ~100, validation ~40). 2-3 days. **Unlocks**: blaze req (1) exactly (topk disjoint from SDPA/other-op cores); required for any sub-device co-residency (a persistent producer owns its cores; the consumer must be steerable off them — same reason ttnn.topk's routing predicate treats sub_core_grids as a disqualifier today, glm-callsite-map.md §E); independently useful to any ttnn caller sharing a chip with a resident op.

## Increment 2 — HEIGHT_SHARDED L1 input (resident, not yet streamed)

**Contract (the deterministic shard->rect assignment)**: input is HEIGHT_SHARDED ROW_MAJOR bf16 in L1; the shard spec's CoreRangeSet must be a full rectangle; shard i (row-major core order, matching `corerange_to_cores(..., true)` used at program_factory.cpp:930) IS tree slice i of the rectangle whose root is the region origin — i.e. the shard spec *is* the placement, and `core_grid`/`num_slices` must be omitted or equal to it. For rows==1 (decode shape): num_slices = number of shard cores, each shard = [1, chunks_per_core * llk_k] of one row. For rows>1: shard height = rows per rect, one rect per shard column group — v1 restricts to rows==1 (the decode/streaming use-case; prefill keeps DRAM-interleaved input).
- Slice geometry must match `compute_slice_runtime`'s even chunk split (program_factory.cpp:685-716): validate shard width == ceil(chunks/P)*llk_k for the leading slices (loud error otherwise), so `slice.start_element` equals the shard's global column base and the emitted indices stay row-global with no remap.

**Changes**:
- Relax `TT_FATAL(!input.is_sharded())` (device_operation.cpp:38) to allow HEIGHT_SHARDED + BufferType::L1 with the contract above; add shard-grid fields to the hash (memory_layout/buffer_type are hashed already at device_operation.cpp:196-197, but the shard GRID is not — add it).
- Factory: derive `ColumnSplitConfig` FROM the shard spec (the inversion): num_slices = shard cores, local_grid_x/y = shard grid extent, num_rects = 1, rect origin = shard grid origin — bypassing the cost-model/override search entirely.
- Reader: new `reader_shard.cpp` variant (following the byte-identical-kernel discipline of reader_local.cpp:5-9): the shard is resident in this core's own L1, so the reader is a local L1->CB copy loop — same structure as reader_local.cpp:37-61 with the TensorAccessor swapped for the shard's base address (or generic `TensorAccessorArgs(*buffer)` which handles sharded layouts), reading its OWN shard only (slice_offset becomes 0; the global index base is carried by `slice.start_chunk`, already a compute runtime arg at program_factory.cpp:829). Alternative considered and rejected for v1: aliasing cb_in onto the shard via `set_globally_allocated_address` — it removes the copy but breaks the compute kernel's chunked cb_wait cadence and double-buffering; keep the copy (L1->L1 local reads are cheap) and revisit.
- Bit-stamping hook (cheap adjacency): because shard->slice assignment is deterministic, a per-call `index_base`/`index_or_mask` attribute (single uint32 OR-ed into emitted indices by the writer) closes blaze req (3) for within-device bank stamps at near-zero cost; device bits stay with the cross-device merge (blaze-side). Optional +50 LOC.

**Size**: ~300 LOC (validation/contract ~120, factory derivation ~80, reader ~60, hash ~20). 4-5 days. Depends on increment 1's region machinery. **Unlocks**: blaze req (2)'s resident-input half — no DRAM materialization or read of the score tensor; independently useful even non-streamed (producer writes shards, one dispatch later topk reads L1 — removes the DRAM round-trip from the GLM prefill critical path even before overlap).

## Increment 3 — streaming handshake (producer credits) WITHOUT FusedProgram

**Mechanisms available in tt-metal today** (verified):
- **GlobalCircularBuffer + SubDevice** (tt_metal/api/tt-metalium/global_circular_buffer.hpp:129-159: `CreateGlobalCircularBuffer` with sender->receiver core mapping, `CreateCircularBuffer(program, ..., global_cb)` binding a program-local CB onto the global address space, `UpdateDynamicCircularBufferAddress`). Kernel-side remote-CB flow control exists in tt_metal/hw/inc/api/remote_circular_buffer.h (pages_sent/pages_acked counters, `resize_remote_sender_cb_interface`, remote_cb_wait/pop — lines 24-142). Proven at model scale: ttnn.dram_prefetcher takes an optional GlobalCircularBuffer (prefetcher/dram_prefetcher.cpp:11-17), matmul consumes `global_cb` in its program config (matmul_device_operation_types.hpp:28), and llama3_70b_galaxy runs the producer persistently on a dedicated **SubDevice** while consumers run on a worker SubDevice (`ttnn.SubDevice` + `mesh_device.create_sub_device_manager`, prefetcher_common.py:67-93; same pattern in models/tt_transformers/tt/prefetcher.py:211-224). Two programs, two sub-devices, concurrent on one chip — co-residency without FusedProgram.
- **Global semaphore + persistent L1 tensor**: `ttnn.create_global_semaphore` exists (CCL infrastructure); producer raw-NOC-writes into a persistent tensor and `noc_semaphore_inc`s the consumer core — byte-for-byte blaze's `stream_sem` design (indexer_local_topk/op.py:86-87).
- **In-tree single-program fusion**: models/experimental/ops/descriptors/fusion exposes `Sequential`/`Parallel` over OpDescriptors with GlobalCB communication in ONE program (tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential_global_cb.py:8-15, fusion/__init__.py:12-38) — the in-tree FusedProgram analog. Rejected for the minimal increment: it requires descriptor-form kernel re-wrapping of both ops.

**Recommended design (blaze-shaped, minimal op-side delta)**: global-semaphore credits over the increment-2 persistent HEIGHT_SHARDED tensor.
- Op-side: add `stream_sem: std::optional<GlobalSemaphore>` (or its address) + implicit credit unit = one LLK chunk (llk_k elements — for k=2048 that is 2048 positions, exactly blaze's STREAM_CHUNK, config.py:20). `reader_shard.cpp` gains a gated loop: before pushing chunk c, `noc_semaphore_wait_min(sem_ptr, c+1)`; everything downstream (compute, tree, writer) is untouched because CB backpressure already serializes. `valid_length` keeps working as the total-credit bound. Runtime-only (semaphore address is a runtime arg; `stream_sem.has_value()` selects the reader variant, hashed).
- System-side: producer program on SubDevice A / CQ or workload slot 1, topk on SubDevice B; enqueue both non-blocking so they are in flight together (exactly the prefetcher deployment shape, prefetcher_common.py:88-93). Under trace this becomes one captured superstep.
- GlobalCircularBuffer alternative: strictly more machinery (sender-side interface in the producer, receiver-side in our reader) but gives hardware-managed backpressure in BOTH directions (producer can't overrun a slow topk). Choose it only if the producer must run more than 2x ahead; blaze gets away with a 2-page input CB + semaphore, so start with the semaphore.

**Size**: op-side ~150 LOC (reader gate + attrs/hash + validation); wiring/example ~150 LOC (sub-device setup, producer stub). 4-6 days; this is the riskiest increment (no hardware validation possible under current constraints — flow-control bugs are hang-shaped). **Unlocks**: with increment 4, GLM prefill score->topk overlap; standalone, it lets ANY producer (blaze SDPA included, if it targeted our op) stream into topk_large_indices.

## Increment 4 — indexer_score side

Producer is `ttnn.experimental.ring_indexer_score_dsa` (models/demos/deepseek_v3_d_p/tt/mla/indexer.py:717-737 — the sole GLM prefill call site feeding topk at indexer.py:737). Its output spec is hard-wired: ROW_MAJOR bf16 inheriting q's memory_config (indexer_score_device_operation.cpp:500-512), q validated interleaved (line 66).

**Changes**:
- `output_memory_config: std::optional<MemoryConfig>` (or preallocated `optional<Tensor> out`) on the op: when HEIGHT_SHARDED-L1 on the topk core set, `compute_output_specs` uses it instead of q's config; validation asserts the shard geometry matches the increment-2 contract (shard width = chunks_per_core*llk_k over the [Sq, T] score plane — note prefill has Sq=160 rows, so this needs the rows>1 shard contract or a per-row-band variant; the decode-shaped rows==1 case is the v1 target).
- Sharded writer variant of writer_indexer_score.cpp: route each score column band to its consumer core's shard via NOC write (the score work-split already produces column-banded partials; the writer's DRAM interleaved TensorAccessor swaps for per-core shard destinations).
- Streaming (+increment 3): after each chunk-band's `noc_async_write_barrier`, `noc_semaphore_inc` the consumer core's stream_sem — the blaze SDPA pattern (dsa_indexer_sdpa/op.py:237-266) transplanted.

**Size**: ~400 LOC (output-spec/validation ~100, sharded writer ~200, semaphore inc + attrs ~100); 5-8 days — the score op's ring/work-split coupling (ring_indexer_score_dsa_program_factory.cpp) is the complexity driver. Depends on increments 2 (contract) and 3 (semaphore semantics).

## Dependency order and what each stage unlocks

1 -> 2 -> 3 -> 4, each landable alone:
- **After 1**: blaze req (1) closed; op is sub-device/co-residency compatible; ttnn callers can pin topk off contended cores. (Also removes the routing predicate's sub-grid disqualifier for topk.cpp:306 sampling callers, a separate follow-up.)
- **After 2**: L1-resident input — DRAM round-trip eliminated. For the 160x65536 GLM last chunk, the hybrid's 467.0us (paper-topk/evidence/glm-hybrid-composite/RESULTS.md:11) is reader-DRAM-bandwidth heavy; L1-resident input attacks that term even without overlap (unquantified — needs measurement).
- **After 3**: producer-agnostic streaming consumer. Blaze decode replacement becomes geometrically expressible: 32 topk cores, 64 chunks/dev at 128k (P=32 -> per-row units 2*ceil(64/32)+log2(32) = 9 vs row-parallel 128), credits at 2048-position granularity — matching the 24.4us comp3 fused cell's topology (glm-callsite-map.md:51, §D). Remaining blaze-only deltas: bank/device bit-stamp (approximable via the increment-2 index_base hook + host remap) and single-program dispatch latency (amortized under trace).
- **After 4**: GLM prefill overlap. Per-layer-per-chunk indexer cost bound falls from t_score + t_topk to max(t_score, t_topk) + drain (one chunk tail + ceil(log2 P) merge units); since ring_indexer_score_dsa is the indexer's dominant cost (indexer.py:735-737 commentary: the removed all-reduce it replaced was "the indexer's dominant cost"), t_score >= t_topk is plausible and the ~467us/last-chunk topk would leave the critical path — but this must be measured, not asserted.

## Evidence

- /home/nachiket/tt-blaze/blaze/ops/distributed_indexer/config.py:16-34 (index bit layout, STREAM_CHUNK=2048, topk row below SDPA)
- /home/nachiket/tt-blaze/blaze/ops/distributed_indexer/config.py:120-124 (SDPA/topk disjointness validation)
- /home/nachiket/tt-blaze/blaze/ops/distributed_indexer/op.py:84-106,122-125,309-320 (HEIGHT_SHARDED L1 topk_input, shard shape)
- /home/nachiket/tt-blaze/blaze/ops/distributed_indexer/op.py:161-202,336-362 (FusedProgram chain, shared stream_sem)
- /home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/op.py:57-63 (compose raises: input raw-NOC-written + shared stream_sem, inexpressible outside FusedProgram)
- /home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/op.py:86-87,255-263 (DM1 stream_sem credit wait)
- /home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/op.py:110-159 (split bank stamp)
- /home/nachiket/tt-blaze/blaze/ops/dsa_indexer_sdpa/op.py:237-266 (producer topk_input_tensor/topk_stream_groups/external_stream_sem params)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:38 (interleaved-only TT_FATAL)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:150-158 (allow_multi_row=false auto path)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:183-206 (program hash fields)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:293-328 (hybrid_row_split uses full grid)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:153,246-248 (row-parallel full-grid work split + CBs)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:520-540,629-643 (rect search against full grid)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:925-930 (rect origins tiled from (0,0), corerange_to_cores row-major)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:685-716 (compute_slice_runtime even chunk split)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:818,835,986-989,1053-1054 (placement-agnostic tree: runtime partner coords, rectangle-spanning recv CB, program semaphores)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader_local.cpp:16-61 (slice reader structure to clone for reader_shard)
- /home/nachiket/tt-metal/tt_metal/api/tt-metalium/global_circular_buffer.hpp:129-159 (CreateGlobalCircularBuffer / program CB binding / dynamic update)
- /home/nachiket/tt-metal/tt_metal/hw/inc/api/remote_circular_buffer.h:24-142 (kernel-side remote CB pages_sent/acked flow control)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/prefetcher/prefetcher/dram_prefetcher.cpp:11-17 (op takes GlobalCircularBuffer)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation_types.hpp:28 (consumer-side global_cb in program config)
- /home/nachiket/tt-metal/models/demos/llama3_70b_galaxy/tt/prefetcher_common.py:67-93 (SubDevice + create_sub_device_manager producer/worker split)
- /home/nachiket/tt-metal/models/tt_transformers/tt/prefetcher.py:211-224 (generalized sub-device manager wiring)
- /home/nachiket/tt-metal/tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential_global_cb.py:8-15 (in-tree Parallel/Sequential fused-program + GlobalCB, single program)
- /home/nachiket/tt-metal/models/experimental/ops/descriptors/fusion/__init__.py:12-38 (Sequential/Parallel/FusedOp exports)
- /home/nachiket/tt-metal/models/demos/deepseek_v3_d_p/tt/mla/indexer.py:717-737 (ring_indexer_score_dsa -> topk_large_indices producer/consumer pair)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.cpp:59-66,500-512 (interleaved-only inputs; output inherits q.memory_config ROW_MAJOR bf16)
- /home/nachiket/tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:294-306 (sub_core_grids as routing disqualifier precedent)
- /home/nachiket/tt-metal/paper-topk/evidence/glm-callsite-map.md:66-71,87 (blaze placement/streaming/stamping/FusedProgram requirement statement, §E gap analysis)
- /home/nachiket/tt-metal/paper-topk/evidence/glm-hybrid-composite/RESULTS.md:11-14 (467.0us GLM hybrid, 98.8us 30-row rect numbers)

## Risks

- Increment 3 is hang-shaped and cannot be validated under the current no-hardware constraint: semaphore-credit flow control bugs present as CB-wait deadlocks (CWFW/NSW), and the sub-device concurrent-enqueue semantics (two programs genuinely in flight, not serialized by dispatch) need an on-silicon smoke test before any claim of overlap.
- CoreRangeSet in the program hash: verify tt::stl hash_operation handles optional<CoreRangeSet> deterministically (matmul precedent suggests yes, but the op's hash currently avoids all set-typed fields — a hash miss here silently forks cache entries, a hash collision silently reuses wrong placement).
- Increment 2's v1 rows==1 restriction leaves GLM prefill (160 rows) on the DRAM-interleaved path until the rows>1 shard contract (shard height = rows-per-rect band) is designed; the prefill overlap claim in increment 4 therefore needs either that extension or a per-row-band streaming scheme — the max(t_score,t_topk) bound is stated, not measured, and t_score has no measured cell in our evidence base.
- Blaze replacement remains partial by design: device-level bit-stamping (bits 17-19) and the cross-device tree merge stay blaze-side; if blaze requires bit-identical index words at the local-topk output boundary, the increment-2 index_base OR-mask must reproduce the split-stamp layout exactly (bits 14-16 at LOCAL_BANK_SHIFT) — feasible but must be validated against sparse_k_filter_sfpu.hpp's decoder.
- Sub-device co-residency vs the hybrid wrapper: with core_grid set, hybrid_row_split's concat path launches TWO programs plus a concat on the restricted region — interaction with a persistent producer occupying the rest of the grid needs an explicit rule (likely: core_grid disables the hybrid wrapper, single launch only).
- The in-tree descriptor fusion framework (models/experimental/ops/descriptors) is moving; if it graduates to cover TTNN device ops, a descriptor-form topk could obsolete the sub-device design — worth a checkpoint with its owners before investing in increment 3's wiring.
