# AutoDebug: Ornith TP4 batched full-attention correctness

**Leading diagnosis: the candidate's singleton core ranges put Q users on different cores from the ones SDPA reads.** For batch32, the candidate places user `i` at `(i//4, i%4)`, while SDPA expects user `i` at `(i%8, i//8)`. Consequently, SDPA row1 consumes **user4's query** with **user1's KV cache and position**. Row0 is correctly paired. This is a concrete source contract mismatch, supported by a host-only check of the actual TTNN core enumeration. It predicts the reported first passing/failing boundary without requiring a page-table padding defect.

The smallest proposed repair is to construct ordered rectangular-prefix shard grids, replacing the failing revision's set of singleton ranges in `shard(y_offset)`. For batch32 these are single rectangles `(0,0)..(7,3)` and `(0,4)..(7,7)`. **The mapping defect is established; recovery of model PCC still requires device validation.** This investigator edited no implementation and ran no devices, reproduction, reset, or AutoDebug launcher.

## Scope, revisions, and direct observations

- Checkout HEAD: `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`, with pre-existing working-tree changes. Investigation date: 2026-11-12. Only this report was written by this investigator.
- At initial inspection the [live candidate](models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_attention_batch_candidate.py) had SHA256 `dfc2226e29cf233a380973e8e140bb2ae9a3bccf7979e7c0284678d699e16047`, matching the archived `source_snapshots/multichip_attention_batch_candidate_dfc2226e29cf.py.txt`. During first-draft review the shared workspace changed externally: the live candidate became `17ebeb822af90f3f77842e4a7fbd8c4fb350ab92381de9c980e084ec459d1647`, with rectangle-plus-tail grid construction at lines35–43. **All CAND line references and failure analysis below refer to the frozen `dfc2226...` snapshot**, not that later live implementation. This report does not claim the updated file reproduces the original failure or that its PCC has been validated.
- [attention_batch_adapted_32.log](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/attention_batch_adapted_32.log), lines981–991, records `AssertionError: ('decode_0_1', 0.9720041155815125)` at the `0.995` threshold. Sequential row checking establishes that row0 passed **at least** `0.995`; its exact PCC is not printed. Later rows and steps were not checked after this assertion.
- The failed [stack snapshot](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/source_snapshots/multichip_stack_e211baf24010.py.txt), lines124–138, allocates 1152-token private caches, creates independent shuffled 36-entry tables, and uses prefix lengths `31 + row%3`, followed by `17 + row%2` tokens. First decode positions are `[48,50,50,49,49,51,...]`. The live stack has subsequently gained options; the archived fixture supplies the failure-specific dimensions here.
- [recurrent_adapted_32.json](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/recurrent_adapted_32.json) records batch32, replicated residuals, layers `[0,3]`, four checked trace replays, 64 prefill comparisons, and 128 decode comparisons; minimum decode PCC is `0.999824047088623`. That path retains independent full-attention calls through `BatchedCandidate.decode_batch_packed`, lines82–89. Its recorded decoder hash is `8dbed3ee...`, whereas the inspected/archived decoder is `07fd1a68...`; this is supporting historical evidence, not an exact one-variable A/B reproduction.
- The initial optional-access failure is requester-reported historical evidence. The current source explains why the fragmented grid selects the subcore concat factory; the adapted log establishes the subsequent numerical failure.

## Main source references

Paths below are repository-relative. Short names used in the analysis refer to these files; numbers after them are source lines.

| Name | Source |
|---|---|
| CAND | [failing multichip_attention_batch_candidate snapshot, dfc2226...](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/source_snapshots/multichip_attention_batch_candidate_dfc2226e29cf.py.txt) |
| CORE | [core_coord.cpp](tt_metal/common/core_coord.cpp) |
| DIST | [buffer_distribution_spec.cpp](tt_metal/impl/buffers/buffer_distribution_spec.cpp) |
| I2S | [interleaved_to_sharded_program_factory.cpp](ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp) |
| FUSED | [paged_tiled_fused_update_cache_program_factory.cpp](ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp) |
| CACHE-R | [reader_paged_fused_update_cache_interleaved_start_id.cpp](ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp) |
| SDPA-F | [sdpa_decode_program_factory.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp) |
| SDPA-R | [reader_decode_all.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp) |
| SDPA-DF | [dataflow_common.hpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp) |
| CONCAT-F | [nlp_concat_heads_decode_subcoregrids_program_factory.cpp](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp) |

## First inconsistent mapping: sharded Q into SDPA

### 1. The shard grid is not enumerated in the order suggested by its comprehension

CAND:34–46 constructs a Python **set of 32 singleton CoreRanges**. Although their coordinates are generated as `(i%8, i//8+y_offset)`, the comprehension's iteration order does not define the resulting shard order.

- The Python set constructor binds to `std::set<CoreRange>` in `ttnn/cpp/ttnn-nanobind/tensor.cpp:623–625`.
- `CoreRange::operator<` compares start coordinates (`tt_metal/api/tt-metalium/core_coord.hpp:104–107`), and the coordinate comparator sorts **x before y** (`tt_metal/third_party/umd/tt-umd/api/tt-umd/types/xy_pair.hpp:32–33`).
- `CoreRangeSet(std::set)` copies this sorted order without merging ranges (CORE:161–163).
- `corerange_to_cores(..., row_wise=true)` iterates the stored ranges and orders cores **within each range**, without globally re-sorting the result (CORE:604–623). A singleton has no internal order to change.

The actual batch32 Q/K sequence therefore starts:

```text
user:  0      1      2      3      4      5      6      7
core: (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)
```

This controls real placement, not just a diagnostic listing. HEIGHT_SHARDED tensor layout uses the one-dimensional distribution strategy (`tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:145–153`); DIST:168–173 obtains the core list using the helper above, and DIST:299 preserves it in `cores_with_data`. `ttnn/core/tensor/tensor_utils.cpp:99–100` returns that list for L1 shards. I2S:60,335,381,402–406 assigns successive input blocks to those cores. Each batch user's padded `[32,256]` shard is eight BF16 tiles.

### 2. The selected SDPA path assumes a different physical Q owner

CAND:61–71 supplies an explicit SDPA config: grid8×8, fixed K chunk256, Q chunk32, with no SDPA `sub_core_grids`. The `sub_core_grids` argument at CAND:74 belongs to the later head-concat operation and does not configure SDPA.

Effective SDPA parameters are:

| Parameter | Value / consequence |
|---|---|
| Q logical / padded shape per chip | `[1,32,4,256]` / `[1,32,32,256]` |
| K/V pooled shape per chip | `[1152,1,32,256]`; 32 private pools ×36 pages |
| Data formats | BF16 Q/K/V updates; BFP8_B cache storage (`multichip_stage4_reference.py:80,399`; same policy in `tt/multichip_decoder.py:81,425`) |
| Page table / positions | `[32,36]` INT32 ROW_MAJOR / `[32]` INT32 ROW_MAJOR, DRAM |
| Query sharding | HEIGHT_SHARDED L1, 32 cores, `[32,256]` per core |
| Paged / causal / MLA | paged, causal, ordinary attention; MLA unset |
| Query head parallel factor | 1 |
| Tile-space dimensions | `S=1152`, `St=36`, `DHt=vDHt=8`, `PNHt=1`, fixed `Sk_chunk_t=8` |
| Selected SDPA tile variant | `use_half_tile=true`: causal BF16 Q with four heads (`SDPA-F:513`); Q input is read as 16×32 half tiles, final output is 32×32 full tiles |
| SDPA compute config | Explicit `self.compute`: HiFi4, approximate math off, FP32 destination accumulation and L1 packer accumulation on; SFPU exp defaults to the program's approximate-exp mode |
| Available cores / KV heads | 64 / 1 |
| Maximum cores per head/batch | 16 from the explicit Python config constructor (`transformer_nanobind.cpp:47`); the wrapper's no-config default does not apply |
| Actual cores per batch | `min(64,16*32*1)/32 = 2`; 32 reducers and 32 workers |
| Output | Default interleaved DRAM, not Q's sharded memory config |

SDPA-F:202–216 computes the allocation. With sharded Q and no head parallelization or subgrid, SDPA-F:342–355 assigns reducer `b` to:

```text
R(b) = (b % 8, b // 8)
```

The actual candidate placement is:

```text
S(i) = (i // 4, i % 4)
```

SDPA-R:226–238 calls `read_q`; SDPA-DF:575–609 reads directly from the reducer's L1 address, using the local core for reducers and the reducer's NoC coordinates for workers. It does **not** resolve this arbitrary input-grid ordering using `q_batch_offset`. Thus the query consumed for SDPA batch `b` is:

```text
P(b) = S⁻¹(R(b)) = 4*(b % 8) + b//8
O[b] = Attention(Q[P(b)], K_pool_for_user[b], V_pool_for_user[b], position[b])
```

| SDPA batch | Actual location of its intended Q | Core SDPA reads | Query actually consumed | Query's RoPE position | KV/mask position |
|---|---|---|---|---:|---:|
| 0 | `(0,0)` | `(0,0)` | Q0 | 48 | 48 |
| 1 | `(0,1)` | `(1,0)` | **Q4** | **49** | **50** |
| 2 | `(0,2)` | `(2,0)` | Q8 | 50 | 50 |
| 8 | `(2,0)` | `(0,1)` | Q1 | 50 | 50 |
| 31 | `(7,3)` | `(7,3)` | Q31 | 50 | 50 |

Only batches0 and31 are fixed points of this mapping. This is a structural error even where positions happen to be equal: those users have different query contents and independent KV histories.

### 3. K/V updates and later head concat do not cancel this mismatch

K/V sharding follows the same `S(i)` enumeration, with V shifted by four core rows. FUSED:629–644 enumerates those grids identically, and FUSED:678–703 gives both corresponding cores `my_batch_idx=i`. CACHE-R:101,119,138–146 then uses position `i` and page-table row `i`. Therefore K1 at `(0,1)` and V1 at `(0,5)` update **user1's correct pool**, rather than moving that history with the wrongly selected Q4. Position selection in SDPA similarly uses logical batch order (SDPA-R:170).

SDPA returns interleaved DRAM by default (`sdpa_decode.cpp:163`). Its writer stores `O[b]` at the logical batch tile offset `b*8` (`writer_decode_all.cpp:426,507–511`). CAND:74 then re-shards this output using `S(b)`. CONCAT-F:81–92 enumerates the same `S(b)` into its source-coordinate table, and its `reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp:45,55–93` reads eight tiles per batch source. Thus concat restores logical order `[O0,O1,...]`; it neither repairs nor adds a batch permutation. User1's gate and subsequent residual/MLP still consume `O1` computed from Q4.

A normal Q sharded→interleaved roundtrip would also preserve logical row order, because both conversions use `S`. Such a roundtrip passing would **not** refute the SDPA physical-owner mismatch.

## Page-table, position, pool, and RoPE adjudication

### The reported 36-entry width is not a demonstrated consumer stride defect

The selected cache factory is **tiled fused update**, selected by the layout of K/V updates (`paged_fused_update_cache_device_operation.cpp:22–26`), not the standalone update factory or the ROW_MAJOR-cache factory. FUSED:195 and SDPA-F:236–237 both use the page-table buffer's **aligned page size**.

On Blackhole DRAM the alignment is64 bytes (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379–394`; `tt_metal/impl/buffers/buffer.cpp:878–880`). A logical row of `36*4=144` bytes has a192-byte transport pitch. CACHE-R:114–120 reads the indexed page through a tensor accessor; SDPA-DF:72–91 likewise uses the aligned row size and `page_id=cur_batch`. Neither path assumes a144-byte physical pitch for this call.

The rows are bank-interleaved:192 is an aligned page size, **not** a claim that physical row1 lives at `base+192` on the same DRAM bank. No power-of-two or eight-entry logical-width padding requirement follows from these two selected consumers. Padding to40 or48 entries is a useful optional experiment, not the leading repair.

Upstream table concatenation was also inspected. At32 inputs it exceeds the codegen eight-input limit and uses native concat (`concat_codegen_supported.hpp:23`, `concat_codegen_supported.cpp:82–84`). Native concat gives inputs separate accessors, reads144 logical bytes into aligned staging, and writes to each output row through its accessor (`concat/device/concat_program_factory.cpp:65–70,197–205,235–240,398–404`; `reader_concat_stick_layout_interleaved_start_id.cpp:78–88`). Scalar table addition preserves INT32, ROW_MAJOR, and the source memory config (`binary_ng_device_operation.cpp:846–872`). No first-row-versus-later-row pitch mismatch was found in this construction.

### Pool offsets and positions agree across selected read/write paths

For private permutations `π_b`, CAND:52–59 constructs `table[b,j]=36*b+π_b[j]`. Concatenating the private caches on dimension0 gives exactly those physical page ranges. The paged reader computes `physical_block_id * num_heads * block_size_t * Wt`, with no extra user-stride addition (CACHE-R:138–146; SDPA-DF:53–68). Here these factors are `1*1*8`, so offsets are in **pages**, not tokens or bytes.

| First decode calculation | User0 | User1 |
|---|---:|---:|
| Position-vector entry / byte within vector | 0 / 0 | 1 / 4 |
| Position | 48 | 50 |
| Table row / logical page column | 0 / 1 | 1 / 1 |
| Physical page | `π₀[1]` | `36+π₁[1]` |
| First cache tile in that page | `8*π₀[1]` | `288+8*π₁[1]` |
| Token row in that page | 16 | 18 |
| BF16 untilized update offset within the tile-row staging | `16*512=8192` bytes | `18*512=9216` bytes |

The actual permutation entries are not logged; no correspondence between particular numeric page IDs is assumed. All valid entries remain within the user's36-page pool. Scatter-back slices at CAND:77–82 reverse exactly the dimension0 concatenation. No pool-offset or scatter-order contradiction was found.

CAND:23–32 gathers RoPE rows using the same position vector, moves users onto the sequence axis (`[1,heads,n,256]`), applies RoPE, and restores `[1,n,heads,256]`. This is the intended logical association before sharding. The static inheritance chain is `AttentionBatchCandidate → RecurrentCandidate → BatchedCandidate → Candidate → multichip_stage4_reference.MultichipDecoder` (`tests/multichip_optimization.py:8–11`). Its `rope` at `tests/multichip_stage4_reference.py:610–620` rotates the first64 dimensions and retains the rest; the analogous live decoder method at `tt/multichip_decoder.py:651–661` has the same logic. The adapted run's exact external setup is not embedded in the failure log, so these paths are not assumed to be one revision. No earlier logical user permutation was found. The leading defect subsequently combines **Q4 rotated for49** with user1's history at50; changing RoPE positions independently would not repair query ownership. No separate bitwise/numerical equivalence of the batched RoPE kernel was established on hardware.

## Sibling examples and the earlier optional-access failure

- `tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py:475–499` packs a common paged pool and batch×pages table, and lines530–533 construct sharded Q from a **contiguous CoreRange**. Lines593–616 pass `[1,B,heads,D]` Q and a one-dimensional per-user position tensor. These match the candidate's logical tensor contracts and avoid its fragmented-core ordering.
- `tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py:126–150` uses `num_cores_to_corerangeset_in_subcoregrids(...,True)` for K/V grid preparation. Lines157 and191–234 check differing user positions and individual cache results. These tests support the cache contract but do not prove the combined batch32 singleton-grid SDPA path.
- `nlp_concat_heads_decode_device_operation.cpp:133–139` selects subcore mode when the input has multiple ranges, even if its core membership forms a rectangle. Output-spec construction at lines93–99 dereferences `sub_core_grids.value()`. This explains why a singleton-range input with the option omitted can encounter the reported bad optional access. Supplying that option selects a valid concat setup, but leaves SDPA's independent Q-owner assumption unchanged. A canonical single rectangle removes the fragmentation that triggered both problems.

## Asynchronous execution and buffer-state audit

The failed stack snapshot enqueues two warmups and restores snapshots between them, captures the forward, restores state, then updates host-backed inputs/positions and executes the trace **with `blocking=True`** before downloading and checking rows (lines164–185). Therefore the assertion is after a completion boundary, rather than merely after a host enqueue. The log does not isolate the first bad numerical intermediate.

For both positions48 and50, fixed K chunk256 rounds `position+1` to256: `PSt=8`, one active chunk. With two cores/batch, `rt_args_common.hpp:61–89` assigns reducer core0 chunk `[0,1)` and worker core1 `[0,0)`. Reader, compute, and writer use the same helper and skip empty work. External Q shards contain eight32×32 BF16 tiles (16384 bytes); the selected half-tile reader reserves/pushes eight16×32 tiles (8192 bytes), reading1024 bytes and advancing2048 bytes per source tile (`SDPA-DF:592–602`). This optimization does not change which core owns the query. Final output is eight full32×32 tiles (`SDPA-F:544–559`); the interleaved-output writer waits for, writes, and pops those eight tiles (`writer_decode_all.cpp:426–429,507–516`). Thus no multi-worker reduction is needed at this first failure, and inspection found no competing count mismatch necessary to explain it.

Fused-cache program reuse also rebinds caches, updates, positions, and page tables (`FUSED:69–80,722–784`). Positions are read from the tensor on device. No stale position/address explanation was established. Trace lifetime or target math issues remain possible additional defects, but the deterministic core mismatch exists before invoking them.

## Smallest proposed fixes

1. **Repair candidate grid construction first.** Preserve the selected cores and shard shapes, but represent them in the global row order SDPA expects. For `1<=n<=32`, the following replaces only grid construction inside `shard(y_offset)`; it addresses grid ordering and does not certify all other operators' small-batch support:

   ```python
   available = ttnn.CoreRangeSet({
       ttnn.CoreRange(
           ttnn.CoreCoord(0, y_offset),
           ttnn.CoreCoord(7, y_offset + 3),
       )
   })
   cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
       ttnn.CoreCoord(0, y_offset), n, available, True
   )
   ```

   At32 this returns one rectangle. Keep Q/K at offset0, V at offset4, and keep the explicit head-concat subgrid argument. Host-only checks verified this helper enumerates `(i%8,i//8+y_offset)` for every `n=1..32`, at both offsets. This investigator did not apply the fix or validate it on devices; the later live workspace already contains the equivalent rectangle-plus-tail construction described above.

2. **A separate diagnostic bypass:** convert Q to interleaved DRAM immediately before SDPA. Its unsharded-Q reader uses the logical batch tile offset (SDPA-DF:619 onward), avoiding the reducer-local ownership assumption. Hold caches, table, positions, RoPE, compute settings, and concat constant. This adds data movement and is useful as a discriminating control; the rectangle change is the smaller production intervention.

3. **Optional callee hardening:** validate that a sharded-Q input's actual core enumeration equals the selected SDPA reducer mapping, or teach SDPA to use the input's actual Q-core map. The current path silently assumes this placement. Avoid globally changing `corerange_to_cores`: many producers and consumers intentionally share its range-preserving order. This investigation does not require a general kernel rewrite to fix the candidate.

## Discriminating follow-up checks, in execution order

These require the caller's normal device-validation workflow and were **not run** here. Use fresh/restored state for each comparison and synchronize before inspecting values.

1. **Inspect physical Q ownership before SDPA.** For batches0,1,4,8,31 on each TP rank, compare the actual Q shard on SDPA's reducer `(b%8,b//8)` with the correctly normalized/rotated logical Q for that user. The current grid predicts user4 on reducer1, user1 on reducer8, and users0/31 correctly placed. Use row/head/rank-distinct values or explicit shard metadata. A generic shard roundtrip is insufficient.
2. **Change only singleton ranges to ordered rectangular prefixes.** Rerun the exact failed32-user shuffled-table fixture, preserving width36 and differing positions. Compare every row, not just the original first assertion. Recovering all rows would strongly localize the failure to core assignment. Batch1 and batches4–8 are useful prospective controls: one-row singleton grids have the expected ordering; batches>8 first introduce the discrepancy. Batches2–3 have a separate existing head-concat admission constraint and are not suitable clean controls.
3. **Check eager intermediates against independent full-attention calls.** Compare normalized Q/K/V before RoPE; each user's Q/K after RoPE; pooled K/V immediately after fused update; SDPA output before concat; and output after concat/before gating. The leading diagnosis predicts correct logical Q before SDPA and correctly updated KV caches, followed by incorrectly paired SDPA attention rows. Compare SDPA output with a reference explicitly using `Q[P(b)]` and `KV[b]`; do not compare against a mere permutation of correct outputs, since the KV history does not move with Q.
4. **Verify page descriptors exactly.** Download the logical table and position vector outside the guarded forward. Require `table[b]==π_b+36*b`, `positions[b]==starts[b]+step`, INT32 ROW_MAJOR, logical shape32×36, and the actual aligned buffer pitch. Check the updated slot at physical page `table[b,pos//32]`, token `pos%32`, all256 dimensions, and all TP ranks. If this fails, the failure is earlier than the proposed SDPA mismatch and needs separate investigation.
5. **Check evolving state and completion.** Compare each private K/V cache after scatter with an independently updated private cache, including untouched slots. After an eager pass, repeat warmup, program-cache reuse, capture, restore, and four blocking trace replays with changing inputs/positions. Retain the original PCC threshold and inspect all output rows; do not accept tolerance relaxation as validation of the mapping fix.
6. **Use padding only as an orthogonal control if needed.** Hold the repaired grid constant while comparing table widths36 and padded widths40/48 with identical active entries. A change here would motivate a new table-construction/transport investigation. Padding alone does not alter the proven Q-core mismatch.

## Other potential issues and limits

- **Numerical or RoPE-kernel differences:** no hardware equivalence check was performed. The ordinary batched logical RoPE mapping is coherent, and no numerical hypothesis is needed to explain why user0 is the first aligned row and user1 the first mispaired row. The exact PCC magnitude cannot be predicted from static source alone, especially after gating, residuals, projections, and the MLP.
- **Table padding:** no consumer stride contradiction was found for this selected DRAM/tiled-fused/SDPA path. This does not certify all standalone-cache, L1-page-table, geometry-override, or historical-revision paths. They are not the candidate's selected variants.
- **State scatter, target math, or trace reuse:** not proven correct on hardware. Their source ordering is coherent; they remain follow-ups if the rectangle-only intervention fails to recover all comparisons.
- **Batches2–3:** supplying the input's own `n`-core grid to head concat cannot satisfy its requirement for at least four output-head cores (`nlp_concat_heads_decode_device_operation.cpp:61–66`). A general small-batch implementation must supply enough output cores or use another concat setup. This cannot explain the batch32 numerical failure.
- **Evidence coverage:** row31's aligned mapping is a prediction, not an observed pass; checking stopped at row1. The historical recurrent pass used a different recorded decoder hash. Neither fact is treated as a controlled falsification or proof of the entire proposed fix.

## First-draft claim review

The headline was rechecked against actual Python binding/ordering, tensor distribution, sharding conversion, selected SDPA factory and reader, fused-cache indexing, interleaved output writes, and head-concat source enumeration. The host-only constructor/enumerator check confirmed that the singleton list is exactly `[ (0,0),(0,1),(0,2),(0,3),(1,0),... ]`; the rectangle and general prefix-helper checks confirmed the proposed intended ordering without opening a device. An independent xhigh subagent reviewed the first draft and confirmed the causal chain, while identifying the concurrent live-source change, the need to state the half-tile variant, and the small-batch concat constraint; those corrections are incorporated. The analysis explicitly tested whether paired K/V movement or output conversion could cancel the discrepancy; neither does. Claims of a necessary page-table padding repair, a separate RoPE permutation, or a demonstrated scatter/cache-state defect were not promoted. Full-model PCC recovery remains unverified.
