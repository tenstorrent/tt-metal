# AutoDebug: wider SDPA compute grids read Q from the wrong cores

## Finding and confidence

**The candidate changes the SDPA compute-grid width while leaving Q on an 8-wide shard grid. The ordinary sharded-Q decode reader assumes that batch `b`'s Q shard lives on SDPA's output/reducer core, which the default planner places at `(b % compute_width, b // compute_width)`. It does not look up Q's actual shard coordinates.** At widths 10 and 11, the first disagreement is batch 8: Q is at `(0,1)`, but SDPA reads `(8,0)`, where this Q tensor has no shard. All 24 batches from 8 through 31 have a source-coordinate mismatch.

This is a direct producer/consumer contract contradiction and the leading diagnosis for the supplied `decode_0_8` failure. It explains the width-8 passing contrast and the exact first failing batch. Inspection cannot establish the bytes read from a core without a Q shard, predict the precise PCC, or prove that no additional numerical issue remains after correcting the mapping.

**The smallest comparison change is an ordered `sub_core_grids` compute map that puts the native 8×4 Q block first, followed by the remaining side and bottom strips.** It preserves QKV creation, Q/K/V shards, paged update, active physical cores, and the three-core sequence split for both 10×10 and 11×10. A matching wide-grid regression already uses this pattern. Q-only resharding to a width-10/11 prefix is another source-supported option.

**The output reshard/concat and Q8→Q4 reshape do not misidentify batch rows on the inspected path.** The SDPA output is DRAM interleaved, explicitly indexed by batch; the following conversion places those rows onto the fixed 8×4 grid correctly. It propagates attention computed from the wrong Q rather than introducing the original error.

Investigation was inspection-only: source/log reads and CPU-only arithmetic/hash checks; no hardware execution, device queries, resets, or source edits. This report is the only file written. Prepared checkout HEAD: `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`, with pre-existing local model changes and untracked experiment artifacts. The four relevant Python files match the recorded SHA-256 values in the 8×8, 8×10, 10×10 and 11×10 command manifests: `multichip_sdpa_geometry.py`, `multichip_packed_stack.py`, `multichip_qkv_create_candidate.py`, and `tt/multichip_decoder.py`. The manifests do not establish the loaded C++ binary's exact source revision; the lowered mapping below is proven for this prepared tree.

## Direct observations versus interpretation

Artifacts are under `models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/`.

| Compute grid | Chunk | Artifact observation | Source-derived allocation |
|---|---:|---|---|
| 8×8 | 256 | Result JSON: 4096 decode-row PCC records, minimum `0.9987144470`; 128 trace replays | 64 available, 64 active, 2 cores/batch |
| 8×10 | 256 | Same minimum and record count | 80 available, 64 active, 16 idle, 2 cores/batch |
| 11×10 | 256 | `sdpa_g11x10_k256_4k.log:21–33`: first assertion `('decode_0_8', 0.9846675395965576)`, threshold `0.995` | 110 available, 96 active, 14 idle, 3 cores/batch |
| 10×10 | 256 | Command manifest exists; no completed result JSON was present at inspection | 100 available, 96 active, 4 idle, 3 cores/batch |

The harness downloads the packed output and checks rows in ascending order ([multichip_packed_stack.py](models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_packed_stack.py), lines 291–296; threshold at 111–116). Thus the reported exception implies that rows 0–7 passed in that run. It supplies no results for rows 9–31. The exception measures the two-layer decoder-stack output, not an isolated SDPA tensor.

The job list also includes chunks 128 and 512 for these grids. Completed width-8 JSONs exist for both: each has 4096 decode checks; minima are `0.9987497926` (128) and `0.9986872673` (512), identical between 8×8 and 8×10. No completed width-10/11 result JSON was present for these chunks. The same Q-coordinate mismatch applies independently of chunk size; that is a source prediction, not a claim that those planned runs failed.

“Cache update succeeds” is requester-provided context consistent with the source audit below. The failed log does not independently certify cache contents: `--state-checks` is enabled, but its post-decode checks begin at harness line 299, after the failing assertion. A returned Python call alone is not a device-completion check.

Interpretation: the first wrong Q location occurs at exactly the observed first failing row. This ranks ahead of a speculative three-core numeric problem. Exact numerical recovery under an adapted mapping remains a hardware follow-up.

## Actual runtime geometry and selected path

The nominal “4K” is not a decode position of 4095. Harness lines 196, 205 and 223 derive:

- Prefix lengths `4095 + b % 3`, followed by continuation lengths `17 + b % 2`.
- First decode positions `4112..4115`; batch 8 starts at 4114. Across 128 steps, the largest position is 4242.
- Cache capacity `4256`, or 133 pages/request. The pooled K/V shape is `[4256,1,32,256]`: 32 requests × 133 pages, one local KV head ([multichip_decoder.py](models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py), lines 470–492).
- For chunk 256, the candidate extends the attention page table to 136 entries/request, retaining the unextended 133-entry update table (candidate lines 64–84). The SDPA factory sees `B=32`, `S=136*32=4352`, `St=136`, `page_block_size_t=1`.
- Q is BF16 TILE, logical `[1,32,4,256]`, padded `[1,32,32,256]`; `PNHt=1`, `DHt=vDHt=8`. Each batch occupies eight full stored tiles, or 16 KiB. Q/K input shards are `(32,256)` on `(0,0)..(7,3)`; V update shards are `(32,256)` on `(0,4)..(7,7)`. New K/V inputs are BF16, while the allocated K/V caches use `kv_dtype=bfloat8_b` (decoder line 85). SDPA reads those BF8 caches; its intermediate/statistics formats are BF16 (factory lines 497–508).
- The call explicitly supplies grid/chunk configuration, so the wrapper's *omitted-program-config* default cap of one core/head is irrelevant. The actual supplied config defaults to `max_cores_per_head_batch=16` ([transformer_nanobind.cpp](ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp), lines 42–47).
- This is ordinary causal paged MQA, not MLA/speculative mode: no explicit mask, sink, sliding window, cache geometry override, or modulo. Default argument lowering is in [sdpa_decode_nanobind.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp), lines 87–124; wrapper arguments in [sdpa_decode.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp), lines 150–172. Q/K/V order is preserved.
- `self.compute` requests HiFi4, `math_approx_mode=False`, FP32 destination accumulation and packer L1 accumulation (decoder lines 152–154). The supplied config is retained by `init_device_compute_kernel_config`, rather than replaced by architecture defaults ([compute_kernel_config.cpp](ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp), lines 21–40). SDPA's separate `exp_approx_mode` defaults to true (factory lines 221–223), identically in all compared cases. Four logical BF16 query heads select 16×32 internal half tiles (factory lines 510–513); stored Q remains full 32×32 tiles.

## Exact source contradiction

Relevant SDPA files:

- [Factory](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp)
- [Reader](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp)
- [Dataflow helpers](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp)
- [Validation/output specs](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp)

1. The candidate's `shard()` hardcodes width 8 (lines 43–59), then places Q there at line 61. Its independent `self.sdpa_grid` is passed at lines 122–124. There is no intervening relocation of Q to match that compute grid.
2. Factory lines 124–133 derive B/S from the page table. Lines 202–216 calculate `cores_per_batch = floor(grid_area/32)` here: 2 for 64/80 cores and 3 for 100/110. There are 32 reducer/output cores in every case.
3. `q_heads_parallel_factor` remains 1 because the MLA branch is inactive (lines 118 and 142–155); therefore column-major group indexing and its Q-grid check cannot apply (lines 265–267). There is no `sub_core_grids` in the reported call.
4. The selected sharded-Q assignment branch, lines 342–355, places reducer `b` at `(b % grid_size.x, b / grid_size.x)`, interleaving workers from the remaining core list. It does **not** enumerate the actual Q shard grid. Lines 393–406 convert those planned output cores to physical coordinates; lines 1048–1059 pass them to the reader.
5. Reader lines 203–208 select `all_output_noc_[cur_batch]`, then lines 226–238 call `read_q`. In the selected sharded branch, helper lines 575–610 set the source to the output core's coordinates; the root uses its own coordinates, other workers use the supplied root coordinates. The source address is raw `q_addr`, without a batch offset or shard lookup. The half-tile branch reads 1024 bytes from each stored tile and advances by 2048 bytes (lines 592–602). Correct tile strides do not repair a wrong source core.
6. The same batch index still selects positions (reader line 170) and page-table row (lines 281–287). Thus a mislocated Q is paired with batch `b`'s correct K/V cache rather than with the cache belonging to the Q user accidentally read.

Factory lines 923–951 select `reader_decode_all.cpp`, `writer_decode_all.cpp`, and [sdpa_flash_decode.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp). The latter consumes `cb_q_in` in QK multiplication at lines 408–430. There is no architecture condition on the faulty core-assignment/read mapping: the wider available grid exposes this software assumption on Blackhole.

Concrete logical coordinates; “absent” means this Q tensor has no shard there, not that the L1 bytes are zero:

| Batch | Actual Q location | SDPA width 8 | SDPA width 10 | SDPA width 11 |
|---:|---|---|---|---|
| 7 | `(7,0)` | Q7 at `(7,0)` | Q7 at `(7,0)` | Q7 at `(7,0)` |
| 8 | `(0,1)` | Q8 at `(0,1)` | absent at `(8,0)` | absent at `(8,0)` |
| 10 | `(2,1)` | Q10 | Q8 at `(0,1)` | absent at `(10,0)` |
| 11 | `(3,1)` | Q11 | Q9 at `(1,1)` | Q8 at `(0,1)` |
| 22 | `(6,2)` | Q22 | Q18 at `(2,2)` | Q16 at `(0,2)` |
| 31 | `(7,3)` | Q31 | Q25 at `(1,3)` | absent at `(9,2)` |

For batch 8, the actual reduction groups are:

- 8×8/8×10: root `(0,1)`, worker `(0,5)`; both fetch Q from `(0,1)`.
- 10×10: root `(8,0)`, workers `(8,4)`, `(9,4)`; all fetch Q from `(8,0)`.
- 11×10: root `(8,0)`, workers `(4,4)`, `(5,4)`; all fetch Q from `(8,0)`.

Validation lines 138–150 require HEIGHT_SHARDED Q or DRAM interleaved Q, but do not validate this normal-path coordinate agreement, shard height, or orientation against the chosen output-core map. A configuration passing validation is therefore insufficient evidence that this reader will address its Q shards correctly. The source defect is a hidden mapping assumption with missing validation, not an API-only assertion that wider grids are unsupported.

## Why the reshapes and output conversion are not the first error

The Q8→Q4 reshape explicitly preserves padded shape `[1,32,32,256]` (candidate line 28). In the reshape implementation, the codegen contract rejects explicit padded/volume-changing reshapes ([reshape_codegen_supported.cpp](ttnn/cpp/ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.cpp), lines 397–403); this logical-volume change reaches the tile view fallback ([reshape.cpp](ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp), lines 778–787). It changes which head rows are logical while retaining each user's 32-head physical stride. It does not compact eight-head or four-head users into adjacent portions of one shard. Norm/RoPE precede the explicit Q reshard at line 61; the identical preparation is present in the passing cases.

The SDPA call omits output `memory_config`. Wrapper line 163 supplies `DEFAULT_OUTPUT_MEMORY_CONFIG`; [operation.hpp](ttnn/api/ttnn/operation.hpp), line 149 default-constructs it, and [memory_config_impl.hpp](tt_metal/impl/tensor/spec/memory_config/memory_config_impl.hpp), lines 40–42 default to INTERLEAVED/DRAM/no shard. Output-spec lines 624–634 retain logical `[1,32,4,256]`, with the corresponding tiled padding.

In [writer_decode_all.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp), line 426 sets `out_tile_id = cur_batch * out_chunk_tiles`, here `8*b`. The selected MQA/interleaved branch at lines 506–515 calls `write_tiles_to_memory`; helper lines 459–468 write pages `8*b..8*b+7`. Moving the reducer changes where computation happens, not which batch row it writes in DRAM.

Then candidate lines 127–132:

1. `to_memory_config(out, shard(0))` selects the interleaved-to-sharded path ([to_memory_config_op.cpp](ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp), lines 340–351), placing eight tiles per core in the fixed 8×4 shard order. The interleaved-to-sharded [factory](ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp), lines 335–406 advances source start indices by each shard's tile count.
2. Concat receives that same 8×4 core grid explicitly. Its [subcoregrid factory](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp), lines 78–92 enumerates actual input cores. Its [reader](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp), lines 55–93 writes batch row `q`, advancing to the next input core after eight head-dimension tiles. It produces logical `[1,1,32,1024]`, one concatenated four-head vector per batch ([concat output specs](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp), lines 90–110).
3. The L1 interleaved conversion and `:n` slice preserve those batch rows. The matching gate rows are applied at candidate line 140, followed by output projection/reduction and residual/MLP in decoder lines 1198–1199 and 615–621.

Both buffer distribution and concat use range-first, then x-fast row-major enumeration ([buffer_distribution_spec.cpp](tt_metal/impl/buffers/buffer_distribution_spec.cpp), lines 168–173; [core_coord.cpp](tt_metal/common/core_coord.cpp), lines 517–527 and 604–625). At batch 32, `shard(0)` is a single rectangle, so no range-order ambiguity exists. Changing only the post-SDPA reshard cannot repair attention already calculated using the wrong Q.

## Preserve native QKV and paged-update contracts

Native creation at candidate lines 18–25 uses `num_heads=8`, `num_kv_heads=1`, `overlap_qk_coregrid=True`. [QKV wrapper](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.cpp), lines 34–46 retains the requested grid but rebuilds the output memory config. Its [output specs](ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp), lines 127–156 pad Q and KV heads independently to 32; Q/K/V initially share the native 8×4 grid. The candidate subsequently places V at y=4 and retains K at y=0.

For fused paged update, [validation](ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp), lines 180–209 requires sharded inputs, full head-dimension shard width, ROW_MAJOR orientation and BF16/FLOAT32 inputs; lines 237–246 require disjoint K/V core sets with equal counts. The existing K8×4/V8×4 placement satisfies this. Its [factory](ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp), lines 629–643 enumerates each actual grid and pairs ordinal `i`; lines 684 and 694 set `my_batch_idx=i` for both inputs. The [update reader](ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp), lines 101–124 reads `positions[i]` and page-table row `i`; the [writer](ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp), lines 85–127 computes the physical cache block and row from that same index. None depends on the later SDPA compute width.

This explains how correct cache update and incorrect SDPA can coexist. Keep `head_memory`, K's `shard(0)`, V's `shard(4)`, page tables, offsets, and positions intact for the geometry comparison.

## Minimal legal adapted mappings to compare

### Preferred: retain all tensor shards and adapt compute-core order

For batch 32, width `w` in `{10,11}`, height 10, add the following to the existing SDPA program config. This is a proposal, not an applied edit or a hardware-verified fix:

```python
compute_cores = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(w - 1, 3)),
    ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(w - 1, 9)),
])
program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(w, 10),
    sub_core_grids=compute_cores,
    q_chunk_size=32,
    k_chunk_size=256,
)
```

Use the **ordered list** as shown, without merging into one bounding rectangle or converting to an unordered/sorted set. The ranges are nonoverlapping and cover all selected cores: `32+8+60=100` for width 10; `32+12+66=110` for width 11. Their first 32 enumerated cores are exactly the current Q8×4 grid in batch order.

Factory lines 182–189 retain `sub_core_grids` and require its count to equal `w*10`. Lines 333–341 use the explicit core enumeration, taking the first B cores as reducers/output cores and subsequent cores as workers. This makes `read_q`'s assumed Q source correct without changing tensor data. It retains 96 active cores, 3 cores/batch, and the same active physical core set as the corresponding default rectangle; batch/group placement changes. The current DRAM output and fixed post-SDPA concat preparation remain appropriate.

This pattern is directly corroborated by [test_sdpa_decode_sharded_q_block_wide_grid](tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py), lines 326–352, whose comment explicitly preserves the model's 8×4 Q block while using a wider grid. That test uses GQA rather than this exact MQA case, so it is contract evidence, not a hardware pass for this repro.

Order is an actual source contract: the Python list overload retains vector order ([tensor.cpp](ttnn/cpp/ttnn-nanobind/tensor.cpp), lines 617–622); CoreRangeSet stores it (core_coord.cpp lines 156–158), enumerates it (604–625), and includes range order in equality/hash (482–493 and 697–703). The three-range map is not silently equivalent to a merged single-range map for planning/cache purposes.

This proposal is deliberately scoped to batch 32. For a different batch count, make the first B compute cores agree with that batch's actual Q core list; do not assume the three-range split automatically generalizes.

### Alternative: keep default compute order and adapt Q alone

Replace only Q's fixed conversion at candidate line 61 with a HEIGHT_SHARDED/L1/ROW_MAJOR config, shard `(32,256)`, containing these ranges in order:

| Compute grid | Q range 1 | Q range 2 | Batch mapping |
|---|---|---|---|
| 10×10 | `(0,0)..(9,2)` | `(0,3)..(1,3)` | `(b % 10, b // 10)` |
| 11×10 | `(0,0)..(10,1)` | `(0,2)..(9,2)` | `(b % 11, b // 11)` |

Q is already interleaved after norm/RoPE, so this can be its first explicit post-RoPE sharding conversion. Alternatively reshard only Q after the cache update at line 114. Leave the shared `shard()` helper and its K/V/output consumers unchanged. Keep SDPA output DRAM interleaved; converting it back to `shard(0)` before concat remains correct. If choosing a sharded SDPA output later, its placement must agree with the planner too; that is a separate experiment.

Direct same-width HEIGHT→HEIGHT conversion is also source-supported: to_memory_config lines 283–313 route to reshard; [reshard selection](ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation.cpp), lines 60–65 selects `ReshardSameWidthFactory<true>`. Its [factory](ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.cpp), lines 64–74 obtains actual local/remote core lists and lines 123–165 advances through source shards by ordinal. This is not an unsupported irregular-grid dismissal.

A further isolation control is BF16 TILE **DRAM interleaved Q**, with DRAM interleaved output. Validation lines 138–161 require DRAM for unsharded Q and reject ROW_MAJOR interleaved Q. The unsharded `read_q` branch uses `q_batch_offset=8*b` through its TensorAccessor (helper lines 619–641), avoiding the source-coordinate assumption. It also changes the planner's physical grouping branch, so the ordered compute map is the cleaner first comparison.

Changing chunk size, limiting cores/head, or changing output memory alone does not relocate the Q shards and does not correct the default width-dependent Q lookup.

## Reduction/buffer and completion audit

For actual position 4114 and chunk 256, [rt_args_common.hpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp), lines 61–88 rounds `cur_pos+1` to 4352, i.e. **17 chunks**, not 16. The reader, writer and compute all call this helper. The exact local ranges and tree edges are:

| Cores/batch | Local chunk ranges by core ordinal | Reduction path |
|---:|---|---|
| 2 | core 0 `[9,17)`, core 1 `[0,9)` | 1→0 at round 0 |
| 3 | core 0 `[12,17)`, core 1 `[6,12)`, core 2 `[0,6)` | 2→1 at round 0, then 1→0 at round 1 |

The tree helper handles non-power-of-two orphan attachment explicitly ([sdpa_decode_device_operation.hpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.hpp), lines 35–80). All local cores and children are active here. Chunks cover `[0,17)` exactly once with no empty-core branch. For chunk 128, first-step work is 33 chunks (later 34); for 512 it is 9. None creates an empty local worker in these 2/3-core configurations.

Buffer state ledger for the selected BF16 half-tile path:

| Stage | Producer/consumer accounting |
|---|---|
| Positions | Reader reserves/pushes separate one-stick CBs `c_8` and `c_15` (reader lines 127–151); writer consumes `c_8` (writer 122–147), compute consumes `c_15` (compute 173–184). They do not race to pop one shared CB. |
| Q | Reader reserves/pushes 8 half tiles after a read barrier (helper 587–617). Compute waits on 8 Q tiles (compute 270), uses them throughout, then pops 8 (734). The count is coherent, but the source core is wrong in the reported wide layouts. |
| Local K/V | Each chunk carries 64 full tiles per K and V, `8*8`; source factory double-buffers 128 tiles (factory 480–484). Per-core chunk loop counts are 8/9 or 5/6/6 as above. The last chunk is fully read and causally masked; the extended table owns its otherwise inactive tail entries. |
| Tree send | A nonroot produces O=8 half tiles and M/L=1 half tile each. Writer waits on those counts (370–372), writes L/M/O into its parent's staging block, completes writes before signaling (385–409), then pops 8/1/1 (412–414). |
| Tree receive | Receiver waits for that round's semaphore nibble, reads the round's L/M/O block, reserves/pushes 1/1/8 tiles (290–357). Compute consumes/combines that child (compute 609–646) before finalizing or forwarding. Core 1 in the three-core case consumes core 2 before forwarding to root. |
| Staging bounds | One block is `(8+2)*1024=10240` bytes. Two-core staging allocates one block and uses round 0; three-core staging allocates two blocks and can use round 1. Factory lines 493–495 size it accordingly. No staging-count/bounds contradiction was found for this repro. |
| Final output | Root writes 8 full output tiles; writer waits then pops 8 and completes its writes (writer 426–429, 506–516). |

This focused audit does not prove every low-level arithmetic primitive, but it finds no concrete two-versus-three-core queue mismatch that outranks the wrong-Q read. A third-core numerical hypothesis would also need to explain why rows 0–7 survive the same three-core tree while the first coordinate mismatch fails.

First-draft review: three independent xhigh reviewers rechecked the mapping/workload proposal, reshape/output chain, and native QKV/update contracts against source. CPU-only enumeration additionally verified all four default core assignments, complete chunk coverage, ordered-map disjointness/coverage, unchanged active physical sets, tree parent/child correspondence and staging bounds. Report snippet syntax and local source links were checked. No hardware claim was added from these checks.

Asynchronous boundaries: the harness runs two warm forwards and restores state (266–269), captures a forward (272–274), restores again, then refreshes inputs and executes the trace with `blocking=True` (287–293) before downloading and asserting PCC. Reaching the assertion therefore follows actual replay completion/readback; this is not inferred from a warm forward merely returning. The cache writer itself issues its DRAM-write completion barrier at lines 173–175. These facts support investigating incorrect data mapping rather than treating a host-side cache call returning as proof of its contents.

## Other potential issues and remaining uncertainty

| Candidate | Evidence and contrast | Adjudication / smallest decisive control |
|---|---|---|
| Wrong Q source core | Direct contradiction between fixed Q8×4 and factory/read_q; predicts all width-8 passes and first failure at batch 8 | **Leading diagnosis.** Ordered compute map repairs exactly this boundary while preserving the active physical set and three-core work count. |
| Three-core tree arithmetic or Blackhole numeric primitive | Width 10/11 use 3 cores while width 8 uses 2; this is a real second difference | No demonstrated source violation for this shape. Both early and later batches use the same tree, and its queue ledger is coherent. If adapted mapping still fails, compare caps 2 and 3 with valid mappings and inspect raw SDPA, then target-specific math phases. Hardware confirmation required. |
| Output reshaping/resharding permutes batch | Would require a disagreement between writer page IDs, buffer core order and concat source order | Inspected contracts agree; moved out of the headline. Raw DRAM SDPA versus post-concat downloads would independently confirm. |
| Q8→Q4 view compacts users | Would require the explicit padded stride to change | Source takes a padded tile view with identical physical stride. Shared by passing runs; no contradiction found. |
| Page-table/cache-update corruption | Potentially affects long/ragged contexts | Current updater consumes actual K/V grids by matching ordinal; update table is 133 entries and attention tail is separately extended to 136 for chunk256. No compute-width dependency found. The failed run does not reach final state-content assertions, so do not report them as passed. |
| Test coverage gap | `sdpa_test_utils.py:37–45,530–533` constructs Q with an 8-wide helper independently of compute width; common paged tests exercise 8-wide grids | Existing generic tests are not proof that arbitrary Q/compute layouts agree. The explicit wide-grid subcore test at 326–352 documents the intended adaptation. Do not relax PCC to conceal incorrect user mapping. |
| Native QKV grid-generalization risk | Regular creation factories enumerate output cores from an origin-based bounding box (sharded factory 124 onward; interleaved factory 263 onward), while subcore selection is driven by input-grid properties (QKV wrapper 24–32) | Not the reported cause: creation stays native8×4 in passing/failing cases. Avoid changing native output placement as the first intervention; separate output-grid generalization would need its own contract audit. |

No source-level explanation here predicts the exact value `0.9846675396`; bytes at a Q address on a non-Q core are unspecified and later projections/residuals alter the error. No runtime pass is claimed for 10×10, adapted 11×10, or any proposed control.

## Follow-up for the hardware-owning caller

1. Compare 10×10 and 11×10 with the ordered three-range compute map above, unchanged native Q/K/V and output path, chunk256, the same reference and the original `0.995` gate. Record the ordered core list and effective 96 active cores, not only nominal grid area.
2. Check raw per-batch SDPA output before output reshard/concat and then after concat. Include batches 7, 8, 10, 11, 22 and 31, with explicit batch/core correspondence. This tests the predicted boundary without inferring pairings from unindexed samples.
3. Complete all 128 steps and final K/V state checks. Compare with the Q-only adapted mapping or DRAM-Q control if needed. Record any added data-movement cost separately when comparing performance; the preferred ordered compute map adds no Q conversion.
4. If a residual failure remains with valid Q mapping, isolate the 2→3 core-count change using `max_cores_per_head_batch` on a valid mapping before attributing it to Blackhole arithmetic. Retain chunk128/256/512 as distinct cases.

For a reusable callee improvement, either validate the expected batch→Q-core relationship explicitly or pass actual Q-source coordinates separately from reducer/output coordinates. Merely changing output tensor metadata would leave the raw Q NoC reads incorrect. The caller configuration change above is the smallest intervention needed to make this geometry experiment meaningful.
