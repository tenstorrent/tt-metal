# AutoDebug: Qwen3.8-27B native-context BF16 KV SDPA allocation failure

## Finding

**The BF16 KV candidate retains a decode SDPA geometry whose static circular buffers overlap live L1 by exactly 18,304 bytes.** The model promotes both KV caches from BFP8 to BF16 without adapting the explicit chunk256 / 32-cores-per-head configuration. With the native 262,144-token page table, the factory requests **971,904 bytes of static CB storage per core**. Adding the Blackhole reserved base of **111,744** produces **1,083,648**, exactly the endpoint in the failure. The recorded live-L1 frontier is **1,065,344**. The allocator is reporting a real capacity conflict; no kernel corruption, stale reset, or numerical failure is needed to explain it.

The smallest practical model-side repair candidate is **BF16-only decode chunk128, retaining the existing mapped-capacity divisor guard and 32-core cap**. Its predicted endpoint is **813,312**, leaving **252,032 bytes against the recorded frontier**. A core-only repair is also legal: cap30 is the nearest smaller cap that fits this snapshot, but leaves only **2,176 bytes**. Cap31 still fails. Cap16 leaves 145,536 bytes and retains chunk256. These are source-proven geometry/budget predictions, **not hardware-verified correctness or performance results**. Preserve native context, page ownership, the current BF4 in-place KV reset repair, and unrelated projection geometry.

**Evidence qualification:** the supplied datatype layer-smoke artifacts for lengths33/4097 actually allocate capacities256/4352, not8192. An8192-capacity BF16 smoke, as described in the request, would also have a much smaller table and fits the same explanation, but that specific observation was not independently located. Neither small-capacity smoke establishes native-context admission.

## Scope, provenance, and direct observations

This investigation only read source/artifacts and ran standard-library arithmetic/hash checks. No model/TTNN execution, device query, hardware reproduction, reset, build, or source edit was performed. Only root AUTODEBUG.md was written. Three xhigh subagents independently traced the CB budget, model geometry/evidence, and kernel legality. AutoDebug was not relaunched.

The failure artifact records HEAD af283dd6d375f8f7ffe057b2c1e271f9e4aef3e9, branch exp26-qwen38-bringup, four Blackhole p300c devices, mesh1×4, B1, configured native context262144, and a readiness input of203 tokens. All13 recorded model-source SHA256 values and the candidate-runner hash match the current inspected files. The input has not been processed when the failure occurs. The artifacts do not independently fingerprint the loaded C++ binary; the current factory's exact byte match strongly corroborates this path.

Direct observations:

| Evidence | What it establishes |
|---|---|
| [Failure JSON](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/failures/kv_bf16_initial.json), [failure log](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/failures/kv_bf16_initial.log) | All64 layers load. build_generator → prepare_decode → first _decode → SDPA program compilation fails. Endpoint1,083,648 versus live frontier1,065,344, core range(0,0)–(10,9). No candidate accuracy/timing result. |
| [Baseline result](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/results/baseline_bfp4_lofi.json) | BFP4 weights / **BFP8 KV**, full64 native-context candidate status pass. The filename's bfp4 refers to weights. |
| [Repaired BFP4-KV result](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/results/kv_bfp4.json) | Full64 native-context status pass, after the separate reset repair. |
| [BF16 length33 smoke](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/kv_bfloat16_s33.json), [BF16 length4097 smoke](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/kv_bfloat16_s4097.json) | Real layer3, B1, four decode steps, capacities256/4352, effective chunk256/core cap32. These are multichip-only runtime/replica/replay smokes, not full-model accuracy measurements. |

The last two artifacts predate the current decoder hash. Their recorded decoder cdfeb049… matches the saved after_bfp8_geometry snapshot; the decode SDPA configuration is unchanged, while the intervening change concerns BF16 CCL projection geometry. Do not claim identical full-model live allocations for those reduced tests. Their actual capacities are explicit at JSON line2 and derive from [run_multichip_decoder.py:78](models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py#L78): round up seq+steps to256. [smoke_precision_kv.sh:7](models/autoports/qwen_qwen3_8_27b/tests/smoke_precision_kv.sh#L7) supplies no8192 context override.

The explanation below is an interpretation derived from these observations and source. Proposed fixes and their numerical/performance effects remain unmeasured.

## Lowered path and earliest missing adaptation

1. [precision.py:70](models/autoports/qwen_qwen3_8_27b/tt/precision.py#L70) maps kv_cache_dtype to kv_dtype. [multichip_decoder.py:25](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L25) inherits sdpa_cores32 from [optimized_decoder.py:41](models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py#L41). The apparent .get(...,16) fallback at the callsite is therefore **not** the effective default. The constructor sets SDPA LoFi and sdpa_fp32=False at lines51–52 and applies supplied policy at84–85. Existing dtype-dependent projection repairs at89–108 do not adapt SDPA to BF16 KV.
2. [generator.py:27](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L27) accepts native262144. Its generated table at38–40 has shape[1,8192], INT32 row-major DRAM, replicated across TP ranks. Each KV cache on each rank has shape[8192,1,32,256], tiled DRAM. The physical page size remains32 tokens. The large KV allocation is in DRAM; its dtype separately controls the size of the SDPA K/V **L1 staging buffers**.
3. Preparation preserves the table shape and physical pool geometry. [generator.py:231](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L231) builds a valid temporary map with the original shape; owned caches remain bound. preparation_reset sets positions to0, then line247 calls _decode before prefill warmup or trace capture. Model.decode_forward reaches the first full-attention layer after the initial linear-attention layers.
4. [multichip_decoder.py:741](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L741) projects, splits local Q6/KV1, normalizes Q/K, applies RoPE, and permutes to decode layout. Q is tiled BF16, logically[1,1,6,256], padded[1,1,32,256]. K/V updates use temporary height-sharded[32,256] updates and paged_update_cache. Those update tensors are distinct from the tiled DRAM K/V caches passed to SDPA. Q is interleaved on this path; MLA/sharded-Q special handling is inactive.
5. The model selects chunk256, verifies power-of-two ≥32, then halves only until chunk divides the actual mapped row capacity (lines774–784). For8192×32 this remains256. It supplies the complete11×10 compute grid, q_chunk_size32, k_chunk_size256, cap32, scale1/16, LoFi, fp32_dest_acc_en=False, and exp_approx_mode=False. There is no explicit attention mask, attention sink, sliding window, cache geometry override, cache modulo, or speculative mode. The output uses the default interleaved DRAM configuration and Q's BF16 dtype.
6. [Paged API:114](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp#L114) only constructs defaults when program_config is absent; this caller supplies it. Lines133–166 check fixed chunk power-of-two/multiple32 and forward it unchanged. No operand swapping, dtype conversion, or automatic L1 budget selection intervenes. [Device output:637](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp#L637) constructs output storage from the spec at588–634 before the descriptor uses it. Causal paged validation checks layout/shape and chunk multiple32, but the capacity-divisibility check at455–466 applies only to non-causal mode. The model's existing divisor guard must remain.
7. [Factory:122](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp#L122) derives S=page_table.padded_shape[-1]×block_size=262144, not current_pos+1. It derives B1, Hkv1, PNHt1, DHt=vDHt8, Sk_chunk_t8. Lines202–216 select32 static worker cores for the head. The selected files are reader_decode_all.cpp, writer_decode_all.cpp, and sdpa_flash_decode.cpp (factory923–950).

The earliest missing adaptation is model-side selection of an explicit geometry after promoting KV dtype. The page-capacity guard addresses read ownership, not L1 capacity. The normal factory has no earlier live-capacity admission/resizing step; its special speculative-only check at722–756 is not reached and in any case compares nominal L1 minus base rather than live tensors. The final generic overlap validator correctly refuses the resulting layout.

## Exact per-core CB budget

Let F denote [sdpa_decode_program_factory.cpp](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp). Tile counts come from F479–495, formats and tile geometry from497–559, and actual allocations from637–720.

**Half tiles are essential to the arithmetic.** F513 selects16×32 tiles because causal BF16 Q has **six logical heads**, even though its head-count axis is padded to32; D remains256. Q, masks, intermediate outputs and softmax/reduction statistics therefore cost1024 bytes/tile. K/V retain full32×32 tiles: BF16=2048, BFP8=1088, BFP4=576 bytes. Final c20 output tiles are full BF16=2048. Do not multiply every buffer by2048 or infer half-tile selection from padded heads.

| CB(s) | Count and byte size for failing invocation | Bytes |
|---|---|---:|
| c0 Q | 8 half tiles ×1024 | 8,192 |
| c1 K | Double buffer:2×8 sequence tiles×8 head-width tiles×2048 | 262,144 |
| c2 V | Same as K | 262,144 |
| c3 causal mask | 8 half tiles×1024 | 8,192 |
| c5 scale; c12 identity | 1 half tile each | 2,048 |
| c6,c7,c27,c28,c29,c30,c31,c21,c22,c17,c18 | Eleven statistic buffers,1 half tile each | 11,264 |
| c8 writer position; c15 compute position | Separate64-byte DRAM-aligned row sticks | 128 |
| c9 page table | Full8192-entry INT32 row,8192×4 | 32,768 |
| c10 row-major-Q staging | 8 half tiles; still allocated although Q is tiled | 8,192 |
| c11 column identity | 1 full BF16 tile | 2,048 |
| c24 QK intermediate | 8 half tiles | 8,192 |
| c25,c26,c23,c16 | Four output-size buffers,8 half tiles each | 32,768 |
| c19 reduction staging | (8 output+2 statistics)×(32−1) half tiles | 317,440 |
| c20 final output | 8 full BF16 tiles | 16,384 |
| **Static CB payload** | No optional c4/c13/c14; no aliases to global L1 here | **971,904** |
| **Static base** | Blackhole default unreserved base | **111,744** |
| **Static endpoint** | Base+payload | **1,083,648** |

Position storage is not4 bytes: generator positions are row-major DRAM; [Buffer::aligned_page_size:842](tt_metal/impl/buffers/buffer.cpp#L842) rounds to Blackhole's64-byte DRAM alignment ([NoC parameters:379](tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h#L379)). F236 and657–664 use that aligned size twice for distinct consumers. c9 is a complete per-user row; for interleaved tables it is not multiplied by batch. All listed sizes are multiples64, so [CB placement:2390](tt_metal/impl/program/program.cpp#L2390) adds no inter-buffer alignment holes.

The static base is independently source-derived: Blackhole MEM_MAP_END=41088 from [dev_mem_map.h:111](tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h#L111), plus69×1024 kernel-config bytes and64-byte alignment at [bh_hal_tensix.cpp:43](tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp#L43) and68–69 yields111744. Program placement starts at the persistent arena high-water mark ([program.cpp:474](tt_metal/impl/program/program.cpp#L474)); the exact endpoint is consistent with no extra persistent-arena prefix for this program. A future changed arena base must be included, not assumed constant.

For this Q/head geometry, no optional inputs, equal K/V tile size T, chunk K, core count C, table width W, batch B, the budget is:

    endpoint = base + 80,896
             + 2 × align64(4×B) + align64(4×W)
             + 32 × (K/32) × T
             + 2 × (K/32) × 1,024
             + (C−1) × 10,240

Here C=min(110,cap×B)/B with integer division and one KV head. The80,896 term includes all chunk/core-independent tiled buffers. This formula is specific to the selected half-tile, interleaved, non-MLA path; do not reuse it for full-tile Q, sharded output, optional masks, or speculative decoding.

F626–629 assigns **every CB to the full11×10 core_grid**, even though only32 cores are statically active and only one gets data at position0. This explains the error's full(0,0)–(10,9) range. Runtime early returns do not shrink compiled buffers. Reducing cores helps because it shrinks c19, not because the remaining idle cores disappear from CB placement.

### Context and dtype contrasts

| KV / mapped capacity | c9 aligned bytes | Endpoint, chunk256/cap32/B1 | Margin against recorded1,065,344 |
|---|---:|---:|---:|
| BF16 / native262144 | 32,768 | **1,083,648** | **−18,304** |
| BFP8 / native262144 | 32,768 | 837,888 | +227,456 |
| BFP4 / native262144 | 32,768 | 706,816 | +358,528 |
| BF16 / reported8192 contrast | 1,024 | 1,051,904 | +13,440 |
| BF16 / actual4352 layer smoke | 576 | 1,051,456 | +13,888 |
| BF16 / actual256 layer smoke | 64 | 1,050,944 | +14,400 |

The last-column frontier is a **controlled arithmetic reference**, not a measurement from the passing runs. Reduced layers can also have a higher frontier because fewer L1 tensors are live. Native versus8192 adds31,744 bytes of c9 alone, enough to cross this recorded boundary. BFP8→BF16 adds245,760 bytes in K/V CBs; BFP4→BF16 adds376,832. No change to full64 layer count in the CB formula is necessary; layer count affects the separate live-allocation side of the comparison.

### Allocation timing and synchronization

Persistent model/collective/request allocations already exist before prepare_decode. Reset work and preceding decoder operations may have been enqueued. SDPA output storage is allocated before descriptor creation; here it is DRAM, so it does not itself explain the L1 frontier. Existing residual/projection/collective temporaries and persistent L1 storage remain relevant. The artifact does not identify which tensor owns address1,065,344; do not attribute it to a particular cache, output, leak, or all64 layers.

[program.cpp:2515](tt_metal/impl/program/program.cpp#L2515) reads the current occupied compute-L1 frontier; lines2617–2624 compare it with the static region end. [compile_and_allocate:3426](tt_metal/impl/program/program.cpp#L3426) allocates/checks CBs before successful enqueue of this SDPA program. This is a synchronous host allocation rejection, not a deferred report of an executing SDPA stall. Generator cleanup synchronizes at [generator.py:357](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L357), but that later boundary is not the origin of the exception.

Any future automatic budget selector must include live inputs, persistent allocations, and guaranteed subsequent L1 output/workspaces at the point they become live. Do not compare only to nominal1.5MiB hardware L1 or take an early free-space snapshot and omit later allocations. The existing final validator uses live state and also rechecks cached programs; no stale capacity selector was found on this explicit-config path.

## Legal minimal interventions

| BF16 native/B1 intervention | Endpoint | Margin vs recorded frontier | Assessment |
|---|---:|---:|---|
| chunk256, cap31 | 1,073,408 | −8,064 | **Still fails**; one c19 block saves only10,240. |
| chunk256, cap30 | 1,063,168 | +2,176 | Nearest cap-only fit; too little margin to generalize beyond this snapshot. |
| chunk256, cap28 | 1,042,688 | +22,656 | Legal non-power-of-two cap; modest headroom. |
| chunk256, cap24 | 1,001,728 | +63,616 | Legal cap-only candidate. |
| chunk256, cap16 | 919,808 | +145,536 | Conservative cap-only candidate; less sequence parallelism. |
| **chunk128, cap32** | **813,312** | **+252,032** | **Recommended first controlled repair**; power-of-two chunk halves staging, retains parallelism. |
| chunk64, cap32 | 678,144 | +387,200 | Legal fallback, more chunk/softmax iterations. |

Chunk128 saves262,144 bytes of K/V staging plus8,192 mask/QK bytes. It is the next smaller legal fixed chunk:192/224 are rejected by the public wrapper's power-of-two requirement. Page32 imposes no restriction that chunk equal a page; the reader maps each sequence tile separately and a128-token chunk spans four owned pages. Changing q_chunk_size does not change this decode factory's query geometry: it uses PNHt from Q instead. Fidelity alone does not shrink these fixed-chunk buffers. k_chunk_size0 is rejected by the model guard; even bypassing it, current sdpa_fp32=False gives an eight-tile/256-token dynamic CB capacity and therefore no memory saving. Setting fp32=True would lower the dynamic cap, but would combine unrelated policy changes and is unnecessary.

Core caps need not be powers of two. [Tree helper:36](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.hpp#L36) explicitly handles non-power-of-two orphan senders at65–78, and F245–258 uses ceil(log2 C). The scalar work divider also uses integer quotient/remainder, not a power-of-two mask. Cap30 is legal at this boundary; it is a capacity-margin issue, not an unsupported tree shape. Native full context has1024 chunks at K256:32 cores do32 local iterations,30 cores do34/35,16 cores do64. K128/cap32 has2048 chunks and64 local iterations. BF16 intermediate/statistics storage stays BF16 with all candidates, so changed reduction ordering and iteration count require accuracy checks, especially at long context.

Prefer resolving the BF16-specific cap at the shared decode callsite from the **actual bound K/V dtypes**, before the existing divisor loop. Generator.__init__ accepts external caches without checking that their dtype matches the model policy, so a policy-only fix can miss caller-owned BF16 pools. For equal-dtype owned caches, a setup policy change after final overrides and before copying _batched_decode_impl/_prefill_impl ([multichip_decoder.py:219](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L219)) is also sufficient. Changing only the parent policy after construction misses the copied batched view. Preserve explicitly supplied smaller legal choices; record requested and effective chunk/core cap. Keep the original power-of-two validation before capping, so an invalid requested chunk is not silently made valid. Apply this only to the BF16 KV decode geometry and retain BF8/BF4 configurations. A small policy/callsite change is sufficient; modifying the allocator or reducing advertised context is not required.

Keep the **existing mapped_tokens divisor loop** after any dtype cap. Let M=32×logical table-row length, K divide M, and0≤p<M. Then ceil((p+1)/K)×K≤M, proving full-chunk reads do not index beyond the supplied table entries, even for nonaligned prompts and final positions. This is a table-index bound, **not validation of the physical page IDs stored in those entries**. The current native generated/preparation map contains valid IDs; caller-owned sparse maps can contain−1 and must meet the existing valid-read ownership contract for all touched entries. Causal masking occurs after reads and cannot replace that contract.

Preservation is stronger than the index bound: power-of-two divisors are nested, so applying a128 cap before the loop gives new_effective_chunk=min(old_effective_chunk,128). For every position, its rounded read prefix is therefore a subset of the old prefix. A previously valid full-chunk mapping stays valid; a core-cap change leaves the union of pages read unchanged. Native262144 is divisible by128; smaller nonaligned-context pools retain their256-rounded allocation without changing logical prompt lengths. Never fabricate page IDs from alignment padding or shorten the native table during preparation just to pass compilation.

Keep [generator.reset:366](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L366) unchanged: BF4 uses same-dtype ttnn.fill(...,output_tensor=value) to preserve buffer addresses and trace bindings; other types use the existing zero/copy path. Its separate repair is present in the failure-matched source. No evidence links it to this exact CB overlap.

### All supported batches

The generator accepts B1…32, not only the named B1/8/32 benchmark points. On110 cores with Hkv1:

| Batch | Effective cores/head at cap32 | BF16 endpoint K256 | BF16 endpoint K128 |
|---|---:|---:|---:|
| 1,2,3 | 32 | 1,083,648 | 813,312 |
| 4 | 27 | 1,032,448 | 762,112 |
| 7 | 15 | 909,568 | 639,232 |
| 8 | 13 | 889,088 | 618,752 |
| 16 | 6 | 817,408 | 547,072 |
| 17 | 6 | 817,536 | 547,200 |
| 32 | 3 | 786,816 | 516,480 |

B17–32 positions need128-byte sticks each instead of64, accounting for the B16→17 difference. Table-row width remains8192 per user. Cap30 alters only B1–3 relative to cap32; B4+ is grid-limited already. Batched production paths have different live tensors, so these endpoints do not prove admission for every batch. Do not restrict supported batches to powers of two as a workaround.

Separate this per-core L1 calculation from full-model DRAM feasibility. One native BF16 K/V pair costs256MiB per layer/device at B1; the16 full-attention layers cost4GiB, or128GiB at B32. The latter exceeds the recorded device DRAM capacity even before weights and workspaces. Use reduced real layers for native-map B32 coverage and a measured feasible working set/external pool for full64 batched coverage; do not mistake an infeasible B32×262144 full64 pool for an SDPA repair failure. [kv_memory_forecast.json](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/kv_memory_forecast.json) independently records16,384 BF16 KV bytes/token/device across the stack, and warns that even its B32≈32768 peak is only an estimate. Preserve advertised native context and valid batch support within the memory contract; do not claim all32 users can simultaneously own maximal-length caches.

## Kernel work and buffer ledger

These are instantiated source calculations, not device traces. [rt_args_common.hpp:35](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp#L35) rounds p+1 to a whole chunk and assigns chunks in reverse core order. Reader, writer and compute use the same helper and return consistently when no work is assigned.

| Position p / geometry | Rounded tokens; chunks; active data cores | Concrete ownership |
|---|---|---|
| p0 or33, K256/C32 | 256;1;1 | Core0 reads pages0…7; other31 assigned workers have no chunks. |
| p0 or33, K128/C32 | 128;1;1 | Core0 reads pages0…3. |
| p4097, K256/C32 | 4352;17;17 | Core0 owns chunk16/pages128…135; core16 owns chunk0. |
| p4097, K128/C32 | 4224;33;32 | Core0 owns chunk32/pages128…131; core31 owns chunks0,1. |
| p262143, K256/C32 | 262144;1024;32 | Core0 owns chunks992…1023,32 chunks. |
| p262143, K128/C32 | 262144;2048;32 | Core0 owns chunks1984…2047,64 chunks. |

For K256, the reader reserves64 full K tiles, reads/barriers/pushes64, and similarly64 V tiles per local iteration, against128-tile double-buffered capacities. K128 uses32 tiles per iteration against64-tile capacities. Q is retained as8 half tiles. QK/mask has8 versus4 half tiles; PV produces8 half tiles and each max/sum statistic is one. Fixed chunk subblocks divide these dimensions exactly: with fp32=False, QK is1×8 versus1×4, PV output is1×8 and its K dimension is8 versus4. The matching compute helpers consume the corresponding K/V tiles; no tail count mismatch is introduced by128.

Relevant source: [reader:187](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp#L187), its full-chunk read calls312–362, [dataflow helpers:755](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp#L755), [compute:372](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp#L372), and [writer:254](ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp#L254). The reader copies positions into distinct c8/c15 mailboxes, so writer and compute consume separate entries. The page-table reader stages the entire row irrespective of current position. Masks are generated for the causal final chunk; masked rows are still read, which is why full mapped-capacity safety remains necessary.

Tree participation prunes children without data using child_id<k_num_chunks in compute220–239 and writer182–197. Parent IDs precede child IDs in the selected tree, so the active prefix remains connected. At p0 there are no active reduction children, **but c19 still reserves31 blocks at compile time**. This distinction explains why the initial zero-position decode can fail before exercising long-context reduction.

## Other potential issues and competing explanations

| Candidate | Adjudication / intervention |
|---|---|
| **Missing BF16-specific model geometry** | Exact endpoint match, reaches the recorded sequence, predicts dtype and capacity contrasts. Leading finding. Change only BF16 chunk/cap, then verify. |
| Legacy c19 overprovision | Real shared-factory optimization opportunity. F487–495 documents that ordinary mode allocates C−1 blocks although current tree accesses by round; speculative mode already limits to ceil(log2 C). For C32,31→5 blocks would save266,240 bytes, predicting endpoint817,408 without changing chunk/core assignment. This is broader than the requested minimal model repair and needs a complete c19 send/receive lifetime audit and generic SDPA regression matrix before changing shared code. Do not silently enable speculative mode. |
| c10 allocated for tiled Q; CBs allocated on idle cores | Real overhead, shared by passing/failing cases. c10 is8,192 bytes and alone cannot fix18,304 overlap. Merely shrinking the grid while still retaining32 cores/head leaves per-core bytes unchanged; reducing grid below32 helps through C and can break batch admission. These are not the first repair boundary. |
| Page-table capacity mismatch / undefined padding IDs | Existing divisor guard prevents this for the current native map. It is a separate historical failure mode that a repair must not reintroduce. Preserve the guard and actual page mappings. |
| Reset corruption, inaccurate LoFi, bad BF16 cache-update kernel, CCL race | None is required to reach the synchronous static-region rejection or explains its exact byte count. Those operations may warrant their own future accuracy tests; this artifact has no failing SDPA output. |
| L1 leak or unexpected tensor owner | Unproven. The logged frontier is sufficient to explain the collision. Need an allocation map to identify owners or establish a leak. The reduced-layer frontier cannot be assumed equal to full64. |
| Nominal L1 admission or stale-snapshot bug | No automatic normal-path selector is executed; explicit geometry reaches the live final validator. The speculative-only nominal-size check is outside this repro. Avoid introducing an early snapshot bug in a new selector. |
| Full-model pass means all native long prompts pass | False inference. Passing sweep results establish their recorded prompt/trace workload with native-sized storage, not full262144-token attention accuracy or every batch/prompt. Repairs require boundary and long-context tests. |

## Discriminating test plan for the implementing parent

These are proposed follow-ups; none was executed here. Preserve the owner's hardware scheduling/recovery rules. Start with static configuration checks and reduced exact geometry before expensive full-model accuracy/timing.

1. **Exact resource contrast at native geometry.** Use real layer3 Q/K/V with BF16 Q, local Q6/KV1/D256, physical KV[8192,1,32,256], INT32 map[1,8192], position0 and the full11×10 grid. Compare BF8 K256/C32, BF16 K256/C32, BF16 K128/C32, BF16 K256/C31, C30 and C16. Log actual formats, half-tile flag, CB sizes, persistent base, and lowest L1 frontier **after output allocation**. Expected endpoints:837888,1083648,813312,1073408,1063168,919808. Match full-owner live pressure or measure it explicitly; isolated-layer success of the original geometry does not refute this diagnosis. The failure-matched full64 first prepare is the definitive admission control. The existing layer runner's --capacity-alignment262144 with --seq33 or4097 and --steps4 creates native-width storage without changing logical prompt lengths; a first-position-zero control still needs an appropriate direct-call or generator preparation probe.
2. **Vary map width independently from current position.** Hold p0 and all other inputs fixed, compare W8,136,256,8192; the only table contribution is align64(4W). The current failing configuration should add32704/32192/31744 bytes relative to the three small maps. If bytes do not match, inspect logical versus padded width, sharding, Q logical heads, fp32 mode, and actual C before attributing it to nondeterminism.
3. **Preserve logical prompts and ownership.** Check lengths1,31/32/33,127/128/129,203,255/256/257,4095/4096/4097 and native-end decode positions262142/262143, with valid mapped caches. Generator physical lengths are1 or ceil(length/32)×32, not necessarily the logical lengths. Exercise one-token prefill tails, continuation, and randomized/nonidentity page maps. Keep native table width during preparation. Check exact table-end positions of small pools and external maps; never substitute unowned alignment bytes for page IDs. Direct-decoder maps W1/2/3/4/5 should resolve a128 cap to32/64/32/128/32. The generator's separate multiple-of-eight table requirement remains intact. Include an external BF16 cache bound to a nominal BF8 model policy to check that actual dtype drives the repair.
4. **Cover batch-dependent work.** At minimum B1,2,3,4,7,8,16,17,32, including ragged current positions and inactive(-1) slots. Check both ordinary and _batched_decode_impl configurations, requested/effective chunks, no lost pages, output/cache ownership, and unchanged BF8/BF4 behavior. Add a cheap exhaustive policy/work-assignment check for B1…32; do not use lower per-core CB payload as evidence of identical live-L1 headroom.
5. **Separate numerical comparisons from memory admission.** On the same finite real Q/K/V, compare BF16 chunk128/cap32 with a legal BF16 chunk256 control under sufficient L1 headroom, and with the reference attention. Compare every head/rank, finiteness, absolute/relative error and PCC. Native-end synthetic bounded inputs can cheaply test indexing; real long-context inputs must also test the changed BF16 online-softmax accumulation chain. Cap30/16 change the reduction tree/work division; chunk128 doubles native per-core softmax iterations. Do not assume bitwise equality across geometry or accept a compile-only result as accuracy evidence.
6. **Complete queued work and replay/reset coverage.** Synchronize/read back after first execution and each discriminating comparison. Warm and capture/replay with changed token/position/map contents, including transitions across128/256 boundaries. Retain the BF4 reset/replay control and confirm address stability after reset. Inspect output after completion; reaching a host enqueue is insufficient.
7. **Finish the full Stage08 candidate.** Re-run the failure-matched full64 generator preparation with native262144, then the203-token readiness workload and accuracy gates, then warmed timing. Record effective BF16 geometry and all source hashes. Preserve baseline BF8/BF4 and unrelated weight/CCL dtype geometry. The prior BF16 runtime failure has no accuracy/performance score and must not be treated as evidence that BF16 KV is numerically worse.

The stock [paged SDPA unit test:230](tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py#L230) currently uses BF8 KV,4096 sequence, D128, block64 and a different grid. Its helper supplies a chosen chunk and does not recreate full64 live L1. Existing tests establish useful contracts but do not cover this exact BF16/D256/page32/native-map admission boundary.

## Report review

The draft was checked against the exact failure hashes, constructor defaults rather than fallback values, Q logical versus padded heads, actual half-tile selection, both position mailboxes, full aligned page-table row, CB creation/aliasing/core ranges, static base, and live validator. Two independent xhigh reviewers checked the written draft and confirmed the budget/legality; their wording and citation corrections were applied, including distinguishing table-index bounds from valid physical page IDs. Standard-library calculations independently reproduced every endpoint above; checked work coverage for B1…32, six caps, four chunks and19 boundary positions; checked tree parent/child agreement for N1…64 and every active prefix; and checked divisor-guard bounds for every mapped width W1…8192. These are static translations of source, not execution tests. Review explicitly corrected two tempting mistakes: cap31 is insufficient because a reduction block costs10,240 rather than20,480 bytes; and the current SDPA policy has fp32_dest_acc_en=False despite the constructor's generic fallback beingTrue. The supplied small-smoke capacities are distinguished from the request's8192 description.

The headline diagnosis explains the exact initial failure and all supported dtype/capacity contrasts. Hardware execution, numerical quality, performance, and phase-specific live capacity after the proposed change remain for the parent; no such result is claimed here.
