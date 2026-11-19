# AUTODEBUG: Ornith full-context prefill compilation latency

## Diagnosis

**The observed run completed successfully. The captured apparent stall was a host-side compilation wait, with two concrete sources of unnecessary position-dependent specialization in the original model caller. There is no remaining evidence of a permanent device hang in this run.**

1. The original caller passes a host integer `chunk_start_idx=position`. Legacy chunked SDPA deliberately includes that value in its program key and derives compile-time KV geometry from it. The 262144-token traversal therefore has **256 distinct SDPA program configurations**, despite identical query shapes.
2. Before SDPA, the original caller slices the entire remaining page-table suffix for cache filling. Its width decreases on every chunk. This changes both the row-major slice writer's accessor configuration and the downstream paged-fill writer's compile-time page-table row size. **Changing SDPA alone leaves this second compilation family.**
3. RoPE slices also have position-specific operation-cache keys, but their changing offsets are runtime arguments in the selected tiled slice kernels. Offset changes alone do **not** establish another family of distinct kernel binaries.

The minimal model-local correction identified here is to bound the fill table to the padded chunk, select the existing device-tensor SDPA offset API, and align the attention page-table row for that API. **These changes appeared through an external edit to the shared source during this investigation.** They are described and inspected below; this investigator did not implement or execute them. Their performance improvement and hardware correctness remain unverified here.

## Scope and source identity

Inspection only: read prepared triage, logs, Python/C++/kernel source, and existing result JSON; ran pure-Python AST, hash, and integer-planning checks. No target execution, device access, reset, process termination, source edit, or AutoDebug relaunch was performed. This report is the only file written by this investigator. Three isolated xhigh reviewers checked SDPA, slicing/fill, and the evidence chronology.

- Checkout HEAD: `9e8a45111a69fe0531f65025022ab887ca59df5a`.
- Model revision in the result: `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
- Original `tt/functional_decoder.py` SHA256: `d8ed75fb8c62f10282ada9c4340455b62c981e8a206908e457270c6cb6204db9`. This matches `doc/functional_decoder/full_long.json:124` and the source initially inspected.
- The shared decoder changed externally at **2026-11-12 16:22:25 UTC**, after the original run closed. The subsequently inspected SHA256 is `15cd5c5d872665eccaa2fa1dcd34d0d4acb7d45a680cb00481d43ab59e264985`.
- Reversing only the visible local-table/attention-table/start-argument changes **in memory** exactly recovered the original recorded SHA256. Thus the original run and present correction are kept revision-consistent; the passing original result is not evidence that the correction was tested.

Unless qualified otherwise, model paths below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`. Original decoder line references describe the initial source; shared-source references explicitly describe the later version. TTNN and Metal references are checkout-relative.

## Direct evidence and its limits

| Direct observation | Supported conclusion | Limit |
|---|---|---|
| `doc/functional_decoder/triage/full_long.txt:4-11` lists prefetch, dispatch, and profiler service loops; lines 13-14 list no running operation. | Consistent with device idle while waiting for the next host submission. | A snapshot does not diagnose all previous or subsequent device intervals. |
| `doc/functional_decoder/triage/full_long_host_stack.log:710-718` shows the main thread in `sync_build_steps -> ProgramImpl::compile -> compile_and_allocate -> MeshWorkloadImpl::compile -> EnqueueMeshWorkload`. | The sampled submission is waiting for host kernel compilation. | The stack ends at frame 7: no Python call site, operation identity, chunk position, or compile hash is recorded. |
| The same stack, lines 574-581, 604-606, 634-636, and 674-676, includes waits for linker/compiler children. | Actual compile/link dependencies are present, rather than just an unexplained generic future wait. | One stack cannot prove that children changed over time. That detail is requester-provided context, not independently captured progress evidence. |
| `doc/functional_decoder/triage/full_long_summary.txt` reports all listed checks passing. | Those checks reported no detected fault. | This is not a whole-run correctness or performance guarantee. |
| `doc/functional_decoder/full_long.log:20-36` eventually contains comparisons for all three long prefills, `LONG_DECODE_PCC`, and `LONG_CONTEXT_PASS`. Lines 39-43 record clean closure. | The original run completed its long-context test and closed naturally. | Prefill accuracy is sampled against complete HF K/V history, not whole-output PCC. |
| `full_long.log:37` reports `320/2110` JIT cache hits, `15.2%`, and `2748 build-once deduplications`. | The completed run had substantial JIT activity and reuse. | Aggregate JIT telemetry is not the TTNN program-cache hit rate and does not attribute elapsed time to specific operations. |

The evidence grew during inspection: the first read of `full_long.log` contained only its 19 startup lines; subsequent reads contained completed windows, decode, and finally the pass. The triage file mtime was **16:17:17**, host stack **16:17:38**, and final close **16:20:54 UTC**. The driver opened at **16:13:35.783**: about **7 minutes 19 seconds** elapsed through clean closure. No per-chunk or per-operation duration is available. In particular, do not reinterpret the saved stack as a proven second-pass compilation stall merely because a later log read contains first-pass results.

`full_long.json` now records fifteen sampled prefill windows, with minimum output PCC **0.9982076883** and minimum residual-subtracted contribution PCC **0.9981271625**, exceeding the harness's 0.995 thresholds. Decode at position 262143 has PCC **0.9997731447**, three deterministic traced replays. This is evidence from the already-running original test, not a reproduction launched by this investigator.

## Effective repro and completion boundaries

`tests/long_context.py:34,47-48,85-96` defaults to context 262144, opens one device, prepares a random INT32 page permutation, and performs three independent prefills. The completed invocation used layer 3 (`full_long.json:4-8`). `FunctionalDecoder.from_state_dict` defaults to **1024 tokens**, not 512 (`tt/functional_decoder.py:31`). The original loop pads full-attention tails to 32 tokens (`:245-274`).

| Logical prefill length | First 255 chunks | Final absolute start | Final logical count | Final attention count after padding |
|---:|---:|---:|---:|---:|
| 262139 | 255 x 1024 | 261120 | 1019 | 1024 |
| 262143 | 255 x 1024 | 261120 | 1023 | 1024 |
| 262144 | 255 x 1024 | 261120 | 1024 | 1024 |

Every traversal uses starts `1024*j`, `j=0..255`. After projection, normalization, permutation, and RoPE, Q is BF16 tiled `[1,16,1024,256]`; K/V are BF16 tiled `[1,4,1024,256]`. Each allocated cache is BF16 `[8192,4,32,256]`, **512 MiB**; the full row-major INT32 page table is `[1,8192]`. SDPA uses a fixed 8x8 grid and 32-token Q/K chunks, scale 1/16, HiFi4 with FP32 destination accumulation (`functional_decoder.py:40-42,89-100,172-220`). The SDPA factory consequently selects its non-streaming compute path (`sdpa_program_factory.cpp:82,1398`). There are 512 head/query-chunk tasks distributed over 64 cores, eight per core (`:1365-1386`).

The complete forward, including its final output concatenation, occurs before `ttnn.synchronize_device` at `long_context.py:95-97`. Comparisons and their first progress prints occur only afterward (`:98-113`). The `device_only()` guard (`tests/validate.py:23-36`) restricts host tensor work; it is not a completion barrier. Cache clearing also enqueues device operations. Therefore silence before the first comparison does not locate a device failure. Conversely, the emitted comparisons prove the first two full forwards crossed that explicit completion boundary. Decode replay is explicitly blocking, followed by readback and exact repeated-output comparisons (`long_context.py:128-140`); final clean close rules out a deferred teardown hang for this run.

The scalar-offset and suffix-width domains repeat on the second and third traversals. They explain initial specialization cost, not recompilation forever at identical keys. Other input logical lengths, tail crop/pad shapes, and final concatenation sizes do differ between the traversals.

## Source evidence: two independent compilation families

### 1. Host-valued absolute SDPA offset specializes real compiled geometry

Original call: `tt/functional_decoder.py:210-220`, specifically `chunk_start_idx=position` at line 215.

The lowering is explicit:

1. `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp:285-310` chooses the integer overload unless a start tensor is supplied.
2. `sdpa/sdpa.cpp:104-141` forwards that integer to `prim::sdpa`, with causal mode forced true and no device start tensor.
3. `sdpa/device/sdpa_device_operation_types.hpp:15-22` stores the integer in `SDPAParams`. SDPA has no custom program hash; `ttnn/api/ttnn/device_operation.hpp:73-86` hashes the operation type, attributes, and tensor arguments. Its inclusion is intentional, not an incorrect shared-cache implementation.
4. `sdpa/device/sdpa_program_factory.cpp:1225-1258,1267-1270` derives legacy `Sk = position + Sq`, `Skt = ceil(Sk/32)`, and K chunk counts. The derived dimensions enter **reader, writer, and compute compile-time arguments** at `:1556-1569,1647-1657,1689-1697`.
5. The selected kernels are `dataflow/reader_interleaved.cpp`, `dataflow/writer_interleaved.cpp`, and `compute/sdpa.cpp` (`sdpa_program_factory.cpp:1986-2008`). `tt_metal/impl/kernels/kernel.cpp:882-887` hashes compile-time arguments. `kernel_compile_utils.cpp:23-32` incorporates that kernel hash into the binary compile hash.

For this model's padded Q length of 1024:

| Absolute start | Query-chunk offset `position/32` | Legacy `Sk` | Legacy `Skt` / K chunks | Highest required virtual page |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1024 | 32 | 31 |
| 512, a valid continuation example | 16 | 1536 | 48 | 47 |
| 1024 | 32 | 2048 | 64 | 63 |
| 261120, final default chunk | 8160 | 262144 | 8192 | 8191 |

Even though the scalar offset itself also becomes a runtime argument, its derived KV extent changes the binaries. This completes the causal chain beyond merely observing a changing cache key.

Distinct hashes mean distinct binary configurations. They do **not** prove a compiler runs on every invocation: Metal deduplicates by kernel hash (`tt_metal/impl/program/program.cpp:445-458`) and can reuse existing build artifacts. A cold encounter with many configurations is the source-supported latency mechanism; the saved stack does not identify which one was being compiled.

### 2. The overlong fill table varies shape before SDPA is reached

Original `functional_decoder.py:207` is:

```python
local_table = page_table[:, position // 32 :]
```

At chunk `j`, this has width `8192 - 32*j`: **8192, 8160, ..., 32**, although each padded chunk fills only **32** pages. This is the earliest unnecessary changing *shape* in the cache-fill/attention chain. The two preceding RoPE slices already change operation keys, but not their offset-only binary configuration.

Python indexing calls `ttnn.slice` (`ttnn/ttnn/operations/core.py:124-149`). A whole-tensor slice is bypassed (`slice/slice.cpp:197-200`), so the first suffix is an alias and there are 255 actual suffix-slice operations. For the remaining calls:

- `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:459-489,517-525` hashes bounds, steps, factory, and full input/output specifications. Both bounds and output width vary.
- ROW_MAJOR selects `SliceRmProgramFactory` (`:425-456`). Its tensor specifications are bound with static accessor page geometry (`slice_program_factory_rm.cpp:332-338,507-515`). `tt_metal/impl/metal2_host_api/program_spec/construction/resource/tensor.cpp:113-132` emits aligned page size as a compile-time argument. Each 128-byte decrease in this workload changes the output writer's row-page configuration.
- The same suffix then feeds both `paged_fill_cache` calls. `experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:216-224` hashes tensor arguments; `paged_fill_cache_program_factory.cpp:155-160,340-346` explicitly compiles `page_table_stick_size` and its log2 into the fill writer. There are **256 distinct page-table widths** supplied to fill, independently of the SDPA offset. K and V have matching specifications and can share programs; this is not 512 independent width configurations.

The fill kernel only indexes chunk-relative pages. `experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp:21-38` maps local `seq_tile_idx` to `local_table[seq_tile_idx / block_size_t]`; here `block_size_t=1` and a chunk contains 32 such rows per head. The unused suffix pages are unnecessary. Replacing the suffix with exactly those 32 entries preserves every selected physical page, including the random permutation, while making the fill-table geometry constant.

### 3. RoPE slices are a different cache issue

`functional_decoder.py:204-205` slices BF16 TILE `[262144,64]` cos/sin tables into fixed `[1024,64]` tables. Slice keys still include absolute start/end, so distinct offsets produce operation-cache entries. Cos and sin's differing contents do not require separate keys when their tensor specifications and slice bounds match.

However, TILE selects `SliceTileProgramFactory`. In `slice_program_factory_tile.cpp:69-100,129-140,192-210`, start offsets, shape traversal values, and work counts are runtime arguments; explicit kernel compile parameters include rank, while the accessor page remains a BF16 tile. With the same shapes/grid/instrumentation, changing only the absolute RoPE offset does not change the kernel binary. The same distinction applies to many fixed-shape input slices. Do not headline all slice cache misses as repeated compiler execution.

## Minimal model-local correction identified and its safety contract

The later shared source already contains all three changes at `tt/functional_decoder.py:207-223`:

```python
# n is the full-attention chunk length AFTER padding to 32 tokens.
local_table = page_table[:, position // 32 : (position + n) // 32]
ttnn.experimental.paged_fill_cache(state.key, k, local_table)
ttnn.experimental.paged_fill_cache(state.value, v, local_table)

attention_table = page_table
if page_table.shape[1] % 8:
    attention_table = ttnn.pad(page_table, [(0, 0), (0, 8 - page_table.shape[1] % 8)], 0)

# In the existing SDPA call:
#   page_table positional argument -> attention_table
#   chunk_start_idx=position ->
#   chunk_start_idx_tensor=self.positions[position : position + 1]
```

This is a caller-side preparation correction; the shared SDPA cache does not need a hash change or a new kernel. The existing `self.positions` is a device INT32 ROW_MAJOR full-context vector (`functional_decoder.py:78-83`). A one-element slice satisfies flexible SDPA's `[1]`, INT32 device-tensor contract (`sdpa_device_operation.cpp:234-251`).

Tensor contents and device addresses do not participate in reflected tensor hashing (`ttnn/api/ttnn/tensor/tensor.hpp:252-253`, `storage.hpp:204-205`). The start tensor also appears in `SDPAInputs`; the ProgramDescriptor binds its current buffer (`sdpa_program_factory.cpp:1341-1344,2043`). The adapter resolves these bindings and reapplies current tensor arguments on hits (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:649-654,798-803`). Consequently, different same-spec start tensors can reuse the SDPA configuration while retaining the correct current value.

The flexible factory compiles `Sk = Sq + page_table.padded_width * block_size`, **263168 tokens / 8224 tiles** for this case, independent of absolute position (`sdpa_program_factory.cpp:1228-1258`). This conservative extent is not an extra cache allocation or an instruction to read beyond context. Runtime start controls the causal work:

- Reader reserves, reads, and pushes one start value into each of two distinct one-entry CBs (`reader_interleaved.cpp:350-374`). Compute and writer each wait, read, and pop their own one entry (`compute/sdpa.cpp:98-107`, `writer_interleaved.cpp:192-203`). Both CBs receive exactly what their consumers expect.
- All three divide the start by the configured Q chunk size, **32**, and use the same absolute query-chunk coordinate.
- Reader limits the effective valid K extent by runtime start plus Q length (`reader_interleaved.cpp:408-416`) and each causal Q chunk's high K tile by its absolute query end (`:575-587`). Compute uses the matching absolute coordinate (`compute_common.hpp:1707-1724`); masking follows absolute Q/K positions (`:1299-1347,1907-1923`).
- Starts 0, 1024, and 261120 therefore still access at most 32, 64, and 8192 K tiles, respectively. The last accessed virtual page remains 8191, not 8223.

**Preserve these conditions:**

1. **Keep the existing arbitrary-start bridge.** Original `prefill_forward:251-262` decodes one token at a time until the absolute start is divisible by 32. For start 33, it consumes positions 33 through 63, then chunked prefill begins at 64. Supplying 33 directly to flexible SDPA would floor it to 32. No 1024-token alignment requirement should be introduced.
2. **Use padded `n` for the bounded fill map.** A 27-token tail is processed as 32 rows and requires one page. With aligned start and a real context divisible by 32, rounding a valid tail to 32 remains within that context. Later continuation overwrites padded future cache rows before consuming them causally.
3. **Keep the full logical page map for attention.** Only filling uses the local chunk map. Rebasing attention to that map and resetting the start to zero would discard all preceding KV history; retaining the full map but resetting the start would put the causal diagonal in the wrong place.
4. **Pad the SDPA table to eight INT32 entries per row when needed.** Flexible mode requires `padded_width * 4` divisible by 32 (`sdpa_program_factory.cpp:163-166`); legacy mode instead used the buffer's aligned page size (`:168-172`). Existing short validation allocates 2080 tokens, yielding 65 pages: pad to 72 for flexible attention. The real request still accesses only pages 0-64. Zero padding does not authorize longer requests or enlarge the cache.
5. **Retain actual bounds.** Flexible validation cannot inspect the device start value. It derives an extent from the supplied page-table width (`sdpa_device_operation.cpp:275,306-307`) and only checks that this inferred extent covers Q (`:417-423`). Padding 65 entries to 72 increases the inferred extent without increasing physical KV capacity. The model's existing assertion checks only the configured maximum (`functional_decoder.py`, original line 246 / later line 249), not the allocated state's capacity. Safety therefore depends on requests fitting the original real cache/page-map extent; the external correction added no actual-capacity assertion. The advertised 262144 context and all true positions remain unchanged.

This removes the two proven position-dependent binary-geometry families. It does not make the entire model cache constant-sized: the one-element position slice, RoPE slices, bounded map slices, and input slices still have scalar-bound-dependent operation keys. Distinct tail sizes can also create a finite set of different programs. Those remaining costs require measurement before a broader rewrite.

## Other potential issues and rejected explanations

| Candidate | Adjudication |
|---|---|
| Permanent SDPA/NoC/CB device deadlock | Not supported by the captured idle-device/host-compile state; the original run later passed and closed. No concrete failing queue/count contract was found. |
| Shared SDPA hash is accidentally including an offset | False framing. Legacy `Sk` really is compiled from that offset. Removing it from the shared hash without changing setup would risk reusing stale geometry. Use the available flexible API instead. |
| Every RoPE offset recompiles a binary | Unsupported. Slice keys vary, but the selected tiled slice's offset values are runtime inputs. |
| Full context necessarily overflows the allocated KV cache | Not supported for this test: 8192 valid pages, final access page 8191, matching allocation, followed by successful completion. Flexible compiled upper bounds are not the number of pages actually read. |
| Actual allocated-state capacity is enforced at prefill admission | Not generally true: the model checks the configured maximum, and callers can allocate a smaller state. This does not explain the valid full-context run. Any future admission check should use the original page-map/cache capacity before metadata padding; padding must not make an invalid request admissible. |
| Small-context passes disprove the compilation diagnosis | They do not. Existing `full_validation.json` covers context 2080 and relatively few chunk positions; compilation diversity scales to 256 positions at full context. Conversely, small tests alone never proved long-context performance. |
| All elapsed time is compilation | Unproven. Attention over growing history is genuine work, output retention/concatenation also costs memory and time, and no per-operation timing is saved. The source identifies avoidable specialization, not its exact wall-time share. |
| Repeated second/third traversal recompiles because absolute offsets keep changing | Incomplete. Those traversals reuse the same 256 offsets and padded Q shape. Any additional compilation needs another changing specification, a cache-state difference, or an artifact miss. |

Progress visibility is weak: the harness prints nothing inside a 256-chunk forward. Optional per-chunk/interval markers would make subsequent compilation versus execution progress easier to distinguish, but are not the performance repair.

Broader cache-stability work is optional. Replacing RoPE slicing with embedding requires preparing ROW_MAJOR BF16 RoPE tables once: embedding on the current TILE tables first untilizes them (`embedding/embedding.cpp:32-38`). Integer page maps must stay exact; do not use BF16 embedding weights for page IDs up to 8191. Tensor-argument slicing is also not a universal solution: ROW_MAJOR slice paths read tensor bounds back with `.to_vector<T>()` (`slice/slice.cpp:567-568`), violating this decoder's device-only forward intent. None of these broader changes is required for the minimal correction above.

## Claim review and remaining verification

The first draft was checked against both sides of each boundary:

- **Operation key versus kernel hash:** confirmed real compiled geometry changes for scalar SDPA and shrinking RM/fill rows; demoted offset-only RoPE changes to operation-cache overhead.
- **Observed wait versus permanent hang:** reconciled the earlier stack with later completion, including explicit synchronize, blocking trace replay, readback, and final close.
- **Proposed intervention versus source state:** re-read the shared decoder; the correction is already present from an external edit. Its source hash differs from the successful original run. No corrected-run performance or correctness claim is made here.
- **Arbitrary starts and final context:** checked the decode-to-32-alignment bridge, padded-tail page counts, absolute causal limits, complete-prefix page mapping, and short 65-page table alignment.
- **Static local checks:** AST confirmed the 1024 default; pure integer calculations confirmed all three 256-chunk plans, 256 legacy `Sk` values, 256 suffix widths, constant 32-page bounded maps, and aligned/unaligned/context-end examples. A further 261648 aligned position/padded-chunk combinations preserved exactly the filled virtual-page interval. The original-source hash recovery matched the completed result JSON exactly.

For the calling agent's later hardware validation, the decisive comparison is the original and corrected model under the same context, cache state, and instrumentation. Check first-use versus warmed compile activity, confirm that SDPA geometry no longer varies by absolute start and fill row size remains 32 entries for 1024-token chunks, then retain the full long-context windows, deterministic final decode, randomized page maps, and continuation starts 32, 33, 1024, 1025, and 2050. Include the 65-page short-context case to exercise the new alignment requirement. Do not infer speedup merely from a passing run or a larger aggregate cache-hit percentage.

**Confidence:** high in the source-level specialization diagnosis and the original run's successful completion; high in the stated model-local intervention boundary; unmeasured in the runtime contribution of each specialization family and the corrected version's hardware behavior.
