# AutoDebug: Ornith TP4 cross-request trace allocation failure

## Leading finding

**The generator captures decode/sampling after preparing only decode/sampling, then runs previously unseen eager prefill programs while those traces remain live.** Changing the logical prompt from 33 to 34 creates 20 persistent allocations in program-cache miss contexts. The allocation tracker rejects the next model replay before it executes. Equal tile padding does not imply equal program signatures: logical tensor shapes, exact slice offsets, padding amounts, and terminal model work remain request-dependent.

The smallest architectural repair is generator-owned preparation followed by capture, **plus a bounded physical prefill adapter**. Moving the existing decode warmup earlier, warming a few observed prompt lengths, or removing trace retirement alone cannot cover arbitrary prompt lengths and continuation starts. Retain the reviewed decoder's layer-specific chunk sizes, math/precision policy, state representation, and residual/collective strategy. Some preparation/input plumbing must change; passing padded tokens into the current decoder unchanged is incorrect for linear attention.

Confidence is high in the allocation/lifecycle diagnosis. Proposed normalized execution profiles below are repair hypotheses requiring numerical, allocation, and hardware validation. This investigation did not execute TTNN, open devices, modify source, reset hardware, or launch AutoDebug again. Only this report was written.

## Evidence identity and terminology

Inspected checkout: `c2254f91c38840cd4069a1ecbbd3f4874e27d557`, 2026-11-13. `generator.py`, `model.py`, the full-model probes, and full-model log directory are untracked working-tree additions; attributing their contents to HEAD would be incorrect. The optimized `multichip_decoder.py` has no reported working-tree modification.

Paths abbreviated below:

- **G**: `models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py`
- **M**: `models/autoports/ornith_ai_ornith_1_5_9b/tt/model.py`
- **D**: `models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py`
- **L**: `models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/logs/cross_request.log`

SHA-256 at inspection:

| File | SHA-256 |
|---|---|
| G | `cc3325becaf8c8481e7431ec452d5a9abbae5b0e4814b797e6f2e3ff371af3da` |
| M | `170e6de459b0fa817ce2d1ec8d9385a98f24a2936106588e4bdff82a75b92294` |
| D | `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8` |
| L | `1b4460366ba5ca8a819255a0c037d8e272c3832582da4881efa004f2bb3f9394` |

Read the installed `tt-enable-tracing/SKILL.md` at the requested version 0.1.23 and the checkout's `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`. Their relevant contracts agree: prepare all generator paths before the first capture, keep program-cache allocations in the safety gate, retain traces during supported requests, and acknowledge only reviewed individual backing buffers.

## Direct observations, separated from interpretation

| Direct observation | What it establishes |
|---|---|
| L:21–22 loads layers 0 and 3. | This artifact is a reduced stack with real terminal weights, not an all-layer Stage 6 validation. |
| L:25: first generation returns eight tokens; 7 model replays, 7 sampling replays, 2 captures, 0 releases. | Initial traced feedback executes. Token quality is not established. |
| L:26–27: another length-33 prompt succeeds; counters become 14/14 replays, still 2 captures/0 releases. | Changed token contents at the same shape reuse the traces. |
| L:28: `NEXT_REQUEST 34`; L:39–59 fails in `verify_before_replay`. | Length 34 reaches the pre-replay safety gate; that failing model replay never executes. |
| Every reported allocation has `program_cache: <operation>` and a prefill stack. | New persistent program-context allocations occur during eager prefill, after the previous captures. |
| The probe schedules lengths 33, 34, 31, 65 (`tests/full_model_trace_probe.py:31–36`). | The log stops at 34. It supplies no result for 31 or 65. |

Interpretation supported by source: incomplete preparation is the earliest divergent lifecycle decision. Exact logical shapes and offsets turn the changed request into new program signatures. This predicts unchanged-shape passing, changed-shape failure at the next replay, and persistence through Python GC.

The tracker conservatively marks allocations made after trace activation; it does not prove address-by-address overlap. The log therefore establishes a violated allocation-safety contract, **not actual corruption of all 20 buffers**. It does not establish their byte sizes, explain token 220, or prove an allocator, SDPA, CCL, or math-kernel defect.

## Causal chain and concrete allocations

### Lifecycle

1. `generate()` calls `prepare_decode()` before prefill (G:288–289).
2. Preparation executes `_model_step`, sampler precompile, `_sample_step`, and recurrent reset (G:202–210). It does **not** prepare all eager prefill, embedding, terminal norm/head, output-selection, or continuation paths.
3. The first length-33 prefill runs before capture. `decode_forward()` refreshes persistent inputs and captures model and sampling (G:248–261,214–230).
4. On subsequent requests, the state lookup returns immediately (G:189–191) and `capture()` returns when the model trace already exists (G:217–218). There is no request-time release in this path; the source releases only in `close()` (G:315–320).
5. Length-34 prefill now compiles behind the two retained traces. Its ordinary result tensors can die, but cached workload allocations survive.
6. `ttnn.execute_trace` checks allocations **before** native replay (`ttnn/ttnn/__init__.py:139–142`). The check runs GC and re-queries live buffers before raising (`ttnn/ttnn/tools/trace_allocation_tracker.py:138–183`). Thus asynchronous replay or teardown is not the first failure here.

The eager prefill's `host_logits()` reads results back (M:153–155), providing completion/readback before the decode boundary. Model and sampling replay use CQ0 and `blocking=False` (G:264–268); token readback follows sampling (G:272–274). A successful host assertion alone would not establish queue completion, but this particular exception is host-side rejection before replay submission.

### All 20 reported allocation IDs

These are allocation IDs from the log, not a claimed count of unique source lines or per-chip IDs.

| Buffer ID | Originating operation | Source call site | Log line |
|---:|---|---|---:|
| 2764 | Embedding | M:111 | 60 |
| 2774 | Embedding output tilize with padding | M:111, composite embedding lowering | 78 |
| 2786 | Prefill input RMS norm | D:1043 → 689 | 96 |
| 2795 | Norm output copy to L1 | D:696 | 116 |
| 2810 | Linear-attention `z` slice | D:792 | 136 |
| 2820 | Beta slice | D:793 | 168 |
| 2833 | `a` slice | D:794 | 200 |
| 2844 | `a` BF16-to-FP32 conversion | D:794 | 232 |
| 2857 | Packed-input tail padding | D:803 | 254 |
| 2875 | Untilize with unpadding within history extraction | D:819, composite slice lowering | 276 |
| 2884 | Row-major convolution history slice | D:819 | 308 |
| 2897 | `g` tail padding | D:834 | 340 |
| 2906 | Beta tail padding | D:835 | 362 |
| 2951 | Out-projection input copy to L1 | D:895 → 633 | 384 |
| 2964 | Linear-attention reduce-scatter | D:895 → 440 | 408 |
| 2989 | Short-prefill gate/up matmul program variant | D:721 → 762 | 432 |
| 3002 | Down-projection input copy to L1 | D:732 → 633 | 456 |
| 3020 | Full-attention input padding | D:1042 | 478 |
| 3142 | Final RMS norm | M:131 | 496 |
| 3151 | LM-head matmul | M:134 | 514 |

Why these persist: `ttnn/api/ttnn/device_operation.hpp:403–451` computes the workload cache key and routes misses through `create_and_cache_mesh_workload_with_allocation_context`. Lines 377–393 install the operation-specific `program_cache:` label seen in L; lines 357–368 enqueue and cache the workload. Ordinary output creation occurs separately, before invocation/cache lookup (lines 630–646).

`tt_metal/distributed/mesh_workload.cpp:183–219` allocates a replicated DRAM mesh buffer for program binaries on initial load. Workload ownership retains that buffer (`mesh_workload_impl.hpp:77`), and the program cache retains the workload (`tt_metal/api/tt-metalium/program_cache.hpp:66–86,149–150`). Inspection of the relevant program factories supports persistent binary-buffer attribution; the log itself prints allocation context, not C++ object identity. An ID/address/size/owner dump would conclusively identify each object.

A workload's binary storage is one mesh allocation replicated across participating chips. **Do not reinterpret 20 as five programs times four chips.** It is 20 mesh allocation regions, ordinarily with a backing region on each of the four chips. The allocator tracker marks post-capture allocations without address-overlap analysis (`tt_metal/impl/allocator/trace_allocation_tracker.cpp:183–215`). JIT artifact cache hits at L:542 do not prevent creation of new workload binary buffers.

### Instantiate 33 versus 34 through lowering

| Quantity | Logical 33 | Logical 34 |
|---|---:|---:|
| Token upload width (G:130) | 33 | 34 |
| Embedded logical sequence height (M:112) | 33 | 34 |
| Tiled embedded height | 64 | 64 |
| Linear layer's logical chunk height | 33 | 34 |
| Linear qkv/GDN physical height | 64 | 64 |
| Packed/g/beta tail padding | 31 | 30 |
| Last three convolution input rows (D:819) | `30:33` | `31:34` |
| Full-attention padded chunk height (D:1042) | 64 | 64 |
| Full-layer final trim / terminal logical height | 33 | 34 |

Embedding explicitly uses `runtime_rows=false` in L:60. Its wrapper derives sentence size from the input, chooses fused tilize only when the indices' padded width is a multiple of 32, and otherwise converts the result to tile layout (`ttnn/cpp/ttnn/operations/embedding/embedding.cpp:44–83`). The ordinary embedding hash includes tensor arguments; the row-independent option is a separate branch (`device/embedding_device_operation.cpp:173–193`). Merely enabling that option would not fix downstream signatures.

Python indexing fills omitted bounds from the **logical** tensor shape (`ttnn/ttnn/operations/core.py:124–149`). For the `z`, beta, and `a` slices, lowering rounds the row end to 64 for both requests. Nevertheless `SliceDeviceOperation::compute_program_hash` includes input logical shape as well as padded shape and slice bounds (`ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:459–489`). Consequently identical logged padded ranges can still represent different programs.

History extraction has a second independent change: its begin row changes from 30 to 31. Both are unaligned. Composite slice lowering takes the row-major path for unaligned tile begins (`slice/slice.cpp:224–235`), accounting for both untilize and RM slice allocations. L:276 and :308 show the length-34 untilize end `[0,0,33,3135]` and range `[0,0,31,0] → [1,1,34,2048]`.

The full-attention layer already pads its main work to 64, but padding itself receives a different logical input, and its final trim restores the request's exact length (D:1050). Terminal norm/head therefore still see 34. This explains why “both are in the 64-token bucket” does not resolve the reported failure.

## Minimal repair hypotheses

### H1: explicit generator preparation, then a sealed capture phase

Introduce idempotent `prepare(...)` owned by the generator and called before request state is admitted. Standalone generation and the low-level prefill/decode entrypoints must share it; relying only on serving warmup callbacks is insufficient.

**Phase A, no traces live:** allocate the served caches, stable input/binding slots, CCL/semaphore resources, zero/reset tensors, masks, index tensors, and normalized prefill staging. Warm every supported physical execution profile, decode profile, supported sampler graph, reset/state-update operation, and output/postprocessing path. Use the actual served KV allocation/spec, including page-table and pool shapes. Include eager prefill and single-new-token generation, even though they do not themselves replay a trace. Restore real cache/state, tokens, positions, tables, and applicable sampling RNG state after preparation. Synchronize before switching phases.

**Phase B:** record from prepared state with program-cache misses forbidden. Refresh the exact intended capture inputs immediately beforehand. Capture each execution signature and stable buffer binding at most once. Deferred first capture of a prepared variant is acceptable; compiling or allocating unrelated persistent helper state at that time is not. Keep every existing trace live until explicit unload/reconfiguration/recovery/close.

The current `if self.states:` guard (G:192–193) must become an explicit preparation/sealed-state check: it rejects a second binding even before the first trace is captured. Raising earlier on a genuinely unregistered new physical cache allocation is preferable to unsafe replay, but is not a substitute for preparing the configured supported request variants.

Keys should describe actual graph/spec/bindings: cache backing allocations and pool geometry, physical batch/layout, prepared input slots, table capacity, operation modes, and sampler graph flags that actually change execution. Logical lengths, token values, current positions, and page-map contents are refreshed data. An exact slot tuple only needs its own trace if it changes binding or execution; a scheduler can instead copy rows into prepared physical slots. The present `(id(kv_cache), tuple(slots))` key (G:189) does not validate table shape or backing storage replacement.

For this tree, sampling is fixed greedy (G:72–73); do not claim regular/seeded support from ignored `**kwargs`. If new modes are supported, prepare their distinct graphs and resources before the first capture and reuse earlier traces on return to a mode.

### H2: normalize prefill at model boundaries and inside chunk plumbing

Keep the layer-major schedule and default chunk caps **32768 for linear attention and 16384 for full attention** (D:156–160). Retain BF8 attention/MLP/KV policy, HiFi2 attention, LoFi MLP, FP32 recurrent state, BF16 residuals, current reduction choices, and packed local L1 decode residuals (D:116–124,195–205,429–469,658–699,1301–1355). Forcing every layer into small common chunks would change the reviewed strategy and is not the proposed fix.

Separate `(valid_count, absolute_start)` from the fixed declared tensor shape used by TTNN. A bounded adapter can use the following **candidate catalogue of declared sequence heights** while retaining the explicit Python branch/config regimes:

| Path | Prepared heights | Selection |
|---|---|---|
| Linear prefill | `1, 32, 64, 128, 511, 512, 640, 768, 960, 1024, 4096, 8192, 32768` | Smallest listed height ≥ real chunk count. Height 511 has tiled backing height 512. |
| Aligned full-attention prefill | `32, 64, 128, 480, 512, 768, 1024, 4096, 8192, 16384` | First form reviewed height `m=32*ceil(real_count/32)`, then smallest listed height ≥ m. |
| Full-attention unaligned prefix | One-token layer-decode profile | Repeat at most 31 times, changing persistent position data. |
| Token embedding | One fixed tiled input width, e.g. 1024 | Stream whole-prompt token input through this width; dummy indices remain valid token IDs. |
| Terminal norm/head | One fixed height, e.g. 32, or a deliberately prepared small set for throughput | Stream all-logit requests; gather the real last hidden row before last-only head work, or crop a fixed output on host. |

The 511/512 split is deliberate. Using logical 512 for real counts 129–511 would prematurely enable BF8 norm/reduction and the special matmul branch (D:436,601,697). Height 1 retains the existing single-row projection/norm/reduction and prefill-GDN combination; it is not interchangeable with arbitrary 32-row prefill.

The linear heights 640/768/960 preserve changes in `ceil(n/320)` for its packed-input grid and `ceil(n/256)` for output projections (D:605–615). Full attention uses `ceil(m/256)` for both. Counts ≤64 retain short-MLP `ceil(n/32)`; 65–128 retain its other short path; larger counts retain minimal fused-SwiGLU (D:711–770). A cheap standalone integer-arithmetic check exhaustively compared these explicit profiles for all 32768 linear counts and all 512 aligned full counts and passed. This does **not** prove identical program hashes, inferred matmul plans, numerical outputs, memory feasibility, or optimal performance.

This catalogue is a bounded implementation proposal, not a claim that these modes already exist. It requires all of the following plumbing:

1. **Linear state must use the real count.** Mask `g=0` and `beta=0` after the final real row, including dummy rows within the declared bucket, so `exp(g)=1` and padded GDN updates leave the FP32 state unchanged. Existing protection only covers rows beyond the current logical `n` (D:832–835). Zero input embeddings alone fail: `sigmoid(0)=0.5` and bias/softplus/decay generally give a nonzero update/decay. Extract convolution history at the actual real end, not bucket end. A fixed-shape gather over old three-row history plus projected inputs handles real counts 1 and 2 without negative indices. Keep backing state identities and per-slot writes intact.
2. **Runtime start/row lookup.** Replace absolute cos/sin slices, local page-table slices, `self.positions[absolute:absolute+1]`, exact history/end slices, and variable full-layer result trims (D:819,972–983,1024–1050) with prepared fixed-shape inputs or index-driven operations. For eager prefill, host construction of fixed-shape RoPE indices, positions, masks, and fill maps followed by copies outside capture is acceptable; it must not create persistent device buffers or programs after sealing.
3. **Full-attention alignment.** For interval `[s,s+n)`, preserve the current prologue of `r=min(n,(-s)%32)` one-token calls, then aligned 16384-token bodies and one normalized tail. This accepts every start, including `s=1,31,33,262143`. Do not replay cached prefix tokens to force alignment or overwrite their linear state. Keep absolute RoPE/cache positions and the flexible tensor-valued SDPA start. Linear chunks independently retain their 32768 cap.
4. **Separate fill and attention maps.** Construct a fixed-size chunk-local fill map containing the request's real physical pages for `ceil(valid_count/32)` pages and skip sentinel `-1` elsewhere. Insert the sentinel after adding any pool offset, or preserve it explicitly; `-1 + offset` is not the sentinel. Existing paged-fill kernels explicitly drain/discard input tiles for `0xFFFFFFFF` (`experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp:15–33,277–294`). Cast K/V to the actual BF8 cache dtype as the current decoder does. Use safe local page-zero mappings, **not -1**, for inactive entries in the attention map. Last partial-page dummy rows are future rows and must remain causally invisible to real queries.
5. **Prepare physical tails at the context limit.** A coarse bucket can extend past 262144 even when the real request fits. For example, `s=250016,n=12000` ends at 262016, but its full-attention physical bucket 16384 ends at 266400. Clamp dummy RoPE indices to an in-range value. Prepare a separate prefill attention map with at least 512 safe extra pages for the maximum 16384-row bucket: 8192+512=**8704 pages** at the configured maximum, aligned as required by SDPA. The flexible SDPA validator only checks KV length ≥ Q length, not start+Q (sdpa device operation:417–423); its reader must have valid table entries for dummy queries as well. The existing cache physical allocation need not grow when extra virtual entries alias a safe in-range page. Retain the configured decode table/capacity and its selected decode key-chunk policy; do not accidentally change decode policy by passing the extended prefill table. Discard every dummy query result. Validate this bound under watcher before accepting it.
6. **Bound whole-model work and staging.** M:136–140 embeds and applies the head to the entire logical prompt; decoder bucketing alone leaves those programs length-specific. Stream embedding/head through prepared fixed-size buffers. Keep a fixed-capacity residual staging arrangement so layer-specific chunks can differ without concatenating a variable list into a new request-shaped device tensor. One concrete conservative option is two interleaved BF16 residual buffers sized for configured context plus maximum bucket slack, with fixed-shape row gather and writes into runtime row offsets. This consumes substantial memory and needs a measured budget; it is not permission to reduce logical context. Reuse layer chunk workspaces and host-crop all-logit outputs to real positions; last-only generation should not materialize the whole prompt-by-vocabulary tensor.

Useful lowered API boundaries for this adapter:

- `ttnn.gather` accepts runtime uint index tensors and `out=`, with matching input/index layout and an index shape equal to output shape. It does **not** broadcast row indices across columns (`data_movement/gather/gather_nanobind.cpp:23–27,55–70`; device validation:54–104). A full-row history/logit gather must supply the required index shape and warm the wrapper's transpose/layout programs. BF16 projected history and residuals are suitable input types; do not assume BF8 gather support.
- Tensor-argument `ttnn.slice` is not a generic replacement for all these slices. Its device-only path requires TILE/unit-step and `slice_dim`/`num_devices`; output size is inferred as input dimension divided by `num_devices`. RM input falls back through host `to_vector`, and tile start coordinates lose sub-tile row bits (`slice/slice.cpp:477–585`; device operation:312–340; `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:69–77`). In particular it does not directly solve the three-row history lookup or RM page-table slicing.
- For eager staging writes, interleaved RM `slice_write` excludes outer-dimension starts from the key and recomputes offsets on a cache hit (`experimental/slice_write/device/slice_write_device_operation.cpp:86–96`). Input/output specs must still be fixed. Use already-interleaved RM staging: its public wrapper converts and reassigns tiled output storage (`slice_write/slice_write.cpp:119–125`), which would defeat stable backing identities. This runtime-host-offset path applies to eager work; a captured scalar offset would remain fixed on replay.

An alternative finer catalogue is every tile multiple up to each current chunk cap, with an explicit special policy around logical 1 and the 511/512 boundary. It reduces coarse padding but greatly increases preparation cost. Exact logical-length/start Cartesian enumeration is not an acceptable substitute for normalization. The proposed small catalogue should first be validated on representative layers and the real configured cache; its startup memory/time feasibility is not established by inspection.

### H3: preserve correct trace/buffer lifetimes

Keep the existing generator-owned sampler wrapper (G:179–184,227–229). The common high-level `SamplingGenerator.capture_trace` contains a capture-wide `corruptible_allocation_scope` (`models/common/sampling/generator.py:610`); switching to that wrapper would weaken the requested gate.

The existing exact acknowledgment of `state.logits` (G:224–226) has a defensible serial lifetime: a model replay produces it, then its matching sampling trace consumes it before another model replay. Preserve and test this ordering across **every** live binding. Token and position inputs allocated before the first capture need no blanket exemption. If new trace output buffers are acknowledged, list their aliases and consumers and prove each producer/consumer lifetime individually. Never acknowledge the 20 program buffers or wrap a whole forward, warmup, or capture to hide late compilation.

## Other concrete issues and test gaps

These are real code concerns or scope gaps, but they do not explain the sequence-matched 33→34 allocation failure and should not displace it as the headline.

### First low-level decode can destroy a prefilled cache

`decode_forward()` invokes preparation before copying the real token/position (G:248–253). On first use, preparation runs a mock decode at position zero on the supplied cache, then calls `self.reset()` (G:194–210). If callers first perform low-level prefill, this can erase the owned cache's real linear prefix. For external cache bindings, mock updates remain in the external cache because reset touches only `self.cache`. A pooled owned reset clears the entire linear backing through `states[0]`, including unrelated slots (G:104–107; D:505–520). High-level `generate` avoids this particular order by preparing first; that is why it is not the cause assigned to L.

Move preparation before real prefill/admission. Any necessary state preservation must restore the actual touched external/owned cache in place, not reset an unrelated pool or merely restore position inputs.

### Empty-cache resume negative control retains full-attention prefix

G:153 resets before resumed prefill, but G:101–107 deliberately leaves full-attention KV intact. This is appropriate for a fresh position-zero prefill that overwrites its valid prefix. It is not an empty cache for `prefill_prefix=False` at nonzero start. The installed readiness contract explicitly requires an empty-cache negative control (`runtime/readiness_check/contract.py:214–226`). Clear the relevant full-attention prefix in place for that diagnostic control using prepared operations, or use an independently prepared empty diagnostic binding. This does not imply clearing all KV at every normal request.

### Current binding/mode/test coverage is narrow

Changing table width on an existing G:189 key attempts to copy into old fixed device tensors (G:254–257); it does not select a new prepared signature. Decode batch/pool paths specialize on batch, complete ordered pool versus subsets, and table width (M:143; D:1194–1245,1322–1353). Normalize these into configured stable bindings or prepare their actual execution variants. Do not multiply traces by row order when normalized runtime inputs can represent it.

Regular/seeded sampling is not currently wired through generation kwargs. Preparation executes sampling and writes the token tensor; any future sampled mode needs explicit RNG restoration/initialization. Current greedy output is independent of RNG, so absent restoration is not evidence for this greedy failure.

The reduced trace probe prints tokens/counters but asserts neither reference correctness nor trace IDs. It does not cover continuation, changed page maps, external caches, batches/slot permutations, or mode switches. Identical token 220 from a reduced stack is not a full-model correctness result. No numeric/kernel theory for that token was established here.

## Discriminating tests for the repair owner

All device tests below are **proposed, not executed**. Use fresh processes with these settings before TTNN import:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
TT_METAL_TRACE_ALLOC_TRACEBACKS=1
TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=10
```

1. **Locate the first divergent program.** After creating the original length-33 traces, forbid program-cache misses around length-34 eager prefill. Expect the first miss to name embedding. Record program-cache entry count before/after prefill, not only capture/replay counters. A different first miss revises the ordering, not the broad late-preparation evidence. Dump binary buffer owner/ID/address/size if exact allocation attribution is needed.
2. **Controlled ordering contrast.** In a fresh diagnostic run, warm both exact 33 and 34 prefills before any capture, restore all request state, then execute 33→33→34→33. Passing supports H1 for these exact signatures; it does not establish arbitrary-length coverage or replace H2. Forbid misses across the request sequence after preparation.
3. **Signature normalization.** Compare requests 33 and 34 after adaptation: declared shapes and program IDs should match, while valid masks/history indices differ. Separately test same padded input with logical 33 versus 34 and fixed logical input with history start 30 versus 31 to distinguish tensor-spec and scalar-offset specialization. Cover `return_all_logits` true/false and single-new-token generation, followed by replay so late allocations cannot escape the check.
4. **State-neutral padding.** For logical counts 1,2,3,31,33,34,65,129,481,511,512,513 and a long awkward tail, compare the normalized layer's real outputs, convolution history, FP32 recurrent state, and subsequent decode against the unnormalized eager reference from identical initial state. Poison dummy tokens/hidden rows. Inactive `g` and beta must remain exactly zero regardless of their dummy values. Use the stage's existing accuracy tolerances, not newly relaxed tolerances.
5. **Planner boundaries.** Check immediately below/at/above 32,64,128,512,640,768,960,1024,4096,8192,16384,32768, plus 131071/131072/131073 and 262143/262144 where valid. The cheap arithmetic audit only covers explicit profile selection; actual cache keys, auto-plans/free-L1 behavior, and memory budgets must be measured. Compare valid lengths within the same bucket and across buckets.
6. **Arbitrary continuation and maximum physical extent.** Start at 1,2,31,32,33,127,128,16383,16384,32767, and near the context end. Use short suffixes and suffixes crossing a chunk boundary. Specifically test `[250016,262016)` (physical full bucket extends to 266400) and `[262143,262144)`; inspect fill sentinel handling, attention table index bounds, RoPE indices, untouched prefix pages, linear history, and the first subsequent decode. Compare full versus prefix+suffix on the same prepared generator; include the corrected empty-cache negative control.
7. **Generator-wide lifetime and state order.** Prepare all configured physical variants before the first capture, then interleave eager prefills, continuations, ragged multi-row prefills, and decode on disjoint slots while retaining the initial trace IDs. Exercise changed page-map contents and rows/slot orders supported by the configured bindings. Assert no new program entries/persistent resources, at most one capture per actual signature/binding, zero request-time releases/invalidations, and idempotent repeated setup. Prefill owned/external caches before first decode as a separate test to expose the current preparation mutation; assert unrelated pool slots are unchanged.
8. **Feedback, sampling, completion, and tracker control.** Prove model replay N+1 consumes the token written by sampling N and increments the actual captured position tensor once. Test changed and unchanged page tables. If sampled modes are implemented, exercise greedy→sampled→greedy with original trace reuse, and verify host/device RNG state across warmup. Complete CQ/readback work before comparing tensors. End each mixed workload with a decode replay. Keep a separate positive control allocating a deliberately live device tensor after capture and show that the tracker rejects it; then remove it and verify the real workload passes. Keep the exact logits acknowledgment and no broad allocation scope. Run full-stack correctness/qualitative checks after reduced probes pass; collect performance separately without tracking diagnostics.

## Adjudication after drafting

The report's headline was checked against the actual passing/failing sequence and lowered paths:

| Candidate | Direct support / required preconditions | Decision |
|---|---|---|
| Late eager prefill compilation behind live traces | All 20 stacks and contexts; decode-only warmup; same-shape reuse passes; 34 fails before replay | Leading diagnosis; explains observed boundary without speculative kernel behavior. |
| Only a tile-size change | Both 33 and 34 pad to 64, but logical specs/history/padding arguments differ | Rejected as an explanation; padded-shape-only warmup remains insufficient. |
| Retained ordinary outputs / missing Python GC | All IDs are program contexts; tracker reruns after GC; cache owns workloads | Not supported as cause of these 20 allocations. |
| Decode/sampler recapture or request-time trace eviction | Existing-state/capture early returns; 2 captures/0 releases in passing sequence | Not the observed path; retirement is not a permissible repair. |
| Wrong first-use warmup order corrupts cache | Concrete low-level API order; high-level generate prepares before real prefill | Separate real issue; does not explain L's 33→34 event. |
| SDPA, CCL, or precision kernel defect | No failing replay/incorrect-value evidence in this log | Unproven and unnecessary for the headline. |

The finite catalogue was additionally checked against every explicit Python branch/config class using a host-only integer script. The maximum-context physical-padding caveat was retained because flexible SDPA does not protect that bound automatically. No proposed runtime result is presented as verified evidence.
