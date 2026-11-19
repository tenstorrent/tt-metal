# AutoDebug: sampling frees the decoder's persistent all-gather output

**Leading diagnosis in the original snapshot:** `SamplingCCL.line_all_gather()` returns borrowed decoder scratch to a caller that explicitly frees its gather results. The first eager sampling call force-deallocates the cached BF16 values output. The next sampling call retrieves that deallocated tensor and passes it as an optional persistent-output argument to the all-gather device operation. Generic argument validation can emit exactly **`Input Tensor is not allocated`**. This is a source-proven ownership violation and a strong match for the reported warmup-then-capture sequence. The saved log lacks a Python traceback or capture-key marker, so the exact failing call remains inferred.

**Smallest ownership repair:** return an independent `ttnn.clone()` from the sampling adapter, in the requested memory configuration. Alternatively, give this adapter sampler-owned all-gather outputs without using the decoder's persistent-output cache. Keep the decoder's persistent CCL policy intact. A cache-key change, another reshape, or another Python reference does not repair ownership. The live source gained a clone during this investigation; see provenance for the unverified concurrent repair.

Two additional trace defects and a preparation hazard need separate attention: reset capture performs a hidden host upload for row-major convolution state; failed capture leaves the queue recording and loses its trace ID; and preparation omits prefill signatures whose persistent allocations can outlive later trace replays. Reset and later prefill behavior do not explain the first invalid-tensor fatal.

## Scope and provenance

Inspection date: **2026-11-14**. Git HEAD: `7be8c4348666622686a6bdf5efa72e73ecf4c9b0`. Generator, model, and generator probe are untracked development files; HEAD alone does not identify this implementation. This investigation performed no source edits, target execution, device access, reset, process termination, or AutoDebug relaunch. It wrote only this report. Two independent xhigh investigators examined ownership and trace lifetime, followed by an independent xhigh report review.

**Concurrent source change:** after the first draft, the shared generator/model were edited outside this investigation. All generator/model findings and line references below diagnose the original captured source, rather than asserting that the later implementation still has each defect. The complete original 335-line generator dump is retained in `Q/doc/full_model/autodebug_sampling_retry.log:412-746`; the original 160-line model dump is at `:3889-4048`. Reconstructing the numbered lines in memory gives the original-source hashes below. An intermediate on-disk generator hash recorded before the draft was `796e6f4b26022db7f47cab722a7839bd539de921c46078a6d763abb0b8ceffc9`; that is a separate observation, not the checksum of the initial dump. The critical ownership/reset/capture code is shown in the original dump. No changed source was reverted.

At the **01:32 UTC** review check, live generator SHA256 was `ebc26ff457bdf9f32d30fa863b53fd248e0b0f2c577335f325307b5c097dc4bd` and live model SHA256 was `215bb03453abb38a89b6ac1c28bb65100e45f5c2f03653ce8fcf07b9d0e3db3f`. The generator then contained an explicit clone at line35, preallocated reset sources at lines56-65, device-only reset copies at lines159-161, and initial exception cleanup at lines167-175. These implement the proposed ownership/reset boundaries in source. This inspection does not verify their execution or certify complete exception safety: an error in end-capture/cleanup can still bypass release or mask the original error. Prefill preparation remained explicitly unfinished at lines183-202. Later edits may supersede these review-time hashes.

Abbreviations below:

- **Q/** = `models/autoports/qwen_qwen3_6_27b/`
- **S/** = `models/common/sampling/`
- Other paths start at the repository root. Generator/model line references describe the original captured source; native/common references describe the stable inspected checkout.

| File | Inspected SHA256 |
|---|---|
| Q/tt/generator.py, original numbered dump reconstructed with LF newline | `2bad87b853653117c9a4d15275392f1e5ae147be98407808ad5a9fa89d76eed4` |
| Q/tt/model.py | `0097cd34f4f7f0dd89b164c5b733eac205ef14254dfd909215ef7f7c6b475e5b` |
| S/tt_sampling.py | `7ae3725caa32d8143963117b42035101608f46e6954d17cb0a8c33a9dcb67afb` |
| Q/tt/multichip_decoder.py | `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b` |
| Q/tests/probe_full_generator.py | `af829d2d08cadeff1e59086e6f8d2e4408057650d2d3f2b7273927cf3c67e109` |
| Q/doc/full_model/generator_reset.log | `e917c4b11f8300328b52a07e2a0dd5c841d9bc0e4112c88ff53aa0474d89fe29` |

## Direct observations and sequence inference

| Evidence | Direct observation |
|---|---|
| `generator_reset.log:21-24` | Layers 0 and 3 load. At `01:21:20.774`, an active-trace allocation warning immediately precedes the invalid-input fatal. No sampler/capture phase markers or host stack appear. |
| `Q/tests/probe_full_generator.py:23-31` | Inspected probe uses a 1x4 mesh, 100,000,000-byte trace region, default B1, max sequence 1024, reduced layers `[0,3]`, and a 33-token eager `prefill_logits()` before `generate(...,5)`. |
| `Q/tt/generator.py:123-140` | Preparation warms decode, sample, reset; then captures decode, sample, reset in that order. |
| `Q/doc/full_model/work_log.md:19` | Requester records capture failure followed by shutdown hanging. This is context, not a host stack identifying the shutdown stop site. |
| Saved `triage/sampling_capture.txt:167-168,256-273` | Inactive-CB pass; Ethernet NoC-counter mismatches whose diagnostic lists multiple possible interpretations. Subsequent device stacks do not identify the Python fatal call. Summary checker “pass” labels mean the checker ran, not that every embedded diagnostic was healthy. |

**Inference:** the allocator warning is enabled when a trace is registered at `end_mesh_trace()`, rather than by opening the first capture (`tt_metal/distributed/mesh_device.cpp:1752-1757`; `tt_metal/impl/allocator/trace_allocation_tracker.cpp:103-112`; `allocator.cpp:127-140`). On this probe's flow, that strongly suggests decode capture reached its end and a later capture was running. End-capture registers even on an exception, so the warning alone does not prove success. Combined with the ownership chain below, **sample capture is the leading failure site**. The log does not directly prove warm sampling completed; the current source and reported sequence predict it.

## 1. The first sample frees a borrowed persistent values buffer

### Effective configuration

This is ordinary local-top-k sampling, not force-argmax. `Q/tt/generator.py:57-67` supplies no enabling model config; `S/tt_sampling.py:265-266,301-307,348-364` gives multi-step reduction=false, physical sampling batch32, `max_top_k=32`, DRAM sampling memory, and force-argmax disabled. Greedy parameters (`k=1,p=0,temp=1`) still run that pipeline.

`Q/doc/full_model/probe_positions.json:7-16` reports vocabulary 248320, giving local BF16 TILE/DRAM logits **`[1,1,32,62080]`**, including 31 inactive rows for B1. Power-of-two padding makes top-k input **`[1,1,32,65536]`** (`S/tt_sampling.py:1331-1365`). On Blackhole, this eligible unstable call routes through the `topk_large_indices` composite (`ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:282-373,619-645`). Final outputs are BF16 values and UINT16 indices, each **`[1,1,32,32]`**. Route finish chooses UINT16 at padded width 65536; widening requires width strictly greater than 65536 (`ttnn/cpp/ttnn/operations/reduction/topk/device/topk_route_finish_device_operation.cpp:138-149`; `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_utils.hpp:65-69`). Both gathers produce **`[1,1,32,128]`**. The ownership bug does not depend on the top-k engine, but these values establish the actual persistent branch and dtype separation.

### Ownership ledger

| Step | Operation | Ownership/state |
|---|---|---|
| 1 | Sampler gathers local candidates with `buffer_key="SAMPLING_VALUES"` (`S/tt_sampling.py:1370-1378`). | Local top-k values are caller-owned gather inputs. |
| 2 | Adapter removes a leading singleton and calls `model.io._collective()` (`Q/tt/generator.py:28-32`). | Adapter ignores `buffer_key`; input becomes `[1,32,32]`. |
| 3 | `_collective()` selects DRAM, repacks to `[1,1,32,32]`, enables persistence for rows32<=32 (`Q/tt/multichip_decoder.py:446-475`). | Layer0 defaults include `persistent_ccl=True`, `ccl_decode_memory="DRAM"` (`:57-59`). Width32 bypasses optional dtype conversion (`:457`). |
| 4 | First call allocates `[1,1,32,128]`, retains it in `_ccl_buffers[signature][1]`, supplies it as persistent all-gather output (`:476-508`). | Decoder owns this buffer. Native output creation returns the supplied tensor itself (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.cpp:281-285`). |
| 5 | Decoder returns `[1,32,128]`; adapter returns `[1,1,32,128]`. | Both are metadata views of the same root. No output typecast runs because dtype remains BF16 (`Q/tt/multichip_decoder.py:511-513`). |
| 6 | Sampler aliases the result as `topk_values_gathered_bf16_interleaved` because sampling memory is DRAM (`S/tt_sampling.py:1393-1394`). | Still borrowed decoder scratch. |
| 7 | Sampler calls `ttnn.deallocate()` on that result (`:1498`). | **Decoder's cached root allocation is forcibly freed.** |
| 8 | Second sample finds the same signature; no allocation-state guard exists (`Q/tt/multichip_decoder.py:475-495`). | Stale tensor becomes the next gather's optional persistent-output argument; host validation rejects it. |

These reshapes preserve their final two dimensions and take the view branch (`ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:757-770`). Device view construction retains the root owner (`ttnn/core/tensor/tensor_ops.cpp:465-478`; `storage.cpp:157-161`). This is not an assumption based on Python object identity.

Public `ttnn.deallocate()` defaults to **force=True** (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:185-195`). Forced deallocation bypasses shared-reference and shared-device-memory protection (`ttnn/core/tensor/tensor.cpp:125-148`); `DeviceStorage::deallocate()` explicitly frees a view's root (`storage.cpp:230-240`). Retaining a cache reference cannot save this allocation.

### Why the fatal calls an output an “Input Tensor”

`AllGatherAsyncInputs` contains both `input_tensor` and optional `persistent_output_buffer` (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation_types.hpp:120-123`). Generic launch reflects **all Tensor fields**, then validates them before output creation or dispatch (`ttnn/api/ttnn/device_operation.hpp:617-630`). The allocation assertion at line565 uses exactly the observed text. A bad optional output can thus be reported as an invalid input.

The native composite predicate does not divert these tiled, gather-dimension-aligned candidates into a copying route (`ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp:286-305`). The supplied persistent output reaches that validation. No kernel execution, queue timing race, or allocator exhaustion is required.

### Affected and unaffected intermediates

- **Freed:** persistent BF16 values root, through `S/tt_sampling.py:1498`.
- **Not freed by that chain:** persistent UINT16 indices root. Its dtype gives a distinct cache key. `typecast(...int32)` at `:1421-1423` allocates separate storage (`ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp:162-167`); deallocating the cast at `:1441` does not free the UINT16 root.
- Local top-k tensors freed at `:1380,1405` are distinct collective inputs.
- There is no demonstrated decoder-statistics collision for this B1 probe. Default decode norms gather hidden width1280 (`Q/tt/multichip_decoder.py:598-606,1305`). Distributed statistics preserve logical row count; B1 does not become logical row32 merely because it occupies a tile (`ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_device_operation.cpp:91-101`).

### Minimal remedies

**Smallest local change:** clone the reshaped result before returning from `SamplingCCL.line_all_gather()`, honoring requested memory configuration. Clone always allocates independent output (`ttnn/cpp/ttnn/operations/data_movement/clone/device/clone_device_operation.cpp:65-68`). Warm that added path before capture. Common cleanup then frees its own clone and leaves the decoder root allocated.

**Cleaner ownership boundary:** implement sampler gather directly using the existing initialized CCL semaphores/topology, without a persistent output buffer. The common sampler then owns its output. This is the sibling pattern in `models/autoports/aleph_alpha_kolibri_1_bf16/tt/generator.py:51-60`. `models/demos/muse_glimmer_30b_qb2/tt/model.py:83-111` explicitly excludes `SAMPLING_*` outputs from persistent manager buffers for this ownership reason. A per-call nonpersistent option in `_collective()` is another localized implementation. Avoid temporarily mutating shared decoder policy inside a graph.

Adding `buffer_key` to cache signatures can separate namespaces but cannot stop the sampler freeing its cached values output. Another reshape is another alias; matching `to_memory_config(...DRAM)` returns its input (`ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp:261-265`). An allocation-state guard would recreate stale state without establishing ownership/trace lifetime. Do not remove common cleanup globally to accommodate this adapter.

## 2. Reset capture still performs a hidden host write

`Q/tt/generator.py:104-108` captures `ttnn.copy(ttnn.zeros_like(tensor), tensor)` for every cache tensor. Layer0 convolution history is **BF16 ROW_MAJOR `[B,3,2560]`** (`Q/tt/multichip_decoder.py:758-763`). This zeros_like is not device-only: `full_like_impl()`'s device-fill path requires supported **TILE** input; row-major falls through to `full_impl()` (`ttnn/cpp/ttnn/operations/creation/creation.cpp:282-313`). That function excludes row-major device fill (`:206-219`), builds a host BF16 tensor, and calls `to_device()` (`:236-248,68-89`). `ttnn/core/tensor/tensor_ops.cpp:156-158` enqueues its host write; `tt_metal/distributed/fd_mesh_command_queue.cpp:621,667,876` rejects writes during capture.

The eager warm reset can therefore succeed while later reset capture fails with a write-not-supported error. Device residency of the zeros_like input does not change that implementation. This is distinct from the current invalid-tensor signature and would be reached after repairing sampling. The earlier `generator_params.log:23-42` documents a removed in-place RM multiply error; it does not certify the replacement's trace safety.

**Minimal fix:** allocate and retain immutable same-layout zero-source tensors before any capture, then capture only device copies into state. Keep the sources until traces are released. A verified device-only RM fill could avoid extra storage, but current zeros_like and the previously rejected RM in-place multiply are not that implementation. Tiled BF16 recurrent and tiled BF8 KV state use other paths; this finding does not say every zeros_like call uploads.

## 3. Failed capture leaves the queue recording and loses the partial ID

`Q/tt/generator.py:110-117` stores a trace ID only after graph execution and end-capture succeed. If the sample graph raises, capture is never ended, the partial trace is never released, and its ID never enters `self.traces`. `close()` (`:325-328`) releases only known successful traces.

`FDMeshCommandQueue::record_begin()` sets `trace_id_` and enables sysmem bypass (`tt_metal/distributed/fd_mesh_command_queue.cpp:1626-1643`). `record_end()` clears trace state, restores dispatch state, and disables bypass (`:1886-1902`). `release_mesh_trace()` releases resources/unregisters the ID (`tt_metal/distributed/mesh_device.cpp:1686-1703`); releasing alone does not end recording.

**Minimal fix:** preserve each opened ID; on failure, end an open capture through the supported runtime sequence, release the partial trace, and re-raise the original graph error without masking it with cleanup errors. Handle end-capture failures too, checking whether recording remains active. Do not mark failed traces prepared/executable.

This can plausibly contribute to shutdown hanging because mesh close receives altered queue state. It does **not** prove the first stuck producer in the saved shutdown. The evidence lacks a host close stack and an operation-to-fatal mapping. No independent fabric/NoC root cause is established.

## 4. Preparation omits later persistent prefill allocations

`prepare()` warms only fixed-batch decode/sample/reset (`Q/tt/generator.py:123-138`); its docstring says prefill preparation is unfinished. `generate()` prepares before request prefill (`:277-286`); `prefill_forward()` accepts varying lengths/starts and runs eager model prefill (`:188-206`). The probe separately primes length33; normal construction has no such call.

For a concrete missing lifetime, consider a later **B1 length2** prefill after length33 priming and trace capture. `Model.terminal(...decode=False)` (`Q/tt/model.py:126-127,144`) reaches layer0 distributed norm (`Q/tt/multichip_decoder.py:598-633`). New statistics and hidden-gather shapes have two logical rows. `_collective()` enables persistence for <=32 rows and lazily retains exact-shape outputs (`:469-494`). Decode warmed one-row hidden gathers; sampler warmed32-row candidates; the earlier33-row prefill did not create these two-row entries. New buffers are allocated after live traces and retained when decode next replays.

Trace replay reuses recorded scratch addresses without rerunning the allocator, so a surviving later allocation can overlap them. New shape-dependent program-cache buffers carry the same risk (`tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md:40,44-62`). Missing setup and surviving allocations are source facts; actual overlap/corruption requires runtime evidence. This is a **later cross-request hazard**, not the first capture fatal. The same33-token retry alone does not prove this length2 scenario occurred.

**Repair:** prepare persistent helpers, supported physical prefill/chunk/start variants, terminal/slicing paths, sampler and reset before first capture, then capture against prepared state. Physical buckets/runtime inputs can preserve supported logical lengths. For this reduced probe, also prewarm its exact return-device/last-logit slicing and padding path: an eager `prefill_logits(...return_all_logits=True)` does not certify that terminal path. Reject program-cache misses during capture to expose missing preparation; separately check allocations made during later eager prefill.

## Candidate adjudication and other potential issues

| Candidate | Source evidence | Sequence match and uncertainty | Intervention |
|---|---|---|---|
| Borrowed gather output force-freed | Proven root ownership violation | Predicts first sample succeeds, second sample/capture raises exact generic assertion; exact capture marker/stack missing | Independent returned storage or sampler-owned gather output |
| RM reset zeros_like upload | Proven host fallback inside reset graph | Predicts later reset-capture write error, after current failure | Preallocated zero sources/device copies |
| Partial trace cleanup omitted | Proven missing end/release path | Queue stays recording after graph exception; actual shutdown wait remains unresolved | Exception-safe capture lifecycle |
| Prefill setup omitted | Proven surviving later exact-shape allocations | Potentially unsafe surviving allocations on later varied requests; overlap/corruption unverified | Prepare physical variants before capture |
| Values/indices collision from ignored buffer_key | Current dtypes already distinguish keys | Not needed to explain B1 fatal; future same-dtype users could collide | Namespace any retained cache, plus ownership repair |
| Decoder temporary dies after copying logits | No contradiction found | Ordinary trace scratch can be freed after capture construction | No blanket intermediate retention |
| Generic missing decode/sample warmup | Selected mode uses the exact same warm/capture functions, including plus_one | No separate missing signature established | Fix invalidated buffers instead of repeating identical warmup |
| Kernel arithmetic, fidelity, capacity or fabric failure | No primary violation established for this fatal path | Host validation explains fatal before native dispatch; triage does not map another cause to it | No kernel/precision/resource-policy rewrite justified |

Tokens, positions, logits, caches/page tables, weights, and RoPE tables are created and retained before capture (`Q/tt/generator.py:44-66`; `Q/tt/model.py:63-90`). `_decode_graph()` copies its local output into persistent logits (`generator.py:95-98`); discarding that local output is valid trace scratch behavior. Token feedback already uses `tt_out_tok=self.tokens` (`:100-102`). No host-token reconstruction is needed for the ownership repair.

## Completion boundaries and focused follow-up checks

Warm Python calls primarily enqueue work on CQ0; reaching the graph's end alone is not device correctness proof. Capture records operations rather than running the modeled step. `_execute()` is nonblocking (`Q/tt/generator.py:119-121`); token/logit reads (`:87-93`) provide caller-visible readback boundaries. The current log records no successful generation/readback. The ownership assertion is synchronous host validation: waiting for device completion cannot restore its freed root. An allocation warning alone does not prove corruption.

Recommended checks for the repair owner, **not executed here**:

1. Add capture-key/graph-entry/exit markers. After eager sampling, inspect `_ccl_buffers` for `[1,1,32,32]` inputs: the current BF16 output should report unallocated while UINT16 remains allocated. This is the strongest falsification check for the leading chain.
2. Run two eager samples and sample capture after the ownership change. Verify cloned returns have distinct storage and cached roots stay allocated after common cleanup. Keep semantic greedy parameters and32 tile-shaped local candidates.
3. Capture/reset the actual RM convolution cache; verify every state becomes zero on replay without host writes. Zero sources must predate every live trace.
4. Exercise a graph exception after begin-capture; verify recording ends, partial ID is released, and cleanup preserves the original exception.
5. In a fresh validation process, enable `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` before TTNN import; include a positive control proving tracking is active. Interleave changed-token/position decode, sample, reset, and different-length/resumed prefill; finish with replay to check late allocations. Test repeated token feedback and changed/unchanged page tables. Do not hide failures with broad allocation exemptions.

Static checks here used only Python standard-library AST/JSON/hash inspection: verified absent prefill preparation and capture exception handling, confirmed zeros_like in reset, and derived vocabulary/shard/padding/local/gathered dimensions248320/62080/65536/32/128. No TTNN import, build, or hardware reproduction was used.

## Post-draft review

Completed independent xhigh review against the original source dump and stable native/common code. The reviewer confirmed the ownership/root-deallocation/optional-output validation chain, concrete top-k dimensions and dtype separation, RM host fallback, partial-capture queue state, and later prefill allocations. Review corrections separated concurrent repairs from the original snapshot, replaced categorical prefill-corruption wording with a potential hazard, and made the top-k utility reference explicit. Standard-library reconstruction verified the original dumps, and the shared sampler/decoder hashes remained unchanged. Confidence is high in the source ownership violation, moderate that sampling capture is the saved fatal's precise site, and unresolved for shutdown/fabric causality. No hardware confirmation or successful-repair claim is made.
