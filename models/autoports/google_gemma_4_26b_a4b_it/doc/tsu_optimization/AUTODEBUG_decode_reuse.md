# AutoDebug: retaining Gemma4 decode traces across eager prefill

Source-only investigation, 2026-09-29. I did not run the model, use a device,
modify implementation code, or independently reproduce the measurements in the
TSU work log. The current worktree already contains an experimental configurable
prefill-trace limit and counters; this report evaluates the code as found.

## Verdict

**The current `prefill_prepared` requirement is an unnecessary coupling and is
the direct cause of the long-prompt decode recaptures.** It is not a trace-runtime
safety invariant. A one-entry, decode-only reuse path across a previously warmed,
exact-signature eager prefill is source-defensible and is the smallest promising
alternative to tracing 4K prefill.

The safe claim is deliberately narrow. The eager prefill must use an already
initialized program signature; all request-local device outputs and scratch that
were allocated after decode capture must be dead before replay; all tensors whose
addresses are embedded in the decode and sampling traces must remain alive and at
the same addresses; and queue, decode-binding, and sampling-graph contracts must
still match. Current allocator behavior permits this pattern, but source inspection
cannot prove that the full Gemma4 path has no hidden retained allocation. The
proposed tracker-enabled device test remains required.

This avoids the reported 6.1-second long-prefill capture because, with
`prefill_prepared is None`, decode `_capture()` does not call `_capture_prefill()`
([generator.py:566-595](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L566)).
It still pays ordinary eager-prefill TTFT. No performance improvement is claimed
without a matched serving measurement.

## Observations versus interpretation

Direct observations recorded by the existing work log are:

- Synchronous 4K/C1 serving recaptured decode for about 293-302 ms on every
  warmed request; the matched baseline cohorts were 23.28/23.24 ms TPOT,
  42.95/43.03 TSU
  ([work_log.md:73-89](models/autoports/google_gemma_4_26b_a4b_it/doc/tsu_optimization/work_log.md#L73),
  [work_log.md:149-163](models/autoports/google_gemma_4_26b_a4b_it/doc/tsu_optimization/work_log.md#L149)).
- Raising the prefill trace limit to 8192 produced two 4K cohorts at 20.66 and
  20.86 ms TPOT (48.41/47.93 TSU), with one decode capture and matching text for
  the first cohort,
  but the first 4K capture took 6111 ms; full-model 8K also exceeded the 1 GB
  trace region ([work_log.md:149-174](models/autoports/google_gemma_4_26b_a4b_it/doc/tsu_optimization/work_log.md#L149)).
- A reduced tracker-enabled probe reported matching eager controls for changed
  tokens and page mappings at multiple lengths
  ([work_log.md:99-123](models/autoports/google_gemma_4_26b_a4b_it/doc/tsu_optimization/work_log.md#L99)).

Those are prior runtime reports, not results of this investigation. The
source-supported interpretation is that the approximately 300 ms transition is
decode setup caused by an adapter/generator lifetime decision, rather than an
inherent requirement to record prefill. The measured TPOT contrast is consistent
with that interpretation, but only the proposed decode-only experiment can
separate trace retention from other effects of the 8192 candidate.

## Headline finding: the long-prompt recapture chain is explicit

1. `_serving_prefill_key()` returns no key above
   `prefill_trace_max_length`, whose default is 1024. A reusable prepared prefill
   therefore cannot exist for 4K input
   ([generator.py:96-100](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L96),
   [generator.py:296-316](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L296)).
2. The vLLM adapter derives `reuse_prefill` from that short-prefill key and passes
   it directly as `_reuse_trace` to `configure_sampling()`
   ([generator_vllm.py:198-229](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L198)).
3. When `_reuse_trace` is false, `configure_sampling()` calls the blanket
   `_release_trace()` ([generator.py:167-193](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L167)).
   That method releases both prefill trace IDs, the output trace, every common
   sampler trace, and the model decode trace
   ([generator.py:232-245](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L232)).
4. The long prompt then takes the eager fallback and the adapter sets
   `_sampling_signature = None`
   ([generator_vllm.py:230-250](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L230)).
5. First decode independently requires `prefill_prepared is not None` before it
   will preserve the trace through sampling reconfiguration
   ([generator_vllm.py:286-302](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L286)).
   `can_reuse_serving_decode()` duplicates that prerequisite even though its other
   checks concern only decode/sampling compatibility
   ([generator.py:318-338](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L318)).
6. With the trace gone, `decode_forward()` binds persistent inputs and calls
   `_capture()` again ([generator.py:615-632](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L615)).

Thus extending prefill trace eligibility removes recapture only because it makes
`prefill_prepared` exist. It also makes `_capture()` record the expensive prefill
graphs. The code has no independent representation for “this eager prefill
signature was already initialized and is safe to run while decode traces live.”

## Existing code already demonstrates the intended ordering

Standalone `Gemma4Generator.generate()` retains decode traces across repeated
eager prefills without `prefill_prepared`:

- Its reuse decision checks exact prompt length, sampling representation and
  mode, live decode state, owned-cache identity/capacity, greedy policy, and
  buffered-output capacity
  ([generator.py:741-754](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L741)).
- On a compatible request it retains the trace, refreshes stable table/position
  inputs, and executes eager `prefill_forward()` plus eager first-token sampling
  before replaying decode
  ([generator.py:755-798](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L755)).
- The focused test asserts that exact-length repeated and changed-token prompts
  keep the same trace ID, then compares the changed prompt with a fresh capture
  ([check_full_trace.py:56-73](models/autoports/google_gemma_4_26b_a4b_it/tests/check_full_trace.py#L56)).
  Its work log records a tracker-enabled pass
  ([full_model/work_log.md:143-158](models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/work_log.md#L143)).

This is strong evidence for feasibility, not a general proof. Standalone reuse is
restricted to the same logical length and canonical greedy policy, and its owned
cache/program history is more controlled than arbitrary vLLM scheduling.

## Historical trace-lifetime rule versus current allocator behavior

The earlier TTFT report correctly rejected this ordering: capture a prefill trace,
then let decode `_bind()` allocate new persistent inputs. It recommended preparing
long-lived prefill/decode buffers before capture
([TTFT AUTODEBUG.md:18-28](models/autoports/google_gemma_4_26b_a4b_it/doc/ttft_optimization/AUTODEBUG.md#L18),
[TTFT AUTODEBUG.md:59-66](models/autoports/google_gemma_4_26b_a4b_it/doc/ttft_optimization/AUTODEBUG.md#L59)).
That conclusion was sound for the examined direction, but “no allocation after
capture” is broader than the current allocator's implemented contract:

| Ordering | Lifetime result |
|---|---|
| Prefill trace, then persistent decode allocation | Unsafe: the new allocation remains live when prefill replays. |
| Decode trace, then warmed eager prefill whose temporaries all die | Permitted by the tracker; still requires exact-program and synchronization checks. |

There are two different memory subjects:

1. **Trace command storage is retained.** With a configured trace region, ordinary
   DRAM and `BufferType::TRACE` use separate bank managers
   ([allocator.cpp:43-90](tt_metal/impl/allocator/allocator.cpp#L43)). A live
   `MeshTraceBuffer` owns its host descriptor and device buffer
   ([mesh_trace.hpp:41-51](tt_metal/distributed/mesh_trace.hpp#L41)), and replay
   looks up that retained buffer and its address
   ([fd_mesh_command_queue.cpp:1258-1279](tt_metal/distributed/fd_mesh_command_queue.cpp#L1258)).
2. **The captured execution footprint is not reserved.** Capture copies runtime
   arguments and circular-buffer addresses into trace commands
   ([dispatch.cpp:3334-3385](tt_metal/impl/program/dispatch.cpp#L3334)); after
   assembly, the temporary trace nodes and their program references are cleared
   ([fd_mesh_command_queue.cpp:1638-1648](tt_metal/distributed/fd_mesh_command_queue.cpp#L1638)).
   The allocator states directly that memory used by a trace is no longer tracked
   after capture ([allocator.hpp:149-151](tt_metal/impl/allocator/allocator.hpp#L149)).
   The trace object therefore does not own every tensor whose address its commands
   will access.

The current safety tracker makes the narrower lifetime rule executable. Ending a
trace registers it as active; releasing it unregisters it
([mesh_device.cpp:1366-1424](tt_metal/distributed/mesh_device.cpp#L1366)). Every
later non-TRACE allocation is recorded against every active trace unless an
explicit suppression applies
([trace_allocation_tracker.cpp:117-135](tt_metal/impl/allocator/trace_allocation_tracker.cpp#L117)).
Immediately before replay, Python forces garbage collection; the C++ query drops
IDs that are no longer allocated and reports only surviving buffers
([unsafe_allocation_tracker.py:67-115](ttnn/ttnn/unsafe_allocation_tracker.py#L67),
[trace_allocation_tracker.cpp:138-174](tt_metal/impl/allocator/trace_allocation_tracker.cpp#L138)).
The regression test explicitly permits a post-capture temporary after it is
deleted, and rejects post-capture buffers that remain live
([test_single_device_trace.py:211-259](tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py#L211)).

This tracker is conservative: it compares allocation time and liveness, not
address overlap. A surviving buffer is a possible collision, not proof of an
actual one. Conversely, zero survivors is the structural condition needed here.
Tracking is disabled by default
([rtoptions.cpp:370-373](tt_metal/llrt/rtoptions.cpp#L370)); without it the
allocator only warns that later buffers *may* be corrupted
([allocator.cpp:118-130](tt_metal/impl/allocator/allocator.cpp#L118)). Production
code must enforce the contract rather than depend on the checker.

### Why exact warmup is part of safety

A later TTNN program-cache miss creates and caches a workload; program-cache
misses can be forbidden explicitly
([device_operation.hpp:402-438](ttnn/api/ttnn/device_operation.hpp#L402)). Kernel
binary backing storage is initialized lazily in persistent, top-down DRAM, and
capture itself rejects an uncached binary
([mesh_workload.cpp:131-180](tt_metal/distributed/mesh_workload.cpp#L131)). A new
binary or other module-persistent buffer created by eager prefill would survive
until replay and violate the contract.

“Warmed exact shape” must therefore mean the exact executable program signatures:
logical and physical shapes, dtypes, layouts, memory configurations, relevant
scalar/config values, layer policy, and every chunk variant. Equal logical length
alone is insufficient. A prior investigation found length 31 to 33 introduced 53
surviving program-cache allocations after capture
([AUTODEBUG_trace_lifetime.md:93-120](models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/AUTODEBUG_trace_lifetime.md#L93)).

The former retained sliding-attention tail is no longer a target-path survivor.
The production model constructs `MultichipDecoder`
([model.py:68-79](models/autoports/google_gemma_4_26b_a4b_it/tt/model.py#L68)),
which installs `_LocalAttention(OptimizedAttention)`
([multichip_decoder.py:874-954](models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py#L874)).
Long prefill marks only non-final chunks to retain a tail
([multichip_decoder.py:1243-1269](models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py#L1243)),
and `OptimizedAttention` stores `None` after the final chunk
([optimized_decoder.py:1417-1424](models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py#L1417)).

## Minimal concrete alternative

Add a one-entry **warmed eager-prefill signature attached to the decode trace
bundle**. Do not represent it with `prefill_prepared`, and do not create prefill
trace IDs or persistent prefill staging buffers.

Start with the measured B1/C1, start-position-zero, slot-zero, on-device canonical
greedy path. The state transition should be:

1. With no reusable trace, run eager prefill and first-token sampling normally.
   Let the adapter perform its blocking host token read. `ttnn.to_torch()` calls
   `from_device()` ([core.py:438-441](ttnn/ttnn/operations/core.py#L438)), whose
   default is blocking ([core.hpp:34-35](ttnn/cpp/ttnn/operations/core/core.hpp#L34)).
   Only after successful completion record that eager-prefill signature as warmed.
2. Bind/warm/capture decode and common-sampler traces as today. Attach the warmed
   prefill signature and the decode/sampling binding key to this trace bundle.
   “Decode trace” here means the model trace, common sampler trace, persistent
   bound tensors, and their ownership—not merely `self.trace_id`.
3. Before the next prefill calls `configure_sampling()`, compare the request with
   both keys. On an exact match, pass `_reuse_trace=True`, run eager prefill/sample,
   copy the first token to host, and drop all request-local device outputs before
   first decode replay.
4. Keep the adapter's `_sampling_signature` unset after eager prefill. On the
   first decode, permit trace retention through the pending eager-prefill marker,
   but still take the signature-change/reset path: refresh sampler state, force
   `reset_batch=True`, and copy explicit tokens, positions, and page tables rather
   than using device feedback from the preceding request. Then consume the
   pending marker. Remove the `prefill_prepared` prerequisite from both adapter
   and generator decode reuse. On any key mismatch, release the entire bundle
   **before** prefill can compile or allocate. Clear the attached warmed signature
   whenever the trace is released.

Merely deleting the two `prefill_prepared` checks is not sufficient: prefill has
already made the release decision before decode, and a coarse reuse decision
would admit unwarmed variants.

### Required keys and invalidations

| Key component | Retain only when | What may refresh in place |
|---|---|---|
| Device/queue domain | Same mesh, active sub-device manager, CQ, and synchronous request boundary | Nothing |
| Eager prefill program | Exact logical length, physical token/chunk shapes, start zero, slot zero, dtype/layout/memory configs and model policy were initialized before capture | Token values |
| KV cache | Same outer container **and every K/V tensor physical identity/address/spec** | Cache contents |
| Hybrid page tables | Same count and each table's shape/dtype/layout/memory config; same decode row contract | Page IDs copied into stable bound tables |
| Decode binding | B1, active slots `(0,)`, token-output mode, stable token/position/table/output tensor identities | Tokens and positions |
| Sampling graph | Device mode; same greedy versus sampled topology, penalty/log-prob state, force-argmax state, and common-sampler trace bucket | k/p/temperature/seed values only when their tensor refresh does not change that topology |
| Allocation lifetime | No post-capture prefill allocation survives to replay; all capture-bound tensors remain owned | Fully dead temporaries only |

Replay resolves a trace ID through the *currently active* sub-device manager, so
manager identity is not optional
([mesh_device.cpp:1460-1470](tt_metal/distributed/mesh_device.cpp#L1460)).

The sampling distinction is real: decode capture conditionally records seed
advancement when `sampled_mode` is true
([generator.py:495-512](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L495)).
The common sampler keys traces by penalty, log-prob, force-argmax, and bucket modes
and requires the identical input/output tensor objects
([sampling/generator.py:86-157](models/common/sampling/generator.py#L86),
[sampling/generator.py:290-310](models/common/sampling/generator.py#L290)). Numeric
sampling parameters are copied into persistent state, while a force-argmax mode
change resets traces ([sampling/generator.py:260-289](models/common/sampling/generator.py#L260),
[tt_sampling.py:593-655](models/common/sampling/tt_sampling.py#L593)).
The initial implementation should remain canonical greedy; expansion can use a
normalized graph key rather than `repr(params)`, which confounds values and list
padding with graph topology—the current generator already documents the compact
prefill versus padded decode mismatch
([generator.py:318-324](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L318)).

## Counterexamples that must release before eager prefill

1. **4096 to 4097, or any unseen executable signature.** Even a nearby length may
   select new slice/pad/chunk programs and persistent binary storage.
2. **Same length, different lowering.** A dtype, layout, memory config, layer
   policy, scalar/program attribute, or chunk path can change the program key.
3. **Nested cache replacement.** Current decode invalidation checks only
   `kv_cache is self.cache`
   ([generator.py:615-623](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L615)).
   Replacing a K/V tensor inside the same list can leave replay using a stale
   captured address; the stronger signature must inspect every tensor.
4. **Table/binding change.** Table count/spec/shape, batch, active-slot mask, or
   logits-versus-token output changes the captured graph or its addresses.
5. **Sampling topology change.** Greedy to sampled changes seed advancement;
   host sampling, penalties, log-probs, force-argmax, or bucket changes select a
   different graph. Seeds and numeric values are refreshable only within the same
   topology.
6. **Skipping the first-decode reset.** Setting `_sampling_signature` during
   prefill and then admitting `reset_batch=False` can select `device_feedback=True`
   and leave the prior request's token/position state in captured inputs. Retention
   must be independent of the mandatory request-boundary state refresh
   ([generator_vllm.py:286-324](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L286)).
7. **A retained eager result.** Returning or otherwise retaining device logits,
   sampled output, tail scratch, or a new module buffer leaves a tracked
   post-capture allocation live. The current adapter converts the sampled prefill
   token to host before returning
   ([generator_vllm.py:236-250](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L236)).
8. **Lost captured ownership.** If a captured token, position, cache, sampler, or
   output tensor is deallocated, the trace still has its raw address. The tracker
   detects later survivors, not loss of a capture-time owner.
9. **Async or cross-CQ overlap.** Model and sampler replay are nonblocking
   ([generator.py:661-675](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L661)).
   A new eager prefill must not race a prior replay; another CQ needs an explicit
   event/synchronization. The adapter exposes async decode readback and an event
   ([generator_vllm.py:329-336](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L329)).
10. **Unjustified tracker suppression.** `mark_corruptible` removes a buffer from
   the unsafe set; it does not itself prove that overwriting that buffer is safe
   ([unsafe_allocation_tracker.py:43-65](ttnn/ttnn/unsafe_allocation_tracker.py#L43)).
   The common sampler legitimately uses it for acknowledged bucketed trace I/O
   that another trace is intended to overwrite
   ([sampling/generator.py:26-36](models/common/sampling/generator.py#L26)). Do not
   extend that exception to arbitrary eager-prefill outputs or program-cache
   allocations merely to silence the checker.

## Other potential issues and ruled-out causes

- Decode `_capture()` has no exception cleanup around begin/forward/end capture,
  unlike `_capture_prefill()`'s nested `try/finally` and blanket release
  ([generator.py:406-425](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L406),
  [generator.py:566-595](models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py#L566)).
  A decode capture failure can leave stale metadata or capture state. This does
  not explain successful repeated 4K recaptures, so it is not a headline cause.
- The outer-only KV-cache identity check is insufficient against in-place nested
  replacement, but the adapter currently constructs and retains cache views and
  validates physical ownership
  ([generator_vllm.py:98-127](models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py#L98)).
  No evidence shows that current serving mutates those entries; treat it as a
  hardening requirement for the new signature.
- Final sliding-tail retention was a real historical trace-lifetime problem, but
  the current target path's final chunk clears it as shown above. It cannot explain
  the present long-prompt recapture.
- The tracker proves liveness order, not physical overlap. This limits claims
  about old failures, but the proposed design intentionally targets zero live
  post-capture allocations rather than relying on non-overlap.

## Required validation

Before selecting this alternative:

1. Add CPU contract tests showing `prefill_prepared=None` plus a live compatible
   decode bundle and matching warmed key preserves both model and sampler traces
   through prefill and first decode. Negative cases must cover changed length,
   nested cache identity, table shape/spec, batch/active slots, logits mode,
   greedy/sampled, penalties/log-probs, and host/device mode.
2. On device, run B1/4096 twice with changed token IDs and changed live page
   mappings. Assert the second request keeps both trace IDs, performs zero decode
   captures, and produces the same tokens as a fresh-capture control.
3. Run with `TT_METAL_TRACE_ALLOC_TRACKING=1`, tracebacks enabled, and
   `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. Forbid program-cache misses while
   the trace is live. Before every replay, require zero live unsafe allocations;
   do not add corruptible scopes or annotations for eager-prefill allocations to
   obtain the pass. Preserve the sampler's existing, explicitly justified
   trace-I/O annotation.
4. Include 4096 to 4097 as a mandatory release/warm/recapture negative control,
   plus a repeat of 4096 after the new capture.
5. Exercise the actual serving scheduler's async boundary, or explicitly limit
   the optimization to the synchronous C1 path until event ordering is proven.
6. Record cold eager-prefill, first decode capture, warm request, TPOT, TTFT,
   trace IDs/counters, generated text, and program-cache counts separately.

Static evidence answers the design question: decode-only retention is feasible
and better matched to the measured workload than full 4K prefill capture. It does
not replace the final tracker-enabled hardware qualification, especially for
Gemma4's model-specific CCL and asynchronous destruction paths.
