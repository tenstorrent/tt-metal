# Allocation warning between model and sampler trace captures

2026-09-27. Inspection only; no device, server, or profiler commands were run. The TT tracing skill was consulted. The commands below are proposed verification for the main agent after its serving workload releases the mesh.

## Classification

`readiness_vllm/server.log:156` records the generic allocator warning at 15:00:16.737, immediately before the first `SPLIT_TRACE_READY` marker. The strongest source explanation is the sampler's ordinary temporary allocations during its second trace capture, after the model trace has become active. This is a **conservative lifetime warning, with no demonstrated address overlap or output corruption**. It must not be classified as harmless solely from successful stale-input, page-growth, async, or token-equality tests.

The message cannot identify the exact operation: `AllocatorImpl::verify_safe_allocation` (`tt_metal/impl/allocator/allocator.cpp:118`) only checks `allocations_unsafe_` and emits at most once per host thread for the entire process. It does not compare addresses, distinguish surviving versus freed tensors, or exclude a second trace's internal scratch. Consequently, one warning does not mean there was only one later allocation.

`MeshDeviceImpl::end_mesh_trace` (`tt_metal/distributed/mesh_device.cpp:1418`) registers the trace on exit. With tracking disabled, `AllocatorImpl::register_active_trace` sets `allocations_unsafe_=true`; only release of all active traces clears it. Beginning another capture does not clear it.

## Allocation boundary and ownership

The current `Gemma4Generator._capture` sequence is:

1. Clone the token/position/seed restoration tensors and warm `_forward` before any new trace exists.
2. Precompile common sampling against the warmed logits, also before model capture.
3. Capture `_forward`, retaining `self.trace_logits`, and end the model trace. The model trace is now active.
4. Call `SamplingGenerator.capture_trace(self.trace_logits, tt_out_tok=self.tokens, skip_precompile=True)`. The sampler opens its own trace and runs `_run_sampling`. Normal top-k, mask/padding, gather, type-conversion, and tie-breaking intermediates allocate storage behind the live model trace. The first such allocation is the most likely warning site. The log alone cannot distinguish its exact op.
5. End the sampler trace; restore preallocated token/position/seed buffers; synchronize and print `SPLIT_TRACE_READY`.

The common sampler explicitly provides `precompile` for this split-capture case (`models/common/sampling/generator.py:346`). The autoport already uses the required ordering and `skip_precompile=True`; an accidental eager precompile behind the model trace is therefore not supported by the current source.

Most sampler intermediates are locals, and `TTSampling.forward` explicitly deallocates sampling values, gathered values, global indices, and untilized indices near its return (`models/common/sampling/tt_sampling.py:1139`). Its token output writes the previously allocated `self.tokens` via `output_tensor`; this TP4 canonical path rejects device log probabilities, so no newly retained log-probability output is expected. This makes transient sampler scratch plausible, but only runtime lifetime tracking can establish that no unexpected allocation survives.

`_replay` itself executes the model then sampler traces. `_format_tokens` supplies its preallocated `public_tokens` output; `_copy` creates a host TTNN tensor and copies into an existing device tensor. These call sites do not intentionally create new persistent device tensors per canonical decode step. Internal op or program-cache allocations still need the tracker check.

The common sampler's `_mark_trace_buffers_corruptible` is gated on a non-None trace bucket. This generator does not set a trace bucket, so that helper does not silently exempt its outputs. No new exemption or broad `corruptible_allocation_scope` should be added to make verification pass.

## Tracker availability and meaning

A host-only import verified the actual runtime resolves TTNN to `/home/container_app_user/tt-metal/ttnn/ttnn/__init__.py`, and its compiled trace module exposes `get_unsafe_tracked_ids`, `remove_unsafe_tracked_id`, and `drain_pending_traceback_ids`. The current inspection process had tracking and diagnostics disabled. The matching Python tracker exists in that installed tree.

Set these variables **before starting Python/importing TTNN**:

- `TT_METAL_TRACE_ALLOC_TRACKING=1`: enables allocation accounting and the Python `ttnn.execute_trace` pre-replay verifier.
- `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`: adds allocation stacks and Python referrer diagnostics on failure.
- `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`: retains program-cache allocations in the check.
- Optional `TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=12` broadens object traversal for diagnostics.

The switches are captured at process startup in both Python (`ttnn/ttnn/trace_allocation_config.py`) and C++ runtime options. Setting them after import is insufficient. No explicit report API or source change is required: with accounting enabled, every ordinary `ttnn.execute_trace` calls `UnsafeAllocationTracker.verify_before_replay`, runs host GC, queries live candidate buffers for that specific trace, and raises `RuntimeError` before dispatch if any remain. Redirect stdout/stderr to preserve the buffer IDs, op contexts, stacks, and referrers. There is no automatic standalone JSON tracker report.

The tracker is conservative lifetime accounting, **not a byte-range overlap detector**. It records allocations made after a trace was registered, excluding trace storage and explicitly acknowledged allocations, and retires freed buffers. A failure identifies a concrete live allocation needing explanation; its wording about corruption is stronger than the tracker's actual address-free test. A clean replay proves no unacknowledged post-registration allocation survives for that exercised trace, not universal allocator nonoverlap for untested shapes or paths.

An optional diagnostic query is `ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace_id)`, returning `{buffer_unique_id: allocation_context}`. It is useful **while the trace is live**, before replay/release. Querying after release loses the trace's accounting and cannot establish safety. Comparing only public input/output addresses likewise cannot exclude overlap with freed capture intermediates.

## Minimal functional verification

After the live serving suite finishes and the mesh is free, run the existing reduced direct-adapter test in a fresh process. This is a correctness check, not profiling; no Tracy, device-profiler, or serving-performance collection is required. From `/workspace/tt-metal`:

```bash
env TT_METAL_TRACE_ALLOC_TRACKING=1 \
    TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
    TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=12 \
    TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/runtime_logs \
    python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_adapter \
    --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/adapter_trace_allocations.json \
    > models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/adapter_trace_allocations.log 2>&1
```

Acceptance requires exit0 plus the existing token-equality, stale-token/position, async-read, and changed-page-without-recapture assertions. Both model and sampler `execute_trace` calls are automatically checked. Preserve the log and JSON together with the environment settings and runtime source paths; absence of the generic warning alone is insufficient because tracking replaces that warning path.

If it fails, preserve the first traceback before any release/cleanup. Classify the named survivor: persistent inputs/state must be allocated before model capture; transient tensors must lose all owners before replay; sampler scratch intentionally overwritten before use needs a specific producer/consumer ownership argument. Do not skip all program-cache buffers or mark all sampler outputs corruptible to suppress the evidence. First locate the exact allocation/referrer and verify its lifetime.

This reduced invocation validates the canonical split-sampling lifecycle represented by layers0/5. It does not establish full30-layer, B32, penalties, or host-logits-mode allocation safety. If the reduced trace reports no survivors, retain that bounded result and use the same tracker configuration on the next necessary full-model functional check for a broader claim. The current source-only investigation leaves exact warning-op identity and runtime lifetime safety unproven until such evidence is collected.
