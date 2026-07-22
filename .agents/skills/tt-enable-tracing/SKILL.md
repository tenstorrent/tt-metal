---
name: tt-enable-tracing
description: "Enable or debug TTNN trace capture and replay for models, decoders, generators, and multi-chip TTNN code. Use when eliminating TTNN dispatch overhead, implementing traced prefill/decode, making full-model generation trace-safe, or investigating trace failures such as unsupported writes, reads, event synchronization, stale inputs, or bad replay correctness."
---

# TT Enable Tracing

Use this skill when adding TTNN trace capture/replay or when an existing traced path fails. Trace replay is usually the difference between a good device implementation and a usable inference path: the device kernels may be fast, but eager Python/TTNN dispatch between many ops can dominate decode.

For the readiness harness, teacher-forcing decode must always use traced decode. Do not treat an eager teacher-forcing pass as acceptable evidence for optimized full-model or datatype-sweep performance; fix trace capture/replay first.

## Useful References

Read only what helps the current task:

- `tests/ttnn/tracy/test_trace_runs.py`: minimal trace capture/replay examples.
- `models/tt_transformers/tt/generator.py`: canonical decode trace patterns, including host input preparation, persistent device inputs, replay refresh, and split sampling.
- `models/common/sampling/generator.py` and `models/common/modules/sampling/sampling_1d.py`: common on-device sampling implementations to compare before choosing a token-out sampling path.
- `models/tt_transformers/tt/model.py`: model-side `prepare_decode_inputs_host` and device-only `ttnn_decode_forward` split.
- `advanced_perf_optimizations.md`: deeper examples for TTNN trace capture/replay, multiple command queues, trace plus multi-CQ, and production benchmarking patterns. Search this file for the API or failure mode you are working on before loading it wholesale.

## Mental Model

Trace capture records a static command sequence on a device or mesh device. Replay reuses that exact sequence without paying per-op host dispatch.

The traced region should be device work over stable device tensors. Host-originated input changes happen before replay by copying into those stable tensors. Outputs that are produced by TTNN ops inside capture are fine; host writes, host reads, and host synchronization while the trace is open are not.

Traced warmed replay is the only valid perf-decision metric here; eager/untraced timings can invert the sign of a comparison, so never select a config on them.

When opening the device, reserve trace space:

```python
device = ttnn.open_device(device_id=0, trace_region_size=100000000)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, num_chips),
    trace_region_size=100000000,
)
```

## Trace-Safe Shape

Prefer this structure:

1. Build all weights, caches, page tables, semaphores, persistent CCL buffers, and lazy module state before capture.
2. Create host-side input tensors with `device=None` if useful.
3. Copy or allocate stable device input tensors before capture.
4. Run one warm compile call with the same shapes and mode so every op is compiled (see Program-Cache Warmup).
5. Begin trace capture.
6. Call a device-only forward method that consumes the stable device tensors.
7. End trace capture.
8. For each replay, update stable device inputs outside capture, then call `ttnn.execute_trace`.

Minimal pattern:

```python
trace_input = ttnn.from_torch(
    dummy_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Warm compile outside trace.
compiled = model_forward(trace_input)
ttnn.synchronize_device(device)

trace_id = ttnn.begin_trace_capture(device, cq_id=0)
trace_output = model_forward(trace_input)
ttnn.end_trace_capture(device, trace_id, cq_id=0)

for batch in batches:
    batch_host = ttnn.from_torch(batch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(batch_host, trace_input)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
```

## Program-Cache Warmup

Trace capture cannot compile programs. A program-cache miss inside capture forces a new kernel build, which issues a host->device write and aborts with `Writes are not supported during trace capture`. So every op in the traced region must already be compiled (warmed) with the *exact* program-cache signature it will have during capture.

- Warm with the same shapes, dtypes, layouts, memory configs, and mode as capture; the warm call must drive the identical op sequence and code path so every op variant is compiled.
- The signature can include arguments you would not expect. For example, the integer `begins`/`ends`/`step` passed to `ttnn.slice` are compile-time constants baked into the program hash, so slicing at a different offset, length, or start-tile alignment is a different program that needs its own warm-up. (There is a version which tensor-valued arguments that avoids this.) When in doubt, warm with the same argument *values*, not just the same tensor shapes.
- Warm state-update ops too. Autoregressive helpers such as `ttnn.plus_one`, page/position tensor updates, sampler trace setup, and persistent-output buffer allocation are easy to forget because they are not "the model", but they still compile programs and allocate resources.
- If warm-up mutates persistent trace inputs such as token, position, RoPE index, page table, or KV-cache state, reset those tensors to the exact intended capture state immediately before `begin_trace_capture`.
- If you still hit an unexpected program-cache miss during capture, warm up again immediately before `begin_trace_capture` (re-run the exact forward once more, then capture with nothing else in between). In rare cases an op's program-cache signature depends on transient device state such as free L1, so a warm-up done earlier no longer matches by the time capture runs. See https://github.com/tenstorrent/tt-metal/issues/46533.

## Generator Pattern

Do not trace the high-level generator method unless it is already proven trace-safe. Split the generator into:

- `prepare_decode_inputs_host(...)`: returns host TTNN tensors or torch values for token ids, positions, RoPE indices/tables, page tables, masks, and other per-token inputs.
- `copy_host_to_device(...)` or explicit `ttnn.copy_host_to_device_tensor(...)`: refreshes stable trace input tensors before replay.
- `decode_forward_from_ttnn_inputs(...)` or `ttnn_decode_forward(...)`: device-only model call used for warm compile and capture.
- `decode_next_token_traced(...)`: refreshes inputs, executes trace, and reads back only what the caller truly needs.

For autoregressive decode, page table and position tensors are trace inputs. Bind persistent device tensors before capture and replay the trace over those same tensors. Token feedback stays in the traced device path: do not read a sampled token to host and reconstruct the next token input.

Build the hot loop so the steady-state step is trace replay plus the minimum caller-visible readback. Do not rebuild tokens, current positions, RoPE indices, masks, or page tables on the host every token. Advance device-owned state inside the captured graph where possible, such as `ttnn.plus_one` for current position/RoPE index and `tt_out_tok` feedback for the next token. Host refresh belongs at request boundaries, explicit reset, or actual scheduler-owned input changes; repeated per-token host refresh is an incomplete tracing implementation.

## Canonical Split Sampling

Implement token-out traced decode with two cooperating traces:

1. Capture the model decode trace up to sampler-ready logits.
2. Capture the chosen common sampling implementation, or a correct generator-owned trace wrapper around it, for the active sampling mode. Before choosing, compare `models/common/sampling/` and `models/common/modules/sampling/sampling_1d.py` against the model's state, seed, topology, trace, and logprob requirements.
3. Pass `tt_out_tok=<persistent decode token input tensor>` when calling the sampler, so the sampled token is written directly into the tensor consumed by the next decode replay.
4. Keep current-position/RoPE position state coherent with that token feedback by advancing it on device inside the trace when the model has a fixed-step decode loop. A completed trace does not use host-originated position refresh in the per-token loop.
5. Refresh page-table trace inputs only when the page table changes, and test both unchanged and changed page-table cases. The unchanged-page-table case should perform no per-token page-table copies after setup.
6. For greedy decode, keep the sampled token on device and benchmark the available on-device greedy strategies on the target mesh. Force-argmax is only a candidate. Do not select it by default. On a sharded LM head it can also be trace-unsafe (an output-tensor argmax that races replay and returns wrong tokens); validate any greedy choice with a clean same-logits decision probe.

The canonical pattern is in `models/tt_transformers/tt/generator.py`: capture decode once, bind the model to the same persistent trace inputs that replay refreshes, trace the chosen sampler path, and call sampling with `tt_out_tok` pointing at the decode token input. Untraced sampling hidden inside the model trace is not the canonical token-feedback path.

The normal split-sampling path for greedy decode must be a semantically greedy path. Do not treat a slower generic sampled `top_k=32` or top-p-capable path as proof that split greedy is slow. If `top_k=1` split sampling fails because a gathered top-k tensor has a non-tile inner dimension, fix the sampler-ready shape by preserving or padding the top-k representation, or keep a minimal repro and leave the stage incomplete.

For vocab-sharded logits, split greedy usually should not mean a physical `top_k=1` tensor per shard. Keep candidate tensors tile-shaped: run local top-k with `max_top_k` candidates per shard, usually 32, all-gather those candidates, then use semantically greedy sampling params (`k=1`, `p=0`, `temp=1`). This avoids full-vocab all-gather plus global argmax while still returning the greedy token.

If `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, or another sampling op dominates greedy token-out decode, the LM-head/sampling boundary is still wrong. Fix that before chasing lower-level decoder optimizations or marking tracing complete.

For vLLM decode serving, mirror the production split: bind persistent token/current-position/RoPE/page-table/KV-cache tensors before capture, warm the same mode, capture a device-only decode, and replay with `ttnn.execute_trace(..., blocking=False)` only when the caller implements the async read/host-processing split. If the sampler consumes transformed logits, capture the model trace output in that sampler-ready form so replay returns the same device tensor identity. Reuse the chosen common sampling path from the full-model stage for on-device sampling, and do not use host argmax or full-logits readback in a `sample_on_device_mode=all` path.

Do not collect Tracy, `tt-perf-report`, or `TT_METAL_DEVICE_PROFILER` metrics from a live vLLM server or serving adapter to prove this tracing work. vLLM-stage tracing evidence is functional and serving-level: trace capture/replay succeeds, stale-input tests pass, on-device sampling is wired, async split behavior is correct, qualitative/sampling checks pass, and `run_vllm_server` benchmark JSON records TTFT/ITL/throughput. Use non-serving full-model or reduced profiles from earlier stages for low-level device context if needed.

Keep async readback separate from scheduler overlap. A traced decode path can be safe to submit/read asynchronously while still unsafe for vLLM to build the next step before the previous sampled token has updated scheduler state. If the caller builds token IDs, current positions, or request lengths from host scheduler tables, the trace input refresh for step N+1 must wait for sampled token N to be applied, unless there is a separate test proving the next token/position path is entirely device-owned and cannot be overwritten by stale host state.

## What To Keep Outside Capture

Move these out of the captured region:

- `ttnn.from_torch(..., device=...)`, `ttnn.as_tensor(..., device=...)`, `ttnn.to_device`, and `ttnn.copy_host_to_device_tensor`.
- `ttnn.to_torch`, `.cpu()`, `ttnn.get_device_tensors` followed by host conversion, and full-logits host composition.
- `ttnn.synchronize_device`, event waits/synchronization, and explicit reads.
- Lazy weight loading, model-cache loads, and first-use module initialization.
- Resetting KV cache, page tables, semaphores, or sampling state.
- Python decisions that change the operation sequence, shape, memory config, or selected code path.

If a path needs a host decision, decide before capture and bind that mode at construction time or before the trace is created.

## Multi-Chip And CCL

Mesh traces can include collective operations, but collective resources need care:

- Set fabric config before opening the mesh.
- Create CCL semaphore managers before capture.
- Warm compile with the same mesh shape, page-table shape, and CCL topology intended for capture.
- If an op supports persistent output buffers and trace replay needs stable addresses, make sure the first allocation happens before capture. A warm compile pass often handles this if it takes the same code path.
- Reuse the same replicated or sharded input tensor allocations for capture and replay.

If a single decoder layer traces but the full model does not, inspect terminal work separately: final distributed norm, hidden gathers, LM head, logits gather, sampling, argmax, and token readback.

## Debugging Trace Failures

Common fatal signatures:

- `Writes are not supported during trace capture`
- `Reads are not supported during trace capture`
- `Event Synchronization is not supported during trace capture`
- Trace replay returns stale outputs or ignores updated inputs
- Trace replay passes once and fails on repeated execution

Debug by adding flushed markers around coarse blocks, then bisect:

```python
print("BEGIN_TRACE", flush=True)
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
print("TRACE_MODEL_START", flush=True)
out = model.decode_forward_from_ttnn_inputs(...)
print("TRACE_MODEL_DONE", flush=True)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
```

If the fatal is a write, first check for host input creation, lazy weights, cache loads, page table refresh, sampling-state updates, semaphore resets, first-use persistent CCL buffers, or `copy_host_to_device_tensor` hidden in helper APIs. Also suspect a program-cache miss: an op compiling a new variant mid-capture issues that write. Confirm by setting `device.set_program_cache_misses_allowed(False)` around capture (a miss then fails as `"<Op>: program cache miss occurred, but cache misses are forbidden"`, naming the offending op), then fix it via warm-up (see Program-Cache Warmup).

If replay uses stale inputs, compare the tensors captured by the model to the tensors refreshed before `execute_trace`. `tt_transformers` solves this by binding model-side trace inputs before capture and only refreshing those exact buffers before replay.

Before accepting any reduced input-refresh scheme, add a focused replay test that runs two decode steps with different token and current-position values, inspects the exact persistent trace input tensors, and asserts the output/logits changed. If page-table refresh is skipped, cover both unchanged and changed page tables.

Before accepting token-out decode, add a focused feedback test that proves the sampled token produced by replay N is the token input consumed by replay N+1. This is separate from teacher forcing; teacher forcing can pass while feedback is stale or host-reconstructed.

When benchmarking trace replay, record whether `ttnn.execute_trace` is blocking. Blocking replay may be a valid correctness probe, but a production generator or vLLM async path should use nonblocking replay plus a clear read/output-processing split when the caller can consume it.

When a traced loop is slower than the decoder-stack lower bound, instrument and fix the loop before retuning kernels. Eliminate host token refreshes, current-position/RoPE refreshes, page-table copies, mask rebuilds, cache resets, synchronizations, blocking trace replays, and feedback readbacks from the steady-state path. A line such as `position_refreshes = gen_len - 1` is evidence that the loop is still host-stepped, even if every decoder op inside the step is traced.

If you are still stuck after isolating the failing block, use `$autofix`. It should run diagnosis, then verify or refute each proposed root cause with focused experiments before keeping a fix.

## Symptom Table

These are mechanism signatures, not model properties. When generated or served output matches one, run the focused experiment before forming any precision, accuracy, or model-quality theory.

| Symptom | Likely mechanism | Focused experiment |
|---|---|---|
| Every output token emitted twice (or k times) while the text still advances | Decode loop consumes a stale token/position input - feedback lags replay by one step, often because async scheduler overlap built step N+1 before sampled token N updated host request state | Two-step replay with different tokens/positions; assert the exact tensors the trace reads were refreshed and the outputs differ. In vLLM, repeat with async overlap enabled and disabled |
| Greedy output nondeterministic across runs, or wrong after a sampled request | Trace cache keyed too coarsely - sampling mode/params are not part of the trace key, so replay reuses another mode's captured graph | Alternate greedy and sampled requests back-to-back; log which trace id each replay uses |
| Wrong output at exactly the capture position, correct afterwards | Capture recorded the cache update but never executed it | Execute the trace once immediately after capture, then validate the capture-position cache entry |
| One device/replica diverges after layer N while single-chip is clean | Collective-variant divergence on that axis (numerics or ordering of the reduce path) | Compare per-device outputs at layer boundaries; swap the collective variant for the failing axis |
| Serving output repetitive or garbled while the standalone generator is clean | Stale async-decode state across requests: outputs read before refresh, or per-request state not reset | Serve the same prompt twice with a different prompt between; diff against the single-request generator output, then run a multi-request smoke test |

## Evidence To Leave

Leave compact evidence that the traced path is real:

- Correctness before and after tracing against the same reference.
- Repeated replay determinism across several executions.
- Updated-input replay test proving outputs change when trace inputs are refreshed.
- For vLLM decode: stale-input validation for token/current-position/page-table refresh, explicit async-overlap setting and proof if enabled, on-device sampling trace evidence, and a passing server smoke run with decode trace enabled.
- Split-sampling evidence for token-out decode: internal sampling trace enabled, `tt_out_tok` wired to the persistent decode token input, and greedy benchmarks using the fastest correct on-device sampling strategy measured for this mesh.
- No host fallback in the captured path.
- Warmed trace replay timing, with prefill and decode measured separately where applicable.
- Host-work counters for the replay loop: trace replay count, token refresh count, current-position/RoPE refresh count, page-table refresh count, synchronizations/readbacks, and whether positions/tokens are advanced on device.
- `tt-perf-report` or Tracy evidence for the traced region, except vLLM serving stages where profiler collection is intentionally skipped and replaced by serving benchmark plus trace-contract evidence.
- Clear note of any remaining untraced boundary outside token-out decode. Token-out decode has no host sampling or full-logits readback.
