# AUTOTRIAGE

## Diagnosis

- The most likely root cause is a host-write/trace-start race in `tests/multichip_traced_decode.py`, not a device-side attention, cache, CCL, or fabric hang. The test restores the two full-attention cache tensors with asynchronous `ttnn.copy_host_to_device_tensor(..., cq_id=0)` calls and immediately calls `ttnn.begin_trace_capture(mesh, cq_id=0)` without synchronizing the mesh. If either restore is still being fanned out, its per-device mesh-buffer writes encounter the newly active trace and `FDMeshCommandQueue::write_shard_to_device` intentionally raises `Writes are not supported during trace capture`. The four identical failures match one replicated mesh write reaching the guard once on each of the four local devices.

## Triage Evidence

- `full_trace_write_summary.txt` reports every collected check as `pass`, including ARC, Ethernet, NoC, running operations, call stacks, fast dispatch, lightweight asserts, watcher ringbuffer, binary integrity, and broken components. This directly argues against a wedged worker kernel, CCL credit wait, bad fabric route, LLK assert, ARC failure, or unhealthy chip.
- The detailed capture contains no watcher-ring, lightweight-assert, or device-firmware failure that precedes the host exception. The diagnostic reader also encountered unsupported UMD `noc_read` forms on some probes; those skips do not establish a hardware failure, and the independent health checks passed.
- The reported exception text exactly matches the host guard at `tt_metal/distributed/fd_mesh_command_queue.cpp:665`, in `FDMeshCommandQueue::write_shard_to_device`. That path is a host-to-device mesh-buffer write, not a kernel cache write performed by `paged_update_cache` and not a NoC/CB wait inside a device program.
- The process remaining alive after the exception is plausibly teardown fallout: `trace_id` has been assigned, capture did not reach `end_trace_capture`, and the `finally` block calls `release_trace` on a trace that is still in capture state before closing the mesh. It is downstream of the initial prohibited write.

## Source Evidence

- In `tests/multichip_traced_decode.py`, the sequence immediately before capture is:
  1. eager `decode()` and `ttnn.synchronize_device(mesh)`;
  2. two cache restores through `copy_host(...)`;
  3. `ttnn.begin_trace_capture(mesh, cq_id=0)`;
  4. captured `decode()`.
  There is no synchronization between steps 2 and 3.
- `copy_host` invokes `ttnn.copy_host_to_device_tensor(..., cq_id=0)` and does not request or provide blocking completion. Both cache tensors are replicated mesh tensors, so each logical copy fans out to four physical shards.
- `FDMeshCommandQueue::write_shard_to_device` tests `trace_id_` before dispatching shard data and raises the exact observed message, including the trace id. Therefore an outstanding host copy that reaches this method after capture starts explains both the exception class and its four-device fanout without requiring any defect in the cache-update kernel.
- The full-attention path is the only path here that calls `paged_update_cache`, but that operation is trace-capable in repository coverage: `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py` explicitly compiles it and captures it in warmup and main traces. The Qwen single-chip `full_attention_decode_smoke.py` also captures its full decode after an eager warmup. This demotes “paged update is inherently untraceable.”
- The passing linear trace does not refute the race. The linear and full paths restore different tensors with different layouts/dtypes/sizes, so asynchronous copies can complete at different points relative to `begin_trace_capture`. A scheduling race can be path-specific even though the harness ordering is identical.

### Ranked hypotheses and verify/refute experiments

1. **Asynchronous cache restore overlaps trace start (high confidence).** Add exactly one `ttnn.synchronize_device(mesh)` after the cache-restore loop and before `begin_trace_capture`, then rerun the original command. Elimination of all four write fatals verifies it. As a second control, omit cache restoration for capture; successful capture also verifies that the failure lies outside captured decode.
2. **A cold program/cache miss inside full decode performs an unexpected host upload (medium-low confidence).** With the synchronization above retained, run two eager full decodes before capture and compare against one warmup. If two works and one fails, query the program cache before/after each op or bisect captured prefixes to identify the cold op. This is less likely because the full eager decode already warms every operation, ordinary program enqueue is trace-supported, and the guard identifies tensor data movement.
3. **One full-only operation triggers host data movement during capture (low confidence): `uint32 -> int32` typecast, paged cache update, or paged SDPA.** Capture minimal prefixes in order: typecast only; QKV/RoPE through typecast; first `paged_update_cache`; second update; paged SDPA; output projection/all-reduce. Ensure all inputs and cache restoration are synchronized before every capture. The first prefix that reproduces the guard identifies the operation. Also run a standalone TP4 replicated `paged_update_cache` trace modeled on the DeepSeek test. This hypothesis is refuted if every prefix captures after synchronized setup.
4. **Trace-region exhaustion or CCL incompatibility (very low confidence).** These normally produce allocation/trace-size or operation-specific failures rather than `write_shard_to_device`. Increase `trace_region_size` only if synchronized capture progresses to such an error. The passing linear trace already demonstrates TP4 all-reduce capture on the same topology.

## Downstream Effects

- Because the exception occurs after `begin_trace_capture` and before `end_trace_capture`, the command can linger during trace release or mesh close. That liveness symptom is teardown of an incompletely captured trace, not evidence that attention kernels or fabric are stuck.
- Cache correctness, paged addressing, and attention math are not implicated by this evidence. The eager full-attention run passed before capture, and the prohibited action is a host transfer issued around the capture boundary.
- Reset/recovery after the failed process remains appropriate operational hygiene, but no triage evidence indicates a persistent hardware defect.

## Proposed Fix

- Synchronize the mesh after restoring all caches and before starting capture. Keep mutable token/position writes outside capture and synchronize them whenever capture-start ordering could overlap them.
- Make capture teardown state-aware: only release a trace after `end_trace_capture` succeeds, or explicitly end/abort an active capture on an exception before releasing and closing the mesh. This prevents the secondary teardown hang and preserves the original exception.
- If the synchronized original command still fails, use the ranked prefix bisection above before changing decoder implementation. Do not replace or disable paged KV-cache updates based on the current evidence.

## Uncertainty

- The captured triage artifact is a device-health snapshot and does not contain the Python/C++ host stack for each fatal, so the exact pending source copy (key versus value cache) is inferred from test ordering and the matching mesh-queue guard.
- The four messages are consistent with replicated fanout over four devices, but a host stack or `TT_METAL_LOGGER_LEVEL=Debug` run would make that attribution conclusive.
- If a post-restore synchronization does not eliminate the failure, hypothesis 3 becomes the next actionable path; the minimal prefix experiment is required before proposing an implementation change.
