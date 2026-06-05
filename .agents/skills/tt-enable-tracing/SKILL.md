---
name: tt-enable-tracing
description: "Enable or debug TTNN trace capture and replay for models, decoders, generators, and multi-chip TTNN code. Use when eliminating TTNN dispatch overhead, implementing traced prefill/decode, making full-model generation trace-safe, or investigating trace failures such as unsupported writes, reads, event synchronization, stale inputs, or bad replay correctness."
---

# TT Enable Tracing

Use this skill when adding TTNN trace capture/replay or when an existing traced path fails. Trace replay is usually the difference between a good device implementation and a usable inference path: the device kernels may be fast, but eager Python/TTNN dispatch between many ops can dominate decode.

## Useful References

Read only what helps the current task:

- `tests/ttnn/tracy/test_trace_runs.py`: minimal trace capture/replay examples.
- `models/tt_transformers/tt/generator.py`: production decode trace patterns, including host input preparation, persistent device inputs, replay refresh, and split sampling.
- `models/common/sampling/generator.py`: standalone traced sampling patterns.
- `models/tt_transformers/tt/model.py`: model-side `prepare_decode_inputs_host` and device-only `ttnn_decode_forward` split.
- `advanced_perf_optimizations.md`: deeper examples for TTNN trace capture/replay, multiple command queues, trace plus multi-CQ, and production benchmarking patterns. Search this file for the API or failure mode you are working on before loading it wholesale.

## Mental Model

Trace capture records a static command sequence on a device or mesh device. Replay reuses that exact sequence without paying per-op host dispatch.

The traced region should be device work over stable device tensors. Host-originated input changes happen before replay by copying into those stable tensors. Outputs that are produced by TTNN ops inside capture are fine; host writes, host reads, and host synchronization while the trace is open are not.

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
4. Run one warm compile call with the same shapes and mode.
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

## Generator Pattern

Do not trace the high-level generator method unless it is already proven trace-safe. Split the generator into:

- `prepare_decode_inputs_host(...)`: returns host TTNN tensors or torch values for token ids, positions, RoPE indices/tables, page tables, masks, and other per-token inputs.
- `copy_host_to_device(...)` or explicit `ttnn.copy_host_to_device_tensor(...)`: refreshes stable trace input tensors before replay.
- `decode_forward_from_ttnn_inputs(...)` or `ttnn_decode_forward(...)`: device-only model call used for warm compile and capture.
- `decode_next_token_traced(...)`: refreshes inputs, executes trace, and reads back only what the caller truly needs.

For autoregressive decode, page table and position tensors are trace inputs. If they can change between requests or steps, refresh their stable device buffers before replay. If sampling writes the next token back into the input token buffer, design that as a traced sampling path or a second trace rather than doing host readback inside the model trace.

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

If the fatal is a write, first check for host input creation, lazy weights, cache loads, page table refresh, sampling-state updates, semaphore resets, first-use persistent CCL buffers, or `copy_host_to_device_tensor` hidden in helper APIs.

If replay uses stale inputs, compare the tensors captured by the model to the tensors refreshed before `execute_trace`. `tt_transformers` solves this by binding model-side trace inputs before capture and only refreshing those exact buffers before replay.

If you are still stuck after isolating the failing block, use `$autofix`. It should run diagnosis, then verify or refute each proposed root cause with focused experiments before keeping a fix.

## Evidence To Leave

Leave compact evidence that the traced path is real:

- Correctness before and after tracing against the same reference.
- Repeated replay determinism across several executions.
- Updated-input replay test proving outputs change when trace inputs are refreshed.
- No host fallback in the captured path.
- Warmed trace replay timing, with prefill and decode measured separately where applicable.
- `tt-perf-report` or Tracy evidence for the traced region.
- Clear note of any remaining untraced boundary, such as host sampling or full-logits readback.
