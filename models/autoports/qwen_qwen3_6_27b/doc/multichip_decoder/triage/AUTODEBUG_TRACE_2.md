# AutoDebug: TP4 full-attention trace capture write, second investigation

## Headline finding

The synchronized cache restore refutes the earlier race hypothesis. The remaining failure is a host-to-device **mesh-buffer write issued synchronously by a full-attention-only operation after trace capture begins**. The best current boundary is the first `ttnn.experimental.paged_update_cache` call at `tt/multichip_decoder.py:463`, but source inspection alone cannot distinguish that call from an earlier full-only composite without a capture-prefix experiment.

The four identical fatals are not evidence of four independent cache or fabric faults. They match one distributed mesh write fanning out over the four local chips. The exact message with `trace id` is guarded in `FDMeshCommandQueue::write_shard_to_device` at `tt_metal/distributed/fd_mesh_command_queue.cpp:665`. Device kernels updating KV cache do not use this host API.

## Direct observations

- `tests/multichip_traced_decode.py` now synchronizes after restoring caches, executes a second eager full decode, synchronizes again, restores caches again, and synchronizes immediately before `begin_trace_capture`. The original four write fatals remain. Outstanding cache-restore work is therefore not a viable primary explanation.
- Linear attention uses the same harness, mesh, residual/MLP row reductions, and trace API and captures successfully. The failure is in the full-attention-only portion.
- Full decode's unique sequence starts at `tt/multichip_decoder.py:454`: `uint32 -> int32` typecast, QKV/gate projection and split, QKV head creation, head norms, RoPE embedding/manipulation, two paged cache updates, paged SDPA, head concat/gate, and O projection/all-reduce.
- The fatal is a mesh-buffer host-write guard, not the program-cache-miss guard. A forbidden miss throws `Device operation "...": program cache miss occurred, but cache misses are forbidden` from `ttnn/api/ttnn/device_operation.hpp:390`.
- Single-device trace coverage proves only that `paged_update_cache` can be traced on a single device. It does not prove that this TP4 combination of sharded update tensor, sharded cache, and replicated page/position tensors avoids mesh host movement.
- The previous trace failure left all devices healthy in `full_trace_write.txt` / `full_trace_write_summary.txt`; the lingering process is exception cleanup after an un-ended capture, not a demonstrated device hang.

## Correct program-cache diagnostic

`mesh.set_program_cache_misses_allowed(False)` is the correct API and target. `MeshDevice` owns the program cache used by mesh device operations (`MeshDeviceImpl::get_program_cache`), and the nanobind method explicitly applies to the mesh. Do not iterate physical chips: that would test different per-device caches and can miss the mesh workload cache actually queried by `ttnn/api/ttnn/device_operation.hpp`.

Use it only after the final eager warmup and synchronization and immediately before `begin_trace_capture`, as the current `--forbid-program-cache-misses` path does. Restore `True` in `finally` before releasing/closing the mesh so a failed experiment cannot contaminate cleanup or later operations.

Interpretation is decisive:

1. If capture throws the named `program cache miss occurred` error, the message identifies the first cold device operation. Add an identical eager invocation with identical allocation/layout state, then retry.
2. If the same `Writes are not supported during trace capture` remains, ordinary TTNN program-cache coldness is refuted. A composite/helper is performing host tensor movement independently of program compilation.

## Ranked hypotheses and minimal experiments

### 1. TP4 paged cache update performs a mesh host write during capture (highest confidence)

Likely first failing line: `tt/multichip_decoder.py:463`; second independent occurrence is line 464. The operation uniquely combines a device-sharded one-KV-head-per-chip cache/update with replicated `page_table` and `cache_positions`. Existing trace evidence is single-device, while the observed four-way fanout is mesh-specific.

Minimal experiment: in a fresh process, construct only the already-warmed tensors immediately before line 463 and capture exactly one call:

```python
trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
ttnn.experimental.paged_update_cache(
    decoder.caches["key"], k,
    update_idxs_tensor=cache_positions,
    page_table=page_table,
)
ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
```

Warm this exact call once first, reset only the key cache, synchronize, forbid cache misses, then capture. Run the value-cache call separately in another fresh process. If key-only reproduces four write fatals, the boundary is proven. As a control, repeat with per-device-local page/position tensors matching the cache/update mesh mapping; success there identifies mesh adaptation of replicated runtime tensors as the write source.

### 2. An earlier full-only composite materializes or redistributes a mesh tensor on the host (medium confidence)

Candidate boundary: `nlp_create_qkv_heads_decode` at lines 457-459 or the inherited `_partial_rope_decode` calls at lines 461-462. These manipulate distributed tensors and have no counterpart in linear attention. A `to_memory_config` or composite adapter can hide host-side mesh movement even when every visible argument is device resident.

Minimal experiment: capture monotonically increasing prefixes, each in a fresh process after an identical eager warmup:

1. `typecast(current_positions)` only;
2. through QKV projection/split and `to_memory_config`;
3. through `nlp_create_qkv_heads_decode`;
4. through head norm;
5. through Q RoPE;
6. through K RoPE;
7. add the first paged update.

Set cache misses forbidden for every prefix. The first prefix that changes from clean capture to the host-write fatal identifies the operation; do not infer it from the last printed Python line because mesh worker exceptions can surface later.

### 3. A genuinely cold allocation-sensitive program variant remains (lower confidence, easy to falsify)

The second eager decode reduces this likelihood but allocation addresses can change program runtime arguments without changing the cache key. Ordinary address changes should be handled by cached runtime-argument override and are trace-supported. Run the exact command with `--forbid-program-cache-misses`. The named cache-miss error proves this hypothesis; the original host-write error refutes it.

### 4. Paged SDPA or the output collective triggers the write (low confidence)

These occur after the two cache updates, so they cannot be first if an isolated update reproduces. If both isolated updates capture, extend the prefix through `paged_scaled_dot_product_attention_decode`, then head concat/gate, then O projection, then `all_reduce`. Linear attention already exercises the same all-reduce helper, making the collective alone unlikely.

## Prefix-capture and cleanup requirements

- Run each boundary in a separate process. The public trace API exposes begin/end/release but no abort operation; after an exception inside capture, releasing or closing an active capture can linger and obscure the original fault.
- On successful capture: `end_trace_capture`, then `release_trace`.
- On failed capture: preserve the first exception/log, terminate that process by timeout, perform the `tt-device-usage` health/reset sequence, and start the next prefix in a new process.
- Cache reset copies must precede a synchronization before capture. Do not include host input/cache writes in the captured prefix.
- Record both `--forbid-program-cache-misses` output and the first failing prefix. Together these distinguish compilation from hidden host redistribution.

## Attractive but refuted or unsupported explanations

- **Asynchronous cache restoration:** refuted by explicit mesh synchronization and exact reproduction.
- **`paged_update_cache` is inherently untraceable:** false for single-device coverage; only its TP4 distributed form remains suspect.
- **Fabric/NoC/device hang:** not supported by triage; ARC heartbeat and device health were clean.
- **Four separate kernel failures:** inconsistent with the host mesh-write guard and one-message-per-local-shard fanout.

## Investigation limitation

The required fresh-context `.agents/scripts/autodebug.sh` run was launched, but its Codex sandbox could not start shell commands because the launcher could not initialize `bubblewrap`; its delegated readers hit the same failure and produced no report. This report was therefore completed by direct inspection of the cited checkout sources. No implementation or test file was edited during this investigation.
