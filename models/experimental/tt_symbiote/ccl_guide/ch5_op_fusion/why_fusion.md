# 5.1 Why Fusion Matters

## The Unfused Memory Round-Trip

In the Chapter 4 async model, the execution sequence for an AllGather → matmul layer is:

```
1. Dispatch all_gather_async to ERISC cores
2. ERISC transfers N-1 shards over Ethernet
3. Writer kernel places received shards in the persistent output buffer (DRAM or L1)
4. all_gather_async returns
5. (optionally: dispatch compute to Tensix while CCL is in flight)
6. synchronize_devices — wait for ERISC to complete
7. Dispatch matmul — reads full gathered tensor from the output buffer
8. matmul writes result
```

Even when step 5 exploits overlap, step 3 still writes a full gathered tensor to a buffer, and step 7 reads it all back. This is a mandatory DRAM round-trip if the gathered tensor does not fit in L1.

### Why DRAM is the bottleneck

For a 4-device ring AllGather of a bfloat16 `[1, 1, 4096, 4096]` tensor:

- Each device holds a shard of shape `[1, 1, 4096, 1024]` = 8 MB
- After AllGather the full tensor is `[1, 1, 4096, 4096]` = 32 MB
- Writing 32 MB to DRAM then reading 32 MB back costs ~64 MB of DRAM bandwidth
- On Wormhole, DRAM bandwidth is approximately 200–400 GB/s (chip dependent)
- Round-trip cost: ~0.32 ms at best (32 MB / 100 GB/s effective write + read)

For larger tensors used in models like Llama-70B, the hidden dimension is 8192 and the layer has multiple projections. The round-trip cost compounds across layers.

### Bandwidth formula

For N devices, each holding a shard of byte size S, the unfused DRAM round-trip cost is:

```
round_trip_bytes = 2 × N × S    (write gathered + read gathered)
round_trip_time  = round_trip_bytes / effective_dram_bandwidth
```

The fused op eliminates this entirely: the gathered tiles flow directly into the matmul's L1 CB.

---

## The Fused Pipeline

A fused AllGather+Matmul operation runs inside a single Metal program with the following structure:

```
Tile-level pipeline (single device, conceptual):

   ERISC:   [recv chunk 0] [recv chunk 1] [recv chunk 2] ... [recv chunk N-1]
             ↓ write to CB    ↓              ↓                  ↓
   CB:      [tile tiles of gathered shard in L1 CB — one chunk at a time]
             ↓ signal          ↓              ↓
   Tensix:  [matmul(chunk0)]  [matmul(chunk1)] [matmul(chunk2)] ... [accumulate]
```

The key difference:
- **Unfused:** ERISC fills a full output buffer; matmul starts only after all chunks arrive
- **Fused:** Matmul starts processing chunk 0 while ERISC is still receiving chunk 1

This is sometimes called *wave pipelining* — each wave of tiles enters the matmul CB as soon as it lands on-chip.

### Memory layout comparison

```
Unfused AllGather + Matmul:
  L1/DRAM  ┌──────────────────────────────────────┐
  output   │  gathered tensor (N × shard_size)    │  ← ERISC writes here
  buffer   └──────────────────────────────────────┘
                    ↓ full tensor read by matmul
  DRAM     ┌──────────────────────────────────────┐
  mm_out   │  matmul result                       │

Fused AllGather + Matmul:
  L1 CB    ┌───────────┐
  (small)  │  chunk N  │  ← ERISC writes one chunk at a time; matmul consumes before next chunk arrives
           └───────────┘   (no full gathered tensor ever exists in DRAM)
  L1/DRAM  ┌──────────────────────────────────────┐
  mm_out   │  matmul result                       │
```

---

## The FusedOpSignaler Mechanism

The CCL and matmul kernels are separate Metal programs dispatched to separate Tensix cores (and ERISC). They communicate through a `FusedOpSignaler` structure defined in `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.hpp`.

### Signaler variants

There are five signaler types, selected by `MatmulFusedOpSignalerType`:

| Signaler | Used for | Mode |
|----------|----------|------|
| `AllGatherFusedOpSignaler` | CCL side of AllGather+Matmul | `MULTI` — all matmul cores signaled when slice ready |
| `ReduceScatterFusedOpSignaler` | CCL side of Matmul+ReduceScatter | `SINGLE` — one privileged core signaled |
| `MatmulFusedOpSignaler` | Matmul side (both directions) | Holds ring geometry and worker coordinates |
| `MinimalMatmulFusedOpSignaler` | Matmul side for strided/minimal path | Lighter-weight variant with `input_tensor_Wt`, topology |
| `StridedAllGatherFusedOpSignaler` | Strided AllGather variant | Same structure as `AllGatherFusedOpSignaler` |

### Signal flow for AllGather → Matmul

The CCL (AllGather) side holds an `AllGatherFusedOpSignaler` that is initialized with:
- `fused_op_receiver_cores_noc`: NOC coordinates of the matmul worker cores
- `fused_op_receiver_signal_semaphores`: semaphore IDs on those cores
- `fused_op_signaler_mode`: `MULTI` (signal all matmul cores when a tensor slice is ready)

The matmul side holds a `MatmulFusedOpSignaler` initialized with:
- `ring_size`, `start_ring_index`: ring topology parameters for ordering the input slices
- `tensor_slice_shape_width`: width of each gathered slice in tiles
- `output_page_offset`, `weight_output_page_offset`: where in the output CB to write each slice's result
- `fused_op_receiver_cores_noc`: coordinates of the AllGather workers (for backward signaling if needed)

At runtime:
1. AllGather writer kernel writes a tile chunk to the shared CB
2. AllGather kernel sends a NOC semaphore increment to all `fused_op_receiver_cores_noc`
3. Matmul reader kernel is polling those semaphores — wakes up when all cores see the increment
4. Matmul processes the chunk
5. Steps 1–4 repeat for each chunk in round-robin order around the ring

### Signal flow for Matmul → ReduceScatter

The direction reverses: the matmul output is what triggers the ReduceScatter reader.

The matmul holds a `ReduceScatterFusedOpSignaler` (mode `SINGLE` — only one RS core is woken per chunk). That privileged RS core then fans out to the rest of the ReduceScatter worker grid:
- `ReduceScatterFusedOpSignaler.num_fused_op_cores_to_signal = 1` (default)
- `push_reduce_scatter_fused_op_rt_args` bakes the semaphore address into the matmul kernel's runtime args

For Llama, the matmul uses `MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER`:
- A privileged matmul core atomically decrements `matmul_semaphore_target`
- When the target reaches zero, it signals `rs_cores` via `rs_semaphore`
- The ReduceScatter side waits on `rs_semaphore` before reading from the matmul output CB

### FusedOpSignalerMode

```cpp
enum class FusedOpSignalerMode {
    SINGLE,  // signal one receiver core per chunk
    MULTI    // signal all receiver cores per chunk
};
```

`MULTI` is used when every matmul core needs to start on the same incoming slice simultaneously (e.g., a column-parallel matmul where all rows of the weight matrix are needed at once). `SINGLE` is used when one coordinator core can dispatch to the rest (e.g., ReduceScatter where a single privileged core handles fan-out).

---

## GlobalCircularBuffer (Llama ops)

Llama fused ops use `tt::tt_metal::experimental::GlobalCircularBuffer` (`global_cb` parameter) instead of the standard CB allocation. A `GlobalCircularBuffer` is a pre-allocated circular buffer whose physical backing is shared across multiple devices via the fabric layer. This enables the AllGather to write directly into the remote device's L1 CB without a DRAM intermediate — true zero-copy device-to-device tile delivery.

When `global_cb` is `None`, the Llama ops fall back to DRAM-backed intermediate storage. The `global_cb` path is the high-performance path and requires the CB to be allocated before the fused op is dispatched.

---

## When Fusion Helps vs. When It Doesn't

### Fusion helps when

- The gathered tensor is large enough that the DRAM round-trip is measurable (>8 MB typically)
- The matmul's K dimension is large relative to its M dimension (high arithmetic intensity per tile)
- The model runs at batch=1 or small batch (latency-critical, no amortization from large batches)
- The CCL and matmul naturally follow each other in the compute graph (no independent work between them that would benefit from the async gap)

### Fusion does not help (or hurts) when

- The gathered tensor fits in L1 — the async model already avoids DRAM if the persistent buffer is L1-resident
- Independent compute exists between the AllGather and matmul — the async (unfused) model extracts that overlap; fusion serializes it
- The matmul is very small (low arithmetic intensity) — the signaler overhead is non-trivial relative to compute time
- Debugging — fused ops are harder to isolate and profile than their two-op counterparts

> **Gotcha:** Fusion and async overlap are mutually exclusive for the same CCL–matmul pair. A fused op is a single blocking program: the matmul cannot start until the fused op's AllGather has received at least one chunk, but you cannot dispatch other work to the same Tensix cores while it runs. Use the unfused async model when there is independent compute to overlap; use fusion when there is none.

---

## Summary

| Property | Unfused async | Fused |
|----------|--------------|-------|
| DRAM round-trip | Yes (unless L1-persistent) | Eliminated |
| Overlap with independent compute | Yes | No |
| Dispatch complexity | Two ops + semaphore management | One op |
| Matmul starts when | All chunks received | First chunk received |
| Memory for intermediate tensor | `persistent_output_buffer` (explicit) | L1 CB (internal) or `intermediate_tensor` |
| Debugging ease | High | Lower |

---

*Back to [Chapter 5 Index](index.md)*

*Next: [5.2 Fused Ops](fused_ops.md)*
