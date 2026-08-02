# 13 · Model-Level Levers — Weight Prefetch and the Host Decode Loop

Every lever in 02–12 is scoped to one op: a program config, a dtype, a kernel. The two levers here
are not. They change how the **whole model streams**, so they are recorded once per model with
`op_signature="model:<lever>"` and appear in the report's `Model-level levers` block rather than as a
row in the per-op matrix. Filing one against a single op — as a `shard` attempt on whichever matmul
happened to be the target — makes it invisible to the next run.

Reach for these when the per-op ladder is spent and the decode step is still far off its bandwidth
ceiling. They are the only levers that change *when* work happens rather than how fast one op is.

---

## 1. Read the wait before reaching for either {#model-read-the-wait}
<!-- route
op_class: matmul,eltwise,other
bound: dram,slow
rank: time
regime: decode
lever_type: single-shot
-->

Both levers below attack **waiting**, so measure the waiting first. The per-stage table separates
`DEVICE FW DURATION` (how long the op held the device) from `DEVICE KERNEL DURATION` (how long it
computed):

```
GeGLU mul 128x15360 bf4_b:   40.557ms FW vs  6.151ms kernel    (34.4ms producer wait)
GeGLU mul 32x15360 bf8_b:    26.814ms FW vs  1.619ms kernel
residual add 128x3840 bf16:  11.074ms FW vs  6.417ms kernel
```

An op computing for 6 ms and occupying 40 is not slow — it is idle, spinning on its producer's
semaphore. **No configuration of that op can help**, which is why grid/fidelity/shard/tt-lang/cpp all
measure flat on it. The profiler charges the wait to the consumer, so these ops look expensive and
are not.

Rule: if `kernel` ≪ `FW` across a bucket, stop tuning that bucket. The cost is upstream.

---

## 2. DRAM weight prefetch — overlap layer N+1's fetch with layer N's compute {#model-prefetch}
<!-- route
op_class: matmul
bound: dram
rank: time
memory: dram_interleaved,sharded
regime: decode
lever_type: single-shot
-->

**When it applies.** Decode re-reads every weight from DRAM once per token, so the step is bounded by
`bytes / bandwidth`. If the profile shows a memory-bound decode below ~80% of peak DRAM bandwidth,
and the weights are DRAM-resident rather than already in L1, the gap is usually not layout — it is
that the stream idles while compute runs and compute idles while the stream runs:

```
today:      [read N] [compute N] [read N+1] [compute N+1] ...
overlapped: [read N] [read N+1     ] [read N+2 ]
                     [compute N     ] [compute N+1]
```

**The mechanism.** `models/tt_transformers/tt/prefetcher.py` builds a ring of dedicated *sender*
cores that stream each layer's weights DRAM → L1 through a global circular buffer while the previous
layer computes. `mlp.py` and `attention.py` already accept `global_cb=` and `sub_device_id=` and
already register their weights — the usual missing piece is that the demo's `create_tt_model` never
constructs a `Prefetcher` and passes it to `ModelArgs`.

**Order matters.** The prefetcher must exist *before* `ModelArgs`: the base class derives the
norm/residual/MLP shard configs and the receiver-core grids from it. A `Transformer` handed a
prefetcher its `ModelArgs` never saw will build tensors on cores the global CB does not own.

**Two gates reject models wrongly** (both observed on gemma-3-12b-it):

1. `is_prefetcher_supported` matches `HF_MODEL` against a hard-coded table of verified configs. A
   model whose geometry is absent — or whose entry lists a different `dim`/`hidden_dim` than the
   checkpoint — fails the lookup, and the constructor then dies on an unset `num_receiver_cores`.
2. Its L1 budget is computed at `BYTES_PER_TILE_BFP8 = 1088`. A model whose weights are `bfloat4_b`
   (576 B/tile) is refused for a budget it does not use — at ring size 80 the real block is
   ~553 KB/core against an 850 KB budget, i.e. it fits.

Both are stale checks, not physics. Bypass them deliberately and say so in the note; do not silently
edit the table.

**Cost.** Sender cores are dedicated, receivers are not. A 10×8 ring costs 8 of ~110 worker cores,
not 88.

**What "working" looks like.** This is the part that decides whether a flat measurement is a verdict
or a mis-wire, so check it before recording:

- the sender cores appear in the profile and the weight reads leave the matmul's critical path —
  `FW − kernel` on the decode matmuls shrinks
- `device_ms` may barely move. That is expected: prefetch does not remove work, it moves it off the
  critical path. **Judge it on trace+1cq per-token, not on device_ms.**
- if nothing changes at all, the likely causes in order: the global CB was never built (check the
  constructor ran), the matmuls were not passed `global_cb`, or the weights are already L1-resident
  and there was nothing to overlap.

Record as `model:prefetch`. A flat result is worth recording *with the evidence above*, so the next
run knows which of the three cases it was.

---

## 3. The host decode loop — keep the step on device {#model-host-loop}
<!-- route
op_class: host_fallback,other
bound: host
rank: time
regime: decode
lever_type: single-shot
-->

**The symptom.** The scorecard prints `fully on device: NO` and names the calls:

```
host round-trips: ttnn.as_tensor, ttnn.copy_host_to_device_tensor, ttnn.from_torch, ttnn.to_device
```

Each is a CPU→device transfer *inside* the decode step. They also break the trace region: the capture
has to stop at each host call and resume after, so the gaps belong to no op and appear in the profile
only as `host_overhead`.

**The fix.** Make the step's inputs device-resident once, then write into them per token instead of
rebuilding them:

- allocate the input tensor on device at setup with `ttnn.allocate_tensor_on_device`
- per step, update it in place (`ttnn.copy_host_to_device_tensor` into the *same* buffer, or better,
  keep the sampled token on device and skip the host entirely)
- no `from_torch` / `to_device` inside the loop

Then the whole step traces as one region and the host does nothing but launch it.

**Distinct from prefetch.** Prefetch hides DRAM latency behind compute; this removes CPU stalls
between steps. A model can need both, and neither substitutes for the other.

**What "working" looks like.** `fully on device` flips to YES, the trace covers the whole step, and
`host_overhead` collapses. If `host_overhead` stays large after the round-trips are gone, it is
dispatch gaps rather than host stalls — a different problem, addressed by fusing adjacent ops (06).

Record as `model:host-loop`.

---

## 4. Do not confuse either with batching {#model-not-batching}
<!-- route
op_class: matmul
bound: dram
rank: time
regime: decode
lever_type: single-shot
-->

Batching amortises the weight read across more users: read once, produce B tokens. It raises
aggregate tok/s and leaves **tok/s/u** — the per-user rate the decode ceiling bounds — essentially
unchanged. Both levers above change the per-user rate at fixed batch. If the target is tok/s/u,
batching is not the answer.
