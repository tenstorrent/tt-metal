# Packed multi-request prefill

`Gemma4PrefillRuntime.prefill_batch` packs independent requests for embedding,
Q/K/V projection, normalization, RoPE, output projection, residuals and MLP.
Each attention layer restores individual requests to the original CP cache
layout, writes that request's cache, runs batch-one ring attention, and returns
the useful output rows to the packed layout. Both sliding K/V and global
`[Krot128 | Vordered512]` caches use the existing kernels and migration buffers.

```python
from models.demos.gemma4_d_p.tt.ragged_prefill import PrefillRequest

# runtime.compile(kv_cache) has loaded the model into the engine-owned caches.
# Only valid token IDs are supplied. Starts are absolute positions.
batch = (
    PrefillRequest(request_id=100, slot_id=0, actual_start=8192, token_ids=tuple(a_tokens)),
    PrefillRequest(request_id=101, slot_id=1, actual_start=0, token_ids=tuple(b_tokens)),
)
result = runtime.prefill_batch(batch, kv_cache)
outputs = result.to_torch()  # Owned host tensors keyed by stable request ID.
result.deallocate()
```

The prefix for request 100 must already occupy slot 0. One boundary contains at
most one chunk per slot and one chunk per request. Validation occurs before
staging any request. A start of zero refills a slot; a continuation must match
both its request identity and the preceding full chunk's end. A partial chunk
is final. The caller must finish any migration that reads a slot before refilling
it, as on the single-request path.

For host queues, `runtime.prefill_requests(prompts, kv_cache)` accepts an iterable
of `(request_id, token_ids)` prompts and yields one result per completed boundary.
It fills vacant slots, advances each live request by one chunk, and refills
finished slots when the consumer asks for the next boundary. Copy any outputs
and complete dependent migration before advancing that iterator. `prefill_batch`
lets an external scheduler choose its own boundaries and slot lifecycle.

The existing blocking H2D socket runner remains batch one. Its protocol has no
boundary marker or nonblocking dequeue, so accumulating an arbitrary number of
socket records could deadlock a producer waiting for completion. Queue-aware
serving integrations call the batch API explicitly. No wire protocol changes
are required for the existing single-request comparison path.

## Layout and padding

For CP8/TP4, each request occupies `ceil(valid_length / 32) * 32` packed rows.
The concatenation is padded once to a multiple of 1024, ensuring each CP/TP
partition contains whole tiles. Token IDs and absolute RoPE positions have
identical ordering, including zero positions on padding rows. Sliding and global
RoPE keep their different adjacent/packed transforms.

Packed projections and MLP matmuls use slabs of at most C/8 local rows, across
request boundaries. This avoids the large-M fallback that accumulates at lower
precision. Attention projections retain the original chunk's math fidelity even
for short slabs. Norms retain the batch-one width reduction geometry, splitting
large slabs and padding their final local slab to whole normalization blocks
(64 rows at C=8192). This small norm-only padding is separate from the packed
token rows below.

Packed TP reductions gather partials in rank order, add them locally in FP32,
round the completed sum once, and then partition rows.
Elementwise FP32 additions avoid the NC reduction kernel's intermediate truncation.
Ring reduction order depends on destinations and communication worker ranges; moving packed
tokens otherwise causes numerical drift that accumulates through all 60 layers.
The local sum keeps the addition order independent of token placement, at the
cost of gathering all four TP partials.

The default single-request path retains its original ring reduction. For numerical
comparison with packing, enable `GEMMA4_PREFILL_STABLE_REDUCTIONS=1` before model
construction, or set `runtime.model.stable_prefill_reductions = True` before
warmup/capture. This selects the same FP32 order for independent calls. Treat it
as a compile-time setting: release and warm/capture again after changing it.

CP all-gather collects projected Q/K/V (or Q/packed KV) without gathering TP
heads. Each segment is sliced, padded to the **configured original chunk size**,
and partitioned across CP. With chunk size C and L=C/8, a token at absolute
position `chunk*C + rank*L + row` still occupies local cache row `chunk*L + row`.
This is also the mapping in `kv_chunk_table.py`. Partial requests may have zero
useful rows on some cache ranks, even though tokenwise work is balanced across
all CP ranks. Attention outputs undergo the inverse gather/slice/concat/partition
before TP reduction, row partitioning and shared residual/MLP work.

Cache writes use a separate stable `valid_global` tensor per active request.
The existing BFP8 writer clamps to the last **32-token page** containing valid
tokens; rows within that final page may be padding. Ring attention retains its
fixed padded extent, since its metadata path derives geometry from Q's shape and
does not accept a separate logical-length tensor. Causality prevents a valid
query from reading the padded suffix; padded query outputs are discarded.

There is still up to 31 tokenwise padding rows per request, plus up to 1023 at
the end of the batch. Attention retains a full chunk per request. Redistribution
adds CP collectives at every layer and can outweigh padding savings, especially
for long full chunks. There is no claim of a speedup for every workload.

| Valid lengths | Independent tokenwise rows | Packed tokenwise rows | Rows removed |
| --- | ---: | ---: | ---: |
| 33, 1025 | 16384 | 2048 | 87.5% |
| 8192, 8192 | 16384 | 16384 | 0% |

These are row counts, not throughput measurements. Both workloads still execute
two 8192-token attention calls at each layer.

## Trace and output ownership

The trace key is the ordered tuple of tile-rounded segment lengths, cache chunk
size, CP and TP. Slots, prefixes and exact lengths within a tile can change on
replay. Occupancy changes use another trace; inactive lanes never enter a graph.
Only one graph is resident. Changing shape, selecting eager execution with
`use_trace=False`, or switching to batch one synchronizes and releases the old
graph before allocating buffers. The first call for a new shape warms and
captures it; this cost is separate from steady-state replay throughput.

This conservative policy matters: TTNN traces reuse transient allocation
addresses, so keeping a new variant's metadata alive while replaying an older
graph can corrupt it. All metadata and position buffers are allocated before
capture and updated in place. CCLManager owns ring receive buffers and semaphores;
the separate attention calls reuse them in command-queue order.

Results lease device outputs until the next runtime call or `release_trace`.
`result.to_torch()` makes independently owned copies of valid rows. An expired
lease raises an error. No retained device clone is allocated after capture.

Batch completion emits exactly one acknowledgement per chunk per layer, after
the batch has synchronized. Compile/capture does not emit acknowledgements.
Each chunk's layer records stay contiguous for the existing migration consumer.
`request_id` owns the serving request; optional `completion_id` supplies the
migration chunk sequence number. If omitted, the runtime assigns a monotonically
increasing completion ID, returned in `result.requests`. Socket records retain
`[slot, actual_start, actual_end]`. Migration can start later than on the
single-request path because batch acknowledgements follow the complete replay.
Switching back to a socket trace requires its original warmup/capture handshake.
Using socket acknowledgements for batches also requires that initial service
handshake, so its program-cache buffers exist before the packed trace is captured.

Supported geometry is CP8/TP4 with whole CP/TP-local tiles per cache chunk and
chunk-aligned absolute starts; the serving default is C=8192. Arbitrary unaligned
prefix insertion and continuation after a partial final chunk are rejected.
There is no automatic retile or cache reinterpretation fallback.

## Validation and measurements

For an editable perf-only workload, with the model/cache environment from
`PREFILL_SERVICE.md`:

```bash
OMP_NUM_THREADS=16 pytest models/demos/gemma4_d_p/tests/test_ragged_prefill_perf.py -sv --timeout=3600
```

This uses all 60 layers by default; set `GEMMA4_RAGGED_TEST_LAYERS=6` for a shorter
run. Edit the consecutive `run_batch(lengths, starts=[...])` calls in the test.
Each list index selects its KV slot, and the printed token range is
`[start, start + length)`. Start zero replaces a slot's request; a nonzero start
continues its preceding full 8192-token chunk. The examples include unequal
lengths, three/four-request batches, full chunks and populated-prefix continuations.

Each call invokes `prefill_batch` once. Output shows useful tokens/s for the batch
and completion latency for each request. All requests share the batch's completion
time. New shapes are labeled `capture+replay` (including warmup/capture); matching
consecutive shapes are labeled `replay`. Timings include packing, staging, device
execution and synchronization, excluding model loading, random token generation,
output downloads and external KV transfer.

Run host checks:

```bash
OMP_NUM_THREADS=8 pytest models/demos/gemma4_d_p/tests/unit -q \
  -k 'not test_device_packed_global_transforms_match_reference'
```

With the model/cache environment from `PREFILL_SERVICE.md`, run the hardware
comparison with allocation checks enabled:

```bash
OMP_NUM_THREADS=16 TT_METAL_TRACE_ALLOC_TRACKING=1 \
pytest models/demos/gemma4_d_p/tests/test_ragged_prefill.py -sv --timeout=3600
```

The default uses six real decoder layers, covering five sliding layers and a
global layer, and four allocated slots. Independent batch-one requests occupy
two slots; packed requests occupy the other two. It compares valid hidden states
and all valid written K/V, including global packed KV, across unequal lengths,
different prefixes, alignment boundaries, slot reuse, occupancy changes and
consecutive trace replays. It also checks bitwise isolation when only one request
changes, preservation beyond the cache writer's final page, and raw DRAM reads
through every migration head/config on the first sliding/global cycle.
Set `GEMMA4_RAGGED_TEST_LAYERS=60` for all layers.

The test prints `RAGGED_MEASUREMENTS` and writes `ragged_measurements.json` in its
pytest temporary directory. It measures mixed `(33, 1025)` and full
`(8192, 8192)` workloads. Wall times include host packing, token/metadata/position
staging, redistribution, slicing/padding/concatenation, device execution and
synchronization. Per-request completion latency is measured from the start of
the boundary. Host output downloads and an external migration endpoint are
outside these timings. Cold shape capture is reported separately. End-to-end
migration latency requires the loopback endpoint described in
`PREFILL_MIGRATION.md`; host address-walk tests and device raw reads validate the
unchanged source address mapping without an external endpoint.

### Results on Blackhole 8×4, 2026-10-08

The 60-layer hardware test passed with allocation tracking enabled. Across seven
boundaries and twelve request chunks, all **1,332 output/KV comparisons were
bitwise equal** to independent batch-one execution using the same fixed-order
arithmetic. Isolation, untouched cache
tails, migration-address byte checks and acknowledgement ordering also passed.
The host suite passed 150 tests (two hardware-only transform cases excluded).

Validated shapes use C=8192, one or two active requests, four allocated slots,
32K cache capacity, starts of 0 and 8192, and valid lengths 31, 32, 33, 1023,
1024, 1025, 1055 and 8192. Numerical comparisons have not covered larger packs
or other chunk sizes; the perf-only test also exercises three/four-request packs
without output comparisons. The existing `GEMMA4_ACTIVATIONS_DRAM_ONLY=1` option is
available when activation storage exceeds L1 capacity.

Medians of three steady-state replays, with both requests starting at zero.
The independent column enables fixed-order FP32 reductions to isolate packing's
cost from a change in arithmetic; the default ring-reduction baseline is measured
separately below.

| Valid lengths | Independent FP32 batch time | Packed batch time | Independent FP32 useful tokens/s | Packed useful tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 33, 1025 | 614.7 ms | 269.9 ms | 1,721 | 3,920 |
| 8192, 8192 | 619.3 ms | 922.3 ms | 26,456 | 17,763 |

The mixed workload improves throughput by 2.28× against independent calls using
the same FP32 arithmetic. Two full chunks achieve only
0.67× independent throughput: they save no token rows and pay redistribution,
temporary storage and split/concat costs at every layer.

Median per-request completion latencies from the batch boundary are 307.4/614.7 ms
independently versus 269.9/269.9 ms packed for the mixed case; the full-chunk case
is 311.5/619.3 ms versus 922.3/922.3 ms. These are prefill completion times, excluding
host hidden-state downloads and external KV transfer.

The first call after switching shapes took 9.98 s for the mixed shape and
7.21 s for the full shape, including warmup/capture/replay. Weights and the JIT
cache were warm, and the full shape had already appeared in the correctness
phase; these numbers are not clean-process compilation times. Frequent shape
changes with the one-resident-trace policy can dominate serving latency.

The canonical 60-layer, 256K single-request baseline also passed, using the
requested command with `--timeout=3600` to accommodate cold setup. It processed
262,144 tokens in 6.938 s wall time (6.847 s on device), or **37,782 useful
tokens/s**, using the preserved default ring reduction. The packed comparison
above uses fixed-order FP32 arithmetic in both columns; it does not establish a
speedup against the default ring path. Its 32K cache and fresh prefixes also
differ from the canonical baseline's growing 256K context.

Raw samples and measurement scope are in
[`ragged_prefill_measurements.json`](ragged_prefill_measurements.json). No external
migration endpoint/client was installed at the documented location, so loopback
transfer latency and destination-byte verification were not run. Source-address
correctness was checked directly against device memory.

### GPU accuracy qualification remains open

The additional 8K GPU-reference test did **not** pass its existing accuracy gate.
A control using the original ring reduction also failed its worst-head threshold.
The fixed-order FP32 mode has an additional accuracy gap; bitwise agreement with
independent TT calls does not establish agreement with the GPU model.

| 8K GPU metric | Required | Original ring | Fixed-order FP32 |
| --- | ---: | ---: | ---: |
| Overall PCC | ≥ 0.978 | 0.981975 | 0.977559 |
| Worst head PCC | ≥ 0.928 | 0.918803 | 0.897869 |
| Relative RMSE | < 0.208 | 0.190157 | 0.212173 |

The worst head in both runs was sliding V, layer 39, head 11. Service handshakes,
request processing, acknowledgements and source-address checks completed; the
test failed its numerical thresholds. Packed execution remains experimental
pending that accuracy qualification. The default single-request arithmetic is
preserved. Reproduce the opt-in comparison with:

```bash
GEMMA4_PREFILL_STABLE_REDUCTIONS=1 \
pytest 'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-8k]' -sv --timeout=1800
```
