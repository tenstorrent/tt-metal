# Measuring L1 pressure of a TTNN op across a case matrix

Design notes and findings for a per-case L1 profiler: run one implementation of an op over a
matrix of test cases, emit one self-describing record per case, and join records from different
implementations offline.

Companion documents:
- `README.md` — how graph capture works and how to read a trace, with five captured traces.
- `L1_profiling_critical_findings.md` — the program-cache pitfall: why a `RunMode.NORMAL` capture
  of the real call reports zero on a cache hit, and the two ways around it.
- `../test_graph_l1_breakdown.py` — a Python re-implementation of the peak accounting, asserted
  field-for-field against the C++ extractor, plus tests for each trap below.

Everything marked **[measured]** was run on device in this repo (Blackhole p150b). Everything
marked **[source]** was read out of the tree at the cited line.

---

## Part 1 — Findings

### 1.1 The metric is `peak_total`, and only `peak_total`

`ttnn.graph.extract_resource_usage_per_core(trace)` returns five numbers. All five are L1 bytes
per core. **[source]** `ttnn/api/ttnn/graph/graph_trace_utils.hpp:19-33`

| field | counts | allocated by | lifetime | JSON node |
|---|---|---|---|---|
| `peak_cb` | circular buffers | the program | the program | `circular_buffer_allocate` |
| `peak_dataflow_buffer` | Metal 2.0 dataflow buffers | the program | the program | `dataflow_buffer_allocate` |
| `peak_scratchpad` | Metal 2.0 kernel scratchpads | the program | the program | `scratchpad_allocate` |
| `peak_l1` | device **tensors** in L1 (incl. L1_SMALL) | the allocator | as long as the tensor | `buffer_allocate` / `buffer_deallocate` |
| `peak_total` | high-water of the **sum** of the four | — | — | — |

`peak_l1` is a misnomer: it does not include CBs. The name means "peak L1 *buffer*", where
"buffer" is the `Buffer` class, i.e. a tensor.

Use `peak_total`. Three reasons:

1. **The first four are independent high-water marks that can peak at different instants.**
   Separate accumulators, each `max`'d on its own increments. **[source]**
   `ttnn/core/graph/graph_trace_utils.cpp:309-313, 325-370`. Therefore
   `peak_cb + peak_l1 + peak_dfb + peak_scratchpad >= peak_total`, with equality only when the
   maxima coincide. They did coincide in every trace captured here (conv2d: 76.06 + 46.53 =
   122.59 = `peak_total`), but that is a property of those ops, not a guarantee.
2. **`peak_cb` is zero for a Metal 2.0 program.** Program-scope L1 moves to
   `peak_dataflow_buffer` / `peak_scratchpad`. **[measured]** `ShardedToInterleavedDeviceOperation`
   inside `conv2d.json` reports zero CBs and one `dataflow_buffer_allocate`. Any metric built on
   `peak_cb` alone silently scores a ported implementation as free.
3. **`peak_cb` is zero for a fully tensor-backed sharded op.** **[measured]** `sharded_add.json`:
   `peak_cb = 0`, `peak_l1 = 32768`.

Do keep the four fields — as **diagnostics** that explain why a number is what it is. Never as the
metric.

### 1.2 They are physically different halves of L1

Program-scope L1 grows **up** from the unreserved base; tensor buffers grow **down** from
`L1_END`. They collide in the middle, which is why the validation exists. **[source]**
`tt_metal/impl/program/program.cpp:1868-1935`:

```
"Statically allocated circular buffers in program {} clash with L1 buffers on core range {}.
 L1 buffer allocated at {} and static circular buffer region ends at {}"
```

**[measured]** `matmul_normal_mode.json`, real addresses on a 1.5 MB Blackhole core: the L1 tensor
sits at `1552384` with size `20480`, and `1552384 + 20480 = 1572864` = exactly the top of L1. The
three CBs sit at `111616`, `242688`, `373760` — the bottom.

This is also why a tensor-backed CB is excluded from `peak_cb`: its address *is* the tensor's
address, so those bytes live in the top half and are already counted by `peak_l1`. The exclusion
is the region layout, not a heuristic.

### 1.3 Where the numbers come from

`peak_cb` is the op author's block arithmetic, verbatim — no allocator, no measurement:

```
factory:  in0_CB_size = in0_CB_tiles * in0_aligned_tile_size   (tiles -> bytes, x buffering depth)
          CircularBufferConfig(in0_CB_size, ...) -> total_size_
impl:     CircularBufferImpl::size() { return config_.total_size(); }
tracker:  track_program -> track_allocate_cb(..., cb->size(), ...)
extract:  if (!globally_allocated) current_cb += size
```
**[source]** `matmul_multicore_reuse_mcast_2d_program_factory.cpp:152-171`,
`tt_metal/impl/buffers/circular_buffer.hpp:39`, `ttnn/core/graph/graph_processor.cpp:415`,
`graph_trace_utils.cpp:325`.

It is per core by construction: a `CircularBufferConfig` total size is what *each* core in the
CB's `core_range_set` reserves. **[measured]** Reproducible by hand from `matmul_l1_out.json`:
1024x1024 bf16 is 32x32 tiles on a 2x2 grid, so the output block is 16x16 = 256 tiles, and
256 x 2048 B = 524288 — exactly the third CB node.

`peak_l1` is the allocator's per-bank arithmetic, `Buffer::aligned_size_per_bank()`:

```
aligned_size_per_bank() = calculate_bank_size_spread(aligned_size(), aligned_page_size(),
                                                     num_banks, alignment)
  num_banks = sharded ? num_cores() : allocator->get_num_banks(type)
  num_equally_distributed_pages = 1 + ((num_pages - 1) / num_banks)      // ceil-div
  return num_equally_distributed_pages * round_up(page_size, alignment)
```
**[source]** `tt_metal/impl/buffers/buffer.cpp:767`, `tt_metal/impl/allocator/allocator.cpp:609`.

Consequences: the ceil-div makes it a **worst case per bank**, not an average (100 pages over 64
banks charges 2 pages to every bank, including the 28 holding one); and for a sharded buffer
`num_banks` is `num_cores()`, so the figure is exactly one shard rounded to page alignment.

### 1.4 The capture is a window

A tensor allocated before `begin_graph_capture` emits no `buffer_allocate`, so it contributes
nothing to `peak_l1` even though it physically occupies L1 throughout. **[measured]**
`test_capture_window_bounds_peak_l1` measures the same sharded add at `1 x shard` with inputs
built outside the window and `3 x shard` with them built inside.

### 1.5 Freeing a pre-window buffer corrupts the peaks (bug)

The running totals are `size_t` with no floor on the decrement. **[source]**
`graph_trace_utils.cpp:366-368`. A `buffer_deallocate` whose matching `buffer_allocate` is outside
the window underflows the accumulator.

**[measured]**, both cases, one shard = 32768 B/core:

| scenario | true `peak_l1` | reported |
|---|---|---|
| free 32768/core (pre-window), then allocate 32768/core | 32768 | **0** (the wrap cancels exactly) |
| free 65536/core (pre-window), then allocate 32768/core | 32768 | **18446744073709518848** (2^64 - 32768) |

Metal notices and logs `Can't hook deallocation of a buffer which allocation wasn't hooked`
(**[source]** `tt_metal/impl/graph/graph_tracking.cpp:178`) but the node is still emitted and
still subtracted.

Every op that frees an input hits this: `deallocate_activation=True`, an explicit
`ttnn.deallocate`, or `reallocate`. **This is the single hard constraint on the protocol** — see
2.1. Worth filing upstream; a `std::min(current, size)` clamp plus a warning would fix it.

### 1.6 Exclusions are de-duplications, not losses

- A CB with `globally_allocated == "1"` is **tensor-backed**: the CB *is* a tensor's L1 shard
  (`CircularBufferConfig::set_globally_allocated_address`). Excluded from `peak_cb` because
  `peak_l1` already counts the tensor.
- A dataflow buffer with `borrows_memory == "1"` — same argument, Metal 2.0.

**[measured]** In `sharded_add.json` all three CBs are tensor-backed at `32768` each, exactly equal
to the shard's `max_size_per_bank`. Same bytes, counted once.

Both flags are JSON **strings** (`"0"` / `"1"`), not ints — they are not in the integer coercion
whitelist. **[source]** `ttnn/core/graph/graph_processor.cpp:50`.

### 1.7 In-place implementations allocate nothing

**[measured]** `sharded_add_in_place.json` (`ttnn.add(a, b, output_tensor=a)`) contains **no
`buffer_allocate` node at all**. `input_tensors` has three entries and one `buffer` node carries
three `tensor` children, all at `address: 1474560, tensor_id: 13` — input, preallocated-output
argument and return value are one buffer. The three CBs are still 32768 each, so the op moves
96 KB/core of live L1 while `peak_l1` reports 0.

The 0 is an artifact of the window (1.4), not of the accounting. With operands built inside the
window the figure is correct and an in-place implementation legitimately scores lower than an
out-of-place one.

### 1.8 Per-core peak ignores placement

`extract_resource_usage_per_core` sums CB sizes **ignoring `core_range_set`** — two CBs on disjoint
core sets are added as if they shared a core. Deliberate worst case. **[source]**
`graph_trace_utils.cpp:325-333`.

Two things follow:

- A core-aware aggregation is strictly tighter. `Breakdown.peak_per_core` in the companion test
  does it; `max(peak_per_core.values())` is the true worst core for program-scope L1.
  **[measured]** For matmul the two agree, because every CB in one program shares a core range.
- **The per-core figure says nothing about how many cores are used.** **[measured]** the
  1024x1024 matmul chose a 2x2 grid at 768 KB/core. An implementation spreading the same work over
  64 cores at 48 KB/core scores 16x better per core while occupying 16x more of the chip. Both
  facts are needed and they answer different questions:
  - `peak_total` — does it fit, and how much headroom does a fused neighbour have. Decides OOM.
  - `peak_total x cores_touched` — how much of the device it consumes. Decides concurrency, and
    whether a "win" was real or just spreading.

### 1.9 A CB's core range is not the tensor's shard grid

**[measured]** In `sharded_add.json` the tensor-backed CBs carry
`core_range_set: "{[0-0 - 10-9]}"` = 110 cores, while the shard's `num_cores` is 64. The op
declares the CB over the whole grid it may run on and repoints it at the shard. Do not infer the
shard grid from a CB node.

### 1.10 Composite ops are separable

**[measured]** `conv2d.json` contains four `circular_buffer_deallocate_all` nodes, one per child
program (`InterleavedToSharded`, `Halo`, `Conv2d`, `ShardedToInterleaved`). Each child's
program-scope L1 is therefore attributable rather than blended. Combined with `stacking_level` and
a per-frame running max, you get "peak while this op was live" for every frame in the tree.

**[measured]** conv2d 32->64 3x3 on 128x128: `peak_total` 122.59 KB, reached inside
`Conv2dDeviceOperation`; Halo's own frame never exceeds 37.12 KB. Conv2d's 10 CBs split 6 real
(77888 B = the whole `peak_cb`) and 4 tensor-backed.

Also visible: **`reallocate` appears as a deallocate and an allocate at the same size** (nodes 118
and 122, both 27136), net zero and invisible in the peak.

### 1.11 Smaller traps

- **L1_SMALL reports `type: "L1"`** and is folded into `peak_l1`, though it lives in a separate
  allocator region. **[measured]** Halo's config tensors are `buffer_type: 3` (L1_SMALL) at
  16–48 B/core.
- **`NORMAL` adds a DRAM buffer** for kernel binaries / dispatch data at enqueue.
  **[measured]** 22528 B at node 16 of `matmul_normal_mode.json`, absent from `NO_DISPATCH`. DRAM,
  so harmless to the L1 peaks, but it changes the node count.
- **`device_id` is inconsistent across node types in `NORMAL`**: CB nodes say `0`, buffer nodes
  say `1`, because CB tracking comes from `program.cpp` with the underlying `IDevice` while buffer
  tracking has the `MeshDevice`. Do not key on it. **[measured]**
- **`extract_peak_L1_memory_usage` is a different function** and not a component of `peak_total`.
  It adds buffers by their whole-buffer `size` (all banks) and sums that with per-core CB bytes —
  mixed units. **[source]** `graph_trace_utils.cpp:60-110`. Do not use it here.
- **Sizes are exact in both run modes.** They come from the `TensorSpec` and
  `CircularBufferConfig`; only addresses are faked, by `hook_allocate` setting `address_ = 0`.
  **[source]** `tt_metal/impl/buffers/buffer.cpp:582`, `graph_processor.cpp:956`.
- **Peaks are deterministic** for a given shape and config — pure arithmetic over the trace, no
  timing, no allocator state in `NO_DISPATCH`. One capture per case; no warmup, no repeats.
- **`enable_fast_runtime_mode` must be off** for the full op nesting. With it on you still get the
  device-op frames but not the composite ttnn frames.

### 1.12 What the metric cannot tell you

1. **Fragmentation.** `peak_total` is a high-water mark of a running sum and knows nothing about
   placement, so two implementations with equal `peak_total` can differ in whether they actually
   fit. Escalation: `query_op_constraints_with_initial_state`
   (`ttnn/api/ttnn/graph/graph_query_op_constraints.hpp`) runs the op against a `MockAllocator` on
   a mock device, reproducing real addresses and fragmentation, returns `Error` when it genuinely
   cannot fit, and threads allocator state across a sequence of ops.
2. **Time.** Peak is duration-blind. Holding 100 KB for the whole op and touching it for a hundred
   cycles score identically. `duration_ns` on `function_end` is the only temporal signal here, and
   in `NO_DISPATCH` it is host time, not device time.
3. **Correctness.** `NO_DISPATCH` never executes. A beautiful low peak can belong to a wrong
   kernel. The profiler must be paired with, not substituted for, a correctness run.
4. **Anything outside the window.** By construction.

---

## Part 2 — The measurement protocol

### 2.1 Build every contract tensor inside the capture window

Not stylistic. Forced by 1.5: if the op frees an input that was allocated before the window, the
accumulator underflows and the reported peak is either too low or astronomically large. Build the
inputs — and the preallocated output, if the contract has one — inside the window.

### 2.2 Accumulate globally, maximise locally

Building inputs inside the window means input construction is *in the trace*, with its own ops and
its own CBs (`from_torch` may tilize, `InterleavedToSharded` may run). If that construction ever
peaks higher than the op under test, a whole-trace `peak_total` measures the construction, not the
op.

The fix is not to move the construction out (2.1 forbids it). Run the accumulation over the
**whole** trace, so the baseline and every deallocation are accounted correctly, but take the
maximum only over the sub-range where the op under test is on the stack.

This is exactly `OpStat.peak_total` in the companion test — "peak while live":

```python
def move(kind, delta, cores=None):
    ...
    b.peak_total = max(b.peak_total, total)          # whole-trace
    for i in stack:
        b.ops[i].peak_total = max(b.ops[i].peak_total, total)   # per frame
```

**Primary metric = the op-under-test frame's `peak_total`.** It equals the whole-trace
`peak_total` whenever the op dominates — the common case, and true for every trace here — and is
correct when it does not. Record both and flag cases where they differ; a divergence means the
harness's own scaffolding was the peak, which is a bug in the case, not a property of the op.

Identify the frame as the **last top-level (`stacking_level == 1`) `function_start`**, since the
harness calls the op last. Accept an explicit op name as an override.

### 2.3 Gates

Every case must pass three checks, and a failure must be recorded as data, not raised:

1. **Output spec equality.** `out.tensor_spec()` must match what the contract declares. "Same
   contract" has to be enforced, or an implementation that quietly returns interleaved where
   another returns sharded looks cheaper for a reason that is not L1 discipline.
2. **Underflow tripwire.** `peak_total < 2**32`. Catches 1.5 in the unlikely event 2.1 is
   violated, plus any future accumulator bug.
3. **Frame agreement.** Whole-trace `peak_total` vs the op frame's, per 2.2.

### 2.4 Mode

`NO_DISPATCH`. Sizes are exact, nothing executes, and a case that would OOM still yields a number
— which is the point, since an implementation's whole value may be that it fits where another does
not. Pair with a separate `NORMAL` correctness run.

---

## Part 3 — Profiler design (one implementation, many cases)

The goal is a per-case record, not a comparison. Comparison happens later, offline, by joining
records from separate runs. That inverts one requirement over a direct A/B: **the case identity
must be derivable from the contract alone**, never from anything the implementation chooses.

### 3.1 Separation of concerns

```
case matrix (contract only)  ->  collect (needs device)  ->  records.jsonl + traces/
                                                                 |
                                    analyze / join (no device) <-+
```

Mirrors how `graph_report` splits capture from import: no analysis at capture time.

### 3.2 The case is data, the implementation is a callable

```python
case = {
    "op": "matmul",                    # logical name of the CONTRACT, not the impl
    "inputs": [tensor_spec, ...],      # shape, dtype, layout, memory_config
    "kwargs": {...},                   # everything else the contract fixes
    "output": {"preallocated": bool, "memory_config": ...},
}

impl = lambda device, tensors, **kwargs: ...   # the only thing that varies between runs
```

`case_key = sha1(canonical_json(case))`. Because the key is a pure function of the contract, impl
A's row and impl B's row for the same cell join on it without coordination. Nothing the
implementation picks — core grid, program config, decomposition — may enter the key.

If two implementations have slightly different signatures, absorb that in the `impl` callable, not
in the case. The case stays canonical.

### 3.3 Every case emits a row, including failures

A case that raises is a legitimate outcome and must appear in the output, or the join silently
drops cells and an implementation looks better by supporting less.

| `outcome` | meaning |
|---|---|
| `ok` | measured, all gates passed |
| `unsupported` | the op raised — record exception class and message |
| `gate_failed` | measured but untrustworthy — record which gate |
| `error` | harness fault (input construction failed, etc.) |

### 3.4 Record schema

```json
{
  "schema_version": 1,
  "case_key": "sha1:9f2c...",
  "case": { "op": "matmul", "inputs": [...], "kwargs": {...}, "output": {...} },
  "outcome": "ok",
  "metric": {
    "peak_total": 806912,
    "peak_cb": 786432,
    "peak_l1": 20480,
    "peak_dataflow_buffer": 0,
    "peak_scratchpad": 0,
    "peak_total_whole_trace": 806912,
    "worst_core_program_l1": 786432,
    "cores_touched": 4,
    "device_bytes": 3227648,
    "contract_baseline": 20480,
    "internal": 786432
  },
  "shape": {
    "child_ops": ["MatmulDeviceOperation"],
    "programs": 1,
    "node_count": 22,
    "tensor_backed_cbs": 0,
    "l1_allocs": 1,
    "l1_frees": 1
  },
  "trust": { "underflow": false, "output_spec_match": true, "frames_agree": true },
  "provenance": {
    "arch": "blackhole", "grid": [11, 10], "l1_per_core": 1572864,
    "commit": "edf6276af5a", "mode": "NO_DISPATCH", "ts": "2026-09-07T12:00:00Z",
    "trace": "traces/9f2c....json"
  }
}
```

Field notes:

- **`peak_total`** — the op frame's, per 2.2. The primary number.
- **`peak_total_whole_trace`** — for the `frames_agree` gate.
- **`device_bytes`** = `peak_total * cores_touched`, the occupancy figure from 1.8.
- **`cores_touched`** — the union of `core_range_set` over counted program-scope nodes in the
  frame, widened by `max(num_cores)` over its L1 buffer nodes. Approximate, because the trace
  carries a shard's *count* of cores but not their coordinates, and interleaved buffers report
  `num_cores: 0` (they span every bank — take the grid from the device). Document the definition
  next to the number; consistency across runs matters more than precision.
- **`contract_baseline`** — sum of `aligned_size_per_bank()` over the contract's tensors, via
  `extract_l1_output_buffer_allocation_size_per_core`. A worst-case-on-one-core figure, since the
  per-core accounting charges every buffer to a single core regardless of its real bank spread.
  Identically wrong for every implementation, so the derived `internal = peak_total - baseline`
  ("everything inside") is comparable; quoting `baseline` alone is misleading.
- **`shape.child_ops`** — the decomposition digest. Cheap, and it is how you notice that an
  implementation quietly stopped calling Halo, or started.
- **`provenance.commit`** and **`arch`** — a peak is only comparable within one arch, and a
  factory's block arithmetic changes between commits.

### 3.5 Keep the raw traces

Save the trace JSON per case, keyed by `case_key`. The metric definitions above will change as the
corpus teaches you things — `cores_touched` in particular is a first cut. With the traces kept,
every past run is re-analyzable without a device and without re-running; without them, a metric
change invalidates the corpus.

Budget: the traces here are 10–95 KB each (22–186 nodes). A 500-case matrix is a few tens of MB.
Gate it behind a flag if that is too much, but default to on.

### 3.6 Analysis

JSONL, one file per (implementation, run). Join on `case_key`:

- **Coverage first.** Compare `outcome` before comparing bytes. An implementation with a lower
  median `peak_total` over fewer `ok` cases has not won.
- **Both axes.** `peak_total` for fit, `device_bytes` for occupancy. A 16x per-core improvement
  that is pure spreading shows up as flat `device_bytes`.
- **Per-case, not aggregate-first.** The interesting output is the set of cases where the two
  differ by more than a threshold, with `shape.child_ops` and the four diagnostic fields to
  explain each one. A single mean over a case matrix hides exactly the regime changes worth
  looking at.
- **Headroom.** `peak_total / l1_per_core` is the fit fraction; anything near 1.0 is a case where
  the fragmentation blind spot (1.12) matters and the mock-allocator path is warranted.

### 3.7 Where the case matrix should come from

This repo already has a per-op registry model — `feature_spec.py` (`TARGET` / `INPUTS` /
`INVALID`), `axes.py`, and the `SUPPORTED` / `EXCLUSIONS` block in the op file. `INPUTS` is
already a curated contract matrix with the invalid cells marked. Generating the profiler's case
list from it, rather than a parallel hand-written matrix, means the L1 corpus and the correctness
corpus cover the same cells and the `unsupported` outcomes can be checked against the declared
`SUPPORTED` rectangle instead of being taken on trust.

---

## Part 4 — Open questions

1. **`cores_touched` for interleaved buffers.** They report `num_cores: 0` and span every bank.
   Counting the whole grid overstates occupancy for a small tensor; counting zero understates it.
   Current proposal takes the grid from the device and documents the choice.
2. **Should the input-construction scaffolding be excluded from `contract_baseline`?** Building a
   sharded input runs a real op with its own CBs. Those are released before the op under test
   starts, so they do not enter the frame's peak — but they do enter `peak_total_whole_trace`, so
   `frames_agree` will trip on cases with expensive input construction. Either widen the gate to a
   tolerance or treat the divergence as informational.
3. **Multi-device.** Everything here is single-device. `device_tensors` in the tensor nodes carries
   per-device addresses, and `track_device` records per-device info, but the peaks are not
   partitioned by device.
4. **Whether to also record the mock-allocator verdict** per case. It answers "does it actually
   fit" rather than "how much does it want", needs a mock device and the program cache disabled,
   and is C++-only today — so probably a second, narrower pass over the cases nearest the budget.
