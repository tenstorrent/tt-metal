# Critical finding: the program cache silently zeroes L1 measurements

Applies to any attempt to measure an op's L1 pressure by wrapping its **real** call in a
`RunMode.NORMAL` graph capture — the obvious design, and the one that quietly produces wrong
numbers. Context: `README.md` (how traces work), `L1_PRESSURE_PROFILING.md` (the metric and the
protocol).

## The symptom

Same op, same shape, four consecutive captures on one device:

```
NORMAL      (1st ever = cold)      peak_total= 190464   peak_cb= 190464   n_cbs=13
NORMAL      (2nd = cache hit)      peak_total=      0   peak_cb=      0   n_cbs= 0
NO_DISPATCH (after cache hit)      peak_total= 190464   peak_cb= 190464   n_cbs=13
NORMAL      (3rd = cache hit)      peak_total=      0   peak_cb=      0   n_cbs= 0
```

`NORMAL` reports the op's 186 KB/core only on the **first** dispatch of a given program. Every
subsequent dispatch reports zero. `NO_DISPATCH` reports the correct figure every time, including
after a cache hit, and agrees exactly with the cold `NORMAL` number.

This is not a `peak_cb`-only artifact. `peak_total` is the high-water of a single `current_total`
accumulator, incremented in the *same branch* that increments `current_cb`
(`ttnn/core/graph/graph_trace_utils.cpp:325-333`). Missing CB nodes are missing input data, so
every field that includes them is wrong.

## Why it happens

`circular_buffer_allocate` nodes are emitted from **two different places**, one per run mode:

| mode | emitted from | fires when |
|---|---|---|
| `NO_DISPATCH` | `GraphProcessor::track_program`, iterating `program->circular_buffers()` (`ttnn/core/graph/graph_processor.cpp:426-428`) | every dispatch, unconditionally |
| `NORMAL` | `tt_metal/impl/program/program.cpp:1693,1720`, inside real CB allocation | only when CBs are actually allocated |

`track_program` deliberately hands the `NORMAL` case over to the allocation path:

```cpp
void GraphProcessor::track_program(Program* program, const IDevice* device) {
    track_deallocate_cb(device);              // <-- unconditional
    if (run_mode == RunMode::NORMAL) {
        // we will track real buffer allocations during program run
        return;                               // <-- no CB nodes from here
    }
    for (const auto& cb : program->circular_buffers())
        track_allocate_cb(cb->core_ranges(), 0, cb->size(), cb->globally_allocated(), device);
}
```

On a program-cache hit the cached `Program` is reused as-is: it is not rebuilt and its circular
buffers are not re-allocated, so `program.cpp` never fires. The trace therefore contains no
`circular_buffer_allocate` for that dispatch.

The design is reasonable in itself — `NORMAL` wants the *real* addresses, which only exist at
allocation time — but it makes CB visibility a function of cache state rather than of the program.

**It is worse than omission.** `track_deallocate_cb` is called *before* the early return and
pushes its node unconditionally (`graph_processor.cpp:395-413`), so a cache hit still emits
`circular_buffer_deallocate_all`, which **zeroes** the CB / dataflow-buffer / scratchpad running
totals. A cached dispatch does not merely fail to add its bytes; it clears whatever was standing.

Measured node census of a cache-hit `NORMAL` capture of the same op — note the deallocate is
present, no `circular_buffer_allocate` key exists at all, and the output `buffer_allocate` is
still recorded:

```python
{'capture_start': 1, 'function_start': 14, 'function_end': 11,
 'tensor': 21, 'buffer': 3,
 'buffer_allocate': 1, 'buffer_deallocate': 1,
 'circular_buffer_deallocate_all': 1,          # present -> zeroes the totals
 'capture_end': 1}                             # 'circular_buffer_allocate' absent
```

## Why it is dangerous

1. **Silent.** No warning, no exception. A zero looks like an op with no circular buffers, which
   is a legitimate state (a fully tensor-backed sharded op, or a Metal 2.0 port).
2. **Plausible when partial.** With an L1-resident output, `peak_l1` is still counted, so
   `peak_total` comes back non-zero and merely too small — no obvious tell.
3. **Worst for composite ops.** In a multi-program op where some children are cold and others
   cached, the cold children register in the running max before the next child's
   `deallocate_all` clears it, while the cached children contribute nothing. The result is a
   partial figure that is neither the true peak nor zero.
4. **Order-dependent.** The same case measures differently depending on what ran before it, so a
   suite is not reproducible and two implementations are not comparable.

## Solutions

### A — passive: disable the program cache, capture `NORMAL`

With the cache off, every dispatch rebuilds the program and re-allocates its CBs, so the nodes
reappear on every call. Measured, same op and shape:

```
cache ON,  NORMAL capture              peak_total=      0    wall= 5.13 ms   <- broken
cache OFF, NORMAL capture (1st)        peak_total= 190464    wall= 5.15 ms
cache OFF, NORMAL capture (2nd)        peak_total= 190464    wall= 5.34 ms   <- stable
cache OFF, plain call (no capture)                           wall= 2.06 ms
cache ON,  plain call (no capture)                           wall= 1.10 ms
```

`device.disable_and_clear_program_cache()` exists and this is the supported pairing — the
mock-allocator query API documents the same requirement, since a cached workload outlives the
sub-devices it references and crashes at teardown (`graph_query_op_constraints.hpp`, upstream
issue 45646).

- **For:** no per-call-site wiring; measures the real call with real addresses; a passive read
  from wherever the test harness already sits.
- **Against:** the cache-off penalty applies to *every* op call in the run (1.10 → 2.06 ms here,
  roughly 2x host-side); the capture window then spans input preparation and output readback as
  well as the op, so the peak must be attributed to the op **by frame name** — a heuristic that
  breaks for an implementation whose internals differ from the expected shape.

### B — active: keep the cache, re-invoke under `NO_DISPATCH`

Let the real call proceed untouched, then call the op a second time inside a `NO_DISPATCH`
capture. Nothing dispatches, nothing is allocated, and the trace is exactly the op.

```python
result = op(*args, **kwargs)                 # real call: correctness, device time, cache warm
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
try:
    op(*args, **kwargs)                      # measurement only; output discarded
finally:
    trace = ttnn.graph.end_graph_capture()
peaks = ttnn.graph.extract_resource_usage_per_core(trace)
```

Measured cost of the extra capture: **~3.9 ms** per call.

- **For:** immune to cache state by construction; the window contains the op and nothing else, so
  no frame identification is needed at all; the real run's cache behaviour and any device-time
  measurement are left untouched. `NO_DISPATCH` programs are deliberately not cached
  (`graph_nanobind.cpp` docstring), so the measurement cannot poison the real cache.
- **Against:** the call site must hold `(fn, args, kwargs)`, so *something* has to hand the
  callable over — a passive test-level hook cannot re-invoke on its own; and input tensors built
  before the capture are outside the window, so `peak_l1` excludes the contract unless they are
  rebuilt inside it (see `L1_PRESSURE_PROFILING.md` §2.1 for why that also matters for the
  `size_t` underflow).

### Recommendation

**B.** The exact-by-construction window is the deciding factor: no frame-name heuristic, and
therefore no failure mode that grows worse the more composite the implementation is — which is
precisely the case `peak_l1` exists to measure. The cost is one hook at a site that already has
the call in hand, against A's run-wide behavioural change plus an approximation.

Use A only where no such site exists and the op's internal frame shape is known and stable.

## Tripwire

Whichever design is used, assert on the two failure signatures rather than trusting the number:

- `peak_total == 0` **and** the trace contains a `circular_buffer_deallocate_all` but no
  `circular_buffer_allocate` → the cache-hit signature. Fail the measurement; do not record a zero.
- `peak_total >= 2**32` → the `size_t` underflow from a pre-window deallocation
  (`L1_PRESSURE_PROFILING.md` §1.5).

A cross-check worth running once per op: capture the same case both cold-`NORMAL` and
`NO_DISPATCH` and assert the two agree. They did here, exactly (190464 both ways), which is what
licenses using `NO_DISPATCH` as the measurement mode.
