# Using the streaming profiler with your own callback

The streaming profiler drains device zones to the host and hands decoded records to any
callback you register. (For the on-wire zone packet family and the measurements behind it, see
[STREAMING_PROFILER_ZONES.md](STREAMING_PROFILER_ZONES.md) — none of it changes the consumer
contract described here.)

## Enable (Blackhole)

```
export TT_METAL_STREAMING_PROFILER=1
```

One switch — it also arms the device-side markers, and it is the only thing that reserves any
device memory: without it a Tracy-enabled build costs nothing.

### The defaults, and when to change them

Out of the box you get the configuration every number in
[STREAMING_PROFILER_ZONES.md](STREAMING_PROFILER_ZONES.md) was measured with — no other variable
needs setting.

| knob | default | change it when |
|---|---|---|
| `TT_METAL_PERF_DEBUG_FILLERS` | `6` — 6 fillers + 1 mover, tuned to keep producers unstalled at high offered rates | `4` (4 fillers + 2 movers) for long max-rate captures: the second mover is worth ~2.3x on sustained evacuation |
| `TT_METAL_PERF_DEBUG_ROLE_RING_MB` | `448` per filler | runway scales with capture *length* (~19 MB per 1k iterations per filler), so lower it to hand DRAM back on short captures, raise it for very long ones |
| `TT_METAL_PERF_DEBUG_STAGE_MIN_FILL_PCT` | `0` — ship every core's words immediately | `50` when host cost dominates: frames ship fuller, ~2x fewer frames per zone, paid for out of producer headroom |

The rings live in the DRAM profiler region, reserved per bank while this profiler is enabled —
at the default, 448 MiB per bank.

## Register a callback

From `tools/profiler/perf_debug_consumer.hpp`:

```cpp
auto h = perf_debug::register_consumer("my-sink",
    [](const perf_debug::PerfDebugRecordBatch& b) { /* ... */ });
// later: perf_debug::unregister_consumer(h);
```

Register any time — before the device opens or mid-capture. Your callback runs on its own
thread; if you're slow you drop only your own records (`b.dropped_delta`), never anyone
else's. The batch span is only valid during the call, so copy what you keep.

## What you get

Zones arrive **whole**: a zone is one record with a start and a duration. On the wire the device
already ships most zones atomically (one 3-word packet at scope close, carrying end + duration);
the few remaining legacy start/end pairs (the producer-stall zone, the >3.2 s long-zone fallback,
DRISC drainer self-zones) are paired for you on the host. Either way you never see halves.

```cpp
enum class PerfDebugRecType : uint32_t {
    Zone = 1,   // a complete zone: data.zone = {start, duration}
    Data = 3,   // point marker with payload: data.ts; payload follows via Ext + Cont
    Event = 4,  // point marker, no payload: data.ts
    Ext = 5,    // Data/Event continuation header: data.ext = (id << 32) | payload word count
    Cont = 6,   // one uint64 of Data payload: data.payload
};

struct PerfDebugRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;  // which (core, RISC) stream: lane = core_index * 5 + risc
    uint32_t dev : 3;    // device index into the capture context
    PerfDebugRecType type : 3;
};

struct PerfDebugRec {
    // The active member is decided by meta.type.
    union {
        struct {
            uint64_t start;     // device timestamp of the zone open
            uint64_t duration;  // device cycles
        } zone;
        uint64_t ts;       // Data / Event
        uint64_t ext;      // Ext
        uint64_t payload;  // Cont
    } data;
    uint32_t id;            // structural zone id -> resolves to the zone's name
    PerfDebugRecMeta meta;
    uint32_t prog;          // runtime host-id of the op this lane is executing (0 = none yet)
};
```

Ordering: cross-lane interleaving is arbitrary — key any state you keep by
`(meta.dev, meta.lane)`. A zone is delivered when it **ends**, so under nesting
`data.zone.start` isn't monotonic within a lane; `start + duration` is complete information
either way.

## The call pattern

The ops-CSV consumer in `tools/profiler/perf_debug_ops_csv.{hpp,cpp}` is the full working
reference.

```cpp
void MyConsumer::operator()(const perf_debug::PerfDebugRecordBatch& batch) {
    names_.refresh();  // ZoneNameMirror member: names arrive as kernels JIT, refresh once per batch
    for (const auto& r : batch.records) {
        if (r.meta.type != PerfDebugRecType::Zone) continue;
        std::string_view name = names_.lookup(r.id);                          // zone name
        const auto& lane = batch.context->devices[r.meta.dev].lanes[r.meta.lane];  // chip, core x/y, risc, role
        // ... aggregate: e.g. per-op rows keyed on r.prog, using r.data.zone.start / .duration ...
    }
}
```

## Try it end to end

The built-in CSV consumer:

```
TT_METAL_STREAMING_PROFILER=1 TT_METAL_PERF_DEBUG_OPS_CSV=/tmp/ops.csv python your_model.py
```

One row per op (kernel start/end unions, per-core and per-RISC splits), joinable against a
classic `ops_perf_results` CSV on GLOBAL CALL COUNT.
`tests/ttnn/tracy/test_perf_debug_ops_csv.py` runs exactly this.

## Two rules

Don't register/unregister from inside a callback, and don't block in it.

---

*There is also a built-in callback that pushes zones to a Tracy timeline; enable it with
`TT_METAL_STREAMING_PROFILER_TRACY=1`.*
