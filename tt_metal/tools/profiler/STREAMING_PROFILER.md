# Using the streaming profiler with your own callback

The streaming profiler drains device zones to the host and hands decoded records to any
callback you register.

## Enable (Blackhole)

```
export TT_METAL_STREAMING_PROFILER=1
```

One switch — it also arms the device-side markers.

## Register a callback

From `tools/profiler/streaming_profiler_consumer.hpp`:

```cpp
auto h = streaming_profiler::register_consumer("my-sink",
    [](const streaming_profiler::StreamingProfilerRecordBatch& b) { /* ... */ });
// later: streaming_profiler::unregister_consumer(h);
```

Register any time — before the device opens or mid-capture. Your callback runs on its own
thread; if you're slow you drop only your own records (`b.dropped_delta`), never anyone
else's. The batch span is only valid during the call, so copy what you keep.

## What you get

Zones arrive **whole**: a zone is one record with a start and a duration. On the wire the device
ships most zones atomically (one 3-word packet at scope close, carrying end + duration); the kinds
that still ship as start/end pairs (the producer-stall zone, the >3.2 s long-zone fallback, DRISC
relay self-zones) are paired for you on the host. Either way you never see halves.

```cpp
enum class StreamingProfilerRecType : uint32_t {
    Zone = 1,   // a complete zone: data.zone = {start, duration}
    Data = 3,   // point marker with payload: data.ts; payload follows via Ext (+ Cont)
    Event = 4,  // point marker, no payload: data.ts; complete in itself
    Ext = 5,    // Data continuation: id = payload word count, data.ext = payload words 1-2
    Cont = 6,   // one uint64 of Data payload (words 3 and up): data.payload
};

struct StreamingProfilerRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;  // which (core, RISC) stream: lane = core_index * 5 + risc
    uint32_t dev : 3;    // device index into the capture context
    StreamingProfilerRecType type : 3;
};

struct StreamingProfilerRec {
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
    StreamingProfilerRecMeta meta;
    uint32_t prog;          // runtime host-id of the op this lane is executing (0 = none yet)
};
```

Ordering: cross-lane interleaving is arbitrary — key any state you keep by
`(meta.dev, meta.lane)`. A zone is delivered when it **ends**, so under nesting
`data.zone.start` isn't monotonic within a lane; `start + duration` is complete information
either way.

## The call pattern

The ops-CSV consumer in `tools/profiler/streaming_profiler_ops_csv.{hpp,cpp}` is the full working
reference.

```cpp
void MyConsumer::operator()(const streaming_profiler::StreamingProfilerRecordBatch& batch) {
    names_.refresh();  // ZoneNameMirror member: names arrive as kernels JIT, refresh once per batch
    for (const auto& r : batch.records) {
        if (r.meta.type != StreamingProfilerRecType::Zone) continue;
        std::string_view name = names_.lookup(r.id);                          // zone name
        const auto& lane = batch.context->devices[r.meta.dev].lanes[r.meta.lane];  // chip, core x/y, risc, role
        // ... aggregate: e.g. per-op rows keyed on r.prog, using r.data.zone.start / .duration ...
    }
}
```

## Try it end to end

The built-in CSV consumer:

```
TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_OPS_CSV=/tmp/ops.csv python your_model.py
```

One row per op (kernel start/end unions, per-core and per-RISC splits), joinable against a
classic `ops_perf_results` CSV on GLOBAL CALL COUNT.
`tests/ttnn/tracy/test_streaming_profiler_ops_csv.py` runs exactly this.

## Two rules

Don't register/unregister from inside a callback, and don't block in it.

---

A built-in callback pushes zones to a Tracy timeline; enable it with
`TT_METAL_STREAMING_PROFILER_TRACY=1`.
