# Streaming profiler in the Tracy GUI

Data is streamed off the device as it is produced, so capture runs continuously instead of stopping at
a kernel or program boundary. Host callbacks receive every record and decide what to do with it (see
[STREAMING_PROFILER_DETAILED.md](STREAMING_PROFILER_DETAILED.md)). A Tracy callback ships built in: set
`TT_METAL_STREAMING_PROFILER_TRACY=1` and every record is pushed to the GUI.

The three device-side primitives are below — device code, the host callback that receives it, and how
it renders. All three GIFs come from one 50-iteration demo kernel; one GPU context per worker core,
one row per RISC.

## 1. Zone scope — `DeviceZoneScopedN`

Device:

```cpp
void kernel_main() {
    DeviceZoneScopedN("MY-KERNEL");    // one zone spanning the whole kernel
    for (uint32_t it = 0; it < N_ITERS; it++) {
        DeviceZoneScopedN("compute");  // nested zone: opens here...
        do_compute();                  // ~20 us of work
    }                                  // ...closes at the end of the scope
}
```

Host callback — a zone arrives as one record, whole, when it closes. The public API is
`<tt-metalium/experimental/streaming_profiler.hpp>` (namespace `tt::tt_metal::experimental::streaming_profiler`);
subscribe with a callable taking the `Batch` of the channels you want:

```cpp
auto h = Subscribe("zone-sink", [](const Batch<Channel::Zones>& b) {
    for (const Zone& z : b.zones) {
        fmt::print("{}: {} ns on chip {} core ({},{}) {} (op {})\n",
            z.site.name,                 // "compute"
            z.duration().count(), z.core.chip_id, z.core.logical.x, z.core.logical.y,
            static_cast<int>(z.core.risc), z.runtime_id);
    }
});
// later: Unsubscribe(h);
```

![zone scopes](docs/zone_gifs/zone_scopes.gif)

A named RAII scope, alive until the end of its `{}`. Zones nest: every RISC row shows
`*-KERNEL` (the firmware wrapper) with `MY-KERNEL` under it and the per-iteration `compute`
zones one level deeper. Hovering shows the name and GPU execution time (~20 us here).

## 2. Timestamped data — `DeviceTimestampedData`

Device:

```cpp
uint64_t bytes_moved = 0;
for (uint32_t it = 0; it < N_ITERS; it++) {
    do_compute();
    bytes_moved += 2048;                                // any runtime value
    DeviceTimestampedData("BYTES-MOVED", bytes_moved);  // stamped with the device time
}
```

Host callback — a `TimestampedData` record arrives assembled, its payload as a span of uint64 words:

```cpp
auto h = Subscribe("data-sink", [](const Batch<Channel::TimestampedData>& b) {
    for (const TimestampedData& d : b.timestamped_data) {
        fmt::print("{} @ {}: value={}\n",
            d.site.name,                                     // "BYTES-MOVED"
            d.time().time_since_epoch().count(), d.payload[0]);
    }
});
```

![timestamped data](docs/zone_gifs/timestamped_data.gif)

A point event carrying a 64-bit runtime value. Renders as a triangle above the row; the tooltip
shows name, timestamp, and the value — here `Data: 49152` = `bytes_moved` after 24 iterations.

## 3. Flag — `DeviceFlag`

Device:

```cpp
for (uint32_t it = 0; it < N_ITERS; it++) {
    DeviceFlag("LOOP-START");   // a named instant, no payload
    do_compute();
}
```

Host callback — an `Event` is a name and a time, nothing else:

```cpp
auto h = Subscribe("flag-sink", [](const Batch<Channel::Events>& b) {
    for (const Event& e : b.events) {
        fmt::print("{} @ {} on core ({},{})\n",
            e.site.name, e.time().time_since_epoch().count(),   // "LOOP-START" @ host time
            e.core.logical.x, e.core.logical.y);
    }
});
```

![device flag](docs/zone_gifs/device_flag.gif)

The payload-free point event: a name and a device timestamp, nothing else. Use it to put a
moment (phase boundary, retry, error path) on the timeline. Here one `LOOP-START` per
iteration, next to that iteration's `BYTES-MOVED` on the same row.

## Where to go next

- [`STREAMING_PROFILER_DETAILED.md`](STREAMING_PROFILER_DETAILED.md) — the full reference: the three profiler
  modes and every environment variable, the relay/receiver architecture, the wire format, the offline tools.
- [`STREAMING_PROFILER_FINDINGS.md`](STREAMING_PROFILER_FINDINGS.md) — the dated record of findings and
  benchmarks behind the design.
