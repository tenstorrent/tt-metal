# Streaming profiler

The streaming profiler delivers the zones, events and values of device kernels to the host while the workload runs,
with every processor of every device on one timeline. Kernels mark them with the macros of
`api/debug/kernel_profiler.hpp` (`DeviceZoneScopedN`, `DeviceRecordEvent`, `DeviceTimestampedData`); the host receives
them in callbacks, in Tracy or as CSV files. The records travel through
[tt-insight](https://github.com/tenstorrent/tt-insight), a submodule at `tt_metal/third_party/tt-insight`.

## Using it

```cpp
#include <tt-metalium/experimental/streaming_profiler.hpp>
using namespace tt::tt_metal::experimental::streaming_profiler;

Callback zones = RegisterCallback([](const Batch<Zone>& batch) {
    for (const Zone& zone : batch.records<Zone>()) {
        use(zone.descriptor().name, zone.core().logical, zone.duration());
    }
}, "my-tool");
```

A callback runs on a thread of its own, one call at a time, until its `Callback` is destroyed or reset. A record is
valid until the callback returns; a copy of it can be kept. A capture starts when Metal opens devices and stops when
it closes them, and a callback registered before that receives every record. The last ones can arrive after the
devices are closed: a callback has them all once it has ended, so close the devices, end the callback
(`zones.reset()`), and then read what it gathered.

| Variable | |
|---|---|
| `TT_METAL_STREAMING_PROFILER=1` | Turns it on. Excludes `TT_METAL_DEVICE_PROFILER`, device prints and the tensor prefetcher. |
| `TT_METAL_STREAMING_PROFILER_TRACY=1` | Sends every record to a connected Tracy. |
| `TT_METAL_STREAMING_PROFILER_ZONE_CSV=<path>`, `..._OPS_CSV=<path>` | Write the zones or the ops as CSV when the last capture stops. |
| `TT_METAL_STREAMING_PROFILER_FIFO_MB=<n>` | Host memory for each device's records, 1024 by default, which the device's drainers share: how far a callback can fall behind before it loses records. |
| `TT_METAL_STREAMING_PROFILER_DRAM_MB=<n>` | Device DRAM for each drainer on a DRAM core, off by default: a size lets the drainers take in a burst faster than the link to the host carries, and while they keep records there a workload's own DRAM traffic loses up to a quarter of its rate. |
| `TT_METAL_STREAMING_PROFILER_INLINE_ENABLED=0` | Builds every marker out of line: smaller kernels, one call more a marker. |
| `TT_METAL_STREAMING_PROFILER_SYNC_EVENTS=1` | Records the sync event markers (`SYNC_WAIT`, `SYNC_SIGNAL`). |
| `TT_METAL_STREAMING_PROFILER_SYNC_CHECK=1` | Logs what a capture does with what Metal lends it, and runs tt-insight's clock sync self-test, which measures how closely the devices' clocks line up. For checking a setup, not for runs that are measured. |

## What it needs

- Blackhole devices, and a host whose IOMMU translates the cards' DMA.
- On each device, the DRAM cores Metal does not reach DRAM through, where the profiler's drainers run, and one idle
  Ethernet core, which keeps the device's time (two with the sync check). On a firmware bundle older than 19.12.0,
  or with `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=0`, the idle Ethernet cores drain instead.
- The profiler area of every Tensix and active Ethernet mailbox, and `..._FIFO_MB` of pinned host memory a device.

## What it prints

```
[streaming profiler] active on 1 device(s)
tt-insight: capture of 1 device ran 5.9 s: 418 KB received, none lost
```

The first line is Metal's, at each capture's start. With the sync check on, Metal logs what it measured at each stop:
`[streaming profiler] sync check on 8 device(s): 1960328 readings, error p50 0.180 ns, p99.9 0.560 ns, max 0.940 ns`.
Lines that start `tt-insight:` are on standard error: one at each
stop with what was received and lost, and one for each loss or fault when it is first known.
`tt_metal/third_party/tt-insight/.claude/skills/troubleshooting-tt-insight/` explains each. `TT_INSIGHT_VERBOSE=1`
adds detail.

## Limits

- A full ring makes its processor wait, and the wait arrives as a zone named `PROFILER-STALL`. A callback that
  falls behind loses records instead, counted in `Batch::dropped_bytes()`.
- Metal runs no kernels on DRAM cores while the profiler is on. Programs on idle Ethernet cores and dispatch kernels
  publish nothing.
- The process should close its devices before it exits.
- On a machine of several cards, a capture warns once for each Ethernet link that joins no other device of the
  capture.
