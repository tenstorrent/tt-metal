# Streaming profiler telemetry

A kernel sends a telemetry metric to an external process, a separate process from the Metal process that runs the
workload, with one `DevicePublish` call:

```cpp
// hostdevcommon/telemetry_sample.hpp, which kernels and the external process share
struct TelemetryMetadata {
    String name;
};

// the kernel
#include "api/debug/device_publish.h"
#include "hostdevcommon/telemetry_sample.hpp"

using namespace tt::tt_metal::experimental::streaming_profiler;
DevicePublish<TelemetryMetadata{"tt_bytes_processed_total"_s}, kSendToExternalProcess>(bytes);
```

A telemetry metric's name starts with `tt_`, and its value is a `uint64_t` or a `double`. The external process receives
it as a `TelemetrySample`, from `tt-metalium/experimental/telemetry_sample.hpp`:

```cpp
using TelemetryValue = std::variant<uint64_t, double>;
using TelemetrySample = Record<TelemetryValue, TelemetryMetadata>;

Callback callback = RegisterCallback(
    [](const Batch<TelemetrySample>& batch) {
        for (const TelemetrySample& sample : batch.records<TelemetrySample>()) {
            use(sample.metadata().name, sample.core(), sample.payload(), sample.published_at().time());
        }
    },
    "my-telemetry",
    /*external_process=*/true);
```

The host matches a call site by its metadata type and the type of its value, so a `TelemetrySample` callback receives
only `TelemetryMetadata` records whose value is a `uint64_t` or a `double`. A call site that passes any other type, a
`uint32_t` included, never reaches it: cast the value first.

## Running it

This directory holds the two sides:

- `kernels/publish_telemetry.cpp` sends three metrics at a fixed rate: `tt_loop_iteration`, `tt_bytes_processed_total`,
  a running total, and `tt_progress`, a `double`.
- `telemetry_metal_process.cpp` is the Metal process. It runs that kernel on four cores of device 0.
- `telemetry_external_process.cpp` is the external process. It logs every telemetry sample it receives.

Build with `./build_metal.sh --build-programming-examples`, then start the external process in one shell:

```
./build/programming_examples/metal_example_telemetry_external_process
```

and run the Metal process in another, from the tt-metal root, here for 5 s at 10 samples per second:

```
TT_METAL_STREAMING_PROFILER=1 ./build/programming_examples/metal_example_telemetry_metal_process 5 10
```

The external process prints one line per sample, starting with the time of day (UTC) the core published it:

```
Listening for telemetry. Press Enter to stop.
...
20:08:01.830463 chip 0 core (1,0) tt_loop_iteration: 0
20:08:01.830463 chip 0 core (1,0) tt_bytes_processed_total: 16384
20:08:01.830463 chip 0 core (1,0) tt_progress: 0.02
```

Only one process can be the external process, so stop `metal_example_external_process` before starting this one.
