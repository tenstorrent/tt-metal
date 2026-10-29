# Streaming profiler external process

A kernel sends a metric to an external process, a separate process from the Metal process that runs the workload,
with one `DevicePublish` call:

```cpp
// sample.hpp, which kernels and the external process share
struct SampleMetadata {
    String name;
};

// the kernel
#include "api/debug/device_publish.h"
#include "../sample.hpp"

using namespace tt::tt_metal::experimental::device_observability;
DevicePublish<SampleMetadata{"bytes_processed_total"_s}, kSendToExternalProcess>(bytes);
```

In actual use, a metadata type like `SampleMetadata` belongs in `hostdevcommon/`, which kernels and host code both
include. The example keeps `sample.hpp` beside its sources.

Device observability carries the record off the device, and the external process receives it through a callback, with
the core it came from and its time on the host's steady clock:

```cpp
using Number = std::variant<uint64_t, int64_t, uint32_t, int32_t, double, float>;
using Sample = Record<Number, SampleMetadata>;

Callback callback = RegisterCallback(
    [](const Batch<Sample>& batch) {
        for (const Sample& sample : batch.records<Sample>()) {
            use(sample.metadata().name, sample.core(), sample.payload(), sample.published_at().time());
        }
    },
    "my-external-process",
    /*external_process=*/true);
```

`kSendToExternalProcess` sends a record to the external process as well as the Metal process. Without it, the record
stays in the Metal process.

Registering a callback with `external_process` set makes a process the external process, and a Metal process with device
observability on connects to it when it opens its devices. The only IPC between the two is a Unix domain socket, which
the Metal process uses to send the external process what it needs to read and decode records. The records themselves
stream from the device to the external process over a D2H socket, without passing through the Metal process.

## Running it

This directory holds the two sides:

- `kernels/publish_metrics.cpp` sends two metrics at a fixed rate: `loop_iteration`, and `bytes_processed_total`, a
  running total.
- `metal_process.cpp` is the Metal process. It runs that kernel on four cores of device 0.
- `external_process.cpp` is the external process. It logs every metric it receives.

Build with `./build_metal.sh --build-programming-examples`, then start the external process in one shell:

```
./build/programming_examples/metal_example_external_process
```

and run the Metal process in another, from the tt-metal root, here for 5 s at 10 samples per second:

```
TT_METAL_DEVICE_OBSERVABILITY=1 ./build/programming_examples/metal_example_metal_process 5 10
```

The external process prints one line per record, starting with the time of day (UTC) the core published it:

```
Listening. Press Enter to stop.
...
20:08:01.830463 chip 0 core (1,0) loop_iteration: 1
20:08:01.830463 chip 0 core (1,0) bytes_processed_total: 16384
20:08:01.830463 chip 0 core (0,1) loop_iteration: 1
20:08:01.830463 chip 0 core (0,1) bytes_processed_total: 32768
```

The external process keeps running across Metal processes. Either side may start first: a Metal process retries every
second, and its records reach the external process from the moment it connects.
