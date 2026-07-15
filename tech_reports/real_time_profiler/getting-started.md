# Real-time profiler — getting started

The **real-time profiler** (RT profiler) streams per-program timing from the device over the existing fast-dispatch path (D2H socket). Each completed program yields a `ProgramRealtimeRecord`: `runtime_id`, raw `start_timestamp` / `end_timestamp`, device `frequency` (cycles per ns), `chip_id`, and `kernel_sources` (paths for that program).

You can register **multiple** callbacks; they run on the profiler receiver thread in registration order. Use `UnregisterProgramRealtimeProfilerCallback(handle)` when done (Python: `ttnn.device.UnregisterProgramRealtimeProfilerCallback`).

On some dispatch setups (e.g. ETH dispatch, remote chips without the needed resources) the profiler stays inactive — check `ttnn.device.IsProgramRealtimeProfilerActive()` before asserting on record counts.

**C++:** `tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback` in [`realtime_profiler.hpp`](../../tt_metal/api/tt-metalium/experimental/realtime_profiler.hpp). Example streaming to disk: [`test_realtime_profiler_csv.cpp`](../../tt_metal/programming_examples/profiler/test_realtime_profiler_csv/test_realtime_profiler_csv.cpp).

---

## Register a callback (Python) — append JSON lines

```python
import json
import threading

import ttnn

out = open("rt_records.jsonl", "a")
lock = threading.Lock()

def on_record(record):
    row = {
        "runtime_id": record.runtime_id,
        "chip_id": record.chip_id,
        "start_timestamp": record.start_timestamp,
        "end_timestamp": record.end_timestamp,
        "frequency": record.frequency,
        "kernel_sources": list(record.kernel_sources),
    }
    with lock:
        out.write(json.dumps(row) + "\n")
        out.flush()

handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_record)
try:
    # open device, run workloads, synchronize...
    pass
finally:
    ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    out.close()
```

Same pattern is used in tests, e.g. [`tests/ttnn/tracy/matmul_workload.py`](../../tests/ttnn/tracy/matmul_workload.py) (collects dicts, then writes JSON after the run).

---

## Tracy default support

Metal wires a **`RealtimeProfilerTracyHandler`** that also registers on the same callback list. Records are pushed into Tracy’s Tenstorrent **device** timeline (per-chip context, calibration, program zones, optional sync-check markers). Your custom callbacks still run; you do not replace Tracy, you add alongside it.

For Tracy usage and the GUI, see [MetalProfiler](../MetalProfiler/metal-profiler.md) and the [Tracy profiler docs](../../docs/source/tt-metalium/tools/tracy_profiler.rst).
