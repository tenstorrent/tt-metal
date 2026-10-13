# Real-time profiler — getting started

The **real-time profiler** (RT profiler) streams per-program timing from the device over the existing fast-dispatch path (D2H socket). Each completed program yields a `ProgramRealtimeRecord`: `runtime_id`, raw `start_timestamp` / `end_timestamp`, device `frequency` (cycles per ns), `chip_id`, and `kernel_sources` (paths for that program).

Callbacks are registered on a **MeshDevice** and receive records only for programs that ran on that mesh's local chips. A callback registered on a submesh is served by its parent mesh's profiler and sees only the submesh's chips. Register after opening the device; a callback stays registered until you unregister it or the mesh is closed, and closing first delivers the records still in flight. Unregistering after close is a no-op.

You can register **multiple** callbacks; they are invoked concurrently. If a callback shares a resource with other callbacks or across multiple meshes, access it in a thread-safe way (e.g. with a lock). Use `UnregisterProgramRealtimeProfilerCallback(mesh_device, handle)` when done (Python: `ttnn.device.UnregisterProgramRealtimeProfilerCallback(mesh_device, handle)`).

On some dispatch setups (e.g. ETH dispatch, remote chips without the needed resources) the profiler stays inactive — check `ttnn.device.IsProgramRealtimeProfilerActive(mesh_device)` before asserting on record counts.

**C++:** `tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback(mesh_device, callback)` in [`realtime_profiler.hpp`](../../tt_metal/api/tt-metalium/experimental/realtime_profiler.hpp). Example streaming to disk: [`test_realtime_profiler_csv.cpp`](../../tt_metal/programming_examples/profiler/test_realtime_profiler_csv/test_realtime_profiler_csv.cpp).

---

## Register a callback (Python) — append JSON lines

```python
import json

import ttnn

out = open("rt_records.jsonl", "a")
dropped_total = 0

def on_record_batch(batch):
    global dropped_total
    dropped_total += batch.dropped
    for record in batch.records:
        row = {
            "runtime_id": record.runtime_id,
            "chip_id": record.chip_id,
            "start_timestamp": record.start_timestamp,
            "end_timestamp": record.end_timestamp,
            "frequency": record.frequency,
            "kernel_sources": list(record.kernel_sources),
        }
        out.write(json.dumps(row) + "\n")
    out.flush()

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(mesh_device, on_record_batch)
try:
    # run workloads, synchronize...
    pass
finally:
    # Closing drains the remaining records into the callback first.
    ttnn.close_mesh_device(mesh_device)
    ttnn.device.UnregisterProgramRealtimeProfilerCallback(mesh_device, handle)
    out.close()
```

Same pattern is used in tests, e.g. [`tests/ttnn/tracy/matmul_workload.py`](../../tests/ttnn/tracy/matmul_workload.py) (collects dicts, then writes JSON after the run).

---

## Tracy default support

Each MeshDevice's profiler also feeds a **`RealtimeProfilerTracyHandler`** alongside the registered callbacks. Records are pushed into Tracy’s Tenstorrent **device** timeline (per-chip context, calibration, program zones, optional sync-check markers). Your custom callbacks still run; you do not replace Tracy, you add alongside it.

For Tracy usage and the GUI, see [MetalProfiler](../MetalProfiler/metal-profiler.md) and the [Tracy profiler docs](../../docs/source/tt-metalium/tools/tracy_profiler.rst).
