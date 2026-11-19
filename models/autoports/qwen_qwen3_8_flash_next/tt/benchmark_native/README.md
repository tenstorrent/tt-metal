# Host completion query binding

This model-local extension exposes the existing
`tt::tt_metal::distributed::EventQuery(const MeshEvent&)`. In fast-dispatch mode
that function reads the host completion counters for every participating mesh
coordinate. It does not enqueue a wait, synchronize a device, read a tensor, or
enable the device profiler. Use the same nanobind source and Python ABI as TTNN.

Built in this checkout with:

```sh
cmake -S models/autoports/qwen_qwen3_8_flash_next/tt/benchmark_native \
  -B models/autoports/qwen_qwen3_8_flash_next/tt/benchmark_native/build \
  -DTT_METAL_ROOT=/home/ttuser/qwen38-flash/tt-metal \
  -Dnanobind_DIR=/home/ttuser/qwen38-flash/tt-metal/.cpmcache/nanobind/7965351fcedc97e6ade403d51a150a266d188dc6/cmake \
  -DPython_EXECUTABLE=/home/ttuser/qwen38-flash/vllm-env/bin/python
cmake --build models/autoports/qwen_qwen3_8_flash_next/tt/benchmark_native/build -j 4
```

The generated `_benchmark_event_query.cpython-312-*.so` is a local build output, not a
source artifact. Native binding interoperability with live MeshEvent objects
must be validated by the owned server's collector readiness smoke; compilation
and host tests alone do not prove device behavior.

`PhaseRecorder` buffers `begin`, `end`, completion observation, and host output
processing timestamps. `end(ticket, event=existing_readback_event)` reuses the
normal decode readback marker. Without an event, it records one CQ0 completion
marker after the complete prefill submission. This adds a dispatch marker and
host notification, whose overhead must be disclosed; it never inserts an
explicit wait or blocks the serving thread for completion. The background thread
only queries the host event counters. Python scheduling and GIL contention can
delay observation: every event retains the last-known-incomplete/first-observed-
complete bracket, and the snapshot records maximum observer poll spacing.
The observer timestamp is a host observation, not an exact device timestamp.

The caller must invoke `host_complete(ticket)` after existing readback/token
processing. It is deliberately separate from event completion. `annotate` copies
metadata such as actual internal chunk spans. Create `<log_path>.flush` only
after measured traffic finishes. The observer writes an atomic JSON snapshot to
`log_path` and removes the marker on success. Pending events and observer errors
remain explicit. Call `close()` before mesh teardown. Startup readiness can call
`flush()` before traffic to export the imported module/native paths and recorder
configuration even when no events have run yet.

The recorder does not invent durations by subtracting overlapping request
latencies. The collector must preserve CQ submission order, actual request/slot
mapping, event uncertainty, and separate host completion markers when partitioning
the shared timeline. Reusing HTTP TTFT/TPOT as phase times is not supported.

Host-only validation (avoids the repository's hardware-importing conftest):

```sh
/home/ttuser/qwen38-flash/vllm-env/bin/python -m pytest --noconftest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_benchmark_phases_cpu.py
```

The initial build needed `SPDLOG_FMT_EXTERNAL` and `FMT_HEADER_ONLY=1` for the checkout headers. The first live pilot then exposed an ABI registry mismatch: TTNN is non-limited, while a STABLE_ABI extension uses a separate nanobind registry and cannot accept its MeshEvent objects. CMake now uses the same non-limited ABI. Import success alone was insufficient; a live prefill/decode completion smoke is required. Query failures immediately export their error and buffered events before stopping, so diagnostics survive without an explicit close.
