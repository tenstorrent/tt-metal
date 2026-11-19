# Full resident PLE streaming profiler path

Status: source-supported capture path; five CPU tests pass, including the real
Tracy host-log parser/exporter and the installed `tt-perf-report` executable.
No streaming hardware capture has been run by this author. CPU fixture times
are deliberately synthetic and are not model performance measurements.
Evidence: `streaming_profile_cpu_tests.log`.

## Why continuous collection is needed

The Blackhole HAL profiler reservation is 38,400 bytes per supported program:
20 rounded cores per bank × five RISCs × 48 bytes × eight DRAM banks.
`profiler_state_manager.cpp` derives the 48-byte record allowance from two
program-ID words plus four guaranteed markers, two UINT32 words per marker.
`bh_hal.cpp` reserves for 154 cores rounded across eight banks, even when fewer
cores participate. Thus support counts 1,000/6,000/10,000/20,000 reserve
38.4/230.4/384/768 MB respectively. The prior 288 MB estimate for 10,000 is
incorrect for this HAL reservation.

Using the measured 514,951,680-byte free-resident baseline, which already
includes the default 38.4 MB profiler reservation, count 10,000 leaves about
169,351,680 bytes before activations; count 20,000 exceeds the resident budget
by 214,648,320 bytes. These are arithmetic estimates, not new capacity tests.
The final runtime may change the baseline.

`TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1` disables the device GUI push only;
it does not reduce the standard device buffer or continuously drain it.
Standard `ReadDeviceProfiler` finishes command queues before collecting data;
inserting such calls inside the measured forward changes its execution.
`--dump-device-data-mid-run` only processes explicitly requested dumps.
Accumulation discards the operation identity needed for the required report.

Streaming supports `TT_METAL_STREAMING_PROFILER_DRAM_MB=0` (direct to host).
Its default 128 MiB per bank would reserve 1 GiB and cannot fit this baseline.
Zero removes the 38.4 MB classic reservation, giving a theoretical
553,351,680 bytes before other streaming resources and activations. A 16 MiB
per-bank spool instead leaves approximately 419,133,952 bytes. Host FIFO
512 MiB is per relay/socket, uses pinned host memory, and does not consume
this device DRAM budget. Backpressure can stall producers; completed capture
logs and merge provenance retain reported stall counts. Hardware activation,
capacity, losslessness, and overhead still require an actual serialized run.

Source anchors: `tt_metal/impl/profiler/profiler_state_manager.cpp`,
`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp`,
`tt_metal/llrt/rtoptions.cpp`, and `tt_metal/impl/streaming_profiler/{ops_csv,
receiver,service,device_programs}.cpp`.

## Capture and offline report

Run from the checkout after the owner releases the serialized hardware slot.
The following is the prepared hardware command, not a claimed completed run.
Use a distinct directory for each mode. `profile_streaming_capture` delegates
to normal Tracy capture and CSV export, replacing only its post-run attempt
to join absent standard device logs with the existing host CSV exporter.
The standard profile path remains unchanged without `--streaming-profile`.

```bash
PLE_PROFILE_MODE=prefill
PLE_PROFILE_DIR="$PWD/models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/ple_streaming/$PLE_PROFILE_MODE"
mkdir -p "$PLE_PROFILE_DIR"
env -u TT_METAL_DEVICE_PROFILER -u TT_METAL_WATCHER \
  TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_STREAMING_PROFILER=1 \
  TT_METAL_STREAMING_PROFILER_DRAM_MB=0 \
  TT_METAL_STREAMING_PROFILER_FIFO_MB=512 \
  TT_METAL_STREAMING_PROFILER_OPS_CSV="$PLE_PROFILE_DIR/streaming_ops.csv" \
  TT_METAL_STREAMING_PROFILER_TRACY=0 \
  PYTHONPATH=. \
  timeout 1800 python_env/bin/python \
    -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_streaming_capture \
    -r -p -v --no-device --no-web-server \
    --tracy-tools-folder build/tools/profiler/bin -o "$PLE_PROFILE_DIR/raw" \
    -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_decoder \
    --snapshot /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa \
    --layer 1 --seq-len 32 --mode "$PLE_PROFILE_MODE" --expert-dtype bfloat8_b \
    --cache-dir /home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide \
    --streaming-profile --output "$PLE_PROFILE_DIR/profile.json" \
    > "$PLE_PROFILE_DIR/capture.log" 2>&1
```

After success, repeat with `PLE_PROFILE_MODE=decode`. Each process loads the
complete real table and weights, warms the same shape, restores state, and
measures exactly one prefill or one warmed trace replay. All tensor transfers
remain outside the measured model invocation. The standard profiler-drain
API is omitted in streaming mode because its implementation is a no-op when
the classic profiler is disabled; device close flushes and verifies streaming.

Select the single generated `ops_perf_results*.csv` under that run's
`raw/reports` directory and pass it as `PLE_HOST_CSV`. The wrapper generates
only one report per capture. The following offline command opens no device:

```bash
PYTHONPATH=. python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.report_streaming_performance \
  --host-csv "$PLE_HOST_CSV" \
  --streaming-csv "$PLE_PROFILE_DIR/streaming_ops.csv" \
  --profile-json "$PLE_PROFILE_DIR/profile.json" \
  --capture-log "$PLE_PROFILE_DIR/capture.log" \
  --output-dir "$PLE_PROFILE_DIR/report"
```

The merger requires activation plus final verification logs; rejects dropped
or missed data, stranded device words, duplicate/missing IDs, inconsistent
execution counts, incorrect/repeated markers, and reversed cycle intervals.
It checks every raw dispatcher ID in the saved half-open counter range.
Host CSV IDs encode `(raw_id << 10) | device_id`; streaming CSV IDs are raw.
Decode metadata comes from the marked capture, while durations come from
the final execution of those IDs, after exactly one unmeasured replay and
one measured replay. It allows capture to execute or only record commands,
but every ID must have the same complete execution ordinal sequence.
Captured host durations are cleared rather than represented as replay times.

Original host signposts are retained, all selected IDs and per-execution
cycle ranges are archived in `streaming_selection.json.xz`, and input/output
SHA256 identities and selection proof are in `streaming_merge.json`.
The existing report tool runs with `--tracing-mode` to preserve the verified
operation order and must retain exactly the selected operation count.
Artifacts include human-readable `<mode>_perf_report.txt`, CSV/XZ, report
console logs, `latency_summary.json`, and full `provenance.json` commands.
Device per-operation durations and their sum are measured; device gaps and
whole-pass elapsed latency are unavailable and stay blank/null. The report
does not invent gaps or substitute host elapsed time for device latency.

## CPU validation

```bash
PYTHONPATH=. python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.test_streaming_profile_merge_cpu
```

Five tests validate the final replay, capture-without-execution alternative,
sole eager execution, failure controls, raw host-log import with marker/ID
preservation, and actual human-table/CSV generation with null gap summaries.
The suite asserts that TTNN was not imported. Applicable pre-commit hooks pass.
