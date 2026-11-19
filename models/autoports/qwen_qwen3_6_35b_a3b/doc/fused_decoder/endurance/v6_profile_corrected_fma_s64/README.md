# Rejected S64 profiler capture

This capture is incomplete and cannot supply valid prefill/decode performance tables or top-operation conclusions. The decoder correctness checks succeeded and the device closed cleanly, but Tracy terminated host capture during measured prefill. No clean latency claim is made.

`../v6_profile_corrected_fma_s64.log:34` records:

> Instrumentation failure: Too many source locations. You cannot have more than 32K static or dynamic source locations.

The generated raw CSV has 33,013 rows: 33,012 device operations and only `PERF_PREFILL` (zero-based row27,513). `PERF_PREFILL_END`, `PERF_DECODE`, and `PERF_DECODE_END` are absent. There are **zero** trace replay sessions, so the required 20 total identical sessions and16 measured sessions fail. All retained operation durations are present and positive. There are no device dropped-marker/buffer-full warnings; that does not establish capture completeness. Overlap cannot be validated for either requested phase because neither window is complete.

The original raw CSV is preserved and copied byte-for-byte to `ops.csv`; `ops_source.txt` names it. `profiler_completeness.json` and `rejected_capture.json` contain hashes, checks and failure evidence. `validate_capture_host.py` is the executed standard-library validator; it imports neither TTNN nor Torch and uses no device.

No normal phase report was generated. In tt-perf-report1.5, `python_env/lib/python3.12/site-packages/tt_perf_report/perf_report.py:984-1040` merely warns when requested start/end signposts are missing and can proceed with a different/incomplete window. Generating those tables under ordinary prefill/decode names would misrepresent the retained data. Once a replacement capture passes signpost/session validation, the required commands are:

```bash
tt-perf-report ops.csv --active-experts 8 --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --csv prefill_perf_report.csv
tt-perf-report ops.csv --active-experts 8 --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary --no-color > prefill_perf_report.txt
tt-perf-report ops.csv --active-experts 8 --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --csv decode_perf_report.csv
tt-perf-report ops.csv --active-experts 8 --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary --no-color > decode_perf_report.txt
```

## Source-supported repair

Use `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1`, retaining `TT_METAL_DEVICE_PROFILER=1` and file dumps. `rerun_csv_only_device_zones.sh` records the exact fresh-path invocation, with the existing chunk-draining wrapper and snapshot unchanged. The script has not been executed by this source-only agent. The parent owns scheduling and hardware validation.

Python line profiling was already disabled: `tools/tracy/__main__.py:20-21` defines `-p` as enabled-zones-only and `-l` as line profiling; this invocation supplied `-p` without `-l`. `tools/tracy/__init__.py:77-86` does not install automatic function profiling in partial mode. Removing a nonexistent line-profile flag cannot solve this failure.

The device GUI path explains the source-location growth: `tt_metal/third_party/tracy/public/tracy/TracyTTDevice.hpp:211-220` retains a unique `OP ID:<runtime_host_id>` function string for BRISC/ERISC firmware zones, explicitly noting the32K source-location limit. `PushStartMarker` at285-304 allocates the source location using that run-ID string. `tt_metal/third_party/tracy/server/TracyWorker.cpp:3488-3503` and3887-3897 reject static/dynamic location tables exceeding signed16-bit capacity. This mechanism matches termination near33K retained operations; the memo does not claim an independent census of the internal Tracy location table.

The supported runtime switch is parsed at `tt_metal/llrt/rtoptions.cpp:1375-1385`; it is effective when the device profiler is enabled. `tt_metal/impl/profiler/profiler.cpp:2663-2671` returns before GUI device-marker injection when the switch is set. The same collection path independently generates device analyses and queues `writeDeviceResultsToFiles` before calling that injection function (`profiler.cpp:2514-2522`). File output has its own independent disable flag at2635-2640. Thus keep `TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES` unset.

This removes the device timeline visualization from the Tracy GUI; it does not remove device operation timing CSVs. `tools/tracy/process_ops_logs.py:351` parses `cpp_device_perf_report.csv`, and `append_device_data` at984 merges those measured durations with host operation metadata. TTNN operation profiling and trace tracking remain enabled by `tools/tracy/__main__.py:333-336`; signposts remain explicit host messages from `tools/tracy/__init__.py:41-49`. No shared/native edit, runtime arithmetic change, or new profiling wrapper is needed. The repaired capture must still pass20 identical total trace sequences,16 measured sessions, complete windows, positive durations, no drops, and zero report overlap before producing accepted tables.
