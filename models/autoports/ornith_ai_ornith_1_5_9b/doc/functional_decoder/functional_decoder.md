# Functional decoder measurements

Single Blackhole chip 0, 1x1 mesh, serialized runs, real weights, BF16 weights/activations, HiFi4 with FP32 destination accumulation. Batch 1. Prefill length 1024 from an empty state; decode writes position 1024 against that prefix. Internal GDN chunks are 32; outer prefill chunks are 1024. Watcher and allocation diagnostics are off during profiling and validated separately.

| Layer kind | Mode | Device-kernel sum (ms) | Op gaps (ms) | Sum including gaps (ms) | Device op rows |
|---|---|---:|---:|---:|---:|
| linear | prefill | 13.239976 | 0.103700 | 13.343676 | 103 |
| linear | decode | 1.654715 | 0.084252 | 1.738967 | 116 |
| full | prefill | 8.194264 | 0.053003 | 8.247267 | 70 |
| full | decode | 1.840723 | 0.067933 | 1.908656 | 71 |

Numbers sum the filtered tt-perf-report CSV columns `Device Time [μs]` and `Op-to-Op Gap [μs]`, converting microseconds to milliseconds. They are device execution measurements, not whole-model token latency. Gap sums retain the report’s handling of overlaps; no host setup, weight conversion or HF reference is included. One warmed invocation per mode is reported, with no statistical confidence interval.

The separately measured synchronized traced-decode host wall times are 1.819328 ms (linear) and 2.011775 ms (full). These include replay dispatch/synchronization overhead and are not substituted for device time.

Both reports select explicit `PERF_PREFILL` / `PERF_PREFILL_END` or `PERF_DECODE` / `PERF_DECODE_END` windows. Compile/warmup, state restoration, and input setup precede the measured windows. Decode rows carry trace IDs and replay-session IDs; `performance_summary.json` retains them. Numerical comparisons from replay and runtime guards passed in the same captures (`linear_profile.json`, `full_profile.json`).

Per-kind artifacts under `tracy/linear/` and `tracy/full/`:

- `prefill_perf_report.txt`, `decode_perf_report.txt`: human-readable tt-perf-report tables.
- `prefill_perf_report.csv`, `decode_perf_report.csv`: filtered timing rows.
- `prefill_ops.csv.gz`, `decode_ops.csv.gz`: lossless raw postprocessed Tracy ops CSVs; originals remain local.
- `prefill_provenance.json`, `decode_provenance.json`: source hashes and exact report commands.
- `*_perf_report.console.log.gz`: report command/status output.

Capture: `bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_profiles.sh`; render: `python models/autoports/ornith_ai_ornith_1_5_9b/tests/perf_reports.py`. The report renderer restores an archived original ops CSV when needed.

The measurement-only kernel overlay uses byte-identical symlinks to disambiguate two 16-bit profiler source IDs in this checkout path. See `profiler_collision_diagnosis.json`, `profiler_collision_resolution.json`, `AUTOFIX.md` and `tests/run_profiles.sh`. No shared kernel or profiler assertion was changed. Failed captures under `linear_hash_collision` and `linear_overlay_missing_headers` are excluded. The successful metadata contains zero collisions; every decode row in each selected window has a trace ID.

Large intermediate device-marker streams and Tracy binaries remain local, listed with hashes in `archive_manifest.json`. Postprocessed ops CSVs, filtered CSVs, human tables, target logs and metadata are committed. Restore any archived raw file with `gzip -dc path.gz > path`; archived bytes are verified by the manifest hash.
