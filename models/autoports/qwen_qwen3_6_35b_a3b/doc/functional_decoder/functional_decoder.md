# Functional decoder device performance

Measured signposted warmed passes on physical Blackhole chip 0, serialized, without watcher.
Prefill uses one warmed 256-token pass; decode uses 16 warmed trace replays at position 256.
The CSV column `Device Time [μs]` is already in microseconds. Each operation row is counted once;
tt-perf-report's CSV does not include the terminal table's total row. Decode totals are divided by 16.
Device time below excludes dispatch gaps and is not a host-wall latency or full-model throughput claim.
Profiler buffers are drained between complete decode replays; gaps include this measurement overhead.
The gap column records the report's separate op-to-op gaps; no overlap is permitted by this summarizer.

| Layer kind | Phase | Device time/pass (μs) | Op-to-op gap/pass (μs) | Passes | Human table |
|---|---|---:|---:|---:|---|
| linear_attention | prefill | 69685.930 | 470.589 | 1 | [report](tracy/layer_0/prefill_perf_report.txt) |
| linear_attention | decode | 60378.858 | 35941.092 | 16 | [report](tracy/layer_0/decode_perf_report.txt) |
| full_attention | prefill | 68881.854 | 476.021 | 1 | [report](tracy/layer_3/prefill_perf_report.txt) |
| full_attention | decode | 60137.201 | 21419.468 | 16 | [report](tracy/layer_3/decode_perf_report.txt) |

Source commands: `tests/collect_perf.sh`; CSV hashes and run provenance: `perf_summary.json`.
