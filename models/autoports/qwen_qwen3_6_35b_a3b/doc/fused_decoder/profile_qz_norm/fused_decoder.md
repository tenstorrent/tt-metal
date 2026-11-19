# Decoder device performance

Measured signposted warmed passes on physical Blackhole chip 0, serialized, without watcher.
Prefill uses one warmed 256-token pass; decode uses 16 warmed trace replays at position 256.
The CSV column `Device Time [μs]` is already in microseconds. Each operation row is counted once;
tt-perf-report's CSV does not include the terminal table's total row. Decode totals are divided by 16.
Device time below excludes dispatch gaps and is not a host-wall latency or full-model throughput claim.
Profiler buffers are drained between complete decode replays; gaps include this measurement overhead.
The gap column records the report's separate op-to-op gaps; no overlap is permitted by this summarizer.

| Layer kind | Phase | Device time/pass (μs) | Op-to-op gap/pass (μs) | Passes | Human table |
|---|---|---:|---:|---:|---|
| linear_attention | prefill | 165106.828 | 3251.017 | 1 | [report](tracy/layer_0/prefill_perf_report.txt) |
| linear_attention | decode | 1761.720 | 113432.408 | 16 | [report](tracy/layer_0/decode_perf_report.txt) |
| full_attention | prefill | 8780.662 | 646.002 | 1 | [report](tracy/layer_3/prefill_perf_report.txt) |
| full_attention | decode | 574.795 | 70790.514 | 16 | [report](tracy/layer_3/decode_perf_report.txt) |

Source commands: `tests/collect_fused_perf.sh`; CSV hashes and run provenance: `perf_summary.json`.
