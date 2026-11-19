# Functional decoder results

See [contract and results](functional_decoder/README.md),
[acceptance index](functional_decoder/acceptance.json), and
[context contract](context_contract.json).

Warmed timing totals use the filtered tt-perf-report CSV column
`Device Time [μs]` (microseconds); inter-op gaps use `Op-to-Op Gap [μs]`.
Human tables, CSVs, raw compressed ops and provenance are in
`functional_decoder/perf/{linear_attention,full_attention}/`.

## Warmed per-layer performance

Batch 1, prefill 512 tokens, traced decode at context 513; one Blackhole chip.
Numbers are **kernel sum / kernel plus inter-op gaps, milliseconds**. These are
decoder-layer timings, not full-model throughput.

| Layer kind | Warmed prefill ms | Warmed traced decode ms |
|---|---:|---:|
| linear_attention | 16.051 / 18.580 | 2.735 / 2.798 |
| full_attention | 6.317 / 6.479 | 2.326 / 2.371 |

Tables, filtered CSVs, compressed raw ops and provenance are in
[linear attention reports](functional_decoder/perf/linear_attention/metrics.json) and
[full attention reports](functional_decoder/perf/full_attention/metrics.json). Each directory has
`prefill_perf_report.txt/.csv` and `decode_perf_report.txt/.csv`.
Totals use the filtered `Device Time [μs]` and `Op-to-Op Gap [μs]` columns;
reported overlap is zero. `tt-perf-report==1.5.0 --arch blackhole` uses its P150
nominal roofline reference for this P300 host; measured durations are unchanged.
The unclassified reshape warning concerns category labeling only.

[Profiler audit](functional_decoder/profiler_audit.json) confirms final collection has no dropped
markers. Capacity is 5000 programs, with profiler drains between complete
passes and outside signposts. Earlier overflow runs are historical evidence.
Raw `ops.csv.gz` can be decompressed to replay the exact report commands saved
in metrics.json. The raw compressed console logs preserve compiler output.

Accepted performance was rerun with allocation tracking and tracebacks disabled,
Watcher unset, and program-cache exclusion explicitly disabled. See
[review remediation](functional_decoder/work_log.md) and the environment in each metrics JSON.
Prior tracked profiles remain under `perf_tracked/` as historical evidence.
