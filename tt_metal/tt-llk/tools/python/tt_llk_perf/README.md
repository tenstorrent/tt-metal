# tt_llk_perf

Host side of the perf counter infrastructure. Stdlib only, so it imports in the
metal wheel, the LLK CI container and a bare interpreter alike.

- `headers.py` parses the C++ headers in `tools/include/perf_counters/`:
  `counter_type_names()` gives ordinal -> name from the `PerfCounterType` enum in
  `types.h` (the firmware tags records with the ordinal); `bank_tables(arch)` gives
  bank -> `[CounterEntry(name, select, l1_mux)]` from `blackhole.h` / `wormhole.h` /
  `quasar.h` (the LLK harness decodes its config words with these; the Quasar L1 bank
  is empty). `find_include_dir()`
  resolves the header directory: explicit argument, `LLK_HOME`, this source tree,
  `TT_METAL_HOME`, then the ttnn wheel's package data.
- `metrics.py` is the derived-metric engine: adapt your data to `CounterView`
  and call `compute_metrics(view)`. Keys end in `_pct` (bounded) or `_ratio`
  (unbounded); `METRIC_LABELS` maps them to display names. Absent counters read
  `None`, never 0. The Quasar l1_client CSR has no fixed rows: `quasar_l1_client_label(sel)`
  and `quasar_l1_client_selection_is_valid(sel)` name and validate a `subport*8 + event`
  selection, `compute_l1_client_metrics(view, names)` produces its per-run metric and
  `metric_label(key)` resolves both the static and the dynamic keys.

Consumers: `tools/tracy/perf_counter_analysis.py` (metal profiler) and
`tests/python_tests/helpers/{counters,metrics}.py` (LLK perf harness). Metal's
`setup.py` installs the package; the LLK pytest plugin puts `tools/python` on
`sys.path`. Adding a counter needs no Python change: the names are parsed from
the headers. Adding a metric means one formula plus one `METRIC_LABELS` entry.
