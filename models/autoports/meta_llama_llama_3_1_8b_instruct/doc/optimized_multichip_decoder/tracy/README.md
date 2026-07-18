# Pre-review topology Tracy evidence

This initial capture established the collective rewrite at batch 1 and logical sequence
length 18.  It contains three warmed prefill iterations and one traced decode
replay; `_trace_latency` also performs one eager warm and one capture forward,
so each decode summary contains three complete decoder executions.

The exact capture command and report-generation procedure are recorded in
`../../work_log.md`.  The final raw ops CSV is retained losslessly as:

`reports/2026_07_18_13_09_56/ops_perf_results_2026_07_18_13_09_56.csv.gz`

The uncompressed SHA-256 is
`1d2abf8898d1f0f9ced308c327364e8bd3a408c877b21919a3d2c5cc061768bb`.

`analysis/` contains advice-enabled filtered CSVs, category summaries, console
provenance, and complete human-readable tables for:

- `single_prefill`: `PERF_SINGLE_PREFILL` to `PERF_SINGLE_PREFILL_END`;
- `single_decode`: `PERF_SINGLE_DECODE` to `PERF_SINGLE_DECODE_END`;
- `multi_prefill`: `PERF_MULTI_PREFILL` to `PERF_MULTI_PREFILL_END`;
- `multi_decode`: `PERF_MULTI_DECODE` to `PERF_MULTI_DECODE_END`.

The authoritative final replay-only profile after reviewer-driven QKV tuning is
in `../tracy_replay_final/`.  This older capture remains useful for pre/post
topology comparison but is not used for final device/E2E accounting.

The reviewable authority is the CSV/table output.  Transient Tracy host traces,
raw 50+ MB device logs, generated summary PNGs, and the earlier superseded
capture were pruned after the final reports were verified.
