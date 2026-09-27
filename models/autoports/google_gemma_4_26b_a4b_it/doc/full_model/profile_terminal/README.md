# Reduced terminal profile

This captures one real sliding-attention layer (0), one global-attention layer
(5), and the real embedding, final norm, vocabulary-sharded LM head and common
split-greedy sampler on TP4. Prefill is4096 tokens; the signposted window is one
warmed token-out decode replay. It is not a full-model performance result.

`summary.json` derives each device's complete firmware window from first start
to last end, including gaps. The longest window is3078.76us. The sampling trace
is at most512.37us (16.6% of that reduced window), so it does not dominate even
this two-layer path. The slowest-device-per-op report attributes944.789us to the
BF16/HiFi4 vocabulary-sharded LM-head matmul and279.04us to local top-k. Those
are individual op durations, not whole-layer denominators. The normal common
split greedy path remains selected over the measured slower force-argmax path.
No full-vocabulary all-gather or host argmax occurs in this default path.

Reports: `decode_perf_report.csv`, `decode_perf_report_stacked.csv`,
`decode_perf_report_stacked.png`, `summary.json`. Raw captures remain local in
`reports/` and `.logs/` and are ignored by git. No reduced result is copied to
full-model telemetry device-time or roofline fields.

Command: `python_env/bin/tt-perf-report
models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/profile_terminal/reports/terminal_two_layers/2026_09_27_07_41_22/ops_perf_results_terminal_two_layers_2026_09_27_07_41_22.csv
--start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --tracing-mode
--active-experts 8 --csv
models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/profile_terminal/decode_perf_report.csv`.
The tool warns about unclassified op names, but retains their durations in the
CSV. The profile's generic allocation warning is covered by the separately
passing trace-allocation-tracked tests; Watcher was disabled for profiling.
