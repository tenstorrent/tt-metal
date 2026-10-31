# Reduced terminal attribution

The final reduced profile uses real layers 0 (sliding) and 5 (full attention), the real TP4 embedding, norm, vocabulary head, cache shapes and canonical split sampler. It executes 4096 input tokens and 128 output tokens at batch 1. It is not a full-model device measurement.

`terminal_attribution.json` records complete per-device FW windows, including gaps, and takes the maximum across the four devices. Raw CSV row order is used between signposts because trace replay rows retain capture-time host metadata. The FW cycle/duration ratio is 1.35 cycles/ns.

| Scope | Complete device window (microseconds) |
|---|---:|
| Terminal norm, head and softcap | 1255.196 |
| Reduced model decode, excluding sampler | 2462.830 |
| Canonical greedy sampler | 519.469 |
| Combined reduced token-out decode | 3001.274 |

The sampler occupies about 17.3% of this deliberately reduced two-layer decode window, and is far smaller than the approximately 29 ms inherited 60-layer stack reference. It does not dominate full-model token-out work. The head matmul is about 1092 microseconds in the combined report; it reads the local BFP8 vocabulary shard at approximately 327 GB/s. The common candidate sampler moves selected candidates across chips, not the complete vocabulary logits. Its largest top-k work and candidate gathers are retained in `perf_sampler_only.txt`. The semantic-greedy force-argmax alternative was already measured at roughly 2.9 ms and rejected in `sampling_choice.md`.

`tt-perf-report` completed successfully with advice enabled for all five windows: `perf_prefill.txt`, `perf_decode.txt`, `perf_model_only.txt`, `perf_sampler_only.txt`, and `perf_terminal.txt`. Generic advice flags the head as low arithmetic utilization, but its 32-row physical tile represents one logical user and is bandwidth dominated. Generic advice also recommends tracing for a sampler gap; these windows already replay traces, so that recommendation does not identify missing host tracing. Host orchestration/readback boundaries remain visible in the separate clean full-model end-to-end measurements and counters.

The prefill whole window includes intentional profiler drains after each chunk, so its 1.533-second span is diagnostic only. It is not a clean prefill latency or a full-model TTFT. No reduced measurement is copied into full-model telemetry device-time or roofline fields.

Collection command is in `tests/profile_full_terminal.py` and `work_log.md`. Source CSV: `tracy_reduced/reports/2026_10_31_03_45_33/ops_perf_results_2026_10_31_03_45_33.csv`. Raw traces and tables remain local and are excluded from publication commits and telemetry.

The selected stage-05 layer profiles give a nominal B1 stack reference of 50 × 475.2245 + 10 × 527.8178 = 29.0394 ms. Clean full-model token-out measures 29.2267 ms. These profiling regimes differ: adding isolated terminal and sampler windows to the profiled layer sum would exceed the unprofiled full result. The sum is an indicative comparison, not an additive decomposition or strict lower bound. Do not infer a 0.1873 ms total wrapper cost from their difference. `layer_stack_comparison.json` records the same calculation for every configured batch.
