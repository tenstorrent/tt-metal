# Benchmark: google/gemma-4-26B-A4B-it

Status: failed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for both serving profiles; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Run details

Subset: `ef36dff6d479319adc5b0438e79acc4e139bd108300e209479ab6ee864c4c994`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|

Error: RuntimeError: command exited -15; inspect /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/benchmark/run/accuracy-shared.log

Final status: failed. RuntimeError: command exited -15; inspect /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/benchmark/run/accuracy-shared.log

Total client-stage wall time: 942.6 seconds.
