# Benchmark: google/gemma-4-31B-it

Status: failed. Accuracy concurrency: unrecorded.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Run details

Subset: `3dd6b5fe3878c8138817180626bf5f551ae6a966d11777d3eac935db695fcc38`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|

Error: RuntimeError: command exited 1; inspect /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/benchmark/run/roofline-b32-check.log

Final status: failed. RuntimeError: command exited 1; inspect /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/benchmark/run/roofline-b32-check.log

Total client-stage wall time: 226.4 seconds.
