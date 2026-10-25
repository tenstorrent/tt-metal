# Benchmark: Qwen/Qwen3.8-27B

Status: failed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Run details

Subset: `fb01e42fbed84a24ad4f7c57b6355afa2abbd27e7cc3a27418e20b861fd6b5d7`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|

Error: InterruptedError: benchmark interrupted by signal 15

Final status: failed. InterruptedError: benchmark interrupted by signal 15

Total client-stage wall time: 940.0 seconds.
