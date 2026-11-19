# Benchmark: Qwen/Qwen3.6-35B-A3B

Status: failed. Accuracy concurrency: 32 (largest measured batch; server prefix caching: unrecorded).

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Each batch profile sets server max-num-seqs and request concurrency to the batch size. The two one-user rows run on the largest server: the FUNCTIONAL 128/128 reference and the long-context cost a lone user pays on that server.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every batch profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

- Sample reduction, mmlu_pro: 280 → 14. Four native-thinking pilot requests completed in 485.635s on the 32-slot server at concurrency4; individual elapsed294.670–485.633s and outputs1916–3908 tokens. Serial Stage10 prefill is about22ms/input token. Reduce the common frozen ranking before changing thinking, cap or tasks; retain one MMLU-Pro question per subject and18 IFEval questions to limit accuracy time within the7200s client bound. Fixed performance repetitions remain a separate documented budget conflict.
- Sample reduction, ifeval: 256 → 18. Four native-thinking pilot requests completed in 485.635s on the 32-slot server at concurrency4; individual elapsed294.670–485.633s and outputs1916–3908 tokens. Serial Stage10 prefill is about22ms/input token. Reduce the common frozen ranking before changing thinking, cap or tasks; retain one MMLU-Pro question per subject and18 IFEval questions to limit accuracy time within the7200s client bound. Fixed performance repetitions remain a separate documented budget conflict.

## Run details

Subset: `204ea844a375ed212ff836af30565a151c8a55ac32e2ddd239607a52d4ead3e5`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|

| Profile | Completed / requested | Wall seconds | Requests/s |
|---|---:|---:|---:|

| Profile | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|

Error: ValueError: server launch command does not match recorded capacity

Final status: failed. ValueError: server launch command does not match recorded capacity

Client-clock wall time: 0.0 of 7200 seconds. Server setup (launch, compile, warmup): 0.1 of 10800 seconds. Total: 0.1 seconds.
