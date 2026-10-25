# Benchmark: Qwen/Qwen3.8-27B

Status: failed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| gpqa_diamond_cot_zeroshot | 10 / 198 | exact_match,strict-match | 0.00 | — | — | Unavailable |
| gpqa_diamond_cot_zeroshot | 10 / 198 | exact_match,flexible-extract | 50.00 | 89.20 | -39.20 | [reference](https://huggingface.co/Qwen/Qwen3.8-27B) |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

Reference protocol notes:
- gpqa_diamond_cot_zeroshot, exact_match,flexible-extract: Published exact-model GPQA Diamond score; local10/198 zero-shot lm-eval variant and output cap may differ. Published complete evaluation recipe not established from model card.

## Run details

Subset: `fb01e42fbed84a24ad4f7c57b6355afa2abbd27e7cc3a27418e20b861fd6b5d7`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [gpqa_diamond_cot_zeroshot](gpqa_diamond_cot_zeroshot/results.json) | 10 | 4 (40.0%) | 4 | 2226.0 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|

Error: ValueError: performance input_lens: expected 32 requests of 4096 tokens

Final status: failed. ValueError: performance input_lens: expected 32 requests of 4096 tokens

Total client-stage wall time: 2598.9 seconds.
