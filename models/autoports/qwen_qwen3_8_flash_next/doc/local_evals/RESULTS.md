# Qwen3.8-Flash-Next on QB2 M1 (4x Blackhole P300x2): local benchmark + evals

**Status: IN PROGRESS** (tables filled as results land; see STATUS.md for timeline).

Model `Qwen/Qwen3.8-Flash-Next` @ `812d96e10a0dd82b69d749e68693698cf6c36daa`, bring-up port
`models/autoports/qwen_qwen3_8_flash_next`, selected precision `combined_shared4_mixed`
(`doc/datatype_sweep/selected_precision_config.json`).

## Server (one server for everything)

Bring-up final serving command `doc/optimized_vllm/run_server.sh`, unchanged except `timeout 86400` -> `604800`
(copy: `evals/server/run_server.sh`, launched as `run_server.sh evals/server/run1`):
resident `--max-num-seqs 2`, `--max-model-len 262144`, block 32, `--max-num-batched-tokens 9`,
prefix caching off (recurrent-state exclusion), async scheduling, device sampling, trace mode all,
FABRIC_1D_RING, prefill alignment 8, decode interleave 1 prefill : 2 decode, reasoning parser qwen3,
tool parser qwen3_xml. Served model id = snapshot path (no `--served-model-name`).

## 1. Benchmark sweep

Client: stage-11 exact-token wrapper `tools/benchmark_client.py` over upstream `vllm bench serve`
(random dataset, exact ISL, `--ignore-eos`, temperature 0, `--request-rate inf`, `--max-concurrency C`,
num_prompts = C, except c1: 4 prompts at 128, 2 at 1024/4096, 1 at 32768). Script: `evals/bench/run_sweep.sh`;
raw JSON in `evals/bench/results/`; table from `evals/bench/summarize.py`.

Important: the server has **2 resident slots**; concurrency 8/16/32 queue behind them, so aggregate
throughput is flat and TTFT at high concurrency is dominated by queueing. tok/s/user = 1000 / mean TPOT.

<!-- BENCH TABLE (regenerate: python3 bench/summarize.py) -->
| ISL/OSL | conc | reqs (ok/fail) | TTFT mean s | TTFT p50 s | TTFT p99 s | TPOT mean ms | ITL p50 ms | ITL p99 ms | tok/s/user (1000/TPOT) | output tok/s total | total (in+out) tok/s | wall s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 4 (4/0) | 7.4 | 7.4 | 7.5 | 103.1 | 103.6 | 112.6 | 9.70 | 6.23 | 12.47 | 82 |
| 128/128 | 8 | 8 (8/0) | 74.7 | 74.8 | 141.1 | 230.5 | 202.2 | 571.4 | 4.34 | 6.23 | 12.46 | 164 |
| 128/128 | 16 | 16 (16/0) | 156.4 | 156.3 | 303.8 | 234.0 | 202.1 | 567.8 | 4.27 | 6.24 | 12.49 | 328 |
| 128/128 | 32 | 32 (32/0) | 320.5 | 320.5 | 630.3 | 236.0 | 202.1 | 567.2 | 4.24 | 6.24 | 12.48 | 656 |
| 1024/1024 | 1 | 2 (2/0) | 59.5 | 59.5 | 59.5 | 105.3 | 105.3 | 106.6 | 9.50 | 6.13 | 12.25 | 334 |
| 1024/1024 | 8 | 8 (8/0) | 604.6 | 604.4 | 1143.8 | 232.8 | 205.2 | 573.5 | 4.30 | 6.14 | 12.28 | 1335 |
| 1024/1024 | 16 | 16 (16/0) | 1273.3 | 1273.1 | 2473.9 | 236.6 | 205.2 | 574.1 | 4.23 | 6.13 | 12.27 | 2671 |
| 1024/1024 | 32 | 32 (32/0) | 2611.1 | 2611.2 | 5135.4 | 238.6 | 205.4 | 574.3 | 4.19 | 6.13 | 12.26 | 5348 |
| 4096/512 | 1 | 2 (2/0) | 239.5 | 239.5 | 239.8 | 108.0 | 108.0 | 109.4 | 9.26 | 1.74 | 15.64 | 589 |
| 4096/512 | 8 | 8 (8/0) | 1285.9 | 1285.8 | 2311.0 | 318.6 | 123.2 | 579.3 | 3.14 | 1.72 | 15.44 | 2387 |
| 4096/512 | 16 | 16 (16/0) | 2478.6 | 2478.4 | 4674.4 | 333.4 | 123.4 | 579.2 | 3.00 | 1.72 | 15.44 | 4774 |
| 4096/512 | 32 | 32 (32/0) | 4866.5 | 4866.0 | 9401.4 | 340.8 | 124.0 | 578.8 | 2.93 | 1.72 | 15.44 | 9549 |
| 32768/512 | 1 | 1 (1/0) | 1945.1 | 1945.1 | 1945.1 | 114.7 | 114.8 | 116.3 | 8.72 | 0.26 | 16.61 | 2004 |
| 32768/512 | 8 | 8 (8/0) | 8995.6 | 8994.4 | 15901.3 | 326.4 | 135.3 | 585.4 | 3.06 | 0.25 | 16.54 | 16101 |
<!-- /BENCH TABLE -->

Rows 4096/512 and 32768/512 ran alone on the server (owner decision, 2026-11-18: one workload at a time). 128/128 and 1024/1024 also ran alone. All 22 rows: 0 failed requests, 0 vLLM preemptions.

**Skipped: 32768/512 at c16 and c32.** The context contract can serve them, but the server has only 2 resident slots, so those rows would measure only queueing (TPOT and tok/s per user are the same as c8; TTFT just grows with queue depth). At ~17 tok/s prefill they would take about 26 h. Skipped by owner decision (Vladan, 2026-11-18).

## 2. Evals

| Eval | Ours | Model card | Notes |
|---|---:|---:|---|
| GPQA Diamond (198 q, thinking, card sampling) | pending | 91.7 | |
| SWE-bench Pro (5 tasks, leak-free, mini-swe-agent) | pending | 62.5 (informational) | card uses Claude Code harness on a refined task set; not comparable |

### GPQA Diamond
- Client `evals/gpqa/run_gpqa.py --out evals/gpqa/run1 --seed 0`; scorer `evals/gpqa/score_gpqa.py`.
- Prompt: OpenAI simple-evals zero-shot CoT template ("ANSWER: $LETTER"), choices shuffled per question (seed 0).
- Sampling (card, thinking mode): temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0,
  repetition_penalty 1.0; `enable_thinking: true`, template default `reasoning_effort` = xhigh.
- max_tokens = 262144 - prompt tokens (largest the context contract allows).
- Extraction: last `ANSWER: X` in content, then `\boxed{X}`, then "answer is X"; falls back to reasoning text
  if content blank.

### SWE-bench Pro (leak-free)
- Owner branch `vkovacevic/qwen38-flash-next-qb2-swepro-clean` cloned to `evals/tt-inference-server`,
  local branch `qwen38-flash-local-evals` (local commits listed in PIPELINE-NOTES.md).
