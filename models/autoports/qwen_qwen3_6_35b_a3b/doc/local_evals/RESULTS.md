# Qwen3.6-35B-A3B on M2 (4x Blackhole, P300x2, 1x4 mesh): local benchmark and eval results

Model `Qwen/Qwen3.6-35B-A3B` @ `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, tt-metal autoport
`models/autoports/qwen_qwen3_6_35b_a3b`, precision `baseline_bfp8` (the pipeline's selected config).
Everything ran on this one machine against one server, strictly one workload at a time.

## Server (the pipeline's final serving command)

```bash
# evals/scripts/start_server.sh  (tmux session qwen-server)
cd ~/qwen36-35b-a3b/tt-metal
QWEN_VLLM_AUDIT=0 QWEN_VLLM_REDUCED=0 QWEN_VLLM_ASYNC=1 TT_METAL_TRACE_ALLOC_TRACKING=0 \
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 PYTEST_ADDOPTS='-p vllm' QWEN_VLLM_SERVER_LIFETIME_SECONDS=1209600 \
bash models/autoports/qwen_qwen3_6_35b_a3b/tests/run_vllm_server.sh <outdir> 32 2048 --stages serve
```
This is vLLM 0.31 with the TT plugin: max-num-seqs 32, max-model-len 262144, block size 32,
max-num-batched-tokens 2048, async scheduling, decode-only traces, on-device sampling mode `all`,
`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`, and no prefix caching
(the model does not support it). The KV pool is 8224 pages x 32 = 263,168 tokens. The server takes about 30 minutes
to boot. The only change from the pipeline's command is a longer server lifetime (14 days instead of 8 hours).

## 1. Benchmark sweep

Client: the same one stage 11 used (`vllm bench serve` v0.26.0 from `benchmark-vllm-env`): random dataset,
`/v1/completions`, `--ignore-eos --temperature 0 --seed 4108 --request-rate inf`. Each concurrency level runs
one wave (num_prompts = concurrency; c=1 uses 4/2/2/1 prompts). Everything ran on the one 32-slot server.
Script: `evals/scripts/run_sweep.sh`; raw JSON and logs: `evals/bench/results/`.

SWEEP_TABLE

The tok/s/user column is 1000 / mean TPOT, and mean TPOT includes the decode stalls caused by other users' prefills.
The output tok/s total column is total generated tokens / benchmark wall time.

What the numbers show:
- **Single user decode is about 31-33 ms/token (about 31 tok/s) at every context length**, up to 32K tokens (32.7 ms).
  The stage-11 figure of about 46 tok/s came from a 1-slot server (21.8 ms); the 32-slot server always runs at about 31 ms.
- **Prefill is serial and slow, about 22 ms per input token**: 3 s for 128 tokens, 23 s for 1K, 91 s for 4K and 729 s for 32K.
  Requests are prefilled one after another, so TTFT at concurrency N is roughly N times the single-request prefill.
- **With 5 or more users, each decode step takes about 410-430 ms.** The adapter pads any batch of 5-32 rows to its fixed
  32-row trace. So 8 users get only about 2.4 tok/s each and less total throughput than 1 user. Throughput reaches its
  best, about 75 tok/s, only near 32 users (stage 10 measured 72.8 tok/s at 32x6000).
- With long prompts at 16 or 32 users, a running decode stalls for the length of each new prefill: ITL p99 is about
  92 s, which is one 4K prefill. That is why the 4096/512 c16/c32 TPOT means are 1.8-3.2 s.
- **32768/512 ran at c=1 only.** At c=8, 8 x (32768+512) = 266,240 tokens is more than the 263,168-token KV pool,
  so the server would have to preempt. Serial prefill also means c=8/16/32 would spend 1.6/3.2/6.5 hours mostly
  waiting in the prefill queue (8/16/32 x 729 s), which is not a meaningful throughput measurement.

## 2-4. Evals

EVAL_TABLE

## Sampling settings

SAMPLING

## Notes

NOTES
