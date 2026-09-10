# moonshotai/Kimi-Linear-48B-A3B-Instruct on Tenstorrent Blackhole (p300x2) — bring-up report

generated 2026-09-10T23:53:51Z

## Verified

- p300x2 (1x4): teacher-forced accuracy on the AIME24 reference (100 tokens, traced decode): {"top1": 0.96, "top5": 1.0, "top100": 1.0, "matches_top1": 96.0, "matches_top5": 100.0, "matches_top100": 100.0}
- p300x2 (1x4): whole-model golden test passed (2 cases; top-1 0.960 top-5 1.000 top-100 1.000; eager decode 76 ms/token)
- p300x2 (1x4): greedy chat fixtures answered (eager decode 111 ms/token, prefill 1.24 s)
- p300x2 (1x4): context validated to 131072 tokens
- p300 (1x2 submesh): teacher-forced accuracy on the AIME24 reference (100 tokens, traced decode): {"top1": 0.94, "top5": 1.0, "top100": 1.0, "matches_top1": 94.0, "matches_top5": 100.0, "matches_top100": 100.0}
- p300 (1x2 submesh): whole-model golden test passed (2 cases; top-1 0.960 top-5 1.000 top-100 1.000; eager decode 88 ms/token)
- p300 (1x2 submesh): greedy chat fixtures answered (eager decode 114 ms/token, prefill 1.67 s)
- kda module tests on 1x1: 7 passed (min PCC 0.9998)
- moe module tests on 1x1: 6 passed (min PCC 0.9949)
- mla module tests on 1x1: 6 passed (min PCC 0.9995)
- decoder module tests on 1x1: 12 passed (min PCC 0.9969)
- kda module tests on 1x4: 7 passed (min PCC 0.9998)
- moe module tests on 1x4: 6 passed (min PCC 0.9963)
- mla module tests on 1x4: 6 passed (min PCC 0.9995)
- decoder module tests on 1x4: 12 passed (min PCC 0.9988)
- datatype sweep selected: {"name": "bfp8_kvbfp8", "experts": "bfp8", "kv": "bfp8", "top1": 0.96, "top5": 1.0, "top100": 1.0, "decode_ms": 112.2}
- host vLLM p300x2: OpenAI API proof passed (identity, greedy, determinism, kimi_k2 tool call + round trip, non-aligned prompt, advisory findings)
- container profile `p300x2`: tt-model serve / API proof / stop passed with advisory findings

## Not verified

- container profile `p300`: not verifiable on this box: the p300 profile opens a direct (1,2) mesh, which needs a physical P300 host. On this QuietBox 2 a direct 2-chip mesh cannot initialise fabric; TP2 was validated as a (1,2) submesh of the 1x4 parent (stage 07).
- TTI release workflow (meta_ifeval / meta_gpqa_cot): not run (optional stage 14)

## Advisory findings and known limitations

- host vLLM readiness runner (sampling/qualitative/benchmark) exit 1
- KDA prefill runs the chunked fp32 WY recurrence (tt/kda/chunked_prefill.py, gate clamp -2.5); kernel-vs-exact PCC on real activations at 1024 tokens: outputs 0.9997-0.9999, carried state 0.99993-0.99999 (layers 0/2/8/17/25). The reused ttnn.experimental.kda kernel is kept selectable but drifts (bf16 gate chain) and overflows on layer 25.
- Decode is eager in the bare-metal generator paths unless traced (vLLM traces decode); on-device sampling not enabled (host sampling).

## User-facing latency (host vLLM, p300x2)

| test | value |
|---|---|
| short_chat_c1_128tok | {"ttft_ms": 586.2, "total_s": 15.04, "completion_tokens": 92, "prompt_tokens": 22, "decode_tps": 6.29} |
| short_answer_c1 | {"ttft_ms": 572.4, "total_s": 0.73, "completion_tokens": 2, "prompt_tokens": 19, "decode_tps": 6.55} |
| prompt_1024_ttft | {"ttft_ms": 3411.0, "total_s": 5.06, "completion_tokens": 11, "prompt_tokens": 1038, "decode_tps": 6.06} |
| prompt_8192_ttft | {"ttft_ms": 18436.5, "total_s": 20.91, "completion_tokens": 16, "prompt_tokens": 8208, "decode_tps": 6.06} |
| concurrency_8_100out | {"concurrency": 8, "ok": 8, "wall_s": 37.06, "completion_tokens": 800, "aggregate_tps": 21.59, "per_user_tps": 2.7, "latency_p50_s": 37.05, "latency_max_s": 37.06} |
| concurrency_32_100out | {"concurrency": 32, "ok": 32, "wall_s": 60.56, "completion_tokens": 3200, "aggregate_tps": 52.84, "per_user_tps": 1.65, "latency_p50_s": 60.55, "latency_max_s": 60.55} |
| tool_call_latency | {"status": 200, "seconds": 7.55, "finish_reason": "tool_calls", "tool_calls": 1, "completion_tokens": 39} |

## Optimisation rounds

| round | teacher forcing | decode | prefill |
|---|---|---|---|
| round_20260910T200524Z | {"top1": 0.95, "top5": 1.0, "top100": 1.0, "matches_top1": 95.0, "matches_top5": 100.0, "matches_top100": 100.0} | 110.5 ms/tok | 1.46 s |
| round_20260910T224405Z | {"top1": 0.94, "top5": 1.0, "top100": 1.0, "matches_top1": 94.0, "matches_top5": 100.0, "matches_top100": 100.0} | 110.6 ms/tok | 0.65 s |

## Datatype sweep

```
[
 {
  "name": "bfp4_kvbfp8",
  "experts": "bfp4",
  "kv": "bfp8",
  "top1": 0.94,
  "top5": 1.0,
  "top100": 1.0,
  "decode_ms": 110.1,
  "prefill_s": 1.01,
  "degenerate": false,
  "paris": true,
  "meets_bars": true
 },
 {
  "name": "bfp8_kvbfp8",
  "experts": "bfp8",
  "kv": "bfp8",
  "top1": 0.96,
  "top5": 1.0,
  "top100": 1.0,
  "decode_ms": 112.2,
  "prefill_s": 0.68,
  "degenerate": false,
  "paris": true,
  "meets_bars": true
 }
]
```

## Published

- repo: https://huggingface.co/jashansinghTT/kimi-linear-48b-a3b-instruct-blackhole (public: yes; catalog tag: yes)
- clean-pull proof on p300x2: PASSED (`tt-model serve --profile p300x2 jashansinghTT/kimi-linear-48b-a3b-instruct-blackhole`; prove exit 1)
