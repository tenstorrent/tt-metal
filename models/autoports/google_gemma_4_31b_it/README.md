# Gemma 4 31B text autoport

Full-model TP4 token-out autoregressive generation on four local Blackhole p300c ASICs, **4096 input / 128 generated tokens**. Batch 1 is first; values are medians of three warmed requests with diagnostics disabled. Decode includes model and sampling traces, device token feedback and caller token readback.

| Batch / concurrent requests | TTFT (ms) | Decode tokens/s/user | Decode ms/token | Evidence |
|---:|---:|---:|---:|---|
| 1 | 659.37 | 34.22 | 29.23 | [measurement](doc/full_model/performance_b1.json) |
| 32 (agent target) | 20677.09 | 23.18 | 43.14 | [measurement](doc/full_model/performance_b32.json) |
| 16 | 10327.03 | 29.13 | 34.33 | [measurement](doc/full_model/performance_b16.json) |
| 8 | 5162.31 | 32.31 | 30.95 | [measurement](doc/full_model/performance_b8.json) |

The separate AIME24 teacher-forcing workload is 161 input / 100 output tokens: **94.56 ms TTFT and 37.38 tokens/s/user**. It includes on-device sampling and explicit reference-token feedback; it is not logits-only timing or the headline workload.

Prefill top-1/5/100 agreement is **94% / 100% / 100%**; decode is **93% / 100% / 100%**, over all 100 positions of one pinned AIME24 chat reference. These are token-agreement results, not a contest score.

The full stack preserves **262144-token context**, including non-aligned prompts. The target pool holds 32 concurrent 32768-token working sets. Full-model and allocation-tracked reduced checks cover batches 1/8/16/32, mixed prompts, inactive slots, remapped pages, deterministic logits and on-device split sampling.

HF and tokenizer revision: `842da3794eaa0b77d5f08bae87a17459d91ff475`. This new text-only implementation preserves this run's optimized multichip decoder policy and layout. Stochastic top-k supports 1–32 candidates; explicit host sampling is a separate compatibility mode.

[Full-model report](doc/full_model/README.md), [qualitative review](doc/full_model/qualitative_verdict.md), [independent review](doc/full_model/stage_review.md), [work log and commits](doc/full_model/work_log.md). No vLLM integration is included in this stage.
