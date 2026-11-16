# Ornith-1.5-9B TTNN autoport

Optimized full-model performance on four P300c Blackhole chips (TP4, mesh1×4):

| Warmed batch1, 128 input / 128 output | Before | After |
|---|---:|---:|
| TTFT | 90.017 ms | 89.839 ms |
| Resident traced token-out, no per-token host transfers | 112.477 t/s/user | 114.842 t/s/user |
| Traced token-out with caller readback | 112.029 t/s/user | 114.599 t/s/user |
| Traced teacher forcing with token uploads/readback | 111.582 t/s/user | 113.956 t/s/user |

Context2048, three warmed serial repetitions for TTFT and token-out. Resident
latency is8.708ms/token. No material TTFT gain is established; an earlier run
measured92.049ms and remains in the evidence. The main gain is full-model B32
with31 active fixed slots: **18.017→11.473ms/token (36.3% lower latency)**.
See the [optimized full-model report](doc/optimized_full_model/README.md) for
controls, exact regimes, profiler tables and accuracy/capacity evidence.

The full 32-layer model preserves the optimized multichip decoder policy and
supports a **262,144-token context**, including non-aligned prompts. Batch 32 is
validated separately at context 512. Global cache-pool sizing constrains aggregate
occupancy; maximum context for all 32 slots simultaneously does not fit DRAM.

| Readiness gate | Top-1 | Top-5 | Top-100 |
|---|---:|---:|---:|
| AIME24 chat prefill, 100 reference tokens | 99% | 100% | 100% |
| AIME24 traced teacher-forced decode, 100 tokens | 98% | 100% | 100% |

`tt/model.py` implements the HF text stack; `tt/generator.py` provides standard
Metal readiness generation and explicit serving prefill/decode, caller-owned
caches, page tables, fixed slots, inactive rows and async outputs. Greedy token
feedback and position advance remain on device. Explicit host-logits/host-greedy
compatibility is separate from the measured path. The full-model saved-boundary
resumed-prefill check passes; reduced serving forced-preemption trajectories
still differ from uninterrupted generation. Scheduler prefix caching retains
the documented recurrent-state exclusion.

The completed full-model stage has an independent round-1
[clean-pass review](doc/full_model/stage_review_round1.md). Optimized-full-model
received an independent round2 [clean-pass](doc/optimized_full_model/stage_review_round2.md);
local checkpoint details are in its [work log](doc/optimized_full_model/work_log.md).
The vLLM adapter and full32-layer B32 serving path complete Stage9 with an
independent round2 [clean-pass](doc/vllm_integration/stage_review_round2.md).
Primary B1 serving measures236.249ms TTFT and114.941tokens/s/user
(128 input/128 output, one request, concurrency1, TP4/context262144).
The earlier tracked server passed capacity acceptance with262143 input tokens
and1 output token, totaling262144; this predates the prompt-horizon and
intermediate-logit correction and does not establish32 simultaneous maximum-length
requests. See the [serving report](doc/vllm_integration/README.md) for current
evidence and limits, and the [full-model report](doc/full_model/README.md),
[work log](doc/full_model/work_log.md), [serving contract](doc/full_model/serving_contract.json),
and [context contract](doc/context_contract.json) for commands, evidence and limits.

Exact logs, generated text and rendered prompt files are kept as lossless gzip
archives to preserve whitespace. Raw local copies remain available;
[`archive_manifest.json`](doc/full_model/archive_manifest.json) records their SHA256 hashes and restoration command.
