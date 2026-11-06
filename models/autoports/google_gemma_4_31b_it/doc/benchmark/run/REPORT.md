# Gemma 4 31B final benchmark attempt — incomplete

| Benchmark | Completed / population | Measured | Published | Delta | Coverage |
|---|---:|---:|---:|---:|---|
| GPQA Diamond, upstream zero-shot CoT | 0 / 198 (planned frozen 128) | — | [84.3%](https://ai.google.dev/gemma/docs/core/model_card_4) | — | Gated dataset unavailable; accounting readiness failed before accuracy |
| SWE-bench Verified | 0 / 500 | — | Not established | — | Tool-use harness required in Stage 12 |
| Terminal-Bench 2.1 | 0 / 89 | — | Not established | — | Terminal harness required in Stage 12; 2.0 figures not substituted |

| Decode profile | Concurrent requests | Server slots | Complete requests | TTFT mean ms | TPOT mean ms | Decode tok/s/user (1000/TPOT) | Output tok/s aggregate | Prefill FLOP % | Decode DRAM % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1, 4096/128 | 1 | 1 | 8 / 8 | 753.499 | 28.306 | 35.328 | 29.434 | unavailable | unavailable |
| B8, 4096/128 | 8 | 8 | 24 / 24 | 2402.053 | 61.438 | 16.277 | 99.903 | unavailable | unavailable |
| B16, 4096/128 | 16 | 16 | 48 / 48 | 3719.953 | 111.105 | 9.001 | 114.002 | unavailable | unavailable |
| B32, 4096/128 | 32 | 32 | 96 / 96 | 6551.272 | 231.336 | 4.323 | 112.641 | unavailable | unavailable |

**Status: incomplete.** These are latency-only measurements retained after the required runner failed its roofline readiness check. No accuracy verdict or threshold is applied. No host time is represented as device time.

Target workload: 32 concurrent multi-turn agents, 32768 live tokens per agent, typical context 4096 (Stage 10 estimate pending Stage 12). These profiles measure distinct single-turn 4096-token prompts and 128 generated tokens. They do not measure long-history reuse, tools, agent waiting, or growing-context working sets.

Elapsed from the original benchmark_stage invocation through this report: 1392.49s of 7200s. All profile launches, reloads, warmups, client verification and reporting count. Initial supplied server: none; Stage 10 had stopped it. No server startup is excluded from the timed interval. Setup before invocation includes client installs, source inspection, manifest download attempt and host checks; see RUN_NOTES.md.

## Measurement details

Upstream vLLM 0.26.0 empty-target benchmark client; serial use of all 4 Blackhole ASICs (P300x2, TP4 mesh1x4). Same 60-layer autoport, head_bfp4 precision policy and 262144-token context in each profile. Prefix caching enabled. Request concurrency equals server slots. Greedy temperature 0, ignore_eos=true; requests must each report4096 actual input and 128 output tokens. Seeds differ between warmups and measurements; random prompts are distinct. The client initial probe is excluded by its upstream result accounting. No live device profiler, extra synchronization, or plugin runtime patch.

Each profile retains the actual process launch argv, API model capacity, imported module path, layer list, precision and source hashes in configuration-bN.json. Source commit and startup duration are in perf-bN-server.json. Model/tokenizer revision:842da3794eaa0b77d5f08bae87a17459d91ff475. Server commands preserve operator top_k 32, checkpoint temperature 1/top_p 0.95, async scheduling and chunked prefill. The server wheel identifies as vLLM 0.31.0+empty; its source commit is unavailable in installed package metadata and remains null.

| Batch | Metric | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | ttft | 753.499 | 753.060 | 756.720 | 757.806 |
| 1 | tpot | 28.306 | 28.308 | 28.312 | 28.313 |
| 1 | itl | 28.306 | 28.313 | 28.899 | 33.216 |
| 1 | e2el | 4348.376 | 4348.196 | 4350.130 | 4350.279 |

B1: measured wall34.790s; request throughput0.230/s; input tokens32768; output tokens1024. [Raw result](partial-performance/perf-b1.json).

| Batch | Metric | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 8 | ttft | 2402.053 | 2010.807 | 5440.924 | 6174.466 |
| 8 | tpot | 61.438 | 64.437 | 73.277 | 73.588 |
| 8 | itl | 61.438 | 30.127 | 54.851 | 799.778 |
| 8 | e2el | 10204.696 | 10212.027 | 13599.812 | 14334.909 |

B8: measured wall30.750s; request throughput0.780/s; input tokens98304; output tokens3072. [Raw result](partial-performance/perf-b8.json).

| Batch | Metric | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 16 | ttft | 3719.953 | 2120.238 | 11440.892 | 13044.297 |
| 16 | tpot | 111.105 | 122.710 | 130.149 | 132.332 |
| 16 | itl | 111.105 | 39.302 | 839.151 | 848.000 |
| 16 | e2el | 17830.227 | 17701.641 | 27076.946 | 28655.566 |

B16: measured wall53.894s; request throughput0.891/s; input tokens196608; output tokens6144. [Raw result](partial-performance/perf-b16.json).

| Batch | Metric | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 32 | ttft | 6551.272 | 2331.859 | 24968.556 | 28480.319 |
| 32 | tpot | 231.336 | 261.448 | 264.823 | 271.373 |
| 32 | itl | 231.336 | 60.917 | 924.696 | 934.165 |
| 32 | e2el | 35930.932 | 35534.298 | 58281.108 | 61788.087 |

B32: measured wall109.090s; request throughput0.880/s; input tokens393216; output tokens12288. [Raw result](partial-performance/perf-b32.json).

## Missing phase accounting

Whole-server sampling-mode logs (including warmups) count host-compatible prefill steps B32=128, B16=64, B8=32 and B1=9; all observed decode steps use device sampling. These are lifetime counts, not an isolated measured-window attribution. See [sampling_modes.json](sampling_modes.json). No full-phase percentages can be reported. AutoDebug and an independent AutoFix hypothesis check found no lossless stock-plugin export spanning scheduler work, asynchronous completion, host sampling and final token handling with step/request identity. Adapter enqueue timers and summed request TTFT/TPOT would be invalid substitutes. The fail-closed collector records this failure before accuracy. Runtime/interface changes remain unauthorized under AGENTS.md; the unapplied PC-04 proposal is not a validated collector. See [diagnosis](../setup/AUTODEBUG.md), [hypothesis review](../setup/completion_export_review.md), and [proposal](../setup/PC-04-completion-observer.md).

## Accuracy protocol and coverage

The packaged ci-v1-reasoning manifest is retained verbatim with 128 selected GPQA documents from 198, per-document hashes, population hash and seed 0 choice shuffle. Zero-shot: 0 few-shot examples. Dataset content could not be downloaded or verified: HF authentication was unavailable. No questions were sent, scores computed, responses collected or stop/truncation counts measured. Those counts remain unknown rather than inferred as zero. No questions were removed or replaced.

Frozen manifest digest: `3dd6b5fe3878c8138817180626bf5f551ae6a966d11777d3eac935db695fcc38`. Population hash and selected document hashes remain in [manifest.json](manifest.json).

Intended accuracy protocol uses structured chat messages with native pinned HF template exactly once, enable_thinking=true, temperature 1, top_p 0.95, top_k 20 and 126976 output cap with EOS retained and no textual stops. The pinned release source uses r1_gpqa_diamond and a 14400s HTTP timeout; the packaged upstream runner uses gpqa_diamond_cot_zeroshot and 3600s timeout. These differences are recorded, not described as reference parity. The published 84.3% is for this model, but no measured delta exists. See [protocol](../protocol.json).

## Handoff

The stage checker exits 2 because the original runner is incomplete; its generic budget message is not a timeout. All four latency/identity validations and the context check pass; see [checks.json](checks.json). Owned servers were stopped at idle and the final device-owner scan is empty. All latency results are supplemental partial evidence; the original generated runner report and failed summary are preserved. Completion requires a validated full-phase observer/collector and valid accuracy coverage or an explicit recorded coverage exclusion. GPQA needs authorized dataset access. Stop/truncation statistics and upstream scores cannot be produced without responses.

Release matrix derives only from Qwen/Qwen3-32B P300X2 in pinned tt-inference-server411057222c2dc5a4268e693b0bd57bc0e2571f38: 128/128/C1, 8 requests. Its time estimate is separate from measured 4096/128 profiles; Stage 12 must measure that row. No old Gemma implementation settings were used as evidence.

Context contract was preserved. Evidence/context checker outputs, ownership cleanup, commit records and setup limitations are in RUN_NOTES.md and setup/. No push, CI dispatch, issue comment or message was made.
