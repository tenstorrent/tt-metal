# Benchmark: Qwen/Qwen3.6-35B-A3B

Status: completed. Accuracy concurrency: 32 (largest measured batch; server prefix caching: False).

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| mmlu_pro | 14 / 12032 | subject_macro:exact_match,custom-extract | 78.57 | 85.20 | -6.63 | [reference](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| ifeval | 18 / 541 | prompt_level_strict_acc,none | 83.33 | — | — | Unavailable |
| ifeval | 18 / 541 | inst_level_strict_acc,none | 90.62 | — | — | Unavailable |
| ifeval | 18 / 541 | prompt_level_loose_acc,none | 83.33 | — | — | Unavailable |
| ifeval | 18 / 541 | inst_level_loose_acc,none | 90.62 | — | — | Unavailable |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 83884.32 | 21.79 | 45.89 | 1.48 | 0.23 | 10.46 |
| 8 users | 8 | 8 | 4096 / 128 | 266244.23 | 3611.12 | 0.28 | 1.41 | 0.22 | 9.71 |
| 16 users | 16 | 16 | 4096 / 128 | 396011.51 | 8401.13 | 0.12 | 1.40 | 0.22 | 9.43 |
| 32 users | 32 | 32 | 4096 / 128 | 666284.62 | 18731.86 | 0.05 | 1.34 | 0.21 | 8.69 |
| 1 user on largest server (FUNCTIONAL 128/128) | 1 | 32 | 128 / 128 | 3032.59 | 31.37 | 31.87 | 18.24 | — | — |
| 1 user on largest server (long context) | 1 | 32 | 32768 / 128 | 742008.39 | 32.51 | 30.76 | 0.17 | — | — |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Each batch profile sets server max-num-seqs and request concurrency to the batch size. The two one-user rows run on the largest server: the FUNCTIONAL 128/128 reference and the long-context cost a lone user pays on that server.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are present for every required batch profile. The two lone-user rows do not require roofline accounting; their dashes mean not applicable. HTTP concurrency is not a fixed device batch size.

## Answer statistics

| Benchmark | Extracted by filter | Blank at cap | Answer tokens median / p90 / max | Output cap | Recommended cap |
|---|---|---:|---:|---:|---:|
| mmlu_pro | custom-extract 14/14 (100.0%) | 0 (0.0%) | 1521 / 2295 / 2560 | 32768 | 32768 |
| ifeval | none 18/18 (100.0%) | 0 (0.0%) | 2438 / 6025 / 16352 | 32768 | 32768 |

Targets: extraction at least 95% per filter; blank answers at the cap at most 5%. Recommended cap is p90 × 1.5 rounded up to a power-of-two K, never below the current cap.

Owner-approved client budget: 28800 seconds; approved by Vladan, full autonomy delegated to the hub. Owner-approved third attempt: measured fixed performance prefill alone exceeds the default client bound. Approval source: /home/ttuser/qwen36-35b-a3b/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/benchmark/run/owner-approval.json.

- Sample reduction, mmlu_pro: 280 → 14. Four native-thinking pilot requests completed in 485.635s on the 32-slot server at concurrency4; individual elapsed294.670–485.633s and outputs1916–3908 tokens. Serial Stage10 prefill is about22ms/input token. Reduce the common frozen ranking before changing thinking, cap or tasks; retain one MMLU-Pro question per subject and18 IFEval questions to limit accuracy time within the7200s client bound. Fixed performance repetitions remain a separate documented budget conflict.
- Sample reduction, ifeval: 256 → 18. Four native-thinking pilot requests completed in 485.635s on the 32-slot server at concurrency4; individual elapsed294.670–485.633s and outputs1916–3908 tokens. Serial Stage10 prefill is about22ms/input token. Reduce the common frozen ranking before changing thinking, cap or tasks; retain one MMLU-Pro question per subject and18 IFEval questions to limit accuracy time within the7200s client bound. Fixed performance repetitions remain a separate documented budget conflict.

Reference protocol notes:
- mmlu_pro, subject_macro:exact_match,custom-extract: Published model-card MMLU-Pro figure. This report uses the upstream five-shot CoT subject recipe with structured multiturn chat and a frozen 14/12032 subset; published population, scorer and generation recipe are not established as identical. Delta is descriptive.

## Run details

Subset: `204ea844a375ed212ff836af30565a151c8a55ac32e2ddd239607a52d4ead3e5`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [mmlu_pro](mmlu_pro/results.json) | 14 | 0 (0.0%) | 0 | 3844.0 |
| [ifeval](ifeval/results.json) | 18 | 0 (0.0%) | 0 | 3844.0 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Profile | Completed / requested | Wall seconds | Requests/s |
|---|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 693.22 | 0.011540 |
| [8](perf-b8.json) | 24 / 24 | 2174.83 | 0.011035 |
| [16](perf-b16.json) | 48 / 48 | 4390.07 | 0.010934 |
| [32](perf-b32.json) | 96 / 96 | 9141.29 | 0.010502 |
| [functional](perf-b32-c1-functional.json) | 4 / 4 | 28.07 | 0.142503 |
| [long_context](perf-b32-c1-long_context.json) | 2 / 2 | 1492.28 | 0.001340 |

| Profile | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| 1 | TTFT | 83884.32 | 83810.73 | 84385.57 | 84421.64 |
| 1 | TPOT | 21.79 | 21.79 | 21.80 | 21.80 |
| 1 | ITL | 21.79 | 21.80 | 22.00 | 22.59 |
| 1 | E2EL | 86652.06 | 86578.60 | 87152.90 | 87188.81 |
| 8 | TTFT | 266244.23 | 221334.36 | 607902.02 | 689444.44 |
| 8 | TPOT | 3611.12 | 3961.26 | 5005.78 | 5007.71 |
| 8 | ITL | 3611.12 | 127.47 | 145.40 | 88683.92 |
| 8 | E2EL | 724856.52 | 725105.95 | 1110832.76 | 1192923.27 |
| 16 | TTFT | 396011.51 | 224705.99 | 1221436.46 | 1390514.61 |
| 16 | TPOT | 8401.13 | 9751.89 | 10548.42 | 10815.93 |
| 16 | ITL | 8401.13 | 233.31 | 89642.88 | 90629.25 |
| 16 | E2EL | 1462954.92 | 1462856.29 | 2463123.23 | 2632135.47 |
| 32 | TTFT | 666284.62 | 234850.74 | 2555901.50 | 2910673.35 |
| 32 | TPOT | 18731.86 | 22127.22 | 22484.82 | 23227.65 |
| 32 | ITL | 18731.86 | 431.17 | 94140.57 | 94836.77 |
| 32 | E2EL | 3045230.57 | 3044888.85 | 5363873.83 | 5720430.74 |
| functional | TTFT | 3032.59 | 3028.28 | 3074.54 | 3078.25 |
| functional | TPOT | 31.37 | 31.39 | 31.40 | 31.40 |
| functional | ITL | 31.37 | 31.37 | 31.63 | 32.20 |
| functional | E2EL | 7017.07 | 7011.11 | 7059.49 | 7062.81 |
| long_context | TTFT | 742008.39 | 742008.39 | 744144.72 | 744334.62 |
| long_context | TPOT | 32.51 | 32.51 | 32.51 | 32.51 |
| long_context | ITL | 32.51 | 32.63 | 33.05 | 33.40 |
| long_context | E2EL | 746137.36 | 746137.36 | 748273.98 | 748463.90 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 8 concurrent request(s): [record](perf-b8-server.json).

Server configuration for 16 concurrent request(s): [record](perf-b16-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 8918.9 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b32-evidence.json).
- Concurrency 32, decode: 222.37 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b32-evidence.json).
- Concurrency 16, prefill: 4283.86 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b16-evidence.json).
- Concurrency 16, decode: 106.187 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b16-evidence.json).
- Concurrency 8, prefill: 2120.82 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b8-evidence.json).
- Concurrency 8, decode: 53.99 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b8-evidence.json).
- Concurrency 1, prefill: 671.013 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b1-evidence.json).
- Concurrency 1, decode: 22.1952 s. Source-derived useful-work model; see per-operation terms, MoE routing/reuse assumptions and bounds in linked evidence. Estimated bytes, not hardware counters. Peak: https://tenstorrent.com/hardware/cards; https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md; https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md. Timing: Union of accepted submit-to-existing-host-completion intervals on CLOCK_MONOTONIC; queued decode overlaps counted once; intervening host gaps assigned to next phase; cross-phase overlaps rejected; no added device synchronization. [Evidence](roofline-b1-evidence.json).


## Protocol and retained inputs

Accuracy uses upstream lm-evaluation-harness 0.4.13 on the 32-slot server with 32 concurrent HTTP requests. All selected questions remain in the denominator. MMLU-Pro uses task metadata 3.1, five validation examples selected by upstream `first_n` within each subject, subject-specific chain-of-thought prompts, and the upstream `custom-extract` filter: `answer is \(?([ABCDEFGHIJ])\)?`, followed by `take_first` and exact match with case and punctuation ignored. IFEval uses metadata 4.0, zero-shot prompts, the upstream train-split evaluation population and upstream strict/loose prompt/instruction scoring. Only final-answer content is scored; reasoning remains in the raw responses.

The native HF chat template is applied exactly once by the server to structured messages. Thinking is enabled. Accuracy generation uses a 32768-token cap, temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0, sampling enabled, and no added stop strings (`until=[]`); native EOS remains active. Python random seed is 0; NumPy, Torch and few-shot seeds are 1234. Template/tokenizer revision is `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.

The frozen common subsets were reduced before observing scores, using their existing ranked order: MMLU-Pro 280 to 14 questions (one per each of 14 subjects; full population 12032), and IFEval 256 to 18 (full population 541). The measured setup speed motivated this recorded `sample_reduction`; thinking, output cap and task recipes were retained. GPQA preparation encountered an authenticated dataset-access restriction; its preparation evidence is retained. Two applicable tasks completed. Per-document IDs/content hashes, full populations, few-shot hashes and task variants are recorded in [manifest.json](manifest.json). All 32 questions and contexts are preserved in [benchmark-inputs.jsonl](benchmark-inputs.jsonl), with offline reconstruction and request-link verification in [input-export-provenance.json](input-export-provenance.json). Each task directory retains upstream scores, samples, raw HTTP responses and request links.

The MMLU-Pro model-card reference is 85.2%; the reported difference is descriptive. This 14-question subset and the model-card full evaluation are not a matched statistical comparison. No exact-model IFEval reference was located. No accuracy acceptance verdict is assigned.

## Serving identity and measurement interpretation

Every measured server preserves all 40 configured layers, selected `baseline_bfp8` precision, the generated autoport adapter and full 262144-token capacity. The model and tokenizer are pinned to `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. Hardware is two P300 boards / four Blackhole p300c chips, TP4/EP4 on a 1x4 FABRIC_1D mesh, with 110 worker cores and eight DRAM banks per chip. The selected policy uses BFP8 for the named dense/expert weight groups, BF16 activations/KV/CCL, and the explicit FP32 state/router and fidelity exceptions in each server identity; it is not a blanket BFP8 claim.

The imported module is `models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm`, SHA-256 `f6d414c865f8408a5d54f0bd64abd6fb119bb0bed0534e1eeb7408cbc698e6ab`. Each `perf-b*-server.json` records its observed launch command, imported file/hash, layer counts, precision, hardware and configuration hash. Each referenced `configuration-b*.json` retains the launch evidence. The servers use block size 32, max-num-batched-tokens 2048, async scheduling, decode-only tracing with 805306368 trace bytes and 8224 KV pages. Prefix caching is false for accuracy and all performance rows, preserving the Stage 9/10 setting.

Required profiles use server slots equal to configured concurrent requests (1, 8, 16 or 32), repeated distinct 4096-token prompts and exactly 128 generated tokens. Performance uses greedy decoding with EOS ignored. The two concurrency-1 rows use the 32-slot server: 128/128 for FUNCTIONAL and 32768/128 for long context. Warmup measurements are setup evidence and are excluded from the measured rows. HTTP concurrency is not a fixed device decode batch size; prefill and decode scheduling can vary within a profile.

Upstream raw `max_concurrent_requests` counts requests touching inclusive one-second buckets. It reports 2 for concurrency 1, 10 for concurrency 8, 18 for concurrency 16 and 34 for concurrency 32 because sequential replacements share a bucket. The actual benchmark semaphore and transport connection limit remain 1, 8, 16 and 32; reconstruction from unrounded stream timing peaks at those bounds. Raw values are retained unchanged. See [the concurrency audit](attempt3-support/setup/concurrency-metric-audit.md). The report labels configured concurrent requests separately from server slots.

Long prefill intervals for replacement requests interrupt decode progress in the multi-user profiles. This explains the gap between mean TPOT and median ITL: batch 32 has mean TPOT 18731.86 ms and median ITL 431.17 ms. The raw timelines and complete phase ledgers reconcile these measured client costs; they are not isolated device-kernel timings.

Full-phase roofline timing uses host wall clocks at existing blocking prefill completion and real asynchronous decode completion boundaries. No extra device synchronization or live device profiler was introduced. The collector binds the measured HTTP window to request lifetimes, complete prefill chunks and decode positions, checks for unfinished or cross-phase-overlapping calls, unions same-phase overlaps and accounts for intervening host gaps. It collects each profile before the owning server stops. Prefill FLOPs are shape-derived; decode DRAM bytes include modeled weights, KV and recurrent-state traffic with expected unique-expert occupancy and explicit bounds. These are estimates rather than hardware byte counters. Peaks use participating worker cores/fidelity and nominal four-chip DRAM bandwidth; low utilization is reported without a minimum threshold. Evidence and detailed work records are linked from [roofline.json](roofline.json).

## Source provenance and earlier attempts

The frozen serving-source baseline is tt-metal `18527ae01cfdb0b5e6ab58ce6714f0f3f0eb6a6a`. Later bookkeeping commit `bfebe7b0d9816529ab59e13fa6a657c7cb38ef82` adds the offline input-export mode and validation; serving source is unchanged, as recorded in [serving-source-baseline.json](attempt3-support/setup/serving-source-baseline.json). The TT plugin commit is `5d1785b76d17a8989c9495f6939abea6cef62f09`. Server vLLM is `0.31.0+empty`; `db9527a46873454610df6dbedf79a36d6bf1a7f6` identifies the v0.31.0 release, not an embedded installed-build SHA. Performance client vLLM is 0.26.0 (release `568afb3a13806beb53bb2e6bd518269357b237c0`). Both environments use Transformers 5.12.1 and tokenizers 0.22.2.

Editable benchmark workflow `f2873162ea1b8b4971aa9b41c83a551e7c1d53e5` (0.1.26) includes the HTTP timeout repair and explicit owner-approved budget. Installed cached package 0.1.24 was left unchanged. The HTTP repair passed CPU tests and the complete third-attempt accuracy run, including the response that exceeded the former 3600-second child timeout. Workflow tests passed 766 tests with one skip; package, version, host and import checks are retained. The enclosing multigoal runner still refers to its older cached checker, which cannot accept the explicit 28800-second budget; the repaired standalone stage checker is used and its actual result is retained, with the cached-checker incompatibility disclosed separately.

Attempts 1 and 2 remain preserved. Attempt 1 failed before requests because its launch-argument alias was not recognized by the evidence checker. Attempt 2 failed on a hardcoded child HTTP timeout; its partial IFEval responses and recovered scores are historical evidence, not substituted into this attempt. The third attempt was explicitly approved by Vladan with an eight-hour client limit and three-hour setup limit. The owner directed a truthful partial-result handoff if this final attempt reached its limits, with no fourth attempt. See [owner-approval.json](owner-approval.json). All stage commits are local; nothing is pushed.

## Finalization and clocks

The historical sample-reduction rationale refers to the original 7200-second bound. The same frozen selection was retained for this explicitly approved 28800-second third attempt. No batches were skipped. All 182 measured performance requests completed, including the six lone-user requests. All 32 accuracy responses ended with native `stop`; none were blank or token-limited.

Client-clock wall time through 2026-11-18T13:37:31.783076+00:00: 22939.786 / 28800 seconds. This includes 1133.332 seconds of continuous post-run export, cleanup, validation, reporting and commit work since the preserved runner summary. Server setup (launch, compile and explicit warmup): 7209.514 / 10800 seconds. Total invocation-to-handoff time: 30149.300 seconds. Separately recorded preparation before invocation: 242.460 seconds from the documented minute-precision preparation start.

All four owned servers were stopped after collection using ownership and idle checks. The final device-holder check was empty; no reset was used and the hub-owned reservation was left unchanged. [Cleanup evidence](attempt3-support/setup/cleanup-verification.json). Earlier failed evidence remains under [attempts/attempt2](attempts/attempt2/archive-index.json) and the other historical run directories.

Independent saved-artifact accounting audits: [batch 1](attempt3-support/setup/completed-b1-accounting-audit.md), [batch 8](attempt3-support/setup/completed-b8-accounting-audit.md), [batch 16](attempt3-support/setup/completed-b16-accounting-audit.md), [batch 32](attempt3-support/setup/completed-b32-accounting-audit.md). [Frozen accuracy audit](attempt3-support/setup/completed-accuracy-audit.md).

Evidence and context check outcomes: [checks.json](checks.json).

Independent stage review: [stage-review.md](attempt3-support/setup/stage-review.md).

Local commit provenance: [commits.json](commits.json). No commits were pushed.

The [post-commit closing record](attempt3-support/setup/closing-record.log) records the final bookkeeping SHA and continuous clock through that commit; the summary above is explicitly timestamped before the bookkeeping checkpoint.

Runner-check remediation: the exact cached 0.1.24 check still exits 2 because it rejects the owner-approved eight-hour client budget. The 0.1.26 check exits 0 on the same measurements. The original approval, reproduction and explicit refutation are retained in [the stage work log](../work_log.md).
