# Benchmark: ornith-ai/Ornith-1.5-9B

Status: completed. Accuracy concurrency: 32 (largest measured batch; server prefix caching: False).

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| mmlu_pro | 140 / 12032 | exact_match,custom-extract | 79.29 | — | — | Unavailable |
| gsm8k_cot | 128 / 1319 | exact_match,strict-match | 85.94 | — | — | Unavailable |
| gsm8k_cot | 128 / 1319 | exact_match,flexible-extract | 95.31 | — | — | Unavailable |
| ifeval | 64 / 541 | prompt_level_strict_acc,none | 78.12 | — | — | Unavailable |
| ifeval | 64 / 541 | inst_level_strict_acc,none | 84.31 | — | — | Unavailable |
| ifeval | 64 / 541 | prompt_level_loose_acc,none | 79.69 | — | — | Unavailable |
| ifeval | 64 / 541 | inst_level_loose_acc,none | 85.29 | — | — | Unavailable |

| Profile | Concurrent requests | Server slots | ISL / OSL | Mean TTFT ms | Mean TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 393.79 | 8.92 | 112.12 | 83.84 | 8.52 | 44.66 |
| 32 users | 32 | 32 | 4096 / 128 | 3530.74 | 139.21 | 7.18 | 186.74 | 8.36 | 8.73 |
| 1 user on largest server (FUNCTIONAL 128/128) | 1 | 32 | 128 / 128 | 230.25 | 17.08 | 58.56 | 53.35 | — | — |
| 1 user on largest server (long context) | 1 | 32 | 32768 / 128 | 5901.70 | 21.30 | 46.95 | 14.87 | — | — |
| 8 users | — | — | not measured: Stage 10 did not separately validate this server slot configuration; only slots1 and32 have launch and measurement evidence. | — | — | — | — | — | — |
| 16 users | — | — | not measured: Stage 10 did not separately validate this server slot configuration; only slots1 and32 have launch and measurement evidence. | — | — | — | — | — | — |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Each batch profile sets server max-num-seqs and request concurrency to the batch size. The two one-user rows run on the largest server: the FUNCTIONAL 128/128 reference and the long-context cost a lone user pays on that server.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are present for both measured batch profiles. Roofline estimates are not required for the two additional lone-user rows, where — means not applicable. HTTP concurrency is not a fixed device batch size. Decode tokens/s/user is the reciprocal of mean TPOT; aggregate output throughput includes prefill and scheduling.

## Answer statistics

| Benchmark | Extracted by filter | Blank at cap | Answer tokens median / p90 / max | Output cap | Recommended cap |
|---|---|---:|---:|---:|---:|
| mmlu_pro | custom-extract 140/140 (100.0%) | 0 (0.0%) | 473 / 2213 / 13388 | 32768 | 32768 |
| gsm8k_cot | flexible-extract 128/128 (100.0%); strict-match 113/128 (88.3%) | 0 (0.0%) | 235 / 622 / 2641 | 32768 | 32768 |
| ifeval | none 64/64 (100.0%) | 0 (0.0%) | 810 / 3014 / 8650 | 32768 | 32768 |

Targets: extraction at least 95% per filter; blank answers at the cap at most 5%. Recommended cap is p90 × 1.5 rounded up to a power-of-two K, never below the current cap.

- gsm8k_cot: filter strict-match extracted 113/128 answers (88.3% < 95%); check answer format, few-shot/template and scorer regex

- Sample reduction, mmlu_pro: 280 → 140. Pre-invocation native-thinking pilot: 18 responses, 14886 output tokens in 152.218s (97.79 output tokens/s). Task-weighted full subset projects 14016s; selected subset projects 3885s, leaving margin within7200s for tail variation, performance, collection, scoring and reporting. Pilot is small; estimate is not a latency guarantee. See pilot_statistics.json and pilot_accuracy/.
- Sample reduction, gsm8k_cot: 256 → 128. Pre-invocation native-thinking pilot: 18 responses, 14886 output tokens in 152.218s (97.79 output tokens/s). Task-weighted full subset projects 14016s; selected subset projects 3885s, leaving margin within7200s for tail variation, performance, collection, scoring and reporting. Pilot is small; estimate is not a latency guarantee. See pilot_statistics.json and pilot_accuracy/.
- Sample reduction, ifeval: 256 → 64. Pre-invocation native-thinking pilot: 18 responses, 14886 output tokens in 152.218s (97.79 output tokens/s). Task-weighted full subset projects 14016s; selected subset projects 3885s, leaving margin within7200s for tail variation, performance, collection, scoring and reporting. Pilot is small; estimate is not a latency guarantee. See pilot_statistics.json and pilot_accuracy/.

## Run details

Subset: `195d043bddd1648f5894a2b310c14ab4907550f82eb2bd9aca49cefd731d38ab`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [mmlu_pro](mmlu_pro/results.json) | 140 | 0 (0.0%) | 0 | 574.3 |
| [gsm8k_cot](gsm8k_cot/results.json) | 128 | 0 (0.0%) | 0 | 574.3 |
| [ifeval](ifeval/results.json) | 64 | 0 (0.0%) | 0 | 574.3 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Profile | Completed / requested | Wall seconds | Requests/s |
|---|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 12.21 | 0.65 |
| [32](perf-b32.json) | 96 / 96 | 65.80 | 1.46 |
| [functional](perf-b32-c1-functional.json) | 4 / 4 | 9.60 | 0.42 |
| [long_context](perf-b32-c1-long_context.json) | 2 / 2 | 17.21 | 0.12 |

| Profile | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| 1 | TTFT | 393.79 | 398.62 | 404.86 | 405.65 |
| 1 | TPOT | 8.92 | 8.92 | 8.92 | 8.92 |
| 1 | ITL | 8.92 | 8.93 | 9.06 | 9.14 |
| 1 | E2EL | 1526.45 | 1530.96 | 1537.64 | 1538.41 |
| 32 | TTFT | 3530.74 | 833.08 | 14880.20 | 16965.48 |
| 32 | TPOT | 139.21 | 154.52 | 163.64 | 163.76 |
| 32 | ITL | 139.21 | 13.26 | 596.00 | 604.60 |
| 32 | E2EL | 21209.95 | 21240.87 | 35261.77 | 37755.52 |
| functional | TTFT | 230.25 | 245.95 | 246.54 | 246.55 |
| functional | TPOT | 17.08 | 17.08 | 17.08 | 17.08 |
| functional | ITL | 17.08 | 17.07 | 17.25 | 17.66 |
| functional | E2EL | 2399.08 | 2414.79 | 2415.31 | 2415.34 |
| long_context | TTFT | 5901.70 | 5901.70 | 5934.32 | 5937.22 |
| long_context | TPOT | 21.30 | 21.30 | 21.31 | 21.31 |
| long_context | ITL | 21.30 | 21.41 | 21.59 | 22.23 |
| long_context | E2EL | 8606.75 | 8606.75 | 8640.06 | 8643.02 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 28.3613 s. TP4 aggregate. Useful projection/head and causal-attention/recurrent contraction FLOPs, excluding tile padding, discarded work, scalar/SFPU ops and redundant chunk rebuilds. Peak is work-weighted harmonic LoFi/HiFi2/HiFi4 from observed compute grid at published 1.35GHz. Decode reads actual stored projection/head/norm bytes once per fixed-capacity step (batch shared), logical page-rounded BF8 KV read/write, FP32 recurrent/conv traffic including inactive snapshot/restore, embedding/logit/sample traffic. Includes all executed steps, including asynchronous pipeline tail beyond delivered tokens. DRAM is a logical traffic estimate, not physical bus transactions; repeated SDPA tile fetches and small scalar/dispatch traffic are not modeled. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md. Timing: Host perf_counter: prefill adapter entry through terminal host readback; decode through existing async completion and host materialization. Union/complement on one timeline includes scheduler gaps; no added synchronization.. [Evidence](roofline-b32-evidence.json).
- Concurrency 32, decode: 37.3814 s. TP4 aggregate. Useful projection/head and causal-attention/recurrent contraction FLOPs, excluding tile padding, discarded work, scalar/SFPU ops and redundant chunk rebuilds. Peak is work-weighted harmonic LoFi/HiFi2/HiFi4 from observed compute grid at published 1.35GHz. Decode reads actual stored projection/head/norm bytes once per fixed-capacity step (batch shared), logical page-rounded BF8 KV read/write, FP32 recurrent/conv traffic including inactive snapshot/restore, embedding/logit/sample traffic. Includes all executed steps, including asynchronous pipeline tail beyond delivered tokens. DRAM is a logical traffic estimate, not physical bus transactions; repeated SDPA tile fetches and small scalar/dispatch traffic are not modeled. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md. Timing: Host perf_counter: prefill adapter entry through terminal host readback; decode through existing async completion and host materialization. Union/complement on one timeline includes scheduler gaps; no added synchronization.. [Evidence](roofline-b32-evidence.json).
- Concurrency 1, prefill: 2.31978 s. TP4 aggregate. Useful projection/head and causal-attention/recurrent contraction FLOPs, excluding tile padding, discarded work, scalar/SFPU ops and redundant chunk rebuilds. Peak is work-weighted harmonic LoFi/HiFi2/HiFi4 from observed compute grid at published 1.35GHz. Decode reads actual stored projection/head/norm bytes once per fixed-capacity step (batch shared), logical page-rounded BF8 KV read/write, FP32 recurrent/conv traffic including inactive snapshot/restore, embedding/logit/sample traffic. Includes all executed steps, including asynchronous pipeline tail beyond delivered tokens. DRAM is a logical traffic estimate, not physical bus transactions; repeated SDPA tile fetches and small scalar/dispatch traffic are not modeled. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md. Timing: Host perf_counter: prefill adapter entry through terminal host readback; decode through existing async completion and host materialization. Union/complement on one timeline includes scheduler gaps; no added synchronization.. [Evidence](roofline-b1-evidence.json).
- Concurrency 1, decode: 9.89006 s. TP4 aggregate. Useful projection/head and causal-attention/recurrent contraction FLOPs, excluding tile padding, discarded work, scalar/SFPU ops and redundant chunk rebuilds. Peak is work-weighted harmonic LoFi/HiFi2/HiFi4 from observed compute grid at published 1.35GHz. Decode reads actual stored projection/head/norm bytes once per fixed-capacity step (batch shared), logical page-rounded BF8 KV read/write, FP32 recurrent/conv traffic including inactive snapshot/restore, embedding/logit/sample traffic. Includes all executed steps, including asynchronous pipeline tail beyond delivered tokens. DRAM is a logical traffic estimate, not physical bus transactions; repeated SDPA tile fetches and small scalar/dispatch traffic are not modeled. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md. Timing: Host perf_counter: prefill adapter entry through terminal host readback; decode through existing async completion and host materialization. Union/complement on one timeline includes scheduler gaps; no added synchronization.. [Evidence](roofline-b1-evidence.json).

Client-clock wall time: 707.8 of 7200 seconds. Server setup (launch, compile, warmup): 114.2 of 3600 seconds. Total: 822.0 seconds.


## Protocol and published references

MMLU-Pro uses the upstream 5-shot, 14-subject generative recipe and custom-extract scorer. GSM8K-CoT uses the upstream 8-shot recipe with both strict-match and flexible-extract scorers. IFEval uses the upstream zero-shot recipe and all four prompt/instruction strict/loose metrics. The harness is lm-eval 0.4.13. The [frozen manifest](manifest.json) records each document ID/content hash, population hash, task variant, few-shot count and few-shot hash. The selected documents are nested in the packaged common subset, verified in [common-subset checks](../subset/common_subset_verification.json), with seed `tt-bringup-benchmark-v1`. All selected questions remain in their denominators.

The server applied the pinned native HF chat template exactly once to structured messages, with few-shot examples as separate turns. The [template parity probe](../native_template_parity.json) matched the native token IDs exactly. Eval explicitly enabled thinking despite the Stage10 server's default of false. Generation used temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0 and max output 32768. Ordinary EOS remained enabled. Textual task stops were cleared (`until: []`) to avoid cutting reasoning on incidental benchmark strings. These are recorded protocol differences. Only final `message.content` was scored; raw reasoning and API responses remain in each task's `responses.jsonl`, with questions, prompts and upstream scores in `samples_*.jsonl` and `results.json`.

All 332 final responses stopped normally: **zero token-limit responses and zero blank-at-cap answers**. The answer-statistics table was read and checked. MMLU-Pro, IFEval and GSM8K's flexible filter extract 100% of responses. GSM8K's strict filter misses 15 nonblank finals: 11 currency-prefixed answers, one expression, one bold-marked answer and two without its required answer phrase. The [format audit](gsm8k_format_audit.json) retains every miss and both filtered outputs. Reporting the existing upstream flexible filter alongside strict-match resolves the identified format mismatch without rewriting answers or changing scorers. The strict 85.94% result remains a protocol-sensitive upstream score; flexible 95.31% is reported separately. Flexible extraction can itself select a later number in a multi-number answer, so extraction completeness is not an accuracy verdict.

No exact-model MMLU-Pro, GSM8K-CoT or IFEval score was found in the primary model card or author announcement; published columns and deltas therefore remain unavailable. The exact model publishes **GPQA Diamond 86.4%**, averaged over five runs, in its [pinned model card](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/489cb97981b8654bcfcf30ce1f94ed1b62e07b53/README.md). GPQA was not run because the upstream dataset gate denied access; [setup evidence](../pilot_subset/prepare.log) is preserved. The publication also omits a complete matching recipe, so no cross-task delta is inferred. See the [protocol record](../protocol.json) for sources and selection details.

## Implementation, accounting and clocks

The [identity](../identity.json) and [32-slot](configuration-b32.json)/[1-slot](configuration-b1.json) startup records prove the imported generated autoport `models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm`, all 32 configured layer indices, HF model/tokenizer revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, selected `bfp8_all_lofi` policy, TP4 mesh1x4 and four Blackhole p300c chips (P300x2). Both servers retained context 262144, block 32, scheduling budget 4096, integrated64 device sampling, async scheduling, decode-only traces and Stage10 responsive prefill/decode interleaving. Effective prefix caching is **false for both profiles and accuracy**. Slots 8 and 16 lack separate Stage10 validation and are explicitly skipped.

Measured source commit: `af2350b0493fbd9be8b36942a59415a7c5f4fec4`; vLLM TT plugin: `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`. The server is the working Stage10 vLLM 0.31.0+empty PyPI distribution; its upstream tag association is recorded separately and is not claimed as an embedded build SHA. The performance client is upstream vLLM 0.26.0 at `568afb3a13806beb53bb2e6bd518269357b237c0`; [client setup](../client_setup/) records dependency versions and installation times.

Performance uses a model-owned sample-preparation adapter around the unchanged upstream `vllm bench serve` request/metric path. Every synthetic prompt is a distinct preserved integer-ID list with the pinned tokenizer, avoiding a demonstrated client/server text-retokenization mismatch. This workload excludes server text-tokenization cost. Actual client usage and captured scheduler lengths agree. `perf-*.json.inputs.json` retains every input ID; raw performance JSON retains all per-request counts and latency distributions. All measured and warmup requests completed at their exact requested lengths. Failed text-transport setup measurements are excluded and preserved in [setup_failed_transport](../setup_failed_transport/), with diagnosis and repair in [AUTOFIX.md](../AUTOFIX.md).

Full-phase accounting was prepared and validated before accuracy. Both profiles were collected before their servers stopped, at existing terminal-prefill readback and asynchronous decode materialization boundaries, with no added device synchronization or live device profiler. The [accounting specification](../PHASE_ACCOUNTING.md), roofline evidence and raw phase records distinguish modeled useful FLOPs/logical DRAM traffic from physical bus measurements. No minimum utilization or accuracy threshold was applied.

Pre-invocation setup consumed 2258.1 seconds of wall time, including installation, investigations, invalid and repaired pilots, subset preparation and idle time; it is recorded separately in [invocation.json](../invocation.json) and [setup_probe.json](../setup_probe.json). The initial 32-slot server became ready after approximately 140 seconds under its own bounded startup. Timed attempt 1 completed with the runner clocks shown above. Post-run validation and final reporting are recorded separately in `completion_timing.json` and remain within the 7200-second client bound when added to runner client time. Only one fresh timed attempt was needed.

The [evidence check](evidence_check.log) found valid measurements with one advisory for strict GSM8K extraction; the paired flexible score and complete format audit address that protocol limitation. The [context check](context_check.log) passed at 262144, and `doc/context_contract.json` is byte-for-byte unchanged. The check's historical small-cap advisories refer to earlier stage logs, not these launches. Final process cleanup and local commit records are in [RUN_NOTES.md](../RUN_NOTES.md). No accuracy acceptance verdict is issued; the bringup owner judges the results.

The repository commit hook normalized final newlines in performance JSON. [Normalization evidence](hook_normalization.json) proves unchanged parsed measurements and retains original hashes; required byte-level roofline links were updated and the evidence check rerun successfully with the same single advisory. The final source adds only a post-benchmark output newline normalization to prevent recurrence; measured model execution remains at the recorded source commit.
