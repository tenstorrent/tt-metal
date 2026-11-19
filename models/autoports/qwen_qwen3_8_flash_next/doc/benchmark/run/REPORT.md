# Benchmark: Qwen/Qwen3.8-Flash-Next

Status: completed. Accuracy concurrency: 2 (largest measured batch; server prefix caching: False).

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| gpqa_diamond_cot_zeroshot | 1 / 198 | exact_match,flexible-extract | 100.00 | 91.70 | +8.30 | [reference](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| gpqa_diamond_cot_zeroshot | 1 / 198 | exact_match,strict-match | 0.00 | 91.70 | -91.70 | [reference](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| ifeval | 1 / 541 | prompt_level_strict_acc,none | 100.00 | — | — | Unavailable |
| ifeval | 1 / 541 | inst_level_strict_acc,none | 100.00 | — | — | Unavailable |
| ifeval | 1 / 541 | prompt_level_loose_acc,none | 100.00 | — | — | Unavailable |
| ifeval | 1 / 541 | inst_level_loose_acc,none | 100.00 | — | — | Unavailable |

**Protocol and scope:** Each accuracy task uses one frozen document. GPQA’s strict-match row and its delta are literal-format diagnostics (0/1 extracted), not an accuracy estimate: the upstream zero-shot prompt does not request its required phrase. The upstream flexible filter extracted 1/1; IFEval extracted 1/1. Both responses stopped normally, with zero blank-at-cap answers. All upstream scores are retained. Published comparisons are arithmetic comparisons to full-dataset figures, not matched reproductions. See [protocol review](protocol-review.json).

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 users | 2 | 2 | 4096 / 128 | 439650.88 | 315.25 | 3.17 | 0.50 | 0.02 | 10.59 |
| 1 user on largest server (FUNCTIONAL 128/128) | 1 | 2 | 128 / 128 | 7368.59 | 103.62 | 9.65 | 6.24 | — | — |
| 1 user on largest server (long context) | 1 | 2 | 32768 / 128 | 1947330.08 | 113.79 | 8.79 | 0.07 | — | — |
| 1 users | — | — | not measured: Stage 10 did not validate a dedicated one-slot server at the selected full context and precision; concurrency-1 rows use the validated two-slot server. | — | — | — | — | — | — |
| 8 users | — | — | not measured: Stage 10 did not validate an eight-slot resident server for this full48 selected-precision implementation. | — | — | — | — | — | — |
| 16 users | — | — | not measured: Stage 10 did not validate a sixteen-slot resident server for this full48 selected-precision implementation. | — | — | — | — | — | — |
| 32 users | — | — | not measured: Stage 10 did not validate a thirty-two-slot resident server for this full48 selected-precision implementation. | — | — | — | — | — | — |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Each batch profile sets server max-num-seqs and request concurrency to the batch size. The two one-user rows run on the largest server: the FUNCTIONAL 128/128 reference and the long-context cost a lone user pays on that server.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are supplied for the measured batch profile. The two extra one-user latency rows do not have separate roofline estimates. HTTP concurrency is not a fixed device batch size.

## Answer protocol statistics

| Benchmark | Extracted by filter | Blank at cap | Answer tokens median / p90 / max | Output cap | Recommended cap |
|---|---|---:|---:|---:|---:|
| gpqa_diamond_cot_zeroshot | flexible-extract 1/1 (100.0%); strict-match 0/1 (0.0%) | 0 (0.0%) | 1826 / 1826 / 1826 | 32768 | 32768 |
| ifeval | none 1/1 (100.0%) | 0 (0.0%) | 12504 / 12504 / 12504 | 32768 | 32768 |

Targets: extraction at least 95% per filter; blank answers at the cap at most 5%. Recommended cap is p90 × 1.5 rounded up to a power-of-two K, never below the current cap.

- gpqa_diamond_cot_zeroshot: filter strict-match extracted 0/1 answers (0.0% < 95%); check answer format, few-shot/template and scorer regex

- Sample reduction, gpqa_diamond_cot_zeroshot: 128 → 1. Measured shared 2+2 pilot: 1913.705s / 17735 completion tokens (9.267 tokens/s). Per-task observed p90 sensitivity totals 1539.1s for one each; two each plus required performance reserve 6049.5s and reporting/check reserve600s forecasts 9727.6s, above9000s client bound. One each forecasts 8188.5s. Retain fixed common seed/ranking, native xhigh policy and32768 cap; no score-based selection. See setup/sample-sizing.json and pilot-answer-statistics.json.
- Sample reduction, ifeval: 256 → 1. Measured shared 2+2 pilot: 1913.705s / 17735 completion tokens (9.267 tokens/s). Per-task observed p90 sensitivity totals 1539.1s for one each; two each plus required performance reserve 6049.5s and reporting/check reserve600s forecasts 9727.6s, above9000s client bound. One each forecasts 8188.5s. Retain fixed common seed/ranking, native xhigh policy and32768 cap; no score-based selection. See setup/sample-sizing.json and pilot-answer-statistics.json.

Reference protocol notes:
- gpqa_diamond_cot_zeroshot, exact_match,flexible-extract: Exact model card reports GPQA Diamond 91.7; its task-specific prompt/scorer, repetitions and generation cap are unspecified. This run uses frozen common subset, lm-eval 0.4.13 zero-shot CoT, deterministic seed-0 choice shuffle, native xhigh thinking, model-card sampling, and reports upstream strict and flexible filters. Not a matched published recipe. The published scorer is unspecified; the flexible-filter delta is an arithmetic comparison, not a matched reproduction.
- gpqa_diamond_cot_zeroshot, exact_match,strict-match: Exact model card reports GPQA Diamond 91.7; its task-specific prompt/scorer, repetitions and generation cap are unspecified. This run uses frozen common subset, lm-eval 0.4.13 zero-shot CoT, deterministic seed-0 choice shuffle, native xhigh thinking, model-card sampling, and reports upstream strict and flexible filters. Not a matched published recipe. Pilot inspection found that the upstream zero-shot prompt does not demand the strict regex phrase. Strict extraction0/2 versus flexible2/2 with nonblank normal stops is a format diagnostic; both upstream scores remain unchanged. The strict-filter arithmetic delta is not an ability comparison.

## Run details

Subset: `c007b76687e457cab48465c3ab9e403be0d582abe6aeda0e3ea3696777814c9d`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [gpqa_diamond_cot_zeroshot](gpqa_diamond_cot_zeroshot/results.json) | 1 | 0 (0.0%) | 0 | 1569.5 |
| [ifeval](ifeval/results.json) | 1 | 0 (0.0%) | 0 | 1569.5 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Profile | Completed / requested | Wall seconds | Requests/s |
|---|---:|---:|---:|
| [2](perf-b2.json) | 8 / 8 | 2030.51 | 0.00 |
| [functional](perf-b2-c1-functional.json) | 4 / 4 | 82.11 | 0.05 |
| [long_context](perf-b2-c1-long_context.json) | 2 / 2 | 3923.56 | 0.00 |

| Profile | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| 2 | TTFT | 439650.88 | 464159.45 | 482991.29 | 490779.53 |
| 2 | TPOT | 315.25 | 344.82 | 345.49 | 345.57 |
| 2 | ITL | 315.25 | 122.34 | 575.30 | 576.55 |
| 2 | E2EL | 479687.09 | 507924.52 | 526731.64 | 534496.50 |
| functional | TTFT | 7368.59 | 7369.37 | 7376.29 | 7376.86 |
| functional | TPOT | 103.62 | 103.63 | 103.65 | 103.65 |
| functional | ITL | 103.62 | 103.58 | 104.35 | 104.65 |
| functional | E2EL | 20528.03 | 20529.16 | 20533.82 | 20534.33 |
| long_context | TTFT | 1947330.08 | 1947330.08 | 1950592.06 | 1950882.02 |
| long_context | TPOT | 113.79 | 113.79 | 113.82 | 113.82 |
| long_context | ITL | 113.79 | 113.69 | 115.48 | 119.09 |
| long_context | E2EL | 1961781.68 | 1961781.68 | 1965040.73 | 1965330.43 |

Server configuration for 2 concurrent request(s): [record](perf-b2-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 2, prefill: 1920.38 s. Sum 2*M*K*N over full48 executed logical chunk rows, active top10 experts, replicated rank work, actual causal index/selected-attention positions, and useful recurrence; include replay tails and every chunk's logits. Exclude tile padding and minor scalar/nonlinear FLOPs. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html; tech_reports/GEMM_FLOPS/GEMM_FLOPS.md. Timing: Exclusive ordered CQ0 completion intervals including intervening host gaps; nonblocking EventQuery observations; final host readback tail included. [Evidence](phase-accounting-b2.json).
- Concurrency 2, decode: 110.127 s. Conservative source-derived payload plus explicit DRAM materializations over TP4/EP4, padded prepared weight storage and BFP exponents, active expert union lower estimate, selected KV, recurrent state/copies, indexer terminal materializations and output traffic. Per-slot attention/head; shared bucket MoE. Not measured DRAM transactions. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html. Timing: Exclusive ordered CQ0 completion intervals including intervening host gaps; nonblocking EventQuery observations; final host readback tail included. [Evidence](phase-accounting-b2.json).

Client-clock wall time: 7627.7 of 9000 seconds. Server setup (launch, compile, warmup): 2503.1 of 3600 seconds. Total: 10130.8 seconds.

Evidence checks: [benchmark evidence](evidence-check.log) is valid with the documented strict-format advisory; [context contract](context-check.log) passes at 262144 tokens. The [source integrity check](source-integrity.json) confirms unchanged implementation, precision and context files. [Final observer export](final-observer-export.json) and owned shutdown completed without pending events or device holders. Full raw evidence and restoration instructions are described in [EVIDENCE.md](../EVIDENCE.md).

Final reporting and independent review completed at 2026-11-18T05:42:25.058775+00:00. The client clock through reporting was **8298.676009 / 9000 seconds**, including checks and review; setup remained **2503.071863 / 3600 seconds**, with **5256.310899 seconds** of pre-invocation preparation recorded separately. See [reporting timing](reporting-timing.json), [check summary](check-summary.json) and the independent [clean-pass evidence review](STAGE_REVIEW.md). This is a measurement/protocol review, with no model-accuracy verdict. Packaging and local commit closure are logged in [RUN_NOTES.md](../RUN_NOTES.md).

The upstream raw peak-concurrency field uses inclusive one-second bins and reads 3/2/2. Independent reconstruction from request intervals gives actual peaks **2/1/1**, matching the three configured concurrency limits; server slots remain two throughout. See the review for the calculation and scope.

<!-- qwen-benchmark-provenance:start -->

## Provenance and measurement qualifications

The observed server imported `models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm` with 48/48 layers, precision `combined_shared4_mixed`, and context capacity 262144. Hardware: P300x2: four Blackhole ASICs, TP4/EP4 mesh 1x4, visible devices 0,1,2,3; serial serving. Model revision `812d96e10a0dd82b69d749e68693698cf6c36daa`; tokenizer revision `812d96e10a0dd82b69d749e68693698cf6c36daa`. See [identity](<../identity.json>) and [constructed module/precision evidence](<../setup/b2-1794966797189440608/phases.jsonl.identity.json>). Per-profile records above retain the observed launch commands, server slots and prefix-caching settings.

Runtime/source commits: tt-metal `45bdcf46660436ef99f5c1c502b65ed2865cf544`; vllm-tt-plugin `629f07cb385494b14d0ee3d138de039d71d09546`; vllm `db9527a46873454610df6dbedf79a36d6bf1a7f6`; lm_eval `ddd67220430a2470529f25fd5c05a576ca1057a0`; vllm_benchmark_client `568afb3a13806beb53bb2e6bd518269357b237c0`.

Setup before invocation spans 2026-11-18T01:14:47+00:00 through 2026-11-18T02:42:23.310899+00:00 (5256.3 s elapsed). This separate wall interval includes installation, downloads, instrumentation repairs, initial launches, warmup and pilots; it is not active compute time and is not added to the runner's client/setup clocks printed above. See [setup timestamps](<../setup/pre-invocation-timing.json>) and [run notes](<../RUN_NOTES.md>).

Recorded throughput with six decimal places (the upstream tables round to two):

| Profile | Requests/s | Output tokens/s | Total tokens/s |
|---|---:|---:|---:|
| B2 | 0.003940 | 0.504307 | 16.642130 |
| FUNCTIONAL (128/128, concurrency 1) | 0.048713 | 6.235291 | 12.470582 |
| Long context (32768/128, concurrency 1) | 0.000510 | 0.065247 | 16.768428 |

Initial launch-to-readiness intervals completed before invocation:

| Launch | Server slots | Seconds |
|---|---:|---:|
| [b2-1794965173630864396](<../setup/b2-1794965173630864396/launch.json>) / [readiness](<../setup/b2-1794965173630864396/ready.json>) | 2 | 263.5 |
| [b2-1794966797189440608](<../setup/b2-1794966797189440608/launch.json>) / [readiness](<../setup/b2-1794966797189440608/ready.json>) | 2 | 232.6 |

These readiness intervals include any ready servers later replaced; they are setup evidence, not serving measurements.

Template evidence: [server-interpreter smoke](<../setup/template-smoke-server.json>) records `Qwen2Tokenizer` and native template SHA-256 `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375` with identical rendering to the client smoke. The evaluator sends structured messages with its tokenizer backend disabled; the server applies the native HF template once. This setup smoke establishes rendering compatibility; per-task result metadata records the actual run's template contract. [gpqa_diamond_cot_zeroshot](<gpqa_diamond_cot_zeroshot/results.json>) records {'gpqa_diamond_cot_zeroshot': 0} few-shot settings; [ifeval](<ifeval/results.json>) records {'ifeval': 0} few-shot settings.

Generation dictionaries (including thinking, sampling, token caps and textual stops) are frozen in [run_config.json](<run_config.json>). Final content is scored separately from retained reasoning; no raw reasoning is substituted for an empty final answer. The task variants, document hashes, populations and upstream scorer settings remain in [manifest.json](<manifest.json>) and task artifacts. Answer extraction and truncation facts are reported above.

Recorded thinking/sampling settings:

- gpqa_diamond_cot_zeroshot, ifeval: `{"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "xhigh"}, "do_sample": true, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "until": []}`.

Measured decode batches are derived from server events, separately from HTTP concurrency and configured slots:

| Profile | Server slots | Observed active decode batches | Maximum polling gap s | Maximum completion bracket s |
|---|---:|---|---:|---:|
| [B2](<phase-accounting-b2.json>) | 2 | 1 active: 1016 steps | 34.357589 | 35.298675 |

| Profile | Phase | Nominal FIFO attribution s | Completion-bracket lower s | Completion-bracket upper s |
|---|---|---:|---:|---:|
| B2 | prefill | 1920.376579 | 1642.077869 | 21956.884759 |
| B2 | decode | 110.126940 | 108.998801 | 393.153629 |

The headline roofline percentages use nominal FIFO attribution from host-observed completion events. They are not precise phase occupancy measurements. Delayed observer polling widens completion brackets; conservative summed upper brackets can exceed the client wall interval. The retained evidence includes FIFO validation, event timestamps, uncertainty and client-request mapping. No device synchronization or live device profiler was added.

Decode bytes model stored precision, padded weight scans, active-expert sharing, selected KV and explicit materialized traffic. They are conservative payload/materialization estimates, not measured DRAM transactions. Expert-union sensitivity bounds apply only to that category; unmodeled staging, scratch, cache effects and repeated reads remain explicit omissions. Prefill uses useful executed-row FLOPs, including replayed chunks, and a fidelity-weighted nominal four-chip peak; its matrix-equivalent normalization does not establish an achievable SFPU ceiling.

Stage-local helper/evidence commits (separate from the launched runtime identity):

| Commit | Purpose |
|---|---|
| `7184c2f226222eb86a6e09c037ca36ae2ecea1f2` | Prepare Qwen benchmark phase completion recorder and owned server profiles |
| `48220c1057d97ce92da36a1d660e0e6d8e9092b2` | Repair benchmark observation integration and add full-phase accounting |
| `772f7a07e58a35697d9353cafbe6adc16db1dbf7` | Bind benchmark token transport and complete phase accounting |
| `45bdcf46660436ef99f5c1c502b65ed2865cf544` | Match TTNN event ABI and retain observer failure evidence |
| `377741b2ea24bc2b2ae496e50b9eb121e3811e98` | Validate native event order in benchmark phase attribution |
| `690672a64c270c0b8b8c08b80a4ee37f67245b06` | Add benchmark report provenance finalizer |
| `4f4524de2c2a9cdd81582f027ba3bf70beab671b` | Use complete pre-invocation setup interval in benchmark report |
| `6ad77f962804512eae44c28551dd36d404e1e847` | Add lossless split benchmark evidence archives |
| `c3c54169304cda0af3ff76d15e35f2120d71d9af` | Preserve small benchmark throughput rates in report appendix |

This appendix adds provenance and measurement qualifications; upstream scores, denominators, raw responses and summary values are unchanged.

<!-- qwen-benchmark-provenance:end -->
