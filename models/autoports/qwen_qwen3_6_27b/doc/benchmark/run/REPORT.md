# Benchmark: Qwen/Qwen3.6-27B

Status: completed. Accuracy concurrency: 32 (largest measured batch; server prefix caching: False).

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| mmlu_pro | 196 / 12032 | exact_match,custom-extract | 90.82 | 86.20 | +4.62 | [reference](https://huggingface.co/Qwen/Qwen3.6-27B) |
| ifeval | 160 / 541 | prompt_level_strict_acc,none | 89.38 | — | — | Unavailable |
| ifeval | 160 / 541 | inst_level_strict_acc,none | 92.77 | — | — | Unavailable |
| ifeval | 160 / 541 | prompt_level_loose_acc,none | 92.50 | — | — | Unavailable |
| ifeval | 160 / 541 | inst_level_loose_acc,none | 94.78 | — | — | Unavailable |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 887.38 | 26.80 | 37.31 | 29.83 | 10.87 | 43.03 |
| 8 users | 8 | 8 | 4096 / 128 | 3378.17 | 98.29 | 10.17 | 63.79 | 8.32 | 31.46 |
| 16 users | 16 | 16 | 4096 / 128 | 7510.35 | 237.28 | 4.21 | 53.48 | 6.36 | 20.48 |
| 32 users | 32 | 32 | 4096 / 128 | 20040.49 | 676.46 | 1.48 | 37.78 | 4.37 | 12.11 |
| 1 user on largest server (FUNCTIONAL 128/128) | 1 | 32 | 128 / 128 | 278.75 | 26.58 | 37.63 | 35.03 | — | — |
| 1 user on largest server (long context) | 1 | 32 | 32768 / 128 | 16351.02 | 27.77 | 36.01 | 6.44 | — | — |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Each batch profile sets server max-num-seqs and request concurrency to the batch size. The two one-user rows run on the largest server: the FUNCTIONAL 128/128 reference and the long-context cost a lone user pays on that server.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are complete for all four batch profiles. Dashes on the two lone-user rows mean roofline accounting is not required for those extra rows. HTTP concurrency is not a fixed device batch size.

## Answer statistics

| Benchmark | Extracted by filter | Blank at cap | Answer tokens median / p90 / max | Output cap | Recommended cap |
|---|---|---:|---:|---:|---:|
| mmlu_pro | custom-extract 196/196 (100.0%) | 0 (0.0%) | 2016 / 4362 / 14725 | 32768 | 32768 |
| ifeval | none 160/160 (100.0%) | 0 (0.0%) | 2484 / 4638 / 32724 | 32768 | 32768 |

Targets: extraction at least 95% per filter; blank answers at the cap at most 5%. Recommended cap is p90 × 1.5 rounded up to a power-of-two K, never below the current cap.

- Sample reduction, mmlu_pro: 280 → 196. 18-request native-thinking probe completed in 347.057 seconds. Conservative full-subset projection 347.057*536/32=5813.2 seconds plus 1800 seconds reserved for performance/reporting exceeds 7200 client seconds. 356 samples project 3861.01 seconds, leaving tail margin. Same deterministic document ranking; thinking and 32768 cap unchanged.
- Sample reduction, ifeval: 256 → 160. 18-request native-thinking probe completed in 347.057 seconds. Conservative full-subset projection 347.057*536/32=5813.2 seconds plus 1800 seconds reserved for performance/reporting exceeds 7200 client seconds. 356 samples project 3861.01 seconds, leaving tail margin. Same deterministic document ranking; thinking and 32768 cap unchanged.

Reference protocol notes:
- mmlu_pro, exact_match,custom-extract: Published model-card full-set score. This run uses frozen common subset and upstream lm-eval 0.4.13 5-shot, first_n subject examples as structured turns, answer-is-(X) formatting and upstream custom-extract; published exact recipe is not specified and best-practice model-card JSON answer format differs.

## Run details

Subset: `e0632cc5d8bc6b4494aaf56441fbbd48ab06462aef183157a8fe4b93da282002`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [mmlu_pro](mmlu_pro/results.json) | 196 | 0 (0.0%) | 0 | 5683.3 |
| [ifeval](ifeval/results.json) | 160 | 0 (0.0%) | 0 | 5683.3 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Profile | Completed / requested | Wall seconds | Requests/s |
|---|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 34.33 | 0.23 |
| [8](perf-b8.json) | 24 / 24 | 48.16 | 0.50 |
| [16](perf-b16.json) | 48 / 48 | 114.89 | 0.42 |
| [32](perf-b32.json) | 96 / 96 | 325.24 | 0.30 |
| [functional](perf-b32-c1-functional.json) | 4 / 4 | 14.62 | 0.27 |
| [long_context](perf-b32-c1-long_context.json) | 2 / 2 | 39.76 | 0.05 |

| Profile | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| 1 | TTFT | 887.38 | 886.99 | 888.97 | 889.15 |
| 1 | TPOT | 26.80 | 26.80 | 26.80 | 26.80 |
| 1 | ITL | 26.80 | 26.81 | 27.04 | 27.53 |
| 1 | E2EL | 4291.12 | 4290.68 | 4292.95 | 4293.19 |
| 8 | TTFT | 3378.17 | 2071.19 | 9650.84 | 11016.85 |
| 8 | TPOT | 98.29 | 108.27 | 113.10 | 113.83 |
| 8 | ITL | 98.29 | 35.78 | 342.80 | 511.93 |
| 8 | E2EL | 15860.70 | 15887.47 | 23403.25 | 24797.48 |
| 16 | TTFT | 7510.35 | 2888.81 | 27520.96 | 31394.63 |
| 16 | TPOT | 237.28 | 268.38 | 271.86 | 276.55 |
| 16 | ITL | 237.28 | 83.07 | 680.35 | 689.43 |
| 16 | E2EL | 37644.41 | 37092.02 | 61673.87 | 65608.49 |
| 32 | TTFT | 20040.49 | 4460.43 | 85920.56 | 98226.30 |
| 32 | TPOT | 676.46 | 780.39 | 783.11 | 789.93 |
| 32 | ITL | 676.46 | 727.90 | 1038.43 | 1048.36 |
| 32 | E2EL | 105951.43 | 103546.13 | 185166.42 | 197358.37 |
| functional | TTFT | 278.75 | 278.91 | 282.21 | 282.42 |
| functional | TPOT | 26.58 | 26.58 | 26.58 | 26.58 |
| functional | ITL | 26.58 | 26.57 | 26.88 | 27.25 |
| functional | E2EL | 3654.05 | 3654.05 | 3657.76 | 3658.02 |
| long_context | TTFT | 16351.02 | 16351.02 | 16563.18 | 16582.04 |
| long_context | TPOT | 27.77 | 27.77 | 27.78 | 27.78 |
| long_context | ITL | 27.77 | 27.88 | 28.19 | 28.50 |
| long_context | E2EL | 19877.36 | 19877.36 | 20090.65 | 20109.61 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 8 concurrent request(s): [record](perf-b8-server.json).

Server configuration for 16 concurrent request(s): [record](perf-b16-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 210.883 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html ; https://github.com/tenstorrent/tt-perf-report#understanding-the-performance-report. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b32-accounting.json).
- Concurrency 32, decode: 114.348 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b32-accounting.json).
- Concurrency 16, prefill: 72.481 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html ; https://github.com/tenstorrent/tt-perf-report#understanding-the-performance-report. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b16-accounting.json).
- Concurrency 16, decode: 42.3992 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b16-accounting.json).
- Concurrency 8, prefill: 27.7033 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html ; https://github.com/tenstorrent/tt-perf-report#understanding-the-performance-report. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b8-accounting.json).
- Concurrency 8, decode: 20.4489 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b8-accounting.json).
- Concurrency 1, prefill: 7.06783 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html ; https://github.com/tenstorrent/tt-perf-report#understanding-the-performance-report. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b1-accounting.json).
- Concurrency 1, decode: 27.2594 s. See retained accounting components and doc/benchmark/ROOFLINE_METHOD.md; useful architecture FLOPs and actual per-rank stored weight metadata with KV/recurrent/staging traffic estimates. Peak: https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html. Timing: FIFO observed CQ completion timeline for every prefill and existing decode readback event, with actual decode host conversion intervals attributed to decode and removed from overlapping prefill; host gaps included; no added synchronization.. [Evidence](roofline-b1-accounting.json).

Client-clock wall time: 6299.8 of 7200 seconds. Server setup (launch, compile, warmup): 1533.0 of 7200 seconds. Total: 7832.9 seconds.


## Evaluation protocol and provenance

Both model and tokenizer are pinned to `Qwen/Qwen3.6-27B` revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`. Accuracy used upstream lm-evaluation-harness **0.4.13**, one shared pool of **32 concurrent requests on the 32-slot server**, and the Stage 9/10 prefix-caching setting: **disabled**, because recurrent GDN state is not reconstructible from cached attention pages. The server's default non-thinking chat setting was overridden explicitly by `chat_template_kwargs.enable_thinking=true` on every evaluation request. The [identity](../identity.json) retains the generated autoport import, all 64 layers, selected precision, TP4 hardware, source commits and 262144-token context capacity.

| Task variant | Frozen samples / population | Few-shot and upstream scorer |
|---|---:|---|
| `mmlu_pro`, group v2.0; 14 subject tasks v3.1 | 196 / 12032 | Five `first_n` validation examples per subject as separate user/assistant turns; `RegexFilter` with `answer is \(?([ABCDEFGHIJ])\)?`, then `TakeFirstFilter`; `exact_match,custom-extract`, ignoring case and punctuation |
| `ifeval`, task v4.0 | 160 / 541 | Zero-shot original prompts; unchanged final-text filter; upstream strict/loose prompt-level and instruction-level accuracy metrics |

MMLU-Pro subject counts/populations are biology 12/717, business 13/789, chemistry 18/1132, computer science 7/410, economics 14/844, engineering 16/969, health 13/818, history 6/381, law 18/1101, math 22/1351, other 15/924, philosophy 8/499, physics 21/1299 and psychology 13/798. The [frozen manifest](manifest.json) retains every selected document ID/hash, full-population hash and few-shot hash. Its SHA-256 is `e0632cc5d8bc6b4494aaf56441fbbd48ab06462aef183157a8fe4b93da282002`; selection seed is `tt-bringup-benchmark-v1`. The final questions are nested within the common 280/256 subsets; [sample-sizing evidence](../setup/sample_sizing.json) records the measured-speed reduction. Questions were not selected or replaced using scores.

Both tasks use `max_gen_toks=32768`, `do_sample=true`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`, `until=[]` and `chat_template_kwargs={"enable_thinking":true}`. HTTP sampling seed is 1234; harness seeds are Python 0, NumPy 1234, Torch 1234 and few-shot 1234. Native EOS is retained. The harness passes structured messages for the native HF template to be applied once by the server. It does not send a preformatted native-template string. Reasoning is retained separately and only final content reaches the scorer. Completion-token statistics include reasoning and final-answer tokens.

The [published exact-model MMLU-Pro figure](https://huggingface.co/Qwen/Qwen3.6-27B) is **86.2%** over the full dataset. This report's subset delta is arithmetic, not a matched-protocol reproduction: the published exact recipe is unspecified, and the model card recommends JSON choice formatting while this run preserves upstream five-shot `the answer is (X)` formatting and its scorer. IFEval has no figure in the consulted model-card/announcement sources. GPQA Diamond was considered before evaluation but its official gated dataset was unavailable with the existing credentials; the [failed preparation](../setup/prepare_common.log) is retained.

| Task | Raw responses | Normal stops | Token-limit stops | Blank at cap | Extracted/nonempty |
|---|---:|---:|---:|---:|---:|
| MMLU-Pro | 196 | 196 | 0 | 0 | 196/196 |
| IFEval | 160 | 160 | 0 | 0 | 160/160 |

The shared accuracy interval was **5683.337 seconds**; task wall times refer to that same interval. The [protocol audit](protocol_audit.json) verifies all 356 document/request/response bindings, frozen document hashes, generation settings and exact equality between raw final content and scored input. All reasoning remains separate, and upstream extraction is reproduced without modifying scores. Full questions, raw responses, request links, sample outputs and task configurations remain under [MMLU-Pro](mmlu_pro/results.json) and [IFEval](ifeval/results.json). The final [native-template check](native_template_check.json) passed all 356 inputs: pinned local HF token IDs exactly match the server’s native structured-chat tokenization and the original response prompt-token counts. Local templating was used only for verification; each measured request was templated once by the server.

## Protocol and integration repairs

- MMLU-Pro's upstream textual `Question:` stop was removed before evaluation (`until=[]`) to prevent it cutting off thinking; native EOS and the 32768-token cap were retained. The original few-shot prompts and scorer remain unchanged.
- Performance-only random prompts originally changed length through the upstream text decode/re-encode roundtrip. [The exact-ID adapter](../bench_exact.py) preserves original token IDs through upstream vLLM 0.26's completions/timing client. Each performance row retains complete prompt IDs and hashes in its `.prompts.jsonl`/`.prompts.json` sidecars. The [original mismatch evidence](../setup/perf-probe/perf-b32.json), [repair and CPU tests](../setup/EXACT_TRANSPORT_AUTOFIX.md), and [passing live preflight](../setup/perf-probe-exact/perf-b32.json) are retained. Accuracy chat inputs were unaffected.
- The native-template verifier was corrected for upstream batched serialized arguments and the Transformers return type. Requests were unchanged. [Failed verifier log](../setup/template_probe.log) and [successful 18-question setup verification](../setup/template_probe_retry.log) remain available.
- The nonblocking event-query extension deployment and launch-argument recorder were repaired during setup. [AutoFix diagnosis and validation](../AUTOFIX.md) links the preserved failures; direct evidence includes [initial extension failure](../event_query_smoke.log) and [initial configuration-recorder failure](../setup/initial_server.log). These repairs preceded timed accuracy.

## Implementation, hardware and measurement provenance

Every server imported `models.autoports.qwen_qwen3_6_27b.tt.generator_vllm` from this checkout, with all 64 layers and precision policy `gdn_attn8_head8_mlp4_lofi`. The participating hardware was P300x2: four Blackhole p300c ASICs (chips 0–3), mesh 1×4, TP4, DP1, serialized device ownership. The retained configuration snapshots include observed registration, precision settings, actual process argv, environment, effective capacity and hardware identification. Each profile has its own [1-slot](configuration-b1.json), [8-slot](configuration-b8.json), [16-slot](configuration-b16.json) or [32-slot](configuration-b32.json) snapshot.

All profiles preserve Stage 10 async scheduling, 512 scheduler-token budget, trace mode `all`, request-seeded sampling policy, disabled prefix caching and full 262144-token context. The unchanged [context contract](../../context_contract.json) has SHA-256 `316de5af81c2e1c693159eeb2cbd5de45d3200e69703d3acafbdf637a1050581`. No batches were skipped. Synthetic performance uses greedy generation, ignored EOS, distinct prompts and exactly 128 generated tokens per request. All **182/182 measured performance requests** completed (176 batch-profile requests plus six lone-user requests).

| Component | Recorded source |
|---|---|
| tt-metal / generated autoport during all timed profiles | `eeb42050d6563f711e136b918ee351fd614e04bc` |
| TT vLLM plugin | `da1587235b84b13f2826cfd7e8b448b340322a54` |
| Serving vLLM 0.31.0+empty release source | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Upstream performance vLLM 0.26.0+empty | `568afb3a13806beb53bb2e6bd518269357b237c0` |

The installed serving wheel does not embed a Git SHA; its release-tag mapping, original installation method and package RECORD fingerprint are stated in [package provenance](../server_package_provenance.json). The performance adapter changes token transport only; upstream scheduling, streaming timing and usage remain intact. [Performance protocol audit](performance_protocol_audit.json) verifies prompt lengths, distinctness, output counts, timing fields and source bindings. The upstream raw `max_concurrent_requests` statistic counts one-second buckets; exact response intervals verify actual configured concurrency instead.

[Roofline methodology](../ROOFLINE_METHOD.md) documents the event-completion timeline, work formulas, actual stored tensor metadata, precision/fidelity peaks and physical batch-state traffic. Each profile was collected before its server stopped. No added device synchronization or live device profiler was used. These are useful-work and modeled-traffic estimates against nominal participating-chip peaks, not measured bus counters or instruction occupancy. Host gaps and completion-observation delay are included. The independent [batch-32 audit](roofline_audit_b32.json) matched all requests, excluded 51,269 prior submissions and reconciled 325.231 seconds of nonoverlapping phases with 325.243 seconds of measured client time.

## Validation, clocks and local handoff

The first timed attempt completed; no second attempt was needed. Runner client time was **6299.826 / 7200 seconds**, and launch/compile/warmup setup time was **1533.026 / 7200 seconds**. Setup completed before invocation took **2488.525 seconds**, separately recorded in [the setup receipt](../setup/pre_invocation_setup.json). Additional validation and final reporting time is recorded in [the closeout clock](post_run_validation.json) and remains inside the client bound. The runner’s original clocks and raw summary are preserved.

The [benchmark evidence check](evidence_check.log) passed, including recomputed answer statistics, exact token counts, server configurations, phase accounting and both clocks. The [context check](context_check_final.log) passed at the full HF capacity. Its historical advisory matches concern old process-return-code text, not a reduced capacity. Native-template verification and independent protocol audits passed. These are protocol and evidence checks; no accuracy acceptance verdict is applied.

Only stage-owned servers were stopped, after drain checks and completed phase collection; [final shutdown](final_server_stop.log) and [ownership check](cleanup.json) retain the result. No reset was used and no enclosing reservation was acquired or released. Integration failures and their repairs remain in [AutoFix notes](../AUTOFIX.md) and setup logs.

Stage implementation commits are `21fddded` (nonblocking event query), `367a4c4c` (collector and controller), `3e7473e6` (accounting audit), `305a4735` (exact-token transport) and `eeb42050` (frozen protocol). Final report/evidence commit SHAs are logged in `../final_commits.log`; nothing is pushed. [Artifact manifest](../artifact_manifest.json) inventories retained evidence. Compact evidence and stage-owned source are committed; large and immutable raw outputs remain unchanged in this persistent checkout, including all questions, raw responses, full phase logs and prompt-token sidecars. The repository’s 500-KB file limit is respected without rewriting hashed evidence.
