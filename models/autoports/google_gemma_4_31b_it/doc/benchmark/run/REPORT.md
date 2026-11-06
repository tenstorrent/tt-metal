# Benchmark: google/gemma-4-31B-it

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| GPQA Diamond (zero-shot CoT planned) | 0 / 198; frozen subset 128 | Not covered: gated dataset credentials unavailable, operator decision 5 | — | 84.3 | — | [Exact-model card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| SWE-bench Verified | 0 / 500 | Not covered: tool-use harness, Stage 12 | — | Unavailable | — | — |
| Terminal-Bench 2.1 | 0 / 89 | Not covered: terminal harness, Stage 12 | — | Unavailable | — | No substitution of 2.0 scores |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 754.95 | 28.32 | 35.31 | 29.41 | 12.77 | 32.84 |
| 8 users | 8 | 8 | 4096 / 128 | 2449.17 | 62.19 | 16.08 | 98.53 | 12.33 | 35.12 |
| 16 users | 16 | 16 | 4096 / 128 | 3751.35 | 111.35 | 8.98 | 113.59 | 12.10 | 33.19 |
| 32 users | 32 | 32 | 4096 / 128 | 6541.89 | 232.00 | 4.31 | 112.42 | 11.42 | 25.24 |

**Measurement status: completed.** No local accuracy score was produced. The frozen subset and population remain preserved; published figures are references only. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Run details

Subset: `3dd6b5fe3878c8138817180626bf5f551ae6a966d11777d3eac935db695fcc38`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 34.82 | 0.23 |
| [8](perf-b8.json) | 24 / 24 | 31.18 | 0.77 |
| [16](perf-b16.json) | 48 / 48 | 54.09 | 0.89 |
| [32](perf-b32.json) | 96 / 96 | 109.30 | 0.88 |

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | TTFT | 754.95 | 753.51 | 760.59 | 761.88 |
| 1 | TPOT | 28.32 | 28.32 | 28.34 | 28.34 |
| 1 | ITL | 28.32 | 28.32 | 29.05 | 33.64 |
| 1 | E2EL | 4351.71 | 4351.33 | 4356.32 | 4357.84 |
| 8 | TTFT | 2449.17 | 2010.61 | 5475.28 | 6208.99 |
| 8 | TPOT | 62.19 | 64.27 | 73.42 | 75.97 |
| 8 | ITL | 62.19 | 30.12 | 50.14 | 801.13 |
| 8 | E2EL | 10347.00 | 10586.14 | 13637.39 | 14374.48 |
| 16 | TTFT | 3751.35 | 2122.74 | 11592.09 | 13175.48 |
| 16 | TPOT | 111.35 | 122.81 | 130.63 | 133.11 |
| 16 | ITL | 111.35 | 38.90 | 840.58 | 845.45 |
| 16 | E2EL | 17893.28 | 17714.83 | 27219.54 | 28797.25 |
| 32 | TTFT | 6541.89 | 2336.70 | 24894.53 | 28406.66 |
| 32 | TPOT | 232.00 | 261.83 | 264.81 | 272.39 |
| 32 | ITL | 232.00 | 62.80 | 926.19 | 937.25 |
| 32 | E2EL | 36005.38 | 35612.31 | 58262.14 | 61787.13 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 8 concurrent request(s): [record](perf-b8-server.json).

Server configuration for 16 concurrent request(s): [record](perf-b16-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 79.6329 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b32-evidence.json).
- Concurrency 32, decode: 29.6693 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b32-evidence.json).
- Concurrency 16, prefill: 37.5818 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b16-evidence.json).
- Concurrency 16, decode: 16.5063 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b16-evidence.json).
- Concurrency 8, prefill: 18.4401 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b8-evidence.json).
- Concurrency 8, decode: 12.7371 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b8-evidence.json).
- Concurrency 1, prefill: 5.93663 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b1-evidence.json.gz).
- Concurrency 1, decode: 28.8792 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b1-evidence.json.gz).

Total client-stage wall time: 1175.3 seconds.


Target workload beside these profiles: 32 concurrent multi-turn agents, 32768 live tokens per agent and typical context 4096 (Stage 10 estimate pending Stage 12). The measured profiles use distinct single-turn 4096-token inputs and exactly 128 generated tokens. They do not cover long-history prefix reuse, tool execution, agent waiting or the growing live working set. B1/C1 is the single-user regression and telemetry headline.

Protocol: upstream vLLM 0.26.0 benchmark client, serial use of all four Blackhole ASICs, P300x2 TP4 mesh1x4; generated autoport with all 60 layers and selected head_bfp4 policy. Full context remains 262144. Prefix caching enabled; concurrency equals server slots. Each profile warms N requests and measures max(8,3N) complete requests with distinct seeded random prompts, temperature 0 and ignore_eos=true. The client's readiness probe and extra warmups are explicitly disabled; the separate collector probe and runner warmup are outside the measured window. Raw input_lens/output_lens establish actual4096/128 lengths for every request. Performance is plain token-input completion; it intentionally does not apply a chat template. Server defaults retain checkpoint temperature1/top_p0.95 and operator top_k32 override. The official model card recommends top_k64; the served32 and intended release GPQA20 are explicit protocol differences. No live device profiler or new device synchronization was enabled.

The frozen GPQA manifest retains all128 selected document identifiers/content hashes, full198 population and population hash, shuffle seed0 and zero few-shot examples. The dataset was not downloaded using an unofficial mirror. No accuracy questions were sent and no upstream scores or raw accuracy responses exist. Accuracy stop/truncation counts are unavailable, not inferred. The intended accuracy policy is structured messages with the native pinned HF chat template applied exactly once, enable_thinking=true, temperature1, top_p0.95, top_k20, max_gen_toks126976, EOS retained and no textual stops. The available lm-eval0.4.13 gpqa_diamond_cot_zeroshot exact-match scorer differs from the release r1_gpqa_diamond extractor; its3600s request timeout differs from release14400s. These are recorded protocol gaps, not reference parity. No accuracy acceptance verdict or score threshold is applied.

Phase accounting uses consecutive adapter-observed actual completions. Each disjoint interval is assigned to its ending phase; plugin host sampling, output construction, scheduling, communication and idle gaps after the previous return lie in the next interval. The first interval starts with the first measured client request; the final client tail belongs to the final phase. Existing async readback events are awaited by the unchanged plugin before adapter host processing; no synchronization was added. This attribution is explicitly authorized by operator decision4. Submission-to-completion spans are not summed, avoiding overlap double-counting. Raw event populations and effective decode positions are retained. Useful prefill FLOPs omit pointwise operations and padding from the numerator while retaining their time; decode DRAM is a shape/precision-based estimate including weights, KV and modeled intermediate traffic. These are host-wall estimates, not device-time utilization. See per-profile assumptions and measured sampling-mode counts in roofline evidence.

Model and tokenizer revision:842da3794eaa0b77d5f08bae87a17459d91ff475. Each configuration-bN.json preserves actual process argv, source hashes, imported module path, all layer indices, precision, devices and API context capacity. Launch identity validates frozen source hashes and HEAD. Plugin runtime stays stock ca7a18246e4111629ed6cb4a7d1192002749fccf; local commit90f33d2e392eb8bfda6948df9dabe712a724cb1a contains authorized tests only. See setup/server_vllm_source.json for the installed server Python source comparison; build/native provenance is separately limited.

Before this invocation, setup installed clients, reviewed sources, prepared the frozen manifest, implemented and host-tested the collector, and committed its source. Setup wall time was not fully instrumented. No initial server was supplied: Stage10 cleanup and the startup ownership scan show none; every launch/startup is inside the timed stage. The earlier incomplete attempt is preserved under ../attempts/incomplete-96fb930d. The rejected PC-04 plugin patch remains unapplied historical evidence. GPQA noncoverage and model-owned accounting supersede the earlier blockers under operator decisions4–6.

The installed evidence checker has two stale assumptions: a3600s cap and nonempty accuracy. The local contract checker retains all identity/performance/roofline/context evidence requirements, explicitly validates authorized noncoverage, and uses1800s per configured batch. Its source hashes and exact corrections are retained in evidence_check.json; setup/workflow-check-contract.patch is the generic upstream correction. Installed skills remain read-only. The outer multigoal script still invokes that stale installed checker; its failure is reported separately from measurement validity.

| Profile | Actual prefill tokens | Active decode tokens | Host-sampled prefill steps | Device-sampled prefill steps | Device-sampled decode steps |
|---|---:|---:|---:|---:|---:|
| B1 | 32768 | 1016 | 8 | 8 | 1016 |
| B8 | 98304 | 3048 | 24 | 24 | 392 |
| B16 | 196608 | 6096 | 48 | 48 | 400 |
| B32 | 393216 | 12192 | 96 | 96 | 416 |

| Profile | Maximum active decode rows | Mean active decode rows | Full-batch decode steps / total |
|---|---:|---:|---:|
| B1 | 1 | 1.000 | 1016 / 1016 |
| B8 | 8 | 7.776 | 360 / 392 |
| B16 | 16 | 15.240 | 336 / 400 |
| B32 | 32 | 29.308 | 288 / 416 |

Scheduler ramp-up and completion tails are included; server slots are capacity, not a claim that every step is full.

These are measured-window step counts, not lifetime counters. Stock-plugin mixed intermediate prefill uses the operator-accepted host sampling path. Performance outputs all reach the requested128 tokens with EOS ignored; the upstream performance dump does not retain per-request finish_reason, so stop reasons are not inferred.

Release matrix: adopted128/128/C1 row from same-device dense Qwen3-32B, without borrowing Gemma implementation settings. Estimated 28.947s; actual 26.931246740045026s for8 requests. See [release matrix](../release_benchmark_matrix.json).

| Server slots | Startup inside budget (seconds) | Source commit |
|---|---:|---|
| 32 | 228.301 | 27ea28bafd0195ba8ce89dd36da722b15fa6fa24 |
| 16 | 198.256 | 27ea28bafd0195ba8ce89dd36da722b15fa6fa24 |
| 8 | 186.232 | 27ea28bafd0195ba8ce89dd36da722b15fa6fa24 |
| 1 | 172.224 | 27ea28bafd0195ba8ce89dd36da722b15fa6fa24 |

Verification: the corrected evidence gate and full-context check pass. The unchanged installed outer gate exits2 with `missing required accuracy tasks`; its narrow correction awaits the requested exception to the installed-package read-only rule. [Check results](checks.json).

Raw log/event originals remain locally under run/. Large originals and command logs are also preserved byte-for-byte in gzip archives for Git; [raw_archives.json](raw_archives.json) maps filenames and SHA256 values. After a fresh checkout, restore originals with `python tools/archive_benchmark_raw.py doc/benchmark/run --restore` from the model directory before replaying evidence checks.

Client stage through this final report: 1473.426s / 7200s, including launch/reload, warmup, verification, inference, collection and reporting. Upstream runner elapsed: 1175.322s.
