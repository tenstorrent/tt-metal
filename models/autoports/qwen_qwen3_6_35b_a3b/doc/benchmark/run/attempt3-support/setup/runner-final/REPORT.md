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

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every batch profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Answer quality

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
| [1](perf-b1.json) | 8 / 8 | 693.22 | 0.01 |
| [8](perf-b8.json) | 24 / 24 | 2174.83 | 0.01 |
| [16](perf-b16.json) | 48 / 48 | 4390.07 | 0.01 |
| [32](perf-b32.json) | 96 / 96 | 9141.29 | 0.01 |
| [functional](perf-b32-c1-functional.json) | 4 / 4 | 28.07 | 0.14 |
| [long_context](perf-b32-c1-long_context.json) | 2 / 2 | 1492.28 | 0.00 |

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

Client-clock wall time: 21806.5 of 28800 seconds. Server setup (launch, compile, warmup): 7209.5 of 10800 seconds. Total: 29016.0 seconds.
