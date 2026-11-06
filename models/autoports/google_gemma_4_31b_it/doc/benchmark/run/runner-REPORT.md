# Benchmark: google/gemma-4-31B-it

Status: completed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 754.95 | 28.32 | 35.31 | 29.41 | 12.77 | 32.84 |
| 8 users | 8 | 8 | 4096 / 128 | 2449.17 | 62.19 | 16.08 | 98.53 | 12.33 | 35.12 |
| 16 users | 16 | 16 | 4096 / 128 | 3751.35 | 111.35 | 8.98 | 113.59 | 12.10 | 33.19 |
| 32 users | 32 | 32 | 4096 / 128 | 6541.89 | 232.00 | 4.31 | 112.42 | 11.42 | 25.24 |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

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
