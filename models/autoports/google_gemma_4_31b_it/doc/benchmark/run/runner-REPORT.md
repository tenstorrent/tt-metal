# Benchmark: google/gemma-4-31B-it

Status: completed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 754.57 | 28.32 | 35.31 | 29.41 | 12.76 | 32.85 |
| 8 users | 8 | 8 | 4096 / 128 | 2418.45 | 61.47 | 16.27 | 99.71 | 12.52 | 35.36 |
| 32 users | 32 | 32 | 4096 / 128 | 6560.48 | 232.24 | 4.31 | 112.26 | 11.40 | 25.22 |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Run details

Subset: `none (no accuracy tasks)`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 34.81 | 0.23 |
| [8](perf-b8.json) | 24 / 24 | 30.81 | 0.78 |
| [32](perf-b32.json) | 96 / 96 | 109.46 | 0.88 |

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | TTFT | 754.57 | 755.06 | 758.15 | 758.72 |
| 1 | TPOT | 28.32 | 28.32 | 28.34 | 28.34 |
| 1 | ITL | 29.70 | 28.31 | 29.81 | 106.95 |
| 1 | E2EL | 4351.44 | 4351.95 | 4353.88 | 4354.22 |
| 8 | TTFT | 2418.45 | 2011.89 | 5505.41 | 6235.36 |
| 8 | TPOT | 61.47 | 64.29 | 73.47 | 73.57 |
| 8 | ITL | 64.96 | 30.16 | 59.23 | 802.41 |
| 8 | E2EL | 10224.74 | 10180.54 | 13667.02 | 14401.97 |
| 32 | TTFT | 6560.48 | 2341.04 | 24985.72 | 28485.51 |
| 32 | TPOT | 232.24 | 261.56 | 264.70 | 273.53 |
| 32 | ITL | 244.44 | 61.88 | 927.51 | 1133.77 |
| 32 | E2EL | 36054.32 | 35790.18 | 58239.63 | 61763.96 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 8 concurrent request(s): [record](perf-b8-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 79.7639 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b32-evidence.json).
- Concurrency 32, decode: 29.6964 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b32-evidence.json).
- Concurrency 8, prefill: 18.1601 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b8-evidence.json).
- Concurrency 8, decode: 12.6501 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b8-evidence.json).
- Concurrency 1, prefill: 5.93946 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b1-evidence.json.gz).
- Concurrency 1, decode: 28.8745 s. tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence Peak: doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC. Timing: Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time.. [Evidence](roofline-b1-evidence.json.gz).

Total client-stage wall time: 900.0 seconds.
