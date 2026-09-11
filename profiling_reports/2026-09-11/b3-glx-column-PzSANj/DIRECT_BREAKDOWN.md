# B3 direct capture validation

Validated `ops_perf_results_2026_09_11_15_38_59.csv.gz` independently using Python CSV parsing.

Exactly one MLA_START/MLA_END region contains 192 device operations: 24 identical operation identities on each of eight devices. Every kernel duration is finite and positive.

| Category | Device-max operation sum (ms) | Share |
|---|---:|---:|
| SDPA | 7.195124 | 83.81% |
| Matmul | 0.737823 | 8.59% |
| Other | 0.652105 | 7.60% |
| Total | 8.585052 | 100% |

The independent sum matches the wrapper exactly. No separately named TP collective occurs; ring communication is included within RingJointSDPA.

One instrumented first forward, no warmup/repeats; not a steady-state benchmark.
Kernel-counter durations exclude host compilation gaps, but cold execution and instrumentation may affect device timing.
Sum of per-operation device maxima is an accumulated operation budget, not wall-clock MLA latency or TTFT.
RingJointSDPA includes distributed ring communication; CCL category zero does not mean no communication.
Reference=None: execution and coverage validated, numerical accuracy not independently compared.
