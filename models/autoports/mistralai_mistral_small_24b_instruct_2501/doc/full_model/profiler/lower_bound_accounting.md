# Full-model decode lower-bound accounting

The inherited final batch-1 optimized TP4 decoder result is 0.414822 ms per
layer over 100 warmed traced replays. Forty identical layers therefore imply a
16.592880 ms/token decoder-stack lower bound.

The final-source free-running full model measured 55.93636 token/s/user, or
17.877457 ms/token, across 53 post-capture model+sampler trace replays. The
full-model-only gap over the decoder-stack lower bound is therefore 1.284577
ms/token (7.18% of the measured interval). This gap contains embedding,
terminal norm, the four-piece BF16 TP LM head, split sampling, device position
advance, trace orchestration, and the caller's single-token observation.

The required reduced profile uses one exact optimized layer, exact embedding,
final norm, all four LM-head pieces, Sampling1D, `tt_out_tok` feedback, signed
and unsigned device position advance, and separate model/sampler traces. Ten
replays contain 13,785.89 us of merged device-operation time, or 1,378.589 us
per replay. `tt-perf-report` attributes:

| Reduced full terminal category | Share | Ten-replay device time |
| --- | ---: | ---: |
| matmul | 69.21% | 9,540.94 us |
| TopK | 9.06% | 1,249.63 us |
| Sampling | 1.97% | 271.77 us |
| ManualSeed | 1.25% | 172.04 us |
| async all-reduce | 2.74% | 378.18 us |
| all-gather, including embedding and sampler gathers | 2.57% | 354.18 us |
| final/layer norms | 1.90% | 261.86 us |

The explicit TopK + Sampling + ManualSeed operations total 169.344 us/replay,
12.28% of this one-layer terminal probe; they do not dominate. On the actual
40-layer token-out path, the final-source separately measured complete
Sampling1D call is 0.314182 ms, 1.76% of the 17.877457 ms interval.
SamplingGenerator force-argmax is semantically identical but 1.266483 ms and
is rejected.

The observed 1.284577 ms full-model-only gap is consistent with the reduced
terminal envelope and remains small relative to the layer stack. The BF16
four-piece LM head is the largest added terminal compute in the reduced report;
it is already DRAM-sharded and its passing block-four configuration is retained
because block eight exceeds physical L1 circular-buffer capacity.

Evidence: `final/perf_report.csv`, `final/perf_summary.csv.csv`,
`final/perf_report_table.txt`, `final/ops.csv.gz`, and
`../../logs/reduced_terminal_trace_profiler.log`.
