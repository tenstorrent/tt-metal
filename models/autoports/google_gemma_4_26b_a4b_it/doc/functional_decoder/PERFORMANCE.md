# Functional layer performance

The workload is 4096 input rows followed by 128 successive decode rows,
batch 1 and one request, on one Blackhole ASIC. The harness supplies layer
inputs, not full-model autoregressive token generation. Device profiling and
watcher runs are separate. Setup, HF reference, uploads and output comparisons
are outside each measured forward/replay. Input refreshes between decode replays
occur inside the enclosing signpost region, but outside the measured device
windows.


| Layer type | Prefill device us | Traced decode device us (mean128) | Useful prefill FLOPs % | Estimated decode DRAM % |
| --- | ---: | ---: | ---: | ---: |
| sliding_attention | 4998536.82 | 9811.07 | 0.106427 | 24.109186 |
| full_attention | 5009442.54 | 10894.37 | 0.146257 | 27.049574 |

Each `tracy/<kind>/whole_layer.json` records the first native firmware start to
the last native firmware end of the complete prefill pass and each traced
replay. The decode headline is the mean of all 128 replay windows. All layer
operations and device gaps within each window remain in the denominator.
Host/input-copy gaps between replays are excluded. These are device durations,
not TTFT or end-to-end generation latency. Firmware cycles are converted at
1.35 GHz using the recorded duration/cycle ratio. Summed kernel time is retained
as a diagnostic; it is not used as the whole-layer denominator.

`tests/summarize_perf.py` implements the calculation and writes all 128 replay
windows to `whole_layer.replays.csv`. `prefill_perf_report.txt` is the rendered
full prefill table. `decode_perf_report.txt` is the rendered first measured
replay, made from `decode_example_ops.csv`; the complete 128-replay rendered
table remains in `decode_all_replays_perf_report.txt`. Full filtered report
CSVs, stacked summaries and original `ops.csv` remain locally available.
`report_commands.json` records the exact tt-perf-report commands and exit codes.
Large captures and full tables are intentionally not committed; compact human
reports, summaries, replay windows and provenance are checkpointed.

Useful FLOPs count logical Q/K/V and output projections, shared MLP, top-8
active MoE experts, router and causal QK/PV attention. The full-attention tied
K/V projection is counted once as useful work, even though the implementation
packs duplicate K/V. Padding, masked attention, extra experts actually computed
during prefill, scalar normalization and transcendental work are omitted from
the useful numerator; all their device time remains included. Sparse prefill
currently computes all 128 experts per tile; this is a functional baseline.
The separate per-op utilization columns in tt-perf-report do not define the
whole-layer roofline and are never averaged into it.

Estimated decode DRAM bytes sum one padded operand read and output write per
native operation. Sparse weights use the actual recorded nnz fraction. Paged
cache updates read and write one 32-token page across KV heads. Embedding and
slice inputs count selected rows; whole-cache layout conversions before
embedding count their full pool. Additional per-core rereads, metadata and
profiler traffic are excluded. This is a stated traffic model, not measured
DRAM transactions. Fresh profiles disable op-info caching because this runtime
can cache invocation shapes by a volume-insensitive program hash.

The common theoretical peak basis is one ASIC with 120 physical Tensix cores,
1.35 GHz and 4096 FLOP/core/cycle divided by four for BF16 HiFi4:
165.888 TFLOP/s and 512 GB/s DRAM. The measured runtime exposes 110 worker cores;
the denominator uses the theoretical participating ASIC peak. The layer mixes
BF16/FP32, HiFi4 matmul and SFPU work; this common useful-work reference is not
an assertion of homogeneous FPU utilization. Weights and caches are BF16.
No percentage is clamped.

Hardware basis: [P300 specifications](https://docs.tenstorrent.com/aibs/blackhole/p300.html),
[per-ASIC bandwidth](https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html),
and the installed tt-perf-report 1.3.0 Blackhole 4096-FLOP/cycle/HiFi4 model.

The prior `whole_layer_stale_metadata.json` is diagnostic only: its DRAM estimate
was withdrawn because cached shapes and whole-pool cache-update accounting
inflated traffic. Final values come only from the fresh uncached profiles.
