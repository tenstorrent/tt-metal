# Functional decoder performance

Warmed single-layer workload: 4096 input tokens followed by 128 sequential decode
positions, with batch 1/concurrency 1 as headline and B32/16/8 as agent-workload rows.
Real target weights, synthetic layer inputs, BF16 activations/cache/weights, HiFi4.
Host input/RoPE/position refresh occurs outside each captured layer forward. These
are layer timings, not full-model generation or serving throughput.

| Layer kind | Batch | Prefill device ms | Mean decode device ms | Useful prefill FLOPs / peak | Estimated decode DRAM / peak |
|---|---:|---:|---:|---:|---:|
| sliding_attention | 1 | 97.224 | 3.039 | 25.07% | 62.77% |
| sliding_attention | 32 | 3060.013 | 4.518 | 25.49% | 67.58% |
| sliding_attention | 16 | 1537.799 | 3.665 | 25.36% | 67.19% |
| sliding_attention | 8 | 768.890 | 3.249 | 25.36% | 66.69% |
| full_attention | 1 | 136.684 | 3.344 | 21.72% | 64.49% |
| full_attention | 32 | 4338.214 | 6.602 | 21.90% | 65.61% |
| full_attention | 16 | 2176.647 | 4.870 | 21.82% | 65.89% |
| full_attention | 8 | 1089.629 | 4.029 | 21.79% | 65.70% |

The target agent-workload estimate is 32 concurrent conversations, 32768 working-set
tokens per agent, typical 4096-token context (`agentic_serving.json`). This stage
profiles the typical context at all four batches; it does not claim 32 simultaneous
full 262144-token contexts or a full-model KV capacity result.

## Timing and calculation basis

`functional_decoder/profile_provenance.json` records exact collection, report and
calculation commands, CSV hashes and paths. Raw collections and both human-readable
and CSV tt-perf-report tables are local under
`functional_decoder/tracy/<layer_type>/batch<batch>/`. The rendered tables are
`prefill_perf_report.txt` and `decode_perf_report.txt`; corresponding CSVs have the
same stem. Their `Device Time` column is microseconds and describes individual op
kernel durations. These sums are diagnostic and are not the headline denominator.

Headline device time uses raw Tracy `DEVICE FW START CYCLE`/`DEVICE FW END CYCLE`
across every operation in the whole layer. Cycles are converted using the median
`DEVICE FW DURATION [ns] / (END-START)`, approximately 0.74074074ns/cycle. Prefill
spans all four chunks and their gaps. Decode is the mean of 128 complete captured
layer windows, including all internal gaps. Between-replay host refresh, profiler
drains and idle gaps are excluded; no host-wall latency is substituted.

Useful prefill FLOPs include all Q/K/V/O and gated-MLP projections plus causal
QK/PV work (window 1024 or full). Small elementwise work is omitted. Decode DRAM
is an explicit estimate: weights once, live KV once, logical intermediate reads/
writes, cache-tile read/write and page metadata. It excludes redundant core reads
and padding; percentages are estimates, not hardware counter measurements.
`../tests/summarize_profile.py` and each `performance_*.json` record formulas/components.

The whole participating ASIC denominator is 120 Tensix cores × 4096 ops/cycle × 1.35 GHz
/ 4 HiFi4 phases = 165.888 TFLOP/s and 512 GB/s DRAM. The profiler's 110 available worker
cores are not substituted for the whole-ASIC theoretical peak. Sources:
https://docs.tenstorrent.com/aibs/blackhole/p300.html and
https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html;
per-core fidelity convention matches tt-perf-report ArchitectureSpec.

The first undrained profile lost markers and is rejected, retained as
`functional_decoder/tracy/sliding_attention/batch1_overflow`. Accepted profiles drain outside layer
windows and require exactly 128 complete decode sessions with equal op counts.
PROGRAM CACHE HIT=False is stale serialized profiler metadata, explained with
source locations in `functional_decoder/work_log.md`. Optional WASM-copy and pandas mixed-type warnings
do not remove records; accepted CSV completeness is checked directly.
