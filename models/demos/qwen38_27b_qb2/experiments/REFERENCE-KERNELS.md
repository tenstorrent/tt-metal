# Reference kernels for long-context Qwen optimization

Reviewed 2026-10-07 against the native `a08819ddbe23077f8037d3802303939064868ff6`
runtime. These are experiment priorities, not established model speedups.

## DRAM reader scheduling

[Tenstorrent's DRAM report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md)
uses bank-local readers, routing-aware placement, separate virtual channels and
transaction-tagged outstanding reads. Its reported 92% result is for Wormhole
and Grayskull, not this Blackhole allocation. The transferable experiment is to
keep the next block in flight while waiting for the current block.

Exact local references:

- `tests/tt_metal/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/kernels/reader_dram.cpp`
- `tests/tt_metal/tt_metal/perf_microbenchmark/10_dram_read_remote_cb_sync/`
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp`

The matmul factory uses architecture-specific preferred NoCs and optimal
bank-to-worker assignment. Reuse these helpers rather than hard-coding the
Wormhole example's coordinates. Our GDN candidate instead assigns work in
row-major core order and accesses interleaved state. Measure bank placement and
NoC routing before concluding that more active cores improve bandwidth.

## Full attention at 128K–256K

[Flash-Decoding](https://pytorch.org/blog/flash-decoding/)
splits the KV sequence into independently computed partial attention results,
then combines them with a numerically stable reduction. This increases
parallelism for small decode batches.

Our pinned `sdpa_decode_program_factory.cpp` already divides sequence work among
cores per KV head and allocates double-buffered K/V circular buffers. Adding
generic split-KV or double buffering is therefore not a new optimization.
`dataflow_common.hpp::read_k/read_v` still issues a completion barrier before
publishing each chunk; tagged lookahead and chunk/core-budget tuning are the
specific candidates. Preserve page-table mapping, causal bounds, grouped-head
reuse and the final reduction.

[TT FlashAttention's pipeline](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashAttention.md)
is a reference for reader/compute/writer overlap and contiguous tile transfers.
[FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf) additionally
illustrates overlapping softmax with matrix operations; Hopper's TMA/WGMMA
instructions do not directly map to Tensix.

The first attention experiment failed its unchanged 2% relative-RMS limit:
8K/B1, native chunk 128, PCC 0.999652, relative RMS 0.0270035 on all four ranks.
It is not evidence of a page-table corruption. The next diagnostic must compare
default HiFi2/approximate math with explicit HiFi4/FP32 accumulation and accurate
exponentiation before tuning speed. The deployed model also omits an explicit
SDPA compute config; causality for any model-score difference is unproven.

## Recurrent state update

[FLA's fused recurrent GDN kernel](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py)
partitions value columns while retaining the whole key dimension per program.
It retains FP32 state locally through the recurrence and supports fused Q/K
normalization and gates. For ordinary one-token decode, cross-token state reuse
does not eliminate the need to retain every layer's state between calls.

Our standalone candidate follows the same independent value-column partition.
The hardware receipt `../galaxy-evidence/gdn-step-candidate-v3/candidate.json`
passes recurrence accuracy, allocation rebinding and long-run checks. Selected
latencies in microseconds (recurrence only, warm trace including dispatch):

| Batch | One partition | Two partitions | Four partitions |
|---:|---:|---:|---:|
| 1 | 66.67 | 39.18 | 26.01 |
| 8 | 83.53 | 85.80 | 87.63 |
| 16 | 154.43 | 152.69 | 140.10 |
| 64 | 453.86 | 432.80 | 462.27 |

Every case still misses the P1 latency target. The strong B1 gain and small
large-batch gains motivate different partition choices by batch and investigation
of per-item stalls. They do not isolate DRAM as the cause. Current input CBs hold
one work item; the reader cannot prefetch another state while that item remains
live. Prototype two-item buffering, then tagged lookahead, without adding a
second DRAM state pass. Also measure repeated FP32 operand reconfiguration and
pack/reload costs before redesigning state layout.

The two-item prototype has now passed all 30 geometry variants and the long-run
and trace-rebinding checks. At four partitions, B8/B16/B32/B64 latency decreases
by 25.4%/29.2%/31.0%/32.1% against the same partition count with one buffered
item. All still miss P1's target; full-model integration remains pending. See
`../galaxy-evidence/kernel-diagnostics-v1/README.md` for measurements and scope.

The explicit attention math comparison also completed. HiFi4/FP32 reduces
error but all four geometries still exceed the unchanged 2% relative-RMS gate;
accurate exponentiation produced the same outputs in this experiment. This
does not establish the cause of the model's GPQA gap.

## Experiment order and acceptance

1. Resolve attention numerical-baseline failure without loosening tolerances.
2. Establish a Blackhole read-only and read/write bandwidth baseline with the
   relevant tile sizes, bank layout and active-core counts.
3. Sweep attention chunk size and core budget at 128K/B1/B8 and near-256K/B4.
4. Compare GDN one/two-item buffers and 1/2/4 value partitions at B1/8/16/32/64.
5. Attribute reader/compute/writer stalls using compact profiler output, then
   integrate only candidates that retain accuracy and improve full-model TPOT.

Report useful bytes/time separately from actual DRAM counters and all serving
metrics. Preserve FP32 recurrent state, long-run accuracy, reference evals and
the 8K regression controls. No candidate is promoted by these notes.
