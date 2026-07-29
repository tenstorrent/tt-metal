# Phi-3.5 Mini optimized decoder

This stage implements the single-device optimized TTNN layer in
`tt/optimized_decoder.py`; tests import `OptimizedDecoder` directly and do not
route through `FunctionalDecoder`.

## Result

The selected path keeps packed QKV and separate gate/up projections, BF16
activations and norms, BFP4/LoFi attention and MLP decode weights, BFP8 paged
KV cache, a width-sharded L1 residual stream, sharded norms, DRAM-sharded
decode matmuls, and explicit paged-SDPA configuration. Prefill uses BFP4
attention-projection weights, BFP8 MLP weights, and large 2D matmul configs
through 4096 rows; larger non-aligned
logical lengths use the functional stage's proven bounded TTNN program and
chunked-attention contract.

Current-run warmed latency on one Blackhole p300c:

| Workload | Functional | Optimized | Change |
| --- | ---: | ---: | ---: |
| Prefill, batch 1, seq 128 | 1.917 ms | 1.825 ms | -4.8% |
| Traced decode, batch 1, context 128 | 1.051 ms | 0.522 ms | -50.3% |
| Traced decode, batch 32, context 128 | 1.217 ms | 0.723 ms | -40.6% |
| Prefill, batch 32, seq 32 | 26.165 ms | 20.277 ms | -22.5% |

The primary batch-1 decode target beats the best current functional baseline,
and batch 32 does not regress. The prefill regression is retained because this
decoder-stage objective prioritizes decode and the selected cumulative path
preserves correctness and capability.

## Operation-topology audit

| Area | Functional/current topology | Candidate | Action and evidence |
| --- | --- | --- | --- |
| Attention projections | One packed QKV matmul | Split Q/K/V | Kept packed: adapted split path (three matmuls, three sharded-to-interleaved rows, concat) passed at 0.570 ms; packed QKV with selected split MLP passed at 0.550 ms. |
| MLP input projections | One packed gate/up matmul plus slices | Separate gate and up | Selected separate: three 200-replay runs were 0.5500-0.5505 ms versus packed 0.5549-0.5557 ms. The final profile has two `3072x8192` rows and fused SiLU-multiply. |
| Residual/norm | Functional DRAM interleaved, one-core norms | Width-sharded L1 chain | Kept: both norms are 16-core 7 us rows and residual adds stay sharded. |
| Decode matmuls | BF16/interleaved baseline | DRAM-sharded BFP4/LoFi | Kept: profiler proves all five dominant rows are `BF16 x BFP4`, LoFi, on 12 DRAM cores. |
| Attention | Paged BF16 cache and paged SDPA | BFP8 cache plus explicit SDPA config | Kept: SDPA is 19 us at batch 1; real-weight cache-consuming traced PCC is 0.999806. |
| Head/RoPE boundary | Phi head width 96 requires explicit rotate-half | Keep heads sharded throughout | Rejected with exact contract: 48-wide half slicing is not tile-shard legal. Narrow DRAM crossings remain around RoPE/head helpers. |
| Large prefill | Static large-M config | Default bounded program above 4096 | Kept bounded path: static config requested 13,041,408 B CB vs 1,572,864 B L1 at seq 32769; adapted default passes 32769 and 131071. |
| DRAM geometry | 32-core working shard, blocks 3/8 | 8/16-core phase shards and blocks through 16 | Selected 16 cores: three 200-replay runs were 0.5545-0.5549 ms vs 0.5565-0.5570 ms at 32 cores. An 8-core/block-16 candidate exceeded L1. |

There are no collectives in this single-device stage. The final decode profile
contains 63 device ops and zero host ops. Remaining movement is localized to
Phi's 96-wide head/RoPE/cache helper boundaries: one initial
interleaved-to-sharded conversion, QKV head conversion, cache-update layouts,
one concat-head reshard, and the MLP working-shard transition.

## Correctness and capability

The acceptance floor is PCC >= 0.995.

- Real-weight batch-1 prefill/decode PCC at sequence/context 128:
  0.9999450 / 0.9998056.
- Real-weight batch-32 prefill/decode PCC: 0.9998948 / 0.9995409.
- Non-aligned sequence 33 prefill/decode PCC:
  0.9999967 / 0.9999982.
- Repeated fresh-cache runs are deterministic.
- Non-aligned prefill executes at 32769 and 131071, exact-limit prefill at
  131072, and decode executes at logical
  context 131072. `doc/context_contract.json` remains at the advertised
  131072-token capability and records the BFP8 cache.
- The official Hugging Face snapshot `2fe192450127e6a83f7441aef6e3ca586c338b77`
  supplies the real layer-0 tensors.

## Profiler and accounting

### Mandatory shard-advisor seed

The required fresh advisor capture is saved under `shard_advise/`.
`report.json` records 38 ops, 35 final choices, one spill, and
`dram_sharding.dram_sharded_considered=5` with all five dense linears advised
as DRAM-sharded. `final_ir.mlir` is authoritative: it recommends
8-bank DRAM-sharded inputs/weights, block 12 for the four hidden-input
projections, block 32 for down, and broad 96/86-core L1 outputs.

The final path applies the advised DRAM-sharded matmul family and an L1
residual/norm/MLP chain. It rejects the exact 8-core/block-32 geometry after
real-weight whole-layer A/B at both required batches:

| Candidate | Batch-1 traced decode | Batch-32 traced decode | PCC |
| --- | ---: | ---: | ---: |
| Advisor seed, 8 cores, blocks 12/32 | 555.650 us | 756.075 us | 0.999778 / 0.999530 |
| Final default, 16 cores, blocks 6/16 | 521.679 us | 722.930 us | 0.999790 / 0.999541 |

The advisor's large output grids are likewise not kept: the DRAM-sharded
factory computes its legal round-robin output grid, and restoring the
rectangular 16-core residual grid is faster as part of the complete traced
layer. Exact commands and the two first-attempt capture repairs are in
`work_log.md`.

The post-advisor final-default reports are
`tracy/final/decode_b1_shard_advise.txt`,
`decode_b32_shard_advise.txt`, and `prefill_b1_shard_advise.txt`; all keep
advice enabled. Batch-1 decode is 521.679 us end-to-end in the unprofiled
same-environment default A/B; the paired final profiler run is 499 us device
time and 566.429 us end-to-end. The BFP4 decode
weights move approximately 30 MiB per layer invocation; at the profiler's
512 GB/s Blackhole peak this gives a coarse 61 us weight-only roofline.
Cache reads are context-dependent and small at context 128. The 499 us device
time is dominated by many small Phi head/RoPE/cache operations plus the five
matmuls, not weight traffic alone.

Dominant matmul rows:

| Role | Shape | Policy | `in0_block_w` | Device time |
| --- | --- | --- | ---: | ---: |
| Packed QKV | 32x3072x9216 | BFP4/LoFi, DRAM sharded | 6 | 55 us |
| Output | 32x3072x3072 | BFP4/LoFi, DRAM sharded | 6 | 22 us |
| Gate | 32x3072x8192 | BFP4/LoFi, DRAM sharded | 6 | 50 us |
| Up | 32x3072x8192 | BFP4/LoFi, DRAM sharded | 6 | 50 us |
| Down | 32x8192x3072 | BFP4/LoFi, DRAM sharded | 16 | 48 us |

The report labels these `SLOW` because DRAM utilization is 42-53% and the
DRAM-sharded factory does not expose output-subblock fields. The adapted
candidate matrix measured BFP4/LoFi at 0.557 ms, BFP4/HiFi2 at 0.729 ms,
BFP8/LoFi at 0.583 ms, and BFP8/HiFi2 at 0.736 ms using real weights.
BFP4/LoFi is the fastest passing
policy. No first API error is used as the rejection.

TTNN logs a memory-config substitution for DRAM-sharded matmul outputs: both
requested and computed layouts have the same width shard shape and 16 cores,
but the factory chooses a round-robin non-rectangular grid while the residual
and sharded norms require a rectangular 8x2 grid. Passing the computed grid
directly was tested and fails RMSNorm validation because its 16 cores occupy a
22-core bounding box. The measured report shows no hidden host op; explicit
reshard/conversion rows account for boundary changes. The warning is therefore
an op-factory grid-identity limitation, not an untracked topology change.

The Tracy consoles report profiler buffers full after the measured tests and
some later markers dropped. The signposted decode window itself parses as one
complete 63-op replay, and independent unprofiled 200-replay logs reproduce
the selected latency. Profiler row conclusions are limited to that complete
window; later dropped markers are not used as evidence.

## Validation

Watcher and profiler were separate runs. `watcher_shard_advise_console.log`
records five optimized-path tests passing under `TT_METAL_WATCHER=10`, including batch
1, batch 32, non-aligned, repeat, trace, and full-context decode checks.
Nanobind reference-leak diagnostics occur during Python teardown after the
tests pass, the watcher reports no kernel error, and device close succeeds;
they are binding teardown noise rather than a decoder/watcher failure.
Commands and additional evidence are in `work_log.md`.
