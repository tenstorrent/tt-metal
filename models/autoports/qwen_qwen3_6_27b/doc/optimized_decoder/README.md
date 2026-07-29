# Qwen3.6-27B optimized decoder

This stage adds an independent single-device `OptimizedDecoder` while
preserving the functional public contract, paged cache semantics, non-aligned
prefill lengths, both layer kinds, and batches 1 and 32.

## Selected path

- Packed Q/K/V and gate/up projections.
- BFP8 attention/cache, BFP4 gate/up/down, and LoFi kernels.
- DRAM-sharded QKV, output, gate/up, and down decode matmuls.
- Gate/up slices and fused SiLU multiply stay L1 width-sharded into down.
- TTNN SDPA decode and paged cache update; no runtime Torch conversion or
  host fallback.
- Phase-specific interleaved prefill weights.
- Checkpoint Q projection canonicalized from HF per-head
  `[q_head, gate_head]` to runtime `[all_q, all_gate]`.

Required advisor artifacts:

- batch 1: [`report.json`](shard_advise/report.json) and
  [`final_ir.mlir`](shard_advise/final_ir.mlir)
- batch 32: [`report.json`](shard_advise/batch32/report.json) and
  [`final_ir.mlir`](shard_advise/batch32/final_ir.mlir)

## Correctness

The acceptance bar is PCC 0.995.

| Check | Result |
|---|---:|
| full prefill b1/b32, seq33 | 0.999991 / 0.999990 |
| linear prefill b1 seq65 / b32 seq5 | 0.999996 / 0.999996 |
| full traced decode b1, steps 1/2 | 0.999004 / 0.999581 |
| full traced decode b32, steps 1/2 | 0.999585 / 0.999817 |
| linear traced decode b1, steps 1/2 | 0.999986 / 0.999987 |
| linear traced decode b32, steps 1/2 | 0.999967 / 0.999990 |
| official-weight full / linear decode | 0.997327 / 0.998821 |
| paged prefill/decode, permuted pages, position 65 | 0.999993 |

The new official-weight full-attention harness exposed an inherited loader
defect that diagonal synthetic weights masked. After repairing Q/gate ordering,
BFP8 attention passes at 0.997327. BFP4 attention reaches only 0.987364 and is
rejected. BFP4 down is 0.000471 below the precision-locked BFP8-down
official-weight trial (0.999292), remains above the bar, and wins latency.

## Performance

Times are warmed host medians. Decode is ten trace replays; prefill is five
synchronized iterations. Baselines are the best correct fused-stage results.

| Path | Fused baseline | Optimized | Change |
|---|---:|---:|---:|
| full decode b1 | 2.444631 ms | 1.168046 ms | -52.2% |
| full decode b32 | 2.650596 ms | 1.354522 ms | -48.9% |
| linear decode b1 | 3.130992 ms | 2.196347 ms | -29.9% |
| linear decode b32 | 21.475945 ms | 20.497374 ms | -4.6% |
| full prefill b1 seq33 | 3.150490 ms | 2.495279 ms | -20.8% |
| full prefill b32 seq33 | 68.639999 ms | 42.279556 ms | -38.4% |
| linear prefill b1 seq5 | 11.354058 ms | 10.371300 ms | -8.7% |
| linear prefill b32 seq5 | 312.701355 ms | 290.853394 ms | -7.0% |

The preserved final [`decode report`](profiler/decode_b1/tt_perf_report.txt)
shows QKV at 161 us/456 GB/s, attention output at 72 us/436 GB/s, packed
gate/up at 328-329 us, and down at 154 us. Its rows verify LoFi BFP8 attention
and LoFi BFP4 MLP weights. A separate
[`prefill report`](profiler/prefill_b1/tt_perf_report.txt) covers seq33.

Static tests assert optimized-path independence and the selected policy.
Changing-input ten-replay traces run with fallback exceptions enabled. Separate
`TT_METAL_WATCHER=10` batch-32 runs for both layer kinds are clean. Compact
correctness logs and the runner-derived
[`decode sweep`](evidence/decode_sweep.log) are under `evidence/`; watcher logs
are under `watcher/`.

The attention head/SDPA/residual boundaries still require interleaved restores;
the material MLP subchain no longer round-trips through DRAM. Further removal
requires coordinated TTNN composite/interface changes.
