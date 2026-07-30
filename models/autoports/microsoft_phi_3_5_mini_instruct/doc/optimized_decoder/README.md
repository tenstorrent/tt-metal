# Phi-3.5 Mini optimized decoder

This stage optimizes the single-device dense decoder while preserving the
functional decoder's paged prefill/decode, LongRoPE, trace, determinism, and
131072-token context contracts. The primary target is traced batch-1 decode;
batch-32 decode must not regress.

## Current baseline

Fresh baseline run on 2026-07-30, one Blackhole p300c:

| Path | Workload | Warmed host latency |
| --- | --- | ---: |
| Prefill | batch 1, sequence 128 | 2.007797 ms |
| Traced decode | batch 1, context 128 | 1.052189 ms mean / 1.050918 ms min |
| Traced decode | batch 32, context 128 | 1.215247 ms mean / 1.211395 ms min |

Command:

```bash
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/functional_decoder_perf.py
```

The functional profiler shows BF16/HiFi2 weights for all four material
projection roles. At batch 1 their representative device rows are QKV 154 us,
attention output 55 us, packed gate/up 268 us, and down 182 us. Both RMSNorms
take about 44 us on one core. The down projection is marked `SLOW`.

## Operation-topology audit

| Current sequence / boundary | Candidate | Action | Evidence |
| --- | --- | --- | --- |
| input RMSNorm -> packed QKV | Sharded norm/residual chain; BFP8/BFP4 and LoFi/HiFi2 projection policies; DRAM-sharded and advisor 1D matmuls | Required sweep | Norm is one-core; QKV is 154 us and DRAM-bound |
| packed QKV -> on-device head creation | Split Q/K/V | Keep packed unless a precision-locked split candidate wins | Functional path already removes repeated same-input Q/K/V matmuls |
| explicit 96-wide rotate-half -> paged cache update -> paged SDPA | Native rotary/composite replacement; BFP8 cache; explicit SDPA config | Test legal composite/config variants; preserve explicit path if head_dim=96 remains unsupported | Functional explicit RoPE creates tilize/untilize/permute traffic; paged SDPA itself is already the correct composite |
| concat heads -> output projection -> residual | L1 width-sharded output/residual; DRAM-sharded output matmul | Required sweep | Output projection is 55 us; current residual returns to DRAM interleaved |
| post-attention RMSNorm -> packed gate/up | Sharded norm; BFP4/LoFi; advisor 1D and DRAM-sharded geometries | Required sweep | Norm is one-core; gate/up is 268 us and largest decode matmul |
| packed gate/up -> two slices -> SiLU -> multiply | Separate gate/up with fused activation versus packed path | Compare whole MLP under the same precision/layout | Packed path avoids a repeated same-input read but pays split/unary/binary rows |
| activated MLP -> down projection -> residual | BFP4/LoFi and BFP8/LoFi/HiFi2; wider working shard and DRAM-sharded geometries | Required sweep | Down is 182 us, 280 GB/s, marked `SLOW` |
| runtime host/device boundaries | Remove any conversion/fallback introduced by optimization | Keep runtime TTNN-only | Functional runtime has 0 host ops; optimized tests will statically audit the optimized class |

There are no collectives in this single-device stage, so CCL fusion,
persistent CCL buffers, and multi-device residual families are not applicable.
QKV and gate/up are already packed at load time. Their split alternatives are
still measured where required by the optimization checklist.

## Final selected path

The optimized class owns prefill, decode, and dispatch. Prefill uses
phase-specific BFP8 attention/output/down weights, BFP4 packed gate/up, and
large multicore DRAM-interleaved matmuls. Decode uses BFP4/LoFi
DRAM-width-sharded weights for all four packed projections, an 8-core
width-sharded residual/RMSNorm input chain, paged BFP8 KV cache, and traced
paged SDPA. There is no functional runtime fallback.

| Path | Functional | Optimized | Change |
| --- | ---: | ---: | ---: |
| Prefill B1/S128 | 1.791829 ms | 1.654356 ms | -7.7% |
| Traced decode B1/context128 | 1.051599 ms | 0.489858 ms | -53.4% |
| Traced decode B32/context128 | 1.216610 ms | 0.631896 ms | -48.1% |

Real-weight prefill PCC is 0.999274, BFP8-cache decode PCC is 0.998946,
advertised-context decode PCC is 0.998839, and LongRoPE traced PCC is
0.999992, all above the functional 0.995 bar.

## Artifacts

- `work_log.md`: commands, candidate decisions, correctness, performance, and
  checklist status.
- `shard_advise/report.json` and `shard_advise/final_ir.mlir`: mandatory
  compiler advisor output captured on the rewritten optimized block.
- `tracy_final/ops.csv`, signpost CSVs/summaries, and PNGs: exact-final
  profiler evidence.
- `watcher_final/generated/watcher/watcher.log`: clean watcher evidence.

Detailed commands, candidate tables, advisor decisions, PCC, profiler
conclusions, and the completed optimize checklist are in `work_log.md`.
