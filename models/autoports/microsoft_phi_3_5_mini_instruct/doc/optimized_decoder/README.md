# Phi-3.5 Mini optimized decoder

`tt/optimized_decoder.py` starts from the reviewed `FusedDecoder` and keeps its
packed QKV, packed gate/up, fused SiLU-multiply, paged SDPA, LongRoPE, page
table, trace, and non-aligned sequence contracts. Construction converts
attention weights to BFP8, gate/up to BFP4, and down to BFP8. Activations,
norms, outputs, and the selected KV cache remain BF16.

## Topology audit

| Current sequence | Candidate | Action and evidence |
| --- | --- | --- |
| norm -> packed QKV -> split heads | packed vs split Q/K/V | Kept packed; already one shared-input matmul. |
| Phi width-96 rotate-half -> cache update -> paged SDPA | fused RoPE / explicit SDPA config | Generic fused RoPE cannot express the 48-wide half rotation. Paged SDPA is already the composite op. |
| concat heads -> output projection -> residual | sharded residual chain | Measured with sharded RMSNorm/residual carry; flat at B1 and slower at B32, so rejected. |
| norm -> packed gate/up -> slices -> fused SiLU multiply -> down -> residual | packed vs split gate/up | Kept packed; it removes one shared-input matmul and the fused stage proved the elementwise form. |

There are no collectives: this is a one-device decoder. Runtime source contains
no Torch conversion or host fallback.

## Correctness

Real layer-0 weights were tested against the Torch oracle. Non-aligned prefill
PCCs are 0.999217 (31), 0.999286 (33), and 0.999257 (65). Decode PCC is
0.999149 at batch 1 and 0.999233 at batch 32, above the inherited 0.995 bar.
Repeated decode calls are bitwise deterministic. The cache/page/context shape
contract is unchanged, so `doc/context_contract.json` is unchanged.

## Performance

One Blackhole p300c, context 128, 100 warmed traced replay pairs:

| Batch | Fused baseline ms | Optimized ms | Change |
| ---: | ---: | ---: | ---: |
| 1 | 1.047274 | 0.667609 | -36.3% |
| 32 | 1.210779 | 0.830255 | -31.4% |

Warmed prefill improves from 1.597804 to 1.400795 ms at batch 1 and from
37.301012 to 30.285878 ms at batch 32 (`prefill_perf.log`).

Primary profiler evidence is `tracy/ops_final.csv` with advice-backed
`decode_b1_final.txt` and `decode_b32_final.txt`. Runtime rows prove BFP8
QKV/output/down and BFP4 gate/up reached the optimized calls. The selected
16-shard, `in0_block_w=8`, BFP8/LoFi DRAM-sharded down path measures 57-58 us
in the final report, versus 182 us for the paired fused baseline. See
`AUTOFIX.md` for the full geometry matrix.

The first 100-pair profiler collection overflowed marker buffers and was
rejected. The two-pair rerun completed without dropped-marker warnings.
Watcher ran separately with `TT_METAL_WATCHER=10`; batch-1 and batch-32
real-weight decode passed cleanly.

## Candidate decisions

- QKV BFP8, output BFP8, gate/up BFP4, and down BFP8 were crossed with
  16/32-core geometry, HiFi2/LoFi, and legal K blocks including 3 and 6.
- QKV/output/gate-up won isolated rows but changed cumulative whole-layer
  timing by less than 0.5 us and sometimes regressed, so automatic configs
  remain selected. The down candidate won cumulatively and is enabled.
- A sharded RMSNorm/residual-chain candidate was flat at B1 and regressed B32.
- BF16 paged KV cache remains selected. The unsupported reduced-cache policy
  surface was removed.

The selected layer weights occupy about 87 MiB, a weight-only lower bound near
0.17 ms at 512 GB/s. Final B1 e2e is 0.668 ms including cache, RoPE, norms,
layout work, and dispatch. `tracy/ops_final.csv` and the final B1/B32 reports
show the optimized down row at 57-58 us, 12 active cores, LoFi BF16 x BFP8,
and 85-87% modeled DRAM utilization. The reports deliberately contain paired
fused and optimized calls; optimized rows are the BFP8/BFP4 calls after each
BF16 fused group.

## Optimize checklist

- Traced decode, paged cache, determinism, non-aligned prefill, real-weight
  PCC, batch 1/32, watcher, and runtime-fallback audits: complete.
- Topology, dtype/fidelity, geometry, DRAM sharding, composite SDPA, packed
  projections, and residual/norm candidates: complete; see `AUTOFIX.md`.
- Final default reproduction, runtime dtype rows, warmed prefill/decode, and
  roofline/device/e2e accounting: complete in the final logs and reports.
- Multi-device CCL, MoE, LM head, sampling, and vLLM items: not applicable to
  this single-device decoder-layer stage.
