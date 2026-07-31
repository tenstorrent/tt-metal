# Phi-3.5 Mini optimized decoder

`tt/optimized_decoder.py` starts from the reviewed `FusedDecoder` and keeps its
packed QKV, packed gate/up, fused SiLU-multiply, paged SDPA, LongRoPE, page
table, trace, and non-aligned sequence contracts. Construction converts
attention weights, gate/up, and down to BFP4. Activations,
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
PCCs are 0.998782 (31), 0.998704 (33), and 0.998763 (65). Decode PCC is
0.998787 at batch 1 and 0.998885 at batch 32, above the inherited 0.995 bar.
Direct paged prefill-to-decode PCC is 0.998744/0.998771, and five traced
replays are bitwise deterministic at both batches. A direct real-weight decode
at the advertised 131072-token context passes at 0.998697. The KV dtype and
cache shape are unchanged, so `doc/context_contract.json` is unchanged.

## Performance

One Blackhole p300c, context 128, 100 warmed traced replay pairs:

| Batch | Fused baseline ms | Optimized ms | Change |
| ---: | ---: | ---: | ---: |
| 1 | 1.047211 | 0.642772 | -38.6% |
| 32 | 1.211028 | 0.807652 | -33.3% |

Warmed prefill improves from 1.574821 to 1.351156 ms at batch 1 and from
37.313263 to 24.148464 ms at batch 32.

Primary shipped-policy profiler evidence is `tracy/decode_bfp4_final.txt`.
Runtime rows prove BFP4 QKV/output/gate-up/down reached
the optimized calls. The selected decode path uses 16 shards,
`in0_block_w=16`, and BFP4/LoFi. Prefill uses an explicit 64-core,
`in0_block_w=8` down projection; QKV and gate/up larger-block candidates at B32
were rejected by exact L1 limits. See `AUTOFIX.md` and
`AUTOFIX_BFP4_PRECISION_FRONTIER.md`.

The first 100-pair profiler collection overflowed marker buffers and was
rejected. The two-pair rerun completed without dropped-marker warnings.
The final full ten-test suite ran separately with `TT_METAL_WATCHER=10` and
passed cleanly, including context, cache transition, and trace replay.

## Candidate decisions

- QKV BFP8, output BFP8, gate/up BFP4, and down BFP8 were crossed with
  16/32-core geometry, HiFi2/LoFi, and legal K blocks including 3 and 6.
- QKV/output/gate-up won isolated rows but changed cumulative whole-layer
  timing by less than 0.5 us and sometimes regressed, so automatic configs
  remain selected. The down candidate won cumulatively and is enabled.
- A sharded RMSNorm/residual-chain candidate was flat at B1 and regressed B32.
- BF16 paged KV cache remains selected. The unsupported reduced-cache policy
  surface was removed.
- A final real-weight frontier crossed BFP4 QKV/output and BFP4 down both
  separately and cumulatively. Combined BFP4 is correct (minimum final-suite
  PCC 0.998697) and faster at both batches, so it supersedes the earlier BFP8
  attention/down checkpoint.

The selected layer weights occupy about 87 MiB, a weight-only lower bound near
0.17 ms at 512 GB/s. Final B1 e2e is 0.642772 ms including cache, RoPE, norms,
layout work, and dispatch. The final B1/B32 reports
show the optimized down row at 47.879-47.992 us, 12 active cores, block16
LoFi BF16 x BFP4, and 51.2-51.3% modeled DRAM utilization. The reports
deliberately contain paired fused and optimized calls; optimized rows are the
BFP4 calls after each BF16 fused group.

## Optimize checklist

- Traced decode, paged cache, determinism, non-aligned prefill, real-weight
  PCC, batch 1/32, watcher, and runtime-fallback audits: complete.
- Topology, dtype/fidelity, geometry, DRAM sharding, composite SDPA, packed
  projections, and residual/norm candidates: complete; see `AUTOFIX.md`.
- Final default reproduction, runtime dtype rows, warmed prefill/decode, and
  roofline/device/e2e accounting: complete in the final logs and reports.
- Prefill keeps automatic large 96/103-core QKV/output/gate-up configs and
  selects an explicit 32-core B1 / 64-core B32 block8 down config. The block4/8
  matrix, exact B32 L1 blockers, bounded final reports, and profiler-overflow
  recovery are recorded in `prefill_explicit_config_runner.txt` and
  `tracy/prefill_program_config_final.txt`.
- Multi-device CCL, MoE, LM head, sampling, and vLLM items: not applicable to
  this single-device decoder-layer stage.
