# Shard-Advice Usage & DRAM-Sharding Analysis — mistralai/Mistral-Small-24B-Instruct-2501

Model: `mistralai/Mistral-Small-24B-Instruct-2501` — optimized_decoder (single decode layer, dense).

This report classifies how the optimize stage used the `$shard-advise` output, and analyses which matmuls
ship DRAM-sharded. Sources: `shard_advise/report.json` (advisor input), `tt/optimized_decoder.py` (final
output), `doc/optimized_decoder/work_log.md` + `README.md` + `evidence/README.md`, and the optimize codex run
log.

## Classification rubric

- **A — used-as-is:** advisor config shipped unchanged.
- **D — used-but-modified:** advisor config taken as starting candidate; the shipped config is a re-tuned
  version of it (same op, altered params/layout).
- **B — not-used:** advisor config not shipped and nothing derived from it. Reason tag: *incorrect* /
  *superseded-on-perf* (implemented + measured correct, but an unrelated faster option shipped) /
  *algorithm-change*.
- **C — could-not-use:** advisor emitted no valid config / hard technical blocker.

## Shard-advice usage

Advisor emitted **19** per-op recommendations (`ops[]`, indices 0–18; `final_choices`=20, `total_ops`=23),
plus 15 reshards, 1 spill, and **1 unfixable op**.

| Advice item(s) | Advisor recommendation | Verdict | Reason | Evidence (source) |
|---|---|---|---|---|
| reshape ×2 (idx 0, 18) | l1/interleaved boundary | A | — | Batch/token flatten shipped; work_log "decode reshapes to `[1,1,32,H]`". |
| RMSNorm ×2 (idx 1, 12) | l1/block_sharded 11-core | A | — | `_advisor_decode_norm_configs` "Reproduce the advisor's 11-core block-sharded decode RMSNorm choice"; `use_advisor_decode_layout=True` (default). OPT-015: "applied exactly in principle; both final rows 8–9 us". |
| qkv_heads / RoPE ×2 / slice_static / repeat / SDPA-decode (idx 3,4,5,6,7,8) | height-shard / DRAM-interleaved | A | — | Layouts match final code; "direct QKV/concat edges retained". |
| residual add ×2 (idx 11, 17) | l1/width_sharded | A | — | OPT-015: "keep attention and MLP residual adds width-sharded → applied; traced decode improved 1.529 → 1.377 ms". |
| SiLU·multiply (idx 15) | l1/width_sharded | A | — | Fused SiLU into multiply, kept width-sharded; "80x64 geometry improved 1.826 → 1.723 ms". |
| QKV matmul (idx 2) | L1 width-sharded 1D `@11x9` | B | superseded-on-perf | Exact advisor 1D implemented behind non-default `use_advisor_1d_matmuls=False`; PCC 0.9998337, measured **1.788 ms vs 1.288 ms** DRAM-sharded default. Shipped is DRAM-sharded BFP4. |
| O matmul (idx 10) | L1 width-sharded 1D `@11x8` | B | superseded-on-perf | Same advisor-1D seed rejected on latency; shipped DRAM-sharded BFP4 block-16. |
| gate matmul (idx 13) | L1 width-sharded 1D `@11x10` | B | superseded-on-perf | Separate DRAM-sharded MLP wins decode by 569 us (packed 1.857 vs 1.288); advisor 1D rejected. |
| up matmul (idx 14) | L1 width-sharded 1D `@11x10` | B | superseded-on-perf | Same; shipped DRAM-sharded BFP4. |
| down matmul (idx 16) | L1 width-sharded 1D `@11x8` | B | superseded-on-perf | Same advisor-1D seed rejected; shipped DRAM-sharded BFP4 block-16. |
| nlp_concat_heads_decode (idx 9) | dram/interleaved | C | — | `unfixable_ops`: `TT_FATAL ... Input tensor must be sharded`; advisor emitted no valid config. Runtime repairs the SDPA result to a height shard and does one direct L1 reshard into O. |

**Counts: A = 13, B = 5, C = 1, D = 0 (of N = 19).**
**B reason breakdown:** superseded-on-perf **5**, incorrect 0, algorithm-change 0.

Note: unlike the Falcon models, Mistral's advisor **norm + residual chain is the shipped default**
(`use_advisor_decode_layout=True`), so those items are A, not B — the advisor's coherent L1 residual chain
was measured faster (1.529 → 1.377 ms) and kept. Only the advisor's matmul *program_configs* (the exact 1D
width-sharded family) lost to DRAM-sharding and are retained solely behind the non-default
`use_advisor_1d_matmuls` flag (with `evidence/advisor_1d_pcc.xml` / `advisor_1d_perf.xml`).

## DRAM-sharding per matmul

All five in-scope matmuls ship **DRAM-sharded BFP4/LoFi**
(`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, weights in WIDTH_SHARDED / `BufferType.DRAM`;
constructor defaults `use_dram_sharded_attention=True`, `use_dram_sharded_mlp=True`). lm-head is out of scope.

| Matmul | Final layout | DRAM-sharded? | Versions tested (measured result) | Why this choice |
|---|---|---|---|---|
| QKV (packed, 32×5120×6144) | DRAM-sharded BFP4/LoFi, block 16 | **Yes** | BF16 interleaved baseline (5.221 ms QKV alone); BFP8 vs BFP4; block 2/4/8/16; exact advisor L1 1D `@11x9` (1.788 ms decode, rejected) | DRAM-sharded BFP4 wins; final row 59 us @ 268 GB/s |
| O (32×4096×5120) | DRAM-sharded BFP4/LoFi, block 16 | **Yes** | block 2/4/8/16 sweep; direct concat-to-O reshard; advisor 1D `@11x8` (rejected) | block-16 kept; one L1 reshard replaces L1→DRAM→L1; 41 us @ 258 GB/s |
| gate (32×5120×32768) | DRAM-sharded BFP4/LoFi, block 4 | **Yes** | packed gate+up (1.857 ms) vs separate DRAM-sharded (1.288 ms); advisor separate 1D `@11x10` (rejected) | separate DRAM-sharded wins decode by 569 us; 292 us @ 287 GB/s |
| up (32×5120×32768) | DRAM-sharded BFP4/LoFi, block 4 | **Yes** | as gate | 294 us @ 285 GB/s |
| down (32×32768×5120) | DRAM-sharded BFP4/LoFi, block 16 | **Yes** | BFP4/LoFi geometry/block sweep; advisor 1D `@11x8` (rejected) | 286 us @ 293 GB/s |

Precision sweep (whole-layer, real weights): BFP4/LoFi **1.288 ms** (selected) beats BFP8 HiFi2 (2.420 ms),
BFP8 LoFi (1.723 ms), BFP4 HiFi2 (2.228 ms); all PCC ≥ 0.9998. Norms, residual adds and the SiLU·multiply stay
**L1 width/block-sharded** (no large streamed weight to amortize; keeps the residual chain coherent and
minimizes reshards).

## Summary

Mistral-Small-24B used **13 of 19** advisor items verbatim (A) — the whole topology/movement scaffolding plus,
notably, the advisor's 11-core block-sharded RMSNorm and width-sharded residual chain (shipped by default and
measured faster). The **5 matmul program_configs** are B/superseded-on-perf: the advisor's exact L1 width-sharded
1D family was implemented and PCC-verified but lost to DRAM-sharded BFP4 (1.788 ms vs 1.288 ms) and survives only
behind a non-default flag. **1** item (`nlp_concat_heads_decode`) is C — the advisor itself emitted no valid
config. Nothing was rejected as incorrect. **All 5 matmuls ship DRAM-sharded** because the MLP dominates the
layer (>89% of the baseline) and decode is DRAM-bandwidth-bound (~42% roofline), so streaming BFP4 weights from
DRAM beats holding them L1 width-sharded — the same pattern as the dense Falcon models.
