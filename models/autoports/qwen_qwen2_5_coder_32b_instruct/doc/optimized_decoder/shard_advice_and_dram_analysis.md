# Shard-Advice Usage & DRAM-Sharding Analysis — optimized_decoder

**Model:** Qwen/Qwen2.5-Coder-32B-Instruct
**Artifact:** `models/autoports/qwen_qwen2_5_coder_32b_instruct/`

This report quantifies how much of the `$shard-advise` output the optimize stage adopted, and analyses
per-matmul weight placement (DRAM-sharded vs L1 width/block-sharded vs interleaved). Sources: the advisor
`shard_advise/report.json`, the shipped `tt/optimized_decoder.py`, and the `doc/optimized_decoder/*.md`
narrative (work_log, README, AUTOTRIAGE, validation).

## Classification rubric

- **A used-as-is** — advisor config shipped unchanged.
- **D used-but-modified** — advisor config taken as the starting candidate; the shipped config is a re-tuned
  derivative of it (same op, altered params/layout).
- **B not-used** — advisor config not shipped and nothing derived from it. Reason tags: *incorrect*,
  *superseded-on-perf*, *algorithm-change*.
- **C could-not-use** — advisor emitted no valid config / hard technical blocker.

## Headline

This is the model where the shard-advisor was **most successful**: the advisor's exact packed decode capture
(the `advisor_packed_bfp8_hifi2_1d` profile) became the first candidate, passed PCC, and **won the latency
sweep** — it is the shipped batch-32 default. From `work_log.md`: *"The final report's exact spill and programs
became the first candidate, passed PCC, and won latency"* and *"final advisor packed family … 1.9410 ms …
selected for batch 32."*

## Shard-advice usage

Advisor capture: `total_ops=28`, `final_choices=25`, per-op recommendations in `ops[]` = **26**, `reshards=21`,
`spill.total_spills=1`, `unfixable_ops=1`. N = **26** classified items.

| Advice item / group (indices) | Advisor recommendation | Verdict | Reason | Evidence (source) |
|---|---|---|---|---|
| reshape boundaries (0, 25) | l1/interleaved | A | — | shipped verbatim as decode I/O boundaries |
| RMSNorm ×2 (1 in-norm, 18 post-attn) | l1 block_sharded 11-core (0,0)-(10,0) | A | — | work_log: "11-core block-sharded RMSNorm and phase-specific width reshards — Applied" |
| packed QKV linear (2) | l1 width_sharded 1D `matmul_multi_core_reuse_multi_cast_1d @11x7`, 75 cores | A | — | shipped `_advisor_final_program(grid=(11,7), per_core_n=3, out_subblock_w=3)`; work_log 168: "advisor interleaved family wins" |
| QKV bias add (3) | l1 width_sharded | A | — | width-sharded add applied (bias kept as separate device add for PCC 0.999995) |
| nlp_create_qkv_heads_decode (4) | l1 height_sharded | A | — | shipped height-sharded |
| slice_static ×6 (5,6,9,10,11,15,20,21) | dram/l1 interleaved | A | — | boundary/split ops shipped as advised (incl. on-device gate/up split) |
| rotary_embedding ×2 (7,8) | l1 height_sharded | A | — | shipped height-sharded |
| repeat (12) | dram/interleaved | A | — | shipped as advised (GQA KV expand) |
| SDPA-decode (13) | dram/interleaved | A | — | explicit composite decode SDPA program applied |
| nlp_concat_heads_decode (14) | dram/interleaved (no valid sharded config) | **C** | — | `unfixable_ops`: `TT_FATAL … Input tensor must be sharded`; dedicated head helper retained |
| output/O projection (16) | l1 width_sharded 1D `@11x8`, 80 cores | A | — | `_advisor_final_program(grid=(11,8), per_core_n=2)`; work_log 35: "Applied advisor 1D output projection into 80-core width-sharded L1" |
| attention residual add (17) | l1 width_sharded 80-core | A | — | work_log 331: 80-core width-sharded residual, add row 1.288 us, no DRAM round-trip |
| packed gate/up matmul (19) | l1 width_sharded 1D `@11x10`, 108 cores | A | — | `_advisor_final_program(grid=(11,10), per_core_n=16, out_subblock_w=8)`; work_log 170: "advisor wins" |
| SwiGLU multiply (22) | l1 width_sharded 108-core | A | — | work_log 229: "multiply on 108 cores and down input reshard to 87 cores — Applied exactly" |
| down projection (23) | l1 width_sharded 1D `@11x8`, 87-core in → 80-core out | A | — | `_advisor_final_program(grid=(11,8))`; work_log 171 |
| final residual add (24) | l1 width_sharded 80-core | A | — | work_log 332: 80-core width-sharded final add before model-visible return |

Advisor spill (spill packed gate/up output to interleaved L1 before the static-split consumers) was **applied
exactly** — work_log: *"Omitting this spill … produced synthetic decode PCC 0.000351; applying it restored
0.999839."*

### Counts

**A=25 · B=0 · C=1 · D=0 (of N=26).**
- B reason breakdown: incorrect 0, superseded-on-perf 0, algorithm-change 0.
- The single non-adopted item (C) is `nlp_concat_heads_decode`, where the advisor itself emitted no valid
  config (constraint query requires sharded input).
- D=0: the advisor's `in0_block_w=2` was independently sweep-verified (blocks 4/8/16 and non-power sizes were
  run and rejected on shard/K divisibility) and **retained unchanged** because it is both fastest and the
  exhaustive legal maximum — so this is A (verified-and-kept), not a re-tuned derivative.

## DRAM-sharding per matmul

The decoder carries a full DRAM-sharded matmul family (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`,
`_decode_matmul_program`) and it is the **non-default** batch-1 fallback, but the shipped **batch-32 representative
decode** uses the advisor's L1 width-sharded 1D programs (`MatmulMultiCoreReuseMultiCast1DProgramConfig`,
`weight_strategy="advisor_interleaved"`) — weights interleaved in DRAM, activations width-sharded in L1.

| Matmul | Final layout (batch-32 default) | DRAM-sharded? | Versions tested (+measured) | Why this choice |
|---|---|---|---|---|
| Packed QKV | L1 width-shard 1D @11x7, 75 cores, block 2, per_core_N=3, subblock 1x3, BFP8/HiFi2 | **No** | advisor-1D vs DRAM-sharded whole family 2.291–2.324 ms; block 4/8/16 trials | advisor interleaved wins (1.941 ms); input shard is 2 tiles so block 2 is maximal; BFP8-fused bias hurt PCC → separate add |
| O / output | L1 width-shard 1D @11x8, 80 cores, block 2, per_core_N=2, subblock 1x2 | **No** | same DRAM-sharded family comparison | advisor wins; sharded L1 output retained into residual/norm (no reshard) |
| Packed gate/up | L1 width-shard 1D @11x10, 108 cores, block 2, per_core_N=16, subblock 1x8 | **No** | DRAM 32-core split & 40-core packed families measured; blocks 4/8/16/10 run | advisor wins; dominant op (42.4% of decode); phase input shard exactly 2 tiles |
| down | L1 width-shard 1D @11x8, 87-core in → 80-core out, block 2, per_core_N=2 | **No** | blocks 4/8/16 fail shard divisibility; non-power 5/10 fail K=864 divisibility | advisor wins; legal block set is {1,2}, 2 is exhaustive max |

(gate_proj and up_proj are fused into the single packed gate/up matmul; the advisor captured them packed too.)

**0 of the in-scope matmuls are DRAM-sharded at the emitted batch-32 geometry.** Precision default is BFP8 /
HiFi2 (functional BF16 baseline 82.37 ms → final 1.941 ms traced decode, 42.4× faster). The DRAM-sharded
alternative is correct but slower — work_log: *"advice to try DRAM-sharded weight programs was addressed by the
32/40-core split/packed families (2.324/2.291 ms), both slower than final 1.941 ms."* Only the **batch-1**
fallback default (`packed_mlp_bfp8_hifi2_dram_gate40c`, 40-core) is DRAM-sharded, because the advisor capture
geometry is specifically batch 32.

## Summary

Qwen2.5-Coder-32B is the clearest shard-advisor success among the analysed models: **25 of 26** advice items
shipped as-is (96%), the lone exception being the `nlp_concat_heads_decode` op the advisor itself flagged
unfixable. Uniquely, the advisor's exact packed 1D layout family also **won the latency sweep** over the
DRAM-sharded candidates and is the shipped default — so despite being a dense 32B model, it ships **zero**
DRAM-sharded matmuls at its batch-32 target, unlike the Falcon models. Local search still ran (block sizes,
DRAM-sharded families, precision) but confirmed the advisor's choices rather than overriding them.
