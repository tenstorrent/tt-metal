# Shard-Advise Usage & DRAM-Sharding Analysis — meta-llama/Llama-3.1-70B-Instruct

Model id: `meta-llama/Llama-3.1-70B-Instruct` · autoport dir `meta_llama_llama_3_1_70b_instruct` ·
stage `optimized_decoder`.

**Device scope:** this stage optimizes a **single-device** dense decoder layer. The code is written against the
mesh API (`mesh_device`, `dram_grid_size()`, smoke-tested with `MeshShape(1,1)`), but README states it
"does not start multichip, full-model, generator, or vLLM work" — so there is **no tensor-parallel weight
sharding in this stage**; multichip is explicitly deferred. Runs on one Blackhole p300c.

Advisor capture: `report.json` → `total_ops=28`, `ops[]=27` recorded recommendations, `final_choices=25`,
spill pass ran with 1 spill, `unfixable_ops=1` (`nlp_concat_heads_decode`), 17 reshards.

## Classification rubric

A used-as-is · D used-but-modified (advisor config was the starting candidate; shipped config is a re-tuned
derivative of it, same op, altered params) · B not-used (not shipped and nothing derived; reason:
incorrect / superseded-on-perf / algorithm-change) · C could-not-use (no valid config / hard blocker).

## Shard-advice usage (N = 27 op recommendations)

| Advice item / group (report.json idx) | Advisor recommendation | Verdict | Reason | Evidence (source) |
|---|---|---|---|---|
| QKV linear (2) | 1-D matmul @11x10, in0=2, per_core_N=3, subblock 1x3, 107-core width-sharded output | **A** | — | work_log disposition: "applied … advisor simple real PCC 0.9999794 prefill / 0.9999808 decode; faster than DRAM family". |
| O_proj linear (18) | 1-D matmul @11x8, in0=2, per_core_N=3, subblock 1x3, 86-core output | **A** | — | "retained in the 2.114 ms final traced decode". |
| gate_proj + up_proj linears (21, 22) | 1-D matmuls @11x10, in0=2, per_core_N=9, subblock 1x3, 100-core outputs | **A** | — | "applied … final-family block/grid sweep rejected block 4/8/16 and grids 11x9/11x8"; advisor block 2 + 11x10 retained. |
| down_proj linear (24) | 1-D matmul @11x8, in0=2, per_core_N=3, subblock 1x3, 86-core output | **D** | re-tuned | "applied as seed, then tuned … real-weight sweep selected 11x6, in0=4, per_core_N=4, subblock 1x4, 64-core; 2.1140 vs 2.1203 ms over 500 replays". |
| SiLU·up multiply (23) | width-sharded 100-core (matches gate/up output) | **A** | — | SiLU folded into `ttnn.mul(...SILU...)`; split gate/up kept (packed exceeds L1). |
| qkv-heads split, 2× RoPE, mask ge/where/repeat, SDPA-decode, slices, reshapes (0,3,4,5,6,7,8,9,10,11,12,13,14,15,17,26) — 16 items | height-shard / block-shard / DRAM / interleaved topology layouts | **A** | — | Topology scaffolding shipped as-is; `nlp_create_qkv_heads_decode` replaced manual split (PCC/trace pass); composite SDPA retained. |
| 2× RMSNorm block-sharded 11-core + 2× residual add width-sharded 86-core (1, 20, 19, 25) — the "exact advisor residual/layout chain" | block-sharded 11-core norms + exact 86-core residual chain | **B** | superseded-on-perf | "tested, rejected … correct at 0.9999794/0.9999797 but 9.926 ms prefill / 2.545 ms decode vs simple advisor 9.761/2.454 ms". Shipped default is a simpler uniform 32-core width-sharded chain; advisor-exact chain kept only behind non-default `advisor_exact_residual_chain=True`. |
| `nlp_concat_heads_decode` (16) | dram/interleaved (advisor query failed) | **C** | — | `unfixable_ops`: `TT_FATAL … Input tensor must be sharded`. Advisor left SDPA output DRAM while concat-heads requires sharded input; runtime supplies a targeted sharding conversion. |

**Counts: A=21, B=4, C=1, D=1 (of 27).**
**B reason breakdown:** superseded-on-perf 4, incorrect 0, algorithm-change 0.

## DRAM-sharding per matmul

Shipped default is `decode_matmul_strategy="advisor_1d"` — **all five projection matmuls use the advisor's
L1 width-sharded 1-D matmul family (`MatmulMultiCoreReuseMultiCast1DProgramConfig`), NOT DRAM-sharded.**
A full DRAM-sharded path (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` +
`_dram_weight_memory_config` → `BufferType.DRAM` width-sharded weights) is implemented but selectable only via
non-default `decode_matmul_strategy="dram_sharded"`.

| Matmul | Final (shipped) layout | DRAM-sharded? | Versions tested (+measured) | Why this choice |
|---|---|---|---|---|
| Fused QKV | advisor 1-D L1 width-shard @11x10, in0=2, per_core_N=3, BFP4/LoFi, DRAM-interleaved weights | **No** | advisor-1D vs repaired 32-core DRAM-sharded; BFP4/BFP8/BF16 precision | advisor 1-D decode 2.454 ms vs DRAM-sharded 2.600 ms; advisor faster + avoids DRAM-sharded prefill corruption hazard |
| O_proj | advisor 1-D L1 @11x8, in0=2, per_core_N=3, BFP4/LoFi | **No** | same family + explicit SDPA/O sweeps | same; O retained in 2.114 ms final |
| gate_proj | advisor 1-D L1 @11x10, in0=2, per_core_N=9, block 2, subblock 1x3, BFP4/LoFi | **No** | blocks 2/4/8/16, grids 11x10/11x9/11x8; split vs packed | advisor block 2 + 11x10 win (alt 2.123–2.451 ms); packed exceeds L1 (2,263,296 B CB > 1,572,864 B) |
| up_proj | advisor 1-D L1 @11x10 (as gate) BFP4/LoFi | **No** | same as gate | same |
| down_proj | advisor-seeded then tuned: 1-D L1 @11x6, in0=4, per_core_N=4, subblock 1x4, 64-core, BFP4/LoFi | **No** | blocks 2/4/7/8/14/16; grids 11x8/11x6/11x5; DRAM down block auto/7/4/2/1 | 11x6/block-4 = 2.1140 ms wins over 11x8/block-4 2.1203 ms; DRAM down 2.632–2.867 ms slower |

**DRAM-sharding verdict: 0 of 5 matmuls DRAM-sharded.** Dominant reason: the advisor's L1 width-sharded 1-D
chain measured **faster** on traced decode (2.454 vs 2.600 ms at matched precision), and the DRAM-sharded
prefill family was additionally a **correctness hazard** — AutoDebug traced a PCC 0.000324 (infinities) failure
to eight-bank DRAM-width-sharded weights consumed by an 11-column 2-D program with a mismatched `per_core_N`.
Norms, residual adds, and the SiLU·multiply stay L1 (no large streamed weight to amortize).

## Summary

Of 27 advisor recommendations, **21 shipped as-is (78%)**, **1 was re-tuned (down_proj geometry, D)**,
**4 were superseded on perf (B — the exact block/width norm+residual chain, ~3.7% slower)**, and **1 was a hard
blocker (C — `nlp_concat_heads_decode`, no valid advisor config)**. Nothing was rejected as incorrect. Like the
8B sibling, this model keeps the advisor's L1 width-sharded 1-D matmul family as the shipped decode path and
ships **zero DRAM-sharded matmuls** — the DRAM-sharded alternative was both slower and prone to silent
weight/program-shape corruption. Final all-BFP4/LoFi decode is 2.114 ms (batch 32), 2.66× faster than functional
at batch 1.
