# Shard-Advice Usage & DRAM-Sharding Analysis — optimized_decoder

**Model:** Qwen/Qwen3-32B
**Branch:** `mvasiljevic/model/qwen-qwen3-32b`
**Artifacts:** `doc/optimized_decoder/shard_advise/report.json` (advisor input), `tt/optimized_decoder.py`
(final), `doc/optimized_decoder/{work_log,README,perf_report,stage_review,AUTODEBUG}.md`, optimize codex log.

## Summary

The shard-advisor ran as a mandatory gate (OPT-015) after the dense attention+MLP rewrite. Its report holds
**24 per-op layout/program_config recommendations** (`total_ops` reported as 26 in narrative, `final_choices`
23–24), plus 19 reshards, 1 spill, and 1 unfixable op. The optimize stage implemented the advisor family as an
executable candidate (`decode_matmul_mode="shard_advisor"`) but **ships a different default**
(`decode_matmul_mode="dram_sharded"`): DRAM width-sharded projection weights with
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, a measured 40-core L1 width-sharded residual/norm/gate/up
chain, a separately-swept 32-core down projection, and production head/RoPE layouts. The advisor's exact L1
width-sharded 1-D matmul + block-sharded norm + height-sharded RoPE configs survive only behind the non-default
`shard_advisor` flag.

Two documented advisor outcomes drive the classification (work_log OPT-015 / README shard-advisor gate / stage_review):
- **Full legal advised layout family → decode PCC 0.985269 < 0.99 (reject on correctness), 1.743 ms.**
- **Advisor 1-D matmuls with production head layouts → correct 0.999272 but slower, 1.737 ms.**
- **Selected DRAM-sharded production matmuls → correct 0.998990, 1.217 ms** (30% faster than the correct advisor-matmul trial).

## Shard-advice usage (consistent rubric)

Rubric: **A** used-as-is · **D** used-but-modified (advisor config = starting candidate, shipped = re-tuned
derivative) · **B** not-used (not shipped, nothing derived; reason tag `incorrect` / `superseded-on-perf` /
`algorithm-change`) · **C** could-not-use (no valid config / hard blocker).

| Advice group (items) | Advisor recommendation | Verdict | Reason | Evidence |
|---|---|---|---|---|
| Topology/boundary + fixed-contract head ops: reshape x2 (op0,23), slice_static x4 (op6,7,10,14), repeat (op11), SDPA-decode (op12), nlp_create_qkv_heads_decode (op3) — **9** | interleaved / dram / height-sharded head split | **A** | — | README keeps decode SDPA + TTNN head composites; these boundary/fixed-contract layouts match shipped code. |
| 5 projection matmuls: QKV (op2 @11x10), O (op15 @11x8), gate (op18 @11x10), up (op19 @11x10), down (op21 @11x8) — **5** | L1 width-sharded `matmul_multi_core_reuse_multi_cast_1d` (1-D) configs | **B** | superseded-on-perf | Shipped default is independent DRAM-sharded matmuls; advisor 1-D configs kept only under `shard_advisor` flag. "Advisor 1-D matmuls, production heads: correct 0.999272 but 1.737 ms; DRAM-sharded selected 1.217 ms" (work_log OPT-015). |
| Q-norm (op4, block 10x4), K-norm (op5, block 8x4), RoPE Q (op8), RoPE K (op9) — **4** | L1 block-sharded Q/K RMSNorm + L1 height-sharded RoPE | **B** | incorrect | "Full advised head/norm/RoPE/layout family: decode PCC 0.985269 fails 0.99" (work_log). Default emits Q/K norm + RoPE to DRAM instead (AUTODEBUG hyp.3: "advised head/norm/RoPE layout family changes numerical behavior"). |
| Input norm (op1, block 1x11), post-attn norm (op17, block 1x11), residual adds (op16, op22, width 1x80), SwiGLU multiply (op20, width 1x100) — **5** | block-sharded norms / width-sharded residual+multiply | **B** | superseded-on-perf | "Sharded residual/norm/linear skeleton: applied directionally; final keeps the measured 40-core L1 width-sharded chain" (work_log). Advisor 80/100-core widths not shipped; 40-core chain measured faster and removes reshards. |
| `nlp_concat_heads_decode` (op13) — **1** | dram/interleaved (advisor emitted invalid config) | **C** | — | `unfixable_ops`: `TT_FATAL ... Input tensor must be sharded`. Runtime requires HEIGHT_SHARDED input; the DRAM/interleaved concat result cannot be requested literally (AUTODEBUG hyp.3). |

**Counts (N=24): A=9, B=14, C=1, D=0.**
**B reason breakdown:** superseded-on-perf **10**, incorrect **4**, algorithm-change **0**.

Note: the `incorrect` tag reflects a family-level failure — the advised head/norm/RoPE layouts fail the 0.99 decode
PCC bar when applied together (RoPE also carries a live-trace per-position-allocation constraint). D=0: no shipped
config is a re-tuned derivative of an advisor config; ops are either shipped as-advised (A), replaced by an
independent implementation (B), or blocked (C).

## DRAM-sharding (per matmul)

All **5 of 5** in-scope projection matmuls ship **DRAM-sharded** (`decode_matmul_mode="dram_sharded"` default →
`_dram_sharded_memory_config` with `BufferType.DRAM` + `TensorMemoryLayout.WIDTH_SHARDED`, and
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`). All BFP4 weights / LoFi compute; BFP8 paged KV cache.
Qwen3-32B is dense (no MoE); it adds Q-norm/K-norm RMSNorms, which are not matmuls. lm-head is out of scope.

| Matmul | Final layout | DRAM-sharded? | Versions tested (+ measured) | Why this choice |
|---|---|---|---|---|
| QKV (packed, 5120x10240) | DRAM-sharded BFP4/LoFi, `in0=4, per_core_N=8`, 40-core storage; ~100–183 us, 260–261 GB/s | **Yes** | advisor L1 1-D @11x10 (correct 0.999272, 1.737 ms); DRAM-sharded (0.998990, 1.217 ms) | Interleaved weights underuse DRAM banks; DRAM width-sharded streaming wins (README "Decode weights"). |
| O (8192x5120) | DRAM-sharded BFP4/LoFi, `in0=8, per_core_N=5`, 32-core; ~80–146 us, 263–264 GB/s | **Yes** | advisor L1 1-D @11x8; DRAM-sharded selected | same DRAM-bandwidth argument |
| gate (5120x25600) | DRAM-sharded BFP4/LoFi, `in0=4, per_core_N=20`, 40-core; ~233 us, 281 GB/s | **Yes** | packed gate/up rejected (1.551 vs 1.367 ms separate); separate + DRAM-sharded ships | Two same-input matmuls; separate path beat packed; DRAM-sharded weights (README "Gate/up") |
| up (5120x25600) | DRAM-sharded BFP4/LoFi, `in0=4, per_core_N=20`, 40-core; ~233 us, 281–282 GB/s | **Yes** | as gate | same |
| down (25600x5120) | DRAM-sharded BFP4/LoFi, `in0=20, per_core_N=4`, 32-core storage; ~222 us, 297 GB/s | **Yes** | down-40 (1.219028 ms) vs down-32 (1.217996) vs down-16/block-25; down-32 selected on 500-replay tie-break | DRAM-BW; role-specific geometry sweep, down-32 wins (README/stage_review) |

Precision sweep (candidate matrix, decode ms / decode PCC): BFP8 HiFi2 2.017 / 0.999931; BFP8 LoFi 1.395 /
0.999890; BFP4-MLP + attention HiFi2 1.368 / 0.999249; **all-BFP4 LoFi (down-32) selected 1.217 / 0.998990**.
All-BFP4 failed an artificial random-distribution stress (synthetic 0.972) but passes real model boundaries; per
OPT-012 the synthetic distribution does not veto the real-weight winner (stage_review). Final decode is 67.4×
faster than functional BF16 and ~10.9% faster than the strongest retained correct earlier baseline (1.367 ms).
The five projection matmuls consume ~0.87 ms (~72% of device time); they remain marked `SLOW` because the
DRAM-sharded program-config class exposes no output-subblock knob (perf_report / AUTODEBUG hyp.4) — an
API/observability limit, not a chosen 1×1 subblock.

## Bottom line

Qwen3-32B **DRAM-shards all five projection matmuls** — the same choice as the two Falcon dense models — because
decode is DRAM-bandwidth-bound and streaming BFP4 weights from DRAM beats the advisor's L1 width-sharded 1-D
matmuls (1.217 vs 1.737 ms at equal correctness). Of 24 advice items, 9 boundary/head-contract layouts ship
as-advised, 14 are not used (10 superseded on measured latency, 4 rejected because the advised head/norm/RoPE
layout family fails the 0.99 decode-PCC bar), and 1 (`nlp_concat_heads_decode`) is a hard runtime blocker the
advisor itself marked unfixable. Nothing was adopted-then-re-tuned (D=0).
