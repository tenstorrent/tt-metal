# Shard-Advise Usage & DRAM-Sharding Analysis — optimized_decoder

**Model:** `meta-llama/Llama-3.1-8B-Instruct`
**Scope:** single-device decode/prefill decoder layer (`tt/optimized_decoder.py`). lm-head is out of scope.

This report records (1) how much of the mandatory shard-advisor (`$shard-advise` / OPT-015) output the
optimize stage actually shipped, and (2) the DRAM-sharded vs L1-sharded matmul decision for every projection.

---

## 1. Shard-advice usage

Advisor input: `doc/optimized_decoder/shard_advise/report.json` (22 per-op layout/program_config items; plus
12 reshards, 1 spill, 1 op the advisor could not model). Disposition source: `work_log.md` OPT-015 table +
final code + optimize codex log.

### Rubric
- **A — used as-is:** advisor config shipped unchanged.
- **D — used but modified:** advisor config taken as the starting candidate; shipped config is a re-tuned
  version of it (same op, altered params/layout).
- **B — not used:** advisor config is not in shipped code and nothing was derived from it. Reason tags:
  `incorrect` / `superseded-on-perf` / `algorithm-change`.
- **C — could not use:** advisor emitted no valid config / hard technical blocker.

### Classification

| Advice group (items) | Advisor recommendation | Verdict | Reason | Evidence |
|---|---|---|---|---|
| QKV / O / gate / up / down decode matmul program_configs (5) | 1-D L1 width-sharded: QKV 11x9 `in0=2,pcN=2,sb=2`; O 11x6 `in0=8,pcN=2,sb=2`; gate/up 11x9 `in0=2,pcN=5,sb=5`; down 11x6 `in0=8,pcN=2,sb=2` | **A** | — | Shipped as the **default** (`decode_matmul_strategy="advisor_1d"`); `_advisor_matmul_program_config(grid=(11,9)...)` in code matches exactly. work_log OPT-015: all four "executed / applied". |
| Matmul output width-shard mem configs (QKV 96-shard, gate/up 90-shard, O/down 64-shard) (bundled with above) | `_advisor_width_sharded_l1` outputs | **A** | — | Used whenever `use_advisor_1d` (the default). |
| qkv-heads split, RoPE (Q/K), decode-SDPA DRAM layout, SwiGLU multiply, 2 residual adds, 7 slice/reshape boundaries (14) | height-shard / DRAM / composite / trivial-boundary layouts | **A** | — | Layouts present in final code; OPT-015 marks head/RoPE/SDPA retained. |
| Exact advisor norm/residual chain: input-norm block-11 + post-norm width-22 (2) | block-sharded input RMSNorm (11 cores) + width-22 post-norm + width-64/56 reshards | **B** | superseded-on-perf | Implemented behind `advisor_exact_residual_chain` (default **False**); measured **correct** (PCC 0.999805/0.999855) but **slower**: traced decode 0.749 ms vs shipped uniform-32-core chain 0.713 ms. work_log: "rejected as default … slower than mixed 32-way residual chain". |
| `nlp_concat_heads_decode` (1) | dram/interleaved (advisor emitted no valid config) | **C** | — | Advisor "reverted choice"; runtime uses the dedicated height-sharded concat contract. `TT_FATAL … Input tensor must be sharded`. |

**Counts:** A = 19 · B = 2 · C = 1 · D = 0 (of N = 22).
**B reason breakdown:** superseded-on-perf 2, incorrect 0, algorithm-change 0.

Note: unlike the Falcon models, Llama's shipped default **is** the advisor 1-D chain — so almost all of the
advisor's matmul + width-shard output advice ships verbatim (A). Only the advisor's coherent norm/residual
chain lost to a hand-tuned uniform 32-core width-sharded chain, and only the concat op was unmappable.

---

## 2. DRAM-sharding analysis

**Headline: the shipped default uses NO DRAM-sharded matmuls.** All five projection matmuls run as **advisor
1-D L1 width-sharded** matmuls (`MatmulMultiCoreReuseMultiCast1DProgramConfig`) with **BFP4 weights held in
plain DRAM-interleaved buffers** (activations/outputs width-sharded in L1). A fully-wired DRAM-sharded path
(`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` + `_dram_weight_memory_config`, selectable via
`decode_matmul_strategy="dram_sharded"`) was implemented and swept, but lost on decode latency.

Governing rule (optimize codex log, OPT-004): *"Keep the advised config only where it beats the DRAM-sharded /
best measured candidate."* Here it does.

| Matmul | Final layout (shipped default) | DRAM-sharded? | Versions tested (+ measured result) | Why this choice |
|---|---|---|---|---|
| Fused QKV | advisor 1-D L1 width-sharded, 11x9/`in0=2`/`pcN=2`/`sb=2`, BFP4×BF16 LoFi, DRAM-interleaved weight | **No** | DRAM-sharded rewrite (decode 0.895→0.803 ms across 32/16/8-core geometry) vs advisor 1-D (0.713 ms); BFP8 vs BFP4 weights | Advisor 1-D beat DRAM-sharded on traced decode; BFP4/LoFi passed PCC (profile: 91 us, 12.6%). |
| O projection | advisor 1-D L1 width-sharded, 11x6/`in0=8`/`pcN=2`/`sb=2`, BFP4 LoFi | **No** | same DRAM-sharded vs advisor 1-D sweep | Advisor 1-D faster; profile 33 us, 4.6%. |
| gate_proj | advisor 1-D L1 width-sharded, 11x9/`in0=2`/`pcN=5`/`sb=5`, BFP4 LoFi | **No** | `out_subblock_w` 1 (0.717 ms) vs 5 (0.713 ms); packed vs split gate/up; geometry 8/16/32 cores | Advisor 1-D + subblock 5; DRAM-sharded/packed lost; profile 113 us, 15.6%. |
| up_proj | advisor 1-D L1 width-sharded, 11x9/`in0=2`/`pcN=5`/`sb=5`, BFP4 LoFi | **No** | same as gate | same; profile 107 us, 14.8%. |
| down_proj | advisor 1-D L1 width-sharded, 11x6/`in0=8`/`pcN=2`/`sb=2`, BFP4 LoFi | **No** | BFP8/HiFi2 (0.892 ms) vs BFP4/LoFi (0.803 ms on DRAM candidate); DRAM 32/16/8-core (0.803/0.803/L1-clash) | Advisor 1-D + BFP4/LoFi; DRAM-sharded matched but did not beat it; profile 110 us, 15.2%. |

### Dominant reason DRAM-sharding is not used
The advisor's L1 width-sharded 1-D matmul chain measured **faster on traced decode** (final 0.713 ms BFP8-cache)
than the best DRAM-sharded geometry (0.803 ms). DRAM-sharding was not rejected for correctness — the
DRAM-sharded rewrite was the initial optimized baseline and was fully geometry-swept — it simply lost the
decode-latency comparison, so the advisor 1-D path shipped as default and DRAM-sharded remains a
precision-locked sweep control.

---

## Summary

- Shard-advice: **19/22 used as-is, 2 rejected on perf (norm/residual chain), 1 unmappable (concat), 0 modified.**
- DRAM-sharding: **0 of 5 matmuls** ship DRAM-sharded; all ship as advisor 1-D L1 width-sharded with
  BFP4 DRAM-interleaved weights, because the advisor 1-D chain beat every DRAM-sharded geometry on traced decode
  (0.713 vs 0.803 ms). The DRAM-sharded path stays available behind `decode_matmul_strategy="dram_sharded"`.
