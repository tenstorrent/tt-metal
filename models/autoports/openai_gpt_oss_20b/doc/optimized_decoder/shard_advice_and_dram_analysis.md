# Shard-Advise Usage & DRAM-Sharding Analysis — openai/gpt-oss-20b

**Model:** `openai/gpt-oss-20b` (MoE decoder: dense attention + top-4 routed sparse experts)
**Autoport dir:** `models/autoports/openai_gpt_oss_20b`
**Sources:** `tt/optimized_decoder.py` (final), `doc/optimized_decoder/{work_log.md,README.md}`, `doc/optimized_decoder/shard_advise/report.json` (advisor input), optimize codex run jsonl.

## Summary

The shard-advisor captured a **rewritten dense-attention + dense-MoE** decode graph (45 op recommendations). The optimize stage adopted the **entire attention half verbatim** (it measurably helped) and **discarded the entire MoE half**: the shipped runtime replaces the advisor's dense batched-expert matmuls with a **routed `sparse_matmul` expert family** (~6.37× faster on traced decode). Separately, **no matmul in the shipped decoder is DRAM-sharded**: the DRAM-sharded attention variant was built and measured but rejected because it fails the sliding-window boundary PCC bar, and the expert path is a sparse kernel family to which DRAM-sharding does not apply.

---

## 1. Shard-advice usage (consistent rubric)

Rubric: **A** used-as-is (advisor config shipped unchanged) · **D** used-but-modified (advisor config was the starting candidate, shipped config re-tuned) · **B** not-used (advisor config not shipped and nothing derived; reason = `incorrect` / `superseded-on-perf` / `algorithm-change`) · **C** could-not-use (no valid config / hard blocker).

Total advice items: **45** (`report.json` ops, `final_choices`=43; + 30 reshards, 0 spills).

| Item group (report.json indices) | Advisor recommendation | Verdict | Reason | Evidence |
|---|---|---|---|---|
| Attention sub-chain — input RMSNorm, fused QKV, head-split, SDPA-decode, O-proj, residual, boundary reshapes/full/zeros (idx 0–16, 17 items) | block-sharded norm on 10 cores; QKV width-sharded 45/80 @11x8 1D mcast; interleaved head ops; DRAM SDPA + KV cache; O input L1 → 11x9 90-core width-shard → DRAM residual revert | **A** | used-as-is | `use_shard_advisor_attention_layouts = True`; exact `@11x8`/`@11x9` program configs present in final code; work_log: "sparse end-to-end timing improves from 0.983908 to 0.928144 ms". |
| Dense expert compute — repeat-input, dense gate/up matmul(+bias), gate/up slices, SwiGLU clamp/mul/sigmoid chain, dense down matmul(+bias), routing multiply, expert-sum reduce, reshapes (idx 26–43, 18 items) | dense BF16 batched matmul over 32 experts + interleaved slices + dense down/reduce | **B** | algorithm-change | Shipped runtime uses routed `sparse_matmul` (`use_sparse_experts = True`); work_log: "routed sparse experts are 6.37x faster on traced decode; selected `sparse_matmul` falls outside the dense advisor capture". |
| MoE wrapper layouts — post-attn RMSNorm, FP32 router linear, topk/softmax, scatter, residual-revert add (idx 17–25, 44; 10 items) | width-90 norm, 1-core block router, block-sharded softmax/scatter, sharded residual revert | **B** | superseded-on-perf | Implemented in full dense A/B and **correct** (real prefill/decode PCC `0.99958–0.99973`) but the whole dense-MoE chain is slower (7.727175/**5.909603** ms vs selected 3.874835/**0.928144** ms); `use_shard_advisor_moe_layouts = False`, sparse router ships plain FP32. |

**Counts:** **A=17, B=28, C=0, D=0** (of 45).
**B reason breakdown:** algorithm-change **18**, superseded-on-perf **10**, incorrect **0**.

No C: the advisor emitted valid, correctness-verified MoE configs — nothing was a hard blocker. No D: shipped attention configs are the advisor's exact configs (A), not re-tuned derivatives.

---

## 2. DRAM-sharding analysis (per matmul)

DRAM-sharded = a `MatmulMultiCoreReuseMultiCastDRAMSharded*` / `get_dram_sharded_matmul_config` program config over a DRAM-sharded weight (`dram_sharded_weight_config`). In this model that path exists in code (`use_dram_sharded_attention`) but is **toggled OFF** in the shipped config.

| Matmul | Final shipped layout | DRAM-sharded? | Versions tested (measured result) | Why this choice |
|---|---|---|---|---|
| Fused QKV | L1 **width-sharded**, advisor `@11x8` 1D mcast (45/80 cores), BF16, separate bias broadcast | **No** | advisor L1 width-shard (**selected**, boundary passes); DRAM-sharded separate-bias: BFP4/LoFi PCC `0.747–0.750` invalid; BFP8/HiFi2 ordinary PCC `0.9957–0.9968` but sliding boundary **fails**; BF16/HiFi4 ordinary `0.9967–0.9977` but sliding pos-130 PCC `0.91306`, 1.042835 ms | DRAM-sharded rejected on **sliding-window boundary correctness**, not speed; advisor L1 width-shard passes boundary and improved decode 0.9839→0.9281 ms. work_log: "Advice to DRAM-shard QKV/O is closed by the corrected ordinary-position and boundary" evidence. |
| O projection | L1 **width-sharded**, advisor `@11x9` (90 cores) → DRAM residual add | **No** | shares the DRAM-sharded attention sweep above (`use_dram_sharded_attention` covers QKV+O); DRAM-sharded 3.881326/0.957894 ms but boundary fails | same as QKV: DRAM-sharded attention path fails the sliding boundary; advisor width-shard shipped ("final O row is 62 us on 90 cores"). |
| Router | FP32 `linear`, interleaved/L1 (small; hidden × 32 experts) | **No** | advisor 1-core block-sharded router layout built in dense A/B (correct) but rejected with dense chain; plain FP32 shipped | FP32 boundary is required by real-weight expert selection; matmul is tiny → no DRAM bandwidth to amortize; advisor sharded router belongs to the superseded dense chain. |
| Expert gate_up | routed **`sparse_matmul`**, BFP8 weights + LoFi, 9×10 grid, `in0_block_w=45`, `subblock_w=1` | **No** (sparse expert kernel, not a DRAM-sharded matmul) | dtype: BFP4 rejected (prefill PCC `0.976 < 0.99`), BFP8 kept (`0.998`); grids 3×4/5×6/8×8/**9×10** → 9×10 fastest (3.973/0.948 ms); blocks 30/45/90 → **45/45**; expert input L1 vs DRAM → DRAM (L1 0.968 ms slower) | routed-sparse replaces dense advisor MoE (6.37× faster). DRAM-sharding is an attention-matmul optimization; it does not apply to the sparse expert kernel family. BFP8 chosen for PCC bar; 9×10/45 tuned for latency. |
| Expert down | routed **`sparse_matmul`** with `is_input_a_sparse=True`, BFP8/LoFi, 9×10, `in0_block_w=45` | **No** (sparse kernel) | same expert sweep; down block 30/45/90 → **45** | same as gate_up: sparse expert family; DRAM-sharding not applicable; grid/blocks tuned. |

**DRAM-sharding verdict for this model: 0 of 5 matmuls DRAM-sharded.**
Dominant reasons: (1) **attention QKV/O** — DRAM-sharded variants fail sliding-window boundary PCC (BF16 pos-130 `0.913`, BFP8 full pos-129 fail), so the correctness-passing advisor L1 width-sharded configs ship instead; (2) **experts** — shipped compute is routed `sparse_matmul`, a different kernel family where DRAM-sharding is not an option; (3) **router** — tiny FP32 matmul, nothing to amortize.

This contrasts with the dense Falcon decoders, where DRAM-sharded BFP4/BFP8 matmuls win the projection matmuls outright.
