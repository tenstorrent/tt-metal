# Decode matmul & shard-advise analysis (autoport optimized decoders)

Cross-model study of the **optimized decoder** stage produced by the agentic
bring-up pipeline (`mvasiljevic/model/*` branches). It started as a deep dive on
**Llama-3.1-8B-Instruct** and is repeated for every model that finishes its
optimized-decoder step. All analysis is **read-only** on the model branches
(via `git show origin/...`); nothing here is pushed to those branches.

Each model has its own page (`<model>.md` + `<model>.html`). This page is the
methodology + the cross-model summary.

---

## What the analysis asks of each model

1. **Shard-advise: which recommendations were applied vs. rejected, and why.**
   Distinguish what the advisor *caused* from what merely *coincides* with the
   pre-advisor baseline or came from an independent sweep.
2. **Decode matmul strategy** actually shipped per role (QKV / O / gate / up /
   down / experts / LM-head): shard-advise **1D-mcast** (DRAM-interleaved
   weights) vs **DRAM-sharded** weights vs `sparse_matmul`.
3. **Per-op device profile** from the committed `tracy/.../decode_perf_report`:
   device µs, cores, **DRAM GB/s and DRAM %**, math fidelity/dtype for every
   dominant matmul; plus any `Copy`/`Reshard` overhead rows.
4. Two recurring **pattern checks** (below).

## Two recurring patterns (found on Llama, checked on every model)

**Pattern A — shard-advise segment-boundary DRAM reverts.**
The advisor captures the decoder block as a *standalone function* whose
signature is **DRAM-interleaved in and out** (the capture harness feeds DRAM
tensors). It therefore brackets the block with an entry conversion
(DRAM → first op's shard) and an **"output revert"** (last op's shard → DRAM at
`func.return`). In a real repeated decoder block the residual stream stays
**L1-resident across all layers**, so those two boundary conversions are
*capture artifacts, not runtime cost*. Reproducing them faithfully (the
"exact advisor chain") is a measurable regression.

**Pattern B — matmul strategy × precision ordering.**
The 1D-mcast-vs-DRAM-sharded choice interacts with weight precision: dropping to
BFP4 halves weight DRAM traffic and can flip which strategy is fastest. If a
model picks its decode matmul strategy at an early/higher precision and then
only layers BFP4 onto that frozen "search base," the strategy decision can be
**stale** — a DRAM-sharded path never re-measured at the shipped precision may
actually be faster.

---

## Llama-3.1-8B — reference findings (on-device verified)

Unlike the other models (analyzed from committed artifacts), Llama-3.1-8B was
re-run on a Blackhole P300 with the real layer-16 weights. Detailed pages:
[`exact_vs_mixed_decode_perf.html`](../../models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/exact_vs_mixed_decode_perf.html)
and
[`dram_sharded_vs_advisor_matmul.html`](../../models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/dram_sharded_vs_advisor_matmul.html).

**Shard-advise verdict.** The advisor's value was the **matmul program-config
geometry** (grids `11×9`/`11×6`, `in0_block_w`, `per_core_N`, output subblock 5)
and projection output-shard geometry — applied, and drives all 5 dominant
matmuls. Its **norm/residual chain was rejected**: implemented faithfully as
`advisor_exact_chain` it kept PCC (0.99981/0.99986) but was **~5% slower**
(0.744 vs 0.710 ms traced decode). One op explains the whole gap:

- **Pattern A confirmed.** A ~**33 µs, 110-core `Copy`** (the advisor's
  L1-interleaved residual + DRAM output revert) that the shipped mixed 32-way
  chain never emits. Root cause: `optimized_decoder.py:866–871` places the
  exact-chain residual in `L1_MEMORY_CONFIG`; the mixed chain keeps it
  width-sharded on 32 cores → entry/exit reshapes are metadata-only.

**DRAM-sharded vs advisor_1d (Pattern B confirmed).** At the shipped all-BFP4
precision, a plain **DRAM-sharded, 32-core** decode matmul path is **~2% faster**
than the shipped advisor_1d default (**0.6965 vs 0.7101 ms**, BFP8 cache,
10 prefill / 100 replay) with **identical PCC**. The driver is **QKV**: the
advisor's 96-core 1-D config sits at **27% DRAM util (91 µs)** while
DRAM-sharded QKV hits **46% (53 µs)**. The bring-up log rejected DRAM-sharded,
but its sweep only ran at BFP8 attention/O precision and was never re-run at the
shipped BFP4 precision — strategy conflated with precision (`work_log:131`,
"retained as search base").

**DRAM utilization ceiling.** For this decode workload (M = 32 = one tile,
`per_core_M=1`, weight-streaming off 8 DRAM banks) DRAM-sharded matmuls top out
~40–55%; the compute grid is fixed by the op, and the per-core blocking has a
sweet spot (32 cores beat both 16 and 64). Sub-saturation here is a workload
ceiling, not a mistuned config.

Two skill improvements were filed from this (branch
`mvasiljevic/optimize-skill-notes`): re-decide matmul strategy at final
precision (OPT-014), and discard the advisor's segment-boundary DRAM
conversions for L1-resident repeated blocks (OPT-015).

---

## Cross-model summary

Decode traced-decode speedup vs functional, shipped decode matmul strategy, how
the advisor residual/exact chain was handled, and whether Patterns A/B appear.

| Model | Decode strat shipped | Decode speedup | Advisor resid-chain | Pattern A (boundary revert) | Pattern B (strategy×precision) |
| --- | --- | ---: | --- | --- | --- |
| [Llama-3.1-8B](llama-3.1-8b.md) | **advisor 1D-mcast** | 49× | rejected (~5% slower) | present; bites in rejected variant (~33 µs copy) | ❌ **stale — DRAM-sharded ~2% faster at BFP4** |
| [Llama-3.1-70B](llama-3.1-70b.md) | **advisor 1D-mcast** (TP→dense) | 2.7× (b1) | rejected (~3.7% slower) | present; ~171 µs reshape (~8%) | ❌ **partial — strategy frozen at BFP8; O row 38.7%** |
| [Mistral-Small-24B](mistral-small-24b.md) | DRAM-sharded | 72× | applied (L1 chain, +helped) | present; no copy, ~119 µs reshape (~9%) | ✅ re-decided at final precision |
| [GPT-OSS-20B](gpt-oss-20b.md) (MoE) | 1D-mcast attn + `sparse_matmul` experts | 6.6× | dense advisor chain rejected | present; **does not bite** (0 copies) | ✅ (attn 1D is correctness-bounded, not stale) |
| [Qwen3-32B](qwen3-32b.md) | DRAM-sharded | 67× | rejected (PCC + slower) | present; no copy, ~118 µs reshape (~10%) | ✅ re-decided at final precision |
| [Qwen2.5-Coder-32B](qwen2.5-coder-32b.md) | **1D-mcast** (BFP8, batch-aware) | 42× | rejected (block sweep) | present; avoided (~3.7% head permutes) | ✅ 1D correctly wins **at BFP8** |
| [Falcon3-10B](falcon3-10b.md) | DRAM-sharded (precision-aware cores) | 5.3× | rejected (whole-layer) | present; **copy 25 µs + reshape 34 µs (~7%)** | ✅ re-decided (+low-util O row: mixed strat untried) |
| [Falcon3-7B](falcon3-7b.md) | DRAM-sharded | 2.3× | rejected (at final precision) | present; **copy 24 µs + reshapes (~10%)** | ✅ re-decided at final precision |

## Cross-model synthesis

1. **Pattern B was Llama-8B's mistake, not the fleet's — and precision picks the
   winner.** The DRAM-sharded-vs-1D choice is *precision-dependent*: BFP4 halves
   weight DRAM traffic and favors DRAM-sharded; BFP8 (2× the bytes) keeps 1D-mcast
   competitive. The fleet bears this out:
   - The four BFP4 dense models (Mistral, Qwen3, Falcon3-10B/7B) re-measured at
     final BFP4 and shipped **DRAM-sharded**.
   - **Qwen2.5-Coder-32B** was PCC-forced to stay **BFP8**, re-measured at that
     precision, and **1D-mcast correctly won** (1.94 vs 2.29 ms) — with a
     batch-aware default (DRAM-sharded for batch-1). This is the clean validation
     of the mechanism.
   - **GPT-OSS** ships 1D attention for a documented **correctness** reason
     (sliding-window PCC), not a stale A/B.
   - **Both Llama-family runs (8B and 70B)** shipped 1D-mcast without a clean
     final-precision DRAM-sharded re-check: 8B is ~2% slower than an all-BFP4
     DRAM-sharded path; 70B adjudicated the choice at BFP8-attention and swept
     all-BFP4 only on the 1D family (O matmul left at 38.7% DRAM), though its
     lower-core DRAM-sharded is L1-infeasible so the residual risk is smaller.

   So the choice itself (1D vs DRAM-sharded) is legitimately model/precision/batch
   specific; the OPT-014/OPT-015 fix targets the *process* error — freezing the
   strategy before the precision is final. **Interestingly the two misses are
   both Llama-family bring-ups**; the Mistral/Qwen/Falcon runs re-decided cleanly,
   which suggests the gap is a per-family workflow habit rather than a universal
   pipeline flaw.

2. **Pattern A (the advisor's DRAM in/out segment boundary) is universal.** Every
   model's `final_ir.mlir` ends with an `(output revert)` shard→DRAM at
   `func.return` plus a DRAM→shard entry. How it lands at runtime splits:
   - **Reproduced as a real per-step `Copy`:** Falcon3-7B/10B (24–25 µs) — the
     Llama-style artifact, ~7–10% of decode.
   - **Copy avoided, reshape boundary remains:** Mistral, Qwen3, GPT-OSS write the
     last residual straight to public DRAM (0 Copy ops) but still pay ~50–68 µs
     entry/exit `ReshapeView` (~9–10%), except GPT-OSS which is cheap.
   In all cases the residual stays L1-resident *within* the block; the DRAM
   round-trip is the single-decoder-layer **public contract**. In a stacked
   N-layer model where the residual is carried in L1 across layers, this
   ~7–10%/layer boundary is the **largest cross-model fuse-away opportunity** —
   exactly what the OPT-015 note calls out.

3. **Small attention matmuls (QKV, O) are the least DRAM-efficient everywhere**
   (~33–52%, and 12% on GPT-OSS's 1D attn), while **MLP gate/up/down are the cost
   center and best-utilized** (47–58%). This is the M=32 decode ceiling (one tile
   of rows, weight-streaming). Falcon3-10B's low-util O row (33%) is the one
   place a per-op **mixed strategy** (DRAM-sharded MLP + 1D-mcast QKV/O) was never
   tried in isolation — a small, bounded, plausible remaining win.

4. **`ttnn.nlp_concat_heads_decode` is the recurring advisor-unfixable op** (needs
   a sharded input; the advisor proposes DRAM interleaved) — flagged on every
   dense model and worked around at runtime.

5. **All six ship BFP4/LoFi projections + BFP8 KV cache**, except GPT-OSS which
   keeps **BF16/HiFi4 attention** and **BFP8 experts** (BFP4 and BFP8-cache both
   failed its PCC gate — sliding window + sinks are precision-sensitive).

6. **Every shipped decoder is single-device dense** — this stage is a per-layer
   study on one Blackhole P300. Llama-3.1-70B is the tell: its source IR was
   captured at TP=4 / 2×2 mesh with all-reduces, but the stage **collapsed the TP
   projections into dense full-weight matmuls** (1×1 mesh), so no collective /
   fused-CCL / persistent-buffer work exists yet. Multi-device decode is deferred
   to a later stage across the fleet.

### Branch status (8 model branches)

| Model branch | optimized_decoder step | analyzed |
| --- | --- | --- |
| meta-llama-llama-3.1-8b-instruct | ✅ done | ✅ on-device |
| mistralai-mistral-small-24b-instruct-2501 | ✅ done | ✅ |
| openai-gpt-oss-20b (MoE) | ✅ done | ✅ |
| qwen-qwen3-32b | ✅ done | ✅ |
| tiiuae-falcon3-10b-base | ✅ done | ✅ |
| tiiuae-falcon3-7b-base | ✅ done | ✅ |
| qwen-qwen2.5-coder-32b-instruct | ✅ done | ✅ |
| meta-llama-llama-3.1-70b-instruct | ✅ done | ✅ |

**All 8 model branches analyzed.**

_Method note: Llama-3.1-8B is on-device-verified; the other models are analyzed
read-only from each branch's committed artifacts (`README.md`, `work_log.md`,
`shard_advise/report.txt`, `tracy/.../decode_perf_report` or the Tracy ops CSV)
without re-running on hardware. Per-op device times, DRAM GB/s and DRAM % are
quoted from those committed `tt-perf-report` tables. Nothing is pushed to the
`mvasiljevic/model/*` branches._
