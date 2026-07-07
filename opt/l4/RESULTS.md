# LTX 3s opt run — results & honest verdict (overnight 2026-07-07)

**Ask:** get LTX fast-mode 6s/1080p AV generation from ~5.7s down to **3s** on galaxy.
**Branch:** `smarton/optimizer/ltx-3s-2026-07-07` in worktree `.claude/worktrees/ltx-perf-clean`.
**Prewarm cherry-pick `d90194e3de4`:** applied, built, validated (100% JIT cache hit, zero cold
compile in the device window). Iteration-time win confirmed.

---

## TL;DR

1. **The "5.7s fast mode" you were shown is not a real operating point — it fails quality.**
   It's `LTX_FAST=1` = bf8 quant + a **7-step (6+1) schedule**, measured with quality gates OFF.
   Frame-PCC vs the 11-step reference: **min 0.36 / mean 0.53** — a *different generation*, not a
   degraded one (exactly the documented 7-step cliff). Don't ship it.

2. **Honest quality-gated floor today: 6.52s** (bf8 quant + 8-step 6+2), mean-PCC **0.85** vs the
   bf8 11-step full-quality ref. The step cut (11→8) holds; going below 8 steps breaks the scene.

3. **3s is not reachable overnight at quality — and not with config levers at all.** At 32 chips
   this workload is **communication-bound** in the DiT, not compute-bound. That's why galaxy is only
   ~1.8× faster than loudbox's 11.8s, not the ~4–8× the chip count suggests: adding chips adds
   AllGather/ReduceScatter traffic that offsets the compute parallelism (strong-scaling wall).

4. **But there is a real, funded path to a faster DiT: depth-adaptive temporal-window sparse
   attention, training-free.** On real activations, mid/late DiT blocks tolerate a ±2–3-frame
   temporal window at PCC 0.96 (2.7–3.8× attention sparsity, no finetune); early blocks need dense
   attention. This is the one genuinely new result — see §4. It needs a new kernel (~days) + a
   per-block mask-search, not a finetune. Realistic ~6.0–6.15s; 3s needs it stacked with more.

---

## 1. Operating points (all measured this run, gen#1 traced steady-state, 4×8, 1088×1920, 145f)

| Config | S1 | S2 | VAE | Audio | **Total** | qgate (mean-PCC vs 11-step) |
|---|---|---|---|---|---|---|
| 11-step bf16 (ref, full quality) | 2.84 | 3.49 | 1.14 | 0.99 | **8.71s** | 1.00 (reference) |
| 11-step bf8 (ref, prod full quality) | 2.69 | 3.28 | 1.06 | 1.49* | **8.75s** | — (the gate reference) |
| **8-step bf8 (GATED BASELINE)** | 2.04 | 2.28 | 1.10 | 0.88 | **6.52s** | **0.85** (holds) |
| 7-step bf8 = `LTX_FAST` (INVALID) | 2.05 | 1.24 | 1.08 | 0.87 | 5.44s | **0.36** (different scene) |

\* the 11-step bf8 ref ran with the audio flags that regressed audio (see §2, lever L-audio).

## 2. Levers tried — the honest sweep (most were null/dead; measurement caught the fakes)

| Lever | Class | Result |
|---|---|---|
| **SDPA→LoFi+fp32acc** (`all_bf8_lofi_sdpa_lofi_fp32acc`) | gated | **wash: 6.46s (−0.06s), PCC 0.848.** Proves DiT is CCL-bound, not SDPA-compute-bound. |
| **Audio flags** `LTX_VOC_TRACE=0`+`LTX_AUDIO_CHANNEL_TP=1` | "neutral" | **REGRESSION +0.6s** (audio 0.88→1.5s). Traced vocoder is faster here — the in-code "eager wins" claim is wrong at this config. Dropped. |
| **Lever 2a** audio AdaLN chunk → tile-preserving | neutral | **null: 6.50s (−0.02s).** Bit-identical; the audio tilize cost is negligible at scale. Kept as a video-path-consistency cleanup, not a perf win. |
| **RoPE fold** (mirror WAN) | — | **infeasible.** `wan_fused_rmsnorm_post_allgather` is head-uniform; LTX rope is genuinely per-head (31/32 heads differ, max\|Δcos\|~2.0). Would silently corrupt attention; needs a C++ per-head-cos kernel extension. |
| **Euler-in-trace** | — | **flawed** (folding the i2v pin into the trace crashes the t2v→i2v sequence) and ~10–30ms anyway. |
| **D2H round-trip elision** | — | sound but ~1–2% (the big stage-2-noise H2D stays). Not worth it. |
| **num_links 2→3/4** | — | **infeasible.** num_links=2 is the BH-galaxy fabric wall (`"BHGLX":(2,2)`), not a knob. |
| **Ring→Linear topology** | — | Ring is optimal (it's what unlocks the fused AG/RS dataflow). |
| **CCL reshards** | — | already tight (Megatron AG-at-entry / RS-at-exit; no removable round-trip). |
| **SDPA-input bf8 CCL transfer** (Lever 5) | gated | **REJECT: 6.48s (−0.04s, noise) at PCC 0.876** (down from ~0.90). Wired the dead `_sdpa_input_dtype` hook + bf8 K/V gather buffers; the added q/k/v typecasts (48 blk × 2 stages) offset the halved fabric bytes. Spends quality margin for ~0 speed — the ring gather isn't the bandwidth-limited critical path at this scale. |

## 3. Why 3s is a communication-bound wall (the technical answer to "galaxy should hit 3s")

DiT denoise = 66% of the 6.52s (~4.3s). Per-DiT-block: matmul/CCL ~35%, RingJointSDPA ~29%
(dense), tilize/typecast ~20%, RMSNorm ~4%. Dropping SDPA math fidelity bought 0.06s and bf8 quant
bought ~4% — because the **matmul/attention math is not the bottleneck; the AllGather/ReduceScatter
fabric traffic is.** num_links, topology, and reshards are already at the hardware/library optimum.
So there is no config knob to 3s. The only structural lever is reducing the *amount* of attention —
which is §4.

## 4. The one real path forward: temporal-window sparse attention (TRAINING-FREE) — FUND IT

I captured **real** post-RoPE attn1 Q/K/V (block 24, Stage-2, 8 heads × 38912 tokens) from a live
denoise step and measured windowed-mask output PCC vs dense:

| Window | kept% | ideal× | PCC_avg | PCC_min |
|---|---|---|---|---|
| **T=±2 frames, full spatial** | 26% | 3.8× | **0.964** | 0.912 |
| **T=±3 frames, full spatial** | 37% | 2.7× | 0.973 | 0.929 |
| T=±4 frames, full spatial | 47% | 2.1× | 0.980 | 0.946 |
| T=±1 frame, full spatial | 16% | 6.3× | 0.948 | 0.891 |
| any *spatial* window | — | — | 0.85–0.89 | **0.56–0.64 (fails)** |

(dense sanity = 1.0000.) **A temporal window — each token attends only to ±2–3 neighboring frames,
full spatial — holds quality well above the 0.85 gate at 2.7–3.8× attention sparsity, with NO
finetune.** This reverses the earlier synthetic-Q/K probe (PCC ~0.5): synthetic tensors have no
learned locality; real video attention is strongly temporally local (this is exactly what STA/VSA
exploit, and FastVideo has *not* applied it to LTX).

**Why it's buildable:** the frame-major F×H×W token layout makes a temporal window a **contiguous
per-query-frame K-range** — a banded K-skip, which the ring SDPA kernel can express (it already has
causal_k_limit skipping). Effort: a temporal-banded K-skip ring SDPA kernel, ~days of C++ (NOT the
existing `windowed_scaled_dot_product_attention`, which is dense+mask = zero speedup).

**CRITICAL: viability is per-block and non-monotonic — not a clean depth threshold.** I mapped 9
blocks (single Stage-2 step); min-PCC ≥0.85 at a ±3-frame window (2.7×):

| block | 4 | 8 | 12 | 16 | 20 | 24 | 32 | 40 | 47 |
|---|---|---|---|---|---|---|---|---|---|
| ±3f min-PCC | 0.28 | 0.74 | **0.92** | 0.56 | 0.81 | **0.93** | **0.94** | **0.90** | **0.98** |

Late blocks (32/40/47) hold robustly and get *more* local toward the output (b47 min 0.98); early
blocks are mixed — **b12 (early) holds but b16/b20 (mid) fail**, so it's not "dense first N, windowed
rest." A **per-block mask-search** is required. Notes: avg-PCC ≥0.85 for every block except b4 (the
min failures are localized query regions), and block-level PCC is a *conservative screen* — errors
across 48 blocks + VAE may not compound, so the real gate is an e2e decoded-video run with the
kernel. Realistic win: ~half the blocks windowable → est **~0.3–0.5s off → ~6.0–6.2s**. Full map +
per-block ±2f/±3f/±4f data: `opt/l4/depth_map.txt`.

**Confidence:** 9-block depth map (single step). The temporal-locality signal is strong and
trends with depth; per-block calibration + across-step confirmation is the mask-search (step 1).

## 4b. Audio: a stale-fork regression — the biggest FREE win (−0.42s)

Measured the standalone girl-clip audio decode (`test_audio_decode_girl`, galaxy 4×8, traced, warm):

| | this branch (deffb9) | ltx-perf tip |
|---|---|---|
| **Audio decode (warm, galaxy 4×8)** | **0.92s** (vocoder+BWE 795ms) | **0.5s** |

ltx-perf tip's own committed perf table (`models/tt_dit/models/LTX2.md`, refreshed Jul 1) reports
0.5s on galaxy 4×8; my measured 0.92s on the same mesh/checkpoint/content is a **~0.42s regression**.
**Root cause: this branch (`deffb9`, ~Jun) is a stale fork that predates ltx-perf's audio-decode
refactor** (940 commits diverged; Jun 30–Jul 6: `d976d711a16` slim decode, `9dd7c436fc4` on-device
weight formatting, `ec0c2496463` conv1d reshard+untilize fold, plus vocoder/bwe/decode_audio
rewrites = −772 lines in `pipeline_ltx.py`). It is NOT one missing commit — it's an entire divergent
(older) audio subsystem.

**Fix = rebase, not re-optimize.** Do NOT hand-optimize vocoder+BWE on the stale fork (that
duplicates shipped ltx-perf work). Rebase the fast-mode/quant/sigma/L4 work onto ltx-perf tip → the
audio drops to 0.5s for free (**e2e fast baseline 6.52s → ~6.1s**), and you inherit the current VAE
+ pipeline too. This is the single biggest available win — larger than any config lever and larger
than L4's estimate — and it stacks with L4. Caveat: the rebase is 940 commits and will have
conflicts (the fast-mode video features — LTX_FAST bundle, quant_config, sigma overrides — are not
on ltx-perf); budget it as a real integration, and re-baseline all numbers on the new base.

## 5. What's delivered / committed

- Prewarm cherry-pick built & validated (`386ee5b0e2f`).
- Gated baseline **6.52s @ PCC 0.85** characterized and reproducible (multiple runs ~6.5s).
- tt-opt harness (`ttw.toml`, `opt/ttw/knowledge/context.md`, `opt/iters.jsonl` ledger).
- L4 investigation tooling: capture hook + real-activation probe (the decisive experiment).
- Lever 2a consistency cleanup (audio tile-preserving chunk).

## 6. Recommended next steps — ranked path toward 3s

Synthesized from external literature (adversarially-verified deep research) + internal TT knowledge
(Glean). **Honest headline: ~3s for the exact 6s/1080p/145f deliverable is NOT reachable by
training-free levers alone** — the verified literature puts the training-free ceiling at ~1.5–2×
off 6.5s (→ ~3.5–4.5s), short of the ~2.2× needed. Reaching ~3s requires a **retrained ≤4-step
CFG-free LTX** on top of the kernel work, OR a frame-count/resolution product decision. Context: the
official roadmap target was **6.0s and it was hit** (Townhall 06-17); 3s is ~2× beyond roadmap.

### Tier A — free/cheap, do first (quality-safe)
1. **Rebase onto ltx-perf tip** (see §4b). Audio 0.92→0.5s = **−0.42s → ~6.1s**, free, inherits
   current VAE/pipeline. Biggest single win. 940-commit rebase w/ conflicts — real integration.
2. **Per-block × per-step mask-search** for L4 — sweep all 48 blocks × the denoise steps with the
   `opt/l4/` capture+probe tooling to map which blocks tolerate which ±k temporal window (the 9-block
   map shows it's block-specific and non-monotonic). Cheap; prerequisite for the kernel.

### Tier B — kernel work (~days each), training-free, STACKS
3. **L4 depth-adaptive temporal sparse attention.** DON'T build from scratch — **reuse the internal
   MiniMax-M3 Sparse Attention (MSA) machinery** in tt-blaze/tt-metal, already on the *same Galaxy
   SP8×TP4 mesh*: `indexer_score_msa` (merged, tt-metal PR #48205), `FlashGQASparseDecode` (tt-blaze
   PR #1788, attends only to selected 128-token blocks), `MiniMaxM3IndexerSelect` (block-select).
   Re-targets: decode-time→prefill, GQA→MHA, content-index→(static temporal window OR indexer).
   Owners skrsticTT / nmaurice / handrews; loop in Colman Glagovich (TT FlashAttention owner, >70%
   BH math-util FA). Realized ~1.3–1.7× on attention (lit + our probe) → **~−0.3–0.5s → ~5.7s**.
4. **VAE decoder acceleration.** Video VAE is 1.1s and NOT trace-replayed today. Flash-VAED-style
   decoders report ~6× on the *exact* LTX VAE in the literature (bounded e2e since DiT dominates) →
   **~−0.4–0.6s → ~5.2s**. Kevin is already profiling the VAE (`prof_vae_ltx.py`, `LTX_USE_FUSED`).

### Tier C — the decisive lever for 3s, but needs model training
5. **Few-step CFG-free distilled LTX (8→4 steps, or fewer).** For a *communication-bound* DiT this
   is the #1 lever: each removed denoise step removes a full round of AllGather/ReduceScatter. NOT
   training-free — needs offline distillation (Phased DMD arxiv 2510.27684 hits a 4-step floor on
   14–28B video DiTs; Wan2.2-Lightning ships 4-step + CFG-free at ~20× fwd-pass reduction). 4 steps
   is the safe quality floor; 3-step aspirational. **LTX checkpoints are hot-swappable** (Sulphur
   weight-swap; AnimateDiff-Lightning 4-step precedent internally), so a distilled checkpoint drops
   in. Path: (a) request/wait for a Lightricks few-step LTX-2.3, or (b) an internal distill run.
   8→4 steps ≈ −40–45% DiT ≈ **−1.5–1.7s**. This is what closes the gap to ~3s.

### Dead ends — do NOT pursue (evidence)
- **Feature caching** (TeaCache/PAB/FBCache/∆-DiT/toca): dead at ≤8 steps on distilled models — two
  independent 2026 primary sources (Chorus, DisCa) agree caching & distillation exploit the *same*
  inter-step redundancy; also matches the internal `LTX_PERF_PLAN.md` finding.
- **CFG removal**: already off — the distilled pipeline is `_denoise_no_guidance` (verified). Spent.
- **"Batched denoising"** (the roadmap's named next lever): a **throughput/cost** optimization
  (larger batch → larger CCL messages → better mesh utilization), NOT single-clip latency — a
  single clip's steps are sequential. Won't help the 3s number; helps $/video at serving scale.
- **Config knobs** (num_links, topology, reshards, SDPA-LoFi, SDPA-bf8): the HW/comms wall (§2/§3).

### Honest 3s verdict
Tier A+B (all quality-safe, ~weeks of kernel work) → **~5.0–5.3s**. Reaching **~3s requires Tier C**
(the 4-step distilled model — the decisive lever), OR a product decision: 145f→~73f *is* a 3s clip
but a different deliverable; 121f (FastVideo's) → ~5.3s. Sparse-attention CUDA kernels and all
published FPS numbers are NVIDIA-specific — only the *algorithms* port to Blackhole; budget kernel
implementation as new BH work.
