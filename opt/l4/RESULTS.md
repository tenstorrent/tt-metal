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
| **SDPA-input bf8 CCL transfer** (Lever 5) | gated | **untested** — attacks the confirmed comms bottleneck (halves K/V ring-gather bytes) but adds q/k/v typecasts that may offset it; the one marginal lever left. Uncertain ~1–3%. |

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

## 5. What's delivered / committed

- Prewarm cherry-pick built & validated (`386ee5b0e2f`).
- Gated baseline **6.52s @ PCC 0.85** characterized and reproducible (multiple runs ~6.5s).
- tt-opt harness (`ttw.toml`, `opt/ttw/knowledge/context.md`, `opt/iters.jsonl` ledger).
- L4 investigation tooling: capture hook + real-activation probe (the decisive experiment).
- Lever 2a consistency cleanup (audio tile-preserving chunk).

## 6. Recommended next steps (in order)

1. **Per-block mask-search** — sweep all 48 blocks × a few steps with the capture+probe tooling to
   map the dense→windowable transition depth and the tolerable ±k per block. (block 24 holds, block
   4 fails; the boundary is unmapped.) Cheap, and it's the prerequisite for the kernel.
2. **Build the depth-adaptive temporal-banded K-skip ring SDPA kernel** (dense early blocks, ±2–3
   frame window later). The real path to a faster DiT and toward 3s. ~days.
3. Marginal: test **Lever 5** (SDPA-input bf8) for the last ~1–3% on the comms side (gated).
4. 3s at the full 6s/145f/1080p deliverable will likely need L4 **stacked** with a frame-count or
   further step-schedule decision — flag as a product call, since 145f = the 6.04s clip.
