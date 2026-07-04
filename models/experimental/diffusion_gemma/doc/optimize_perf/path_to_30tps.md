# DiffusionGemma — path to 30 t/s (roadmap, #47465)

Rigorous projection of the DiffusionGemma 26B-A4B-it decode throughput from the current baseline
toward the 30 t/s target on QB2 (`bh-qbge-06`, P150x4, mesh `(1,4)`, TP=4). **Analysis + roadmap
only — no device workload run here.** All device numbers are cited from the already-measured
artifacts in this directory: `perf_progress.md` (Lever A true-sparse MoE, LANDED), `README.md`,
`work_log.md` (§3 roofline, §3c traced per-block), `perf_summary.json`, and the bench sources
(`bench_gather_moe.py`, `bench_ondevice_dispatch.py`).

Optimization unit = **denoise step over the 256-token canvas** (≤48 steps/block) **+ commit**.
`t/s` = tokens-per-block ÷ block-seconds = `256 / (ms_per_block / 1000)`. 30 t/s ⇔ **`ms_per_block ≈ 8533 ms`**.

## Established baseline (fixed 48-step, traced)

| quantity | value | source |
|---|---|---|
| per denoise step (dense-128 MoE, 30L, traced) | **4175.7 ms** | `work_log.md` §3c |
| — per-layer denoise | 137.55 ms/layer | §3c |
| — of which dense-128 experts forward | 137.60 ms/layer (~99% of the layer) | `perf_progress.md` bench_moe |
| — fixed overhead (embed + self-cond + LM-head + final norm + terminal sampling) | 49.24 ms/step | §3c |
| commit (256 single-token decode-appends, 30L) | 31,472 ms/block | §3c |
| denoise @ 48 steps | 200,434 ms | §3c |
| **ms_per_block @ 48 steps** | **231,906 ms** | §3c |
| **t/s @ 48 steps** | **1.10** | §3c |

Real blocks halt adaptively at **18–38 steps** (observed `[27]`, `[36,18]`, `[38]`), giving the
1.3–2.3 t/s seen in `ttft_ts_sweep.md`. The fixed-48 budget (required by the static Metal trace) is
the pessimistic anchor; the whole table below is stacked on it for apples-to-apples comparison, and
step count is treated as an explicit lever (row 3).

The **decisive landed lever** is the true-sparse token-gather MoE (`perf_progress.md` Lever A): the
dense-128 experts forward (137.60 ms/layer, the wall) is replaced by an on-device GShard-capacity
token-gather MoE at **10.54 ms/layer trace-safe** (13.0×, PCC 0.99969). Per-layer denoise therefore
drops 137.55 → ~11 ms/layer and the per-step drops toward ~0.38 s. This is the foundation of the
projection.

---

## (a) Roofline of the TRUE-SPARSE path

The single most important — and counterintuitive — roofline fact: **true-sparse does NOT reduce
per-step weight traffic for a 256-token canvas.** The win is compute and data-movement, not bytes.

### Weight-byte model (bf16, 2 B/param, TP=4, per chip)

From `work_log.md` §3 (H=2816, 30 layers, 128 experts × moe_inter 704, shared_inter 2112, vocab 262144):

| component | params | bf16 GB total | GB / chip |
|---|---|---|---|
| non-expert (embed + LM-head + 30× attn/shared-MLP/router/norms) | 2.32B | 4.64 | **1.16** |
| experts (30 × 128 × gate/up/down) | 22.84B | 45.68 | **11.42** |
| **model total** | **25.16B** | **50.3** | **12.58** (matches 13.24 GiB build) |

### Why the expert weight floor is unchanged under top-8 gather

A 256-token canvas with top-8 routing produces `256 × 8 = 2048` (token, expert) assignments over
128 experts. Expected distinct experts activated:

```
E[distinct] = 128 · (1 − (1 − 1/128)^2048) ≈ 128 · (1 − e^-16) ≈ 128.0
```

**Essentially all 128 experts get tokens** (≈16 each; capacity C=32 captured all 2048 pairs, 0
dropped, PCC 0.9997). The token-gather path in `bench_gather_moe.py` therefore still iterates
`E=128` experts (each on its 32-token capacity tile) and **reads the full expert bank once per
step** — the same 11.42 GB/chip the dense path reads. Top-8 sparsity would only cut *weight* traffic
if fewer than 128 experts were active, which does not happen for a canvas this wide.

### The true-sparse per-step weight floor (unchanged) vs the measured step

| scenario | GB/chip/step | @256 GB/s | @512 | @1024 (QB2 peak) |
|---|---|---|---|---|
| all 128 experts (dense **and** true-sparse, 256-canvas) | 12.58 | **49.1 ms** | 24.6 ms | 12.3 ms |
| hypothetical 8/128 experts (perfect reuse — NOT reached at S=256) | 1.88 | 7.3 ms | 3.7 ms | 1.8 ms |
| dense-only (no experts) | 1.16 | 4.5 ms | 2.3 ms | 1.1 ms |

- Dense-128 measured 4175.7 ms/step → **85× the 49 ms @256 floor** (op/movement bound).
- True-sparse measured ~379 ms/step → **~7.7× the 49 ms @256 floor** (31× the @1024 peak floor).

The step moved from *deeply op-count bound* toward *weight-bound*, but ~6–7× headroom to the @256
weight roofline remains — and it is no longer in the expert matmul.

### Where the 13× actually came from (compute + transpose, not weights)

| | dense-128 | true-sparse gather |
|---|---|---|
| token-expert products / layer | 8 chunks × 128 × 32 = **32,768** | 128 × 32-cap = **4,096** (8× fewer) |
| expert-major transpose (`Permute`) | ~87 ms/layer (the dominant op) | **eliminated** (experts are the leading batch dim) |
| gather + scatter overhead | — | ~2 ms/layer (measured, not a washout) |
| on-device dispatch build (topk→scatter→cumsum→gather→col-math) | — | 1.87 ms/layer |
| measured MoE | 137.6 ms/layer | **10.54 ms/layer** |

### The new per-step floor and its composition

Per-layer denoise after the swap = `137.55 − 137.60 + 10.54 ≈ 10.5 ms` (≈11 ms conservative,
including the small non-expert remainder: router, shared-MLP, attention, norms). Per step:

```
new per-step ≈ fixed_overhead + 30 × per-layer
            ≈ 49.24 ms + 30 × 11.0 ms
            ≈ 379 ms   (range 364–379 ms; ≈0.38 s)
```

Composition of the ~379 ms step (this is the new bottleneck map):
- **~330 ms (87%)** — 30 layers × ~11 ms (sparse MoE 10.54 + attn/norms/shared-MLP remainder). Of
  this, the sparse MoE per-layer is ~6.6× the 1.6 ms/layer @256 weight roofline — residual headroom
  is gather/scatter (~2 ms), combine-matmul, per-layer all-reduce (TP=4), and dispatch (~1.9 ms).
- **~49 ms (13%)** — fixed overhead: LM head over 262144 vocab (produces the `[256,262144]` logits),
  terminal sampling (42.3 ms: two ROW_MAJOR argmaxes + entropy + accept/renoise), self-cond, final
  norm. **This was 1.2% of the dense step; it is now the #2 cost.**

---

## (b) Stacked-lever projection

Each row adds one lever to the row above. Assumptions are stated per row. Canvas = 256 tokens/block
throughout; 30 t/s ⇔ `ms_per_block ≈ 8533 ms`.

| # | configuration | per-step | denoise (48 steps unless noted) | commit | **ms_per_block** | **t/s** | multiplier assumption |
|---|---|---|---|---|---|---|---|
| 0 | **Baseline** dense-128, 48 steps | 4175.7 ms | 200,434 ms | 31,472 ms | **231,906** | **1.10** | measured (§3c) |
| 1 | **+ true-sparse denoise MoE** (LANDED) | 379 ms | 18,204 ms | 31,472 ms | **49,676** | **5.15** | 137.6→10.54 ms/layer MoE = 13.0× measured; per-step 4175.7→379 (**11.0×**). Commit **unchanged**. |
| 2 | **+ commit-batching** (256 decodes → 1 causal 256-tok prefill-append) | 379 ms | 18,204 ms | 4,496 ms | **22,700** | **11.28** | commit 31.5 s→4.5 s = **7×** (one dense backbone pass amortizes weight traffic once instead of 256×; sub-win: LM-head skip already LANDED). |
| 3 | **+ step-count reduction** 48 → 24 | 379 ms | 9,102 ms | 4,496 ms | **13,598** | **18.83** | **~2×** on the denoise budget. Real blocks halt at 18–38 steps; 24 is mid-range. Free IF the trace-fixed-budget ↔ adaptive-early-halt tension is solved; otherwise a `max_denoise_steps` config cut (quality risk). |
| 4 | **+ 2CQ / terminal-overlap / dedup** | ~330 eff | — | — | **11,824** | **21.65** | **~1.15×** on the block. After true-sparse the step is no longer 99% MoE; the ~28 ms host readback + LM-head/logits I/O + one redundant full-vocab argmax (RUN-first: sampled == clean argmax, ~14 ms) can overlap/dedup. Low headroom on the dense step (Lever 1/5) but more now the step is small. |

**Result of the requested stack: ~21.7 t/s.** Remaining gap to 30 t/s: `11,824 / 8,533 = 1.39×`.

Reading the stack:
- Row 1 → the block is suddenly **63% commit** (the 31.5 s commit becomes the dominant term the
  instant the step gets cheap). This is why commit-batching (row 2) is the mandatory *second* lever.
- Row 2 → back to **80% denoise** at the fixed-48 budget; step count (row 3) is now the lever.
- Rows 3–4 are the ones with *soft* multipliers (data-dependent step count; established-low-headroom
  2CQ). Rows 1–2 are the *hard*, structural, no-quality-tradeoff wins.

---

## (c) Remaining gap to 30 t/s + ranked additional levers

After the requested stack we are at **~21.7 t/s (11,824 ms/block)**; 30 t/s needs **8,533 ms/block**,
a further **1.39×**. Ranked by expected impact × confidence, with the in-repo / out-of-repo boundary
flagged (the hard rule: **no `models/demos/gemma4/` or shared-ttnn edits**):

| rank | lever | mechanism | expected | risk | in-repo? |
|---|---|---|---|---|---|
| 1 | **Sparse commit** | Extend token-gather MoE to the batched *causal* 256-token commit: commit 4.5 s → ~0.4 s (~12×). Removes commit as a floor entirely. | commit ~12× (block 4,496→~400 ms) | **med** — commit is causal + writes KV; token-gather proven only for the bidirectional denoise canvas so far. Needs causal-attention + `paged_update_cache` validation and a capacity-safety PCC. | ✅ DiffusionGemma-local (`tt/commit_decode.py`) |
| 2 | **Effective step-count → adaptive ~20** | Recover data-dependent early-halt (real blocks halt 18–38) inside a trace-safe shape: multi-length captured traces or chunked-halt at trace-segment granularity. Free (no quality loss) vs a blunt `max_denoise_steps` cut. | up to **~2.4×** on denoise | **med** — collides with static-trace fixed budget (skill difference #4). Data-dependent: hard prompts may still need ~40 steps. | ✅ `tt/denoise_loop.py` |
| 3 | **DRAM-sharded gather/combine matmul + all-reduce tuning** | The sparse MoE is now ~6.6× the @256 weight roofline; the gather-matmul, combine-matmul and per-layer TP=4 all-reduce carry the residue. Sweep `in0_block_w`, core grid, persistent CCL buffers (OPT-004/009). | ~1.2–1.5× on the MoE | low–med | ✅ DiffusionGemma-local program configs |
| 4 | **bfp8 experts in the sparse path** | Precision was ruled out for the *dense* path (transpose-bound, weight was only 1.6 ms). Post-transpose the sparse path is closer to weight-bound, so halving expert bytes (11.42→5.71 GB/chip) now *can* help the weight component. | ~1.1–1.3× on the MoE | **med** — #48291 fidelity is already marginal; bfp8 small-prob drift can flip a diffusion accept/renoise decision. Must gate on decision-fidelity, not top-k. | ✅ local dtype policy |
| 5 | **Terminal-path trim** | Dedup the two full-vocab argmaxes (RUN-first sampled == clean argmax, ~14 ms/step), fuse entropy intermediates, shard the LM head tighter. Now ~13% of the step. | ~1.05–1.1× on the step | low | ✅ `tt/sampling.py`, `tt/denoise_loop.py` |
| 6 | **Kernel-level fused MoE** | A `sparse_matmul` variant emitting the down-projection layout directly (kill the residual reorder), or a fused gather-experts-combine kernel, would push the per-layer MoE from 10.54 ms toward the ~1.6 ms @256 (or 0.4 ms @1024) weight roofline. | ~1.5–3× on the MoE | high | ❌ **shared ttnn/gemma4 — violates the no-edits gate; an upstream contribution** |

Levers 1–2 are what close the 1.39× gap (see verdict). Levers 3–5 are the polish that buys margin
for hard prompts. Lever 6 is the only path that is *out of scope* — and, notably, the token-gather
approach already bought 13× **without** it.

---

## (d) Honest verdict — is 30 t/s reachable purely in-repo?

**Firm answer: the two clean, structural, no-quality-tradeoff in-repo levers get you to ~11 t/s, not
30. Reaching 30 t/s in-repo is possible but not guaranteed — it is contingent on two further in-repo
rewrites landing AND on real blocks halting near ~20 steps.**

### The firm in-repo ceiling: ~11 t/s

True-sparse denoise MoE (**LANDED**, DiffusionGemma-local, 13×, PCC 0.9997) + commit-batching
(scoped, DiffusionGemma-local, ~7×) take the block from **1.1 → ~11.3 t/s at the pessimistic fixed-48
budget, with no quality tradeoff and no gemma4/kernel edits.** This is the number to state with high
confidence today. Both compose *over* the untouched backbone (`git diff main -- models/demos/gemma4/`
stays empty), so neither is gated on shared-directory work.

### Reaching 30 t/s in-repo: arithmetically yes, in the favorable regime

Stacking the two additional in-repo levers (sparse commit + adaptive step count) on top:

```
sparse denoise:  379 ms/step
sparse commit:   ~0.4 s  (token-gather in the causal commit)
adaptive steps:  ~21     (mid of the observed 18–38 halt range)
  →  21 × 379 + 400 = 7,959 + 400 = 8,359 ms/block  →  30.6 t/s   ✓
```

or, equivalently, the row-3 config (24 steps) + sparse commit + row-4 2CQ:
`(24 × 379 + 400) / 1.15 = 8,262 ms → 31.0 t/s`. **30 t/s is arithmetically in-repo-reachable.**

But it stands on a conjunction of soft conditions:
1. **Sparse commit delivers ~12×** on the commit (unproven for the causal + KV-append path).
2. **Blocks halt at ≤~22 steps** — data-dependent. Easy prompts (observed `[18]`) clear it; a hard
   prompt needing ~40 steps lands at ~13–16 t/s even with everything else.
3. **The trace-fixed-budget ↔ early-halt tension is solved** without giving up the trace-safe shape.

So the realistic in-repo range is **~11 t/s (firm) → ~20–31 t/s (favorable regime, easy/median
prompts)**. Purely mechanical, all-prompts, no-quality-tradeoff 30 t/s at the full 48-step budget is
**not** reachable in-repo.

### What 30 t/s would *robustly* require (all prompts, 48-step budget)

Either of:
- **Fewer denoise steps by design** — train/config the diffusion schedule for ~16–20 steps. This is
  a model/quality decision (fewer refinement iterations), not a perf edit, and it is the cheapest
  robust route: at 379 ms/step, `18 × 379 + 400 (sparse commit) = 7,222 ms → 35 t/s` for *every*
  block regardless of content.
- **Kernel-level fused MoE** (out of the no-edits gate; shared ttnn/gemma4). Pushing the per-layer
  MoE from 10.54 ms toward ~2–3 ms (still 1.5–2× above the @1024 weight roofline) gives a ~139 ms
  step, so even the pessimistic fixed-48 budget clears it: `48 × 139 + 400 = 7,072 ms → 36 t/s`.
  This is the only way to hit 30 t/s robustly at 48 steps **without a quality tradeoff** — and it is
  the one lever that must leave the module.

### Bottom line

| target | reachable in-repo, no quality tradeoff? | how |
|---|---|---|
| ~5 t/s | ✅ firm | true-sparse denoise MoE (LANDED) |
| ~11 t/s | ✅ firm | + commit-batching |
| ~20 t/s | ✅ likely | + sparse commit + adaptive step count (median prompts) |
| **30 t/s** | ⚠️ favorable-regime only | + all of the above, easy/median prompts halting ≤~22 steps; **NOT** for hard prompts at 48 steps |
| 30 t/s, all prompts, 48-step | ❌ needs out-of-scope work | fewer denoise steps by design (quality/model decision) **or** a kernel-level fused MoE (shared ttnn/gemma4) |

The campaign's structural bet — token-gather true-sparse MoE — already bought the hardest 13× **in
repo, without a kernel edit**. The remaining distance to 30 t/s is not another structural wall; it is
(1) doing the same token-gather trick to the commit, (2) not paying for 48 steps when a block
converges in ~20, and (3) accepting that the last ~1.4× for *hard* prompts is a step-count (quality)
or kernel-fusion (out-of-repo) decision, not a further in-module optimization.
