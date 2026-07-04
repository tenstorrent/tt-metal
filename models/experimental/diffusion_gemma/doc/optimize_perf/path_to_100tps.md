# DiffusionGemma — path to 100 t/s (roadmap, #47465)

Rigorous projection of DiffusionGemma 26B-A4B-it decode throughput from the **current landed
state** (true-sparse token-gather MoE, `tt/sparse_moe.py`) toward a **100 t/s** target on QB2
(`bh-qbge-06`, P150x4, mesh `(1,4)`, TP=4). **Analysis + code-reading + roofline only — no device
workload run here** (the QB2 box is owned by another agent). Every device number is cited from the
already-measured artifacts in this directory: `perf_progress.md` (Lever A landed, Lever B scoped),
`path_to_30tps.md` (roofline + stacked levers), `work_log.md` (§3 roofline, §3c traced per-block),
`bench_ondevice_dispatch.py`, `bench_gather_moe.py`, and the source of `tt/sparse_moe.py`.

Optimization unit = **denoise step over the 256-token canvas** (≤48 steps/block) **+ commit**.
`t/s = 256 / (ms_per_block / 1000)`. **100 t/s ⇔ `ms_per_block ≈ 2560 ms`.**

This doc is the successor to `path_to_30tps.md`. That doc established the in-repo ceiling at
~11 t/s (fixed-48 budget) and ~20–31 t/s in the favorable early-halt regime. 100 t/s is a **~3.3×
further** step on top of the full 30 t/s stack, and the levers change character: at 100 t/s the
denoise step must fall to ~120–150 ms, at which point **the terminal/fixed overhead and the
per-layer MoE efficiency gap — not the weight floor — are the binding constraints.**

---

## Current landed state (the starting line for this doc)

| quantity | value | source |
|---|---|---|
| sparse MoE (on-device dispatch, trace-safe) | **10.54 ms/layer** (13.0× vs dense 137.6) | `perf_progress.md` Lever A; `bench_ondevice_dispatch.py` |
| per-layer denoise (sparse MoE + attn/norm/router/shared) | ~11 ms/layer | `path_to_30tps.md` (a) |
| per denoise step (30L, traced, F=fixed overhead) | **~379 ms** (461 ms measured L2/L4 fit) | `path_to_30tps.md`; `prof_denoise_step.py` |
| — fixed overhead (embed + self-cond + LM-head + final norm + terminal sampling) | **49.24 ms/step** | §3c |
| — of which terminal sampling (2× argmax@262144 + entropy + accept/renoise) | **42.3 ms** | `path_to_30tps.md` (b) |
| commit (256 single-token decode-appends, 30L) | **31.4 s/block** (commit-bound) | §3c; `perf_progress.md` |
| **ms_per_block @ 48 steps** | ~49,700 ms → **4.78 t/s** | `perf_progress.md` Lever A |
| block @ early-halt (18–25 steps) | ~6 t/s | ibid |

The block is currently **commit-bound** (commit 31.4 s = 59% of the 53 s block). The `path_to_30tps`
plan closes that with commit-batching / sparse commit and reaches ~11 t/s (firm) → ~20–31 t/s
(favorable). **This doc assumes that whole 30 t/s stack is landed** and asks what a further 3.3×
requires.

---

## (a) Roofline for 100 t/s

### The target decomposed

With a **sparse commit** (Lever B / Lever-1 of `path_to_30tps`, a 256-token causal prefill-append
that reuses the sparse MoE — one denoise-step-equivalent of compute), a block is:

```
ms_per_block ≈ (steps + 1_commit) × step_ms         (commit ≈ 1 sparse step)
step_ms      ≈ fixed_overhead + 30 × per_layer_ms
100 t/s      ⇔ ms_per_block = 2560 ms
```

Solving for the **required per-layer MoE time** at each step count (two fixed-overhead assumptions):

| steps | commit | step_ms budget | **per_layer budget** (fixed=49 ms, today) | **per_layer budget** (fixed=25 ms, trimmed) |
|---|---|---|---|---|
| 48 (full trace budget) | +1 | 52 ms | **infeasible** (step < fixed) | 0.9 ms (< @256 floor) |
| 24 | +1 | 102 ms | 1.77 ms | 2.57 ms |
| 20 | +1 | 122 ms | 2.43 ms | **3.23 ms** |
| 16 | +1 | 151 ms | 3.40 ms | **4.19 ms** |
| 12 | +1 | 197 ms | 4.93 ms | 5.73 ms |

**Reading the table.** 100 t/s is **arithmetically impossible at the full 48-step budget** — the
per-step budget (52 ms) is barely above today's *fixed overhead alone* (49 ms). 100 t/s is a
**short-step regime target**: it needs early-halt/schedule to ~16–20 steps AND a per-layer MoE of
~3–4 ms (from today's ~11) AND the fixed overhead roughly halved (49 → ~25 ms). None of the three
alone suffices; they multiply.

### The weight-traffic floor (@256 and @1024 GB/s/chip)

Weight-byte model (bf16, 2 B/param, TP=4, per chip), from `work_log.md` §3 and `path_to_30tps.md`
(a). Experts are TP-sharded on the `moe_intermediate` dim (704 → padded 96×4=768/chip); expert
weights = 128 × 3 × (2816 × 704) = **761 M params/layer**, /4 chips × 2 B = **415 MB/chip/layer**
(padded).

**Top-8-expert floor (the number the mission asked for).** With true top-8 routing, only 8 experts'
weights would be read *if only 8 distinct experts were active*:

| granularity | GB/chip | @256 GB/s | @1024 GB/s (QB2 peak) |
|---|---|---|---|
| **top-8 distinct**, per layer | 0.0259 | 0.10 ms | 0.025 ms |
| **top-8 distinct**, per step (×30) | 0.778 | 3.04 ms | 0.76 ms |
| **all-128** (S=256 reality), per layer | 0.415 | 1.62 ms | 0.41 ms |
| **all-128**, per step (×30 + non-expert LM/norms) | 12.58 | 49.1 ms | 12.3 ms |
| all-128 **bfp8**, per step | 6.3 | 24.6 ms | 6.1 ms |

**The counterintuitive fact carried from `path_to_30tps` (a): at S=256 the top-8 floor is
unreachable.** A 256-token canvas with top-8 routing produces 256×8 = 2048 (token, expert) pairs
over 128 experts, so `E[distinct] = 128·(1 − e⁻¹⁶) ≈ 128.0` — **essentially all 128 experts
activate** (≈16 tokens each; capacity C=32 dropped 0, PCC 0.9997). So each denoise step reads the
**full expert bank once**, and the binding weight floor is the **all-128** row, not top-8. Top-8
sparsity buys *compute and data-movement*, never *weight bytes*, for a canvas this wide.

### What the floor says about feasibility

Block weight floor @1024 (all-128 bf16), 20 steps + 1 commit: `21 × 12.3 = 258 ms`. The 100 t/s
target is **2560 ms — a 10× headroom over the weight floor** (20× under bfp8). **100 t/s is not
weight-floor-limited.** The entire distance to 100 t/s is closing efficiency gaps: the ~6.6×
MoE-vs-roofline gap, the fixed/terminal overhead, and the step count. Physics is not the wall here;
implementation efficiency is.

---

## (b) Where the sparse MoE's 10.5 ms/layer goes vs the ~1.6 ms roofline (the ~6.5× gap)

The sparse MoE per-layer is **10.54 ms**, which is **6.6× the 1.62 ms/chip @256 weight roofline**
(26× the 0.41 ms @1024 peak floor). The dense path's dominant cost — the ~87 ms expert-major
`Permute` — is already **gone** (the batched matmul emits experts as the leading batch dim). So the
residual 6.6× is *new* structure, decomposed below from `bench_ondevice_dispatch.py` timings and the
`tt/sparse_moe.py` source.

### Op-level decomposition (from `bench_ondevice_dispatch.py` + `bench_gather_moe.py`)

| phase | ms/layer | roofline attribution | source |
|---|---|---|---|
| dispatch-build (topk→scatter→cumsum→gather→col-math→scatter) | **1.87** | pure overhead (not in roofline) | `RESULT_DISPATCH_BUILD` |
| gather-matmul `disp^T @ hidden` + combine-matmul `comb @ down` + TP all-reduce | **~1.7** | pure overhead (scatter/gather realized as dense matmuls) | full − batched − dispatch |
| **batched experts** (gate/up/geglu/down over 128 experts) | **~8.9** | contains the 1.62 ms weight read + ~7.3 ms inefficiency | `bench_gather_moe` stage 1 = 8.89 |
| **full on-device sparse MoE** | **10.54** | 6.6× the 1.62 ms floor | `RESULT_FULL_ONDEVICE` |

The gap is two pieces: (1) **~3.6 ms of pure dispatch + gather/combine + all-reduce overhead** that
does not exist in the roofline at all, and (2) **~7.3 ms of inefficiency inside the batched-experts
matmul**, which reads the 415 MB weight bank at only ~46 GB/s effective (415 MB / 8.9 ms) — **~18%
of @256, ~4.6% of @1024 peak.** Both are addressable; neither is fundamental.

### Why the batched-experts matmul is DRAM-inefficient (the dominant ~7.3 ms)

`_batched_experts` (`tt/sparse_moe.py:131–153`) issues three batched matmuls:

```
gate = ttnn.matmul(gathered[1,E,C,H], weights.gate_proj[1,E,H,I])   # sparse_moe.py:137
up   = ttnn.matmul(gathered[1,E,C,H], weights.up_proj[1,E,H,I])     # sparse_moe.py:140
down = ttnn.matmul(down_input[1,E,C,I], weights.down_proj[1,E,I,H]) # sparse_moe.py:146
```

with `E=128`, `C=32` (one 32-row tile), `H=2816` (88 tiles), `I=96/chip` (3 tiles). Each is a
**batch of 128 tiny matmuls**: gate/up are `M=1 tile · K=88 tiles · N=3 tiles`; down is
`M=1 tile · K=3 tiles · N=88 tiles`. Three structural problems:

1. **M = 1 tile (32 rows) ⇒ zero weight reuse across M.** A weight-stationary matmul amortizes a
   loaded weight column over its M rows; here M is the minimum, so each expert's weight is read to be
   multiplied by only 32 tokens (≈16 real, ≈half padding). Arithmetic intensity is floored → the
   matmul is purely DRAM-read-bound, and it doesn't sustain peak DRAM because the reads are small and
   interleaved.
2. **No program config, no output sharding, DRAM-interleaved.** All three calls (and the gather
   `:192` and combine `:204`) pass only `memory_config=ttnn.DRAM_MEMORY_CONFIG` and
   `compute_kernel_config` — **no `program_config`, no core-grid choice, no L1-sharded output**
   (`sparse_moe.py:137–151, 192, 204`). This path was a functional GO/NO-GO prototype
   (`perf_progress.md` Lever A); it has **never had OPT-004 matmul-geometry tuning**. Contrast the
   dense path, which at least builds an explicit `MatmulMultiCoreReuseMultiCast1DProgramConfig` per
   role (`decode.py:20–52`, though with `in0_block_w=1`, itself a floor per OPT-004).
3. **128 experts likely (partly) serialize across the core grid.** A default 4D batched matmul with
   `M=1 tile, N=3 tiles` per batch element does not obviously pack 128 experts onto a 64-core grid;
   if experts run largely one-at-a-time, per-launch overhead dominates — consistent with the ~46
   GB/s effective read.

### The gather/combine/dispatch overhead (~3.6 ms)

- **Gather as a matmul** (`sparse_moe.py:190–192`): `disp^T[1,1,EC,S] @ hidden[1,1,S,H]` with
  `EC = 128×32 = 4096`. This is a 4096×256×2816 dense matmul whose *only* job is to permute 2048
  real rows into per-expert bands; it moves a 23 MB output and a 2 MB one-hot mask that carry no
  model information.
- **Combine as a matmul** (`sparse_moe.py:203–204`): `comb[1,1,S,EC] @ down_flat[1,1,EC,H]`, the
  mirror-image 256×4096×2816 matmul that scatters back weighted by the routing scores.
- **Dispatch-build** (`build_capacity_dispatch`, `sparse_moe.py:70–128`): topk (`:97`) → scatter
  mask (`:102`) → cumsum (`:105`) → gather slot (`:110`) → column math (`:113–118`) → two scatters
  (`:121–122`). ~1.87 ms of many small ops.
- **Per-layer TP=4 all-reduce** (`sparse_moe.py:209–210`) after the row-parallel down projection.

These ~3.6 ms are the *cost of doing gather/scatter with dense ops* because TTNN has no
gather-experts primitive that the sparse path could call. They are the strongest argument for a
fused kernel (lever below).

**Summary of the 6.6× gap:** ≈7.3 ms untuned batched-matmul DRAM inefficiency (in-repo,
OPT-004) + ≈3.6 ms dispatch/gather/combine/all-reduce overhead (part in-repo tuning, part a fused
kernel) — sitting on top of a 1.62 ms weight floor. **None of it is the weight bank; all of it is
op geometry and gather realization.**

---

## (c) Candidate levers to close the gap

Each lever: mechanism, expected multiplier, **in-repo vs upstream** (the hard gate:
`git diff main -- models/demos/gemma4/` must stay empty; no shared-ttnn edits without upstreaming),
and risk. "In-repo" = DiffusionGemma-local (`tt/sparse_moe.py`, `tt/commit_prefill.py`,
`tt/denoise_loop.py`, `tt/sampling.py`, local program configs / dtype policy) composing over the
untouched backbone.

| # | lever | mechanism | expected | in-repo? | risk |
|---|---|---|---|---|---|
| 1 | **Tune the 5 batched matmuls (OPT-004)** | Add explicit `program_config` + core-grid + **L1-sharded outputs** to `_batched_experts` gate/up/down (`sparse_moe.py:137–151`) and the gather/combine matmuls (`:192,:204`). Pack the 128 experts across the 64-core grid; raise `in0_block_w` above the `decode.py:20` floor of 1; keep intermediates in L1 (`bench_moe_l1.py` showed L1 barely moved the *dense* path, but the sparse path is now DRAM-round-trip-bound, a different regime). | **1.8–2.5×** on the MoE (10.5 → ~4–6 ms) | ✅ local | **low–med** — pure config; the batched-matmul core-packing behaviour is the uncertainty. |
| 2 | **DRAM-sharded expert weights for the batched matmul** | Weights load DRAM-*interleaved* (`weights.py:117,126,135`). A DRAM-width-sharded expert layout + matching L1-sharded activation lifts effective BW toward 60–75% of peak (OPT-004). | **1.5–2×** on the batched experts | ⚠️ local **but memory-gated** | **med** — needs a local DRAM-sharded expert copy (+~11.4 GB/chip → OOM risk vs the 13.24 GiB build + 256K KV) OR a batched-matmul-compatible shard scheme; batched matmul does not map cleanly to the 2D DRAM-sharded matmul primitive. |
| 3 | **bfp8 experts in the sparse path** | The sparse path is now (partly) weight-bound (unlike the transpose-bound dense path where bfp8 was ruled out — `perf_progress.md` lever-4). Halve the 415 MB read → 207 MB. Also *halves* the DRAM-sharded copy in lever 2, resolving its OOM tension. | **1.3–1.5×** on the MoE weight component | ✅ local dtype policy | **med** — #48291 fidelity is already marginal (argmax ≈50% vs HF); bfp8 small-prob drift can flip an accept/renoise decision. **Must gate on decision-fidelity, not top-k.** |
| 4 | **Terminal-path trim** | Fixed overhead (49 ms) is ~40% of the *target* 122 ms step. Dedup the two full-vocab argmaxes over 262144 (RUN-first: sampled == clean argmax, ~14–20 ms), fuse entropy intermediates, vocab-shard the LM head tighter (LMHead1D). | **~2×** on fixed (49 → ~25 ms) | ✅ `tt/sampling.py`, `tt/denoise_loop.py` | **low** — RUN-first dedup is safe; verify accept/renoise unchanged. |
| 5 | **Fused gather-experts-combine kernel** | Replace the two dense gather/combine matmuls (`:192,:204`) + the batched matmul with **one kernel** that reads each expert's C tokens by index (indexed DMA), runs gate/up/geglu/down, and scatters back — killing the ~3.6 ms overhead AND the EC=4096-wide intermediates, pushing per-layer toward the 1.6 ms floor. | **1.5–3×** on the MoE | ❌ **upstream** shared ttnn kernel (a partial index-DMA gather *might* be built local from ttnn data-movement primitives — investigate first) | **high** — new low-level kernel; trace-safety + TP correctness. |
| 6 | **`sparse_matmul` variant emitting the down layout directly** | The dense path's 87 ms `Permute` came from `sparse_matmul` emitting expert-major and needing `transpose(1,3)` (`prefill.py:93,108`) + `permute` (`:134`). A `sparse_matmul` that natively (a) accepts per-token top-8 sparsity (today `TT_FATAL: sparsity.logical_volume()==batch_length`, `perf_progress.md` lever-3) and (b) emits the down-proj layout would make the *dense* sparse path competitive without the token-gather scaffolding — a cleaner floor than levers 1–2. | **1.5–2×** vs current sparse MoE | ❌ **upstream** shared ttnn `sparse_matmul` op change | **high** — op-contract change; large blast radius. |
| 7 | **Sparse commit (causal prefill-append)** | Carried from `path_to_30tps` Lever-1 / `perf_progress` Lever B. Extend token-gather MoE to the causal 256-token commit: 31.4 s → ~0.6–1.0 s (~30–50×). **Prerequisite** for the (a) block model above (commit ≈ 1 step). | commit ~30–50× | ✅ `tt/commit_prefill.py` (scoped) | **med** — offset RoPE ✓, needs prefix-attending SDPA + offset KV write + KV-PCC gate. |
| 8 | **Adaptive / fewer steps** | Recover data-dependent early-halt (18–38 observed) inside a trace-safe shape (multi-length captured traces / chunked-halt), or a schedule/config cut to ~16–20 steps. The single largest block-level multiplier. | **~2–2.4×** on denoise | ✅ `tt/denoise_loop.py` (mechanism) / ⚠️ model decision (quality) | **med** — trace-fixed-budget ↔ early-halt tension (skill diff #4); hard prompts may still need ~40 steps. |
| 9 | **Larger batch + 2 command queues** | 2 CQ overlaps the ~28 ms host readback + LM-head/logits I/O with compute; batching independent blocks/users raises M per expert matmul (amortizes the weight read — the one way to beat the M=1-tile floor without fewer steps). | **~1.1–1.3×** on the block | ✅ orchestration | **low–med** — batch changes the sparse-MoE capacity math (more tokens/expert → bump C); preserve batch-1 latency (OPT-005). |
| 10 | **Multi-denoise-step trace batching** | Capture N denoise steps as one trace to amortize dispatch/launch across steps. Complementary to 2 CQ; helps the many-small-op sparse path (which already gained eager→traced 20→13.7 ms/layer, `perf_progress.md` Lever A). | **~1.1–1.3×** on the step | ✅ `tt/denoise_loop.py` | **med** — steps are data-dependent (canvas N feeds N+1); fixed-window capture only. |

**In-repo vs upstream boundary.** Levers 1, 3, 4, 7, 8, 9, 10 are DiffusionGemma-local and clear the
gate. Lever 2 is local but memory-gated. Levers **5 and 6 are the only out-of-gate levers** — shared
ttnn/gemma4 kernel work that must be upstreamed. Notably, the whole token-gather campaign bought 13×
*without* touching a kernel; levers 1–4 are the analogous "tune what we already have" wins and should
be exhausted before reaching for 5/6.

---

## (d) Honest verdict — is 100 t/s reachable?

**Verdict: 100 t/s is reachable, but only in the favorable early-halt regime and only if an
aggressive-but-plausible in-repo MoE efficiency win lands. It is NOT robustly reachable for all
prompts, and NOT reachable at the full 48-step budget by any in-repo means. If the batched-matmul
efficiency plateaus in-repo above ~5–6 ms/layer, the last ~1.5× requires one upstream lever (fused
MoE kernel) or a model-level step-count reduction.**

### The arithmetic that reaches 100 t/s (favorable regime)

Stacking levers 1+3+4 (per-layer MoE 10.5 → ~4 ms, fixed 49 → ~25 ms) + 7 (sparse commit ≈ 1 step)
+ 8 (steps → ~16):

```
step_ms = 25 (fixed) + 30 × 4.0 (per_layer) = 145 ms
block   = (16 + 1) × 145 = 2465 ms  →  104 t/s   ✓
```

or at 20 steps with per_layer 3.2 ms: `(20+1) × (25 + 30×3.2) = 21 × 121 = 2541 ms → 101 t/s`. So
100 t/s wants the conjunction **per_layer ≈ 3–4 ms · fixed ≈ 25 ms · steps ≈ 16–20 · sparse
commit** — every one of them in-repo, but three of them (per_layer, steps) carrying real risk.

### The make-or-break: can in-repo tuning get per_layer to ~3–4 ms?

This is the single load-bearing uncertainty. Today per_layer ≈ 11 ms (MoE 10.54). The requirement is
2.5–3.5× further. The budget:

- **~3.6 ms of dispatch/gather/combine/all-reduce overhead** must fall to ~1 ms. Lever 1 (tuning the
  gather/combine matmuls) helps; the rest needs lever 5 (fused kernel, **upstream**). This is the
  piece most likely to force an upstream contribution.
- **~7.3 ms of batched-matmul inefficiency** must fall to ~2 ms. Lever 1 (program config + core
  packing + L1 outputs) + lever 3 (bfp8) is the plausible path, but the batched-matmul core-packing
  behaviour is unmeasured — if 128 M=1-tile matmuls fundamentally serialize, in-repo tuning plateaus
  at ~5–6 ms/layer and 100 t/s slips to the ~70–80 t/s band without lever 5.

**Honest confidence:** per_layer to ~5–6 ms in-repo is **likely** (lever 1 + 4 alone); to ~3–4 ms is
**plausible but unproven** and probably needs lever 3 (bfp8, fidelity-gated) and possibly lever 5
(upstream). So the realistic in-repo-only ceiling is **~60–80 t/s** (favorable regime), with 100 t/s
sitting just past it, gated on (a) bfp8 passing decision-fidelity and (b) the batched matmul tuning
delivering, or (c) one upstream kernel.

### What 100 t/s would *robustly* require (all prompts)

At the full 48-step budget 100 t/s is **impossible in-repo** (per-step budget 52 ms < today's 49 ms
fixed overhead). Robust 100 t/s (content-independent) needs **fewer denoise steps by design** — a
model/schedule decision to run ~12–16 steps — which is a quality decision, not a perf edit, plus the
upstream fused MoE kernel to hold per_layer near the ~1.6 ms floor:

```
step_ms = 20 (trimmed fixed) + 30 × 2.0 (fused-kernel per_layer) = 80 ms
block   = (14 + 1) × 80 = 1200 ms  →  213 t/s  (robust headroom, but 2 out-of-gate levers)
```

### Bottom line

| target | reachable, no quality tradeoff? | how |
|---|---|---|
| ~11 t/s | ✅ firm, in-repo | sparse MoE (landed) + commit-batching (`path_to_30tps`) |
| ~20–31 t/s | ✅ favorable regime, in-repo | + sparse commit + adaptive steps (`path_to_30tps`) |
| ~60–80 t/s | ✅ likely, in-repo | + matmul-geometry tuning (lever 1) + terminal trim (lever 4) + steps ≤~18 |
| **100 t/s** | ⚠️ **edge, favorable regime** | + bfp8 experts (fidelity-gated) OR fused MoE kernel (upstream); steps ≤~16; per_layer ≈ 3–4 ms |
| 100 t/s, all prompts, 48-step | ❌ needs out-of-scope work | fewer denoise steps by design (quality) **and** fused MoE kernel (upstream ttnn) |

100 t/s is not blocked by a physics wall (the block sits 10× above the @1024 weight floor). It is
blocked by three efficiency gaps that stack multiplicatively: the untuned batched-matmul geometry
(mostly in-repo, lever 1), the terminal/fixed overhead (in-repo, lever 4), and the step count
(in-repo mechanism / model-decision quality, lever 8). The token-gather win already proved the MoE
is *not* on a structural wall; the remaining distance is OPT-004-grade tuning of a path that has
never been tuned, plus a fidelity-gated bfp8, plus not paying for steps a block doesn't need.

---

## Ranked build plan (concrete, in-repo first)

Ordered by (expected block impact × confidence), in-repo before upstream. Each row cites the file it
lands in and its verification gate. Prerequisites: the full `path_to_30tps` stack (sparse MoE landed;
commit-batching + sparse commit + adaptive steps).

| rank | build | lands in | gate / verify | expected |
|---|---|---|---|---|
| 1 | **Sparse commit (Lever B / `path_to_30tps` L1)** — mandatory: the (a) block model assumes commit ≈ 1 step. | `tt/commit_prefill.py` (new) | KV-cache PCC vs 256-decode path; offset RoPE + prefix SDPA + offset KV write | commit 31.4 s → ~0.8 s |
| 2 | **Matmul-geometry tuning of the 5 sparse matmuls (OPT-004, lever 1)** — the biggest untapped in-repo MoE win; the path has never been tuned. | `tt/sparse_moe.py:137–151,192,204` | traced per-layer before/after; MoE PCC ≥ 0.9997; `tt-perf-report` rows show `in0_block_w`>2, L1 outputs | MoE 10.5 → ~5–6 ms |
| 3 | **Terminal-path trim (lever 4)** — dedup the 2 full-vocab argmaxes; fuse entropy; tighter LM-head shard. | `tt/sampling.py`, `tt/denoise_loop.py` | accept/renoise decisions unchanged vs current | fixed 49 → ~25 ms |
| 4 | **Adaptive / fewer steps (lever 8)** — recover early-halt in a trace-safe shape (or schedule to ~16–20). | `tt/denoise_loop.py` | trajectory decision-fidelity vs torch ref; no host-branch in trace | denoise ~2× |
| 5 | **bfp8 experts, fidelity-gated (lever 3)** — halves the weight read; unblocks lever 2's OOM. Gate hard on #48291 decision-fidelity, NOT top-k. | local dtype policy over `experts.weights` | accept/renoise flip-rate vs bf16 on real activations | MoE ×1.3–1.5 |
| 6 | **2 CQ + multi-step trace batching (levers 9, 10)** — overlap terminal I/O; amortize dispatch across steps; batch preserves batch-1 latency. | orchestration / `tt/denoise_loop.py` | batch-1 latency preserved; batch-32 correctness (OPT-005); capacity C re-checked | block ×1.1–1.3 |
| 7 | **DRAM-sharded expert weights (lever 2)** — only after bfp8 resolves the memory tension; batched-matmul-compatible shard scheme. | local weight copy / shard | build fits vs 256K KV budget (`QB2_MEMORY_BUDGET.md`); MoE PCC | batched experts ×1.5–2 |
| 8 | **Upstream: fused gather-experts-combine kernel (lever 5)** — the out-of-gate lever that removes the ~3.6 ms overhead and pushes per_layer toward the floor. Investigate an in-repo index-DMA gather first; upstream if it needs a new op. | shared ttnn (upstream PR) | trace-safe; TP=4 all-reduce correctness; PCC | MoE ×1.5–3 |
| 9 | **Upstream: per-token / down-layout `sparse_matmul` variant (lever 6)** — alternative floor that retires the token-gather scaffolding entirely. | shared ttnn `sparse_matmul` (upstream) | op-contract PCC; `sparsity.logical_volume` per-token | MoE ×1.5–2 |

**Minimal set to *attempt* 100 t/s in-repo:** ranks 1–5 (sparse commit + matmul tuning + terminal
trim + adaptive steps + bfp8). If ranks 1–5 land and bfp8 passes decision-fidelity and the batched
matmul tunes to ~4 ms/layer, 100 t/s is reached in the favorable (≤16–20 step) regime. If the
batched matmul plateaus above ~5 ms/layer, rank 8 (upstream fused kernel) becomes required, and the
gate forces it out of the module.
