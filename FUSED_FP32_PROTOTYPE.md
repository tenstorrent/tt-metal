# Fused FP32 K-chunk prototype for the ACCURATE / BALANCED SDPA recipes (research, not for merge)

Base: SDPA precision recipe stack, #60198 tip 5dd8d6a0fa1. Enabled per run with env `TT_SDPA_FFP32=1`; extra kernel
knobs through `TT_SDPA_EXTRA_DEFINES="A;B=v"` (e.g. `FFP_QK_FID=3`, `FFP_QUAD`, `FFP_GRID9`, `FFP_OS`). Default env runs
the unchanged recipes. Code: `device/kernels/compute/streaming/recipe_fused_fp32.hpp` (new), dispatch in
`recipe_streaming.hpp`, exp grid/refine knobs in `recipe_sfpu.hpp`, host hook in `sdpa_recipe.cpp`.

## Design
- Reference max = exact TF32 row max of the Q chunk's first (reduce-path) K chunk. One extra QK inner step (1 fidelity
  phase, inner half) multiplies [Q | M2] x [K^T ; -e0 - e1], where M2 holds the max as hi (7 significant bits) + lo
  (m - hi), so DEST receives s - m exactly in FP32: no S pack, no S - m pass, no max reduce on fused chunks.
- FP32 exp grid with 9 fraction bits: window [-21.5, +22.9] nat (vs [-21.5, +0.72]), grid ripple +-0.068 %, so scores
  above the reference max are representable.
- Denominator kept (P·1, one tile per row group into a scratch CB the UNPACK thread scans). A row sum >= 2^24 redoes
  that group on the reduce path (real max; O and l rescaled; new reference). Unchanged groups' sums fold into l once at
  chunk end. Pipeline: QK/exp of group g || denominator + check of g-1 || PV of g-2; O and l updated in place.
- Bugs fixed on the way: UNPACK RISC must `tensix_sync()` before reading the new max from L1 after the pack->unpack
  barrier; a "add sums before the check, subtract in the redo" scheme gives non-finite outputs (replaced by the fold).

## Results (Blackhole P150, bh-31)
Full grid, ms (base -> fused):

| shape | BALANCED | BALANCED + QK HiFi3 + quad exp refine | ACCURATE |
|---|---|---|---|
| Wan2.2 720p | 37.88 -> 33.39 (1.13x) | 30.24 (1.25x) | 50.24 -> 39.85 (1.26x) |
| MiniMax-H3 5 s | 24.96 -> 21.42 (1.17x) | 19.71 (1.27x) | 32.95 -> 25.56 (1.29x) |
| D64 S8192 | 5.62 -> 4.38 (1.28x) | 3.99 (1.41x) | 6.74 -> 4.81 (1.40x) |

18-case suite worst/median rel-L2 %: BALANCED .483/.382 -> .430/.318; HiFi3+quad .599/.345 -> .449/.324;
ACCURATE .420/.180 -> .407/.183. Tests passing with fusion on: large_logits, offset_rows, accuracy, nightly rising_max.
Single core (resident): fused loses at K128/K256 for BALANCED D128 (-6..-13 %, fixed per-row-group cost: reconfigs,
saturation check, M2 setup/syncs, pipeline fill/drain, end-of-chunk fold) -> production should fuse only at K >= 512.

## Open problems
- Kernel size: dense programs fit (max 65.9 KB), but key-range builds (causal / sliding window / chunked) are
  90.7-94.4 KB vs the 70656 B kernel config buffer (`-Os` on the fused chunk: 84.8 KB and -10 % speed). Needs a
  code-size project (out-of-line cold paths, redo shared with the reduce path) before causal/ring can fuse.
- Not done: BFP8/BFP4 K (the -1 tile), Wormhole, ring/exp ring, host chooser gating (K >= 512), cutting per-group
  fixed cost (per-chunk saturation check) to move the break-even toward K256.
- Effort estimate for production: dense K >= 512 path ~1.5-2 weeks; key-range/ring size work +1-2 weeks.

Full notes (bench harness, knockouts, kernel sizes): see `docs/FUSED_FP32_NOTES.md` in this branch.
