# Tier-1 Chunk-Skip Win Forecast — ttnn.experimental.topk_large_indices (column-parallel)

**Date:** 2026-08-16 · **Silicon:** Blackhole p150a (13×10 grid, 1.35 GHz assumed) · **Method:** exact host-side simulation of the leaf loop + factory split model, calibrated against the five pinned measured cells.
**Simulation:** `sim_tier1_skip.py` (same directory). Baseline model reproduces every measured cell by construction (±0.0 µs).

## Verdict

**Do not build Tier-1 for the pinned column-parallel cells.** The skip fraction is **0.00 % on every realistic distribution** (standard normal = the sweep stimulus, uniform, softmax-peaked, ascending) for **every** cell, under both the safe llk_k-window threshold and the more aggressive (still provably sound) user-k threshold. The only winner is the degenerate all-equal input. On real stimuli the change is a small pure **regression** (+0.1 – +0.4 µs, the skip-test overhead on the critical path). The mechanism is structural, not statistical noise: the factory's own parallelization destroys the opportunity.

## 1. Factory parameters per pinned cell

`snap_to_llk_target_k` (`topk_large_indices_program_factory.cpp:38-46`): k ≤ 512 → 512, ≤ 1024 → 1024, else 2048. Split: a×b rectangle search, `cost(P) = 2*ceil(chunks/P) + ceil(log2 P)`, P ∈ [2, min(chunks, 128)], ties prefer fewer cores (`:396-410`, cap `max_column_slices = 128` at `:330`). Chunks distributed contiguously, base + (s < extra) (`compute_slice_runtime`, `:500-531`).

| cell | llk_k | chunks | P (rect) | chunks/slice | tree levels | later (testable) chunks/slice |
|---|---|---|---|---|---|---|
| k32@65536 (routed) | 512 | 128 | 64 (8×8) | 2 | 6 | 1 |
| k64@65536 (routed) | 512 | 128 | 64 (8×8) | 2 | 6 | 1 |
| k512@262144 | 512 | 512 | 104 (13×8) | 5/4 | 7 | 4/3 |
| k512@65536 | 512 | 128 | 64 (8×8) | 2 | 6 | 1 |
| k2048@262144 | 2048 | 128 | 64 (8×8) | 2 | 6 | 1 |
| (k2048@65536) | 2048 | 32 | 32 (4×8) | 1 | 5 | **0 — no opportunity, confirmed** |

## 2. Skip fractions (simulated, exact bf16 sign-magnitude streaming)

Skip rule as specified: `chunk_max <= running K-th survivor min`, chunk 0 always seeds. Two threshold variants:

- **cons** — K = llk_k (the survivor window the DST actually holds; safe by construction).
- **aggr** — K = user k (sound: an element below the leaf's running user-k-th largest can never enter the global top-k — fewer than k elements beat any global winner, so every global winner is in its leaf's top-k and survives every llk_k-wide tree merge). Only differs from cons for k32/k64.

Stimulus match: the canonical sweep generates `torch.randn(..., dtype=bfloat16)` (`tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py:569,594,621,640`); sim uses standard normal → RNE-rounded bf16. 5 seeds averaged.

| cell | normal cons/aggr | uniform | softmax | ascending | all-equal |
|---|---|---|---|---|---|
| k32@65536 | 0 % / 0 % | 0 / 0 | 0 / 0 | 0 / 0 | **100 %** |
| k64@65536 | 0 % / 0 % | 0 / 0 | 0 / 0 | 0 / 0 | **100 %** |
| k512@262144 | 0 % / 0 % | 0 / 0 | 0 / 0 | 0 / 0 | **100 %** |
| k512@65536 | 0 % / 0 % | 0 / 0 | 0 / 0 | 0 / 0 | **100 %** |
| k2048@262144 | 0 % / 0 % | 0 / 0 | 0 / 0 | 0 / 0 | **100 %** |

**Why exactly zero, analytically.** For iid data the skip event is distribution-free (pure rank statistics — which is why normal ≡ uniform ≡ softmax). Chunk c+1 (M = llk_k elements) is skippable iff the top-K of the union of the first (c+1)·M elements all come from the first c·M:
P(skip) = C(cM, K) / C(cM+M, K).
With the conservative threshold (K = M): P = 2×10⁻³⁰⁷ at c=1, 7×10⁻⁵⁷ at c=4 — the window is exactly one chunk wide, so a fresh chunk of 512 iid draws virtually always lands something in the running top-512. Even the aggressive k=32 threshold gives 1.4×10⁻¹⁰ at c=1. Column-parallel slices only ever reach c = 1–4. Skips require **long streams**:

| stream position c (chunks seen) | P(skip) cons K=512 | P(skip) aggr k=32 |
|---|---|---|
| 8 | 1.4e-28 | 2.3 % |
| 32 | 1.1e-07 | 37 % |
| 128 | 1.8 % | 78 % |
| 256 | 13.5 % | 88 % |

The a×b rectangle search that raised P (`:389-410`) is precisely what caps c at ceil(chunks/P) = 2–5. **Tier-1 skip and the column split fight over the same resource; the split already won.**

## 3. Predicted µs per cell × distribution

Calibration: `measured = unit × (leaf_units(c_max) + levels) + overhead`, leaf_units(c) = 1 + 2(c−1) (chunk0 sort ≈ 1 unit, later chunks ≈ 2 units — mission constants, matching the factory comment `:382-387`). Routed small-k cells decompose as fixed routing overhead + inner op ≈ the structurally identical k512@65536 (13.1 µs): overhead 23.5 µs (k32) / 29.3 µs (k64). Units: **1.46 µs (≈1966 cyc) per merge unit at llk_k=512; 5.51 µs (≈7440 cyc) at llk_k=2048**.

Skip-test cost (silicon-validated components): 81 cyc cross-TRISC Dst readback (`tt_metal/tt-llk/tests/sources/cgtceq_perf.cpp`, CGTCEQ_RUNBOOK.md) + ~68 cyc SFPU max fold + ~130 cyc/tile chunk unpack (reused by process_chunk when not skipped) → **279 cyc = 207 ns per skipped chunk (409 cyc = 303 ns at K2048); 149 cyc = 110 ns pure overhead per non-skipped chunk**.

Full tree-completion simulation (per-slice leaf times, pairwise level dependencies, root = slice 0):

| cell | measured | predicted: normal/uniform/softmax/ascending | predicted: all-equal | Δ realistic | Δ all-equal |
|---|---|---|---|---|---|
| k32@65536 | 36.6 µs | **36.7 µs** | 33.9 µs | **+0.1 µs (regression)** | −2.7 µs |
| k64@65536 | 42.4 µs | **42.5 µs** | 39.7 µs | +0.1 µs | −2.7 µs |
| k512@262144 | 23.3 µs | **23.7 µs** | 12.5 µs | **+0.4 µs (−1.9 %)** | −10.8 µs (1.87×) |
| k512@65536 | 13.1 µs | **13.2 µs** | 10.4 µs | +0.1 µs | −2.7 µs |
| k2048@262144 | 49.6 µs | **49.7 µs** | 38.9 µs | +0.1 µs | −10.7 µs |

**Headline: expected win on the sweep's stimulus = 0 µs on every cell; expected cost = 0.3–1.9 % regression.** The all-equal column (100 % skip via the `<=` tie rule) is the theoretical ceiling and only materializes on degenerate/constant inputs.

## 4. Reader-bound floor (Tier-1 keeps fetch unchanged — sanity check)

Per-slice fetch is tiny: 2 KB (k512-window cells, 2 chunks × 1 KB) to 8 KB (k2048, 2 × 4 KB); at 64 B/cyc NoC-class that is 0.02–0.10 µs/slice. Aggregate DRAM stream (row of N bf16 across banks, ~400 GB/s): 0.33 µs (65536) / 1.31 µs (262144). Both are 10–100× below the 12–50 µs op times — **the reader never becomes the floor**, even at 100 % skip. (Consistent with the IMPL-4 decision to leave CB flow untouched: there is no bandwidth reason to skip fetches either.)

## 5. Where the idea DOES pay (redirect)

Simulated single-stream contrast (all chunks through one core — the **row-parallel** path when num_rows ≥ grid, or a hypothetical P=1):

- k=32, N=65536, 128-chunk stream: cons 0 %, **aggr (user-k threshold) 54.3 % skip** → would cut leaf work nearly in half on the multi-row shape.
- k=512, N=262144, 512-chunk stream: **15.9 % skip** (cons = aggr).

So the profitable variants are, in order of leverage:
1. **Row-parallel multi-row shapes** (each core streams all N/512 chunks of its row) with the **user-k threshold** — 50 %+ skip at small k on random data, and the routed small-k path always has k ≤ 64.
2. **Threshold seeding (a Tier-2)**: the column-parallel leaves could adopt a global threshold from a cheap pre-pass (e.g. packer exponent histogram, 128 cyc per SORTING.md §threshold-search) instead of their own cold 1-chunk window — that converts the c=1 stream into an effective c=chunks stream. That is a different design with its own sync cost, not Tier-1.
3. If column-parallel Tier-1 is built anyway for robustness on duplicate-heavy data (all-equal 1.87× at k512@262144), gate the test to slices with ≥ 2 later chunks to bound the regression, and use the user-k threshold (free: same Dst read at a different offset; soundness argument in §2).

## Key evidence

- `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:38-46` — snap_to_llk_target_k (k≤512→512, ≤1024→1024, else 2048).
- `...program_factory.cpp:396-410` (+`:330`, `:329`) — a×b rectangle search, cost = 2·ceil(chunks/P)+ceil(log₂P), cap 128, 13×10 grid.
- `...program_factory.cpp:504-524` — contiguous base/extra chunk split per slice.
- `.../kernels/compute_tree.cpp:88-103` — the unconditional process_chunk + merge + rebuild loop Tier-1 would gate.
- `.../kernels/topk_large_indices_compute_common.hpp:74` — elements_per_tile = 1024 (K512 chunk = 1 DST tile).
- `tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py:569` — sweep stimulus is `torch.randn` bf16.
- `tt_metal/tt-llk/tests/sources/cgtceq_perf.cpp` — 81-cyc cross-TRISC Dst readback used as the decision cost.
