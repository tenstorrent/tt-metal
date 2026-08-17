# Brief 02 — Background

**Section file:** `sections/02-background.tex`
**Budget:** 2.0 columns (1.0 page), of which the GPU selection recap gets AT
MOST 1 column.

## Single job

Give the reader exactly the architecture facts needed to follow §3–§6 —
no more. The test for every sentence: does §3's ingredient table, §4's tree,
or §5's characterization use this fact? If not, cut it.

## Content plan (three subsections)

### 2.1 Blackhole / Tensix (≈0.7 col + Fig B1)

Facts to establish (each is load-bearing later):
- 130 programmable Tensix workers on a 13×10 grid (p150a; harvesting varies
  per unit), 2D NoC, per-core ~1.5 MB software-managed L1. Evidence: pack
  header ("13×10 grid (130 workers)"), Hot Chips citation
  `vasiljevic2024blackhole` for the architecture description.
- Five RISC-V processors per core: two data-movement (NoC reader/writer),
  three compute (unpack → math → pack pipeline); SFPU = 32-lane SIMD unit;
  DST register file between math and pack. (Used by: C1 counting arms, C4
  Dst layout, chunk-skip Dst-MMIO rendezvous.)
- **What the machine lacks — the sentence the whole paper leans on:** no
  atomic read-modify-write across cores, no scatter/compaction primitive in
  the compute path, no global barrier; inter-core sync is NoC semaphores.
  (Backs C1-5, C2-4.)
- Producer/consumer circular buffers in L1; kernels are three cooperating
  programs per core.
- bf16 is the working dtype of the LLM call sites (scenarios CSV: every row
  bf16); fp32 passes through the datapath bit-exactly (C4-1).

### 2.2 The incumbent bitonic engines (≈0.5 col)

- The vendor stack routes top-k through a bitonic local sort
  (`ckernel topk_local_sort`; cite the vendor docs/GitHub issues as
  engineering artifacts, per related-work.md §1 "Engineering artifacts").
- Two shipping engines pre-study: (a) multi-core bitonic — gated to
  pow2 widths, W < 65,535, k ≤ 64, N ≥ 4096; (b) a linear single-core factory
  at ≈137 ns/element that catches everything else (H6:
  695 µs @ W=5000 → 17.95 ms @ W=65,536, `baselines/smallk_routefix/
  stock_nonpow2.csv`, `baselines/scope51/canonical_sweep.csv`).
- The measured cliff between them: ≈89× at the W<65,535 gate (H8,
  scope51). State the cliff here as *architecture context*; §6 owns the
  before/after table.
- The design-space vocabulary teaser: the bitonic engine is data-oblivious —
  identical instruction stream for every input, ≈2 cyc/elem on all N every
  pass (RBG §7.1's (A)+(C)+(E) — but introduce the Q1/Q2/Q3 labels in §3,
  not here).

### 2.3 GPU selection family recap (≤1 column — HARD CAP)

One paragraph per branch, each ending in the property Blackhole must replicate
to import it:
- **Radix/bucket select** (Alabi'12, RadiK ICS'24, SC'23 study): 8–12-bit
  exact histograms via shared-memory atomics, scatter compaction, few passes.
  Needs: wide counting + cheap materialization.
- **In-register selection** (WarpSelect, Johnson'19): per-lane running
  thresholds feeding bitonic queues. Needs: cheap lane-local compare + a
  place to park survivors. (Set up here; §4.C3 contrasts.)
- **Threshold-guess families** (GVR arXiv:2604.22312): predicted threshold,
  count-verify-refine. Needs: cheap counting passes.
- **The accelerator escape hatch**: TPU-KNN and two-stage approximation
  (chern2022tpuknn, twostage2025) *avoid* exact selection — frame as the
  industry's confession that exact top-k is hostile to matmul-centric
  machines; this paper measures the exact path anyway.

## Claims owned

- Architecture facts above (cores/NoC/SFPU/packer/no-atomics/no-scatter).
- H6 (137 ns/elem linear factory) and H8 (≈89× cliff) as *context*
  statements — the numbers, not the fix (the fix is §6's).
- The routing-gate description (pow2 / W<65,535 / k≤64 / ≥4096) — matches
  scenarios CSV notes column and RBG §7.3 item 2.

## Figures/tables owned

- **Fig B1** (1 col): Tensix core + mesh schematic — 5 RISC-Vs, SFPU/packer,
  L1 CBs, NoC links; annotate "no atomics, no scatter" on the NoC edge.
  Hand-drawn TikZ/SVG; caption ≤8 words, e.g. "Tensix core and NoC mesh."

## Style directives

1. This is the ONLY didactic section — Wilton-style patience allowed, but at
   Kapre pace: every architecture fact in ≤2 sentences, then move.
2. No GPU-recap sprawl: 1-column hard cap; each family gets
   mechanism → required property → one citation cluster.
3. Numbers with units and configuration even here (≈137 ns/element, 130
   cores, 13×10) — background is not exempt from quantitative style.
4. Define once, reuse everywhere: C (chunks), P (cores/slices), K (user k),
   M = llk_k (leaf window: k≤512→512, ≤1024→1024, else 2048 — forecast.md §1).
   All later sections inherit this notation.
5. Grid-size honesty: say "130 workers on this unit; harvesting varies" —
   never hardcode as an architectural constant.

## Hazards

- Do not describe Wormhole; the paper is single-arch (G1 acknowledged in §7's
  future work, not here).
- Do not enumerate the packer histogram / canonicalization here — those are
  C4 *findings* (§5), not background. Background states only what vendor docs
  already say; anything discovered by this campaign belongs in §5.
- L1 size: say "≈1.5 MB per core" only if citing the Hot Chips talk; do not
  invent budget numbers beyond it.
