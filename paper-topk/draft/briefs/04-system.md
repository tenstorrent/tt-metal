# Brief 04 — System: Log-Tree Top-K on a NoC Mesh (C2) + Chunk-Skip (C3)

**Section file:** `sections/04-system.tex`
**Budget:** 4.0 columns (2.0 pages) — C2 ≈2.3 cols, C3 ≈1.7 cols.

## Single job

Present the two constructive artifacts as consequences of §3's map: C2 is the
best (A)+(C)+(E) engine the mesh admits (parallelize the oblivious network,
pay log-depth rendezvous); C3 is the *sound* import of (D) — the one shrinking
mechanism that survives without scatter. The section must read as "the design
space told us what to build," not "here are two tricks."

## Part 1 — C2: column-parallel log-tree operator (≈2.3 cols)

### 4.1 Structure

- Row of N elements split into C = N/M chunks (M = llk_k window from §2
  notation); P slices, each streaming ⌈C/P⌉ chunks through the bitonic leaf,
  then a ⌈log₂P⌉-level in-place merge tree across cores.
- Tree levels = semaphore levels; one 2-sequence receive CB per core; writer
  kernels extend to 7 partner slots (P=128 ⇒ 7 levels). **No atomics, no
  global barrier, no dedicated DRAM scratch** — position against GPU practice
  (atomic counters in radix select), NOT against mesh theory, which never had
  atomics (related-work.md C2 instructions, verbatim rule).
  Evidence: C2-4 (kernels `compute_tree*.cpp`, `writer_tree*`).
- The serial-chain ancestor it replaced: 69.6→41.9 µs @ k2048/65,536 and
  64.2→32.0 µs @ k512/262,144 at fixed P; P became monotone (H11 — quote as
  the design delta, then hand absolute numbers to the post-P-cap values).

### 4.2 Cost model + rectangle embedding

- **cost(P) = 2·⌈C/P⌉ + ⌈log₂P⌉**, ties prefer fewer cores, P ∈ [2,
  min(C,128)]. The 2× on the leaf term: later chunks cost ≈2 merge units
  (sort + merge + rebuild), chunk 0 costs 1 (forecast.md §3 calibration:
  1.46 µs/unit at M=512, 5.51 µs at M=2048 — cite as measured unit constants
  with the 1.35 GHz caveat on the cycle forms 1966/7440 cyc).
- Cost-optimal a×b rectangle search on the 13×10 grid: the full-rows-only
  fit silently capped k2048@65,536 at 13×2 = 26 cores when 8×4 = 32 was
  free. Fix wins 8–27% op-level: 41.9→34.4 µs (26→32c), 58.6→49.6 (52→64c),
  32.0→23.3 (52→104c), 15.0→13.1 (52→64c). Evidence: C2-2,
  `baselines/smallk_routefix/pcap_repin.csv`.
- Frame per related-work.md: the model is *the mesh-theory bound instantiated
  with measured constants* — one sentence tying 2⌈C/P⌉ to Krizanc &
  Narayanan's multi-packet N/p regime.

### 4.3 Measured scaling (forward pointer to §6's Fig 1)

- State the two endpoints here, defend in §6: k2048@65,536 183.0 µs (P=2) →
  34.4 µs (P=32); k512@262,144 560.2 µs (P=2) → 23.3 µs (P=104), monotone,
  flattening near the 130-core grid capacity; L1 never bound P (C2-3, C2-5;
  raw: `baselines/comp3/psweep4_full.csv` — 15 measured points, in-repo).

## Part 2 — C3: chunk-skip in a streaming bitonic cascade (≈1.7 cols)

### 4.4 Mechanism + soundness sketch

- Skip rule: strict `chunk_max < T` where T = running USER_K-th survivor
  minimum; chunk 0 always seeds; T monotone nondecreasing; boundary ties are
  never skipped; deterministic per input. Soundness: an element below the
  leaf's running user-k-th largest can never enter the global top-k — fewer
  than k elements beat any global winner, so every global winner is in its
  leaf's top-k and survives every M-wide tree merge (forecast.md §2 "aggr"
  argument + the committed 288-line proof header, C3-5). Print a 4–6 line
  proof sketch; cite the artifact for the full proof.
- Adversarial validation one-liner: 36-check battery (all-equal, ascending,
  last-chunk winners, boundary ties × 64 chunks) bit-exact both configs;
  hang battery 20/20 twice (C3-5).

### 4.5 The skip law + compile-time gate

- **P(skip at chunk c) = C(cM,K)/C((c+1)M,K) ≈ e^(−K/(c+1))**, distribution-
  free for iid inputs (pure rank statistics — normal ≡ uniform ≡ softmax).
  Print the exact binomial ratio as primary and the exponential as its
  approximation — evidence.md §5 flags two inconsistent normalizations in
  the working notes; the binomial form is the safe citable one.
- Consequences: conservative K=512 threshold: P(skip) = 2×10⁻³⁰⁷ at c=1,
  1.8% at c=128; aggressive k=32: 37% at c=32, 78% at c=128 (C3-1).
- **Compile-time gate: first_tested = max(2, K/4)** — below c = K/4,
  P(skip) < e⁻⁴ ≈ 1.8%; the gate zeroes the measured ungated overhead
  (+6.8% k512@128-chunks, +1.8% k1536@25-chunks) while forfeiting <1
  expected skip per row (C3-2).
- Decision cost: ≈150 ns/tested chunk at M=512, ≈350 ns at M=2048, derived
  from the ungated A/B (126 tested chunks = +18.9 µs) and consistent with
  the 81-cyc rendezvous + fold components (C3-6). 1.35 GHz caveat applies.

### 4.6 The forecast no-go story (the section's methodological teeth)

- Column-parallel Tier-1 was forecast BEFORE building: exact host bf16
  sign-magnitude streaming simulation, calibrated to reproduce all 5 pinned
  measured cells ±0.0 µs by construction; predicted **0.00% skip on every
  realistic distribution on every cell** and a +0.1–0.4 µs pure regression
  (C3-3, forecast.md verdict + §2 + §3).
- The structural cause, stated as a law of the design space: *the rectangle
  split and the skip law fight over the same resource — stream length c.*
  The a×b search that raised P caps c at ⌈C/P⌉ = 1–5, and the law needs
  c ≳ K/4. The split already won (forecast.md §2 close).
- The build therefore went to the **row-parallel** variant (each core streams
  all C chunks of its own row — long streams, small k): measured
  rows=2, k=32, N=65,536: 279.14 → 153.19 µs (**−45.1%, 1.82×**); rows=8:
  → 177.94 µs (−36.3%); gated no-win cells +0.04% to +0.51%; column-parallel
  guard cell 13.153 µs unchanged, column kernels byte-identical (C3-4,
  rowskip-implementation.md §4).

## Claims owned

C2-1..C2-5 and C3-1..C3-6 (evidence.md §1.3, §1.4). Also M3 (forecast-
before-build) as a *narrated event* here; §7's methodology subsection owns
its generalization.

## Figures/tables owned

- **Fig S1** (1 col): the operator schematic — chunk stream → bitonic leaf →
  log₂P semaphore tree on the a×b rectangle; annotate cost terms 2⌈C/P⌉ and
  ⌈log₂P⌉ on the two phases. Caption ≤8 words: "Column-parallel log-tree
  top-k operator."
- **Fig S2** (1 col): skip law — P(skip) vs c curves (binomial exact, K=512
  cons and k=32 aggr), K/4 gate marked, with the two operating regions
  (column-parallel c=1–5 shaded dead, row-parallel c up to 128 shaded live).
  Source: forecast.md §2 tables + `sim_tier1_skip.py`. Caption:
  "Skip probability versus stream position."
- **Tab S1** (1 col, may live here or §7): WarpSelect vs GVR vs chunk-skip
  contrast — signal source / soundness / granularity / verification passes
  (related-work.md action item 3). Default placement: here, closing 4.4.

## Style directives

1. This is the systems-builder core — highest active-voice density in the
   paper ("the writer extends", "the gate zeroes", "the split already won").
2. Give the cost model its own display equation; reuse the \costmodel macro;
   never re-derive it inline elsewhere.
3. Report the no-go with the same typographic weight as the win: the
   forecast table (predicted 0.00%, +0.1–0.4 µs) earns two sentences BEFORE
   the −45.1% row-parallel number appears.
4. Contrast citations inline at mechanism introduction: WarpSelect
   (johnson2019billion) at the skip rule, GVR (gvr2026) at "sound by
   construction — no verify pass," Bonsai (samardzic2020bonsai) at the merge
   tree, Krizanc & Narayanan at the cost model.
5. Percentages and ratios both, Kapre-style: "−45.1% (1.82×)".

## Hazards

- Anonymity: describe kernels/factory structurally; never print repo paths
  or commit hashes; "the released artifact" is the pointer.
- The −45.1% cell runs 2 rows on a 130-core program (active cores =
  min(rows,130)) — say so, or a reviewer reads it as a 130-core win.
- Do NOT present the K/4 gate as tuned: it is derived from the law
  (e⁻⁴ ≈ 1.8%) — derivation, not sweep.
- The aggressive-threshold soundness argument is load-bearing for both the
  skip rule AND the law's K parameter; write it once in 4.4 and reference it
  in 4.5.
- Skip-rate telemetry was never measured on-device (G4): the measured
  quantity is end-to-end time. Fig S2 plots law + simulation; the measured
  deltas live in §6's A/B bars. Keep the epistemic chain explicit:
  law → simulated skip → predicted time → measured time.
