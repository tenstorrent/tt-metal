# Brief 03 — Design Space: Why GPU Selection Economics Do Not Transfer (C1)

**Section file:** `sections/03-design-space.tex`
**Budget:** 2.5 columns (1.25 pages).

## Single job

Deliver the paper's negative result as a *decomposition*, not an anecdote:
three orthogonal questions define every exact top-k engine; Blackhole was
measured ingredient-by-ingredient against each; the GPU corner of the space
is unreachable and the reason is materialization, not counting. This section
is what makes the paper a design-space study instead of a kernel paper.

## Content plan

### 3.1 The map (Q1/Q2/Q3) — from RBG §7.1 (verbatim structure, tightened)

- **Q1 — winner identification:** (A) comparison networks (data-oblivious) vs
  (B) counting/partitioning (data-dependent decisions).
- **Q2 — data touched per pass:** (C) full-data every pass vs (D) shrinking
  candidate sets (compaction or count-guided skip).
- **Q3 — decision cost between passes:** (E) none vs (F) a synchronization
  rendezvous.
- GPU radix select = (B)+(D)+cheap-(F): shared-memory-atomic histograms,
  scatter compaction, grid-sync/fused decisions. The shipping bitonic engine
  = (A)+(C)+(E). The question the section answers: can Blackhole move to (B)?
- Small taxonomy figure or a 3-axis table (Fig D1) — this is the paper's
  conceptual signature; invest in it.

### 3.2 The measured ingredient table (Tab D1 — from RBG §7.2, evidence C1-1..C1-5)

| Ingredient | GPU form | Blackhole measured | Verdict |
|---|---|---|---|
| Wide exact counting | 8–12 bits/pass, shared-mem atomics | SFPU: 1 bit @ 2.0000 cyc/vec exact; 3 bits @ 3.0; additive on the 3.94 cyc/vec unpack floor (single +2.42, dual +4.52). RISC: 256-bin histogram 7.56–7.90 cyc/elem vs 3.02 pure-load floor; L1-resident 15.5–19.9 | counting narrow (SFPU) or slow (RISC) — radix degenerates to bisection |
| Cheap decisions | grid sync / fused kernels | 81 cyc/decision best (tensix_sync + cross-lane fold + Dst-MMIO read); semaphore 101; sentinel 98; no-fold 756–773 (folding mandatory); MMIO ≈10 cyc/word | affordable — NOT the blocker |
| Materialization | scatter/compaction | dense RISC emit 12.97–17.81 cyc/elem vs a 0.5 bar; SFPU has no scatter; sole survivor: packer compressed stream at 0.63 cyc/orig-elem (gray band) | **the load-bearing gap** |
| Slow incumbent | thrust::sort | incumbent moved to 34.4 µs / 23.3 µs during the study | the bar moved −15–27% during evaluation |

Evidence: C1-1 (`evidence/validate/cgtceq-debug.md` §(i), RBG §7.2 table),
C1-2 (risc_scan_bench, 21/21 bit-exact), C1-3 (rendezvous matrix 3 folds ×
3 syncs), C1-5 (RBG §7.2, §4.4). Every cell in the printed table must carry
its bench name (the RBG table is tagged per-cell — keep that).

### 3.3 Degeneration to threshold bisection (C1-4)

- Without (D), counting passes are strictly additive to the bitonic finish.
- The exact arm collapses to bisection for the K-th threshold: p50 14
  decisions / 2,313 cycles (random), worst-case 17 pinned (clustered /
  all-equal / ties); ≈165 cyc/decision including composition check.
  Evidence: cgtceq-debug.md §(iii) — 57 cells incl. K∈{31,32,33} straddle,
  ties, ±0/Inf/NaN/denorm specials, invariant Cgt<K≤Cgt+Ceq field-exact.
- Punchline (RBG §7.2 close): Blackhole's answer to the design space is
  Floyd–Rivest economics, not radix economics — the family fits only in the
  degenerate corner (B)+(C)+(F).

### 3.4 Why the track closed (C1-6) — the honest kill

- Selector's honest composed model: ≈21–30 µs per large cell, priced from
  the measured components above.
- Pre-registered stop rule (RBG §5.3/CRIT-4, written BEFORE measurement):
  incumbent must be ≥2× the model to justify building.
- The incumbent finished at 34.4 µs (k=2048, N=65,536) — inside the model's
  uncertainty band — *because the study's own Gate-1 side effects (routing
  fix, P-cap raise) made the incumbent faster*. One crisp sentence: "a
  selector that must outrun its own evaluation's side effects is not a
  production bet" (RBG §7.4 — reuse the thought, rephrase in-voice).
- Forward pointer: the counting machinery wasn't wasted — it priced the
  chunk-skip decision (§4) and banked the characterization (§5).

## Claims owned

C1-1, C1-2, C1-3, C1-4, C1-5, C1-6, C1-7 (all of §1.2 in evidence.md).
Also owns the "counting works only in corner (B)+(C)+(F)" synthesis claim.

## Figures/tables owned

- **Fig D1** (1 col): the Q1/Q2/Q3 design-space map with GPU-radix, shipping
  bitonic, and the degenerate BH corner marked. Caption ≤8 words:
  "Exact top-k design space." 
- **Tab D1** (1 col): measured ingredient table (above), each cell tagged
  with its bench.
- **Fig D2** (1 col, shared candidate with §6 — see BUDGET.md): engine
  shootout bars, cyc/elem per mechanism: RISC load floor 3.02, RISC hist
  7.6–8.2, L1 hist 15.5–19.9, dense emit 13.0–17.8, compressed consumer
  0.63, SFPU count 2.0/vec-normalized, bitonic leaf ≈2 bar, 81-cyc decision
  annotation. Source: commit-message numbers mapped in evidence Fig-4 row;
  regenerate raw CSVs from committed benches if reviewers ask (evidence §2
  Fig-4 status note).

## Style directives

1. Negative results in Kapre voice are *quantified verdicts*, not apologies:
   "dense emit measures 13.0 cyc/elem against a 0.5 cyc/elem bar — dead" —
   verdict words allowed after the number, never before it.
2. The (A)–(F) labels must be used consistently from here through §7's
   reopening conditions; define once in 3.1.
3. Cycle numbers stay in cycles here (they are slope-measured); convert to µs
   only with the 1.35 GHz caveat attached (methodology §6 owns the caveat
   text; this section footnotes it on first conversion, e.g. the ≈1.7 µs/row
   bisection figure).
4. Cite the GPU lineage per mechanism inline (Alabi'12/RadiK for counting,
   Merrill/Herf for the bit-flip order trick if mentioned, TPU-KNN for the
   escape hatch) — the reader should never wonder "compared to what?"
5. Sentence budget: this section wants tables to carry the load; prose
   analyzes the table, never restates it row by row.

## Hazards

- The 0.63 cyc/orig-elem compressed-stream figure is CONSUMER-only, measured
  on uniform input; the producer-side composition was never measured (G7,
  reopening condition #1). The text must say so — it's also §7's first
  reopening condition, so the phrasing here and there must match.
- Do not claim the 2.42/4.52 additive costs and the 2.44/4.53 MATH_ISOLATE
  numbers are different measurements of different things — they are the same
  quantity two ways (C1-1); quote one pair and parenthesize the other.
- "Decisions are affordable" must not be softened into "cheap" — 81 cycles
  ≈ 60 ns is affordable *per row*, and the bisection needs ≈14 of them; keep
  both numbers adjacent so the reader can multiply.
- SORTING.md carries [DISPUTED] tags on some LLK floors (3.855 vs 3.938 vs
  4.175) — quote only the adjudicated values used here (3.94 unpack floor per
  evidence C1-1) and do not cite SORTING.md numbers directly.
