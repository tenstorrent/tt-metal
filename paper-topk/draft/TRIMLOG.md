# TRIMLOG — cuts made to reach the 10-page body limit (2026-08-16 integration pass)

Starting point: 12.0 body pages (14 total with `\nocite{*}` bibliography).
End point: **10.0 body pages exactly** (references start at top of page 11; 12 pages total).
Rule followed: BUDGET.md overflow order first, then dedup-only prose trims; no
evidence-bearing number was deleted from the paper unless it survives at
another location (noted per item). The never-cut list (clock caveat, blaze
fairness, scenarios canonical-form caveat, reopening conditions, forecast
no-go story) is verified intact.

## Exhibits cut (4 figures, 2 tables)

| # | Exhibit | Disposition of its content |
|---|---|---|
| T1 | Fig D1 "Exact top-k design space" (map) | Conceptual only; the Q1/Q2/Q3 × (A)–(F) taxonomy is fully carried by §3-A prose. No numbers lost. BUDGET first-cut item. |
| T8 | Fig D2 "engine shootout" (bars) | Every plotted cost lives in Table I (ingredients) cells. |
| — | Fig B1 "Tensix core and NoC mesh" (schematic) | Architecture drawing, no data; §2-A prose carries all facts (130 workers, 13×10, 5 RISC-V, unpack→math→pack, 1.5 MB L1). *Re-add first if a column ever frees up — reviewers like an architecture figure.* |
| — | Fig S2 "skip law" (curves) | Eq. 2 (exact binomial + exponential approx) and the prose anchor points (2×10⁻³⁰⁷ @c=1, 1.8% @c=128 for K=512; 37%/78% for K=32) all remain in §4-D. |
| T2 | Table "Small-k routing before/after" (10 rows) | BUDGET rule 2: collapsed to the three evidence-pack pairs (695→17.2 µs = 40×; 9.50 ms→38.5 µs = 247×; 17.95 ms→42.4 µs = 423×) + the 40×–423× range + the std/bit-stability note, now prose in §6-B. The 18 per-cell routed values remain only in the released CSV (`baselines/smallk_routefix/routed_after.csv`). |
| — | Table "Running-threshold pruning compared" (WarpSelect/GVR/chunk-skip) | The mandated 3-way contrast survives twice in prose: §4-C (threefold contrast sentence: granularity, soundness/verification, law+gate+forecast) and §7 related-work paragraph. No cell content lost. |

## Structural prose consolidations (duplication removals)

- §4 "Measured Scaling" subsection deleted; its P-sweep endpoints, monotonicity,
  and L1-never-bounds content live in §6-C (the "read stream 10–100× below op
  time" clause was *moved* there, not dropped).
- Forecast story now told in full once (§4-F); §6-D and §8 carry one-line
  recaps. "The forecast said no; we did not build it/that variant" kept once (§4-F).
- Chunk-skip measured µs (279.14→153.19, →177.94) only in §6-D; §4-F quotes
  ratios (1.82×/−45.1%, 1.57×/−36.3%) + the min(rows,130) active-core caveat.
- 89×-cliff and 137 ns/elem factory numbers stated once (§2-B); §6-B references back.
- Merge-unit constants (1.46/5.51 µs) stated once (§4-B); §6-C references back.
- bf16 canonicalization rules live in Table II row 1 + abstract + intro bullet;
  §5-A prose now points at the table instead of restating.
- Rendezvous/fold floors: §3-B compressed; full matrix stays in Table II + §5-D.
- Counting-stream increments (+2.42/+4.52 in-stream; 2.44/4.53 isolated) moved
  from §3-B prose into the Table I cell.
- herf/merrill/IEEE-754 totalOrder discussion kept in §5-A; §7 restated as a
  one-clause delta; the third occurrence (§3-C bisection) reduced to a §5 pointer.
- §2-C GPU recap compressed ~35% (BUDGET rule 1); all cites kept.
- Intro paragraph 2 no longer enumerates the three GPU affordances (they are
  §2-C's job); cites kept inline.
- Conclusion enumerate → run-in list; all four reopening conditions retained
  (condition 2 lost the "bitonic merge state stops fitting" clause — implied by
  the LLK ceiling; condition 3 lost "entirely"; condition 4 lost "and
  count-guided skipping compounds" — the compounding point remains in §4-D/§7).

## Formatting

- Tables II (characterization), III (competition), IV (scenarios):
  footnotesize → scriptsize; table note blocks likewise.
- Table II "Probe" column merged into "Measured behavior" as italic
  parentheticals (all probe descriptions preserved verbatim).
- Figures scaled: S1 → 0.82 linewidth, E1/E2 → 0.85 columnwidth.

## Sentence-level trims (~450 words across all sections)

No measured value deleted; casualties worth flagging:
- Intro bullet 3 now quotes only the exponential approximation of the skip law
  (exact binomial form remains as Eq. 2 and in the abstract).
- §6-A: "Exactly one device process…" tightened; clock-caveat paragraph
  tightened without losing the 1.35 GHz / 800 MHz idle content.
- Table III footnote ‡ (blaze) shortened to point at §6-B, which carries the
  full fairness caveats (includes-attention-work, not-a-callable-operator,
  no per-core breakdown committed).
- §8 methodology: stop-rule story shortened to a pointer at §3-D; audit
  paragraph tightened (kept ≈55/60, none fabricated, mutation controls,
  reward-hacking cite, both harness bugs, both ordinary failures).
- §5-B: fixed a wrong cross-reference while compressing — the sampled
  threshold *seed* was described as "a reopening condition in §7" but is not in
  the conclusion's reopening list; now reads "only a sampled threshold seed
  survives".

## Also merged at integration (not trims)

- refs-extra-*.bib (6 files) merged into refs.bib: 9 new entries, 1 duplicate
  key (floydrivest1975) deduped. `\nocite{*}` removed per main.tex note; 8 bib
  entries are now uncited (amd2025adaptivetopk, charikar2002countsketch,
  heavyhitters2023fpga, jalilvand2023sorting, metwally2005spacesaving,
  papaphilippou2021flims, parravicini2021topkspmv, qiao2022topsort) — they stay
  in refs.bib for a future related-work sentence if a column frees up.

# TRIMLOG — revision 1 (2026-08-16, problem-statement + scenarios post-audit pass)

Additions paid for: new §2-A "Problem Statement and Data Contract" (~0.75 col,
owner-directed) + scenarios post-audit footnotes (~0.1 col). End state after
cuts: **10.0 body pages exactly** (references start at top of page 11; 12
total), 0 overfull. BUDGET overflow order honored (rules 1 and 4 newly
applied; 2 and 3 were already spent). Never-cut list verified intact: clock
caveat (§6-A + §3/§4 footnotes), blaze fairness caveats (§6-B), scenarios
canonical-form caveat (§6-E), all four reopening conditions (§9), forecast
no-go story (§4-F).

## Superseded numbers (replaced, not trimmed)

- Scenario table + prose updated to POST-AUDIT scenarios1_table.csv:
  op 193.6→193.8, 120.8→121.0, 187.9→188.1 µs; MoE gate op cells
  3.82/3.83 µs @k=16 ceiling probe → 4.07/4.04 µs measured at production k
  (round-to-16 + slice, values output included); ratios 49.6/1.42/47.5/
  6.3×/20.3× → 49.5/1.4/47.4/5.9×/19.2×. The k=16 ceiling-probe pair
  (3.82/3.83, 6.3×/20.2×) survives only in the CSV notes column.

## Cuts (dedup-only; survival location noted per item)

- §1: "Production numbers motivate this paper better than any argument."
  opener sentence (rhetorical frame; hook paragraph itself intact).
- §1 ¶2: absences list compressed to one sentence; full statement now §2-A(c).
- §1 ¶3: accelerator name list (Cerebras/Groq/SambaNova/Occamy/Esperanto) →
  "the dataflow-accelerator class"; full list survives in §7 class-wide-gap.
- §1 ¶3: mesh-theory sentence compressed (values-with-indices delta kept);
  full treatment in §7. §1 ¶4: three questions shortened to noun phrases.
- §1 bullet 3: "unlike per-lane or predicted thresholds..." clause (the
  mandated WarpSelect+GVR contrast survives twice: §4-C + §7).
- §1 bullet 4: "including the general BH microbenchmark report" cite
  (survives §5 intro + §7). §1 closing sentence tightened.
- §2-B: "load-bearing absences" paragraph and bf16-datatype paragraph
  deleted — both absorbed verbatim-in-substance into new §2-A.
- §2-B: "≈2 cycles/element" dropped from the data-oblivious sentence
  (survives §3-A prose and Table I context).
- §2-D (BUDGET rule 1): threshold-guess + escape-hatch paragraphs merged and
  tightened; all cites kept.
- §3 intro/§3-A/§3-B: pointer parentheticals and "We priced each ingredient
  separately" removed; per-stream increments stay in Table I.
- §3-C: connectives tightened (57 cells, 14/17 decisions, 2,313 cyc,
  invariant, ≈165 cyc composition, ≈1.7 µs, Floyd–Rivest all kept).
- §3-D: "not wasted" and "outrun the side effects" phrasing tightened.
- §4-A: M-snap parenthetical → pointer at §2-B (sole carrier);
  "(⌈log2 128⌉ = 7 levels)" arithmetic; "three ingredients GPU radix select
  leans on" → pointer at §2-A. §4-B: "The lesson generalizes:" frame;
  mesh-theory sentence tightened (cites kept).
- §4-C: "chunk-granular descendant of WarpSelect" sentence deleted — the
  threefold contrast two sentences later carries it.
- §4-D: gate-floor parenthetical tightened.
- §4-F: guard-cell numbers (+0.04–0.51%, 13.15 µs) now stated once in §6-D
  (TRIMLOG-era duplication removed); §4-F keeps the qualitative statement.
- §5 intro: slope-method sentence deduped against §6-A (canonical);
  archival-reference sentence tightened.
- §5-B (BUDGET rule 4): histogram subsection compressed to table-row pointer
  + 2 sentences; 25.175→25.104, mod-64 pattern proof, 4.034/3.855/+4.6%,
  |T|-pow2 rule all survive verbatim in Table II rows 3–5.
- §5-A: totalOrder discussion tightened (all 3 cites kept); 62-cell suite
  mention → pointer (§2-A + §6-A carry it). §5-C: "min→VD" clause (Table II
  row 7 carries it). §5-D: floors paragraph now points at Table II rows 8–9;
  "MMIO ≈10 cyc/word" prose copy dropped (Table II row 8 + Table I carry it).
- §6 intro, §6-A subprocess sentence, §6-B chain sentence, §6-B small-k
  spread sentence, §6-B blaze paragraph: tightened, every number and all
  three blaze caveats kept. §6-D forecast recap → one line (full story §4-F).
- §7: related-work intro line; Fagin "not a per-hop mesh" tail; Jia-genre
  clause (survives §5 intro); totalOrder delta tightened; knuth corollary
  sentence tightened. §8: "we disclose its structure", "of the campaign's
  working notes" tightened.
- §9: "any one, on future silicon or workloads, falsifies the closure."
  closing clause — semantics survive in "reopens if any one holds".

## Formatting

- Table I measured column 3.3→3.9 cm; Table II columns 8.6/4.9→9.6/5.5 cm
  with tabcolsep 4pt (both fit; 0 overfull) — fewer wrapped cell lines.
- Figures: S1 0.82→0.75 linewidth; E1/E2 0.85→0.76 columnwidth.

## Additions (for the record)

- refs.bib: +2 UNVERIFIED entries (blacher2023vqsort, intel_x86simdsort)
  for §2-A's CPU-selection contrast — resolve before camera-ready.
- Table IV note: +2 mandated honesty footnotes (values-output cost
  ≤0.2%/≈6%; launched-grid vs min(rows,130) active cores).

# TRIMLOG — revision 2 (2026-08-17, why-we-win attribution + pseudocode/Tufte-figures pass)

Additions paid for (owner-directed): new §6-F "Attribution: Where the
Speedup Comes From" (~1.0 col, absorbing the old §6-F roofline subsection);
Algorithm 1 (log-tree + dominance-skip pseudocode, scriptsize, ~0.35 col);
Fig. 2 skip-law re-added to §4-D (~0.35 col; was cut rev 0); Eq. 1 extended
to the argmin form with the embedding constraint; core-for-core sentence +
P≥2 clause (coordinator-supplied data); intro + conclusion attribution
pointers; §6-E bit-exactness-audit clause. End state: **10.0 body pages
exactly** (references start at top of page 11; 12 total), 0 overfull
hboxes (three ≤1.7 pt vbox warnings on float/refs pages — invisible),
0 undefined refs/cites, 52 cited entries unchanged, 1 \missing flag (G6,
by design). Never-cut list verified intact: 1.35 GHz clock caveat (§6-A +
§3/§4 footnotes), blaze fairness caveats (§6-B), scenarios canonical-form
caveat + tokens/s-unmeasured (§6-E), all four reopening conditions (§9,
condition 1's "restoring true (D)" gloss trimmed — the restoration
semantics survive in §3-B/§9 framing), forecast no-go story (§4-E).

## Exhibits: cut / added / redesigned

| Exhibit | Disposition |
|---|---|
| Tab. "GPU radix-select ingredients" (old Tab. I) | CUT. Every cell survives in prose: counting row → §3-B para 1 (in-stream increments +2.42/+4.52 and 2.44/4.53, 8.23 4-way, 15.5–19.9 L1-resident absorbed there); decisions row → §3-B para 2 + Tab. I (characterization) row 8 (semaphore 101 / sentinel 98 / MMIO ≈10); materialization row → §3-B para 3; incumbent row → §3-D (34.4/23.3, −15–27%). Rationale: §3-B prose already restated nearly every cell (rev-0 cut Fig D2 for the same duplication). |
| Fig. "chunk-skip A/B bars" (old Fig E2) | CUT. All five plotted cells' numbers verbatim in §6-D prose (279.14→153.19/−45.1%, →177.94/−36.3%, +0.04%, +0.11%, +0.51%, guard 13.15 µs); full grid in tileskip CSVs. |
| Fig. 1 operator schematic (fig-s1) | REDESIGNED (Tufte pass): grayscale, Times-matched, tournament-bracket tree, NoC-traffic annotation (P−1 M-wide window transfers; losers never leave their slice), embedding panel; generated by fig/src/make_whywin_figs.py. |
| Fig. 2 skip-law (fig-s2) | RE-ADDED (was cut rev 0) + redesigned: exact-binomial + exponential curves, both gates, dead/live regime shading, measured −45.1%/−36.3% callouts (computed in-script from tileskip CSVs); caption says the curves are the model. |
| Fig. 3 P-sweep (fig-e1) | REDESIGNED: grayscale, chosen-P* rings with computed deviations (3.8%/0.3%), honest +33%-at-P=2 model note, "factory floor: P≥2" axis note; per-point fit table printed by the script on every regeneration. |
| Algorithm 1 | NEW: leaf stream + skip guard (first-tested max(2,K/4)) + semaphore-tree merge with NoC-traffic comments. |

## Prose cuts (dedup-only; survival location per item)

- §2-D "What GPU Selection Assumes" → heading dropped, compressed to one
  paragraph at the end of §2-C (BUDGET rule 1 terminal form); all 9 cites
  kept in place; affordance details survive: radix → §3, WarpSelect/GVR →
  §4-C+§7, escape hatch → §1+§7.
- §7-d totalOrder-delta sentence CUT — survives verbatim-in-substance in
  §5-A with all three cites (herf/merrill/ieee754).
- §7-c knuth corollary: k ln(n/k)+O(k) formula dropped; cite + no-published-
  corollary claim kept.
- §6-B: "Fourth headline" (23.3 µs @104c) cut — survives in abstract, §1
  bullet 2, §6-C, §9; "Four"→"Three headlines". "Each speedup chain is
  named where used..." meta-sentence cut (the discipline is enacted;
  anchor-twin disclosure §6-A kept). proxy→op 43× tail moved to §6-F
  (with shape).
- §6-C model paragraph compressed — Fig. 3 now carries no-fitting, +33%
  at P=2, chosen-P* deviations, bump (prose keeps 4.3%/0.3%/44.1/38.6/
  42.8/39.0 verbatim).
- §1 bullet 1: the three microbenchmark numbers (2.0 cyc/vec, 81 cyc,
  13.0 vs 0.5) dropped from the bullet — abstract + §3 carry them
  verbatim; bullet keeps the narrow/affordable/wall verdict. Bullet 3
  "(−45.1%)" dropped (1.82× kept; −45.1% in abstract/§4-E/§6-D/Fig. 2).
- §5-B/§5-C/§5-D, §2-A/§2-B, §4-A/§4-C/§4-F, §6-A/§6-E, §7/§8/§9:
  sentence-level dedup trims (connectives, restated pointers, "on 104
  cores" in §9 dup of §6-C); every measured value kept at its canonical
  location. §8 audit sentence: "the AI CUDA Engineer's" → "the" (cite
  kept; the named system remains in the §8 opening cite cluster).
- §4-A: "communication structure is selection's..." sentence (added this
  rev) compressed to a Fig.-1 pointer after §6-F took ownership.

## Superseded/new numbers (added, not trimmed)

- §6-F factor chain 1.17× × 1,511× × 10.4× = 18,401× (arithmetic on
  Tab. II's printed arms; header comment carries the trace).
- Core-for-core ≈1.5×/32× decomposition of the 49.5× sampling win
  (arithmetic on Tab. III row 1: 9,596.3/32 vs 193.8).
- i5 bit-exactness audit numbers (27,756 diffs all ties over 3,200 rows)
  — evidence/i5-sampling-relaxation/i5-landing-report.md §3.
- P≥2 factory-validation floor (§6-C + Fig. 3 axis) — program factory
  :443 TT_FATAL + psweep_p1 UNSUPPORTED cells.
