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

# TRIMLOG — revision 3 (2026-08-17, comp4 re-measure + TILE-native / carve-out / telemetry landings)

Context: three landed wins re-measured (comp4) added ~1.0 col of required new
content (§6-B TILE-native paragraph, §6-D telemetry + gate ablation, §6-E
carve-out rewrite, §6-F envelope-limb rewrite, Table II/III refresh, Fig. 2
measured overlay). Paid in full below — body ends on page 10 exactly (12 pp =
10.0 body + 2 refs, 0 overfull hboxes, same 3 sub-visible vbox warnings ≤1.7 pt).

## Superseded numbers (replaced, not trimmed — all from baselines/comp4/)

- Tab. II routed column, all 24 cells (comp3 → comp4; e.g. anchor 63.4 → 51.5,
  k512@65536 91.7 → 20.9, k1024@2048 134.2 → 21.3) + pre→routed speedups
  (span 56×–22,481× → 329×–31,905×; anchor 9,956× → 12,265×). Swept through
  abstract, §1 (hook 9,596 → 9,587 µs; thesis 63.4/9,956 → 51.5/12,265), §6-B,
  §9.
- Tab. III, all rows (scenarios1/comp3 → scenarios7/comp4): sampling
  9,596.3/217.0/193.8 → 9,586.9/213.6/193.6 (44.2×/49.5× → 44.9×/49.5×);
  split-vocab 8,923.6/215.2/188.1 → 8,924.0/210.9/188.0 (41.5×/47.4× →
  42.3×/47.5×); TP=8 171.4/171.3/121.0 → 171.2/171.2/120.8; DSA routed
  892.7/691.1 → 895.3/682.2 (routing cost 20–25% → 22–26%); MSA routed
  116.1 → 114.2; gate rows restructured: today = routed 8.18/8.34 (130)
  (carve-out landed; was 24.2/77.5 single-core with op-only 5.9×/19.2×);
  §6-F core-for-core arithmetic 9,596.3/32 → 9,586.9/32.
- §6-D epistemic-limit sentence ("no on-device skip-rate telemetry ... law →
  simulated skip → predicted time → measured time") DELETED — replaced by the
  measured telemetry + gate-ablation result (G4 closed). Fig. 2 caption's
  "the only silicon numbers are the annotated end-to-end deltas" DELETED
  (now false — the dots are silicon).
- §6-F envelope limb rewritten: "envelope = 46% ... the next lever" →
  "named lever now banked" (deleted at k_rounded ≤ 1024, anchor −19%,
  residual 9.8 op-internal + 7.4 kept-tilize); fusion (blaze 24.4 µs, 1.4×)
  remains the open lever.

## Cuts (dedup-only; survival location per item)

- §6-A: routed re-measure sentence — survives verbatim-in-substance in the
  Tab. II caption ("re-measured on the final tree ... 5-iteration medians,
  0 wrong"); single carrier now.
- §6-A: "(e.g., 9,492,327 ± 1,391 ns at N=65,534)" → dropped the cell tag
  (value kept; full row in scope51 CSV / evidence H6). "taken at operator
  granularity" → "at operator granularity"; "runs at a time" → "at a time";
  "cannot leak between cells" → "cannot leak".
- §6-B small-k: middle example (9.50 ms → 38.5 µs at W=65,534, 247×) cut —
  BUDGET rule 2 direction; endpoints kept (40× and 423×); full 18-cell grid
  in baselines/smallk_routefix/. Proxy footnote tightened (same content).
- §6-B TILE paragraph: BH DRAM alignment parenthetical (16 B writes /
  64 B-congruent staged reads) cut per the "only if budget allows" rule —
  survives in evidence.md §7 B2 + the 06-evaluation.tex header comment +
  i2-tile-native-report.md.
- §6-C: "the sweep starts at P=2 because the factory refuses P=1" →
  parenthesized; bump's "(measured 42.8 vs. 39.0 µs; model 44.1 vs. 38.6 µs)"
  cut — numbers survive in the fig-script fit table printed on every
  regeneration + evidence A6; "— the property the log-tree bought (§4)" →
  "(§4)" (§4-A owns the claim); "— the limit is the ⌈C/P⌉ work term itself"
  cut (the model paragraph carries it).
- §6-D: gated no-win cells "+0.04% and +0.11% sit within the sub-0.1% spread,
  and the largest residual +0.51% ..." → "+0.04% to +0.51%, the largest being
  gate-off loop-code growth" (all three values in tileskip CSVs + C3-4);
  "three arms each: baseline, skip ungated, and skip with the compile-time
  K/4 gate" → "(baseline, ungated, gated)"; second "279.14 →" dropped.
- §6-E: "at real shapes" (dup of "call-site shapes"); "on a single chip";
  "row-parallel with the chunk skip live" (survives in Tab. III caption);
  "landed under a pre-registered rule (routed×1.05 ≤ stock at both cells)"
  → evidence.md §7 B5 carries the rule; "k rounded up to 16 and sliced, the
  route's own internal mechanism" → Tab. III footnote carries it.
- §6-F: "(routed −67 to −84%, Table II)" in the envelope limb (dup of the
  §6-B TILE paragraph, which owns it); "because the native k=2048 scatter
  measured slower" (same dup); "stock bitonic kernels streaming the row"
  (dup of arm definition + † footnote); "9.7×–21.1× at N ≥ 32,768" secondary
  roofline stat (24-cell span kept); "in one sentence" / "not merely the
  incumbent one" (verdict sentence tightened).
- §4-C: threefold WarpSelect/GVR contrast tightened to point at §7 (which
  keeps the full contrast); "that neither GPU mechanism has" absorbed.
- §4-D: "consistent with the 81-cycle rendezvous plus fold components of §3
  (footnote 1)" cut — §3-D states the same link ("the rendezvous and fold
  primitives price §4's chunk-skip decision").
- §4-E: guard-cell sentence → pointer ("Section 6-D quotes the A/B, the
  guard cell, and the on-device telemetry"); §6-D owns the numbers.
- §5 intro: "Every cycle figure is a two-point slope measurement with a
  2.000× control tripwire" → "Cycle figures are slope-measured with a 2.000×
  control tripwire" (§6-A owns the full method sentence).
- §7-a: "(Section 4 instantiates the mesh bound with measured constants)"
  cut — §4-B states it ("mesh-selection theory's multi-packet regime with
  measured constants").
- §7-d: "— itself an argument for archival characterization" cut.
- §8: contract gate "(semantics and harness pinned first)" cut (§2-A/§6-A
  enact it); forecast sentence's "the redirect measured 1.82×" cut (§6-D/§9
  carry 1.82×); vendor-bug parenthetical "(a stimuli configuration never
  written; a semaphore leak that deadlocks later tests)" cut — evidence C4-9
  carries it; "re-adjudicated disputed figures on fresh silicon" → "…disputed
  figures"; "both are logged;" cut.
- §9: "from chunk-skipping in the forecast-selected row-parallel regime" →
  "from forecast-selected chunk-skipping" (§4-E owns the regime); "--- the
  weak form of (D) ---" → "(weak-form (D))".

## Additions (for the record)

- §6-B: TILE-native I/O paragraph (landing summary + per-shape measured
  policy) — the §4 "one tight paragraph" option was declined for budget; §6-B
  sentence form used instead, per the mission's fallback.
- §6-D: on-device telemetry (400 rows, 120 positions, E[skips] 65.11 vs 66.62,
  −2.3%; ELF-diff byte-identity) + gate-constant ablation (2/4/8; keep /4).
- §6-E/§6-F: MoE-gate carve-out (24.2→8.18, 77.4→8.34; 2.96×/9.29×, zero
  model-code changes).
- Fig. 2: measured per-position skip-rate overlay (g4_curve.csv) + caption
  rewrite; Tab. II caption: routed re-measure provenance note.

## Post-pass addition (coordinator-directed): §6-F tree-limb applicability boundary

Added (2 sentences, evidence row B7 — arithmetic on printed cells): the tree
converts cores into latency at ≈1.0 core-efficiency (P=2 = 183.0 µs = 366
core-µs/row vs the per-row kernel's 357.0 µs, +2.5%), so 10.4× serves
latency-bound low-row callers; rows ≥ grid callers are already core-optimal
(160-row indexer 712.3 µs ≈ 2 × 357.0 row-waves, Tab. III 1.00×/0.80×);
per-core work there = the fusion lever. Paid by (all dedup/compression;
survival locations noted):

- §6-F structural term compressed to a Fig. 1/§4-A pointer ("selection,
  never sorting — only M-wide windows cross the NoC, P−1 in total") — the
  full statement lives in §4-A + Fig. 1's edge annotation.
- §6-F roofline contributors parenthetical "(leaf critical path, tree
  levels, unpack floors)" → section pointer only (§3–§5 own them).
- §6-E: "The residual over the bare op (4.07/4.04 µs) is the composite
  launch envelope at these tiny shapes" CUT — survives in evidence §7 B5
  (~4.2 µs composite envelope); op cells + footnote still printed. "The
  table carries its own controls as features" frame sentence CUT (the
  controls speak). Carve-out "of any kind" dropped.
- §6-C: "(+33% at P=2; the flat two-units-per-chunk charge is conservative
  on long streams)" → "(+33% at P=2, Fig. 3)" — Fig. 3 carries the full
  note in its annotation. "by validation" dropped from the P=1 refusal.
- §6-D: "on the in-order RISCs" dropped from the gate-off growth clause;
  ": the tree path is untouched" dropped (byte-identical kernels states it).
- §6-A: seed-0 chunk-skip detail dropped (C3-4 owns it); "(counting)/(RISC
  scan)" labels folded to a trailing parenthetical; "report medians of"
  dedup in the chunk-skip clause.
- Tab. II † footnote: "byte-identical stock bitonic kernels in the
  operator's row-parallel harness" → pointer to the §6-B arm definition
  (verbatim dup).

Body still ends page 10 exactly; 0 overfull hboxes.
