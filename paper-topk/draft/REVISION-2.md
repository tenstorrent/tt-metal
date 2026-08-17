# REVISION-2 — why-we-win attribution + pseudocode + Tufte figure pass
2026-08-17. All changes uncommitted. Build: `latexmk -pdf` clean from
scratch — 12 pages = **10.0 body + 2 references** (body ends page 10,
references start at top of page 11), 0 overfull hboxes (3 vbox warnings
≤1.7 pt on float/refs pages — sub-visible), 0 unresolved `\cite`/`\ref`,
52 cited entries (unchanged), exactly 1 `\missing{}` flag (G6, by design).

## Job 1 — §6-F "Attribution: Where the Speedup Comes From" (owner-directed)

New closing subsection of Evaluation (label `sec:eval-attribution`,
absorbing the old §6-F roofline subsection), carrying the explicit,
measured decomposition of WHY the design wins:

1. **Honest attribution first.** The anchor's 18,401× pre→op chain
   factors *exactly* along Table II's own arms: 1.17× stock-kernel
   micro-op wins (§5-D) × 1,511× available on a *single* core (proxy arm
   357.0 µs vs stock 539.5 ms — an algorithm the incumbent shipped but
   declined to route this shape to) × 10.4× from the tree on 32 cores.
   "Escape from the fallback, not a magic kernel, carries the orders of
   magnitude." Honest apples-to-apples named: TP=8 row, 171.4→121.0 µs,
   1.4×. Core-for-core at small k: one row-parallel core does top-32 of
   65,536 in ≈194 µs wall vs the stock factory's ≈300 µs/row
   (9,596.3/32), so 49.5× = 32× parallelism × ≈1.5× per-core algorithm —
   a factor that grows to 1,511× at k=2048 (stock's cost explodes with k).
2. **Algorithmic content itemized** with pointers to canonical owners:
   log-tree 1.66–2.01× (Alg. 1, §4-A); cost model + rectangle embedding
   8–27% (Fig. 3, §4-B); law-gated chunk skip −45.1%/−36.3%, provably ≈0
   on 1–5-chunk column streams — hence the gate (Fig. 2, §4-D);
   structural term: selection-not-sort NoC traffic — only M-wide
   candidate windows, P−1 transfers, losers never move (Fig. 1).
3. **Architecture fit** in one sentence (verdict pointer to §3): cheap
   oblivious comparison + affordable decisions + dear materialization ⇒
   parallelized comparison network, work skipped, never data compacted.
4. **What we don't win on**: roofline gap 9.7–29.9× (contributors
   pointered; feasibility still \missing per G6); remaining levers are
   NOT smarter compare kernels (§5-D micro-op audit recovered 5–24%) but
   the envelope (routed−op = 46% of user-facing time at the anchor) and
   producer fusion (blaze 24.4 µs = the existence proof, caveats §6-B).
5. **The dispatch layer is part of the design space**: 44.2×/41.5×
   sampling wins = zero kernel work (eligibility routing + dropping three
   optional call-site args), audited bit-exact — values 100%, all 27,756
   index diffs across 3,200 audited rows proven bf16 ties, pow2 control
   bit-identical (evidence/i5-sampling-relaxation/). §6-E caveat now
   points at this audit; tokens/s-unmeasured retained.

Intro (§1 candor ¶) and conclusion (§9) gained one attribution pointer
sentence each.

## Job 2 — Pseudocode + cost-model formalization (owner-directed)

- **Algorithm 1** (§4-A, algorithmicx, scriptsize): leaf streaming loop
  with the dominance-skip guard (`row-par. ∧ c ≥ max(2,K/4) ∧
  max(chunk_c) < W_p[K]`, soundness + gate comments) and the
  ⌈log₂P⌉-level semaphore-tree merge with NoC-traffic comments (one
  M-wide window per edge; P−1 total; losers discarded in place).
  Referenced from §4-A, §4-C (the rule = the guard), §6-F.
- **Eq. 1 extended to the argmin form**: cost(P) = 2⌈C/P⌉+⌈log₂P⌉ with
  P* = argmin over P ∈ [2, min(C,128)] s.t. P = a×b embeds — the
  routing predicate/design-space argument deliberately NOT pseudocoded
  (prose/diagram material per direction).
- main.tex: +`algorithm` + `algpseudocode` packages.

## Job 3 — Tufte figure pass (fig/src/make_whywin_figs.py, 6 critique roundtrips)

All three figures regenerated grayscale-first, Times-matched (Nimbus
Roman + STIX math), designed at final print width, every number read
from committed CSVs (script prints the model-fit table each run):

- **Fig. 1 (fig-s1-operator)**: tournament-bracket dataflow — chunk strip
  → per-slice M-wide windows → 2-level tree → root, with per-phase cost
  tags (2⌈C/P⌉ units; ⌈log₂P⌉ levels), the NoC-traffic annotation, and
  the 13×10-grid 8×4-rectangle embedding panel. Direct labels, no legend.
- **Fig. 2 (fig-s2-skip-law)**: BACK IN THE PAPER (cut rev 0). Exact
  binomial (solid) + e^(−K/(c+1)) (dotted) for K=32/512; both max(2,K/4)
  gates; dead column-parallel (⌈C/P⌉=1–5) vs live row-parallel regions;
  measured −45.1%/−36.3% end-to-end callouts computed in-script from the
  tileskip CSVs. Caption states the curves are the model (no on-device
  skip telemetry exists — §6-D epistemic limit intact).
- **Fig. 3 (fig-e1-pscaling)**: the money figure — 15 measured points
  (psweep4_full.csv, G5 closed) vs the unfitted model (measured unit ×
  Eq. 1); chosen-P* rings labeled with computed deviations (34.4 µs,
  3.8% off; 23.3 µs, 0.3% off); honest "+33% at P=2" conservative-tail
  note; ⌈C/P⌉ bump annotated; axis notes the P≥2 factory floor
  (TT_FATAL :443, psweep_p1 UNSUPPORTED — coordinator-measured).

Critique-roundtrip log (rendered + Read as image each time):
- RT1: F1 tree geometry broken + text collisions; F2 two label
  collisions + clipped annotations; F3 staircase artifact
  (fractional-c evaluation), floor-clamp fake plunge, triple text
  collision. All fixed (F1 rebuilt as bracket tree; F3 evaluated at
  integer c only).
- RT2: F1 root/placement-label collision; F2 chosen-P* labels clipped at
  axes, series label on a data point; F3 formula-note/gate/callout
  collisions. Fixed.
- RT3: F1 clean (no findings). F2 P*=104 label crossing model line; F3
  callout vs gate label. Fixed.
- RT4: F2 clean after 2-line labels; F3 callout still on gate label +
  leader speared the 1.8% point → callout made free-standing, moved.
- RT5: F3 callout struck the K=512 identity label → labels repositioned;
  hyphen-minus → true minus.
- RT6/RT7: micro-nudge of the K=512 identity label; then no findings.
- Final in-context check at print size on pages 4, 5, 7, 8: no findings
  (two consecutive clean inspections — convergence).
- F4 (design-space bars) NOT re-added: budget; the three cyc/elt numbers
  stay in §3-B prose (fig-d2 remains available in fig/).

## Job 4 — What gave way (page budget)

Additions ≈ 1.7 col; paid in full — body is 10.0 pages exactly. The two
structural give-ways (both TRIMLOG-logged with survival locations):
old Tab. I (ingredients — every cell already restated in §3-B prose) and
old Fig. E2 (skip A/B bars — all five cells verbatim in §6-D prose).
Plus: §2-D GPU recap → one paragraph merged into §2-C (BUDGET rule 1
terminal form, all cites kept); §7-d totalOrder delta → §5-A (sole
carrier, cites kept); ~45 sentence-level dedup trims across
§1–§9 (full ledger in TRIMLOG "revision 2"). Never-cut list intact.

## Consistency re-checks

- 34.3/34.4 anchor-twin discipline re-verified on the final PDF: all six
  34.4's are P-sweep/re-pin/stop-rule contexts (incl. Fig. 3's chosen-P*
  label — P-sweep data), all 34.3's competition contexts.
- No ASCII "x" ratios; anonymity grep clean; bibliography 52 cited
  (no cites orphaned by the trims — verified per key).
- evidence.md gained §6 "Revision-2 additions" (rows A1–A7: factor
  chain, envelope 46%, core-for-core, i5 audit, P≥2 floor, Fig. 3 fit,
  Fig. 2 model-only caveat).

## Files touched

`main.tex`, `sections/00–07`, `fig/src/make_whywin_figs.py` (new),
`fig/{fig-s1-operator,fig-e1-pscaling,fig-s2-skip-law}.pdf`
(regenerated), `fig/README.md`, `TRIMLOG.md`, `STATUS.md`,
`REVISION-2.md` (this file), `../evidence/paper/evidence.md` (§6
appended). Nothing committed; nothing run on device.
