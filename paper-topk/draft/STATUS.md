# STATUS — revision 2 (why-we-win attribution + pseudocode + Tufte figures), 2026-08-17

Supersedes the revision-1 STATUS. Changes this pass (owner-directed
"explain why we win" + "pseudocode and Tufte diagrams, no slop"):
(1) new §6-F "Attribution: Where the Speedup Comes From" — the explicit,
measured decomposition (routing escape × algorithm × architecture fit ×
limits × dispatch layer), with intro/conclusion pointer sentences;
(2) Algorithm 1 (log-tree + dominance-skip guard pseudocode) and Eq. 1
extended to the argmin-with-embedding form; (3) all three figures
regenerated as grayscale Times-matched Tufte redesigns from committed
CSVs (fig/src/make_whywin_figs.py), Fig. 2 (skip law) re-added, Fig. 3
now the model-vs-silicon money figure with chosen-P* deviations and the
P≥2 factory floor; (4) ~1.7 col of additions paid by cutting old Tab. I
(ingredients; all cells survive in §3-B prose) and old Fig. E2 (skip
A/B; all cells in §6-D prose), merging §2-D into §2-C, and ~45 dedup
trims. Full ledger: TRIMLOG.md "revision 2". Summary: REVISION-2.md.

## Compile

- **Clean.** `latexmk -pdf` from scratch: 0 errors, 0 overfull hboxes,
  0 undefined `\cite`/`\ref`, no duplicate labels. Three vbox warnings
  ≤1.7 pt on float/refs pages (sub-visible; introduced by the new float
  set; noted, not hidden).
- **Pages: 12 total = 10.0 body + 2 references.** Body ends on page 10;
  references begin at the top of page 11 — the 10-page IPDPS body limit
  is met exactly.
- Bibliography: 60 entries, **52 cited** (unchanged — no cites orphaned
  by the trims, verified per key). UNVERIFIED items unchanged (resolve
  before camera-ready).

## What changed (revision 2)

- **§6-F Attribution** (replaces the roofline subsection, which it
  absorbs intact incl. the G6 \missing flag): the 18,401× anchor chain
  factored exactly on Table II's own arms (1.17× micro-op × 1,511×
  single-core proxy × 10.4× tree-on-32-cores); TP=8 honest 1.4×;
  core-for-core ≈1.5× at k=32 (49.5× = 32× × 1.5×); itemized algorithm
  terms with owners; one-sentence §3 verdict; roofline gap + the two
  measured next levers (envelope 46%, blaze fusion 1.4×) vs the 5–24%
  compare-kernel headroom; the dispatch layer as the highest-return arm
  with the i5 bit-exactness audit (27,756 diffs all proven ties over
  3,200 rows, pow2 control bit-identical).
- **Algorithm 1 + argmin Eq. 1** (§4): control flow incl. the
  max(2,K/4)-gated dominance skip and the P−1-window NoC-traffic
  comments; the routing predicate deliberately left as prose.
- **Figures** (fig/src/make_whywin_figs.py; 6 critique roundtrips +
  2 clean in-context inspections — log in REVISION-2.md): Fig. 1
  operator bracket-tree schematic; Fig. 2 skip-law (model curves +
  measured −45.1%/−36.3% callouts, caption says model); Fig. 3 P-sweep
  vs unfitted model (per-point fit table printed by the script).
- **evidence.md §6** appended (rows A1–A7) covering every new number.

## Consistency pass (re-run this pass, on the rev-2 PDF)

- Engine vocabulary unchanged and consistent (stock `ttnn.topk` /
  `topk_large_indices` / routed; five arms defined once in §6-B).
- 34.3 vs 34.4 µs anchor-twin discipline re-verified by grep: every 34.3 is
  competition-context, every 34.4 is re-pin/P-sweep/stop-rule-context; the
  §6-A disclosure stands.
- Scenario numbers: single source is the post-audit
  `baselines/comp3/scenarios1_table.csv` (now Table III after the
  ingredients-table cut; characterization = Table I, competition =
  Table II — all `\ref`-driven, no hardcoded numbers).
- New §6-F numbers are arithmetic on printed table cells or evidence-pack
  artifacts (trace in the 06-evaluation.tex header comment + evidence.md
  §6 rows A1–A7); the factor chain composes exactly (1.17 × 1,511 × 10.4
  = 18,401).
- 1.35 GHz busy-clock caveat: §6-A full statement + §3/§4 footnotes — all
  intact (never-cut).
- `\ref` targets for the new label `sec:bg-problem`: 5 references, all
  resolve; no duplicate labels.
- Double-anonymity grep: clean (no names, repo URLs, branch names, hashes;
  new bib URLs are vendor/third-party artifacts).
- LLM-ism grep: clean. `×` 153+/0 over `x`; `≈` over `~` maintained.
- Abstract: 320 words (unchanged — house style).

## Missing-evidence flags

**1 remaining** (unchanged): roofline-v2 derivation artifact (G6), now
flagged inside §6-F's roofline paragraph.
Adjacent disclosed-not-flagged items unchanged (G4 skip telemetry, G8 blaze
breakdown, cost-model overlay arithmetic).

## Top-10 reviewer attack surfaces (updated)

Two attacks from the previous list are now materially blunted:

- ~~"The problem is never crisply stated / what exactly does the op
  promise?"~~ — **removed** by §2-A: formal contract, tie semantics, data
  provenance/destination, and the CPU/GPU/this-machine delta now live in one
  place, evidence-grounded in the contract suite.
- **"Production wins are conditional / cherry-picked synthetic shapes"** —
  **substantially blunted**: Table IV now carries post-audit numbers with
  values output included (the "indices-only benchmarking" objection is
  pre-empted by the ≤0.2%/≈6% footnote), launched-vs-active core counts
  disclosed, and its own no-change controls. What remains of the attack is
  honest and stated: the call-site change is required (three disqualifiers
  named) and end-to-end tokens/s is unmeasured (G12) — an attacker can still
  ask for the tokens/s demo, but not for hidden conditions.

Still standing (renumbered): 1 variance/one-board-one-day (G3); 2 unverified
busy clock (G9 — capture AICLK before submission if possible); 3 n=1 chip
generality (G1); 4 no measured GPU side-by-side anchor; 5 roofline gap
readable as "still 10–30× off" while G6 unresolved; 6 no on-device skip-rate
telemetry (G4); 7 blaze per-core breakdown (G8); 8 tokens/s unmeasured
(G12, above); 9 LLM-agent methodology discount risk; 10 novelty surface
(measurement-first framing must carry it). New minor surface: two UNVERIFIED
CPU-selection bib entries added for §2-A — verify before camera-ready or a
bibliography-checking reviewer will flag them.

## Files

- `main.pdf` (12 pp), `main.tex`, `sections/*.tex` (8), `refs.bib` (60
  entries), `refs.bib.bak`, `fig/src/make_whywin_figs.py` (+ regenerated
  fig-s1/e1/s2 PDFs, updated fig/README.md), `TRIMLOG.md` (integration +
  revision-1 + revision-2 sections), `STATUS.md` (this file),
  `REVISION-1.md`, `REVISION-2.md`.
- All changes uncommitted per instruction; nothing run on device.
