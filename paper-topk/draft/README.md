# Paper draft — "Exact Top-K Selection on a Dataflow Many-Core: A Measured Design-Space Study"

Target: IPDPS 2027, 10 pages IEEEtran conference double-column, references
unlimited, DOUBLE-ANONYMOUS.

## Layout

- `main.tex` — IEEEtran skeleton; `\input`s one file per section.
- `sections/*.tex` — section stubs; each header comment names its brief.
- `briefs/NN-*.md` — **write from these.** Each brief fixes the section's
  single job, the claims it owns (with evidence pointers into
  `../evidence/paper/evidence.md`), its column budget, the figures/tables it
  owns, per-section style directives, and hazards.
- `refs.bib` — full bibliography from `../evidence/paper/related-work.md`;
  entries flagged `UNVERIFIED` must be resolved before camera-ready.
- `BUDGET.md` — the 10-page allocation + pre-agreed overflow cuts.
- `fig/` — figure data-source map + binding plotting rules.

## Non-negotiables (repeated from the briefs)

1. **Evidence discipline:** every number traces to
   `../evidence/paper/evidence.md`. If it is not there, write
   `\missing{what}` — the macro renders red and must be empty at submission.
2. **Clock caveat:** cycle→µs conversions assume a 1.35 GHz busy clock (not
   captured under load; idle 800 MHz measured). Stated in §6.1 and in every
   caption with derived-µs values.
3. **Anonymity:** no repo URLs, commit hashes, branch names, internal tool
   names, or unmasked self-citations. Hardware naming (Tenstorrent
   Blackhole p150a) is fine.
4. **Repositioned claims:** C2 = first *measured* full top-k on commercial
   NoC-mesh silicon (cite Krizanc & Narayanan '92/'93 up front); C4 =
   datapath-numerics/selection-primitives first-ness only (cite the informal
   Blackhole microbench report); C3 must contrast WarpSelect + GVR.
5. **Known missing evidence** (do not print without resolving): roofline-v2
   derivation (G6), blaze per-core breakdown (G8), on-device skip-rate
   telemetry (G4 — figures use law→simulation→time instead).

## Build

`pdflatex main && bibtex main && pdflatex main && pdflatex main`
(requires IEEEtran; no tt-metal build involved).
