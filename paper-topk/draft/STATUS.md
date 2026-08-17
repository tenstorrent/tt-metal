# STATUS — revision 4 (silicon validation + attribution + system-context pass), 2026-08-17

Rev-4 delta (summary: REVISION-4.md; evidence §8 B8–B10): every printed anchor
replicated on the final landed tree (712.2/33.96/356.43 vs printed
712.3/34.3/357.0); §6-B's SFPLOADMACRO "worth 5–9%" corrected to the measured
3.6% (op) / 1.9% (proxy) disable-flag A/B — the win is banked, the proxy
inherits it (no longer "byte-identical"); §6-E gains the DSA system-context
lines (top-k ≈0.3% of GLM-5.2 e2e prefill; the live lever is the
regather/re-partition inverse-pair elision, landed model-side as fe1930d50c2
pending 8×4 validation); paid by the BUDGET rule-1 §2.3 compression (cites
kept). Compile clean: 0 errors, 0 overfull, body ends page 10 exactly
(verified by A/B build against the pre-edit tree).

Previous (revision 3) status follows.

# STATUS — revision 3 (comp4 numbers + TILE-native / carve-out / telemetry landings), 2026-08-17

Supersedes the revision-2 STATUS. Changes this pass (orchestrator-directed
"paper update for the three landed wins"): (1) Table II routed column + all
routed speedups re-pointed at `baselines/comp4/` (routed re-measured on the
landed tree; anchor 63.4 → 51.5 µs, 9,956× → 12,265×, swept through
abstract/§1/§6/§9); (2) §6-D chunk-skip epistemic limit CLOSED by on-device
telemetry (400 rows, law matched at all 120 positions, −2.3% aggregate) +
gate-constant ablation (K/4 stands); (3) Fig. 2 upgraded from model-only to
model + measured silicon dots (g4_curve.csv); (4) Table III fully re-sourced
from comp4/scenarios7 — MoE gates now captured automatically by the route
carve-out (24.2→8.18 µs, 77.4→8.34 µs; 2.96×/9.29×, zero model-code changes);
(5) §6-F envelope limb rewritten: the 46%-envelope lever is now BANKED
(TILE-native landing; k=2048 keeps the tilize chain by measured policy);
(6) new §6-B TILE-native paragraph; (7) coordinator-directed §6-F tree-limb
applicability boundary — the tree buys latency at ≈1.0 core-efficiency
(P=2: 366 core-µs/row vs 357.0 on one core, +2.5%), so 10.4× serves
latency-bound low-row callers while the 160-row indexer (712.3 µs ≈ two
row-waves of 357.0) is already core-optimal, guarding the "GLM prefill got
10× faster" misread (evidence B7). Additions ≈1.2 col paid by TRIMLOG
"revision 3" + "post-pass addition" dedup trims. Summary: REVISION-3.md.

## Compile

- **Clean.** `latexmk -pdf`: 0 errors, 0 overfull hboxes, 0 undefined
  `\cite`/`\ref`, no duplicate labels. Same 3 sub-visible vbox warnings
  ≤1.7 pt as rev-2 (float/refs pages).
- **Pages: 12 total = 10.0 body + 2 references.** Body ends on page 10;
  references begin at the top of page 11 — the 10-page IPDPS body limit
  is met exactly.
- Bibliography: 60 entries, 52 cited (unchanged; no cites orphaned by the
  rev-3 trims). UNVERIFIED items unchanged (resolve before camera-ready).

## What changed (revision 3)

- **Numbers**: every routed cell of Table II (24) + pre→routed column
  (span 329×–31,905×); every row of Table III (scenarios7); headline chain
  631.5 ms → 51.5 µs = 12,265×; intro hook 9,587 µs; DSA routing cost
  22–26%; §6-F sampling wins 44.9×/42.3× and core-for-core 9,586.9/32.
  Sources: `baselines/comp4/{competition_table,scenarios7_table}.csv`,
  `paper-topk/evidence/i2-i3-i4-landings/*` — evidence.md §7 rows B1–B6.
- **§6-D**: telemetry (byte-identical disabled path via ELF-diff; per-position
  law match; E[skips]/row 65.11 vs 66.62) + divisor ablation (2/4/8 ties,
  /8 regresses k=512 +3.2% exactly as the law predicts) → K/4 stands.
- **Fig. 2**: 120 measured per-position skip rates overlaid as open circles
  (2 critique roundtrips + in-context print-size check; caption no longer
  claims the curves are the only model content — the dots are silicon).
- **§6-E/Table III gate rows**: today = routed (carve-out fires
  automatically, TP=8-row pattern); footnote carries the 24.2/77.4 before;
  pending-A/B language deleted (the A/B ran and passed both cells).
- **§6-F**: envelope limb = banked lever (deleted at k_rounded ≤ 1024;
  anchor −19%; residual 9.8 op-internal + 7.4 kept-tilize); fusion
  (blaze 24.4 µs) is the remaining lever; dispatch limb gains the carve-out.

## Consistency pass (re-run this pass, on the rev-3 PDF)

- Stale-number grep on the final PDF: 9,956 / 91.7-routed / 134.2 / 9,596 /
  217.0 / 215.2 / 22,481 / 44.2× / 41.5× / 5.9×(gate) / 19.2× / 77.5 — all
  gone; the one surviving "63.4" is §6-F's intentional before→after.
- 34.3 vs 34.4 anchor-twin discipline: untouched by this pass; §6-A
  disclosure stands. Factor chain 1.17 × 1,511 × 10.4 = 18,401 unchanged
  (those arms are pinned comp3 cells, byte-identical in comp4).
- §6-F residual arithmetic checks: 51.5 − 34.3 = 17.2 = 9.8 + 7.4.
- Scenario numbers: single source is now `baselines/comp4/scenarios7_table.csv`
  (provenance-stamped, uniform head/tree/so hashes).
- 1.35 GHz busy-clock caveat: §6-A full statement + §3/§4 footnotes intact
  (never-cut). Blaze caveats, canonical-form caveat, reopening conditions,
  forecast no-go story: intact.
- Double-anonymity grep: clean (landing commit hashes appear only in .tex
  comments / evidence pack, never rendered).

## Missing-evidence flags

**1 remaining** (unchanged, by design): roofline-v2 derivation artifact (G6),
flagged inside §6-F's roofline paragraph. G4 (skip telemetry) is now CLOSED —
removed from the disclosed-not-flagged list; adjacent items G8 (blaze
breakdown) and the cost-model overlay arithmetic unchanged.

## Top-10 reviewer attack surfaces (updated)

Two more attacks materially blunted this pass:

- ~~"No on-device skip-rate telemetry — the skip story is inference from
  end-to-end time" (old #6/G4)~~ — **removed**: measured per-position curve,
  byte-identity proof for the disabled path, and a gate-constant ablation.
- **"The user-facing win depends on layout conversions you don't own"** —
  **substantially blunted**: the envelope is deleted at k_rounded ≤ 1024 and
  the k=2048 keep-the-tilize decision is a measured per-shape policy, which
  also strengthens the methodology story.

Still standing (renumbered): 1 variance/one-board-one-day (G3; comp4 routed
re-measure is same-board, same-day-class); 2 unverified busy clock (G9);
3 n=1 chip generality (G1); 4 no measured GPU side-by-side anchor; 5 roofline
gap while G6 unresolved; 6 blaze per-core breakdown (G8); 7 tokens/s
unmeasured (G12 — sampling call sites still pass disqualifying args);
8 LLM-agent methodology discount risk; 9 novelty surface; 10 (new, minor)
mixed-provenance Table II — comp3 pinned arms + comp4 routed column — is
disclosed in the caption, but a reviewer can ask why the whole grid wasn't
re-run; the answer (pinned arms byte-identical, only the routed path changed)
is in evidence.md §7 B1.

## Files

- `main.pdf` (12 pp), `sections/{00,01,04,05,06,07}` (02/03 untouched),
  `fig/src/make_whywin_figs.py` (+ regenerated fig-s1/e1/s2 PDFs; s1/e1
  content-identical), `fig/README.md`, `TRIMLOG.md` (adds "revision 3"),
  `REVISION-3.md`, `STATUS.md` (this file),
  `../evidence/paper/evidence.md` (§7 + G4-closed + supersede notes).
- All changes uncommitted per instruction; nothing run on device; no
  tt-metal builds.
