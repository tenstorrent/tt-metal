# REVISION-3 — comp4 re-measure + TILE-native / gate carve-out / skip-telemetry pass
2026-08-17. All changes uncommitted (orchestrator reviews/commits). Host-only:
no device runs, no tt-metal builds. Build: `latexmk -pdf` clean — **12 pages =
10.0 body + 2 references** (body ends page 10, references start at top of
page 11), **0 overfull hboxes**, 3 vbox warnings ≤1.7 pt (same sub-visible set
as rev-2), 0 unresolved `\cite`/`\ref`, exactly 1 `\missing{}` flag (G6, by
design). Every changed number verified against
`tests/.../reduction/baselines/comp4/{competition_table,scenarios7_table}.csv`
and `paper-topk/evidence/i2-i3-i4-landings/` — nothing typed from memory.

## What landed since rev-2 (evidence, all committed)

- **I2 TILE-native I/O + u16 + per-shape route policy** (commit 482de67d779;
  `i2-tile-native-report.md`): routed column −67..−84% at k_rounded ≤ 1024,
  anchor 63.4 → 51.5 µs (−19%).
- **I3 MoE-gate route carve-out** (fdb81ed027e; `i3i4-landing-report.md`
  Mission A): 24.2 → 8.18 µs (2.96×), 77.4 → 8.34 µs (9.29×), zero model-code
  changes, pre-registered rule routed×1.05 ≤ stock passed at both cells.
- **I4 skip telemetry + gate A/B** (aa721796e3f; `g4_curve.csv`,
  `gate_ab_summary.txt`): G4 epistemic limit CLOSED — P(skip|c) tracks
  e^(−K/(c+1)) at all 120 tested positions over 400 iid rows; E[skips]/row
  65.11 obs vs 66.62 law (−2.3%); divisor ablation 2/4/8 ties (best −0.8%,
  /8 regresses k=512 +3.2%) → K/4 stands.
- Fresh canonical numbers: `baselines/comp4/` = comp3 pinned arms + routed
  column re-measured on the landed tree (5-iteration medians, 0 wrong);
  `scenarios7_table.csv` fully re-measured with uniform provenance stamps.

## Job 1 — Table II routed column + global number sweep

All 24 routed cells + all 24 pre→routed speedups replaced with comp4 values.
Headline sweep (grep-verified 0 stale in the PDF): 63.4 → 51.5 µs and 9,956× →
12,265× in abstract, §1, §6-B, §9 (the single surviving "63.4" is §6-F's
intentional before→after); routed span 56×–22,481× → 329×–31,905×;
k512@65536 91.7 → 20.9; k1024@2048 134.2 → 21.3 (the I1 anomaly cell, now
unremarkable); intro hook 9,596 → 9,587 µs. Table II caption discloses the
routed re-measure (same harness/gates, 5-iteration medians, 0 wrong).

## Job 2 — §6-B TILE-native paragraph + §6-F envelope limb rewrite

- New §6-B paragraph: TILE input by layout dispatch, opt-in TILE output +
  uint16 indices, envelope deleted at k_rounded ≤ 1024 (−67..−84%); at k=2048
  the tilize chain measured cheaper and is KEPT — per-shape policy by
  measurement. (The §4 "tight paragraph" option was declined for budget; BH
  alignment-rule detail lives in evidence.md §7 B2 per the fallback rule.)
- §6-F "what we don't win on": the envelope lever is now BANKED — was 46% of
  user-facing time at the anchor, deleted by the landing; anchor −19%; the
  17.2 µs residual decomposes as 9.8 µs op-internal (staged TILE-input read)
  + 7.4 µs kept-tilize pair. Fusion (blaze 24.4 µs → next 1.4×) remains the
  open lever. Dispatch limb gains the carve-out (zero call-site changes).

## Job 3 — §6-D telemetry + gate ablation (epistemic limit closed)

The "no on-device skip-rate telemetry exists" limb is deleted and replaced by
the measured result: compile-time-gated telemetry (disabled path proven
byte-identical by ELF-disassembly diff), 400 iid rows, per-position law match
at all 120 tested positions, E[skips]/row −2.3% vs law; divisor ablation
2/4/8 files as the gate-sensitivity study (K/4 stands, law-predicted ties).
§4-E pointer updated ("quotes the A/B, the guard cell, and the on-device
telemetry").

## Job 4 — Fig. 2 measured overlay (2 critique roundtrips + in-context check)

`fig/src/make_whywin_figs.py` extended to read
`paper-topk/evidence/i2-i3-i4-landings/g4_curve.csv` (TOTAL row parsed for the
aggregate print) and overlay 120 measured per-position skip rates as open
circles on the K=32 curves; identity label now three lines
("exact (solid) / e^(−K/(c+1)) (dotted) / silicon (∘, 400 rows)").
- RT1: long identity line struck through by the K=32 curve label — label
  broken to three short lines, K=32 label moved right of the gate.
- RT2: clean.
- In-context print-size check (page 5): clean — dots hug the curve, caption
  now says the dots are silicon (the old "only silicon numbers are the
  end-to-end deltas" line deleted). Script prints the overlay stats on every
  run (120 positions, 65.11 vs 66.62, −2.3%).

## Job 5 — Table III (scenarios7) + §6-E rewrite

All rows re-sourced from comp4/scenarios7_table.csv. Gate rows restructured to
the TP=8 pattern (today = routed, since the carve-out fires automatically):
8.18/8.34 µs (130), footnote carries the 24.2/77.4 µs single-core before and
the k→16 slice mechanism; prose carries 2.96×/9.29× + "zero model-code
changes". Pending-A/B language ("a route extension still needs an A/B against
the layout-envelope cost") deleted — the A/B was run and passed. DSA
routing-cost 20–25% → 22–26%; sampling caveat paragraph (canonical form,
tokens/s unmeasured) intact — still true, the sampling call sites still pass
disqualifying args.

## Job 5b — §6-F tree-limb applicability boundary (coordinator addition)

Two sentences guarding the "k=2048, W=65,536 ⇒ GLM prefill got 10× faster"
misread (evidence row B7; arithmetic on printed cells only): the tree
converts cores into latency at ≈1.0 core-efficiency — P=2 = 183.0 µs
(§6-C) = 366 core-µs/row vs the shipped per-row kernel's 357.0 µs (Tab. II
proxy) on one core, +2.5% — so the 10.4× serves latency-bound low-row
callers (decode-shape indexer, low-batch tails), while rows ≥ grid callers
are already core-optimal: the 160-row prefill indexer's 712.3 µs (Tab. III)
≈ two 130-core row-waves (2 × 357.0), reported there 1.00×/0.80×; per-core
work at that shape is the fusion lever, closing into the existing fusion
sentence. Paid by TRIMLOG "post-pass addition" trims; page-9 print-size
check clean.

## Job 6 — What gave way (page budget)

Additions ≈1.0 col, paid in full; body is 10.0 pages exactly. No never-cut
item touched (clock caveat, blaze caveats, scenarios canonical-form caveat,
reopening conditions, forecast no-go story all intact). Full ledger: TRIMLOG
"revision 3" — highlights: mutual-pointer dedups (§4-D↔§3-D rendezvous link,
§4-E↔§6-D guard numbers, §5↔§6-A slope method, §7-a↔§4-B mesh bound), §6-B
small-k middle example (BUDGET rule 2 direction), §6-C bump model-vs-measured
parenthetical (survives in the fig-script fit table + evidence A6), §8/§9
sentence-level trims.

## Consistency re-checks

- 34.3/34.4 anchor-twin discipline: untouched (34.3 competition contexts,
  34.4 P-sweep/re-pin/stop-rule contexts).
- Stale-number grep on the final PDF: 9,956 / 134.2 / 9,596 / 217.0 / 215.2 /
  22,481 / 44.2× / 41.5× / 19.2× / 77.5 all 0 hits; new numbers all present.
- §6-F residual arithmetic: 9.8 + 7.4 = 17.2 = 51.5 − 34.3 ✓ (44.1 − 34.3 =
  9.8 per the landing report's stage profile).
- Factor chain unchanged (1.17 × 1,511 × 10.4 = 18,401 — op/pre/now/proxy
  arms are pinned comp3 cells, byte-identical in comp4).
- Double-anonymity: new rendered text has no names, URLs, branch names, or
  hashes (landing hashes live in .tex comments / evidence pack only).
- evidence.md: §7 rows B1–B6 added; H1/H4/A7 marked superseded with comp4
  pointers; gap G4 marked CLOSED.

## Files touched

`sections/{00,01,04,05,06,07}` (02/03 untouched), `main.pdf` (rebuilt),
`fig/src/make_whywin_figs.py` + regenerated `fig/fig-{s1,e1,s2}*.pdf`
(s1/e1 byte-refreshed, content unchanged), `fig/README.md`, `TRIMLOG.md`,
`STATUS.md`, `REVISION-3.md` (this file),
`../evidence/paper/evidence.md` (§7 + supersede/close edits). Nothing
committed; nothing run on device.
