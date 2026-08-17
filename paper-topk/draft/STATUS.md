# STATUS — revision 1 (problem statement + post-audit scenarios), 2026-08-16

Supersedes the integration-pass STATUS of the same date. Changes this pass:
(1) new §2-A "Problem Statement and Data Contract" (owner-directed); (2)
scenarios table/prose re-pinned to the POST-AUDIT `scenarios1_table.csv`
numbers with the two mandated honesty footnotes; (3) ~1 column of
TRIMLOG-logged cuts to hold the 10.0-page body limit. Full trim ledger:
TRIMLOG.md "revision 1" section. Revision summary: REVISION-1.md.

## Compile

- **Clean.** `latexmk -pdf` from scratch: 0 errors, 0 overfull/underfull
  boxes, 0 undefined `\cite`/`\ref`, no duplicate labels.
- **Pages: 12 total = 10.0 body + 2 references.** Body ends on page 10;
  references begin at the top of page 11 — the 10-page IPDPS body limit is
  met exactly.
- Bibliography: 60 entries, **52 cited** (+2 this pass: `blacher2023vqsort`,
  `intel_x86simdsort` for §2-A's CPU-selection contrast — both marked
  UNVERIFIED, resolve before camera-ready like the other 18
  empty-author/UNVERIFIED items). 8 entries still held uncited.

## What changed (revision 1)

- **§2-A Problem Statement and Data Contract** (new, first subsection of
  Background): formal top-k definition (bf16^N rows in DRAM → values AND
  original indices, exact, sorted descending; stable=false tie semantics =
  any valid winner set, deterministic per input, bit-identical relaunches),
  grounded in the 62-cell contract suite (`test_topk_contract.py` T1 tiers:
  canonicalized-input multiset under the hardware sign-magnitude order,
  x[index]==value bit-exact); the three production contract variants
  (indices-only vs values+indices, multi-row, valid-length prefix); data
  provenance/destination (upstream matmul/indexer/gate → DRAM-interleaved →
  three consumer classes with real shapes, cross-referenced to Table IV);
  and a CPU-vs-GPU-vs-this-machine contrast paragraph (SIMD quickselect /
  radix-select affordances / no-atomics-no-scatter + 81-cycle rendezvous).
  §1 and old §2-B duplicates were consolidated into it (TRIMLOG).
- **§6-E scenarios re-pinned to post-audit CSV**: op cells now measured
  WITH values output (193.8 / 121.0 / 188.1 µs; ratios 49.5× / 1.4× /
  47.4×); MoE gate rows now the measured production-k op (round-to-16 +
  slice): 24.2→4.07 µs (5.9×), 77.5→4.04 µs (19.2×) — replacing the k=16
  ceiling probe (3.82/3.83, 6.3×/20.3×, retained only in the CSV notes).
  Two new table-note footnotes: values output ≤0.2% over indices-only at
  sampling shapes but ≈6% at tiny gate shapes; core counts are launched
  grids with min(rows,130) active on the row-parallel path. Caveat sentence
  now names all three routing disqualifiers (indices_tensor, sub_core_grids,
  stable=True) individually.
- Cuts to pay for the above: BUDGET rules 1 (§2-D GPU recap) and 4 (§5-B
  histogram subsection → Table II rows + 2 sentences) plus dedup-only
  sentence trims and table/figure formatting — every item and its survival
  location in TRIMLOG.md. Never-cut list verified intact.

## Consistency pass (re-run this pass)

- Engine vocabulary unchanged and consistent (stock `ttnn.topk` /
  `topk_large_indices` / routed; five arms defined once in §6-B).
- 34.3 vs 34.4 µs anchor-twin discipline re-verified by grep: every 34.3 is
  competition-context, every 34.4 is re-pin/P-sweep/stop-rule-context; the
  §6-A disclosure stands.
- Scenario numbers: single source is the post-audit
  `baselines/comp3/scenarios1_table.csv`; no stale pre-audit values remain
  (grep-verified: 193.6/120.8/187.9/3.82/3.83/6.3×/20.3×/1.42×/49.6/47.5
  absent outside unrelated contexts).
- 1.35 GHz busy-clock caveat: §6-A full statement + §3/§4 footnotes — all
  intact (never-cut).
- `\ref` targets for the new label `sec:bg-problem`: 5 references, all
  resolve; no duplicate labels.
- Double-anonymity grep: clean (no names, repo URLs, branch names, hashes;
  new bib URLs are vendor/third-party artifacts).
- LLM-ism grep: clean. `×` 153+/0 over `x`; `≈` over `~` maintained.
- Abstract: 320 words (unchanged — house style).

## Missing-evidence flags

**1 remaining** (unchanged): §6-F roofline-v2 derivation artifact (G6).
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
  entries), `refs.bib.bak`, `TRIMLOG.md` (integration + revision-1
  sections), `STATUS.md` (this file), `REVISION-1.md` (revision summary).
- All changes uncommitted per instruction.
