# REVISION-1 — problem statement, post-audit scenarios, re-hardening
2026-08-16. All changes uncommitted. Build: `latexmk -pdf` clean from
scratch — 12 pages = **10.0 body + 2 references** (body ends page 10,
references start at top of page 11), 0 overfull/underfull, 0 unresolved
`\cite`/`\ref`, exactly 1 `\missing{}` flag (G6, unchanged by design).

## Job 1 — Problem statement (§2-A, new)

New first subsection of Background: **"Problem Statement and Data
Contract"** (`sections/02-background.tex`, label `sec:bg-problem`),
answering the owner's demand in three paragraphs:

1. **Formal definition + contract.** One or more rows x ∈ bf16^N resident
   in device DRAM → the k largest elements' values AND original indices,
   exact (not approximate), sorted descending; stable=false ties = any
   valid winner set, deterministic for a fixed input. Grounded in what
   `tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py`
   actually pins (T1 tiers): values are the exact top-k multiset of the
   input *as canonicalized by the datapath* (forward-reference to §5's
   silicon findings) under the hardware sign-magnitude total order;
   x[index] == value bit-exact; repeated launches bit-identical (I13).
   Production variants named: indices-only vs (values, indices), multi-row
   batches, valid-length prefix (stale-tail masking).
2. **Provenance and destination.** x is always the upstream operator's
   output (lm-head matmul / attention-indexer score / expert-gate
   projection) in a DRAM-interleaved tensor; results return to DRAM. Three
   consumer classes cover every surveyed call site (evidence:
   `paper-topk/evidence/scenarios/callsites.md` +
   `baselines/comp3/scenarios1_table.csv`): sampling (values+indices,
   k=32, vocab shards N=16–65K, every token), sparse-attention KV gather
   (indices only, k=512–2048, contexts to 1M, every layer), MoE dispatch
   (k=4–16 over 128–512 experts, every layer). Consequence stated: a
   selection engine's real cost includes producer/consumer layout
   conversions — why the paper prices composites. Cross-referenced to
   Table IV (§6-E) both ways.
3. **Machine-class contrast.** One tight statement each: CPU
   (x86-simd-sort/vqsort-class vectorized quickselect — coherent caches,
   cheap branchy control, free scalar pivot logic; 2 new UNVERIFIED bib
   entries), GPU (atomic histograms, scatter compaction, grid sync —
   pointer to §2-D, not restated), this machine (no atomics/scatter/global
   barrier, NoC semaphores over software-managed SRAM, 1–3-bit SFPU
   counting, fixed data-oblivious instruction streams as the fast path,
   data-dependent decision = measured 81-cycle rendezvous).

Consolidation, not duplication: §1's absences list compressed to one
sentence pointing here; old §2-B's "load-bearing absences" and
bf16-datatype paragraphs deleted (absorbed); §2-D kept as the detailed GPU
recap the contrast points into.

## Job 2 — Scenarios re-pinned to post-audit numbers (§6-E)

The scenarios table existed but carried pre-audit values. Updated to the
POST-AUDIT `baselines/comp3/scenarios1_table.csv` (op arm measured with
`return_values=True`, values verified elementwise):

| row | today | op (was → now) | ratio (was → now) |
|---|---|---|---|
| Qwen3 TP=4 sampling | 9,596.3 µs (1c) | 193.6 → **193.8** µs | 49.6× → **49.5×** |
| TP=8 pow2 (control) | 171.4 µs (65c) | 120.8 → **121.0** µs | 1.42× → **1.4×** |
| 1-chip split sampling | 8,923.6 µs (1c) | 187.9 → **188.1** µs | 47.5× → **47.4×** |
| DSA k2048 / k512, MSA | already the op | unchanged | 1.0× (controls) |
| MoE gate top-4/128 | 24.2 µs (1c) | 3.82\* → **4.07** µs | 6.3×\* → **5.9×** |
| MoE gate top-10/512 | 77.5 µs (1c) | 3.83\* → **4.04** µs | 20.3×\* → **19.2×** |

\* was the k=16 ceiling probe; now the measured production-k op
(k rounds to 16 + slice — the route's own internal mechanism), so the gate
rows are real measurements, not ceilings. Routed cells unchanged (217.0 /
171.3 / 215.2 µs; 44.2× / 1.00× / 41.5×).

Two mandated honesty footnotes added to the table note: (1) op cells
include the values output — ≤0.2% over indices-only at sampling shapes,
≈6% at the tiny gate shapes; (2) core counts are launched grids,
min(rows,130) active on the row-parallel path. The canonical-form caveat
now names all three routing disqualifiers (indices_tensor, sub_core_grids,
stable=True) individually; tokens/s-unmeasured (G12) retained verbatim.

## Job 3 — Fit and re-harden

- Cost of additions ≈ 0.85 column; paid via BUDGET.md order — rule 1 (§2-D
  GPU recap tightened) and rule 4 (§5-B histogram subsection → Table II
  rows 3–5 + 2 sentences) — plus dedup-only sentence trims across
  §1/§3/§4/§5/§6/§7 and formatting (Table I/II column widening to kill
  wrapped lines; figures S1→0.75, E1/E2→0.76). Every trim logged in
  TRIMLOG.md "revision 1" with the survival location of every
  evidence-bearing number; **no evidence-bearing number was deleted** —
  the only numbers that left the paper are the superseded pre-audit
  scenario values and the k=16 ceiling-probe pair, both replaced by
  post-audit measurements (old pair survives in the CSV notes column).
- Never-cut list verified intact: 1.35 GHz clock caveat, blaze fairness
  caveats, scenarios canonical-form caveat, all four reopening conditions,
  forecast no-go story. (One rhetorical closing clause of the conclusion —
  "any one ... falsifies the closure" — was trimmed; the falsification
  semantics remain in "reopens if any one holds".)
- Consistency checks re-run: vocabulary (arms/engine names), 34.3-vs-34.4
  anchor-twin contexts, stale-number grep (clean), \cite/\ref resolution
  (clean), duplicate labels (none), double-anonymity grep (clean),
  LLM-ism grep (clean), ×/≈ conventions (clean), abstract 320 words.
- refs.bib: +2 entries (blacher2023vqsort, intel_x86simdsort), both marked
  UNVERIFIED per the file's convention — resolve before camera-ready.
- STATUS.md updated: page count, change log, and the reviewer
  attack-surface list — the problem-statement clarity attack is removed;
  the "conditional production wins" attack is substantially blunted
  (values-output cost disclosed, launched-vs-active cores disclosed,
  no-change controls in-table); what remains (call-site change required,
  tokens/s unmeasured) is stated, not hidden.

## Files touched

- `sections/00` (untouched), `01-intro.tex`, `02-background.tex`,
  `03-design-space.tex`, `04-system.tex`, `05-characterization.tex`,
  `06-evaluation.tex`, `07-related.tex`, `refs.bib`, `TRIMLOG.md`,
  `STATUS.md`, `REVISION-1.md` (this file). Nothing committed.
