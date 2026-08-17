# Brief 06 — Evaluation

**Section file:** `sections/06-evaluation.tex`
**Budget:** 4.5 columns (2.25 pages) — the paper's largest section.

## Single job

Defend every headline number with the harness that produced it, then walk the
reader through five exhibits: competition table, P-scaling vs the cost model,
chunk-skip A/B, model-scenario table, roofline gap. Methodology-of-measurement
lives HERE (the agentic-campaign narrative lives in §7).

## 6.1 Harness and methodology (≈0.8 col)

Must state, in this order:
- **Measurement:** Tracy DEVICE KERNEL DURATION at op level; MATH_ISOLATE
  two-point slope with a 2.000× control tripwire at LLK level; one device
  process at a time; per-cell subprocess isolation. (Ledger footer method
  block — reuse verbatim facts, rephrase.)
- **Correctness gates:** every perf cell is gated exact/bit-exact against
  torch goldens — never PCC; 0 WRONG of 121 cells in the competition run;
  suites: contract 62, op 154, routed 220/220, adversarial 36-check, hang
  20/20×2, counting bench 51/51, RISC bench 21/21 (M4).
- **Determinism/provenance:** deterministic sweep (fixed seeds,
  torch.randn bf16 seed 0 for skip cells); every CSV row provenance-stamped
  (build + tree hash — describe mechanism, don't print hashes: anonymity).
- **Trials:** 3 trials/cell medians (competition); 3 trials × 5 measured
  iters for skip cells with spread <0.1%; scope51 CSVs carry stds (e.g.
  9,492,327 ± 1,391 ns). Acknowledge the Betz-standard gap honestly: single
  board, single day; no committed per-cell std for the 121-cell run (G3) —
  one candid sentence.
- **MANDATORY CLOCK CAVEAT (verbatim policy):** cycle→µs conversions assume
  a 1.35 GHz busy clock; the clock under load was not captured (idle
  800 MHz measured). State here once, AND in every caption of a figure/table
  containing derived-µs values from cycle measurements. Tracy-measured µs
  values (op level) are direct and do NOT need the caveat — keep the two
  classes distinct.

## 6.2 Competition table (Tab E1, ≈1 col incl. table)

- Source: `baselines/comp3/competition_table.csv` — 24 cells (k ∈ {512,
  1024, 2048} × N ∈ {2048..262,144}), five measured arms (prebranch stock,
  stock-now, routed, op, blaze at its native cell) + roofline + gap columns;
  every row status=MEASURED.
- Headlines to call out in prose: H1 (9,956× routed chain at k2048@65,536),
  H2 (34.3 µs @ 32c, 18,401× op chain), H4 (op vs prebranch 632×–51,430×;
  routed 56×–22,481×), H3 (23.3 µs @ 104c).
- blaze row (H5): 34.3 vs 24.4 µs = 1.4× — with BOTH caveats printed: blaze
  includes SDPA work; it is a FusedProgram, not a callable op. The
  "core-time equal" claim has NO committed breakdown (G8) — do not print it;
  if wanted, mark [MISSING-EVIDENCE: blaze per-core breakdown].
- Micro-op factors note (ledger footer): replay ≈1.17× is inside the stock
  ttnn.topk column (631.5 ms is without it; 539.5 ms with); SFPLOADMACRO
  5–9% inside the op columns — one footnote so speedup chains are honest.
- Small-k routing before/after (Tab E2 or merged rows): H7 40×–423×
  (k32@5000: 695→17.2 µs; k64@65,536: 17.95 ms→42.4 µs; k32@65,534:
  9.50 ms→38.5 µs) and the ≈89× cliff it removed (H8). Sources:
  `smallk_routefix/{routed_after,stock_nonpow2}.csv`, `scope51/`.

## 6.3 P-scaling vs cost model (Fig E1, ≈0.8 col)

- Data: `baselines/comp3/psweep4_full.csv` — 15 measured points, in-repo
  (the evidence pack's gap G5 is CLOSED; cite the CSV, not the ledger).
  k2048@65,536: 183.04/97.87/57.31/38.96/42.81/34.36 µs at P=2/4/8/16/24/32;
  k512@262,144: 560.19 → 23.30 µs across P=2..104 (9 points).
- Overlay: \costmodel × measured unit constants (1.46 µs @ M=512, 5.51 µs @
  M=2048, forecast.md §3). Call out the non-monotone bump at P=24 (42.81 µs
  > P=16's 38.96) and explain via the ceil terms — the model predicts it;
  that's the validation.
- One sentence each: monotone post-tree (vs the serial chain, H11); flattens
  near 130-core capacity; L1 never bound P (C2-5).

## 6.4 Chunk-skip A/B (Fig E2, ≈0.6 col)

- 5 cells × {baseline, ungated, gated}: rows=2 k32@65,536 279.14→153.19 µs
  (−45.1%, 1.82×); rows=8 →177.94 (−36.3%); gated no-win cells +0.04% to
  +0.51%; column-parallel guard 13.153 µs byte-identical-kernel unchanged.
  Source: rowskip-implementation.md §4 + `evidence/tileskip/{baseline,
  skipon,skipgated}.csv` + logs (3 trials, spread <0.1%).
- Tie to §4's forecast: predicted-vs-measured in one sentence — the no-go
  variant was never built; the built variant hit within the forecast's
  regime. Epistemic honesty: no on-device skip-rate telemetry (G4) — the
  measured quantity is time; skip fractions are simulated.

## 6.5 Model scenarios (Tab E3, ≈0.7 col)

- Source: `baselines/comp3/scenarios1_table.csv` — 8 production call-site
  shapes, all provenance-stamped, iters recorded.
- Rows to feature: Qwen3 TP=4 decode sampling (N=65,536 pad — the one pow2
  width that fails the W<65,535 gate → 9,596.29 µs single-core today;
  routed 217.01 µs, op 193.61 µs = 44–50×); split-vocab Llama
  (8,923.55→215.22 µs); TP=8 pow2 control (1.00× — today already the
  multi-core bitonic, the honest already-fast row); DSA indexer k=2048
  (712.5 µs @ rows=160 — today IS the op, row-parallel, chunk skip live);
  MoE gate no-change controls (k=4, k=10 — routing provably cannot fire).
- MANDATORY caveat (CSV notes column, verbatim sense): production sampling
  call sites pass indices_tensor/stable=True which disqualify routing — the
  routed column is the CANONICAL form (args dropped), not a free win; the
  end-to-end token/s measurement after a call-site change is unmeasured
  (G12). Controls are features: highlight that the table contains its own
  no-change controls — Kapre/Betz honesty style.

## 6.6 Roofline gap (≈0.4 col, prose + Tab E1's gap columns)

- Gap to the vendor llm_perf roofline: 9.7×–29.9× over all 24 cells,
  10–21× on N≥32K cells (H4; roofline provenance: vendor llm_perf
  repository tables + K-multipliers 0.612/0.850/1.000 — describe as "the
  vendor's published roofline," no PR URLs).
- The "roofline itself infeasible for bitonic kernels — sits below the
  comparator critical-path floor; roofline-v2 fits all 11 silicon points
  within 14.1%" claim is **[MISSING-EVIDENCE: roofline-v2 derivation
  artifact — G6]**. DO NOT print unless the derivation document is recovered
  or re-derived. Fallback framing: report the measured gap, decompose the
  known contributors (comparator critical path, tree levels, unpack floors),
  and leave feasibility open.

## Claims owned

H1–H8, H11 (absolute values), C2-2, C2-3, C2-5, C3-4 (measured deltas),
scenario-table claims, roofline-gap claims, M4 (correctness-gate discipline
as measurement methodology).

## Figures/tables owned (with exact sources)

| Exhibit | Content | Source |
|---|---|---|
| **Tab E1** (2-col table) | 24-cell competition: five arms + cores + roofline + gaps | `baselines/comp3/competition_table.csv` |
| **Fig E1** (1 col) | P-scaling, 2 curves + cost-model overlay | `baselines/comp3/psweep4_full.csv`; model constants forecast.md §3 |
| **Fig E2** (1 col) | chunk-skip A/B bars, 5 cells × 3 arms | `evidence/tileskip/{baseline,skipon,skipgated}.csv`, rowskip §4 |
| **Tab E2** (1 col) | small-k routing before/after (18 cells condensed) + cliff | `baselines/smallk_routefix/*.csv`, `baselines/scope51/canonical_sweep.csv` |
| **Tab E3** (1–2 col) | 8 model scenarios, today/routed/op + controls | `baselines/comp3/scenarios1_table.csv` |
| (shared) **Fig D2** | engine-shootout bars — lives in §3, listed there | commit-msg numbers per evidence Fig-4 row |

If space forces a cut: Tab E2 collapses into three prose numbers; Fig E2's
gated no-win cells move to text.

## Style directives

1. Tufte figure rules are binding: no gridlines, top/right spines off,
   0.5 pt axes, frame-free top legends, direct line-end labels, ≤8-word
   captions ("Runtime versus core count."). Consistent color per arm across
   ALL figures (baseline color in Fig E1 = same arm's color in Fig E2).
2. Log-scale y for the P-sweep and any plot spanning 183→23 µs; label the
   cost-model overlay directly on the curve, not in a legend.
3. Ranges + medians, Kapre-style: "632×–51,430× (24 cells, 3-trial
   medians)". Every table row keeps its core count — speedups without
   resource counts are banned.
4. Small deltas get the tool-noise treatment (Betz standard): the +0.04% to
   +0.51% gated cells are "within measurement spread" — say exactly that.
5. Captions self-contained for skimming readers, but ≤8 words — push
   configuration into body text; the caption states the takeaway noun
   phrase.

## Hazards

- The five speedup chains (prebranch→routed, prebranch→op, opstock→op,
  today→routed, today→op) must never be mixed in one sentence; name the
  chain every time.
- k2048@2048 row: op=15.74 µs @ 130 cores vs opstock 15.73 — a 1.00× cell;
  do not average it away; it shows the small-N floor honestly.
- opstock arm is a *proxy* (rows=2 row-parallel, per-row single-core wall
  time, byte-identical stock kernels) — footnote it as the ledger does.
- Derived-µs vs Tracy-µs caveat classes (6.1) — audit every caption at the
  end for the 1.35 GHz footnote.
- No `TOPK_LEDGER.html`, no commit hashes, no branch names anywhere;
  provenance is described structurally ("every row carries a build and
  source-tree fingerprint").
