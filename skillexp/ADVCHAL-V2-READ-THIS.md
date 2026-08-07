# advchal-v2 — read this one file

Stage 02b (`$advisor-challenger`) ran `$shard-advise` on **15 decoder cells** to measure how much it can
contribute to decode performance. Each cell started from a decoder already optimised without the advisor, so
anything the advisor adds is a real gain and not a re-derivation of work already done.

The accounting is strict: the incoming decoder is frozen as the control, never re-tuned, and only what the
advisor's directions add on top is counted. **That understates what the advisor contributed**, in three ways:

- The metric that picks which candidates get measured values the kind of change that produced about half the
  corpus's gains at **zero** — so those candidates were often never tried. *(§3.6)*
- The one direction the advisor reliably gets right is recorded as a win **0 times out of 37**, including in the
  two cells that shipped exactly that change. *(§3.14)*
- **No cell ever applied the advisor's plan as written.** Where that comparison is possible, doing so is worth
  **3.7×** what the cell shipped. *(§3.27)*

So §2 gives two numbers per cell: what the stage credited, and what was actually reachable.

Everything here comes from the cells' own artefacts or from re-measurements on the same hardware.

This is the short version. The detailed analysis is in [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) — a reference like
`§3.11` points there, plain `§1`–`§6` are sections here.

Claims that were corrected along the way are kept in
[`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md), each with the check that would have caught it.

---

## 1. The verdict

**§1a is the stage**, measured directly across 15 cells. **§1b is the advisor**, inferred from how it behaved
inside that stage — fifteen independent decoders is a fair test of a tool, though not a study of one. The two
answers come out differently.

### 1a. What v2 delivered

| question | answer |
|---|---|
| **How much did it ship?** | **13,601 µs/model** across 8 of 15 cells, where **21,368 µs** was reachable from the advisor's own directions on the same decoders — **64 %**. |
| **What is the single biggest miss?** | The **screening order**. It builds candidates up chain by chain and never applies the advisor's plan as written. On the one cell where the counterfactual is measurable, applying it gives **−17.84 % vs the −4.88 % that shipped — 3.7×**, and −10.43 % of that is **bit-identical** to the incumbent. It was never tried. |
| **Did it follow the advice?** | **7 of 15 cells tried the advisor's exact value on at least one advised item** — 6 as their first candidate — and all 7 ended cleanly, shipping it or measuring a regression against it. **4 never tried it and recorded no reason. None applied the whole advised plan.** Of the 9 cells that changed anything, **3 shipped the advised sharding *and* grid**. |
| **Are the zeros failures?** | Mostly not the advisor's. Only **1 of 7 is verified by measurement** (llama-3.1-8B, whose whole ladder was swept). **3 are coverage gaps** — the tracer cannot reach those layers, so the advisor never saw them and this stage produced no verdict either way. The rest are asserted or partial — graded in §2 below. |
| **Whose defects?** | **10 stage defects**, almost all one-file changes with no build, and they account for the larger measured loss. **6 advisor defects**, all needing tt-mlir builds. The ledger is §3. |

### 1b. Is `ttnn-advise` a promising thing to build a stage on?

**Cautiously yes — as a detector and a starting configuration, not yet as a grid chooser.** The evidence on
each side:

**What supports it**

- Its **direction** on the dominant defect class — widen a starved reduction — was right in **4 of 4** cells
  where anyone measured it. *(§3.2, §3.14)*
- Its **exact plan, applied verbatim, contained more than the stage extracted**: **−10.43 % at PCC 1.0** where the
  cell shipped −4.88 %. *(§3.27 — one cell, the only one with the artefacts to test it end to end)*
- It adds **precision** to detection: adding "and the advisor wants more cores" narrows a 7-cell flag list to 5
  with the same recall. *(§3.14)*
- It **declares what it cannot place**, with the exact runtime error — an output the stage discards. *(§3.28)*
- **Deterministic**, and **~18 s** to run end to end. *(§3.29)*

**What limits it**

- **No latency term anywhere in its objective.** Its grid choice scored **82 %** of achievable across the three
  ladders that were swept; a fixed *"closest to 16 cores"* heuristic with no advisor at all scored **99.4 %**.
  This is the strongest evidence against it. *(§3.3, §3.14)*
- **3 of the 4 placement wins are at grids it did not name** — it identified the op, not the value. *(§4)*
- A detection rule using only the shipped profile, **no advisor**, catches all 4 win cells. It buys precision,
  **not recall**. *(§3.14)*
- Its per-op hit rate over the 118 measured rows is **49 %** — though that population is dominated by boundary
  candidates and structurally excludes the direction it gets right, so it understates the advisor. *(§3.14)*
- **Coverage, not placement, decided more outcomes.** Tracer gaps put roughly half the corpus's op cost outside
  it — **≈62 % of one model's decode time** sits in ops its tracer cannot capture. *(§3.5)*
- The corpus's **largest single cost**, 191 ms/model, is a graph-shape choice **no layout advisor could reach**.
  *(§3.18–3.19)*

What limited the advisor in v2 was not the advisor. It was the stage's use of it — ten cheap defects — plus
tracer coverage, and both are more tractable than a cost model. The 82 %-vs-99.4 % gap is what to watch: **until `LayoutScore` prices latency, trust the advisor for *where* to look and *which
direction* to move, and treat its specific core count as one rung on a ladder to sweep.**

Summarised: a **defect detector with a broken cost model** — no latency term anywhere in its objective — used by
a stage that never tested half of what it found.

---

## 2. The 15 cells, and what each shipped

`FN` = `fuse-noadvise`, `B` = `nofuse-noadvise`, `onA` = `nofuse-noadvise-onA`. The `-onA` suffix says where
that cell's *incumbent* came from — **all 15 cells ran on the same host**, so no difference below is a
hardware difference.

The last column is **what was measured to be reachable from the advisor's own directions on that same decoder**:
how much better each cell could have been had the advice been followed more closely. It is per-layer on the named
kind. **A blank is an absence of measurement, not a measured zero.**

| model | cell | control ms/layer | what v2 shipped | v2 result | **reachable (measured)** |
|---|---|---|---|---|---|
| llama-3.2-1B | exp17 | 0.3731 | nothing | 0.0 % | — *ladder never swept* |
| llama-3.1-8B | exp17 | 0.6650 | nothing | 0.0 % | **0.0 % — tested**, full ladder swept, nothing beat the default |
| phi-3.5-mini | **onA** | 0.6570 | `rope_l1_rect32` | **−8.75 %** | — **not tested** |
| phi-3.5-mini | **B** | 0.7888 | `rope_l1_chain`, sharded multiply/add | **−5.74 %** | — **not tested**, but see the note below |
| phi-3.5-mini | **FN** | 0.8072 | rope only, L1 interleaved | −4.91 % | **−17.84 % / layer** — the advised rope (−10.43 %, *bit-identical*) plus the advised 11-core norm |
| phi-3.5-mini | exp17 | 1.1009 | nothing | 0.0 % | — *advised sharding never tried* |
| qwen3.6-27B | **FN** | 1.2083 full / 19.14 linear | `packed_qkv_l1_chain` | −445.7 µs — inside its ±618.5 µs band | — *its `linear` kind, 97 % of model time, was never advised on* |
| qwen3.6-27B | **B** | 1.4494 full / 15.85 linear | nothing | 0.0 % | — *same; geometry hard-failed on the rest* |
| gemma-4-12B | exp11 | 1.2541 / 1.3774 | `Q+K+V+MLP` + output chain | **−1.14 %** | — *52 measurements, no advised grid among them* |
| gemma-4-26B | **B** | 1.2597 | `sliding_attention_o_chain` | −147.9 µs *(−0.34 %/layer sliding)* | **−12.44 % / layer** — a win it wrote, shipped disabled, never screened. **36×** |
| gemma-4-26B | **onA** | 1.8252 | `advisor_norm88` | **−12.98 %** | **−13.63 % / layer** — 44 cores beats the advised 88, bit-identically |
| gemma-4-26B | **FN** | 1.3412 / 1.5394 | `advisor_concat_projection` | **−2.04 %** *(−1.69 %/layer sliding)* | −1.86 % / layer sliding — *only 0.17 pp; the 88-core norm regressed here and was correctly rejected* |
| north-mini | **FN** | 0.5537 MoE | MoE norm at 32 cores | **−10.23 %** *(−10.37 %/layer sliding MoE)* | **−11.28 % / layer** — 16 cores beats both the advised 22 and the shipped 32 |
| north-mini | **B** | 0.6138 / 0.2033 | nothing | 0.0 % | — *all measured geometries slower or stalled* |
| north-mini | **onA** | 0.2918 / 0.8465 | nothing | 0.0 % | — *sparse MoE untraceable; headroom unknown* |

**8 shipped, 7 returned zero.**

*`v2 result` is at the cell's own scope — model-level % where it had one, µs where it did not. `reachable` is
per-layer on the named kind. The two are not directly subtractable; the corpus totals that combine them are in §4
and* [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) *§3.11.*

⚠ **The reachable column covers 6 of 15 cells.** Five have a measured better configuration; llama-3.1-8B was
swept and has nothing further. The other nine were never probed. **So the corpus reachable total — and the 64 %
credited figure derived from it — are lower bounds.**

**The cheapest place to look next is phi B**, on an inference rather than a measurement: it shipped the *sharded
multiply/add* form of the rope chain, which on phi FN's decoder is worth −6.97 % against −10.43 % for the fully
advised form. If that ordering holds on phi B's own incumbent — untested — phi B has headroom too.

### How solid are the zeros?

A zero can mean three different things — no headroom, no measurement, or no way to look — and they carry very
different weight. Only one of the seven is verified by measurement:

| cell | zero because | how well established |
|---|---|---|
| **llama-3.1-8B** | nothing on its ladder beats the default | **Verified.** The whole achievable ladder was swept — {8, 16, 32, 64}, the only counts its knob can express. 16 (≈ the advised 22) and 32 are both **inside the noise floor**; 64 is +3.78 %, 8 is +11.21 %. The advised 22 is not expressible at all; the knob rounds it to 16. There is no hidden norm win |
| llama-3.2-1B | its two candidates regressed | **Asserted, not verified.** Its ladder was never swept. The norm arrived well placed and the advisor wanted *fewer* cores — a direction that wins about half the time — so there is no particular reason to expect a win here, and no measurement either way |
| phi-3.5 exp17 | every direction overlapped its floor or hard-failed | **Partly.** Its floor is the corpus's second-worst (1.092 µs) and the advised rope sharding — the thing worth −10.43 % on phi FN — was **never tried here** |
| qwen3.6-27B B | the geometry hard-failed | **Partly a coverage gap.** Its `linear_attention` kind is 97 % of model decode time, and the trace stops inside it — **63.5 % of that layer is `untraced`**, so ≈62 % of the model was never advised on. The advisor did see the other third. Reading the profile directly finds the corpus's largest single cost there, ~191 ms/model of `retilize` (§3.18) — not a placement defect |
| north-mini onA | sparse MoE untraceable | **Not a placement verdict.** `ttnn.sparse_matmul` rejects tracer tensors, so the advisor never saw the MoE tail. Its placement headroom is **unmeasured**, and no one has read that profile directly either |
| north-mini B | all measured geometries slower or stalled | **Reasonable.** It did screen, and its `advisor_dense_chain_exact` candidate regressed by 15 % |
| qwen3.6-27B FN | its win is inside its own band | **Honest about itself** — the cell said so. But the same coverage gap applies |

**1 verified, 1 asserted, 1 reasonable, 1 partial, 3 coverage gaps.** "7 of 15 returned zero" is true, but it
does not mean seven decoders had no placement headroom. For three of them the advisor never saw the layer, so
the zero records a tracer limitation rather than a measurement.

Per-cell narratives and every cell measurement: [`MEASUREMENTS`](ADVCHAL-V2-MEASUREMENTS.md).

---

## 3. The ledger: advisor not good enough, vs stage not listening

Every defect found, sorted by **whose it is**. The split matters because the two groups go to different
codebases and cost very different amounts to fix.

*`ADV-n` / `STG-n` are IDs local to this table. The bold codes in the last column (`D1`, `C5f`, `F5`…) are
action points in [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).*

### The advisor is genuinely not good enough — tt-mlir changes

| id | defect | consequence |
|---|---|---|
| ADV-1 | **`LayoutScore` has no latency term at any level.** `getOpRuntime` exists in `TTNNOpModel.cpp` and is never consulted. | It cannot rank by speed at all. Everything below follows from this. → **D1** |
| ADV-2 | **`coreCount` is level 6 of 7**, and for norms `NormalizationRules.cpp:77-104` overrides it with the *input* operand's grid volume — exactly 1 on decode shapes — so the term **cannot vary with the candidate**. | Its grid values lose when measured: **3 of the 4 placement wins are at grids it did not name** (§4 below) |
| ADV-3 | **Candidate layouts are deduped by shard shape keeping the *smallest* grid, before per-op legality filters run.** | A legal sibling can be discarded and an illegal representative kept → **D3** |
| ADV-4 | **One selection the trace cannot explain**: llama's MLP norm chose `1x22` while 32 and 64 were valid and outrank it on both documented tiebreakers. | Either the recorded score is not what is compared, or there is an unrecorded criterion → **D5** |
| ADV-5 | **`report.json` renders a multi-range `CoreRangeSet` as its first range only**, and prints no shard shape at all. | The summary understates its own advice. *Shared blame*: the information survives intact in `final_ir.mlir`, so nothing is lost by the optimizer — only by its summary |
| ADV-6 | **It does not emit the legal ladder**, so a challenger has to guess which core counts are even legal. | Wasted device time on illegal geometries → **D4** |

### The stage is not listening to it — skill changes, all cheap

| id | defect | measured cost | fix |
|---|---|---|---|
| STG-1 | **It never applies the advisor's plan as a candidate.** Screens chain by chain, building up from the incumbent. | **3.7× on the one cell where the counterfactual is measurable** — −17.84 % available vs −4.88 % shipped, ≈1.43 ms/model on phi FN alone (§3.27) | **F5** — apply_all first, then ablate |
| STG-2 | **The screening ceiling prices only boundary conversions**, so an in-chain re-grid is worth `0.000 µs`. | **60 % of the disagreed-on cost filed `below_threshold` and never measured.** Two of the three biggest wins in the corpus came from cells whose ceiling said `0` / `not_measurable` and which recorded **0 kept chains** | **D0** |
| STG-3 | **`advised_cores` is parsed from the lossy `cores=` field** when the correct grid string sits beside it. | **58.3 % of advised core counts wrong; 34.4 % of the "disagreement" is phantom.** Two phi cells recorded themselves as *overriding* the advisor while agreeing with it | **C5f** — one line |
| STG-4 | **`unfixable_ops` is ignored.** The advisor names each unplaceable op with the exact runtime `TT_FATAL`; `reconcile.py` reads the field only to annotate the `untraced` bucket. | **54 declarations, 41 screened anyway.** Cells burn measurements rediscovering errors handed to them in writing | **C5g** |
| STG-5 | **The oracle rule rejects anything that moves PCC at all**, implemented as a differential bar ≈1.0. | Discarded a −13.3 % candidate at PCC 0.9999911 that is *more accurate than what shipped* (§3.1). And note §3.27: **−10.43 % was available at PCC exactly 1.0**, so even the strict rule permitted more than shipped | stage: absolute oracle at the model's own bar |
| STG-6 | **`agrees_with_shipped` never compares the memory space** — core count or DS-family only. | 1 of phi FN's 2 such rows is wrong | **C5c** |
| STG-7 | **`pair_confidence: position` is recorded, documented as a guess, then ignored downstream.** | **23.2 % of pairings corpus-wide** are guesses that nothing discounts — enough to turn positional artefacts into apparent findings | **C5e** |
| STG-8 | **It never re-advises**, screening every candidate against one start-of-run capture. | `ttnn-advise` costs **~18 s** — less than a single harness measurement (§3.29) | **F6** |
| STG-9 | **The capture monkey-patches `_decode_rope`**, so the advisor never sees the cell's real RoPE. | The advice for that region is advice for a substitute method | stage/capture, or fix the tracer limitation |
| STG-10 | **It throws away the perf report it runs.** Only 1 of 15 cells saved a before/after profile pair. | Op-level verification is impossible for 14 cells | **B0** |

The two columns are not the same size of problem. The advisor's defects are real, but each needs a tt-mlir
build. **The stage's are almost all one-file changes with no build, and they account for the larger measured
loss** — STG-1 alone is 3.7× on the cell where it can be measured, and STG-2 hid two of the corpus's three
biggest wins. gemma-4-12B is the extreme case: **52 measurements without ever applying one advised grid.**

### Neither — outside any layout advisor's reach

The corpus's **largest** numbers are in neither column: `retilize` at 191 ms/model, the ≈62 % of qwen that its tracer cannot capture,
`sparse_matmul` coverage, and the sharded-GQA kernel gap. They are itemised in §4, and they belong to the decoder
and to tt-metal — a layout advisor could not reach any of them.

---

## 4. What is still on the table

Placement leaves **≈9.2 ms/model** on the table across four cells. All four were located from the advisor's
*own* per-op output, by a one-line static check needing no device time — but only **half** the total is the
advisor's actual recommendation. In three of the four it named the right op and a grid that a sweep of the legal
ladder beat.

| cell | winning change | advised value | µs/model | the advisor's own number? |
|---|---|---|---|---|
| **phi-3.5 FN** | rope as advised + 11-core norm | rope L1 + **11** | **−4,609** | **yes** |
| gemma-4-26B B | residual/norm at 22 cores, sliding only | 88 | −3,918 | no — 26× what it shipped |
| gemma-4-26B onA | norm at 44 instead of the 88 it shipped | 88 | −375 | no — and bit-identical |
| north-mini FN | MoE norm at 16 instead of the 32 it shipped | 22 | −264 | no |

**From outside the advisor — much bigger, and correctly outside its reach:**

| opportunity | scale | whose |
|---|---|---|
| **`retilize` on qwen's conv chain** | **191 ms/model — 24.4 % of its decode time**, 14× every shipped win combined | the decoder's shape choice — a 4-element conv window on the 32-wide tile axis |
| qwen's untraced token mixer | the kind is **97 %** of decode time; **≈62 %** of the model is in ops the tracer cannot capture | tt-metal tracer coverage |
| `ttnn.sparse_matmul` tracer support | unblocks a whole cell; 58–65 % of every gemma-4-26B window | tt-metal |
| sharded GQA SDPA output | blocks two cells' top candidate *and* a 2.6 ms/model wrong-op fix | tt-metal kernel |

Itemised with evidence: [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §8. What to change:
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

---

## 5. Reading the numbers

A few quantities appear with slightly different values in different sections. In every case that is a different
**scope** or a different **run**, not a disagreement:

| looks inconsistent | why |
|---|---|
| phi FN shipped: **−4.91 % / −4.90 % / −4.88 %** | three runs of one configuration — the cell's own `final.json`, a recomputation from its block means, and a fresh re-measurement (§3.27). Run-to-run drift ≈0.1 pp |
| phi FN's discarded candidate: **−13.39 % / −13.4 % / −13.30 %** | same, three runs |
| north-mini FN shipped: **−9.26 % / −10.23 % / −10.37 %** | three **scopes** — its `final.json` (whose `incumbent_ms` is a 27.635 ms multi-layer harness), its own `model_estimate` (24,949 → 22,398 µs), and per-layer on sliding MoE |
| Δ with no unit qualifier | **per layer**. Model-level figures always say so |

Assumptions are labelled as assumptions. Claims that were later refuted are kept in
[`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md) rather than deleted, along with the check that would have
caught each one.

---

## 6. Where everything lives

| file | what's in it |
|---|---|
| **this file** | the verdict, the cells, the ledger — a few minutes |
| [`ADVCHAL-V2-FINDINGS.md`](ADVCHAL-V2-FINDINGS.md) | **the detailed analysis: 29 findings and the method** — this is where `§3.x` lives |
| [`ADVCHAL-V2-CAPTURE-VARIANCE.md`](ADVCHAL-V2-CAPTURE-VARIANCE.md) | **how the 15 cells captured** — 54 to 290 lines for the same job; 4 substitute model methods, 6 replay it instead of calling it — and what that does and does not cost |
| [`ADVCHAL-V2-ANALYST-PITFALLS.md`](ADVCHAL-V2-ANALYST-PITFALLS.md) | **30 corrected mistakes, in 7 patterns**, each with the check that would have caught it — plus what remains unverified |
| [`ADVCHAL-V2-IMPROVEMENTS.md`](ADVCHAL-V2-IMPROVEMENTS.md) | what to change — ideas, then action points |
| [`ADVCHAL-V2-EXPERIMENTS.md`](ADVCHAL-V2-EXPERIMENTS.md) | 8 experiments run on hardware to test the analysis |
| [`ADVCHAL-V2-COUNTERFACTUALS.md`](ADVCHAL-V2-COUNTERFACTUALS.md) | **10 stage settings changed one at a time** — what each would have found, with a scoreboard |
| [`ADVCHAL-V2-ADVISOR-VALUE.md`](ADVCHAL-V2-ADVISOR-VALUE.md) | **was the advisor necessary?** — detection, grid choice, hit rate, and what 7.4 h bought |
| [`ADVCHAL-V2-PERF-REPORT-AUDIT.md`](ADVCHAL-V2-PERF-REPORT-AUDIT.md) | **the perf report the stage runs and throws away** — compute-vs-movement scorecard per cell |
| [`ADVCHAL-V2-ADVICE-FAITHFULNESS.md`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) | **was the exact advice tried, and first?** — the chronology per cell, E25, and the retraction |
| [`ADVCHAL-V2-ADVICE-FOLLOWED.md`](ADVCHAL-V2-ADVICE-FOLLOWED.md) | **how much of the advice was followed, all 15 cells** — per chain verdict, and buffer type vs geometry |
| [`ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md) | **one shipped win, op by op** — original / advised / shipped, with shapes and the stage's own labels |
| [`ADVCHAL-V2-PHI-OP-BY-OP.md`](ADVCHAL-V2-PHI-OP-BY-OP.md) | the same win as a before/after delta, plus the sharding view |
| `phi_TERMINAL_BEFORE.txt`, `phi_TERMINAL_AFTER.txt` | real `tt-perf-report` terminal output, both sides |
| `phi_BEFORE_rope_off.txt`, `phi_AFTER_rope_on.txt`, `trace_ttnn.py` | executed ttnn call sequences + the tracer |
| [`ADVCHAL-V2-STAGE-ANALYSIS.md`](ADVCHAL-V2-STAGE-ANALYSIS.md) | the stage graded: what v2 fixed, 10 defects it kept |
| [`ADVCHAL-V2-ADVISOR-INTERNALS.md`](ADVCHAL-V2-ADVISOR-INTERNALS.md) | why the advisor advises what it does, from tt-mlir source + decision traces |
| [`ADVCHAL-V2-ORACLES.md`](ADVCHAL-V2-ORACLES.md) | every cell's correctness bar, and why they aren't comparable |
| [`ADVCHAL-V2-MEASUREMENTS.md`](ADVCHAL-V2-MEASUREMENTS.md) | all 149 harness measurements, per cell, in run order |
| [`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md) | every op the advisor placed differently |
| [`ADVCHAL-V2-PER-CELL.md`](ADVCHAL-V2-PER-CELL.md) | attribution accounting per cell |
| [`ADVCHAL-V2-RESULTS.md`](ADVCHAL-V2-RESULTS.md) | the headline table |
| `advchal-v2-narrative.json`, `advchal-v2-data.json` | machine-readable |

Everything here is reconstructed from the cells' own session transcripts and artefacts, not from their
self-reported summaries.
