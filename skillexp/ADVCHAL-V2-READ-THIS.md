# advchal-v2 — read this one file

**What was measured.** Stage 02b (`$advisor-challenger`) ran on **15 decoder cells** to answer one question:
how much of a decoder's speed can the shard advisor be credited with, on a decoder already optimised without
it? Every number below is from the cells' own artefacts or from my re-measurements on the same hardware.

**This file is the few-minute version.** The evidence — 29 findings and the method — is in
[`FINDINGS`](ADVCHAL-V2-FINDINGS.md); **every `§` reference below points there**, not into this file. What the
analysis itself got wrong is in [`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md).

---

## 1. The verdict

| question | answer |
|---|---|
| **Is the advisor worth running?** | Yes, but narrowly. Its value is the size of the *placement defect it happens to find* — and that is essentially **one** defect class, a reduction stuck on too few cores. Worth **6–13 %/layer** where it exists; **7 of 15 cells honestly returned zero**. |
| **How well did the stage use it?** | It shipped **13,601 µs/model** of saving where **21,368 µs** was reachable from the advisor's own directions — **64 %**. |
| **What is the single biggest miss?** | The **screening order**. The stage builds candidates up chain by chain and never applies the advisor's plan as written. On the one cell where the counterfactual is measurable, applying it gives **−17.84 % vs the −4.88 % that shipped — 3.7×**, and −10.43 % of that is **bit-identical** to the incumbent. It was never tried. |
| **Was the advice even followed?** | **7 of 15 cells tried the exact advice**, 6 of them first — and all 7 ended cleanly. **4 never tried it and recorded no reason.** Of the 9 cells that changed anything, **3 shipped the advised sharding *and* grid**. |
| **How much is left on the table?** | **≈9.2 ms/model** from placement — and **≈191 ms/model** from one thing no layout advisor can see. |
| **Is the advisor or the stage the problem?** | Both, unequally. **6 advisor defects** need tt-mlir builds; **10 stage defects** are one-file changes and account for the larger measured loss. See §3. |

**The one-sentence version:** the advisor is a **defect detector with a broken cost model** — no latency term
anywhere in its objective — and the stage listens to it selectively enough that half of what it does find is
never tested.

---

## 2. The 15 cells, and what each shipped

`FN` = `fuse-noadvise`, `B` = `nofuse-noadvise`, `onA` = `nofuse-noadvise-onA`. The `-onA` suffix says where
that cell's *incumbent* came from — **all 15 cells ran on the same host**, so no difference below is a
hardware difference.

| model | cell | control ms/layer | what shipped | model-level |
|---|---|---|---|---|
| llama-3.2-1B | exp17 | 0.3731 | nothing | **0.0 %** — honest zero |
| llama-3.1-8B | exp17 | 0.6650 | nothing | **0.0 %** — honest zero, re-verified |
| phi-3.5-mini | **onA** | 0.6570 | `rope_l1_rect32` | **−8.75 %** |
| phi-3.5-mini | **B** | 0.7888 | `rope_l1_chain` | **−5.74 %** |
| phi-3.5-mini | **FN** | 0.8072 | rope only | −4.91 % — **−13.4 % measured and discarded; −17.84 % was reachable (§3.27)** |
| phi-3.5-mini | exp17 | 1.1009 | nothing | 0.0 % — every direction overlapped or hard-failed |
| qwen3.6-27B | **FN** | 1.2083 full / 19.14 linear | `packed_qkv_l1_chain` | −445.7 µs — **inside its ±618.5 µs band** |
| qwen3.6-27B | **B** | 1.4494 full / 15.85 linear | nothing | 0.0 % — geometry hard-failed |
| gemma-4-12B | exp11 | 1.2541 / 1.3774 | `Q+K+V+MLP` + output chain | **−1.14 %** |
| gemma-4-26B | **B** | 1.2597 | `sliding_attention_o_chain` | **−147.9 µs** |
| gemma-4-26B | **onA** | 1.8252 | `advisor_norm88` | **−12.98 %/layer** |
| gemma-4-26B | **FN** | 1.3412 / 1.5394 | `advisor_concat_projection` | **−2.04 %** — *88-core norm regressed here* |
| north-mini | **FN** | 0.5537 MoE | MoE norm at 32 cores | **−10.23 %** — **16 cores is 1 pp better** |
| north-mini | **B** | 0.6138 / 0.2033 | nothing | 0.0 % — all geometries slower or stalled |
| north-mini | **onA** | 0.2918 / 0.8465 | nothing | 0.0 % — sparse MoE untraceable |

8 shipped, 7 returned zero. Of the zeros: **2 honest** (already well-placed), **3 structural** (the advisor
could not see the layer), **1 geometry wall**, **1 unseparable from its own noise floor**.

Per-cell narratives and every measurement: [`MEASUREMENTS`](ADVCHAL-V2-MEASUREMENTS.md).

---

## 3. The ledger: advisor not good enough, vs stage not listening

The single most useful reframing to come out of this corpus. Every defect found, sorted by **whose it is** —
because the fixes go to different places and cost wildly different amounts.

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
| STG-7 | **`pair_confidence: position` is recorded, documented as a guess, then ignored downstream.** | **23.2 % of pairings corpus-wide**; it misled this analysis into reading 7 guesses as findings | **C5e** |
| STG-8 | **It never re-advises**, screening every candidate against one start-of-run capture. | `ttnn-advise` costs **~18 s** — less than a single harness measurement (§3.29) | **F6** |
| STG-9 | **The capture monkey-patches `_decode_rope`**, so the advisor never sees the cell's real RoPE. | The advice for that region is advice for a substitute method | stage/capture, or fix the tracer limitation |
| STG-10 | **It throws away the perf report it runs.** Only 1 of 15 cells saved a before/after profile pair. | Op-level verification is impossible for 14 cells | **B0** |

**The asymmetry is the point.** The advisor's defects are real but need tt-mlir builds. **The stage's defects are
almost all one-file, no-build changes, and they account for the larger measured loss** — STG-1 alone is 3.7× on the
cell where it can be measured, and STG-2 hid two of the corpus's three biggest wins. gemma-4-12B is the extreme
case: **52 measurements without ever applying one advised grid.**

### Neither — outside any layout advisor's reach

The corpus's **largest** numbers are in neither column: `retilize` at 191 ms/model, qwen's untraced 97 %,
`sparse_matmul` coverage, and the sharded-GQA kernel gap. They are listed under *Still on the table* below, and
recorded there so nobody files them against the advisor.

---

## 4. What is still on the table

**From placement — ≈9.2 ms/model, across four cells.** All four were located from the advisor's *own* per-op
output by a one-line static check needing no device time. But only **half** of the total is the advisor's actual
recommendation; in three of the four it named the right op and a grid that a sweep of the legal ladder beat:

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
| qwen's untraced linear attention | **97 %** of its decode time never advised on at all | tt-metal tracer coverage |
| `ttnn.sparse_matmul` tracer support | unblocks a whole cell; 58–65 % of every gemma-4-26B window | tt-metal |
| sharded GQA SDPA output | blocks two cells' top candidate *and* a 2.6 ms/model wrong-op fix | tt-metal kernel |

Itemised with evidence: [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §8. What to change:
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

---

## 5. Reading the numbers

The same quantity sometimes appears with slightly different values in different sections. That is **scope or
run**, not disagreement, and it is worth knowing which is which:

| looks inconsistent | why |
|---|---|
| phi FN shipped: **−4.91 % / −4.90 % / −4.88 %** | three runs of one configuration — the cell's own `final.json`; my recomputation from its block means; my fresh re-measurement (§3.27). Run-to-run drift ≈0.1 pp |
| phi FN's discarded candidate: **−13.39 % / −13.4 % / −13.30 %** | same, three runs |
| north-mini FN shipped: **−9.26 % / −10.23 % / −10.37 %** | three **scopes** — its `final.json` (whose `incumbent_ms` is a 27.635 ms multi-layer harness), its own `model_estimate` (24,949 → 22,398 µs), and per-layer on sliding MoE |
| Δ with no unit qualifier | **per layer**. Model-level figures always say so |

Where a claim is an assumption rather than a measurement it is labelled. Where a claim of mine was refuted,
[`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md) records it rather than deleting it.

---

## 6. Where everything lives

| file | what's in it |
|---|---|
| **this file** | the verdict, the cells, the ledger — a few minutes |
| [`ADVCHAL-V2-FINDINGS.md`](ADVCHAL-V2-FINDINGS.md) | **the evidence: 29 findings and the method** — this is where `§3.x` lives |
| [`ADVCHAL-V2-ANALYST-PITFALLS.md`](ADVCHAL-V2-ANALYST-PITFALLS.md) | **what this analysis got wrong** — 30 corrections in 7 error patterns, plus what is still unverified. Read it before doing the next analysis |
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

Everything is reconstructed from the cells' own session transcripts and artefacts, not their self-reported
summaries. Facts are sourced; where something is my inference it says so, and where a claim of mine was later
refuted it is recorded in [`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md) rather than quietly deleted.
