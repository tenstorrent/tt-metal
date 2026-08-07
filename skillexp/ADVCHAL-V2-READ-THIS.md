# advchal-v2 — read this one file

Stage 02b (`$advisor-challenger`) ran `$shard-advise` on **15 decoder cells**. Each began from a decoder already
hand-optimised without the advisor, so anything it adds is a real gain rather than a re-derivation.

**Two cells gained over 10 %, six gained between 0.4 and 9 %, and seven gained nothing measurable.**

The zeros are mostly not the optimizer's:

- On **nine cell/kinds it was never shown the layer** — the capture tracer stopped at an op it had no handler for.
- **No cell applied its plan as written.** Where that could be measured afterwards, the plan was worth **twice**
  what the cell shipped, at bit-identical output.
- The metric that decides which candidates reach hardware prices its most reliable recommendation at `0.000 µs`.

All three are cheap. The tracer was fixed here to check that — **~230 lines of Python, no rebuild, and 17
screenable candidates appeared that the corpus had never counted.**

The optimizer's own defect is narrower: its objective carries no latency term, so the core grid it names is a good
guess rather than the best legal one. That is the gap between the **−10.43 %** its plan delivered on one cell and
the **−17.84 %** available on the same decoder.

**Three things are named separately throughout, because they fail differently and are fixed in different places:**
**the optimizer** is tt-mlir's `ttnn-to-ttnn-l1-advisor` pass pipeline, which decides placement; **`ttnn-advise`**
is the tool that captures a graph, runs that pipeline and writes the report; **the stage** is the skill that
screens the result on hardware.

Everything here comes from the cells' own artefacts or from re-measurements on the same hardware. `§3.x` points
into [`FINDINGS`](ADVCHAL-V2-FINDINGS.md); plain `§1`–`§5` are sections here. The 39 claims corrected along the
way are in [`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md), each with the check that would have caught it.

---

## 1. The results

| shipped gain | cells |
|---|---|
| **over 10 %** | **2** |
| 5–10 % | 2 |
| 1–5 % | 3 |
| under 1 % | 1 |
| inside its own noise band | 1 |
| **nothing at all** | **6** |

**6 of the 15 were probed afterwards for a better configuration, and 5 had one** — three of those beating what
shipped by more than 10 percentage points. Only llama-3.1-8B was swept and found to have nothing. The other nine
were never probed, so every corpus-level total below is a floor.

`control` lists every layer kind the cell measured, one per line; elsewhere a second line is a second *scope*, not
a second kind.

| model | arm | control ms/layer | shipped | result | better configuration found later |
|---|---|---|---|---|---|
| llama-3.2-1B | `exp17` | `dense` 0.3731 | nothing | 0.0 % | — *ladder never swept* |
| llama-3.1-8B | `exp17` | `dense` 0.6650 | nothing | 0.0 % | **none — swept, nothing beats the default** |
| phi-3.5-mini | `exp17` | `dense` 1.1009 | nothing | 0.0 % | — *advised sharding never tried* |
| phi-3.5-mini | `nofuse-noadvise-onA` | `dense` 0.6570 | `rope_l1_rect32` | **−8.75 %** | — *never probed* |
| phi-3.5-mini | `nofuse-noadvise` | `dense` 0.7888 | `rope_l1_chain` + sharded multiply/add | **−5.74 %** | — *never probed* |
| phi-3.5-mini | `fuse-noadvise` | `dense` 0.8072 | rope only, L1 interleaved | −4.91 % | **−17.84 %** — advised rope (−10.43 %, *bit-identical*) + advised 11-core norm |
| qwen3.6-27B | `fuse-noadvise` | `full_attention` 1.2083<br>`linear_attention` 19.1402 | `packed_qkv_l1_chain` | −445.7 µs/model<br>*inside its ±618.5 band* | — *`linear_attention`, 97 % of model time, never advised on* |
| qwen3.6-27B | `nofuse-noadvise` | `full_attention` 1.4494<br>`linear_attention` 15.8526 | nothing | 0.0 % | — *same gap; geometry hard-failed on the rest* |
| gemma-4-12B | `exp11` | `sliding_attention` 1.2541<br>`full_attention` 1.3774 | `Q+K+V+MLP` + output chain | **−1.14 %** | — *52 measurements, no advised grid among them* |
| gemma-4-26B | `nofuse-noadvise` | `sliding_attention` 1.2597<br>`full_attention` 1.2617 | `sliding_attention_o_chain` | −147.9 µs/model<br>−0.34 % sliding | **−12.44 %** sliding — a win it wrote, shipped disabled, never screened. **36×** |
| gemma-4-26B | `nofuse-noadvise-onA` | `sliding_attention` 1.8252<br>`full_attention` 2.0132 | `advisor_norm88` | **−12.98 %** sliding | **−13.63 %** — 44 cores beats the advised 88, bit-identically |
| gemma-4-26B | `fuse-noadvise` | `sliding_attention` 1.3412<br>`full_attention` 1.5394 | `advisor_concat_projection` | **−2.04 %**<br>−1.69 % sliding | −1.86 % sliding — *only 0.17 pp; the 88-core norm regressed here, correctly rejected* |
| north-mini | `fuse-noadvise` | `dense_full_attention` 0.1727<br>MoE kinds 0.5537 | MoE norm at 32 cores | **−10.23 %**<br>−10.37 % sliding MoE | **−11.28 %** — 16 cores beats both the advised 22 and the shipped 32 |
| north-mini | `nofuse-noadvise` | `dense_full_forced_rope` 0.2033<br>MoE kinds 0.6138 | nothing | 0.0 % | **11 candidates, 632 µs/model** — found only after the tracer was fixed |
| north-mini | `nofuse-noadvise-onA` | `dense_full_attention` 0.2918<br>sparse MoE kinds 0.8465 | nothing | 0.0 % | **2 candidates, 61.9 µs/model** — same |

*`result` is at the cell's own scope, model-level where the cell had one. The last column is per-layer on the kind
named in the same row unless stated. A blank means nobody looked, not that nothing is there. The two right-hand
columns are not subtractable — combined totals are in* [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) *§3.11. All 15 cells
ran on the same host, so an arm name says only how that cell's incumbent was produced.*

### The seven zeros

| cell | zero because | still blocked? |
|---|---|---|
| llama-3.1-8B `exp17` | nothing on its ladder beats the default, **and the whole ladder was swept** | no — a real zero, the only verified one |
| llama-3.2-1B `exp17` | its two candidates regressed; ladder never swept | no — unmeasured. Needs device time, not a fix |
| phi-3.5 `exp17` | everything overlapped its floor or hard-failed; the advised rope was never tried here | no — screening order (**F5**) |
| north-mini `nofuse-noadvise` | it screened honestly — on a layer three quarters of which was untraced | **unblocked**: 2 handlers → 11 candidates, **632 µs/model** |
| north-mini `nofuse-noadvise-onA` | sparse MoE untraceable | **unblocked**: 1 handler → `untraced` 77 % → 14 %, 2 candidates |
| qwen3.6-27B `nofuse-noadvise` | geometry hard-failed; 63.5 % of its dominant layer untraced | **unblocked** — the whole layer captures, 69 ops advised. Gain unmeasured: the cell kept no profile (STG-10) |
| qwen3.6-27B `fuse-noadvise` | its win is inside its own band, as the cell said | probably unblocked by the same handlers — **untested** on this arm |

**Three were never blocked. All four that were are now unblocked** — two re-measured, one unmeasurable for want of
a profile, one untested on its own arm. The two north-mini rows are the ones to learn from: both cells reasoned
correctly from what they could see, and nothing in the stage's own output distinguished that from a real zero.
Detail: [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md). Per-cell narratives:
[`MEASUREMENTS`](ADVCHAL-V2-MEASUREMENTS.md).

---

## 2. What v2 did, and what the optimizer is worth

### 2a. The stage

| question | answer |
|---|---|
| **Biggest single miss** | The **screening order**. It builds candidates up chain by chain and never applies the optimizer's plan as written. Where measurable: **−17.84 % available vs −4.88 % shipped — 3.7×** — and −10.43 % of it bit-identical. *(§3.27)* |
| **Did cells follow the advice?** | **7 of 15 tried its exact value on at least one item**, 6 as their first candidate. **4 never tried it and gave no reason. None applied the whole plan.** Of the 9 that changed anything, **3 shipped the advised sharding *and* grid.** *(§3.25)* |
| **Are the zeros failures?** | Mostly not the optimizer's — 1 of 7 verified, 4 were coverage gaps. §1. |
| **Whose defects?** | **10 stage** (one-file skill changes, no build), **4 `ttnn-advise`** (Python, one already fixed), **3 optimizer** (C++, needs a build), plus 1 unassignable. §3. |

The tooling is young and it shows, which is the good news: `ttnn-jit` is ten months old, and of the ten ops cells
recorded as capture blockers, **four already had handlers and one did not exist**.

### 2b. The optimizer

phi-3.5 `fuse-noadvise` is the only cell whose artefacts allow the whole plan to be reconstructed and run. Its own
harness, fresh process per configuration:

| configuration | median ms | Δ | differential PCC |
|---|---|---|---|
| incumbent (frozen control) | 0.807535 | — | — |
| what the cell shipped, after 14 measurements | 0.768104 | −4.88 % | 1.0 |
| **the optimizer's plan, implemented from its IR** | 0.723320 | **−10.43 %** | **1.0** |
| the cell's own best candidate, which it discarded | 0.700120 | −13.30 % | 0.99999107 |
| **that plan plus its advised 11-core norm** | **0.663507** | **−17.84 %** | 0.99999107 |

**−10.43 % against the −4.88 % that shipped, at PCC exactly 1.0.** Neither the correctness rule nor the optimizer
cost this cell the difference. Nobody tried the plan.

**What holds up**

- **Deterministic and cheap.** It discards the incoming memory configurations and re-places every op; same graph,
  same answer, in **18.4 / 18.4 / 18.1 / 18.6 s** — less than one harness measurement, so it can be re-run between
  changes.
- **Direction right in 4 of 4** cells on the dominant defect class, widening a starved reduction.
- **It declares what it cannot place**, with the exact runtime error — an output the stage discards *(§3.28)*.

**What does not.** `LayoutScore` has no latency term at any level and `getOpRuntime` is never consulted, so it
cannot rank by speed. Its grid reached **82 %** of achievable across the three swept ladders; a fixed *"closest to
16 cores"* rule with no compiler reached **99.4 %**. **3 of the 4 placement wins are at grids it did not name.**

**The bound.** Implementing everything it said was much better than what shipped, and still not optimal — 16 cores
beat its advised 22, and 44 beat its advised 88, both bit-identically. It is a starting configuration, not an
oracle: **apply the whole plan, then sweep the legal ladder around each op it named.** *(§3.14, §3.27)*

---

## 3. The ledger, in the order it should be fixed

Ten defects in the stage, four in `ttnn-advise`, three in the optimizer. The order below is the fix order: the
cheapest and largest first, and the one needing a compiler build last. *Bold codes are action points in*
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

### First: the stage's use of it — skill changes, no build

Here the optimizer's answers arrive correct and complete, and the stage mishandles them.

| id | defect | measured cost | fix |
|---|---|---|---|
| STG-1 | **Never applies the plan as a candidate** — screens chain by chain from the incumbent | **3.7×** where measurable; ≈1.43 ms/model on one cell *(§3.27)* | **F5** |
| STG-2 | **The ceiling prices only boundary conversions**, so an in-chain re-grid is worth `0.000 µs` | **60 % of the disagreed-on cost filed `below_threshold`, never measured.** Two of the three biggest wins came from cells whose ceiling said `0` | **D0** |
| STG-3 | **`advised_cores` parsed from the lossy `cores=` field** when the grid string sits beside it | 58.3 % of counts wrong; **34.4 % of the "disagreement" is phantom** | **C5f** |
| STG-4 | **`unfixable_ops` ignored** — the advisor names each unplaceable op with its exact `TT_FATAL` | **54 declarations, 41 screened anyway** | **C5g** |
| STG-5 | **The oracle rejects anything that moves PCC at all** | Discarded a −13.3 % candidate *more accurate than what shipped* *(§3.1)* | absolute oracle |
| STG-6 | **`agrees_with_shipped` never compares the memory space** | 1 of one cell's 2 such rows is wrong | **C5c** |
| STG-7 | **`pair_confidence: position` recorded as a guess, then ignored** | **23.2 % of pairings** are undiscounted guesses | **C5e** |
| STG-8 | **Never re-advises** — every candidate screened against one start-of-run capture | Re-advising costs **~18 s** | **F6** |
| STG-9 | **The capture monkey-patches `_decode_rope`** | The advice for that region is advice for a substitute method | stage |
| STG-10 | **Throws away the perf report it runs** — 1 of 15 cells kept a before/after pair | Op-level verification impossible for 14 cells, **and it is why qwen's gain cannot be measured today** | **B0** |

### Second: `ttnn-advise` and `ttnn-jit` — Python, no optimizer change

The optimizer's answers are fine; what reaches or leaves it is not.

| id | defect | consequence | status |
|---|---|---|---|
| TOOL-1 | **The tracer lacked five op handlers** and truncated the capture before the optimizer saw the layer | **The largest single limiter in the corpus** — 9 cell/kinds discarded 58–77 % of their window | **fixed** — [branch](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps), 2 commits |
| TOOL-2 | **`report.json` renders a multi-range `CoreRangeSet` as its first range only**, and no shard shape | Understates the optimizer's *own* advice — 58.3 % of counts. `final_ir.mlir` is intact | **C5f** |
| TOOL-3 | **No legal core-count ladder emitted**, though the set is computable in the pass | Challengers guess, and burn device time on illegal grids | **D4** |
| TOOL-4 | **`--help` offers `--tracer interception` as the fallback** — it cannot trace any HF-RoPE decoder | Wrong advice at the moment of failure; cannot be fixed by porting, so narrow the claim | doc |

### Third: the optimizer — C++, needs a build

Defects in how placement is **decided**. The only ones that make the advice worse.

| id | defect | consequence | fix |
|---|---|---|---|
| OPT-1 | **`LayoutScore` has no latency term at any level**; `getOpRuntime` is never consulted | It cannot rank candidates by speed. Root of the 82 %-vs-99.4 % gap | **D1** |
| OPT-2 | **`coreCount` is level 6 of 7**, and for norms `NormalizationRules.cpp:77-104` overrides it with the *input* grid volume — 1 on decode shapes — so it cannot vary with the candidate | Its grid values lose when measured | **D2** |
| OPT-3 | **Layouts are deduped by shard shape keeping the smallest grid, before per-op legality filters run** | A legal sibling can be dropped and an illegal one kept | **D3** |

**That is the whole list, and OPT-1 subsumes most of OPT-2.** One item cannot be assigned: llama's MLP norm chose
`1x22` while 32 and 64 were valid and outrank it on both of the optimizer's own tiebreakers. Either the score the
trace records is not the score compared — reporting — or the beam applies a criterion it does not record. **The
trace cannot distinguish these** → **D5**.

### How they interact

They are not ten plus four plus three independent tickets.

- **STG-3 and TOOL-2 are one defect from two sides.** `report.json` prints a lossy grid; the stage reads it. `C5f`
  fixes it on the consumer side in one line, so TOOL-2 becomes cosmetic — do not do both.
- **STG-2 and TOOL-3 belong together.** Pricing in-chain re-grids makes far more chains screenable, which is only
  affordable if the optimizer emits the legal ladder so the sweep skips illegal grids.
- **STG-10 gates the measurement of TOOL-1.** qwen's layer captures now, but no before/after profile exists, so
  the largest coverage win in the corpus has no number. Keeping the perf report is a precondition for quantifying
  the tracer fix, not an independent nicety.
- **TOOL-1 removes STG-9's justification.** The capture substituted a method because the tracer could not follow
  the real one; it can now.
- **OPT-1 shrinks STG-1's second half.** Applying the plan then sweeping the ladder is needed *because* the
  objective ignores latency. With a latency term the sweep gets shorter — it does not disappear.

**The ratio is the argument.** Three optimizer defects, one subsuming another. Four in the tooling, one already
fixed in an afternoon of Python. Ten in the stage, every one a one-file change with no build — and the cheap
columns are where the corpus lost: STG-1 is 3.7×, STG-2 hid two of the three biggest wins, TOOL-1 discarded most
of nine cell/kinds before the optimizer got a look. gemma-4-12B `exp11` is the extreme case: **52 measurements
without ever applying one advised grid.**

Graded in full: [`STAGE-ANALYSIS`](ADVCHAL-V2-STAGE-ANALYSIS.md),
[`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md).

---

## 4. What the fixes are worth, and what is beyond them

Three tiers, and they do not overlap.

**Tier 1 — already located and measured; recoverable by the stage fixes alone.** ≈**9.2 ms/model** across four
cells, all found from the advisor's own per-op output by a one-line static check needing no device time. This is
what §3's STG column buys; no tooling or compiler change is required to collect it. Half is the optimizer's actual
recommendation — in three of four it named the right op and a grid that a ladder sweep beat.

| cell | winning change | advised | µs/model | its own number? |
|---|---|---|---|---|
| **phi-3.5 `fuse-noadvise`** | rope as advised + 11-core norm | rope L1 + **11** | **−4,609** | **yes** |
| gemma-4-26B `nofuse-noadvise` | residual/norm at 22 cores, sliding only | 88 | −3,918 | no — 26× what it shipped |
| gemma-4-26B `nofuse-noadvise-onA` | norm at 44 instead of the shipped 88 | 88 | −375 | no — and bit-identical |
| north-mini `fuse-noadvise` | MoE norm at 16 instead of the shipped 32 | 22 | −264 | no |

**Tier 2 — newly visible, and not counted anywhere above.** Fixing TOOL-1 surfaced **17 candidates** on cells the
optimizer had never been allowed to see: **632 µs/model** on north-mini `nofuse-noadvise`, **141.6 µs/model** on
gemma-4-26B `nofuse-noadvise-onA` sliding, **61.9 µs/model** on north-mini `nofuse-noadvise-onA`. These sit on top
of tier 1, and they are themselves a floor — six of fifteen cells have been probed at all.

Still unmeasured here: **qwen `nofuse-noadvise`'s `linear_attention`**, which is 97 % of that model's decode time
and now captures in full at 69 advised ops. Nothing measures the gain because the cell kept no profile of its own
(STG-10). **A fresh profile of that one layer is the highest-value single measurement left in the corpus.**

**Tier 3 — outside a layout advisor entirely**, and larger than either tier above. `retilize` on qwen's conv chain
is **191 ms/model — 24.4 % of its decode time, 14× every shipped win combined** — and it is a graph-shape choice in
the decoder, a 4-element conv window on the 32-wide tile axis. The **sharded GQA SDPA output** kernel blocks two
cells' top candidate and a 2.6 ms/model wrong-op fix; that one is tt-metal's. Neither is a placement problem and
neither would be found by fixing anything in §3.

Itemised: [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §8. What to change:
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

---

## 5. Where everything lives

Δ with no unit qualifier is **per layer**; model-level figures say so. Where a quantity appears with slightly
different values it is a different scope or run, never a disagreement —
[`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §10 reconciles each.

| file | what's in it |
|---|---|
| [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) | **the detailed analysis — 29 findings and the method.** Where `§3.x` lives |
| [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md) | **every capture blocker, classed and where cheap fixed** — 4 already handled, 1 did not exist, 5 real; 9 cell/kinds re-captured |
| [`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md) | **39 corrected mistakes in 7 patterns**, each with the check that would have caught it, plus what is still unverified |
| [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) | what to change — ideas, then action points |
| [`CAPTURE-VARIANCE`](ADVCHAL-V2-CAPTURE-VARIANCE.md) | how the 15 cells captured — 54 to 290 lines for the same job — and what that cost |
| [`STAGE-ANALYSIS`](ADVCHAL-V2-STAGE-ANALYSIS.md) | the stage graded: what v2 fixed, 10 defects it kept |
| [`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) | why the optimizer advises what it does, from tt-mlir source + decision traces |
| [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md) | was it necessary? detection, grid choice, hit rate, and what 7.4 h bought |
| [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) | 10 stage settings changed one at a time, with a scoreboard |
| [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) | 8 experiments run on hardware to test the analysis |
| [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) | was the exact advice tried, and first? the chronology per cell |
| [`ADVICE-FOLLOWED`](ADVCHAL-V2-ADVICE-FOLLOWED.md) | how much was followed, all 15 cells — buffer type vs geometry |
| [`MEASUREMENTS`](ADVCHAL-V2-MEASUREMENTS.md) | all 149 harness measurements, per cell, in run order |
| [`ORACLES`](ADVCHAL-V2-ORACLES.md) | every cell's correctness bar, and why they are not comparable |
| [`PERF-REPORT-AUDIT`](ADVCHAL-V2-PERF-REPORT-AUDIT.md) | the perf report the stage runs and throws away |
| [`PER-OP`](ADVCHAL-V2-PER-OP.md), [`PER-CELL`](ADVCHAL-V2-PER-CELL.md), [`RESULTS`](ADVCHAL-V2-RESULTS.md) | every op placed differently; attribution per cell; the headline table |
| [`PHI-BEFORE-ADVISED-AFTER`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md), [`PHI-OP-BY-OP`](ADVCHAL-V2-PHI-OP-BY-OP.md) | one shipped win op by op — original / advised / shipped |
| `phi_*.txt`, `trace_ttnn.py`, `*.json`, `ttnn-jit-tracer-gap-handlers.patch` | raw terminal output, ttnn call sequences, machine-readable data, the tracer fix |

Everything is reconstructed from the cells' own session transcripts and artefacts, not from their self-reported
summaries.
