# advchal-v2 — read this one file

Stage 02b (`$advisor-challenger`) ran `$shard-advise` on **15 decoder cells** to find out how much a compiler can
contribute to hand-tuned decode performance. Every cell started from a decoder already optimised *without* the
advisor, so anything it adds is a real gain rather than a re-derivation.

Two things are being judged at once, and they came out very differently.

**The tt-mlir optimizer** underneath `ttnn-advise` is the mature part: deterministic, ~18 s end to end, and on the
one cell where its whole plan could be implemented and measured, that plan was worth **2.1× what the stage
shipped** — at bit-identical output. **Everything around it is young and was the binding constraint.** The
capture tracer truncated eight cell/kinds, discarding 58–77 % of the profiled window before the optimizer saw it.
The screening harness never once tried the optimizer's plan as written. Both are cheap to fix, and this analysis
fixed the tracer side to prove it: 218 lines of Python, no rebuild, and 17 candidates appeared that the corpus had
never counted.

So the headline is not *"the compiler underperforms hand tuning."* It is that **v2 never gave the compiler a fair
run**, and where it accidentally did, the compiler won.

The accounting below is deliberately strict — the incoming decoder is frozen as the control, never re-tuned, and
only what the advisor's directions add on top is counted. That **understates** the advisor four ways:

- The metric that decides which candidates get measured prices the change that produced about half the corpus's
  gains at **zero**, so those candidates were often never tried. *(§3.6)*
- The stage's per-op ledger has **no way to record its one reliable win.** 37 rows say *"this norm runs on 1 core,
  the advisor wants 8–88"*; every one is filed `below_threshold`, `not_measurable` or `rejected`, and **none is
  filed `kept`** — including all 14 such rows in the cell that shipped that exact change for **−12.98 %**.
  *(§3.14)*
- **No cell ever applied its plan as written.** Where that is measurable, doing so is worth **3.7×** what shipped.
  *(§3.27)*
- On eight cell/kinds it was **never shown the layer at all** — a tracer gap, not a placement verdict.
  ([`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md))

Everything here comes from the cells' own artefacts or from re-measurements on the same hardware. This is the
short version; the detailed analysis is in [`FINDINGS`](ADVCHAL-V2-FINDINGS.md), where a reference like `§3.11`
points. Plain `§1`–`§6` are sections here. Claims corrected along the way are kept in
[`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md), each with the check that would have caught it.

---

## 1. The results

**8 of 15 cells shipped a gain; 7 returned zero.** The shipped total is **13,601 µs/model** against **21,368 µs**
measured as reachable from the advisor's own directions on the same decoders. Both are lower bounds — nine cells
were never probed for a better configuration, and eight cell/kinds were never fully traced.

One row per cell. The `control` column lists every layer kind that cell measured, one per line. Elsewhere a
second line means a second *scope*, not a second kind, and names its own kind where it has one.

| model | arm | control ms/layer | what it shipped | result | reachable (measured) |
|---|---|---|---|---|---|
| llama-3.2-1B | `exp17` | `dense` 0.3731 | nothing | 0.0 % | — *ladder never swept* |
| llama-3.1-8B | `exp17` | `dense` 0.6650 | nothing | 0.0 % | **0.0 %, tested** — full ladder swept, nothing beat the default |
| phi-3.5-mini | `exp17` | `dense` 1.1009 | nothing | 0.0 % | — *advised sharding never tried* |
| phi-3.5-mini | `nofuse-noadvise-onA` | `dense` 0.6570 | `rope_l1_rect32` | **−8.75 %** | — *never probed* |
| phi-3.5-mini | `nofuse-noadvise` | `dense` 0.7888 | `rope_l1_chain`, sharded multiply/add | **−5.74 %** | — *never probed; see the note below* |
| phi-3.5-mini | `fuse-noadvise` | `dense` 0.8072 | rope only, L1 interleaved | −4.91 % | **−17.84 %** — the advised rope (−10.43 %, *bit-identical*) plus the advised 11-core norm |
| qwen3.6-27B | `fuse-noadvise` | `full_attention` 1.2083<br>`linear_attention` 19.1402 | `packed_qkv_l1_chain` | −445.7 µs/model<br>*inside its ±618.5 µs band* | — *`linear_attention`, 97 % of model time, was never advised on* |
| qwen3.6-27B | `nofuse-noadvise` | `full_attention` 1.4494<br>`linear_attention` 15.8526 | nothing | 0.0 % | — *same coverage gap; the geometry hard-failed on the rest* |
| gemma-4-12B | `exp11` | `sliding_attention` 1.2541<br>`full_attention` 1.3774 | `Q+K+V+MLP` + output chain | **−1.14 %** | — *52 measurements, no advised grid among them* |
| gemma-4-26B | `nofuse-noadvise` | `sliding_attention` 1.2597<br>`full_attention` 1.2617 | `sliding_attention_o_chain` | −147.9 µs/model<br>−0.34 % on `sliding_attention` | **−12.44 %** on `sliding_attention` — a win it wrote, shipped disabled, never screened. **36×** |
| gemma-4-26B | `nofuse-noadvise-onA` | `sliding_attention` 1.8252<br>`full_attention` 2.0132 | `advisor_norm88` | **−12.98 %** on `sliding_attention` | **−13.63 %** — 44 cores beats the advised 88, bit-identically |
| gemma-4-26B | `fuse-noadvise` | `sliding_attention` 1.3412<br>`full_attention` 1.5394 | `advisor_concat_projection` | **−2.04 %**<br>−1.69 % on `sliding_attention` | −1.86 % on `sliding_attention` — *only 0.17 pp; the 88-core norm regressed here and was correctly rejected* |
| north-mini | `fuse-noadvise` | `dense_full_attention` 0.1727<br>MoE kinds 0.5537 | MoE norm at 32 cores | **−10.23 %**<br>−10.37 % on `sliding_attention_moe` | **−11.28 %** — 16 cores beats both the advised 22 and the shipped 32 |
| north-mini | `nofuse-noadvise` | `dense_full_forced_rope` 0.2033<br>MoE kinds 0.6138 | nothing | 0.0 % | **coverage-limited, not headroom-limited** — 2 tracer handlers open 11 candidates worth **632 µs/model** |
| north-mini | `nofuse-noadvise-onA` | `dense_full_attention` 0.2918<br>sparse MoE kinds 0.8465 | nothing | 0.0 % | sparse MoE now traceable — 2 candidates, **61.9 µs/model** |

*`result` is at the cell's own scope: the model-level figure it claimed, with the per-layer figure on the named
kind beneath it where the two differ. `reachable` is always per-layer on the kind named in the same row, and is*
**what was measured to be reachable from the advisor's own directions on that same decoder** *— how much better
the cell could have been had the advice been followed more closely. A blank is an absence of measurement, not a
measured zero, and the two columns are not directly subtractable; the corpus totals that combine them are in §4
and* [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) *§3.11.*

*All 15 cells ran on the same host, so nothing above is a hardware difference — the arm name says only how that
cell's incumbent decoder was produced.*

⚠ **The reachable column covers 6 of 15 cells.** Five have a measured better configuration; llama-3.1-8B `exp17` was
swept and has nothing further. The other nine were never probed. **So the corpus reachable total — and the 64 %
credited figure derived from it — are lower bounds.**

**The cheapest place to look next is phi-3.5-mini `nofuse-noadvise`**, on an inference rather than a measurement:
it shipped the *sharded multiply/add* form of the rope chain, which on the `fuse-noadvise` decoder is worth
−6.97 % against −10.43 % for the fully advised form. If that ordering holds on its own incumbent — untested — it
has headroom too.

### How solid are the zeros, and which are still blocked?

A zero can mean three different things — no headroom, no measurement, or no way to look — and they carry very
different weight. Only one of the seven is verified by measurement. The last column says what has happened to the
obstruction since: four of the seven had a real one, and two of those four are now measured through.

| cell | zero because | how well established | blocker since? |
|---|---|---|---|
| **llama-3.1-8B** `exp17` | nothing on its ladder beats the default | **Verified.** The whole achievable ladder was swept — {8, 16, 32, 64}, the only counts its knob can express. 16 (≈ the advised 22) and 32 are both **inside the noise floor**; 64 is +3.78 %, 8 is +11.21 %. The advised 22 is not expressible at all; the knob rounds it to 16. There is no hidden norm win | **none — this zero is real** |
| llama-3.2-1B `exp17` | its two candidates regressed | **Asserted, not verified.** Its ladder was never swept. The norm arrived well placed and the advisor wanted *fewer* cores — a direction that wins about half the time — so there is no particular reason to expect a win here, and no measurement either way | **none — just unmeasured.** Sweeping its ladder needs device time, not a fix |
| phi-3.5 `exp17` | every direction overlapped its floor or hard-failed | **Partly.** Its floor is the corpus's second-worst (1.092 µs) and the advised rope sharding — the thing worth −10.43 % on phi-3.5 `fuse-noadvise` — was **never tried here** | **none — a screening-order choice**, not an obstruction. Fixed by F5 in [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) |
| north-mini `nofuse-noadvise-onA` | sparse MoE untraceable | **Not a placement verdict.** `ttnn.sparse_matmul` had no direct-TTNN handler, so the advisor never saw the MoE tail | **UNBLOCKED.** One handler ported from the sibling tracer. Capture 14 → 39 ops, `untraced` **77.15 % → 14.39 %**, ceiling 0.66× → **10.09×** the noise floor, **2 candidates worth 61.9 µs/model**. [tt-mlir branch](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps), [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md) |
| north-mini `nofuse-noadvise` | all measured geometries slower or stalled | **Graded reasonable, and that grade was wrong.** It did screen and its `advisor_dense_chain_exact` candidate regressed 15 %, but its ceiling sat *below* its own noise floor because three quarters of each MoE layer was never traced | **UNBLOCKED.** Two handlers — `ttnn.ones_like` and `ttnn.sparse_matmul`. `untraced` **~76 % → ~21 %** on both MoE kinds, ceiling 0.45×/1.80× → **16.09×/22.77×**, **11 candidates worth 632 µs/model** — the largest find in the corpus. [tt-mlir branch](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps), [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md) |
| qwen3.6-27B `nofuse-noadvise` | the geometry hard-failed | **Partly a coverage gap.** Its `linear_attention` kind is 97 % of model decode time, and the trace stopped inside it — **63.5 % of that layer `untraced`**, so ≈62 % of the model was never advised on. The advisor did see the other third. Reading the profile directly finds the corpus's largest single cost there, ~191 ms/model of `retilize` (§3.18) — not a placement defect | **TRACER UNBLOCKED, STILL BLOCKED.** Four gaps closed — `ttnn.copy`, `ttnn.softplus`, `TracedTensor.__getitem__` and `ttnn.repeat_interleave` — and the full **71-op** mixer now traces. A native abort inside `mlir::PassManager::run` blocks it instead: **not** a coverage gap, unattributed, and needs a debug build. [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md) §5 |
| qwen3.6-27B `fuse-noadvise` | its win is inside its own band | **Honest about itself** — the cell said so. But the same coverage gap applies to its `linear_attention` kind | **PROBABLY UNBLOCKED, UNTESTED.** Same four handlers should apply; this arm was never re-captured, so that is an expectation, not a measurement |

Three of the seven were never blocked: llama-3.1-8B has a real zero, llama-3.2-1B needs device time, and phi-3.5
`exp17` needs a different screening order. Of the four that were, **two are unblocked and re-measured**, one is
**expected to be but untested**, and one **moved** — qwen `nofuse-noadvise`'s obstruction is no longer coverage but
a pipeline crash.

Note what the two north-mini rows have in common: **both cells screened honestly and both concluded correctly from
what they could see.** The zero was a property of the tracer, not of the decoder or the optimizer — and nothing in
the stage's own output distinguished that case from a real zero. That is the finding worth carrying forward, more
than any individual cell's number.

**So the tally is 1 verified, 1 asserted, 1 screening-order, and 4 coverage gaps** — not the 3 the earlier
accounting recorded. "7 of 15 returned zero" is true and misleading: for most of them nobody had shown the
advisor the layer.

Per-cell narratives and every cell measurement: [`MEASUREMENTS`](ADVCHAL-V2-MEASUREMENTS.md).

---

## 2. What this says about each half

### 2a. What v2 got wrong — the parts around the optimizer

| question | answer |
|---|---|
| **How much did it ship?** | **13,601 µs/model** across 8 of 15 cells, where **21,368 µs** was reachable from the advisor's own directions on the same decoders — **64 %**. Both figures are lower bounds: nine cells were never probed for a better configuration, and eight cell/kinds were never fully traced. |
| **What is the single biggest miss?** | The **screening order**. The harness builds candidates up chain by chain from the frozen incumbent and never applies the optimizer's plan as written. On the one cell where the counterfactual is measurable, applying it gives **−17.84 % against the −4.88 % that shipped — 3.7×** — and −10.43 % of that is **bit-identical** to the incumbent. Nobody tried it. |
| **Did the cells follow the advice?** | **7 of 15 tried the advisor's exact value on at least one advised item**, 6 as their first candidate, and all 7 ended cleanly. **4 never tried it and recorded no reason. None applied the whole plan.** Of the 9 cells that changed anything, **3 shipped the advised sharding *and* grid**. |
| **Are the zeros failures?** | Mostly not the optimizer's. Only **1 of 7 is verified by measurement** (llama-3.1-8B `exp17`, whose whole ladder was swept). **4 of the 7 are coverage gaps** — the tracer never showed the advisor those layers, so the stage produced no verdict either way. Three of those four are now unblocked and re-measured, and the fourth moved from a coverage gap to a pipeline crash: §1 and [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md). |
| **Whose defects?** | **10 in the stage**, almost all one-file changes with no build, and they account for the larger measured loss. **6 in the optimizer or its reporting**, needing tt-mlir work. The ledger is §3. |

**The surrounding tooling is new, and it shows — which is the good news.** `ttnn-jit` is ten months old and
actively developed. Its tracer stopped eight cell/kinds early, and closing every real gap took **218 lines of
Python and no rebuild** ([tt-mlir branch](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps)).
Of the ten ops cells recorded as blockers, **four already had handlers and one did not exist at all**. The skills
that drive the stage are the same kind of young: `advised_cores` parsed from a lossy summary field when the correct
one sits beside it, `unfixable_ops` read and discarded, a screening ceiling that prices half the real
opportunity at `0.000 µs`. None of that is a compiler problem, and none of it needs a compiler change.

### 2b. What `ttnn-advise` is actually worth

**It is a deterministic placement solver, and it is better than the corpus made it look.** The distinction worth
holding onto: it does not sample, search stochastically, or tune incrementally. Given a graph it discards the
incoming memory configurations entirely and re-places every op from scratch, and the same graph yields the same
answer — four runs in **18.4 / 18.4 / 18.1 / 18.6 s**. That makes it something a pipeline can depend on, which no
hand-tuning loop is.

**The one end-to-end test of that claim, on phi-3.5 `fuse-noadvise`**, its own harness, a fresh process per configuration:

| configuration | median ms | Δ | differential PCC |
|---|---|---|---|
| incumbent (frozen control) | 0.807535 | — | — |
| **what the cell shipped** | 0.768104 | −4.88 % | 1.0 |
| **the optimizer's plan, implemented from its IR** | 0.723320 | **−10.43 %** | **1.0** |
| the cell's own best, which it discarded | 0.700120 | −13.30 % | 0.99999107 |
| **that plan plus its advised 11-core norm** | **0.663507** | **−17.84 %** | 0.99999107 |

Read the third row carefully. The optimizer's plan, transcribed and run, is **2.1× the gain a careful human-driven
sweep produced on the same decoder**, at output that is bit-identical. It was never tried.

**What else holds up.** Its *direction* on the dominant defect class — widen a starved reduction — was right in
**4 of 4** cells where anyone measured it. It **declares what it cannot place**, naming the op and the exact
runtime error, which is an output the stage throws away. Adding *"and the advisor wants more cores"* to a
profile-only detection rule narrows a 7-cell flag list to 5 with the same recall. And once the tracer stopped
truncating, it had considerably more to say: eight cell/kinds went from 58–77 % of the window untraced to 4–21 %,
surfacing **17 screenable candidates**, including 11 on a cell that had published a flat zero.

**The real defect is grid choice, and it is not small.** `LayoutScore` has no latency term at any level;
`getOpRuntime` exists in `TTNNOpModel.cpp` and is never consulted. `coreCount` sits at level 6 of 7, and for
normalizations it gets overridden by the *input* operand's grid volume — exactly 1 on decode shapes — so the term
cannot vary with the candidate at all. The measured consequence: across the three ladders that were swept its grid
choice reached **82 %** of achievable, where a fixed *"closest to 16 cores"* rule with no compiler involved
reached **99.4 %**. **3 of the 4 placement wins in the corpus are at grids it did not name.** It found the op; the
right value was a rung or two away. One selection — llama's MLP norm at `1x22`, when 32 and 64 were valid and
outrank it on both of its own documented tiebreakers — the decision trace does not explain.

**So the honest shape of it.** Implementing everything it said was **much better than what shipped, and still not
optimal** — the −17.84 % configuration is above its own plan, and elsewhere 16 cores beat its advised 22 and 44
beat its advised 88, bit-identically. It is a strong starting configuration and a reliable detector, not an
oracle. Use it that way: **apply the whole plan first, then sweep the legal ladder around each op it named.** The
first half is free and the stage never did it; the second half is what closes the 82 %-to-99.4 % gap until
`LayoutScore` prices latency.

And two things are correctly outside its reach, which the corpus's largest numbers happen to be: a 191 ms/model
`retilize` cost that is a graph-shape choice, and a missing tt-metal kernel. A layout advisor was never going to
find those, and holding it responsible for them would misread the result.

---

## 3. The ledger: advisor not good enough, vs stage not listening

Every defect found, sorted by **whose it is**. The split matters because the two groups go to different
codebases and cost very different amounts to fix.

*`ADV-n` / `STG-n` are IDs local to this table. The bold codes in the last column (`D1`, `C5f`, `F5`…) are
action points in [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).*

### Real defects in the optimizer and its reporting — tt-mlir changes

Only ADV-1 and ADV-2 are defects in the placement decision itself. ADV-4 is unexplained. ADV-3, ADV-5 and ADV-6
are the summary and API surface — the information exists, it just is not exposed usefully.

| id | defect | consequence |
|---|---|---|
| ADV-1 | **`LayoutScore` has no latency term at any level.** `getOpRuntime` exists in `TTNNOpModel.cpp` and is never consulted. | It cannot rank by speed at all. Everything below follows from this. → **D1** |
| ADV-2 | **`coreCount` is level 6 of 7**, and for norms `NormalizationRules.cpp:77-104` overrides it with the *input* operand's grid volume — exactly 1 on decode shapes — so the term **cannot vary with the candidate**. | Its grid values lose when measured: **3 of the 4 placement wins are at grids it did not name** (§4) |
| ADV-3 | **Candidate layouts are deduped by shard shape keeping the *smallest* grid, before per-op legality filters run.** | A legal sibling can be discarded and an illegal representative kept → **D3** |
| ADV-4 | **One selection the trace cannot explain**: llama's MLP norm chose `1x22` while 32 and 64 were valid and outrank it on both documented tiebreakers. | Either the recorded score is not what is compared, or there is an unrecorded criterion → **D5** |
| ADV-5 | **`report.json` renders a multi-range `CoreRangeSet` as its first range only**, and prints no shard shape at all. | The summary understates its own advice. *Shared blame*: the information survives intact in `final_ir.mlir`, so nothing is lost by the optimizer — only by its summary |
| ADV-6 | **It does not emit the legal ladder**, so a challenger has to guess which core counts are even legal. | Wasted device time on illegal geometries → **D4** |

### The stage is not listening — skill changes, all cheap

| id | defect | measured cost | fix |
|---|---|---|---|
| STG-1 | **It never applies the advisor's plan as a candidate.** Screens chain by chain, building up from the incumbent. | **3.7× on the one cell where the counterfactual is measurable** — −17.84 % available vs −4.88 % shipped, ≈1.43 ms/model on phi-3.5 `fuse-noadvise` alone (§3.27) | **F5** — apply_all first, then ablate |
| STG-2 | **The screening ceiling prices only boundary conversions**, so an in-chain re-grid is worth `0.000 µs`. | **60 % of the disagreed-on cost filed `below_threshold` and never measured.** Two of the three biggest wins in the corpus came from cells whose ceiling said `0` / `not_measurable` and which recorded **0 kept chains** | **D0** |
| STG-3 | **`advised_cores` is parsed from the lossy `cores=` field** when the correct grid string sits beside it. | **58.3 % of advised core counts wrong; 34.4 % of the "disagreement" is phantom.** Two phi cells recorded themselves as *overriding* the advisor while agreeing with it | **C5f** — one line |
| STG-4 | **`unfixable_ops` is ignored.** The advisor names each unplaceable op with the exact runtime `TT_FATAL`; `reconcile.py` reads the field only to annotate the `untraced` bucket. | **54 declarations, 41 screened anyway.** Cells burn measurements rediscovering errors handed to them in writing | **C5g** |
| STG-5 | **The oracle rule rejects anything that moves PCC at all**, implemented as a differential bar ≈1.0. | Discarded a −13.3 % candidate at PCC 0.9999911 that is *more accurate than what shipped* (§3.1). And note §3.27: **−10.43 % was available at PCC exactly 1.0**, so even the strict rule permitted more than shipped | stage: absolute oracle at the model's own bar |
| STG-6 | **`agrees_with_shipped` never compares the memory space** — core count or DS-family only. | 1 of phi-3.5 `fuse-noadvise`'s 2 such rows is wrong | **C5c** |
| STG-7 | **`pair_confidence: position` is recorded, documented as a guess, then ignored downstream.** | **23.2 % of pairings corpus-wide** are guesses that nothing discounts — enough to turn positional artefacts into apparent findings | **C5e** |
| STG-8 | **It never re-advises**, screening every candidate against one start-of-run capture. | `ttnn-advise` costs **~18 s** — less than a single harness measurement (§3.29) | **F6** |
| STG-9 | **The capture monkey-patches `_decode_rope`**, so the advisor never sees the cell's real RoPE. | The advice for that region is advice for a substitute method | stage/capture, or fix the tracer limitation |
| STG-10 | **It throws away the perf report it runs.** Only 1 of 15 cells saved a before/after profile pair. | Op-level verification is impossible for 14 cells | **B0** |

The two columns are not the same size of problem. **ADV-1 and ADV-2 are the one thing worth a tt-mlir build**:
without a latency term the optimizer cannot rank by speed, which is exactly the 82 %-vs-99.4 % gap in §2b. Everything
else on the left is reporting. **The stage's ten are almost all one-file changes with no build, and they account
for the larger measured loss** — STG-1 alone is 3.7× on the cell where it can be measured, and STG-2 hid two of
the corpus's three biggest wins. gemma-4-12B `exp11` is the extreme case: **52 measurements without ever applying one
advised grid.**

### Neither — outside any layout advisor's reach

Two of the corpus's largest numbers are in neither column and belong outside a layout advisor entirely:
`retilize` at **191 ms/model**, which is a graph-shape choice in the decoder, and the missing sharded-GQA SDPA
kernel, which is tt-metal's. Both are itemised in §4.

The tracer gaps used to be listed here too. They do not belong — they were `ttnn-jit`'s, cheap, and are fixed.

---

## 4. What is still on the table

Placement leaves **≈9.2 ms/model** on the table across four cells. All four were located from the advisor's
*own* per-op output, by a one-line static check needing no device time — but only **half** the total is the
advisor's actual recommendation. In three of the four it named the right op and a grid that a sweep of the legal
ladder beat.

| cell | winning change | advised value | µs/model | the advisor's own number? |
|---|---|---|---|---|
| **phi-3.5 `fuse-noadvise`** | rope as advised + 11-core norm | rope L1 + **11** | **−4,609** | **yes** |
| gemma-4-26B `nofuse-noadvise` | residual/norm at 22 cores, sliding only | 88 | −3,918 | no — 26× what it shipped |
| gemma-4-26B `nofuse-noadvise-onA` | norm at 44 instead of the 88 it shipped | 88 | −375 | no — and bit-identical |
| north-mini `fuse-noadvise` | MoE norm at 16 instead of the 32 it shipped | 22 | −264 | no |

**From outside the advisor — much bigger, and correctly outside its reach:**

| opportunity | scale | whose |
|---|---|---|
| **`retilize` on qwen's conv chain** | **191 ms/model — 24.4 % of its decode time**, 14× every shipped win combined | the decoder's shape choice — a 4-element conv window on the 32-wide tile axis |
| qwen's untraced token mixer | the kind is **97 %** of decode time and **≈62 %** of the model. **The tracer gap is closed** — 4 fixes (`ttnn.copy`, `softplus`, `TracedTensor.__getitem__`, `repeat_interleave`) and the full 71-op mixer traces. A pipeline crash now blocks it instead, unattributed | was `ttnn-jit`; now needs a debug build |
| ~~tracer coverage gaps~~ | **all five real ones fixed** — 218 lines in `ttnn-jit`, no rebuild, on [`mvasiljevic/ttnn-jit-tracer-coverage-gaps`](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps). 8 cell/kinds went from 58–77 % untraced to 4–21 %. **17 new candidates**, the largest being north-mini `nofuse-noadvise`'s 11 worth **632 µs/model** | tt-mlir `ttnn-jit`, not tt-metal |
| sharded GQA SDPA output | blocks two cells' top candidate *and* a 2.6 ms/model wrong-op fix | tt-metal kernel |

Every one of these audited for whether it is real and what fixing it costs — with four of them fixed and
re-measured: [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md). Itemised with evidence:
[`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §8. What to change:
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

---

## 5. Reading the numbers

A few quantities appear with slightly different values in different sections. In every case that is a different
**scope** or a different **run**, not a disagreement:

| looks inconsistent | why |
|---|---|
| phi-3.5 `fuse-noadvise` shipped: **−4.91 % / −4.90 % / −4.88 %** | three runs of one configuration — the cell's own `final.json`, a recomputation from its block means, and a fresh re-measurement (§3.27). Run-to-run drift ≈0.1 pp |
| phi-3.5 `fuse-noadvise`'s discarded candidate: **−13.39 % / −13.4 % / −13.30 %** | same, three runs |
| north-mini `fuse-noadvise` shipped: **−9.26 % / −10.23 % / −10.37 %** | three **scopes** — its `final.json` (whose `incumbent_ms` is a 27.635 ms multi-layer harness), its own `model_estimate` (24,949 → 22,398 µs), and per-layer on sliding MoE |
| Δ with no unit qualifier | **per layer**. Model-level figures always say so |

Assumptions are labelled as assumptions. Claims that were later refuted are kept in
[`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md) rather than deleted, along with the check that would have
caught each one.

---

## 6. Where everything lives

| file | what's in it |
|---|---|
| **this file** | the results, what each half is worth, the ledger — a few minutes |
| [`ADVCHAL-V2-FINDINGS.md`](ADVCHAL-V2-FINDINGS.md) | **the detailed analysis: 29 findings and the method** — this is where `§3.x` lives |
| [`ADVCHAL-V2-BLOCKER-AUDIT.md`](ADVCHAL-V2-BLOCKER-AUDIT.md) | **every blocker the corpus recorded, classed and where cheap fixed** — 4 were already handled, 1 did not exist, 5 were real; 8 cell/kinds re-captured and re-measured |
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
