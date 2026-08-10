# advchal-v3 — expectations, revised after the nmFN shakedown

Supersedes the per-cell targets in [`ADVCHAL-V3-CHANGES.md`](ADVCHAL-V3-CHANGES.md) §6. Those were **v2
headline percentages carried across as targets**; one shakedown cell showed that at least one of them rests on
a measurement that does not reproduce, and that the whole class is an order of magnitude smaller than the
targets implied. Why each number moved, and whether that was a misjudgement or new knowledge, is in
[`ADVCHAL-V3-ANALYST-PITFALLS.md`](ADVCHAL-V3-ANALYST-PITFALLS.md) — read that alongside this.

## 1. What the stage can do, measured rather than assumed

One cell, run cleanly, with the incumbent reproduced to four decimals (`0.1727` v2 vs `0.1727` v3):

| | |
|---|---|
| flagged pool (boundary ceiling + cliff pool) | **14.1 %** of the layer window |
| best candidate measured | **−1.76 %** |
| **realised fraction of the flagged pool** | **12.5 %** |
| what shipped | −0.60 % (a decision defect, see §4) |
| the two MoE kinds, independently | −1.76 % and −1.77 % |

So the stage's yield is roughly **an eighth of what it flags**, not all of it. That single ratio is what the
old targets were missing: they quoted pool-sized numbers as though placement recovered the whole pool.

**The flagged pool is now the expectation, and it is computable before any device time** — from step 0 for the
existing corpus, and from the cell's own reconciliation once it captures.

## 2. Capacity per cell, from step 0

Boundary ceiling + cliff pool as a share of the summed window across that cell's kinds. This is an **upper
bound reached only if every flagged microsecond went to zero**; multiply by the ~12.5 % realised fraction for
an expectation.

| cell | window µs | flagged | = upper bound | expect ≈ | note |
|---|---:|---:|---:|---:|---|
| phi-3.5 `fuse-noadvise` | 725 | 160.4 | **22.1 %** | ~2.8 % | largest pool in the corpus |
| phi-3.5 `nofuse-noadvise-onA` | 569 | 70.4 | 12.4 % | ~1.5 % | boundary-only |
| phi-3.5 `nofuse-noadvise` | 700 | 70.7 | 10.1 % | ~1.3 % | boundary-only |
| gemma-4-26B `nofuse-noadvise-onA` | 3770 | 361.7 | 9.6 % | ~1.2 % | 353 µs of it is cliff |
| phi-3.5 `exp17` | 1016 | 83.6 | 8.2 % | ~1.0 % | boundary-only |
| gemma-4-12B `exp11` | 2517 | 194.9 | 7.7 % | ~1.0 % | |
| north-mini `fuse-noadvise` | 1174 | 56.8 | 4.8 % | ~0.6 % | **measured 1.76 %** — see §3 |
| north-mini `nofuse-noadvise-onA` | 1925 | 82.1 | 4.3 % | ~0.5 % | |
| qwen3.6 `fuse-noadvise` | 1008 | 34.3 | 3.4 % | ~0.4 % | ⚠ its 97 %-of-model kind is absent from this data |
| **llama-3.1-8B `exp17`** | 648 | 4.4 | **0.7 %** | ~0.1 % | the corpus's only **verified** real zero |
| north-mini `nofuse-noadvise` | 1345 | 8.2 | 0.6 % | ~0.1 % | v2's coverage zero; re-derive after capture |
| **qwen3.6 `nofuse-noadvise`** | 17100 | 33.7 | **0.2 %** | ~0.02 % | its real cost is 191 ms of `retilize` |

**Two independent validations that this metric measures something real.** It puts **llama-3.1-8B `exp17`** last
but one — the one cell whose entire ladder was swept and found genuinely empty — and **qwen `nofuse-noadvise`**
last, the cell whose dominant cost is a graph-shape problem no layout advisor can reach. Neither outcome was
put in by hand.

**Two caveats, stated because the table invites over-reading.** nmFN measured **1.76 %** against a 0.6 %
prior — because the live capture with the v3 tracer handlers found *different and larger* kinds than v2's
committed artefacts show, so a step-0 prior derived from v2 data **understates** any cell whose coverage
improves. And gemma-4-26B `fuse-noadvise` and llama-3.2-1B `exp17` are absent entirely: they committed no perf
CSV, so their capacity is **unknown, not small**.

## 3. Per-cell expectations, revised

`was` = the target in CHANGES §6. `why` links to the pitfalls entry.

| cell | was | now | why it moved |
|---|---|---|---|
| **phi-3.5 `fuse-noadvise`** *(step 1, the gate)* | −10.43 % must be reached, else the rebuild is wrong | **process criteria, not a number** — see §5 | E3: −10.43 % was produced by a hand-written `PHI_ROPE_MODE` patch, not by anything the stage can do. Keeping it as a pass/fail gate risks stopping a correct run for a capability reason |
| **gemma-4-26B `nofuse-noadvise`** | −12.44 %, "26× what it shipped" | **unknown; treat as uncalibrated** | Same provenance class as nmFN: no `done` tag, and step 0 could not reproduce its window from its own CSVs. Its number was never checkable |
| **gemma-4-26B `-onA`** | −13.63 % (44 cores over the advised 88) | **~1.2 % from a 9.6 % pool**, and the 44-vs-88 increment ≈ **0.1 pp, not 1 pp** | E2: the per-rung value was contradicted by the same corpus and measured at 0.08 pp |
| **north-mini `fuse-noadvise`** | −11.28 % (16 cores) | **−1.76 % — MEASURED**, 16 best by 0.08 pp | E1: exactly one v2 measurement (32 cores → 0.5184) fails to reproduce; its ladder neighbour reproduces to 0.2 % |
| north-mini `nofuse-noadvise` | ">0 candidates; 11 worth 632 µs/model" | **unchanged in kind**, re-derive capacity after capture | Coverage predictions are the ones that held: nmFN went 69 % → ~14 % untraced, 49/49 layers |
| north-mini `-onA` | 2 candidates, 61.9 µs/model | unchanged | same |
| qwen `nofuse-noadvise` | "shipped ≈0, the finding is the 191 ms" | **unchanged, and now quantified**: 0.2 % pool | The capacity metric agrees with the qualitative call |
| qwen `fuse-noadvise` | untested | **0.4 % on the kinds we can see**; its dominant kind is still unmeasured | The ⚠ row of the coverage table: it reads clean only because its 97 %-of-model kind produced no reconciliation |
| gemma-4-12B `exp11` | ≥ −1.14 %, plus the wrong-op finding | **~1.0 %**, wrong-op finding unchanged | v2's −1.14 % and the capacity estimate agree — the only cell where the old target and the new method concur |
| llama-3.1-8B `exp17` | must stay 0.0 % | **unchanged — the control** | Strengthened: it has the smallest flagged pool in the corpus |
| phi-3.5 `nofuse-noadvise` / `-onA` / `exp17` | ≥ v2 | **~1.0–1.5 %** | Same recalibration as everything else |

**Corpus-level, replacing "≈9.2 ms/model is on the table".** That figure summed four per-cell wins, of which
two are now uncalibrated (nmFN, g26B) and two were sized on per-rung deltas measured at a tenth of the
assumption. Summing the flagged pools and applying the measured 12.5 % gives an expectation on the order of
**1 ms/model across the corpus, not 9**. Stated as an order of magnitude on purpose: one cell's realised
fraction is one data point.

## 4. What the shakedown says about the stage, beyond the numbers

- **The cliff check works and produced the win.** 4 candidates across 2 kinds, ranked by per-model µs, zero
  device time to find. The shipped change came from it.
- **The ladder was swept properly** — 16 / 22 / 32 both kinds, with confirmations — and independently
  reproduced the threshold finding: all the value is the first step off one core.
- **The absolute oracle is buildable per cell** and was, at the model's own bar from its own test file.
- **Coverage held**: 49/49 layers, all three kinds captured.
- **Isolation held**: no blob shared with v2, all 156 parked refs still unnamed, no tree drift, freshness OK.
- **And the decision procedure failed.** The cell measured `norm16` at **0.543590** four times, never built its
  oracle (while `norm22` and `norm32` both got real-weight oracles and **both passed**), and shipped `topk110`
  at **0.550052** — **1.2 pp slower than a candidate it had in hand**. The gate passed it, because nothing
  reconciles `measurements/` against `final_ms`. That is v2's signature failure surviving into v3.

So the capability is better than v2's and the *decision* is not yet. Fixes, all general:

1. fail when any measurement beats `final_ms` without a recorded verdict and reason;
2. require the oracle on the **fastest** candidate, not only the shipped one;
3. `advised_plan_verbatim` becomes apply-all-**expressible** plus `inexpressible[]` with op cost;
4. the ladder emits `not_modelled` rather than `[1]` where its model does not apply;
5. the confirmation script fails on unparseable provenance.

## 5. The step-1 gate, restated

The old criterion — *"phi FN must reach −10.43 % or the rebuild is wrong"* — is withdrawn. It can fail for a
capability reason (no knob for an advised placement) and would then stop a correct run. Replaced with criteria
the stage controls:

1. `advised_plan_verbatim` records a **measured** apply-all-expressible candidate, and `inexpressible[]` names
   every advised placement with no knob, with its op cost;
2. the rope and norm candidates are each measured **and** their product measured (they are disjoint);
3. every measurement faster than what ships carries a verdict and a reason;
4. the fastest candidate carries an oracle at the model's own bar.

If all four hold and the number is small, that is a **result**. If any fails, the stage is still wrong. That is
falsifiable by something v3 is responsible for.

## 6. Should the whole set be updated now? Yes, and why

Two of eleven targets are traceable to measurements that do not reproduce, and the rest were sized on a
per-rung value measured at a tenth of the assumption. Leaving them would mean every subsequent cell gets
graded against numbers we now know are wrong — and the cheapest way to make a run look like a regression is
to compare it against a target from a voided cell. The old table also **encoded v2's headline as the thing to
beat**, which is exactly how a contaminated number propagates: nobody re-derives a target, they only try to
hit it.

What is deliberately **not** changed: the coverage predictions (they held), the negative control (strengthened),
the qwen `retilize` finding (out of scope and unaffected), and the run order.
