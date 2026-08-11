# advchal-v3 — why the results differ from the expectations

Companion to [`RESULTS`](ADVCHAL-V3-RESULTS.md). Every deviation between what v3 was predicted to do and what
it did, sorted into the only three things it can be:

| | |
|---|---|
| **(A) wrong estimate** | the change did what it was designed to do; my prediction of its *size* was wrong |
| **(B) wrongly implemented** | the finding and the intent were right; the code does something else |
| **(C) unexpected side effect** | the change works as specified, and the specification has consequences I did not model |

The distribution is the finding, so it goes first.

| | count | what it cost |
|---|---|---|
| **(A) wrong estimates** | 5 | credibility of the predictions — no measurement was harmed, but **one (§2.0) was refuted by its own source data before the run** |
| **(B) wrongly implemented** | 3 | noise: false positives that make real signals harder to see |
| **(C) unexpected side effects** | 4 | **the actual lost wins** — **−6,055 µs/model measured** on g26onA's sliding kind (§3.2a), −1,285 µs on phiB (§3.2), −340 µs on phiA (§3.1) |

**Estimates cost credibility, implementations cost noise, side effects cost value.** And the side effects are
the category I never looked for.

---

# 1. The headline deviation

**Expected ≈1 ms/model across the corpus. Measured 6.8 ms.** v2 had claimed 9.2 ms.

**Category (A), and it compounds from two errors that point the same way.**

I computed each cell's capacity as `flagged pool × realised fraction`, where:

1. **the pools came from v2's committed artefacts.** I wrote the caveat — *"a step-0 prior derived from v2 data
   understates any cell whose coverage improves"* — and then used the priors anyway. The three former coverage
   zeros contributed **2.7 ms, 40 % of the run**, from priors of 0.1 %, 0.5 % and 0.02 %.
2. **the realised fraction, 12.5 %, came from the shakedown** — a run I already knew had shipped 0.60 % of an
   available 1.76 %. So I calibrated a corpus-wide multiplier on an observation whose defect I had just
   documented. Correcting only that puts the fraction near 37 %, and 6.8 ms is within reach of the pools.

**The thought-process error:** I derived a constant from a defective observation, then multiplied eleven
predictions by it. A calibration constant inherits every defect of the run it was calibrated on, and I had the
defect written down in the same document.

Worth naming the direction. My first estimates were **too high**, all five inflating the value of my own work.
Having been caught, this one was **too low by 7×**. The bias did not go away when it was named — it flipped
sign. Over-correction is the same failure with better manners.

---

# 2. Category (A) — wrong estimates

## 2.0 The one that is not merely wrong: a bound published below a known result

Separated from §1 because it is a different failure. §1 is *"my multiplier was calibrated on a defective
run"* — a bad estimate. This is **an estimate contradicted by the data it was computed from**, which needed no
run at all to catch.

| | |
|---|---|
| v2, gemma-4-26B `-onA`, from its own `final.json` | 54,633.6 → 47,528.2 µs = **−7,105.4 µs = −13.01 %**, shipped, confirmed, oracled |
| v3's incumbent for the same cell | `before_us` **54,633.6** — identical |
| what I published as the **upper bound** | **9.6 %** |
| what I published as the expectation | **~1.2 %** |

**An upper bound below a delivered measurement on the same baseline is not an under-estimate, it is refuted.**
The mechanism: `flagged` is the *advisor-attributable* window share, and this cell's reconciliation reports a
**0.000 µs advisor-attributable ceiling** with 64.7 % of its dominant kind untraced — so the capacity metric
inherited the attribution defect the cliff check exists to bypass. I wrote that caveat directly above the table
and used the number anyway.

**The rule:** *a capacity estimate is floored by what the cell has already been measured to deliver from the
same baseline; if the formula returns less, the formula is refuted for that cell.* One comparison per row,
against v2's `before_us`/`after_us`. Corpus-wide it refutes the headline as well: ~1 ms predicted against
**12.0 ms** already banked on five comparable cells.

**And it is the §4 pattern again, one level up.** §4 says I specified every *rule* by what it should catch and
never by what it would wrongly catch. Here I specified a *metric* by what it should measure and never asked
**what observation would prove this metric wrong** — with the disconfirming observation sitting in the source
file.

## 2.1 nmFN: expected −11.28 %, measured −1.69 %

Already documented as ERROR 1 in [`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md): the target came from the one v2
cell whose run its own driver marked `CONTAMINATED`, and exactly one of its ladder measurements fails to
reproduce while its neighbour reproduces to 0.2 %.

## 2.2 "F5 is unexecutable" — too strong, after one cell

Written after nmFN's `hard_error`. Across 11 cells, **5 applied an expressible subset**, and the run's largest
gain (g26onA, −11.91 %) is the cell that applied the advised plan and measured it. The bound is **partial**.
Generalising a capability claim from one cell is the same error as generalising the ladder rule from one tensor
width — one observation, stated as a property.

## 2.3 The phi family: expected ≥ v2, measured below on all three

phiFN −4.91 → −1.08, phiB −5.74 → 0, phiA −8.75 → −5.97. I predicted these at or above v2 because their pools
are the largest in the corpus. What I did not model is *which class* their wins belong to: phi's v2 wins are
rope and norm re-grids, which is precisely the class §3.1's defect vetoes and §3.2's bound cannot express. The
capacity metric measures *how much cost is flagged*, not *whether the stage can act on it*.

**A capacity estimate needs a reachability term.** Flagged pool × realised fraction is not enough when a whole
op class is unreachable for a structural reason.

## 2.4 Corpus-level: see §1

---

# 3. Category (C) — unexpected side effects. This is where the value went.

## 3.1 The absolute oracle's second clause vetoes reassociation — the defect that matters

**Specification:** *"Ship if the candidate is within the bar **and** no worse than the incumbent."* Taken
verbatim from `IMPROVEMENTS` A1.

**Consequence I did not model:** the incumbent *is* the reference-adjacent configuration, so any re-grid of a
reduction lands microscopically below it. Clause 2 therefore fires on **floating-point reassociation** — the
exact thing A1 exists to stop vetoing.

Measured on phiA:

| candidate | faster by | PCC | incumbent PCC | gap |
|---|---:|---:|---:|---:|
| `dense_advised_plan_ablate_norm` | **−0.90 %** | 0.99999249 | 0.99999261 | **1.2 × 10⁻⁷** |
| `dense_advised_plan_minus_norm_weight_rep` | −0.71 % | 0.99999148 | 0.99999261 | 1.1 × 10⁻⁶ |

Both pass the model's own 0.995 bar by four orders of magnitude. Both were vetoed for being less accurate in
the **eighth decimal place**.

**So I reintroduced v2's differential-oracle failure in absolute clothing.** v2's complaint was that a
differential oracle vetoes anything perturbing the arithmetic; clause 2 is a differential oracle with an
absolute reference bolted on. The clause split is the useful part:

- **clause 1 (below the model's own bar) works** — phiB's `rope_l1_compatible_geometry` at **PCC 0.9173** is
  real breakage, correctly rejected twice. That is the kernel-bug signal doing its job.
- **clause 2 is the defect.** It fires at 10⁻⁷.
- **g26B is the ambiguous middle**: 0.99469 against a 0.995 bar fails clause 1 by 0.0003 — defensible as
  written, and still disputed against the corpus's contradictory 0.99931 (P3).

**Corpus-wide, 21 of ~41 faster-than-shipped candidates carry an oracle veto**, and 7 of 11 cells shipped
something slower than their own best measurement. Two vetoes are provably clause-2 artefacts; g26onA's
fourteen need the same audit.

**Proposed fix, not made:** the veto is the model's own bar, full stop. "Worse than the incumbent" becomes a
recorded observation requiring explanation only when the gap is material — the same treatment I correctly gave
the differential oracle and inconsistently withheld here.

## 3.2 Inexpressible advice does not just lose wins — it manufactures wrong candidates

phiB's best measurement is named `rope_l1_compatible_geometry` and scores **PCC 0.9173**. The name is the
finding: the advised geometry was not expressible, so the cell built a *nearby* one — and the nearby one is
incorrect.

I had modelled the expressibility bound as "some advice cannot be applied, so some value is unreachable". The
consequence I did not model is that a cell facing an inexpressible placement will **substitute**, and a
substitute is a new configuration nobody validated. So the bound does not merely cap the upside; it creates a
correctness surface. `inexpressible[]` should therefore also record *what was substituted*, and a substituted
geometry should be labelled as such rather than presented as the advice.

⚠ **And the follow-up settles the open question the wrong way for v3.** This entry originally concluded that
phiB's `no_change` was correct and left open whether v2 had shipped the same broken geometry. **It had not.**
The two versions wrote *different knobs* into `tt/optimized_decoder.py` — v2's `advisor_rope_l1_chain`, v3's
`advisor_rope_l1` — and they differ in exactly two places: v2 does the `multiply`/`add` in the query's **sharded**
config where v3 does them **interleaved**, and v2 restores the key to `key_memory_config` where v3 returns
**both** tensors in `rope_memory_config`. v2's version measures **−5.09 % at PCC 0.998993** with the identical
oracle harness (`CACHE_SOURCE reference_fill recorded_target_prefix prefix_length=127 batch=32`, string-for-string
the same). So:

> **v3's `no_change` is right about its own candidate and wrong about the cell.** It rejected a defective
> implementation of a change that is achievable, achieved and correct in the other version's tree — and the
> defect is two lines of v3's own substitution, not a property of the model or of tt-metal.

[`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §2 has the code side by side and the per-op profile that confirms it:
v2's `BinaryNg` ×6 run at `L1_HEIGHT_SHARDED`, 13.25 µs, a row v3's interleaved path cannot produce.

## 3.2a One PCC sample vetoed seventeen measurements — the run's single most expensive line

⚠ **An earlier revision of this entry was wrong** and is corrected here. It said gemma-4-26B `-onA` *"tried one
rung and stopped searching that kind"*. Reading all 33 measurement files shows **the sliding ladder was fully
swept** — rungs 2, 4, 8, 11, 22, 44, seven single-norm ablations and the advised plan verbatim, **17 sliding
measurements**, the fastest at **1.581980 ms (−13.28 %)**, which is *faster than the 1.5873 v2 shipped*.

The actual defect: all 17 carry `verdict = rejected_kind_by_absolute_oracle`, and **sixteen** carry
`"oracle_pcc": "not_retested; sliding kind vetoed at its best measured rung"`. Exactly one was tested —
`ladder_11_sliding`, at **PCC 0.99457** against a 0.995 bar, missing it by **0.0003**.

> **The performance search was complete and correct. The correctness verdict was extrapolated from one
> configuration to sixteen that were never measured.** Two of them (8 and 22 cores) are within 0.25 % of the
> tested one.

**−242.2 µs/layer × 25 layers = −6,055 µs/model**, on a cell that shipped −1,198. **73 % of the whole v3-vs-v2
gap.**

**And the isolation makes it worse again: the verdict was not computed.**
[`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md) reads the cell's own `build_evidence.py`:
`oracle_passed = kind == "full_attention"` and `pcc = 0.9945729603715616 if not passed else ...` — **the
correctness verdict for the sliding kind is a constant keyed on the layer kind, and the PCC beside it is a
literal, with no oracle log committed anywhere in the cell.** Four candidate mechanisms for the drop were tested
on device with real weights and all four are ruled out or orders of magnitude too small. So this entry is no
longer "one sample vetoed seventeen measurements" — it is **a veto with no traceable measurement behind it at
all.**

**And the sweep already showed the op cannot be the cause.**
[`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) measured the op in isolation across 79 configurations: the
largest PCC deviation it can produce at *any* grid is **7.3 × 10⁻⁷**, against the **5.06 × 10⁻³** whole-layer drop
that triggered the veto — **6,879× too small.** So the op under test provably did not cause the number it was
rejected for. The veto is not "correct but over-broad"; **the attribution behind it was wrong before the scope
question arises.** Second fix, and the substantive one: **gate a placement candidate on the op's own output**, not
only on the layer's PCC — on a sparse-MoE layer the latter measures expert-selection agreement.

Unmodelled because I specified the ladder as a *performance* search and the oracle as a *gate*, and never asked
what **scope** a gate verdict has. `SKILL.md` says to sweep the ladder and says nothing about how far a
correctness rejection generalises — so the cell picked the widest scope available. Fix: an oracle verdict binds
only the configuration it was measured on; untested rungs are recorded as
`not_screened_after_correctness_rejection`. And **put the advised grid and the prior version's shipped grid on
the ladder** — 88 was the one rung never measured, and both remaining open questions turn on it.
[`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §1.3–1.5.

## 3.3 Parking a cell branch but not its run branch collides at publish

[`RUN-LOG`](RUN-LOG.md) P6. Operational, mine, cost 0 device hours because the driver's publish-only path
recovered the measurement — but it cost two failed publish attempts and would have cost 51 minutes had I
reflexively re-run.

---

# 4. Category (B) — wrongly implemented

All three are **one mistake made three times**: a rule applied to a population it does not hold over.

| # | rule | population it was applied to | population it holds over |
|---|---|---|---|
| B1 | C5c: agreement must match the memory space | every advised/shipped pair | none — the profile has no output space. 14 of 15 rows false |
| B2 | the legal ladder | every cliff candidate | only advice that is a **shard** over the tile axis. `topk` got `[1]` while shipping on 110 |
| B3 | a measurement faster than `final_ms` must ship or explain | every measurement vs **one global** `final_ms` | per layer kind — 12 false positives on nmFN alone |
| B4 | comparing v3's result against v2's | headline **percentages** | the same **layer kind**. On gemma-4-26B `-onA` I read −11.91 % against v2's −12.98 % and wrote *"reproduces v2"* — while v3 got **+0.00 % on the 25-layer sliding kind v2 won** and −12.10 % on the 5-layer minority kind. At model scope: **20 %, not 100 %** |

B1 was caught by step 0 before hardware. B2 was caught by the shakedown. **B3 I introduced *as the fix* for the
shakedown's defect, and it carries the same mistake** — I scoped *which* measurements to consider by kind and
then compared them all against a single number.

**B4 is the same shape as B1–B3, applied to my own reporting rather than to the code**, and it is the one that
reached a published table: a comparison rule (percentage vs percentage) applied across a population where it
does not hold (different layer kinds). It also produced the friendliest possible wrong answer — *"reproduces
v2"* — on a cell that had in fact missed v2's win by 5×. Corrected in
[`RESULTS`](ADVCHAL-V3-RESULTS.md) §1a; **only `µs/model` is kind-weighted and comparable.**

The corrected corpus picture: on the four cells where v2's control and layer counts are both recorded, v3
delivers **−2,731 µs against v2's −10,480 — 26 %** — plus **−2,701 µs** from three cells v2 scored at zero. So
v3 is **a more trustworthy 6.8 ms, not a larger one**, and that sentence should have been the headline from the
start.

**The thought-process error, and it is the sharpest thing in this document:** I specified every rule by **what
it should catch** and never by **what it would wrongly catch**. Clause 2 was specified as "don't ship something
worse" — its false-positive surface is every reduction re-grid, and I never enumerated it. Same for C5c, the
ladder, and B3.

> One question, asked of each rule before shipping it: **"what will this reject that I want kept?"** Asked four
> times, it catches all four. It costs minutes and it is not the same question as "does this address the
> defect", which is the only one I asked.

---

# 5. What actually worked, so the corrections are not read as a verdict

- **Coverage**: 3 cells that reported 0.0 % now report −2.7 ms/model between them, 40 % of the run's total.
  The one prediction that held in kind and exceeded its size.
- **The cliff check**: found the shipped candidate in the majority of cells, at zero device cost.
- **Clause 1 of the oracle**: caught a genuine PCC 0.9173 breakage that would otherwise have shipped as a
  −4.11 % win.
- **The measurements-vs-decision check**: noisy (B3) but it is why this document can quantify what was left on
  the table. v2 could not.
- **Provenance and isolation**: uniform across 11 cells — tracer fingerprint matched, no optimizer drift, no
  blob shared with any v2 run, 156 parked refs still unnamed, exclusive device on every measurement.
- **The incumbents reproduce**: `0.1727` v2 vs `0.1727` v3 on the one cell where v2's control is directly
  comparable, which is what makes every number here mean anything.

# 6. What I would change before the remaining four cells run

1. **Clause 2 stops being a veto** (§3.1). This is a design change to the rule the stage turns on, and it
   needs a decision, not a patch.
2. **B3 compares per kind** against `model_estimate.per_kind` (§4).
3. **`inexpressible[]` records substitutions** (§3.2).
4. **Re-run the affected cells afterwards** — phiA at minimum, and the oracle audit on g26onA's fourteen.
5. **Do not re-estimate the corpus total from this run's realised fraction.** That is how §1 happened.
