# advchal-v2 — the stage itself: what v2 fixed, and what it still gets wrong

Stage 02b `$advisor-challenger` was rebuilt between the v1 corpus (9 cells, tags `done/challenger/**`)
and the v2 corpus (15 cells, tags `done/advchal-v2/**`). This file grades the rebuild against what the
15 cells actually did, tracing each cell behaviour back to the rule that produced it.

**Sources.** `.agents/skills/advisor-challenger/SKILL.md` (477 lines),
`scripts/{reconcile.py, harness_template.py, capture_template.py}`,
`.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.{txt,check.sh}`, all at
`mvasiljevic/qb2/skillexp/challenger-skill-v2` @ `db00a44`. v1 baseline is the same paths at
`67da647d5eb` (2026-07-30).

> **Two corrections from later work.** (1) Advised core counts derived from the reconciliation's
> `advised_cores` are understated on **58.3 %** of ops — `report.json`'s `cores=` field prints only the first
> range of a multi-range `CoreRangeSet`; the grid-string product is the correct value. **59 of 334 `chain` rows
> stop being disagreements once corrected, carrying 34.4 % of chain µs.** Corrected per-op values:
> `advchal-v2-corrected-advice.json`. (2) `dram_resident` rows for ops the advisor declared in `unfixable_ops`
> are a *fallback after a declared failure*, not advice — 54 declarations corpus-wide, 41 presented as
> screenable anyway. → [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §1 and §11.

> **And the biggest defect found after this file was written is one it does not grade: the screening
> ORDER.** The skill builds up chain by chain from the incumbent. Applying the advisor's plan instead, on the
> one cell where the counterfactual is measurable, gives **−17.84 % against the −4.88 % that shipped — 3.7×** —
> with −10.43 % of it bit-identical to the incumbent. → [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) §F5.

## The size of the rebuild

```
 02b-advisor-challenger.check.sh   |  597 +++++------
 02b-advisor-challenger.txt        |   64 +-
 SKILL.md                          |  661 ++++++++----
 scripts/capture_template.py       |   20 +-
 scripts/harness_template.py       |  168 +++     <- entirely new
 scripts/reconcile.py              | 1128 ++++++++++++++++----
 6 files changed, 1851 insertions(+), 787 deletions(-)
```

The framing changed at the title line:

| | |
|---|---|
| **v1** | "turn a finished optimized decoder into **a faster one**, using `$shard-advise` as a challenger that cannot lose" |
| **v2** | "**measure how much** `$shard-advise` **adds** to a decoder already optimized without it" |

An optimisation stage became a measurement stage. Everything good and bad below follows from taking that
seriously.

---

## What v2 got right

### 1. A fixed harness protocol — the single largest improvement

v1 had **no harness template**; every cell wrote its own, and the reported noise floors ranged from
0.03 % to 1.37 % of the measured time — a **45× spread driven by protocol, not hardware**. v2 ships
`harness_template.py` with floors under the protocol and guards that refuse to go under them:

```
WARMUP >= 10   REPEATS >= 5   ITERS >= 50    incumbent_ms = MEDIAN of the block means
```

**Evidence it worked.** Across all 149 measurements in the v2 corpus, the protocol is uniform: every
measurement is 10 warm-ups / 5 blocks / 50 iters except **7 deliberate tightenings** by three cells (all
recorded, all to cure control settling). And the one cell that v1 had classified `aggregate_only` with a
**18.284 µs** floor came back at **0.712 µs** and shipped −1.83 % (gemma-4-12B).

Each rule is justified from a specific v1 failure, in the file:

- `WARMUP >= 10` — "One corpus harness did exactly 1, and its first timed repeat then carried **73 %** of
  the whole reported spread — a settling ramp misread as run-to-run variance."
- `ITERS >= 50` — each block is a *mean*, so between-block spread is `sqrt(ITERS)` tighter.
- median not min — "min-of-n is biased low by an amount that grows with n, so cells with different n stop
  being comparable. **All nine corpus cells recorded the min.**"

### 2. Non-overlap at fixed n

"Ship iff every candidate repeat beats every incumbent repeat", with n fixed at 5 so the false-positive
rate is comparable across cells (`1/C(2n,n)` → 0.40 % at n=5 vs 5 % at n=3). Plus a **fresh-process
confirmation** requirement, because per-process work happens once per process.

**Evidence it worked.** phi exp17 rejected a candidate with a *better median* whose repeats overlapped;
qwen B rejected `advisor_qkv_direct` on non-overlap; llama-1B rejected a DRAM concat on overlap. Three
cells declined a tempting number on a rule, which is what a measurement stage should do.

### 3. `feasibility.verdict` — refuse to spend device time on unmeasurable advice

Compares the advisor's proposed removal against the incumbent's own repeat spread, and returns
`measurable` / `aggregate_only` / `not_measurable` / `unknown` with an instruction for each. In v1 "one
cell had a ceiling of 0.65× its floor and **shipped a win anyway**".

### 4. Full-model estimate with a band

`model_estimate` scales the per-layer window by the layer counts *and scales the noise floor with it*.

**Evidence it worked.** qwen FN headlined that its −445.69 µs/model gain sits inside a **±618.50 µs**
band. Under v1's framing that is a 445 µs win; under v2 it is an unestablished one, stated as such.

### 5. Layer counts at preflight, and per-kind ranking

Requires the layer-kind counts before anything else, and mandates ranking *across* kinds by the per-model
number rather than per-layer. The file gives the counter-example: a 1.629 µs/layer sliding chain is
65.2 µs/model over 40 layers while a 2.146 µs/layer full-attention chain is only 17.2 µs/model over 8.

### 6. "The advice is a whole-graph plan; you apply parts of it" — half right, and the half that is wrong is expensive

A new section, and it changed behaviour: when a candidate's first neighbour rejected sharded I/O, **qwen
FN** extended through the advisor's adjacent `add → concat` reconfiguration and **nm B** extended through
an existing L1 sharded→interleaved conversion, both getting a measured number instead of a blank. **That
extension behaviour is a genuine improvement and it stands.**

⚠ **But the framing — "you apply *parts* of it" — is now measured as the stage's single largest defect.**
Nothing in v2 ever applies the plan *whole*. On phi FN, the one cell with the artefacts to test it end to end,
applying the advised placement as written gives **−17.84 % against the −4.88 % the cell shipped — 3.7×** — and
**−10.43 % of that is bit-identical to the incumbent**, so no correctness rule blocked it. It was never tried.
Recorded below as **D11**. → [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §7.

---

## What v2 still gets wrong

Ordered by measured cost.

### D1. The oracle rules jointly force a faster, *more accurate* candidate to be discarded

This is the worst defect in the stage and it is not a cell error. Three rules combine:

**SKILL.md §4** instructs rejection on any PCC movement, and recommends a differential oracle:

> If a placement-only candidate **moves PCC at all**, do not tune the threshold: **reject the candidate
> however fast it is**, and report the op and the shard spec as a tt-metal bug.

> A **differential** oracle catches this well and cheaply: same weights on both sides, candidate against
> the frozen incumbent.

**The same section then warns the opposite** — that this exact oracle is not an oracle:

> An oracle that compares the implementation against itself cannot fail. … "preservation relative to the
> frozen incumbent" [passes] automatically for any placement change that keeps tracing working.

**check.sh** makes the result binding, and endorses the differential form:

```
line 285:  # A differential oracle against the frozen incumbent is the RIGHT oracle for a placement change.
line 263:  if f.get("oracle_passed") is not True:
line 264:      crit.append("oracle_passed is not true -- a faster decoder that fails its oracle is a regression")
```

The only way to satisfy *"differential is right"* **and** *"it must be able to fail"* **and**
*"reject on any PCC movement"* is a differential bar at ≈ 1.0. phi FN wrote exactly that
(`comp_pcc(..., 0.999999)` in its own `oracle.py`) and the gate then made the result binding.

**What it cost, measured.** [`ADVCHAL-V2-EXPERIMENTS.md`](ADVCHAL-V2-EXPERIMENTS.md) §E1: the discarded
candidate is **13.39 % faster**, deterministic, passes the model's own 0.995 bar, and is **more accurate
against the HuggingFace reference than the configuration that shipped** (0.99904 vs 0.99890). The stage
discarded **−3,466 µs/model** and shipped **−1,267 µs/model**.

**Why the rule cannot work as written.** It conflates two things a PCC delta cannot distinguish by size
alone:

| | signature | correct response |
|---|---|---|
| a kernel bug under a shard spec | PCC drops **materially** (the skill's own word) | reject, file a tt-metal bug |
| floating-point **reassociation** | PCC moves in the 7th decimal | benign — it is *guaranteed* whenever a reduction's core count changes |

For any reduction op, case 2 happens **by construction**. So "moves PCC at all" makes every norm re-grid
unshippable — and norm re-grids are the highest-yield class in this corpus.

**The corpus contains the discriminating test already.** north-mini shipped the same class of change
because it used an **absolute** oracle against a reference decoder at the model's 0.995 bar, not a
differential one. Both readings are endorsed by the skill; the two cells got opposite outcomes on the same
kind of change.

### D2. `reconcile.py` never fills the verdicts the gate demands

`reconcile.py` writes `"verdict": "pending", "measured_ms": None, "repeats_ms": None` (lines ~453, ~511)
and never fills them. The gate then **requires** resolved verdicts, and the skill requires the
reconciliation to be tool-generated rather than hand-edited. That is not satisfiable, and four cells
invented four different ways out:

| cell | workaround |
|---|---|
| llama-8B | added `reconcile.py --results <evidence manifest>` |
| g26 FN | added `reconcile.py --evidence`, with stale-identifier rejection |
| north-mini FN | a separate `decisions_*.json` + `annotate.py` |
| others | edited the JSON, or added `record_results.py` |

**Consequence:** whether a cell ended up tagged records *which violation it chose*, not the quality of its
measurements.

### D3. The grid-sweep rule points cells in the wrong direction

SKILL.md tells cells to *"never sweep only at or below an advised core count; always measure at least one
exactly-dividing grid"*. So cells swept **at and above** the advice.

**Measured refutation** ([`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E4): north-mini's advised value was
22; the cell swept 22 / 32 / 64 and shipped 32. The true optimum is **16 — below the advised value** — and
it is worth a further **5.4–5.7 µs/layer on both MoE kinds**, ≈ **−264 µs/model**, at unchanged accuracy.
The response curve is **non-monotonic and bimodal**, with a local *maximum* at the advised 22.

Two further measured facts kill the "sweep higher" instinct entirely:

- **phi**: 11 → 48 cores is a **plateau within 3 µs**, and the *uneven* advised 11 is marginally best.
  Exact tile division bought nothing (§E2).
- **north-mini**: 40 / 44 / 48 / 55 / 88 cores are **illegal** — `shard_spec_validation.cpp:104` rejects a
  padded width exceeding the tensor by ≥ one shard width. The cell's three points were nearly the entire
  legal ladder above 22.

The rule should be: **enumerate the legal ladder and sweep it on both sides of the advice.**

### D4. The advised core count is treated as meaningful, and it is not

The advisor has **no latency term at any level** of its objective, and core count is only its *sixth*
lexicographic tiebreaker — see [`ADVCHAL-V2-ADVISOR-INTERNALS.md`](ADVCHAL-V2-ADVISOR-INTERNALS.md).
For normalization ops the `coreCount` term is overridden with the *input's* grid volume, so it cannot vary
with the candidate at all. Nothing in the skill tells a cell this, so cells anchor their sweeps on a
number that carries no throughput information.

**And there is a second, separate problem underneath it: the number the stage reads is not the number the
advisor chose.** `reconcile.py:194` parses `advised_cores` out of `report.json`'s `cores=(x0,y0)-(x1,y1)`, which
prints only the **first range** of a multi-range `CoreRangeSet`. **58.3 % of advised core counts corpus-wide are
understated**, and **59 of 334 `chain` rows stop being disagreements once corrected — 34.4 % of the disagreed-on
µs.** The `AxB` grid string printed beside it is correct. Two phi cells recorded themselves as *overriding* the
advisor while agreeing with it. So D4 is really two defects: the count is misread, *and* even the correct count
carries no throughput information. → [`ANALYST-PITFALLS`](ADVCHAL-V2-ANALYST-PITFALLS.md) Pattern 1.

### D5. The ceiling misprices in-chain re-grids at zero

The reconciliation ceiling counts *boundary conversions the advice does not place*. A re-grid of an op
that stays inside its L1 chain removes no boundary, so it prices at **0.000 µs**. g26 onA recorded a
0.000 µs ceiling on both layer kinds and then measured **236.8 µs/layer** from exactly such a re-grid.
A cell that trusts the ceiling ships a false zero — which is what nm onA and g26 B did.

### D6. Products are only mandated *across* layer kinds, not within one

§6 requires the candidate space to be "the **product** of per-kind winners, not the union of
one-kind-varied sets". There is no equivalent instruction for two independent winners *within* a kind — and
that is where the value was: phi FN's rope+norm product was **−13.24 %** against a best isolate of
−7.60 %, and gemma-4-12B's `q_k_v_mlp` product was −2.82 % against a best isolate of −1.86 %. Only
**2 of 15** cells built a within-kind product; both gained.

### D7. Nothing checks that a candidate is control-plus-one-knob

Only nm FN is recorded as having verified it, and **the check failed** — its candidate policies had
inherited constructor defaults rather than cloning the frozen incumbent, silently changing several dormant
fields. Six candidates and two confirmations had to be remeasured with `dataclasses.replace`. Fourteen
cells never checked, and the gate does not ask.

### D8. The noise floor is treated as within-process, but a large part of it is cross-process

Measured in §E3: the **first** harness process of a session recorded a floor of **11.838 µs**; the
identical configuration in a later process recorded **0.196 µs** — a **60× difference** from JIT-cache
warmth *between* processes. Per-process warm-up cannot remove it, and the stage requires one process per
configuration. Any cell whose control was the first thing it ran carries an inflated floor, which
directly changes `feasibility.verdict`.

Three cells raised warm-ups after seeing a settling signature — all three only for the **control**, never
to separate an overlapping **candidate**, even though the skill's own `not_measurable` guidance says to
"tighten the harness (more replays per timed block)".

**And that guidance is wrong.** Measured on phi exp17: going from 250 replays per measurement to 1,800 made
the block spread **3–4× worse** (0.4–0.7 µs → 1.3–3.0 µs) and still did not separate the candidate. The
`sqrt(ITERS)` argument in `harness_template.py` assumes i.i.d. noise within a run; drift is not i.i.d. and
does not average down. See [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E8.

### D9. The profiler and the timing protocol conflict, undocumented

Three cells hit three different versions of this and invented three different workarounds: nm onA
overflowed device profiler buffers at 250 replays (re-ran with Tracy mid-run dumping); g26 FN found
trace-replay rows carry no host markers between signposts so the bounded window came back **empty**
(added a profile-only single-eager-replay wrapper); g26 B used `*_bounded` profile variants. The skill
should prescribe the profile-only path rather than let each cell rediscover it.

### D10. Soft name/position pairing produces false boundaries

Two cells caught the reconciliation claiming removed boundaries that the authoritative IR **retains**
(qwen FN via IR inspection; nm B disproving several as rotary-input layout conversions). Both excluded
them. A cell that trusted the tool would have over-attributed.

---

### D11. Nothing ever applies the advisor's plan as written

The screening procedure is build-up: start from the frozen incumbent, add one chain at a time
(`SKILL.md` §4 — *"Each chain as one unit, one variable per measurement"*). The advised plan as a whole is never
a candidate, and `final_ir.mlir` — the only artefact carrying the complete plan with its shard shapes — is never
mentioned in the skill.

**Measured cost, phi FN, its own harness, fresh process per form, all at differential PCC 1.0 except where
noted:** incumbent 0.807535 → what shipped 0.768104 (−4.88 %) → rope as advised 0.723320 (**−10.43 %**) → rope as
advised + the advised 11-core norm 0.663507 (**−17.84 %**, PCC 0.99999107). Strict non-overlap throughout.
**3.7× what the cell shipped, ≈1.43 ms/model on that cell alone.**

Two mechanisms make build-up lose. **Sub-floor chains are never tested** — 60 % of the disagreed-on cost
corpus-wide sits in `below_threshold` chains that are individually unmeasurable and collectively obvious (phi
FN's own norm chains were 178 µs and 196 µs). And **"not tried" becomes indistinguishable from "tried and lost"**
— four cells' unproven deviations look identical in the artefacts to a measured rejection.

The skill already half-knows this: its `aggregate_only` feasibility verdict says *"apply the top chains together
as one candidate first"* — but only as a fallback when no single chain clears the floor. The corpus says it
should be the default. → action **F5**.

### D12. `unfixable_ops` is read, then discarded

The advisor declares every op it could not place in `report.json`'s `unfixable_ops`, each with the exact runtime
`TT_FATAL`, obtained by querying tt-metal's own constraint machinery. `reconcile.py:603` reads the field — but
**only** to annotate the `untraced` bucket's informational note. An unfixable op landing in `dram_resident` or
`chain` is never cross-referenced against it, and `nlp_concat_heads_decode` lands in `dram_resident` in every
cell, where the reconciliation labels it *"advisor placed it in DRAM — that is advice"*.

**54 declarations corpus-wide, 41 still presented as screenable advice.** Cells then spend device time
rediscovering errors handed to them in writing: phi FN's `advisor_sdpa_concat_l1` knob and its `dense:b43` chain
both record the identical string from `unfixable_ops`. `SKILL.md` and the stage prompt never mention the field.
→ action **C5g**.

### D13. Capture scope is unconstrained, unrecorded, and varies 6× between cells

Not a template defect — a **per-cell** one, and it is the most under-examined part of the stage. Fifteen cells
wrote fifteen capture scripts, **54 to 290 lines**, and nothing compares them.

- **Four cells substitute model methods before tracing.** phi FN, B and onA each replace `_decode_rope` (the
  tracer cannot resolve `memory_config()` before layout assignment, so they write a stand-in with a declared
  config); **qwen B replaces `_rms_norm_decode`, `_decode_linear` and `_partial_rope_decode`** — three of the op
  classes this corpus's findings concern. The advisor's advice for those regions is advice for a stand-in.
- **Six never trace the model's own `decode_forward`**, hand-writing the traced path instead.
- **At one shared terminal — `ttnn.sparse_matmul` — five cells stopped in four different places**, from 30 ops
  captured down to 5. A 6× spread on the same wall in the same model family.
- **Two invented private env knobs** (`CHALLENGER_CAPTURE_ATTENTION_ONLY`, `CHALLENGER_FINALIZE_CAPTURE`) whose
  values are not recorded anywhere, so a reader cannot tell which mode produced a given capture.

The fix is cheap and it is one field: **record the capture's own scope in `report.json`** — ops attempted,
methods substituted, knobs and their values. Without it, cross-cell coverage numbers mix captures that attempted
very different amounts of the layer, and nothing in the artefacts says so.
→ [`CAPTURE-VARIANCE`](ADVCHAL-V2-CAPTURE-VARIANCE.md).

---

## Verdict on the rebuild

**v2 fixed measurement and broke shipping.**

Everything about *detecting* an effect got better: one protocol, comparable floors, a stated
false-positive rate, a feasibility gate, and a model-level number with an error bar. The corpus's numbers
are trustworthy in a way v1's were not, and the two cells that reported zero after exhaustive screening
(llama-8B, llama-1B) are now demonstrably right — I re-measured llama's entire achievable ladder and found
nothing (§E3).

What v2 broke is the path from a measured win to a shipped one, in three places. Its oracle rules are
internally contradictory and jointly forced the corpus's largest measured win into the bin. Its grid guidance
pointed the best-executed cell away from a further 1 pp. And — found last, and the largest of the three —
**it never applies the advisor's plan as written**, which on the one cell where that can be measured cost
**3.7×** (D11). The first two defects hit the *same op class* — the 1-core reduction — which is where nearly all
of the corpus's value turned out to live. The third is not op-specific: it is the screening order itself.

**A note on where that leaves the advisor.** D4, D11 and D12 are all cases of the stage mis-reading, under-using
or discarding something the advisor got right. That is a more encouraging position than the reverse, because
these are one-file changes with no build — but it also means v2's numbers understate what the advisor was worth.
→ [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md) §7.
