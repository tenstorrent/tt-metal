# advchal-v2 — what to change

Derived from the 15-cell corpus, the tt-mlir source at the pin, eight hardware experiments, and five
counterfactuals that change one stage setting at a time.
Evidence for every claim is in [`READ-THIS`](ADVCHAL-V2-READ-THIS.md) and the files it points to.

**Part 1** is the general ideas — the principles worth arguing about.
**Part 2** is the action points — concrete changes, ordered, each naming the file and the test that
would show it worked.

---

# Part 1 — General ideas

## I1. A measurement stage needs a correctness rule that distinguishes *different* from *wrong*

The stage's rule is "if a placement-only candidate moves PCC at all, reject it however fast it is". That
is sound for the failure it was written against — a kernel that computes the wrong answer under a
particular shard spec — but it cannot tell that failure apart from **floating-point reassociation**, which
is *guaranteed* whenever you change how many cores a reduction is spread over.

So the rule makes the corpus's highest-yield transformation permanently unshippable. It already cost one
cell 8.5 percentage points on a candidate that was measurably *more accurate* than what shipped.

The discriminating test is cheap and the corpus already contains it: compare **both** the candidate and the
incumbent against an *absolute* reference. Reassociation moves the candidate's error either way and keeps it
in family; a kernel bug moves it materially and in one direction. Ship if the candidate is no worse than the
incumbent against the reference.

## I2. "One number from the advisor" should never anchor a sweep

The advisor's objective has **no latency term**, core count is its sixth lexicographic tiebreaker, and for
normalization ops the core-count term is overridden with a value that doesn't depend on the candidate at
all. It also never executes anything.

Treating its recommended grid as a starting point for a *directional* sweep ("go above it") encodes an
assumption the objective does not support. The measured response curves are non-monotonic and bimodal, and
in one case the advised value sat at a local *maximum*. The advice is one legal point; the ladder is what
matters.

## I2a. The win is a threshold, not a gradient — so find the knee, don't chase the peak

Measured on both gemma-4-26B arms and on north-mini: the norm's response to core count is **flat over a wide
middle** (8→44 cores on gemma varies by 0.2 %) with a **cliff at the bottom** (1 core costs 13.7 %) and a
mild penalty at the top. Almost all the value is the *first step off one core*; which of the middle rungs you
land on is worth ~1 pp.

That reframes the search. A cell does not need to find the optimum — it needs to (a) detect that an op is on
the cliff, and (b) land anywhere in the middle. Detecting the cliff is a *static* check on the shipped graph;
only the second needs the device. And it explains why the advisor's specific number matters so little: any
legal middle rung captures nearly the whole win.

## I3. The searchable set should be computed, not discovered by crashing

Three different mechanisms restrict which grids can run: a shard-padding validation rule, per-model grid
helpers that only accept exact tile divisors, and per-op layout rulebooks. Every cell rediscovered these
by launching processes that died on `TT_FATAL`, and two cells drew wrong conclusions from grids that were
never legal.

All three are computable before touching a device. A stage that prints the legal ladder up front turns a
guessing game into an enumeration.

## I4. Accounting that can read zero next to a real win is not accounting

The ceiling counts boundary conversions the advice doesn't place. A re-grid of an op that stays inside its
chain removes no boundary, so it prices at exactly zero — while measuring 236.8 µs/layer on hardware. Any
metric that can report zero on a cell with a 13 % win available will produce false zeros, and it did.

Attribution needs a second channel for **in-chain re-grids**, with a direction, and the ceiling must never
be a stopping condition on its own.

## I4a. Coverage beats placement

Every structural zero in this corpus is a tracer-coverage zero, and the biggest one hides **97 %** of a
model's decode time behind a single unhandled op. Placement work on the visible 3 % cannot compete with making
the other 97 % visible. Coverage is the highest-leverage investment available.

## I5. A per-process protocol cannot control a cross-process effect

The stage fixes warm-ups, blocks and replays *inside* a process and mandates one process per
configuration. Measured: the first process of a session had a noise floor **60×** the same configuration's
floor later. That is JIT-cache warmth between processes, and per-process warm-up cannot touch it. Since
the floor decides `feasibility.verdict`, this silently changes what a cell is allowed to screen.

## I6. Where a stage under-specifies, cells will differ — and one of them will be right

Thirteen situations recurred across cells and were handled differently ([`READ-THIS`](ADVCHAL-V2-READ-THIS.md) §7).
In most of them one handling measurably beat the others. That is not a discipline problem; it is a
specification problem, and each divergence is a cheap place to encode the better handling once.

The clearest case: an impossible contract (a tool that never fills the fields its own gate requires)
produced four different workarounds, and which one a cell chose is what determined whether its work got
published — not the quality of its measurements.

---

# Part 2 — Action points

Ordered by (measured value) / (effort). "Test" is what would show the change worked.

## A. The stage — correctness rules

### A1. Replace the "moves PCC at all" rule with a two-sided absolute comparison ⭐ highest value

**Change.** In `SKILL.md` §4, replace reject-on-any-movement with:

> Build an **absolute** oracle: candidate vs a reference the change cannot move (HF layer, functional
> decoder), at **the model's own PCC bar** — read it from the model's own test, do not invent one. Also
> measure the **incumbent** against the same reference. Ship if the candidate is within the bar **and no
> worse than the incumbent**. Record both numbers.
>
> A **differential** oracle vs the frozen incumbent is a useful *observation* and must be recorded, but it
> is **not** a veto: any change to a reduction's core count reassociates the arithmetic and will move a
> differential PCC in the last few decimals by construction. A differential delta that is *large* is the
> kernel-bug signal — treat "large" as a departure the absolute comparison also sees.

**Also resolve the contradiction:** the same section currently recommends a frozen-incumbent differential
oracle *and* warns that one "cannot fail". Say plainly which is the veto (absolute) and which is the
observation (differential).

- **Files:** `.agents/skills/advisor-challenger/SKILL.md` §4; `02b-advisor-challenger.check.sh` lines
  ~262–295.
- **Gate:** require `oracle_kind ∈ {absolute, differential}`, `oracle_pcc_bar`, `oracle_bar_source`
  (file:line the bar was read from), and `incumbent_pcc_vs_reference`. **Fail** a cell whose bar is
  tighter than the model's own test bar without a recorded justification.
- **Test:** re-run phi FN. The combined candidate must ship. Expected: −13.4 %/layer, −3,466 µs/model,
  PCC 0.99904 vs the incumbent's 0.99890.
- **The strongest single argument for this change:** gemma-4-26B onA's shipped norm change moves a
  differential PCC by **0.0177** and shipped (absolute oracle); phi FN's moved it by **0.0000089** and was
  rejected (differential oracle). The rule as written punishes the *less* perturbing change.
- **And a differential oracle cannot identify which side moved.** Built the absolute version for
  gemma-4-26B B's discarded candidate: against the model's own bfloat16 `FunctionalDecoder`, the candidate
  scores **0.99931** and the **shipped incumbent 0.98347** — the incumbent is the one failing the model's bar.
  The differential number for that pair (0.98322) reads as "the candidate moved"; it did not.
  → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E9. This makes A1 not just a *permissive* change but
  a **correctness** one: the current rule can ship the less accurate configuration and reject the better.
- **Evidence:** [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E1, [`ORACLES`](ADVCHAL-V2-ORACLES.md).

### A2. Ship the three wins the corpus already has

- **phi FN**: `advisor_rope_l1="query_key"` + `advisor_norm_cores=11`. Measured −13.39 %/layer; oracle
  under A1 passes.
- **north-mini FN**: `advisor_moe_norm_cores=16` instead of 32. Measured −5.4 µs/layer (sliding MoE) and
  −5.7 µs/layer (full MoE), ≈ −264 µs/model; PCC 0.99951 vs the model's 0.995 bar.
- **gemma-4-26B B**: `GEMMA4_OPT_RESIDUAL_SHARD_CORES=22` for **sliding attention only** (it regresses
  +4.64 % on full attention, so ship per-kind as the stage's own product rule requires). Measured
  1.2583 → 1.1017 ms/layer, ≈ **−3,918 µs/model, 26× what that cell shipped**, and **more accurate than the
  shipped configuration** against the functional reference (0.99931 vs 0.98347). The cell had *written* both
  geometries and shipped `0`.
- **gemma-4-26B onA**: `GEMMA4_ADVISOR_NORM_CORES=44` instead of 88. Measured −12.2 µs/layer (sliding) and
  −12.4 µs/layer (full), ≈ −375 µs/model; **bit-identical** to the shipped 88 on sliding (PCC 1.0), so it
  inherits the real-weight oracle already passed. Also **relax that cell's own `cores % 11 == 0` check** —
  it is the cell's assumption, not the hardware's; north-mini ran 16- and 32-core norms on the same device.
- **Test:** the existing per-model real-weight tests, unchanged, plus a fresh-process non-overlap
  confirmation.

## B. The stage — search

### B0. Do not trust the advised grid at all — treat it as one sample ⭐

Measured over the three cells where both are defined: a fixed heuristic *"pick the legal grid closest to 16
cores"*, which never consults the advisor, captures **99.4 %** of the achievable win, while **the advisor's own
recommended grid captures 82 %**. It wins the grid choice in 1 of 3 cells, by 0.19 pp. And the advisor's
measured advice overall is a **49 % coin flip** (118 rows), with "use more cores" at 11 %.

**Change.** State plainly in `SKILL.md` that the advised core count carries **no throughput information** —
`LayoutScore` has no latency term and for norms the core-count term does not even depend on the candidate — so
it is a *detection* signal, not a *selection* one. Then B1.

→ [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md) §2–3.

### A0. Consume the perf report the stage already runs ⭐⭐⭐ free, no build, and the most predictive signal available

`tt-perf-report` already emits per op: **`Total %`**, `Bound`, `Cores`, `DRAM %`, `FLOPs %`, math fidelity — plus
a **stacked report with an `Op Category` column** (`Compute`/`TM`/`DM`/`Other`), a roofline summary, and per-op
advice (*"Increase grid size (currently using 2.0)"*, *"✅ Optimized"*). **`reconcile.py` reads three columns.**
Nothing in the skill or gate mentions `--group-by category`, `--summary-file`, `--stacked-csv`, the roofline or
the advice; 14/14 cells saved the CSV, only 4–5/14 saved the rest.

**Four changes, no build required:**

1. **Mandate the full invocation** — `--group-by category --summary-file … --stacked-csv …` — and gate on all
   three artefacts existing.
2. **Add a movement budget to the gate.** Compute the category split from the CSV the cell already saves and
   **warn above ~10 % non-compute, fail above ~25 % without a recorded reason.** On this corpus that flags
   qwen B (53.3 %) and all four phi arms (17–26 %) and passes both llamas — the right answer in every case.
   **This rubric is strictly more predictive of where the wins were than the advisor's own output:** the two
   cells that returned honest zeros are exactly the two cleanest (TM 0.9 %, 1.1 %).
3. **Rank the profile's own conversion ops by cost** — see B0a below.
4. **Record the tool's per-op advice** next to each screening decision; where stage and tool disagree, that is a
   result. ⚠ Treat it as a hypothesis: it says "increase grid size" for DS matmuls where E20 measured widening
   at **+65 %** slower.

→ [`PERF-REPORT-AUDIT`](ADVCHAL-V2-PERF-REPORT-AUDIT.md)

### A0b. Split the movement budget into layout-induced vs graph-structural

Of the mean **18.5 %** non-compute time, **6.0 pp is layout-induced** (tilize/untilize/typecast/fill-pad/
interleaved↔sharded/reshard/copy) and **12.5 pp is graph-structural** (permute/transpose/reshape/slice/concat/
head-ops). Only the first is reachable by a layout assigner, and only with **D0**. The second belongs to
`$optimize` or a new tt-mlir rewrite pass. **Report the two separately** so the right owner picks it up — the
corpus's largest item (qwen's 191 ms `retilize`) is in the second bucket.

One item may be trivially free: **`ReshapeViewDeviceOperation` costs 8.8 % of gemma-12B's window** and 4.0 % of
qwen FN's. A *view* reshape should be metadata-only; if it is materialising, that is free money.

### B0a. Rank the profile's own conversion ops, not just the advisor's plan ⭐

The stage's worklist is derived from the advisor's plan, so **it inherits the advisor's blind spots**. qwen B's
largest cost — 3,983 µs/layer of `retilize`, 191 ms/model — was filed under `boundary` with an advisor ceiling of
**0.000 µs**, which is *correct* (the advice places those conversions too) and therefore never actionable.

**Change.** Independently of the advisor, rank every conversion op in the profile by cost and report the top ones
with their effective bandwidth. A conversion moving 80 KB in 819 µs at **1 % of roofline** is a finding whatever
the advisor thinks of it. In this corpus that one addition surfaces the single largest item in the whole
exercise.

### B1. Emit the legal grid ladder, and sweep both sides of the advice ⭐

**Measured value: two of the three cells with a low-core reduction shipped the wrong rung** — north-mini
(32, best 16) and gemma-4-26B onA (88, best 44). Together ≈ **−639 µs/model** left behind.

**Change.** Replace *"never sweep only at or below an advised core count; always measure at least one
exactly-dividing grid"* with:

> Compute the **legal ladder** for the op before screening: the core counts that satisfy the shard-padding
> rule, the op's layout rulebook, and the model's own grid helper. Print it. Screen the advised value, the
> nearest legal value **below** it, the nearest **above**, and every exactly-dividing rung. The response
> curve is not monotonic — the corpus contains a case where the advised value is a local *maximum* and the
> optimum is below it.

- **Files:** `SKILL.md` §4; add a `--legal-grids <op> <tensor-width-tiles>` mode to `reconcile.py`.
- **Test:** on north-mini, the ladder must print `{1,2,4,8,11,16,22,32,64}` and exclude 40/44/48/55/88;
  the sweep must find 16.
- **Evidence:** [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E4.

### B2. Require the product of non-overlapping winners *within* a layer kind

The skill already mandates products *across* kinds. Extend it: once two or more candidates in the same kind
each pass non-overlap and touch disjoint ops, **their product is a required candidate**. Additivity is not
predictable — the corpus has one super-additive case and one sub-additive case, and in both the product beat
every isolate.

- **Files:** `SKILL.md` §4/§6; gate check that `measured_sets` contains the product when ≥2 winners are
  disjoint.
- **Test:** phi FN's rope+norm and gemma-4-12B's `q_k_v_mlp` are both required, not optional.

### B3. On an overlap, add processes — never longer blocks ⚠ *this replaces my earlier proposal, which I tested and refuted*

I originally proposed re-measuring an overlapping candidate at ≥4× `ITERS`. **I tested that on phi exp17 and
it fails twice over** ([`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E8):

- it did **not** separate the candidate — its two tightened medians straddle the control's;
- it made the noise floor **3–4× worse** (0.4–0.7 µs → 1.3–3.0 µs going from 250 to 1,800 replays).

The protocol's `sqrt(ITERS)` justification assumes i.i.d. noise within a run. It is not i.i.d. — longer
windows pick up slow drift, and drift does not average down. **50 replays/block appears to be near the sweet
spot; 200 is past it.**

**Change.** State in `SKILL.md` that more replays per block is *not* a rescue for an overlap, and that the
term worth attacking is the **cross-process** one (§E3 measured 60×): re-measure more *independent
processes* at the same block size. And record `not_measurable` with its arithmetic without spending device
time chasing it.

### B4. A zero ceiling obliges one screening before a zero may be published

If the reconciliation ceiling is ~0 µs **and** any material op sits on ≤2 cores, the cell must screen that
op before publishing a zero. gemma-4-26B onA got a 0.000 µs ceiling on both layer kinds and shipped
−12.98 % by ignoring it; two other cells trusted a similar ceiling and shipped zeros.

## C. The stage — mechanics

### C1. Make `reconcile.py` able to accept outcomes ⭐ removes the impossible contract

`reconcile.py` writes `"verdict": "pending", "measured_ms": None` and never fills them; the gate requires
them filled; the skill forbids hand-editing its output. Fold the best of the four ad-hoc workarounds
(`--results` / `--evidence` with stale-identifier rejection) into the shipped script as the **one**
supported path, and make the gate check that the reconciliation was regenerated with it.

- **Files:** `scripts/reconcile.py`, `SKILL.md` §3, `check.sh`.
- **Test:** a cell can produce a gate-passing reconciliation without editing any tool output.

### C1a. Record the incumbent's own grid for the op under test

Two arms of gemma-4-26B use the **same env knob name** with different defaults (`GEMMA4_ADVISOR_NORM_CORES`
= 88 in one, 8 in the other), so the "88-core candidate" is a 1→88 change in one arm and an 8→88 change in
the other. Nothing in `final.json` or the measurement records the incumbent's grid for the op being changed,
so the two deltas look comparable and are not. This cost me a wrong mechanism in a published conclusion
until I read the source.

**Change.** Require `op_under_test: {name, incumbent_grid, candidate_grid, legal_ladder}` per screened
candidate. It is the field that makes cross-arm and cross-model comparison meaningful, and it makes the
"is this op on the cliff?" check (I2a) mechanical.

### C1b. Flag any material op on ≤2 cores as a static precondition ⭐⭐ cheapest, highest-value change here

Per I2a the big win is a threshold. `reconcile.py` already knows every op's shipped grid, its advised grid
and its cost, so it can emit **before any device time**:

> `CLIFF: rms_norm on 1 core, 44.5 µs/layer (3.7 % of window), advisor wants 88, legal ladder {11,22,44,88} — screen this first`

**Validated by prediction.** Applying exactly this rule to the corpus's own per-op data flags 5 of 14 cells.
All three double-digit wins are in flagged cells; **no unflagged cell produced one**. I then used it to
predict an unscreened win in gemma-4-26B B and measured **−12.44 %/layer** — a candidate that cell had
written, shipped disabled, and never screened, worth **26× what it did ship**.

- **Files:** `scripts/reconcile.py` (emit it), `SKILL.md` §3 (screen flagged ops first), `check.sh` (fail a
  published zero that has an unscreened flagged op).
- **Cost:** no device time. **Effect in this corpus:** would have surfaced ≈ 8 ms/model.
- **It also fixes the screening order.** Ranking by the cliff instead of by advisor-boundary value moves the
  winning candidate from rank 2/2/2/**4-of-27** to **rank 1 in all four** cells whose win was a low-core
  reduction, and shrinks the list (27 → 2 on phi FN). → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E10.

### C1c. Make the multi-layer capture mandatory when its own trigger fires

`layers_in_window` is **1 in 23 of 23** reconciliations while `spill.ran` is **True in 8 of 8** cells — the
exact condition SKILL.md §2a cites for going to N=2. A recommendation whose trigger holds everywhere and is
followed nowhere is not a recommendation; it is dead text.

**Change.** When `spill.ran` is true (or `layer_handoff.entry_from_dram and exit_in_l1`), **require** N ≥ 2
and gate on it. At N=1 the layer-handoff question is unanswerable by construction — the py↔IR transition pins
both ends to DRAM — and **13 of 23** runs flagged a real per-layer conversion the stage then declared out of
scope without measuring. v1 measured that same cost at 33.6 µs (phi) and 48.0 µs (gemma-12B) per model.

- **Files:** `SKILL.md` §2a (change "consider" to a conditional requirement), `check.sh`.
- → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E11.

### C2. Enforce control-plus-one-knob

Require a recorded diff of executed policy vs frozen policy showing **exactly one changed field**, and gate
on it. The only cell that checked this found it *false* and had to remeasure six candidates and two
confirmations.

- **Files:** `harness_template.py` (record the diff), `check.sh` (fail on >1 changed field).

### C2z. No action needed, but state the reason: traced replay is load-bearing

The skill justifies traced decode replay as "what production does". Measured, it is stronger than that: the
sharded norm **adds ~46 µs of host dispatch and saves ~65 µs of device time**, so eagerly it reads as a
**regression** (+45.6 µs) and under traced replay as a **win** (−65.2 µs). **A cell that timed eagerly would
have rejected every norm win in this corpus.** Say that in the skill, so nobody "simplifies" the protocol.

It also explains why `*_profile` measurements run 1–3 % slower than their timed counterparts, and why one cell
found trace-replay rows carry no host markers between signposts: host and device costs move in *opposite*
directions for this candidate class, so mixing measurement modes produces contradictory rankings.
→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E18.

### C2a. No action needed: the first-repeat floor is fixed

Recorded here so nobody re-fixes it. v1's floors were mostly unfinished warm-up (one harness did **1** untimed
replay; the first timed repeat carried 45–73 % of the spread in 4 cells). Recomputing every v2 control's floor
with the first repeat dropped: **13 of 17 change by exactly 1.00×**, the worst case is phi FN at 1.52×, and
**no cell's `feasibility.verdict` flips**. `WARMUP ≥ 10` did its job.

The floor that remains is **cross-process**, not first-repeat — see C3.
→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E13.

### C3. Model the cross-process floor

Require a **throwaway warm process** before the first timed run, or record `process_ordinal` per
measurement and report the floor excluding the first process. Measured: 11.838 µs vs 0.196 µs for the same
configuration, first vs later process.

- **Files:** `harness_template.py`, `SKILL.md` §1, `reconcile.py` feasibility.

### C4. Prescribe the profile-only path

Trace-replay device rows carry no host op markers between signposts, so a bounded `tt-perf-report` window
comes back **empty**; and 250 timing replays overflow the device profiler buffers. Ship the
single-eager-replay profile wrapper and Tracy mid-run dumping as the documented path instead of letting each
cell rediscover it. Three cells, three workarounds.

### C5. Record `agreed_on: grid | ds_family`, and stop treating ds_family as nothing to screen ⭐

An `agrees_with_shipped` row can carry a different advised core count, because a DRAM-sharded match counts as
agreement regardless of grid (advisor score level 3 beats core count). Without the field the data invites
misreading — it misled this analysis once.

**Quantified.** Matmul/linear ops are **62.3 %** of the profiled window on average (up to 89.8 %), and
**55 rows carrying 64.7 % of all matmul cost differ in grid from the shipped placement yet carry no verdict**
because both sides are DS.

⚠ **RETRACTED, by my own experiment.** I originally wrote that a `ds_family` match with a different grid *"is a
candidate, not an agreement"*. **Measured: it is an agreement.** Turning off DRAM sharding on north-mini's dense
projections — exactly the advisor's 12 → 77-core direction — is **+65.2 % slower** (0.1725 → 0.2849 ms), and the
MoE expert-matmul geometries lose too (+0.96 %, +1.96 %). DS matmuls are DRAM-bandwidth-bound, so core count is
not the limiting resource. Corpus plus probes: **1 win in 7 measured matmul-widening candidates.**

**What still stands:** record the field. It is needed for legibility — `12→99, agrees` reading as agreement
without saying *why* misled this analysis once. **What is withdrawn:** the recommendation to screen those rows.
→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E20.

### C5c. `agrees_with_shipped` must compare the memory space, not only the core count ⭐

`reconcile.py:446-448` decides agreement with:

```python
same = (adv["cores"] is not None and adv["cores"] == dev["cores"]) or \
       (dev["dram_sharded"] and "dram_sharded" in adv["program_config"])
```

**Neither branch looks at the buffer type.** So an op advised into **L1** and shipped in **DRAM** is labelled
"nothing to screen, we already do what the advice says" as long as the core counts happen to coincide.

**Measured hit rate: 1 of 2.** phi FN has exactly two `agrees_with_shipped` rows, and one of them is wrong —
`typecast`, advised `l1/height_sharded/1x1 · 1c`, shipped `1 core · DRAM interleaved`. Advised cores `1` ==
shipped cores `1`, so it agrees; the L1-vs-DRAM disagreement is invisible. The other row (`linear`, 48.0 µs)
matches only through the DS branch, where the grid is 88 advised against 12 shipped.

**Fix:** require the space to match as well, and record `agreed_on: space+grid | ds_family` (C5). One-line
change in the skill's own script; no build.
→ [`PHI-BEFORE-ADVISED-AFTER`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md), code `L`.

### C5d. Fail the gate when a candidate profile parses to zero rows

phi A's `rope_l1_rect32_perf.csv` is **empty** — the profile ran, the CSV was written, and nothing parsed out
of it. Nothing in the stage noticed. A candidate whose profile yields 0 device rows is not a candidate that
performed neutrally, it is a candidate with **no evidence at all**, and it must not be reportable.

**Fix:** assert `len(rows) > 0` per profiled candidate in the gate, naming the file. Same class of hole as the
signposted-window-with-no-device-rows problem that forced `incumbent_large`.

### C5e. Do not let positional pairings read as findings ⭐

`reconciliation_dense.json` carries `pair_confidence: name | position` and states in `limitations[0]` that *"a
positional pair is a guess"*. **In phi FN, 11 of the 31 paired rows are positional** — 35 %. The field is
recorded and then ignored by everything downstream: positional rows sit in the same buckets, get the same
verdicts, and are counted in the same shares as name-paired ones.

This bit *this* analysis: seven rows I had labelled "does not follow the advice" or "op removed" are positional
guesses, and I have downgraded them to *undecidable*
([`PHI-BEFORE-ADVISED-AFTER`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md) → Corrections).

**Fix:** report `pct_paired_by_position` in the accounting block (the field name the tool already promises),
and exclude positional rows from any *count* the stage presents as a finding — keep them for cost attribution,
where a wrong name still lands in the right chain.

### C5a. Drop the "screen DS-matmul advice last" rule

The rule's stated basis — *"it has not won a measurement in this corpus"* — is v1-derived and **v2 contradicts
it**: gemma-4-12B kept a `linear` 12→55-core change worth 129.4 µs, and that cell's shipped win is on linear
ops. Combined with C5, the rule is self-fulfilling: it sends cells last to the op class holding two-thirds of
the cost, most of which the accounting has already exempted.
→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E15.

### C5b. Record the candidate's batch-shape assumption, not just `decode_batch`

The corpus's largest wins are **batch-pinned by construction** and nothing says so. phi's norm memory config
hardcodes a one-tile-row shard height (`shape=(ttnn.TILE_SIZE, width_tiles * ttnn.TILE_SIZE)`), so at batch 64
it fails with `TT_FATAL: Shard height 32 must match physical height 64 for width sharded`; batch 8 fails at
build. A reader of `final.json` sees "−13.4 %/layer" with no indication that it evaporates off batch 32.

**Change.** Require `candidate_shape_assumptions` per shipped knob (tile rows, divisibility, grid shape), and
have the gate fail a shipped change that has none. → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E17.

### C6. Keep the decision trace, compressed, as a deliverable

It is the only artifact that answers "why this grid". One cell committed it **uncompressed at 51 MB**;
another gzipped 118 MB to 7 MB. Require gzip, and `.gitignore` the raw profiler dumps that two cells had to
delete by hand after a 1.2 GB push rejection.

## D. The advisor (tt-mlir)

**First, what will *not* help:** raising the advisor's `optimization_level`. In
`tools/ttnn-jit/_src/shard_advisor.py:166` the level only gates `memory-layout-analysis-enabled` (true at ≥2);
levels 2 and 3 are identical for layout advice. **The stage already uses the only setting that matters**, so
the advice can only be improved by changing the objective below.
→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E14.

### D0. Price conversions as a cost, not a boolean ⭐

Two gaps that together make every layout crossing invisible to the advisor:

- **`requiresReshard` is a boolean** at level 5 of `LayoutScore`, and the struct has no reference to tilize,
  `isTiled`, or element type. **An 819 µs untilize and a 1.5 µs L1 regrid are the same value to it.** Given
  `retilize` is 76.5 % of all boundary cost in this corpus, that is the wrong abstraction.
- ⚠ **The row-major half is WITHDRAWN — tested.** Running the advisor with `row-major-enabled=true` on phi's
  own IR gives **zero row-major layouts**. They are enumerated either way
  (`generateAllPossibleLayouts` loops over {scalar, tiled} unconditionally); the flag lets far more through —
  pipeline log **3.3 MB → 35 MB** — and every one is rejected by op constraint validation:
  `TT_FATAL: Input tensor layout must be TILE but got Layout::ROW_MAJOR`. **The blocker is that TTNN ops reject
  row-major input**, i.e. tt-metal, i.e. out of scope. The flag's only effect was narrowing four matmuls from
  `width_sharded/1x96` to `block_sharded/1x11`, almost certainly a regression.
  → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E23.

  **So the conversion-cost change is the only advisor-side lever left, and it operates entirely within TILE
  layouts** — which is the whole space the advisor legally has.

⚠ **This would not have found the 191 ms item** (E-1) — that needs a graph rewrite, which the advisor cannot do
by construction, and it cannot place row-major either. But it is the prerequisite for the advisor reasoning
about layout crossings *at all*.

**Also settled by the same sweep, so nobody re-tries them:** the advisor is **deterministic**; **`opt-level 3`
is invalid** (`TTNNPipelines.h:592` validates 0..2 — E14 confirmed and closed); and
`disable-dram-sharded-matmul=true` changed nothing on the graph tested.
→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E23.

### D1. Add a latency term to `LayoutScore` ⭐

`getOpRuntime` already exists in `lib/OpModel/TTNN/TTNNOpModel.cpp` and is wired for many ops;
`LayoutScore::operator>` never consults it. A runtime comparison placed above `coreCount` would let the
model rank grids the way hardware does.

- **File:** `include/ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h`,
  `lib/Dialect/TTNN/Analysis/OpModelStrategy.cpp`.
- **Test:** llama-8B's MLP norm should stop selecting 22 when 32 and 64 are legal and faster.

### D2. Make the normalization `coreCount` override candidate-aware

`RmsNormRuleBook::adjustScore` sets `coreCount` from the *input* operand's grid volume (sharded path) or
tile height (interleaved path) — on a decode shape the latter is always **1**. Modelling the kernel's real
parallelism is right; discarding the candidate's own grid makes the term inert exactly where the corpus
found its biggest wins.

- **File:** `lib/Dialect/TTNN/Analysis/OpRules/NormalizationRules.cpp:77-104`.

### D3. Run op-specific legality filters before the shard-shape dedup

`generateAllPossibleLayouts` dedups by shard shape keeping the **smallest** grid per shape, "to reduce the
search space", before per-op rulebooks run. So the surviving representative of a shard shape can be one the
op rejects while a legal sibling was already discarded.

- **File:** `lib/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.cpp:128-225`.

### D6. ~~The op model accepts shard shapes the runtime refuses~~ — **WITHDRAWN, my error**

I proposed this as a tt-mlir legality fix plus a tt-metal recommendation. **There is no defect.** The advisor
specified shard **(32,64)** — tile-aligned — on **32** cores (`final_ir.mlir`, `#ttnn_layout24`). My probe used
**(32,48)**, which the advisor never specified. An isolated single-op test of the advisor's exact config passes
on every op, and the plan implemented verbatim is **−10.43 % at PCC 1.0**, against the **−4.88 %** the cell
shipped.

Nothing to change in tt-mlir or tt-metal from this. What replaces it is C5f (below) and F4.
→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §4–§5.

### C5f. Derive `advised_cores` from the grid string, not the `cores=` bounding box ⭐⭐ one line

`reconcile.py:194` parses `cores=(0,0)-(10,1)` → 22. The chosen layout's `CoreRangeSet` in `final_ir.mlir` has
**two** ranges, 22 + 10 = **32**. The `AxB` grid string printed immediately before it is already correct.

**Validated** against three decision traces (`beam[0].score.coreCount`, the value `LayoutScore` compares):
grid-string product correct **22/22, 10/10, 17/17**; bounding box correct 2/22, 1/10, 2/17.
**Corpus-wide: 476 of 816 advised ops — 58.3 % — carry an understated core count.**

**This is not cosmetic.** Two phi cells recorded themselves as overriding the advisor while agreeing with it,
and the deviation that followed cost one of them 1.43 ms/model (F4). Fix: use the grid product. One line, no
build. Optionally also stop `report.json` flattening the range set — but the consumer fix removes the
consequence, and `final_ir.mlir` already carries the truth.

### F4. Require the advisor's plan to be candidate #1, implemented from `final_ir.mlir` ⭐⭐⭐

The corpus's clearest single result. Cells that tried the exact advice first (6 of 15) all ended cleanly — they
shipped it or recorded a measured regression. **4 cells never tried it and recorded no reason**, all deviating
the same way: keep the advisor's L1 placement, drop its sharding.

Measured on the one cell with the artefacts to settle it (phi FN, its own harness, fresh process per form,
all PCC 1.0):

| form | median ms | Δ |
|---|---|---|
| frozen incumbent | 0.807535 | — |
| **what the cell shipped** — L1 interleaved | 0.768104 | −4.88 % |
| sharded multiply/add (phi B's form) | 0.751277 | −6.97 % |
| **the advisor's IR verbatim** | **0.723320** | **−10.43 %** |

Strict non-overlap throughout. **The unproven deviation cost 5.55 pp ≈ 1.43 ms/model** in a cell that reported
`improved` and passed its gate.

**Actions, cheapest first:**
1. **The stage must build its first candidate from `final_ir.mlir`, not from the `report.json` summary** — the
   IR carries the shard shape and the full `CoreRangeSet`; the summary carries neither correctly (C5f).
2. **Make "the exact advised layout" a required, named candidate** with its own row and verdict, so
   "not tried" is distinguishable from "tried and lost". Today they are not.
3. **A `TT_FATAL` while implementing advice must be recorded against the reconstruction, not the advisor** —
   with the single-op test that isolates it. Mine looked like an advisor defect for a week.

### D4. Emit the advisable ladder in `report.json`

The set of legal core counts per op is computable inside the advisor and is exactly what a challenger needs
(B1). Emitting it removes the guessing game and the wasted device time.

### D5. Answer the open question in the decision trace

For llama-8B's MLP norm, 32 and 64 cores were valid candidates that outrank the chosen 22 on both
documented tiebreakers (`coreCount`, `outputL1Usage`), and 22 matches neither neighbour. **The trace does
not explain the selection.** Either the recorded per-evaluation score is not what is compared, or the beam
applies an unrecorded criterion. Add the comparison actually used to the trace.

- **Evidence:** [`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) §5.

## E0a. Add a cross-model op-cost comparison — it finds defects the stage's question cannot express ⭐

Everything the stage does is *intra-cell*: one decoder against one advisor plan, asking *"which conversions does
the plan not place?"* — a question about **layouts**. A 15-cell corpus supports a question it never asks: **for
the same logical operation at the same batch, which cell is anomalously slow?**

**It immediately finds the corpus's largest un-screened defect, and of a class the stage is blind to:**

| op | cells | rows | mean µs | max µs | cores |
|---|---|---|---|---|---|
| **`concatenate_heads`** | **gemma-4-12B only** | 2 | **76.9** | **102.6** | **1** |
| `nlp_concat_heads_decode` | 13 others | 18 | **3.4** | 9.4 | 16 / 24 / 32 |

gemma-4-12B calls a **different TTNN op** for the same logical step: 102.6 µs on one core, **7.79 %** of its
full-attention window, against a 3.4 µs corpus mean. Estimated ≈**2.4–2.6 ms/model — 3.9× what that cell
shipped** — in the corpus's *most thoroughly screened* cell.

Nothing in the stage could reach it: the advisor advised **DRAM**, the reconciliation filed it under DRAM-advice,
and the cliff check (C1b) filters it out *because the advisor does not want it widened*.

**Action.** Emit, alongside each reconciliation, a table of this cell's per-op costs against the corpus median
for the same op at the same batch — and fail the gate on a ≥5× outlier that carries ≥2 % of the window without a
recorded explanation. Cost: arithmetic over profiles that already exist.

- **Then fix the instance.** The blocking chain, measured: `nlp_concat_heads_decode` needs a sharded input →
  SDPA must emit sharded → `TT_FATAL: Sharded output not supported for GQA` (recorded independently by phi
  exp17 and gemma-4-26B FN). So this instance depends on **E1a** below. Three in-place routes were tried and
  all failed; see [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E19.
- **Candidates the lens ranks next**, within-family and same core count so shape and parallelism are fixed:
  phi's SDPA (27.46 % of a window, **3.8×** between arms on 110 cores) and phi's
  `nlp_create_qkv_heads_decode` (7.80 %, **3.5×** on 32 cores). ⚠ Not isolated — the arms also differ in math
  fidelity and KV-cache dtype. Candidates, not results.

### E1a. Sharded output support for GQA SDPA — it gates more than one thing

`TT_FATAL: Sharded output not supported for GQA` was recorded as a dead end by **two** cells independently, and
E19 shows it is also what blocks the concat-heads fix: the peers that pay 3.4 µs keep the whole
SDPA → concat → O-projection run sharded, and a decoder that drops to interleaved at SDPA cannot rejoin.
So one kernel gap costs at least: two cells' top-ranked boundary candidate, plus ≈2.5 ms/model in a third.

### E0b. What is NOT a defect, recorded so nobody re-chases it

`nlp_create_qkv_heads_decode` on 1 core looks starved (283 µs, up to 9.27 % of window, 5 cells) and **is not**.
The op height-shards over batch, so its core count *is* the batch size — exactly, across all 23 rows in the
corpus (batch 1 → 1 core, batch 32 → 32 cores). **The advisor advising 1 core is correct.** I flagged this as an
opportunity and disproved it; the check is one line of arithmetic against `decode_batch`.

## E-1. Tiled input for the conv/recurrent composites ⭐⭐⭐ the largest single number in the corpus

**Measured: 3,983.5 µs/layer — 25.2 % of qwen B's 15,833 µs linear-attention layer — in `retilize`, i.e. the
tile ↔ row-major crossing. Across 48 layers that is 191,210 µs/model, 24.4 % of its 783,981 µs decode time, and
14× every shipped win in this corpus combined.**

The ops: `UntilizeWithUnpaddingDeviceOperation` ×3 at **819 µs each** and `TilizeWithValPaddingDeviceOperation`
×2 at **671 µs each**, all on the `add → rms_norm` edge, all **already on 109 of 110 cores**. Not
under-parallelised — the crossing itself. The decoder's own comment names the cause:

> *Conv, reshape, and recurrent composite kernels currently require interleaved tensors; cross that boundary
> once after the packed projection instead of four times before four independent matmuls.*

The decoder has already minimised the *number* of crossings; what remains is forced by the kernels.

**Why nobody looked:** the advisor's ceiling for that layer kind is **0.000 µs and that is correct** — the
advice places these conversions too, because they are legally required. The stage filed the cost under
`boundary`: reported, out of scope, uncredited. Both the advisor and the stage behaved correctly, and a quarter
of a 27B model's decode time sits there.

⚠ **Re-aimed after reading the shapes.** I first wrote "accept tiled input in the kernels". The real cause is a
**shape choice in the decoder**: `linear_conv_kernel_dim = 4`, and the conv state is
**(1, 1, 10240, 4)** — the 4-element window sits on the **32-wide tile axis**, so the tiled form is
**8× padded** (80 KB of real data → 640 KB). The conversions therefore run at
**0.90 and 1.10 GB/s against this machine's measured ~90 GB/s roofline — about 1 %** — 819 µs to move 80 KB on
109 cores. Not an inherent cost; a pathological shape.

**Action, cheapest first — all decoder changes, no tt-metal work required:**

1. **Keep the conv window on a leading axis.** Dims 0–1 are not tile-constrained; reduce over the window there
   and both `permute`s disappear along with the 4-on-32 axis.
2. **Replace shift-and-concat with a circular buffer.** Keep `kernel` slots, overwrite slot `t % kernel`, take
   the weighted sum against rotated weights. No slice, no concat, no permute — **the layout never changes**.
3. **Express the depthwise conv as a matmul** against a small banded matrix — tile-native by construction.

*(A tiled 4-wide variant of the conv composites in tt-metal is a separate, optional question; it is not needed
to recover the 191 ms.)*

`retilize` is **76.5 % of all boundary cost in the corpus**, so this shape pattern is worth checking in every
model with a short-window conv or recurrence.

- ⚠ qwen **FN** almost certainly carries the same cost and it is **unmeasured** — that arm's linear kind was
  declared tracer-unreachable, so no reconciliation exists. Measuring it is the cheapest confirmation.
- → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E21.

## E. Coverage (tt-mlir / tt-metal) — the biggest prize

### E1. Tracer handler for the mutable-state `ttnn.copy` boundary

Blocks **48 of qwen's 64 layers**, and a linear layer costs ~13× a full-attention layer, so roughly **91 %
of that model's decode time has never been advised on**. qwen B showed the residual/norm/MLP envelope *is*
traceable and only the gated-delta token mixer is not — so the work is scoped.

### E2. Tracer support for `ttnn.sparse_matmul`

Blocks north-mini onA entirely and hides **58–65 % of every gemma-4-26B window**.

### E3. Tracer handler for `paged_fused_update_cache`

Blocks phi-3.5 B's fused-cache share.

### E4. Report the unreachable share as a first-class number

Cells already compute it. Put `reachable_by_advisor` / `total_layers` and the unreachable window share in
`final.json` so a contribution of zero can be read as "nothing to find" vs "couldn't look".

## F0. tt-metal — the one trivial fix worth proposing

**`OPERATION_CATEGORIES` in `tt_perf_report/perf_report.py` is missing four op codes that are live in this
corpus**, so their time lands in the meaningless `Other` bucket — **14.7 % of measured time on average, up to
36.2 %** (phi B). The entire `Other` bucket is these four:

| unclassified op code | share of `Other` | should be |
|---|---|---|
| `SdpaDecodeDeviceOperation` | 41.5 % | `Compute` |
| `ReshapeViewDeviceOperation` | 40.4 % | `TM` |
| `RotaryEmbeddingDeviceOperation` | 12.8 % | `Compute` |
| `NLPCreateQKVHeadsDecodeDeviceOperation` | 5.2 % | `TM` |

**Four strings.** The tool already prints *"Please add to OPERATION_CATEGORIES for proper classification"* for
each, so it is a known gap. With them added the category split is usable directly, without a wrapper.

**Everything else tt-metal-side is a recommendation only, not expected to ship:** tiled-input variants of the
conv/recurrent composites; sharded output for GQA SDPA (gates two cells' top candidate *and* the
`concatenate_heads` fix); `ttnn.sparse_matmul` and mutable-state `ttnn.copy` tracer support.

## F. Experiment design (for the next corpus)

### F5. Invert the screening order: apply the advisor's whole plan first, then ablate ⭐⭐⭐ the single biggest change in this document

**What the skill says today** (`SKILL.md` §4): *"Screen, in the order the reconciliation gives. Each chain as one
unit, one variable per measurement, against the frozen incumbent."* That is build-up from the incumbent, and
`final_ir.mlir` — the only artefact with the complete plan and its shard shapes — is never mentioned.

**Measured, on phi FN, its own harness, fresh process per configuration:**

| configuration | median ms | Δ | differential PCC |
|---|---|---|---|
| incumbent | 0.807535 | — | — |
| **what the cell shipped** | 0.768104 | **−4.88 %** | **1.0** |
| rope as advised | 0.723320 | **−10.43 %** | **1.0** |
| the cell's own best, discarded | 0.700120 | −13.30 % | 0.99999107 |
| **rope as advised + the advised 11-core norm** | **0.663507** | **−17.84 %** | 0.99999107 |

**3.7× what it shipped**, strict non-overlap throughout. And the −10.43 % is at **PCC exactly 1.0** — so the
oracle is not what cost this cell the difference; not trying the advice is.

**Corpus support:** the three best outcomes in the corpus (gemma-4-26B onA −12.98 %, north-mini FN −9.26 %,
phi onA −7.58 %) are the three cells whose *first* candidate was the advisor's placement. The four worst are
cells that never applied it. Correlation, not proof — cells differ in how much defect there was to find — but
phi FN is the measured counterfactual.

**The change:**

```
1. apply_all = every advised placement at once, built from final_ir.mlir
   (it carries the shard shape and the full CoreRangeSet; report.json carries neither -- C5f)
2. if it will not run, remove ONLY the failing item, with an isolated single-op test naming it
3. measure. apply_all is the floor, not the ceiling
4. ablate one advised item at a time. an item whose REMOVAL is faster is a finding about the advisor
   and gets reported as one
5. build up from the incumbent only for what apply_all could not reach
```

**Why it is strictly better, not just different:**

- **It tests the sub-floor chains.** 60 % of the disagreed-on cost corpus-wide is in `below_threshold` chains
  that are individually unmeasurable and collectively obvious. The skill's `aggregate_only` verdict already
  says *"apply the top chains together as one candidate first"* — but only as a fallback. Make it the default.
- **It inverts the burden of proof.** Today a cell that applies nothing reports `no_change` and passes its
  gate. Under apply-all-first it has to produce a measured regression to justify that.
- **It is the only order that generates advisor feedback.** An advised item that is never applied yields no
  evidence about the advisor. Step 4's ablation is exactly the signal this whole effort wants.
- **It costs the same.** phi FN used 12 measurements building up. apply_all plus the ablations that matter is
  the same order of magnitude.

- **Evidence:** [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §7–§9.

### F1. Hold the incumbent constant when comparing arms

Two nominally identical arms of gemma-4-26B differ by **45 %** in control speed — larger than any advisor
contribution measured anywhere. Any cross-arm claim about advisor value is inside that variance. Either fix
the incumbent, or report advisor contribution only *within* an arm.

### F2. Re-run the v1 comparison cells under one tool version

v1 and v2 differ in harness protocol, so v1-vs-v2 deltas conflate the advisor's contribution with the
measurement change. A single-version re-run separates them.

### F3. Make "what the cell could not do" a required field

llama's zero is honest, but its only knob drove the norm *and* the residual chain together — so a norm-only
effect was **not screenable in principle**. That is a different result from "screened and found nothing",
and nothing in the schema distinguishes them.
