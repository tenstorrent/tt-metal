# advchal-v2 — what to change

Derived from the 15-cell corpus, the tt-mlir source at the pin, and five hardware experiments.
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
- **Evidence:** [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E1, [`ORACLES`](ADVCHAL-V2-ORACLES.md).

### A2. Ship the three wins the corpus already has

- **phi FN**: `advisor_rope_l1="query_key"` + `advisor_norm_cores=11`. Measured −13.39 %/layer; oracle
  under A1 passes.
- **north-mini FN**: `advisor_moe_norm_cores=16` instead of 32. Measured −5.4 µs/layer (sliding MoE) and
  −5.7 µs/layer (full MoE), ≈ −264 µs/model; PCC 0.99951 vs the model's 0.995 bar.
- **gemma-4-26B onA**: `GEMMA4_ADVISOR_NORM_CORES=44` instead of 88. Measured −12.2 µs/layer (sliding) and
  −12.4 µs/layer (full), ≈ −375 µs/model; **bit-identical** to the shipped 88 on sliding (PCC 1.0), so it
  inherits the real-weight oracle already passed. Also **relax that cell's own `cores % 11 == 0` check** —
  it is the cell's assumption, not the hardware's; north-mini ran 16- and 32-core norms on the same device.
- **Test:** the existing per-model real-weight tests, unchanged, plus a fresh-process non-overlap
  confirmation.

## B. The stage — search

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

### C2. Enforce control-plus-one-knob

Require a recorded diff of executed policy vs frozen policy showing **exactly one changed field**, and gate
on it. The only cell that checked this found it *false* and had to remeasure six candidates and two
confirmations.

- **Files:** `harness_template.py` (record the diff), `check.sh` (fail on >1 changed field).

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

### C5. Record `agreed_on: grid | ds_family`

An `agrees_with_shipped` row can carry a different advised core count, because a DRAM-sharded match counts as
agreement regardless of grid (advisor score level 3 beats core count). Without the field the data invites
misreading — it misled this analysis once.

### C6. Keep the decision trace, compressed, as a deliverable

It is the only artifact that answers "why this grid". One cell committed it **uncompressed at 51 MB**;
another gzipped 118 MB to 7 MB. Require gzip, and `.gitignore` the raw profiler dumps that two cells had to
delete by hand after a 1.2 GB push rejection.

## D. The advisor (tt-mlir)

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

### D4. Emit the advisable ladder in `report.json`

The set of legal core counts per op is computable inside the advisor and is exactly what a challenger needs
(B1). Emitting it removes the guessing game and the wasted device time.

### D5. Answer the open question in the decision trace

For llama-8B's MLP norm, 32 and 64 cores were valid candidates that outrank the chosen 22 on both
documented tiebreakers (`coreCount`, `outputL1Usage`), and 22 matches neither neighbour. **The trace does
not explain the selection.** Either the recorded per-evaluation score is not what is compared, or the beam
applies an unrecorded criterion. Add the comparison actually used to the trace.

- **Evidence:** [`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) §5.

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

## F. Experiment design (for the next corpus)

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
