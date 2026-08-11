# advchal-v3 — mistakes made building and analysing v3, and since corrected

**Two phases. ERRORS 1–11 were made building the stage; ERRORS 12–17 were made analysing the run**, and the
second set matters more for reading this corpus: five of the six are the same mistake, and **every one was caught
by the user asking "are you sure?", never by me.** Start at §"The pattern the *analysis* phase added" if you want
the calibration rather than the history.

The v2 corpus kept [`ADVCHAL-V2-ANALYST-PITFALLS.md`](https://github.com/tenstorrent/agentic-research/blob/main/shard-advisor-experiments/03-advisor-stage-v2/analysis/ADVCHAL-V2-ANALYST-PITFALLS.md)
— 39 published claims later retracted, grouped by the error pattern that produced them — and called it the
file where its own errors live. This is the v3 equivalent, started on day one rather than at the end, because
by the end nobody remembers which claims were confident and wrong.

Every entry carries the one distinction that matters when a prediction changes:

| label | meaning |
|---|---|
| **ERROR** | the evidence to get it right was in front of me. I misread it, or did not look. |
| **NEW** | it was not knowable without the measurement that produced it. |
| **BOTH** | the framing was available; the mechanism was not. |

Eight of the eleven below are ERROR. That ratio is the useful part: v3's problems so far are mostly not
missing data, they are me reading the v2 corpus the way the v2 analysts warned against — twice in patterns
their own file names.

---

## ERROR 1 — I used a number from the cell the corpus flagged as least trustworthy, as a target

**Claimed.** north-mini `fuse-noadvise` should reach **−11.28 %**, and that became step 4 of the run plan and
the yardstick for the shakedown.

**True.** −1.76 %, measured on an incumbent reproduced to four decimals. The v2 ladder was
`22 → 0.5432, 32 → 0.5184, 64 → 0.5733`; v3 gets `22 → 0.5441` (reproduces to 0.2 %) but **cannot get below
0.5436 anywhere**. So exactly one v2 measurement fails to reproduce — `0.5184` at 32 cores — and it is the one
the −10.23 % headline shipped and the −11.28 % projection extrapolated from. Its neighbour on the same ladder
is fine.

**Why this is an ERROR and not new information.** I wrote *"three independent reasons to treat its numbers as
the weakest in the corpus"* about this cell **in my own step-0 document**, and used its number as a target in
the same document set. The corpus also records that it had no `done` tag and ran a `reconcile.py` differing by
535/256 lines. When the monitor later surfaced the v2 driver's own `CONTAMINATED, do not tag` marker, that was
confirmation, not discovery.

**The check that would have caught it.** Before any v2 number becomes a v3 target: was that cell **tagged**,
and does its **window reproduce** from its own committed artefacts? nmFN fails both. Step 0 already told me the
second one and I did not connect it.

## ERROR 2 — I encoded a per-rung value the same corpus contradicted

**Claimed.** *"Which middle rung you land on is worth about 1 pp"* — in `SKILL.md` — and two headline
predictions built on per-rung deltas (16-vs-32 → −264 µs/model; 44-vs-88 → −375 µs/model).

**True.** 16 / 22 / 32 measured `0.543590 / 0.544064 / 0.544007` — a spread of **0.08 pp**, and 22 and 32 are
inside each other's noise. The threshold finding is confirmed; the *increment* is an order of magnitude
smaller than I wrote.

**ERROR.** The corpus said both *"flat response from ~8 to ~44 cores (0.2 %)"* and *"the middle rung is worth
~1 pp"*. Those contradict. I propagated the larger one into a skill rule and two predictions instead of
resolving it — and the smaller one was the measured statement.

**Check.** When a source states two numbers for one quantity, resolve which is measured before quoting either.

## BOTH 3 — I read a capability gap as a discipline gap

**Claimed.** *"No cell applied the plan as written… nobody tried"* → so requiring it (F5) recovers 3.7×.

**True.** `advised_plan_verbatim` came back `hard_error`: *"No generic final_ir-to-decoder execution bridge
exists."* Decoders expose **policy knobs**; the plan is **per-op memory configs and shard shapes**. Where no
knob exists the placement cannot be expressed at all.

**BOTH.** The framing was available — 15 of 15 cells failed, and the single success was the analyst's own
hand-written `PHI_ROPE_MODE` patch. "Nobody managed it in fifteen tries and the one success needed bespoke
code" supports *could not*, not *did not bother*. I chose the reading that made my fix look bigger. The precise
mechanism is genuinely new.

**Consequence.** F5 is not wrong, it is **bounded by the knob surface**, and v3 must measure that bound rather
than assume it away. See the revised expectations.

## ERROR 4 — I shipped a bucket rule built on a proxy field

**Claimed.** C5c: agreement must compare the memory space, so `Input 0 Memory` decides it.

**True.** Input-0 is the space of the op's **input**; the advisor states a placement for its **output**, and
the profile has no output-memory column — which the tool's own `limitations[]` already said. As a bucket rule
it moved 15 rows and **14 were false**, six of them on the matmul class a sweep is known to lose on.

**ERROR**, and it is v2's Pattern 1 verbatim: *read a summary field instead of the authoritative artefact*.
Caught by step 0, which 21 unit fixtures had not caught.

## ERROR 5 — a computation that could not apply returned a confident value

**Claimed.** `legal_ladder: [1]` for `topk`.

**True.** The shipped win runs `topk` on **110 cores** over an 11×10 sub-core grid. The rule
`(C−1)·⌈W/C⌉ < W` models a width shard over the tile axis — the norm's parallelism, not topk's. A cell
trusting the ladder would never have tried 110, so **my ladder could have suppressed the win it exists to
find**.

**ERROR.** `[1]` reads as *"1 is the only legal value"* and meant *"I do not model this op"*. Emit
`not_modelled` with the reason.

## ERROR 6 — I blamed the environment for my own invocation

**Claimed.** `openai_codex` is missing from every interpreter; the environment has drifted since v2; this is
*the* blocking readiness item. Published to a branch and to the run record.

**True.** multigoal needs that package *"unless `--codex-bin` points at a Codex binary explicitly"*. The
drivers pass it. **I omitted it.** `codex-cli 0.144.4` was present and healthy the whole time.

**ERROR**, and it is v2's Pattern 2 — *"I blamed the tool for my own reconstruction's failure"* — which that
document calls **the worst error in the corpus**, with the prescribed order: your own setup first, the tool
second. I inverted it and published.

## ERROR 7 — I reported success from a step whose exit status I never checked

**Claimed.** In a message to the operator: six worktrees "removed".

**True.** Every `git worktree remove` had just printed `Permission denied`; my `echo` was unconditional. Only
the registrations were dropped; the root-owned directories remained.

**ERROR**, and it is the failure the v3 `SKILL.md` opens with — the rule I had written days earlier.

## ERROR 8 — I wrote the run's own state to the wrong home

**Claimed.** The isolation state and restore script live in `~/skillexp-logs/advchal-v3/`.

**True.** `$HOME` is `/home/ttuser`; the project is `/home/mvasiljevic`. They went to a directory nothing else
uses, and the restore script defaulted to `$HOME/tt-metal`, which does not exist. A restore would have
silently done nothing.

**ERROR.** On a host where the account name and the project name differ, `~` is not the project.

## ERROR 9 — my own confirmation script printed a note where it owed a failure

**Claimed.** `CONFIRMATION PASSED`, with three `traced_dtypes.json … did not parse` notes above it.

**True.** The files were fine; my Python one-liner was malformed. The provenance fields it was meant to
verify — `tracer_matches_checkout`, optimizer drift, host, device users — went unchecked while the script
announced a pass. I verified them by hand afterwards: all correct.

**ERROR.** In the script written to catch exactly this. A missing or unparseable provenance artefact must FAIL.

## ERROR 10 — I asserted an absence without looking everywhere the data lives

**Claimed**, to the operator: the ladder sweep did not happen (one rung), and the chains were never screened.

**True.** The cell measured **16, 22 and 32 on both MoE kinds** with confirmations — a proper sweep. It lives
in `measurements/`, which my gate and my first reading never opened. All 36 chains carry
`not_measurable`, which is a recorded verdict, not a skip.

**ERROR**, and the more expensive kind because I reported it before checking. An absence is a claim about
every place the data could be.

## ERROR 11 — I put the predictions in the tree of the agent measuring them

**Claimed** implicitly, by committing `ADVCHAL-V3-CHANGES.md` to the skill branch.

**True.** The driver builds each cell's tree with `git checkout -B <work> $SKILL_BR` — the whole branch. §6
names each cell and the number it must reach. A cell could have passed by reading its target.

**ERROR.** Caught before any cell ran, and the fix is verified in a live cell tree (0 `skillexp/` files).

---

## What is genuinely new, and therefore not anybody's mistake

- **NEW.** The knob-vs-IR mechanism: decoders expose policy knobs, plans specify per-op memory configs, and
  the gap is structural rather than a lapse.
- **NEW.** The effect size of the cliff class on a clean, reproduced incumbent: **≈1.7 %/layer**, from two
  independent layer kinds agreeing (−1.76 % and −1.77 %).
- **NEW.** That exactly one v2 measurement for nmFN fails to reproduce while its ladder neighbour reproduces
  to 0.2 % — which localises the defect to a single number rather than to the cell's whole method.
- **NEW.** The **flagged-pool capacity metric** (boundary ceiling + cliff pool, as a share of window) is
  predictive: it puts llama-3.1-8B `exp17` at **0.7 %** — the corpus's only exhaustively-verified real zero —
  and qwen `nofuse-noadvise` at **0.2 %**, the cell whose real cost is the 191 ms `retilize` that placement
  cannot reach. Two independent confirmations that it measures something real.
- **NEW.** An absolute PCC oracle is **uninformative for exact-selection ops**: `topk` on 1 core and on 110
  scored identically to sixteen significant figures, because top-k selection has no reassociation to perturb.
  Not a false pass — a check with no discriminating power, which the skill should say rather than let a cell
  read as strong evidence.

## The pattern behind my eight

Six of the eight are one shape: **a lookup or a claim that failed and returned something comfortable** —
`[1]` for an unmodelled op, a note instead of a failure, an unconditional success echo, an absence asserted
from one directory, a proxy field standing in for the real one, an environment blamed for a missing flag.

That is the same sentence the v2 corpus put at the top of `FUTURE-RUNS.md`, and I wrote it into v3's
`SKILL.md` as the discipline the *cells* must follow. It applies to the person writing the stage at least as
much as to the agent running it.

---

# The root cause, one level below the eleven

The eleven above are individually cheap to fix, and fixing them individually would miss the point. They were
not eleven independent lapses.

## The errors all pointed the same way

| what I did | which way it pointed |
|---|---|
| read *"no cell applied the plan"* as a discipline problem rather than a capability one | made F5 worth **3.7×** |
| took *"the middle rung is worth ~1 pp"* over the same corpus's measured flat 8→44 response | made the ladder sweep worth ~1 pp instead of ~0.1 |
| used nmFN's −11.28 %, from the cell I had myself called least trustworthy | gave the run a headline target |
| generalised the ladder rule from a single tensor width | made it a general capability |
| shipped C5c as a bucket rule without checking the rows it moved | made the fix decisive rather than advisory |

**Five for five, every one inflating the apparent value of the work.** I checked what would make the change
look bigger and skipped the checks that would have made it look smaller. Random carelessness scatters; this
has a sign, and the sign is the diagnosis. It is motivated reasoning, and the specific tell is that
`0.5184` — the outlier on nmFN's ladder, and the number the whole −11 % target rested on — was the one rung
nobody sanity-checked, because it pointed the way the work wanted to go.

## The stance that produced it

I was in **solution mode from the moment I stopped reading.** `IMPROVEMENTS.md` hands you a starred to-do
list; it *reads* like a specification. I converted it into a work breakdown and started editing files. Every
process failure follows from that stance:

- design-then-verify is what you do when you think the answer is known and needs implementing;
- predictions written from the same document as the fix is what you do when predictions are documentation
  rather than experiments — such a prediction **cannot falsify the fix, because it inherits its errors**;
- adopting their ledger as the work breakdown is what you do when the job is delivery rather than inquiry,
  and it silently imports their **causal model** including where it mis-attributes;
- tests authored from my own model of the change is what you do when tests are regression protection rather
  than falsification. Twenty-one fixtures passed while C5c was 14-of-15 wrong.

Underneath all of it: **the corpus establishes its problems with 149 measurements and proposes its remedies
with almost none — and says so, quoting its own 1-in-6 refutation rate. I inherited the confidence of the
diagnosis and spent it on the prescription.** Those are different epistemic objects. `ANALYST-PITFALLS` is a
warning about the reliability of every neighbouring document; I mined it for content and ignored what it was
telling me about the ledger I was implementing.

The operational tell is a question I never asked. For each of ~20 changes I asked *"does this address the
defect?"* I never asked *"what would this look like if it were wrong, and can I check that now?"* Asked
twenty times that is about an hour of work, and it catches all four bad changes.

## What should have been done differently — at the level of thinking, not of code

1. **Separate "what is true" from "what to do", with a hard gate between them.** No remedy written until the
   finding is re-derived from primary data. The corpus replay was the right instrument in the wrong slot: it
   belonged *before* the design, where it would have killed three changes for free, not after, where it
   caught one and the hardware caught the rest.
2. **Write the falsification before the implementation.** Not a unit test of intent — a check whose expected
   value comes from data you did not produce. A change whose wrongness you cannot cheaply describe is a guess
   wearing a citation.
3. **When the remedy is not derivable from the data, instrument instead of fixing.** This is the most useful
   rule to come out of v3 and it was found by accident: C5c became a `space_hint` rather than a bucket rule,
   which keeps the finding, adds no false positives, and leaves the judgement with whoever holds the IR. It
   should have been the default for at least four changes — and it is the structural antidote to the bias
   above, because instrumenting does not let you claim a win.
4. **Attack mechanisms, not attributions.** Ask what made an outcome *possible*, never what the previous
   analyst blamed. The oracle and the missing measurements-vs-decision reconciliation are two doors onto the
   same three numbers in phi FN's data. v3 closed the one the ledger pointed at, and the identical failure
   walked through the other on the first cell.
5. **Distrust a number in proportion to how convenient it is.** The convenient ones are exactly where the
   scepticism went missing.

## The change ledger this implies

Adopted for the remaining action points, and as a re-audit of those already implemented. One row per
proposed change, and **no row means no change**:

| column | why it exists |
|---|---|
| the finding, restated | forces separating finding from remedy |
| the primary artefact it was **re-derived** from, with the number | the column that was empty for all four bad changes |
| how this could be wrong | the question never asked |
| the falsification, and its expected value **from data I did not produce** | a fixture written from intent cannot fail correctly |
| if the remedy is not derivable: what is instrumented instead | makes "measure it" the cheap default, not the fallback |
| which direction the error points if wrong | surfaces the bias while it is still cheap |

The last column is the one a reviewer should read first. If every row points the same way, stop.

---

# The pattern the 11-cell run added: I specified rules by what they should catch

Three of v3's defects are the same mistake, and I made the third one **as the fix for the second**:

| # | rule | applied to | holds over | cost |
|---|---|---|---|---|
| 1 | C5c: agreement must match the memory space | every advised/shipped pair | nothing — the profile has no output space | 14 of 15 rows false |
| 2 | the legal ladder | every cliff candidate | only shard advice | `topk` got `[1]`, shipped on 110 |
| 3 | a measurement faster than `final_ms` must ship or explain | all measurements vs **one global** `final_ms` | per layer kind | 12 false positives on one cell |
| 4 | the oracle's clause 2: "no worse than the incumbent" | every candidate | nothing that re-grids a reduction | **−0.90 % lost on phiA, vetoed at a PCC gap of 1.2 × 10⁻⁷** |

**Every one of these I specified by what it should catch, and never by what it would wrongly catch.** Clause 2
was written as "don't ship something worse". Its false-positive surface is *every reduction re-grid* — the
entire class this stage exists to find — and I never enumerated it. C5c, the ladder and #3 are the same
omission.

> The missing question, asked of each rule **before** shipping it: **"what will this reject that I want
> kept?"** Asked four times it catches all four, in minutes. It is not the same question as *"does this address
> the defect?"*, which is the only one I asked.

**And #4 is worse than an oversight, because it reproduces the exact failure it was written to fix.** v2's
complaint was that a differential oracle vetoes anything that perturbs the arithmetic. My clause 2 is a
differential oracle with an absolute reference bolted on. I took the sentence verbatim from `IMPROVEMENTS` A1
without asking whether the class of change under test can satisfy it.

## And the bias flipped sign rather than disappearing

The first five errors all **inflated** the value of my work. After that was named, my revised corpus estimate
came out **7× too low** — 1 ms predicted against 6.8 ms measured, while v2's original 9.2 ms was closer than my
correction. The mechanism: I calibrated a corpus-wide multiplier (12.5 % realised fraction) on the shakedown, a
run whose decision defect I had **already documented in the same session**. A calibration constant inherits
every defect of the run it was calibrated on.

**So naming a bias does not remove it. It moved from over-claiming to over-correcting, which is the same
failure with better manners** — and it is harder to spot, because under-claiming reads as rigour.

---

# The pattern the *analysis* phase added: I falsified mechanisms in a reconstruction and asserted about the system

ERRORS 1–11 were made while **building** v3. ERRORS 12–17 were made while **analysing** it, over a single
session, and five of the six are one mistake: **I rebuilt a piece of the system in isolation, got a clean
negative, and stated a conclusion about the real system that the reconstruction was not entitled to support.**

Every one was caught by the user asking a version of *"are you sure?"* — never by me. That is the signal to
weigh: **the questions that overturned my conclusions cost the user one sentence each, and cost me between four
minutes and four hours of measurement to answer.** If a single question can overturn a published claim, the claim
was not ready to publish.

## ERROR 12 — I called one observation reported three times "three independent reproductions"

Published a candidate tt-metal defect on the strength of gemma-4-26B's `rms_norm` scoring ~0.9946 in three cells
"that never saw each other's artefacts". They had not — **but all three ran the same candidate policy**, so it is
one measurement of one configuration, replicated by construction. Independence is a property of the *inputs*, not
of the *filesystem*. **Check what the observations share before counting them.**

## ERROR 13 — I inferred "it stopped searching" from absent evidence rather than reading all of it

Wrote that the cell *"tried one rung and stopped searching that kind"* from the `not_retested` strings. The
ladder had in fact been **fully swept — 17 measurements**; the defect was that **one rung's verdict was
extrapolated to sixteen untested ones**. Reading every measurement file first would have given the sharper
finding immediately. Same shape as ERROR 10.

## ERROR 14 — I moved a number between cells

Put phiB's `0.998993` into gemma-onA's row. The correct value, `0.9996293`, was in a file I had already opened.
**A PCC belongs to a (cell, kind, config, oracle-scope) tuple; carrying the value without the tuple is how it
lands in the wrong row.**

## ERROR 15 — "provably cannot" from a reconstruction that held the real variable fixed

Ran 79 isolated configurations of the norm, found the op grid-insensitive to 7.3 × 10⁻⁷, and concluded the op
**"provably cannot"** have caused the layer's 5.06 × 10⁻³ drop — *"6,879× too small"*.

**The reconstruction used the decode-shaped input, `[1,1,1,2816]`: one real row and thirty-one of padding. The
real layer also runs the norm in *prefill*, on `[1,32,2816]` — thirty-two real rows.** That case was never
tested, and **that is exactly where the effect lives.** The isolated result was correct and the conclusion drawn
from it was not.

> **An isolated reconstruction can only falsify a mechanism under the conditions it reproduces.** Before
> concluding, list what the real system varies that the reconstruction holds fixed — here: input shape, execution
> phase, which of the eight call sites, weight present or absent — and say which of those the negative covers.
> I had *four* held-fixed variables and mentioned none.

## ERROR 16 — "the only reading consistent with the numbers", when running the other tree took four minutes

From v2's oracle file matching v3's *incumbent* to 1.3 × 10⁻⁶, I concluded that **v2's oracle had run with the
sharding inactive**, that its 88-core number was "an incumbent-grade measurement filed under two names", and
recommended **striking v2's −5,919 µs/model win** — 47 % of v2's corpus total. I published it.

**Then I checked out v2's tree and ran it. 88 cores reproduces `0.9996293363224806` exactly, with the sharding
active and engaged.** The oracle had exercised the change. The proximity to the incumbent was not evidence of
inactivity — **it was the finding: at one tile per core the sharded reduction is genuinely as accurate as the
interleaved one.** And 11/22/44 fail in v2's tree too, so the ladder is **non-monotonic**, which is what I had
originally guessed and then talked myself out of on the strength of ERROR 15.

Two things to keep from this:

- **"The only reading consistent with the numbers" is a phrase that means I have stopped looking for readings.**
  When it appears, the next action is to enumerate one more reading, not to publish.
- **The experiment that settled it — check out the other version and run its own test — cost four minutes.**
  I had already built the worktree tooling to do it. I inferred where I could have measured, and the inference
  was wrong in the direction that made my analysis more interesting.

## ERROR 17 — I did not update this file as I went

Asked directly whether I was recording each confident-and-wrong finding here. **I was not.** I was correcting
`RESULTS`, `DEVIATIONS` and `PCC-BY-GRID` in place — so the *conclusions* were right by the end, and the
**record of how often my confident conclusions were wrong was missing entirely.** Which is the one thing this
file exists to carry, and the one thing a reader needs in order to calibrate the rest of the corpus.

That is not a clerical slip. **Correcting a claim in place removes the evidence that the claim was made.** Six
retractions in one session is itself the most decision-relevant fact the session produced, and it was the only
fact not being written down.

> **A retraction goes in two places: the document that carried the claim, and this one.** The first keeps the
> corpus correct; the second keeps its author's confidence auditable. Doing only the first makes the corpus look
> like it was right all along.

## The corrected finding these six produced

For the record, because it is the actual result and it took all six errors to reach:

| sliding cores | v2 tree | v3 tree + v2-style guard | v3 tree as shipped (decode-only) |
|---:|---:|---:|---:|
| 0 — interleaved | 0.9996280142258483 ✅ | — | 0.9996280142258483 ✅ |
| 11 | 0.9943331194625922 ❌ | **0.9943331194625922** ❌ | 0.9945729603715616 ❌ |
| 22 | 0.9942874693564726 ❌ | **0.9942874693564726** ❌ | 0.9944099795374435 ❌ |
| 44 | 0.9941146130802025 ❌ | **0.9941146130802025** ❌ | 0.9945729603715616 ❌ |
| **88** | **0.9996293363224806 ✅** | **0.9996293363224806 ✅** | 0.9943716809625597 ❌ |

**The two trees are numerically identical once the guard matches — to sixteen digits, at every rung.** The
weight-resharding difference I chased is irrelevant. The single difference that decides the cell is that
**v2 shards the norm in prefill and decode, v3 shards it in decode only** — so v3 builds its KV cache with
interleaved norms and then reads it with sharded ones. That inconsistency costs ~5 × 10⁻³ of layer PCC at every
grid, and at 88 cores it converts a **pass** into a **fail**.

**So v2's win is real and correctness-established, and v3 lost it to two independent defects of its own:** 88 was
missing from its ladder, *and* its decode-only guard would have failed 88 even if the rung had been there.

## ERROR 18 — the oracle's one fixed test point was the confound, twice in one session

Recommended "ship 88 cores with a phase-consistent guard (v2's condition)" on the strength of a measurement at the
oracle's prefill length, **32**. Sweeping the length showed **v2's guard only fires when prefill ≤ 32 rows** — at
seq 64 `phase=both` returns the decode-only number to sixteen digits. So the recommended fix does not fix
production, and **v2's original passing result does not generalise either**, for the same reason.

Also predicted, from the mechanism, that a *shorter* prefill would make the mismatch *worse* (the newly-appended
inconsistent entry holds a larger share of the attention). The opposite: 9.4 × 10⁻⁵ at seq 4 versus
**5.3 × 10⁻³ at seq 32**.

**This is ERROR 15 again, on the same variable class.** There I concluded from 79 configurations that all held the
input *shape* fixed at the decode case. Here I concluded from four configurations that all held the prefill
*length* fixed at 32. Both times the held-fixed variable was the one that mattered, and both times the fixed value
came from **the harness I was measuring with** rather than from anything about the question.

> **A test fixture's constants are not neutral.** `seq_len = 32`, `layer_idx = 0`, `batch = 1`, one input shape —
> each is a choice someone made for a different purpose, and each becomes an unstated premise of every conclusion
> drawn through it. Before generalising: **which constants of the harness did my conclusion inherit, and which of
> them does the real system vary?** Sweeping the one that mattered cost eight minutes, twice.

And the corollary that would have caught both: **when a configuration passes by a margin that looks lucky, vary
the harness before believing it.** v2's 88-core config passes at exactly the prefill length where its guard fires.
That coincidence was visible in the guard's source the whole time.

## The pattern behind ERRORS 15 and 18, and why it is the same one as the run's own defect

ERROR 15: concluded from 79 configurations that all fixed the input **shape** at the decode case. ERROR 18:
concluded from four that all fixed the prefill **length** at 32. Both times the fixed value came from the harness I
was measuring with.

**The stage made the identical mistake, in code rather than in prose.** Its agents wrote decode-only knobs because
its harness measures decode; its gates passed them because each oracle fixes a prefill length — and north-mini's
fixes it at zero. [`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md).

> So the analyst's pitfall and the stage's defect are one pitfall: **a measurement apparatus silently supplies
> premises, and both the thing measured and the person reading it inherit them.** The fix is the same in both
> places — *name what the apparatus holds fixed, and check whether the deployed system varies it* — and it is
> cheap: sweeping the one constant that mattered cost eight minutes; not sweeping it cost this corpus 5,919 µs and
> me six retractions.
