# advchal-v3 — mistakes made building v3, and since corrected

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
