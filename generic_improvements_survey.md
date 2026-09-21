# Generic improvements to the operation-generation framework

**Compiled 2026-09-21.** One enumerated list, ranked by leverage.

## What this list is

There is a pipeline of AI agents that writes GPU-kernel operations ("TTNN operations") for
Tenstorrent hardware. It lives in a git submodule called `tt_ops_code_gen`, checked out at
`/localdev/dnijemcevic/tt-metal/tt_metal/third_party/tt_ops_code_gen` and symlinked as `.claude`
inside each working copy. Its parts are: agent prompts (`agents/*.md`), shared prompt fragments
(`prompts/*.txt`), user-invocable skills (`skills/*/SKILL.md`, notably `golden-tests`, which writes
each operation's test suite), reference documents (`references/*.md`), and the harness that runs
and scores everything (`eval/*.py`, `eval/eval_test_runner.sh`). Every path below is relative to
that directory unless it begins with `/`.

Over several weeks an AI-generated replacement for the `rms_norm` operation was built and measured
against the hand-written operation it replaces. It exposed many defects. **This list contains only
the defects that would bite an operation generated from scratch, with no existing implementation to
start from and nothing to compare against.** Anything that only matters because that experiment
started from a designated existing implementation, or compared two implementations of the same
operation, or moved code between git branches, has been left out — those are being handled
separately. Where an excluded finding had a generic core inside it, the core is included and the
entry says what was stripped off.

The six items the engineer named are numbers **1, 2, 4, 5, 7, 8 and 10** below; each says so, and
each has been verified against the sources rather than restated.

**Sources.** `/localdev/dnijemcevic/tt-metal/native_parity_checkpoint.md` (2855 lines; the primary
record — sections are cited as "journal §4i.7" and so on); `/localdev/dnijemcevic/2026_09_10_port/CHECKPOINT.md`
(the branch-port and continuous-integration record, cited as "port §13"); the framework's own commit
history on branch `dnijemcevic/codegen_new` (tip `aada63b`); the operation's commit history in
`/localdev/dnijemcevic/tt-metal3`; and the framework's current files, read to check whether an item
is already handled.

---

## Why the top of the list is at the top

The first six items share one property: each is a **small, local, already-half-built change** that
closes a defect class the whole pipeline is currently blind to.

Items 1 and 2 are the same hardware defect from two sides. A check that would catch it *already
runs on every kernel exit* and is one identifier away from working; the agent-side rule that would
prevent it is two sentences. Together they convert a defect that today needs a whole language model
running end to end to detect into one that a twenty-line test catches. Nothing else on this list has
that ratio.

Item 3 is the only item on the list that produced a wrong answer in a real model while every
isolated correctness check passed. The fix is a design rule plus one corrected paragraph in a
reference document that currently tells the planner the opposite.

Items 4, 5 and 6 are each a single missing thing in the test-suite generator — one enum value, one
placement axis, one gate — and each one hid a real defect: 73 wrong-answer cases, a 1.69× speed
deficit, and 152 cases that passed while proving nothing.

Below that the list runs roughly: coverage the generator never produces, then measurement practices
that produce misleading numbers, then things a reference document states wrongly, then method rules,
then small harness plumbing.

---

# The list

## 1. The end-of-kernel safety check tests the wrong counter

**(Engineer's item 1 — confirmed.)** When a kernel finishes on a core, the firmware runs a check
that is supposed to catch the kernel exiting while its memory writes are still in flight. This
matters because each processor keeps a software count of the write acknowledgements it expects, and
that count is seeded by *snapshotting* the hardware register when a kernel starts — so late
acknowledgements from a departed kernel push the register past the next kernel's snapshot, and the
next kernel's wait-for-completion, which is an exact-equality test, can never be satisfied again.
The core is permanently desynchronised. The check exists, and its error text reads *"detected an
inter-kernel data race due to kernel completing with pending NOC transactions"* — exactly this
failure — but the line tests `ncrisc_noc_nonposted_writes_sent`, i.e. writes **issued**, while its
four sibling checks on the same lines test reads and atomics **completed**. Changing that one call
to the completed form (`ncrisc_noc_nonposted_writes_flushed`) makes a twenty-line test — eight
iterations of the operation followed by a matrix multiply on a 32-core resident tensor, no model, no
trace — fire immediately at the exact core, processor and kernel that the production hang occurred
on; with the `sent` form the identical test passes silently.

* **Cost in this experiment:** a reproducible hang in a 32-sample Llama 3.1-8B evaluation, which took
  days to attribute and was initially misread as a network-fabric fault. 416 reference cases, 390
  post-commit cases, 140 production-traced cases and ~23,500 generated cases all passed with the
  defect live; only the full model found it.
* **Evidence:** journal §4k.11 and §4k.12; port §18 ("The decisive experiment").
* **Changes:** `/localdev/dnijemcevic/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc:92` and
  `.../ncrisck.cc:86`. **Outside this framework** — file it as a tt-metal issue.
* **Size:** one identifier per file.

## 2. No agent is told that a kernel must not finish with writes still in flight

Data-movement kernels legitimately use a cheap "the data has departed" fence inside a loop, rather
than the expensive "the data has been acknowledged" barrier. That is sound between loop iterations
and was measured worth about 3%. It is **never** sufficient at kernel exit, and no prompt says so.
The generated operation relied on a trailing write to supply a real acknowledged barrier; on the
plan where input and output are both resident on the core, that trailing write is a no-op, so
nothing was ever acknowledged and the kernel exited dirty. The rule to add is one sentence for the
implementer — *any data-movement kernel that can end on a departed-only fence takes an acknowledged
barrier before returning, reader as well as writer* — plus one concrete question for the
performance agent that optimises a single part of an operation in isolation: *after this change,
what is the last memory operation this kernel can execute on any compiled path, and is it
acknowledged?* That agent's report format already demands it list anything that makes its pattern
incorrect elsewhere; deleting a fence is only correct in the light of what runs on that core
**next**, which optimising in isolation hides by construction.

* **Cost:** the same hang as item 1. The operation's own safety argument was rigorous and correct as
  far as it went — same virtual channel, no reordering, so departure orders the semaphore behind the
  data. It reasoned about the *receiver* and never about the *successor*.
* **Evidence:** journal §4k.11 ("Rules to add"); port §18 ("The mechanism").
* **Changes:** `agents/ttnn-implementer.md`, `agents/perf-part-optimizer.md`; and one line in
  `skills/tune-dm-helper/SKILL.md` + `skills/apply-dm-helper/SKILL.md` requiring any shared helper
  whose completion guarantee is weaker than "acknowledged" to say so in the same breath as its
  buffer-reuse guarantee (the multicast helper used here documents only *"source may be reused when
  send() returns"*, and this operation is its only production caller).
* **Size:** two or three sentences across three files.

## 3. Nothing warns the planner off a 16-bit running sum, and the reference document green-lights it

With the wide-accumulator setting off (which is the default), **every** arithmetic instruction on
the matrix unit lands its result back into a 16-bit destination register — one landing per
instruction, eight per tile per fidelity phase, whatever order the sum is taken in. So "sum it in
the matrix unit" is *not* a wide accumulation, and the only sums that escape narrowing are the
small integer groups inside a single instruction. The generated operation summed the squares of a
row's eight tiles position-by-position in that 16-bit register before summing any columns; the
hand-written operation summed each tile's columns first and then accumulated eight tile totals. Both
are 16-bit accumulations and both are biased; they are biased on *different* data. On
transformer activation rows, which are heavy-tailed (a few values 8–16× the median), the generated
order lost the small squares behind a large one at the same position. The design rule to add: **no
running sum of squares over activations in a 16-bit accumulator, in either order — accumulate across
tiles in 32 bits**, either in the wide destination register or by carrying a 32-bit partial through
an intermediate buffer. Reordering is not a fix.

Separately, the reference document that the planner consults on this subject is actively misleading.
`references/numerical_stability_analysis_reference.md` §3.1 gives a severity guide keyed on the
**number of terms**: *"Accumulation of ≤32 terms in bf16: negligible error"*. The failure here was
**eight** terms. What decides the error is the dynamic range inside the accumulator, not the term
count. The same section never states that the destination register narrows per instruction when the
wide setting is off, and never states the narrowing's measured rounding rule — ties round **away
from zero**, and an increment smaller than half a step is **dropped** (measured on hardware, and
matched by the functional simulator's model of the unit; the instruction-set document leaves it
unspecified, and it is neither round-to-nearest-even nor truncation, so an agent reasoning from
memory will get it wrong).

* **Cost:** Gemma-4-12B whole-model correlation **0.897** with the generated operation against
  **0.973** with the hand-written one — the only genuine red in a twelve-job continuous-integration
  sweep. Forcing the other summation order made the output bit-identical to the hand-written
  operation on all six captured inputs and took the model to 0.9736. The generated operation is, on
  paired measurements of 8192 rows per cell across four input families and five widths, the
  *statistically more accurate* of the two (p ≪ 1e-10) — which is precisely why no isolated
  accuracy metric could catch this.
* **Evidence:** journal §4m.6 (bisection and bit-identity), §4m.7 (the instruction-set reading and
  the Monte Carlo), §4m.9 (the simulator's model of the arithmetic), §4m.10 (an independent silicon
  probe from unrelated work reaching the same rounding rule).
* **Changes:** `references/blocking-model.md` (a precision clause in the blocking model: which axis
  the accumulator runs along, and its width); `references/numerical_stability_analysis_reference.md`
  §3.1 (replace the term-count severity guide with a dynamic-range one; add the per-instruction
  narrowing and its measured rounding rule); `agents/incremental-planner.md`.
* **Size:** a new short section in one reference document, a corrected paragraph in another, and a
  checklist line in the planner.

## 4. Tensors are never spread across cores in column-major order

**(Engineer's item 2 — confirmed.)** When a tensor is spread across a grid of cores, its description
carries an **orientation**: walk the cores across rows, or down columns. The test-suite generator
never produces a column-major case. The property is reachable only through a per-case override that
nothing sets, and the shared helper that builds a spread tensor defaults to row-major, so **zero of
~23,300 generated cases could have hit it**. The generated operation read that field in two places,
both of which only compared two descriptions for equality; it never used it to decide which core
gets which piece. Every core therefore normalised a slice belonging to someone else, and the
cross-core statistic broadcast went to the wrong family of cores. The rule to add is general: **a
property named in the type that describes how a tensor is spread is an axis.** If the platform's own
type carries it, the suite sweeps it.

* **Cost:** 73 failing cases — the single biggest defect found, and found only by running the
  hand-written operation's own tests.
* **Evidence:** journal §4i.7; the general rule at journal §4i "What to change", item 6.
* **Changes:** `skills/golden-tests/SKILL.md` (the sharded-placement section, currently around line
  116, mentions `ShardOrientation` only as something an operation may *pin*, never as something to
  sweep); the feature-specification template.
* **Size:** one axis value plus the rule sentence. Cheap, because orientation crosses with placement
  only — it does not multiply the whole test matrix.

## 5. Where the input lives — main memory or fast on-chip memory — is not an axis, in correctness or in performance

**(Engineer's item 4 — confirmed, and it is wider than the performance half.)** A tensor can sit in
main memory, or in the fast on-chip memory, either spread across cores or interleaved across them.
Those are **three** dataflow regimes, not two. The registry axes carry the layout (interleaved vs
spread) but not the memory kind, and the worked feature specification has **zero** occurrences of
it. The test-suite generator mentions it once, parenthetically, as an *optional* sub-split
(`skills/golden-tests/SKILL.md:98`). So no correctness cell and no performance case could ever
distinguish a main-memory-resident input from an on-chip-resident one. A spread on-chip tensor does
**not** cover the gap: with a spread tensor each core reads its own slice, so the many-to-one read
fan-out that causes the problem never occurs — which is why all 250 spread cases were clean. The
requirement: **every operation that reads an activation over the network needs at least one
on-chip-interleaved correctness cell and at least one on-chip-interleaved performance case.**

A trap found while checking this, worth writing down beside the rule: in the platform's own upstream
tests, 18 of the 24 cases whose identifier says `in0_L1` are a misnomer — both configurations that
identifier labels are interleaved in *main* memory. **Select on the resolved memory configuration,
never on a test identifier.**

* **Cost:** a case where the hand-written operation is **1.69× faster** survived 416 reference cases
  and ~23,500 generated cases undetected. It surfaced only because one upstream nightly file happens
  to parametrise both placements. Six cases in total were affected; after the fix the worst case
  moved from 0.591× to 0.760× of the hand-written operation.
* **Evidence:** journal §4k.3; port §13 ("Two side findings that matter more than the fix").
* **Changes:** `skills/golden-tests/SKILL.md`, the feature-specification template, and the per-
  operation `axes.py` tagger.
* **Size:** one axis with two values, plus one performance case. Note the diagnosis needed the
  operation's own permanent timing instrumentation to localise — that instrumentation is worth
  keeping as a convention.

## 6. The test suite is allowed to invent pieces the operation is missing

If writing a test case requires inventing a Python class, a constant, or a type conversion that the
operation itself does not provide, the suite has quietly completed the operation's public surface —
and then validated the operation against its own completion. No real caller will do that. This
happened twice, independently, in one run: the suite defined stand-in versions of the two
configuration objects the operation should have exported, and it built one of the inputs in the one
physical spelling the operation happened to accept. Both times the suite went green while proving
nothing. It then happened a **third** time when the missing type was supplied by hand: the real
platform type accepts its grid argument either as a proper coordinate object or as a plain pair of
numbers, because its binding converts automatically, and the hand-written stand-in did not. That is
the sharpest available argument against a stand-in — the person who copied the fields still missed
the behaviour. The rule: **a stand-in is a gate failure, not a fixture.** When the generator finds
it needs one, it reports a missing piece of the operation's public surface and stops.

* **Cost:** 32 dedicated input-format cases, all green, all wrong; 120 further cases lost to the
  hand-written stand-in's own divergence. Three days of refinement and two performance tournaments
  were built on top of an operation no real caller could call.
* **Evidence:** journal §4i.2, §4i.3, §4g "Remaining" item 8, §4e TODO-7.
* **Changes:** `skills/golden-tests/SKILL.md`; and a mechanical check in the harness that the
  suite's modules import the operation's exported types rather than defining look-alikes.
* **Size:** a rule paragraph plus a small static check.

## 7. The poison-filled-padding detector is pinned to the one format where its bug cannot happen

**(Engineer's item 6 — confirmed; the pin is at `eval/golden_tests/rms_norm_ttnn/feature_spec.py:572`,
spread into each case around line 637.)** When a tensor's width is not a multiple of the hardware's
32-element tile, the last tile is partly padding. A group of test cases deliberately fills that
padding with a large poison value (1000.0) and uses deliberately *narrow* widths, so a leak is
catastrophic rather than marginal; the group's own comment says it exists because the correlation
check the suite normally uses is blind to a near-uniform scale error. Right shape, right detector —
and it is pinned to a single format corner via a shared base dictionary (`dtype=bfloat16,
fp32_dest_acc_en=False`), which is the one format where the bug it was built to catch cannot occur.
The bug in question: a mask buffer allocated 16-bit unconditionally while the reduce helper
reconfigures both of its inputs from the *reduce's* input buffer, so the mask is read in whatever
format the intermediates carry. At 16-bit intermediates the two agree by coincidence; at 32-bit the
lane pitch halves, the mask's ones land in the wrong places, some real data is zeroed and some
padding is counted. The general rule: **a detector built to cover a specific blindness is crossed
with every format in which that blindness can occur**, and where a group is pinned for cost, the
comment records what the pin *gives up*, not only what it saves. (The current comment says "so each
shape costs 4 cells, one per placement" and stops there.)

* **Cost:** the 32-bit non-aligned cases **ran and passed**. A mis-shifted mask produces a
  near-uniform scale error, the correlation check is blind to that by construction, and the relative
  error tolerance at 32 bits absorbed the rest. Crossing the group with the format would have cost a
  handful of cells.
* **Evidence:** journal §4i.9 (defect 9, at line 1236 onward of the journal) and journal §4i "What
  to change", item 5.
* **Changes:** `skills/golden-tests/SKILL.md` (the rule, so future suites inherit it — the poison
  pattern is currently invented in one operation's specification and appears nowhere in the
  generator); `eval/golden_tests/rms_norm_ttnn/feature_spec.py`.
* **Size:** cross one loose-case group with one more axis value, plus a rule sentence.

## 8. There is no test where the grid of cores is larger than the work

**(Engineer's item 5 — confirmed.)** The suite always sizes the grid to fit the work, so the geometry
where some cores get nothing to do has never been generated. It crashed on first contact: each
core's extent was computed as "my share, or whatever is left, whichever is smaller", and for the
leftover cores "whatever is left" went **negative**. That number is handed to the hardware as an
unsigned value, so negative became enormous and the call crashed. The fix in the operation was to
clamp at zero — a spare core owns zero units of work, never a negative count. Notably, a neighbouring
code path one screen above already had exactly that guard, so this is a defect class the platform
invites, and there is a second-order rule for the implementer: **when one path guards an edge, every
sibling path computing the same quantity guards it too.**

* **Cost:** a hard crash on first contact with the geometry. Zero of the generated cases could reach
  it.
* **Evidence:** journal §4i.6; journal §4i "What to change", item 7.
* **Changes:** `skills/golden-tests/SKILL.md` (add a grid-larger-than-work case to the placement
  section — there is currently no mention of it anywhere in the framework); `agents/ttnn-implementer.md`.
* **Size:** one new kind of test case and one implementer sentence.

## 9. Performance claims are reported as an aggregate, which cancels the losses against the wins

The implementer measured its change on ten cells, got 0.996× overall, and reported "net flat". The
full sweep of 23,458 cases gives the same aggregate — 0.9965× — and **1,324 cases more than 5%
slower** against 177 faster, worst case 0.519×. The losses and the wins cancelled in the sum. The
rule: **a performance claim reports how many cases got slower, how many faster, and the worst one —
never a total or a mean on its own.** Nothing in the performance agents' report formats currently
demands a distribution; the coordinator's ranking rule is about *choosing* a case, not about
*reporting* an outcome.

* **Cost:** a real one-sided regression (1,152 of the 1,324 were a single format-and-shape family,
  one-sided, caused by a fixed additive cost of about +200–325 ns that the change's own analysis had
  priced at zero) shipped behind a flat headline.
* **Evidence:** journal §4j "golden before/after" and journal §4j item 11.
* **Changes:** `agents/perf-coordinator.md`, `agents/perf-part-optimizer.md`,
  `agents/incremental-verifier.md`, `skills/perf-measure/SKILL.md`.
* **Size:** one required-fields sentence in each of four prompts.

## 10. The full-precision-input-with-narrow-accumulator convention is written down but not enforced, and three prompts still say the opposite

**(Engineer's item 3 — confirmed; partly landed, not finished.)** The convention used to tell every
operation to *refuse* the combination of a 32-bit float input with the 16-bit accumulator, on the
grounds it was "nonsensical and lossy", and then conceded two lines later that it was "legal, just
pointless". It is legal and it is not pointless: the accumulation simply carries fewer bits. Worse,
this is frequently the cell an operation's **own default** configuration produces, so an operation
following the old rule literally ends up refusing its own default. Commit `2ed265d` corrected
`references/precision_convention.md` and put one line in the test-suite generator. What remains:
(a) **three operation prompts still restate the old rule**, and one of them says it in capitals —
`eval/prompts/matmul.txt:117` and `:128` (*"MUST be an op-side EXCLUSION"*),
`eval/prompts/flash_attention.txt:182`, `eval/prompts/rms_norm.txt:39`; they were left deliberately
because there was no evidence about those operations, but a prompt that contradicts the universal
convention will be followed; (b) there is **no gate** — nothing checks that this cell is absent from
an operation's refusal list and present with a relaxed tolerance in the test suite.

* **Cost:** not directly measured here, but the shape of the defect is item 12's shape: a refusal
  hides every scrap of evidence about the code behind it (journal §4g, "Remaining — we do not know",
  item 15: 87 wrong answers were invisible until an unrelated refusal came off).
* **Evidence:** framework commit `2ed265d`; `references/precision_convention.md` ("Supported, not
  rejected"); `skills/golden-tests/SKILL.md:1322`.
* **Changes:** the three operation prompts; a check in `eval/verify_supported.py` or the verifier
  prompt.
* **Size:** three prompt edits and one small check.

## 11. A performance round never diffs the whole suite, so regressions outside the focus shape are invisible

The performance tournament measures a focus case, spreads each winning idea across the operation,
and earns a carve-out only for a correctness impossibility or a *measured* regression. So the
carve-out set is the only thing that can narrow a spread, and any regime it omits is a hazard by
construction. In one run the carve-out set covered **one cell** of the precision × layout × operand
rectangle, and five shapes came out materially slower — the worst going from 184,790 ns to 821,128
ns — and **nobody saw them**, because nothing compared device times across the whole suite before
and after the round. Every case already carries a device time in every phase; the diff is a query,
not a measurement campaign. Requirement: **each round ends with a before/after device-time diff over
every case that ran, reported as a distribution (item 9), and any case more than a threshold slower
is either explained or reverted.**

* **Cost:** five unjudged regressions in one run, the worst 4.4× slower.
* **Evidence:** journal §4b "What run 893 revealed"; issue #176 (how a carve-out set should be built).
* **Changes:** `agents/perf-coordinator.md` and a small query helper in `eval/`.
* **Size:** a prompt requirement plus a reporting script.

## 12. There is no written procedure for the target number each performance case is ranked against

Each performance case carries an "achievable" nanosecond figure, and the tournament ranks cases by
measured time divided by that figure and takes the worst. The ranking rule is now written down
(commit `0a8a442`), but **where the figure itself comes from is not**. The numbers were hand-measured
twice, the first set against a wrong core count and silently wrong for a day, with no script. The
consequence is structural, not cosmetic: the ranking is dominated by **reference quality, not
headroom** — the ratios span 0.06 to 0.997, so a case whose target is 12× off can never be picked
however much time it is wasting. One of the thirteen original targets was also **under-specified**:
it was only achievable with a hand-tuned configuration the operation's own defaults cannot reach,
so it was restated from 25,640 to 28,619 ns. There is a good convention already demonstrated
(commit `ba0b374`: a runnable provenance script landed beside the numbers it produced,
`eval/provenance/rms_norm_ttnn_baseline_measurement.py`) — it applies to every operation and lives in
one operation's tree. Lift it, and add the procedure it should follow: a computed floor taking the
maximum of data movement, compute, and dispatch overhead, because a movement-only floor is 61–100×
off on small decode shapes and divides by roughly zero on resident tensors.

* **Cost:** one wrong target for a day; a permanently skewed tournament ranking; and one of the
  thirteen targets unreachable by construction.
* **Evidence:** journal §4b "The references were real, and one was wrong", and "Two caveats" under
  §4j "The perf rounds"; issue #175.
* **Changes:** `skills/golden-tests/SKILL.md` (which today says **nothing at all** about the
  performance group — see item 13), `skills/perf-ceiling-dm/SKILL.md`, and the provenance convention
  lifted out of one operation's directory.
* **Size:** a new section in the generator skill plus a reusable measurement script.

## 13. The test-suite generator has no instruction about performance cases at all

Underneath items 5, 11 and 12 sits one structural gap worth naming on its own:
`skills/golden-tests/SKILL.md` is 1,378 lines and the word "perf" appears **twice**, both times
about something else. The performance group, its target numbers, and its coverage are authored
entirely by hand in each operation's feature specification and each operation's prompt. That is why
the nine performance cases for this operation covered none of the three things production actually
does. Measured against 140 distinct configurations distilled from 334,159 recorded executions across
22 production models: production is **110 interleaved / 30 width-spread and zero block-spread**
(the hand-picked set included a block-spread case); production sets the wide accumulator **on in 110
of 140** (all thirteen original cases pinned it off); and production passes its per-channel weight
in **row-major layout in main memory, constantly, across every trace** (all thirteen used the tiled
layout) — and the row-major form is **faster**, 65,349 against 76,616 ns on an identical
configuration, so the existing targets were loose by about 17% for the layout production actually
uses. The requirement: **the generator authors the performance group as deliberately as it authors
the correctness cross-product**, covering placement (including item 5's memory kind), the
accumulator setting, operand presence, and input-format variants, each with a provenance-backed
target (item 12).

* **Cost:** a performance set that measured a corner production never visits, for three tournament
  rounds at roughly $373.
* **Evidence:** journal §4b "Production does not look like our perf cases" (and its honest caveat
  that the trace may itself be the flawed input — a trace cannot distinguish latent demand from
  absent demand); framework commit `4d5dbe1`, which widened one operation's cases from 13 to 19 and
  found that one of the six added was the largest prize of a round at 1.49×.
* **Changes:** `skills/golden-tests/SKILL.md`.
* **Size:** a new section — the largest single documentation change on this list.

## 14. The operation's own tests never run with the device safety checks enabled

The platform has a development mode that turns on a network-traffic sanitiser, per-core progress
markers, buffer sanitisation and lightweight in-kernel assertions, including the end-of-kernel check
of item 1. The pipeline's test runner (`eval/eval_test_runner.sh`) never enables it — it injects
three measurement plugins and nothing else, and no environment variable in the file turns the
watcher on. So an entire class of runtime invariant is never checked against the operation's own
suite. The requirement: **run the operation's spread-tensor cases under the development build at
least once per generation.** This is deliberately a *check*, not a workload: three synthetic
back-to-back workloads were built to try to reproduce item 1's hang behaviourally and **all three
passed** — only the real model reproduced it — whereas the invariant check catches it outright and
deterministically. Chase the assertion, not the schedule. One limitation to record in the same
place: the development build **cannot reach model scale** (the watcher build of the failing model
dies at startup with the program exceeding the kernel configuration buffer), which is another reason
the unit-scale check is the one that must work.

* **Cost:** item 1's hang reached production continuous integration.
* **Evidence:** journal §4k.12; port §18.
* **Changes:** `eval/eval_test_runner.sh` and the pipeline's stage list.
* **Size:** one extra stage in the runner.

## 15. A failing test leaves its device memory pinned, and the next tests fail because of it

Under the shared-device fixture, a test case that raises leaves the exception's traceback holding
the frame that called the operation, whose local variables still point at that case's on-device
tensors. Those frames sit in a reference cycle that plain reference counting cannot break, so the
memory stays allocated and collides with the next case's static buffer region — a self-sustaining
cascade across a whole shape group. The release hook that fixes this **exists and is correct**
(`eval/golden_tests/conftest.py`, the `_release_device_tensors_in_traceback` function with a full
explanation in its docstring) — and it is only auto-loaded for tests living *under*
`eval/golden_tests/`. The runner injects its three measurement plugins by name and does not inject
this one, so any suite the pipeline drives from elsewhere runs without it. Force-loaded, the affected
run went from 1,253 contaminated cells out of 5,442 to **zero**.

* **Cost:** 49 reported failures and 22 skips, reproduced byte-identically across two replicates and
  both orderings, **none of them real** — a day lost, plus 24 of 45 memory-infeasibility skips
  mis-charged to tensor geometry by `eval/oom.py` when they were residue.
* **Evidence:** journal §4j "RETRACTED: the 71-cell W=8192 regression"; filed as tt_ops_code_gen#193.
* **Changes:** `eval/eval_test_runner.sh` (inject the hook as a plugin for every suite it drives),
  and `eval/oom.py` (stop attributing residue collisions to geometry).
* **Size:** one plugin injection plus a small attribution fix.
* **Uncertainty:** I confirmed the hook exists and that the runner's three `-p` injections do not
  include it. I did not confirm whether the two fix directions filed on #193 have since been taken on
  the framework's `main` branch.

## 16. The measurement noise band is asserted, not measured

`skills/perf-measure/SKILL.md` states a flat "±2–3% is noise" and tells the agent no number of
repeats turns a delta in that band into a win. On this hardware and this harness that figure is
wrong for a whole class of case. Re-running an **untouched** side between two otherwise identical
runs moved it by −2.02% at the 5th percentile, −0.02% at the median, +1.38% at the 95th — but with
**8 cases more than 5% slower and a worst case of +22.79%**, concentrated in the spread mixed-
precision cases. A separately identified group of 18 cases swings 9–78% run to run on one side while
the other side holds to 0.72% on the same cases; their individual ratios are meaningless and only the
group figure is stable. The rule: **before attributing a per-case performance delta to a change,
re-run an untouched side and publish its spread next to the claim.** It is free — it is the run you
are already doing. And where a class is not reproducible per case, quote it as a group.

* **Cost:** a worst-case figure of +3.68% was read as signal against an instrument whose
  repeatability had never been characterised.
* **Evidence:** journal §4k.15; port §18 ("Calibration that changes how S14's table should be read")
  and port §10 ("One class of per-case ratio is NOT reproducible").
* **Changes:** `skills/perf-measure/SKILL.md` (around the "Measurement discipline" section, currently
  lines 260–300), `agents/perf-coordinator.md`.
* **Size:** a corrected paragraph plus one required procedure.

## 17. Repeated dispatches are unnecessary, and unsafe on an operation that writes into its input

The device timing counter has no warm-up transient — a three-repeat spread measures 1.17% at the
median — so repeats re-measure the same number at N× the cost. Worse, they are **unsafe**: three
dispatches per case corrupted **120 of 162** cases in one nightly suite, because several of those
tests pass an accumulator tensor that the operation adds into, so the second and third dispatches
break the very output the test asserts on. The default must be one dispatch; three-and-median only
when adjudicating someone else's number, and never on an operation that writes into one of its
inputs. There is a free corollary worth stating in the same place: **the suite's own pass count is a
check on the instrumentation** — if it moves when you instrument, the measurement is interfering.
That is exactly how the three-dispatch mistake caught itself.

* **Cost:** 120 of 162 corrupted cases in one measurement run.
* **Evidence:** journal §4k.2; port §10 ("Mistake 2").
* **Changes:** `skills/perf-measure/SKILL.md` (it already says measure once; it does not say repeats
  are *unsafe* on a mutating operation, and nothing in the performance agents says it either),
  `agents/perf-coordinator.md`, `agents/perf-part-optimizer.md`.
* **Size:** two sentences.

## 18. Every generated input is drawn from a normal distribution

All test inputs are `torch.randn` — see `eval/golden_tests/rms_norm_ttnn/helpers.py:238`, `:249`,
`:272`, and the generator's own templates. Real transformer activations are not normal: they have a
few scattered per-token outliers (measured at 3–4 elements per 256 at 8–16× the median), and
separately a small number of fixed channels carrying values two orders of magnitude above the rest,
a phenomenon documented in the literature. Item 3's defect is invisible on normal data and obvious on
heavy-tailed data. The generator should author a small family of **few-hot** input distributions
alongside the existing uniform/small/large magnitude regression tests: scattered per-token outliers,
channel-concentrated outliers, and a couple of fixed massive channels. Paired measurements across
those families at widths from 4 to 64 tiles separated the two implementations at p ≪ 1e-10, while the
normal-distribution control at 4 tiles separated nothing at all (p = 0.83).

* **Cost:** the precision refinements that produced item 3's defect were each gated on normally
  distributed data with a relative-error metric and each found "at least as accurate as the
  baseline".
* **Evidence:** journal §4m.7 (the Monte Carlo families and their results), §4m.8, §4m.10.
* **Changes:** `skills/golden-tests/SKILL.md` (the `test_regression.py` template, currently around
  lines 886–950, which has small-magnitude, large-magnitude and uniform variants and nothing
  heavy-tailed).
* **Size:** three new distribution generators in a template.

## 19. Every accuracy tolerance is a magnitude; nothing measures the direction of the error

The suite's metrics are correlation, relative root-mean-square and maximum error. All three are
blind to a **coherent one-directional shift** applied to a whole row — which is exactly what item 3's
defect is, and exactly what a downstream consumer turned out to be sensitive to. The addition:
**a signed test on the per-row statistic against an exact reference.** Two details make it work.
First, the standard is *unbiased against exact*, not *matches the other implementation* — the
hand-written operation fails this test too, drifting to −2.4 units at 64 tiles of width and +12 at
344. Second, the scale must be read **exactly**, through a witness element (place a known
power-of-two value at one position so the output there *is* the scale, read off bit for bit), never
by fitting a scale through the output: the fit is biased about +0.5 units on rows carrying a
per-channel weight, and its per-row readings differ from the true value by a median of 0.3 units,
which invalidated an entire earlier round of per-row conclusions. Precision refinements — fused
compute paths, alternative accumulation strategies, hand-written vector-unit finishers — must be
gated on this metric plus item 18's inputs, and must require 32-bit accumulation (item 3), not on a
correlation figure.

* **Cost:** a genuine one-sided bias in the per-row scale was a real, previously invisible defect
  that no gate in the pipeline could express.
* **Evidence:** journal §4m.4 item 3, §4m.7 §1 (the fit artefact and its correction),
  §4m.4 item 4; §4m.10's independent cross-check ("report gain and bias next to correlation").
* **Changes:** `eval/metrics.py`, `agents/incremental-verifier.md` (the precision baseline it
  measures), `skills/golden-tests/SKILL.md`.
* **Size:** one new metric and its gate wording.

## 20. Two documents state the reduce helper's scaler format rule wrongly

When a reduced dimension is not a multiple of 32, the operation multiplies by a mask tile of zeros
and ones before summing. Two documents tell the agent to make that buffer 16-bit unconditionally:
`skills/partial-scaler-reduce/SKILL.md:171-173` says the format is *"independent of the input dtype;
bfloat16 is the usual choice"*, and `agents/incremental-planner.md:228` carries an unconditional
design-checklist item, *"Reduce scaler CB uses bfloat16 packed format"*, with no carve-out. Both are
false for the mask case: the reduce helper reconfigures **both** of its unpack inputs from the
reduce's *input* buffer, so the mask is read in whatever format the intermediates carry, and at 32
bits the lane pitch halves and the mask lands in the wrong places. Behind both statements is a
precision argument about the wrong property — that 1.0 is exactly representable in 16 bits, which is
true, and is about the scaler's *value*; it silently licensed the *format* for a mask, whose
correctness depends on lane pitch rather than mantissa bits. Fix both: the scaler/mask format follows
the format the reduce reads, and the documents say why.

* **Cost:** a live wrong-answer defect from the operation's first commit, invisible to ~23,300 cases
  (item 7 explains why). Honest note: the skill was **never opened** during the run (zero skill
  invocations across 22 transcripts), so it did not cause this instance — fix it anyway, it will
  cause the next one. The planner checklist line *was* in the agent's input.
* **Evidence:** journal §4i.9, "Where the 16-bit came from".
* **Changes:** `skills/partial-scaler-reduce/SKILL.md`, `agents/incremental-planner.md`.
* **Size:** two corrected sentences.

## 21. An operation's own intermediate values inherit the input's compressed storage format

One of the supported input formats is block-compressed: sixteen numbers share one 8-bit exponent.
That is a reasonable *storage* choice for a tensor whose caller picked it. The generated operation
held its own intermediates — the running sum, the squares, the normalised block — in that same
compressed format, so each of three stages paid a fresh block-float rounding on values the kernel
had just produced. The rule: **an operation's intermediates are sized by what the computation needs,
never inherited from a compressed input format.** For this operation the right width was 16-bit
floating point, not 32-bit — already at the accuracy floor, and half the bytes per tile — which is
the other half of the rule: promote to what the arithmetic needs, not to the widest thing available.
(`references/numerical_stability_analysis_reference.md` §3.1 item 2b already states the converse
correctly — do not make an intermediate 32-bit when the destination register is 16-bit — so this is
the missing companion clause, not a contradiction.)

* **Cost:** measured error 1.708e-2 where the hand-written operation's own gate for the same case is
  1.6e-2. It passed here because the suite's tolerance for that format (correlation 0.99, relative
  root-mean-square 0.10) is about **6× looser**. The generic rule that falls out: **a tolerance
  looser than what the format itself can deliver is a hole with a number in it** — derive each
  cell's tolerance from the arithmetic, do not pick a round number.
* **Evidence:** journal §4i.8; journal §4i "What to change", item 4.
* **Changes:** `agents/incremental-planner.md`, `references/numerical_stability_analysis_reference.md`,
  and the tolerance-authoring section of `skills/golden-tests/SKILL.md`.
* **Size:** two rule paragraphs.

## 22. A caller's tuning value is validated against one quantity and used against another

The operation exposes a block-width knob and validates that it divides the width of one core's
slice. Deeper in, the compute loop walks a *different* quantity — the operation's own internally
resolved chunk width — and the caller's number is never checked against that. At width 72 over 2
cores the two disagree, and an internal assertion fired: a crash, on a call the operation had just
declared valid. The general rule for the implementer: **validate a caller-settable value against the
quantity the kernel actually walks, not against a proxy computed elsewhere.** A second rule comes
with it: the prompt, the design note and the operation's own comment all promised this knob would be
*"HONOURED, never clamped, never absorbed"*, which is **not achievable** on any plan that re-derives
the width internally — and since the operation's derived default measured *better* than the
hand-tuned value the specification held up as the prize (24,497 ns against 25,513), the honest fix is
to change the promise, not to chase the clamp. **Do not write a contract the implementation cannot
keep;** where a knob's ceiling is a hardware limit the caller cannot see, say the operation takes the
nearest legal value.

* **Cost:** a Python-level crash on a valid call. Four independent coverage layers had to line up for
  the tests to miss it (see item 23).
* **Evidence:** journal §4i.10 and §4h ("Exposed but REDUNDANT").
* **Changes:** `agents/ttnn-implementer.md`, `agents/incremental-planner.md`, and the contract
  language in the prompt templates.
* **Size:** two sentences.

## 23. A caller-settable knob reachable only through a per-case override is untested in practice

Item 22's defect needed four things to line up: the knob is not a swept axis, only a per-case
override; every case that does build a tuning object uses a tile-aligned width, so the fallback plan
never triggers; the one non-aligned case that passes a tuning object uses the variant of the object
that has no such field; and the helper that derives a tuning object sets the knob to 1, which divides
everything. The rule: **if a value is part of the operation's public surface, the suite sweeps it —
or the run records in writing that its behaviour is unknown.** A per-case override that nothing sets
is not coverage.

* **Cost:** item 22's crash, plus an unknown amount of untested behaviour behind every other
  override-only value.
* **Evidence:** journal §4i.10 ("Why the tests missed it — four layers"); journal §4i "What to
  change", item 8.
* **Changes:** `skills/golden-tests/SKILL.md`.
* **Size:** one rule plus a mechanical cross-check of the operation's signature against the suite's
  swept axes.

## 24. How deep to let memory reads run is a real design knob, and nobody is told it depends on where the data lives

A deep queue of outstanding reads hides main-memory latency; against the fast on-chip memory of
other worker cores it collapses. The generated reader left all 32 tiles of a row in flight before one
wait — 3,456 concurrent reads across 108 cores — and its wait time went **3,736 → 16,783 ns** when
the same tensor moved from main memory to on-chip memory, while the issue loop got *cheaper*. The
hand-written operation waits every 8 reads and its reader time *halves* on the same move. Three rules
for the blocking model and the data-movement ceiling skill: the wait cadence is a **knob**, its
optimum is placement-dependent and **not monotonic** (measured: 32 → 33,089 ns, 16 → 21,873, 8 →
20,819, 4 → 21,360); cap **transactions, not bytes**, because what saturates a worker's read path is
how many requests are outstanding against it and each read is one request whatever its page size;
and it must be **conditional**, because the same cap costs about 7% on a main-memory input, which is
392 of 416 cases.

* **Cost:** the worst case of 416 sat at 0.591× the hand-written operation; the fix took the tail to
  0.760× with zero of the other 410 cases slowed by more than 5%.
* **Evidence:** journal §4k.4; port §13 and §14.
* **Changes:** `references/blocking-model.md`, `skills/perf-ceiling-dm/SKILL.md`,
  `agents/incremental-planner.md`.
* **Size:** one design rule with its measured sweep as evidence.

## 25. Two ways a change silently fails to take effect, neither of which any prompt warns about

**(a) A hardcoded index next to a named constant.** Adding a compile-time argument requires moving
*both* the host-side list and a literal index inside the kernel (here, `TensorAccessorArgs<33>()`).
The count assertion that exists does **not** catch a mismatch — it only counts. Getting this wrong
hung the board. The lesson generalises: **a literal that duplicates a named constant is a trap**;
prefer deriving placement from data already available (this was ultimately fixed by reading the
memory kind from the accessor's own descriptor bit instead of adding an argument at all, which is
also the better design). **(b) A knob flip that changes no compile-time argument is a no-op.** An
ablation switch was flipped, the output digits were **identical**, and the build reported "JIT cache
stats 44/44 hits" — because the code path being disabled had a fallback that kept doing the same
thing. Identical digits *plus* a 100%-cache-hit line is the tell, and both prompts should say so:
**after flipping a knob, confirm the build key changed and the digits moved; if neither did, the
knob did not reach the kernel.**

* **Cost:** (a) one hung board and a lost debugging cycle; (b) a false-negative ablation that
  briefly pointed the whole precision investigation at the wrong mechanism.
* **Evidence:** journal §4k.4 (last bullet); port §14; journal §4m.6 ("Measurement notes worth
  keeping", note 1).
* **Changes:** `agents/ttnn-implementer.md` (which documents the argument-chaining helper at lines
  570–585 but not this failure mode), `agents/perf-part-optimizer.md`.
* **Size:** two sentences.

## 26. Whole-model runs become a required late step, with a human deciding what a failure means

For an operation that replaces a symbol production code already calls, running whole models end to
end — with real weights, under program-cache reuse and trace capture, ending in a comparison of the
text the model generates — becomes a **required late step in the pipeline**. It is not an automatic
pass/fail gate: when it fails there are several valid responses (fix the operation, change a
precision setting, accept a last-bit difference that a token-exact comparison cannot tolerate,
conclude the entry does not really exercise the operation), and a human chooses among them. It earns
its place because it is the **only** instrument that found item 3: ~23,500 generated cells, 416
reference cases, 140 production-traced configurations and every isolated metric passed.

Three practical notes belong with it, all discovered the hard way. **Finding the entries by name
undercounts badly** — 43 of 91 tiered entries reach this operation and **30 of them never mention it
by name**; the productive pattern is a per-model normalisation module that resolves its
"distributed" flag to false by construction. **Some entries are worthless as evidence and the step
must say which, in writing** — several reach a distributed variant, a different fused operation, or
a hand-rolled chain; a green run there proves nothing and someone will otherwise read it as proof.
And **a green continuous-integration run can mean nothing ran**: an empty job matrix is skipped, a
model filter that matches nothing only warns, and a hardware type with no live runner yields no jobs
— all three conclude successfully. Read the matrix step's output, not the run's conclusion.

* **Cost:** item 3 reached production continuous integration and was found there.
* **Evidence:** journal §4k.8, §4k.12, §4m.4 item 1; port §12.
* **Changes:** `eval/run_eval.py` / the pipeline stage list (there is currently **no** end-to-end
  model stage anywhere in it), plus a documented entry-selection procedure.
* **Size:** a new pipeline stage and a procedure document.

## 27. The measurement instrument: bring the merged fix onto the working branch, and delete a false claim being cited as authority

The device-time figure is the **sum** of every program's kernel duration inside a profiler window,
so the number means whatever the window contains, and a wrong window is **silent** — a plausible
figure, not an error. In one measured comparison the window held one extra program (a layout
conversion the test performed after the operation) which was **98.5% of the recorded time** on one
suite, and which could exceed the operation itself: worst case 42,119 ns of scaffolding charged to
an 8,219 ns operation. A merged framework change (`op_window()`, bracketing exactly the operation's
dispatch, plus a `device_num_programs` invariant so a contaminated window is assertable, plus a
real-time profiler at 0.74% overhead instead of 441.8%) fixes this — and it lives only on the
framework's `main` branch. The consolidation branch this work is being folded into
(`dnijemcevic/codegen_new`, tip `aada63b`) has **zero** occurrences of either name and is 24 commits
behind. Merging is the item. Two things travel with it: an op-agnostic measurement plugin
(`eval/perf_shim.py`) and `op_window()` solve the same problem twice and must be reconciled; and
`eval/profiling.py` (around lines 98–101 on the consolidation branch) carries a **false statement** —
that reading a spread tensor back to the host dispatches a conversion program. It dispatches no
program at all; the read is issued through the hardware command queue and the unspreading happens
inside the read. The conclusion that docstring justifies is still right, but the stated reason is
false and was being cited as authority. (Fixed on `main`; live on the consolidation branch.)

One consequence worth writing down beside the corrected text: **there is a gaming vector** in a
device-program metric. An operation can score better by choosing an output format that is cheap on
device and expensive to read back — host-side conversion of one 8192×1024 tensor costs 3.66 ms,
23.9 ms or 52.7 ms depending on format, none of it visible to the device counter.

* **Cost:** a full set of comparison numbers taken through a contaminated window and superseded; a
  day spent attributing a 98.5% inflation.
* **Evidence:** journal §4k.0 and §4k.9; port §10 ("Three windows", "Mistake 3", "A bare `to_torch`
  dispatches NO program").
* **Changes:** merge `origin/main` into the consolidation branch; correct `eval/profiling.py`;
  reconcile `eval/perf_shim.py` with `op_window()`.
* **Size:** a branch merge plus a docstring.

## 28. A guard whose message and whose condition disagree is invisible to review, and a checker could see it

Third sighting in this project: a compile-time guard read the wrong variable — it tested the
activation's layout flag while its message named a rule about a different operand, two argument slots
away — and broke 48 cells at build time. Previously the same shape appeared as a misspelled
preprocessor name making a condition permanently true. A human reviewer reads the message, agrees
with the rule, and moves on. A checker can compare the symbol the condition tests against the symbols
the message names, and flag a disagreement. This is the same defect shape as item 1 (a check whose
text describes one thing and whose condition tests another), which is why the two belong in the same
report.

* **Cost:** 48 cells failing at build time, plus review time spent not seeing it.
* **Evidence:** journal §4e TODO-4.
* **Changes:** `agents/ttnn-static-analyzer.md` (it currently has no such check) and
  `references/static-analysis-checklist.md`.
* **Size:** one new analyser check.

## 29. A test stage that collects nothing, or a fraction, reports success

A stage recorded a total of zero cases and returned success; the completion marker was written after
29 seconds and the summary read clean. **A stage that passes on zero cases is worse than a failing
stage.** Separately, one module out of thirteen failed to import and pytest abandoned the other 1,002
collected cases with it — so a stage reported the results of 13 cases as though they were the whole
suite. The fix used locally was a **floor on the collected count per stage**, with a shortfall or any
collection error making the stage fail. The framework's own runner has no such floor: it counts
what it gets. The same principle covers two reporting defects recorded alongside: a crash in a late
stage should **fail the run** rather than let it report success with zero evidence from that stage,
and two different populations of test must never be added into one headline number.

* **Cost:** two stages that measured nothing were reported as done; one run reported overall success
  with 112 reference cases failing; one headline count silently merged two populations.
* **Evidence:** port §7 ("S4 collected zero cases", "The harness stamped a stage done having
  collected nothing", "S3b was the harness too"); journal §4e TODO-10 and TODO-11.
* **Changes:** `eval/eval_test_runner.sh`, `eval/run_eval.py`.
* **Size:** a per-stage expected-count floor and a non-zero exit on collection error.
* **Uncertainty:** the zero-collection evidence comes from the port session's own driver script, not
  from `eval_test_runner.sh` directly. I confirmed the framework runner has no collected-count floor
  and treats a low count as a valid result; I did not construct a case that reproduces it there.

## 30. Method rule: read a diagnostic report differentially, against a matched control

The hardware triage tool prints everything it can see, and most of what it prints is normal. One
investigation read a triage report as pointing at the network fabric — eight communication cores
holding reads with zero responses — and held "the operation versus the fabric is genuinely open" for
days on it. The local reproduction of the identical hang, same core, same counter mismatch, showed
**no such finding at all**, on a machine whose links were up. The explanation is structural: that
check compares each core's hardware registers against software counters held in that core's local
memory, and on an idle communication core those counters are zero because no kernel is running there
to maintain them — a false positive by construction. **Rule: run the control, triage the control too,
and diff. A finding present in both is the machine; a finding present only in the failing run is the
bug.**

* **Cost:** days held on a wrong hypothesis; continuous-integration experiments dispatched to settle
  what one local control settled in two minutes.
* **Evidence:** journal §4k.13; port §18 ("The red herring").
* **Changes:** `agents/ttnn-expert-debugger.md`, `skills/debug-ttnn-op/SKILL.md`.
* **Size:** one paragraph.

## 31. Method rule: reproduce a failing case in isolation before blaming the operation, and run the cheapest discriminator first

A result repeating does not mean the operation caused it. In item 15's case, 49 failures reproduced
byte-identically across two replicates, both orderings and eight phases — and were caused by the
previous test's leftover memory, because the environment repeated too. The cells all pass one per
process, which is a single command. **Reproducibility is not intrinsicness; isolation is the cheap
discriminator, and it comes first.** The companion rule, from a different failure: when a problem is
isolated to a one-word switch, **flip the switch before reading a diagnostic in depth**, and prefer a
local reproduction to continuous integration. Continuous integration is the wrong instrument for a
one-word question — its queue is hours, its logs carry no per-core detail, and a run can be cancelled
and take its evidence with it (one such run reports no steps at all and its log is gone; a
*cancelled* conclusion is not a *failure* and is not a verdict).

* **Cost:** a day on the first; a multi-day question that a 45-second local experiment settled on the
  second.
* **Evidence:** journal §4j item 12 and §4k.14.
* **Changes:** `agents/ttnn-expert-debugger.md`, `prompts/` shared method fragment.
* **Size:** one paragraph.

## 32. Method rule: a precision experiment needs a zero-magnitude control and an exact readout

Four separate perturbation experiments in this investigation were **silent no-ops**: multiplying by
1.002, by 1.004, and by (1 + 2⁻⁸) all round back to the original value in 16-bit floating point, so
the "perturbed" run was bit-identical to the baseline and its result meant nothing. A fifth
injection, replicated from one device onto a tensor split across four, produced a dramatic number
with zero noise actually applied. **Every injection needs a zero-magnitude control that returns the
exact baseline digits before its result means anything**, and sub-unit-of-least-precision injections
through a 16-bit format are no-ops by construction. The companion rule, from item 19: read a scale
**exactly**, through a witness element, never by fitting one through the output. And a third, which
this whole investigation earned: **do not describe hardware arithmetic from memory** — read the
instruction-set document, check the functional simulator, and confirm with a witness probe on
silicon; the three disagreed here in ways that mattered.

* **Cost:** several days of injection experiments whose results were uninterpretable, and one full
  round of per-row conclusions invalidated by a biased fit.
* **Evidence:** journal §4m.4 item 5, §4m.7 §1 and note (c), §4m.9.
* **Changes:** a shared method fragment under `prompts/`, referenced by the debugger and the
  verifier.
* **Size:** one paragraph.

## 33. A refinement that widens an axis must re-audit what was hardcoded for the narrower one, and accuracy claims must be measured bitwise

Two rules that share a cause. **(a)** Item 7's mask-format defect was a 16-bit assumption that was
correct for every format the earlier version shipped and wrong for one a later refinement added.
Nothing in the pipeline looks for that. Since a from-scratch generation also widens its supported set
one refinement at a time, the rule applies generally: **when a refinement adds a value to an axis,
re-audit every hardcoded buffer format, width and index against the new value.** **(b)** Where a
design claims an alternative compute path is as accurate as the one it replaces, **require a
bit-level comparison against the original on captured real inputs**, not an argument and not an
aggregate metric — here, 100% bit-identity was one probe away and would have settled item 3 months
earlier.

* **Cost:** item 7's defect was live from the operation's first commit; item 3's mechanism took a
  full investigation to pin down.
* **Evidence:** journal §4i.10 (the generic half, with the starting-from-an-existing-implementation
  framing stripped off), §4m.6 addition 7.
* **Changes:** `agents/incremental-verifier.md` (the refinement queue's acceptance criteria),
  `agents/ttnn-implementer.md`.
* **Size:** two sentences.

## 34. A run should state which framework revision it ran, and how far behind the tip that is

The pipeline pins its framework revision for reproducibility. That pin **silently pins away every
subsequent framework fix**, which is exactly what happened here: one session measured on a harness
predating a merged measurement fix and reported the old behaviour as a live bug. Two consequences:
a run's report should state the harness revision *and* its distance from the framework tip; and
before filing a harness finding as a bug, check whether the tip already carries the fix — the
behaviour may be the pin, not the framework.

* **Cost:** one finding investigated and written up against a fix that had already landed.
* **Evidence:** journal §4k.0.
* **Changes:** `eval/run_eval.py` (record it), the dashboard (show it).
* **Size:** a few lines of code.

## 35. Performance rounds run to a fixed count instead of stopping when they stop earning

Two tournaments at roughly $154 delivered 1.073× overall. Within them, one round measured **0.998×**
— it earned nothing at all in aggregate while producing churn (best 2.74×, worst 0.74× across
cases). **Stop when a round comes out flat rather than running a fixed three.** Note this interacts
with item 9: "flat in aggregate" must be judged on the distribution, not the mean, so the stopping
rule is "no case improved materially and none regressed", not "the total did not move".

* **Cost:** one round of a tournament, roughly $50, for no aggregate gain.
* **Evidence:** journal §4e TODO-3.
* **Changes:** `eval/run_refinements.py`, `agents/perf-coordinator.md`.
* **Size:** a stopping condition.

## 36. Two places the pipeline drops its own evidence

**(a)** The final refinement's timings were never ingested, so its claimed 1.34×–10.46× cannot be
verified from the results database and the cumulative measurement has a hole in it. **(b)** The
review agent whose entire job is finding the kinds of defect on this list returns early when a late
stage's results file is absent (`eval/run_eval.py` around line 680) — so it never ran on the run that
most needed it. It should reflect on whatever exists rather than requiring a particular stage to have
succeeded. A related recording defect from the same family: one run's top-level database rows
predated all three of its performance tournaments, so anything reading the unphased results saw the
operation 3.4× slower than it actually finished.

* **Cost:** one unverifiable performance claim; one skipped review at the worst possible moment; one
  misleading top-level record.
* **Evidence:** journal §4e TODO-5, TODO-12; journal §4b ("The DB's run-level rows").
* **Changes:** `eval/run_eval.py`, `eval/ingest.py`, `eval/db.py`.
* **Size:** small harness fixes.

## 37. A chart must show the cases it could not plot

Two ways this went wrong in one day. 127 of 416 cases failed to pair between two datasets and were
**silently omitted**; because the curve was sorted best-first the drop was invisible and flattered the
headline (3.07× reported against a true 2.68×). And 23,458 bars were drawn one pixel apart on a
thousand-pixel canvas, so everything past the first thousand fell off the right edge — which, on
best-first sorted data, is precisely the slow end. **Any chart states how many data points it could
not plot and why, and a sorted chart never silently truncates the far end.**

* **Cost:** one wrong headline number published and corrected; one chart that hid the regressions it
  was drawn to find.
* **Evidence:** journal §4j item 13.
* **Changes:** `eval/dashboard.py` and the reporting conventions the agents follow.
* **Size:** a reporting rule plus a count in the chart caption.

## 38. The safe test-runner script ignores the environment variable that says where the Python environment is

`scripts/run_safe_pytest.sh` hardcodes `./python_env`, so in any working copy other than the primary
one it silently falls back to the system Python and dies on a configuration import. Every hand-run
probe against a secondary working copy hits this. **Outside this framework** — it is a tt-metal
script — but it costs the pipeline's agents time on every run that is not in the primary tree.

* **Cost:** small and constant; hit on essentially every manual probe against a clone.
* **Evidence:** journal §4e TODO-13; port §18 (the reproduction recipe has to export it by hand).
* **Changes:** `/localdev/dnijemcevic/tt-metal/scripts/run_safe_pytest.sh`.
* **Size:** one line.

---

# Deferred, mentioned once

**Comparing an operation's output after feeding it into whatever consumes it next.** This is a real
and valuable idea, and on the evidence it is the test that would have caught item 3 — the generated
operation's output is indistinguishable from the alternative by every isolated statistic, and yet
three times more harmful than random noise of its own magnitude to the operation that consumes it. It
is **deferred** because which consumer follows which operation is operation-specific, and because the
investigation showed the test only works with the candidate's *real* values in the *real* model:
single-layer reconstructions sat within 5e-6 of exact for both implementations, and every synthetic
perturbation under-reproduced the effect. Item 26 is the practical stand-in.

---

# Already done — do not re-file these

Landed in the framework (`git log 98d50ac..aada63b` in the submodule), listed so the difference from
the list above is visible:

| Commit | What it fixed |
|---|---|
| `4423661` | A constraint must be written as a **rule**, never as one example that satisfies it. Writing an input's shape as a two-number example pinned its rank when the underlying rule said nothing about rank; 32 dedicated cases went green and wrong. Also: a reviewer may now fail a description for being **too narrow**, not only for being untrue. |
| `510bc66` | **Never hand-enumerate a combinatorial set.** N optional inputs means 2^N combinations and all of them are legal; a hand-written list of six turned a test-budget decision into a capability decision and refused two ordinary calls. The rule now lives in `skills/golden-tests/SKILL.md:1124`. |
| `abf8967` | A note about required behaviour must say **what a call does**, not what to avoid. "Do not silently accept a value you discard" is satisfied by refusing the call, which is a different contract; verdicts now carry a machine-readable outcome field. |
| `2ed265d` | The precision convention no longer prescribes refusing a 32-bit float input with a 16-bit accumulator. **Half-done — see item 10** for what remains. |
| `1ee07b4` | The shared accuracy code no longer crashes when an operation legitimately returns an empty result. |
| `812c68f` | The verifier probes what the operation's declarations **refuse**, and handles an empty gap between what is tested and what is supported. |
| `0a8a442`, `4d5dbe1` | The performance tournament's case-ranking rule is written down instead of living in one run's changelog, and one operation's case set was widened from 13 to 19 to cover operands, the accumulator setting and the input's physical layout. **The general procedure is still missing — items 12 and 13.** |
| `ba0b374` | The convention that a performance target ships with a runnable script that reproduces it. **Lives in one operation's directory; lift it — item 12.** |
| `14d2784`, `2a0e441`, `36b6b88`, `49e38fe` | Do not generate inputs for cells the operation will refuse; label loose cases with a group; never drop the runtime axis observer; the case-authoring design written up. |
| `7918c82`, `4650fba` | A declared refusal is a third outcome that scores neither for nor against, and the reference-test layer runs all its files. |
| `10ca301`, `aada63b` | An operation-agnostic measurement plugin and a production-trace replay harness. |
| On the framework's `main` branch only | The measurement-window fix (`op_window()`, `device_num_programs`, the low-overhead profiler) and the corrected profiling docstring — **item 27 is merging them onto the working branch.** |

---

# What was excluded, and why

So the reader can see the boundary:

* **Anything that only matters when generation starts from a designated existing implementation.**
  The gap-analysis skill and its acceptance tests, the "preserve the earlier version's decisions"
  rule, the structural-equivalence fast path, the no-regression-against-the-earlier-version check,
  the effort-split argument for such a run, and the rule about re-reading inherited assumptions —
  except for its generic core, which is **item 33(a)**.
* **Anything that only matters when comparing two implementations of the same operation.** The
  symmetry requirement (both sides measured through the same window, selection in exactly one place,
  proving which side ran), the two "zero-cost asymmetry detectors" and their retraction, the
  before/after switch mechanism and its five load-bearing details, the aliased-type shim, and the
  rule that a tolerance must be at least as tight as the counterpart's — except for its generic core,
  which is **item 21**'s closing paragraph (a tolerance looser than what the arithmetic can deliver
  is a hole with a number in it).
* **Anything about moving code between git branches.** The branch topology proposal, the merge-
  instead-of-port plan, the pre-nuke comparison tree, the restore direction, the declared counterpart
  commit, and the continuous-integration cache-lineage timeout — except **item 34** (a pin hides
  later fixes), which applies to any run.
* **The interface quirks catalogue** (journal §4h): fourteen inconsistencies in the hand-written
  operation's public surface. Properties of that operation, not of this framework — except where a
  generic rule fell out, which is **item 22**.

---

# Group index, for acting on it

**What the test-suite generator must produce** — 4, 5, 7, 8, 13, 18, 23, and the tolerance clause of 21.
**What an agent must be told** — 2, 3, 9, 11, 17, 21, 22, 24, 25, 33, 35.
**What the harness must do differently** — 14, 15, 26, 27, 29, 34, 36, 37, and the enforcement half of 10.
**What is wrong in a reference document** — 3 (the numerical-stability severity guide), 10 (three
operation prompts contradicting the corrected convention), 20 (the reduce-scaler format rule, in two
places), 27 (the false readback claim).
**Method rules for any agent** — 30, 31, 32, and 28's checker.
**Outside this framework entirely** — 1 (the firmware's end-of-kernel check), 38 (the test-runner
script), and two simulator discrepancies recorded for their owners: the functional simulator rounds
each term to 11 significant bits before adding where silicon measures a grid 6 bits below the
format's own step, and its differential-fuzzing reference rounds ties to even where silicon rounds
ties away from zero (journal §4m.10).
