# Post-Port Pass — the procedure

You have been asked to apply **one targeted fix** to **one TTNN operation** that has already been
ported to the Metal 2.0 API. This file is the procedure; the fix recipe your invoker named is the
content. Read this file first, then open the fix recipe when [Step 2](#step-2--survey-every-site)
sends you there.

**If your invoker named more than one fix, you are running a batch.** Read [If you are running a
batch](#if-you-are-running-a-batch) before you begin Step 0 — it sets the output contract for the
whole run. Still one op, and still one fix at a time.

If you are new to this codebase, the next section is your orientation. If you have just finished
porting this op, skip it.

---

## Orientation

**The hardware and the stack.** Tenstorrent builds AI accelerator cards. Each card holds a grid of
cores, and each core runs small RISC-V programs called **kernels** — typically a *reader* and a
*writer* (data movement, moving tiles between DRAM and on-core memory) plus a *compute* kernel
(the math). Two hardware generations matter here: **Gen1** is Wormhole and Blackhole, the current
silicon, and is what you are working on. **Gen2** is Quasar, the next one.

The `tt-metal` repository holds two layers you will touch. **Metalium** is the low-level host API
that builds and launches device programs. **TTNN** is the PyTorch-like operation library above it.

**Ops and program factories.** A TTNN **op** is one operation — `tilize`, `concat`, `softmax` —
living in `ttnn/cpp/ttnn/operations/<family>/<op>/`. Inside it, a **program factory** is the host
code that constructs what the device will run: which kernels go on which cores, which buffers they
use, and what arguments they receive. Its kernels live under `device/kernels/`. The factory is
where nearly all of your work happens.

**What Metal 2.0 changed.** The legacy Metalium API was imperative — create a buffer, set some
arguments, launch — and it passed everything to kernels as bare `uint32_t` values: raw addresses,
magic buffer indices, positional argument slots. Metal 2.0 replaces that with a declarative pair: a
**`ProgramSpec`** describing the immutable structure, and **`ProgramRunArgs`** carrying the
per-execution values. Resources are **named, typed bindings** — a kernel refers to `tensor::input`
or `dfb::in0` rather than to an address or an index — and the framework generates kernel-side
headers so those names resolve at compile time.

**DFBs.** A **DataflowBuffer (DFB)** is Metal 2.0's on-core buffer, replacing the legacy
**CircularBuffer (CB)**. On Gen1 the two are essentially the same thing; the concepts diverge on
Gen2. Legacy code says `cb`, ported code says `dfb`.

**A port** converts an op's factory from the legacy API to Metal 2.0 — a large rewrite. That has
already happened for your op. **A post-port pass**, which is what you are doing, is a small targeted
improvement on top of a working port.

If the fix recipe assumes a concept this section didn't give you, the API and concept reference is
[`migration_guide.md`](../shared/migration_guide.md). Read only the part you need — it is long,
and you are not expected to have read it.

---

## What a post-port pass is

Most of these fixes are purely expressive: the op behaves identically before and after — same
numerics, same performance, same observable side effects — and what changes is how the code says
what it already does. A few do change behaviour deliberately. **Each fix recipe states its own
behaviour-preservation property at the top**; read it, because it tells you what your verification
in [Step 4](#step-4--re-verify) is and is not entitled to conclude. A recipe may also override a
step of this procedure outright, and will say so where it does.

### Style passes and semantic passes

The fix recipes are filed under two directories, and the split is not about how big the change is.

- **`style/`** — the change is expressive. What the program *means* is untouched; what changes is how
  the code says it. Your sentinel set is a real check, and the characteristic way to be wrong is to
  miss a site.
- **`semantic/`** — the change alters something about how the program *runs*, even where the numerics
  come out identical: what work the device actually does, when it is dispatched, what the compiler is
  now free to assume. Most still preserve behaviour, and where one does not its recipe says so — but
  whether it *is* preserved rests on reasoning your tests cannot fully police.

What that changes for you is where the burden sits. A style pass is bounded by its recipe. A semantic
pass is bounded by its recipe **and** by the judgement of the person who launched it, who is closer
to this op than you are and who is expecting to hear what you saw. So on a semantic pass: take the
stop rules literally rather than hunting for a reading that lets you finish, report the things you
noticed even when you are fairly confident they are fine, and treat *"I would have had to decide that
myself"* as a reason to stop rather than as a decision to make.

If your invoker did not say which kind you are running, the recipe's own path tells you.

**Why this is worth doing.** The port that preceded you was mechanical — it got the op compiling
against the new API, but a mechanical conversion leaves the code expressing the old model in new
syntax. What survives these passes is what everyone downstream learns Metal 2.0 *from*: the next
porter working on a sibling op, the engineer writing a new op from an existing one as a template,
the customer reading a reference implementation. A buffer still named `cb_in` or a hand-inverted
argument loop teaches the wrong thing to every one of them. Some of these fixes are also Gen2
preconditions in disguise — a construct that is perfectly fine on Gen1 and cannot survive Quasar,
where the port has to happen anyway and is far more expensive later. Each pass is small; the set
of them is not.

**What counts as done.** A pass has four possible outcomes, and **all four are complete work**:

- **Applied** — sites found, transformed, verified.
- **No sites found** — the op didn't have this pattern.
- **Stopped on a precondition** — the op isn't ported, or its tests weren't green when you arrived.
- **Stopped on a site** — the transformation didn't cover something you found.

The deliverable is not a diff. It is an accurate verdict about this op, and *"needs nothing"* and
*"needs a decision I wasn't given"* are both accurate verdicts — your reader would far rather have
either than a diff that papered over the reason.

There is one real failure mode, and it is the opposite of stopping: **a change that should not have
been made.** A transformation forced onto a site that didn't fit it. A workaround invented so the
pass wouldn't have to stop. Scope widened until something worked. A precondition quietly routed
around. Every one of these comes from treating *"produced a diff"* as the objective — and each
costs far more to find later than the stop would have cost now, because it arrives disguised as
success.

**One fix per pass.** While applying it you will notice other things worth changing — another fix's
target, a stale comment, a helper that could be simpler. Note them in your report; do not do them. A
diff containing two fixes cannot be reviewed as either one, and the before/after check below stops
meaning anything.

**Trust the check, don't replace it with planning.** The op is green when you start, the change is
small, and you can be green again minutes later. That before/after measurement is your safety net,
and it is a better one than any amount of reasoning in advance. So this procedure asks for no
written plan, no inventory, and no audit — what it asks for instead is discipline about the
*baseline*, which is the one thing a fast feedback loop can still get wrong.

---

## Step 0 — Confirm the inputs

Three things. Get them explicitly rather than inferring them.

1. **The op directory** — e.g. `ttnn/cpp/ttnn/operations/data_movement/tilize`.
2. **The fix** — which fix recipe to run. If your invoker named a symptom rather than a recipe,
   ask which they mean; do not choose one yourself.
3. **The sentinel set** — the tests that must pass before and after. Your invoker supplies these.
   If they didn't, ask; do not assemble a set on your own judgement, because a missed test turns
   this whole procedure into a silent false-green.

Your Python venv must be active in this shell. If there is no `python_env`, run `./create_venv.sh`,
then `source python_env/bin/activate`. Porters who inherit a checkout routinely miss this and hit
import failures that look like code bugs but aren't.

**The op must already be ported to Metal 2.0.** If it isn't — the factory still uses
`CreateCircularBuffer`, `SetRuntimeArgs`, `CBDescriptor` and friends — **stop and report that to
your invoker.** Do not port it. Porting is a large, separately-specified job that was not what you
were called for, and attempting it here would produce a change nobody asked for or reviewed.
Reporting an unported op is a complete outcome — you have answered the question that was actually
asked.

## Step 1 — Establish the green baseline

**Before you change anything.** Build, then run the sentinel set.

```bash
./build_metal.sh --build-tests
```

Run builds and tests **in the background and read the log file**, rather than letting output stream
into your context — a failed compile in this repo prints the full clang invocation, hundreds of
include flags per error. You want the failures, not the command lines.

**But let a run finish before you touch a kernel.** Your build does not compile the kernels; the
runtime compiles them from disk at every program-cache miss, so they are read throughout a run
rather than once at its start. Edit one while a run is in flight and it is compiled as-edited
partway through — the run then measures neither the state you launched on nor the one you have now.
That matters most in [Step 4](#step-4--re-verify), where a run spanning an edit can report a pass
for a tree that never existed, and a false green is the one outcome the before/after check cannot
survive. Host-side edits are safe, since the run is executing an already-linked binary; this covers
only the kernel sources and any header they `#include`.

**Run both from the repository root.** From anywhere else, tests fail with *"Root Directory is not
set"* — dozens or hundreds of them at once, which reads like broken code rather than a wrong working
directory, and costs an investigation before anyone thinks to check `pwd`.

Every sentinel must pass. **If the baseline is not green, stop and report.** Do not proceed, and do
not investigate the failure. An op that is already broken is not a valid input to this procedure: a
pre-existing failure carried into the pass destroys your only signal, because afterwards you cannot
tell your regression from the one that was already there. Reporting a red baseline is a complete
outcome, and a useful one — you have found something the person who sent you here did not know.

*Exception:* if you built and ran this op earlier in this session with no intervening changes, that
run **is** your baseline. Don't rebuild to satisfy a ritual.

Record which sentinels passed. You will compare against that list, not against a memory of it.

## Step 2 — Survey every site

Open the fix recipe now and run its recognition step across the op directory. **Enumerate every
site before changing any of them.**

Surveying first is what keeps the pass mechanical. If you fix as you find, you discover the shape
of the work while you are committing to it, and the ninth site teaches you something that should
have changed how you handled the first.

**A survey that finds zero sites is a complete and successful pass.** Report "no sites found" and
stop. The op did not need this fix — that is a result, not a failure, and it is common, because
these recipes are written against a whole corpus rather than against your op.

## Step 3 — Apply

Apply the fix to every site the survey found, and change nothing else.

Follow the fix recipe's transformation exactly. Where it offers a choice it will say so and tell
you how to choose; where it doesn't, there is no choice to make. If a site doesn't fit the recipe's
shape, stop — see [When the fix doesn't fit](#when-the-fix-doesnt-fit).

Three scope limits hold throughout, whatever the fix recipe says:

- **Nothing outside the op directory.** Not shared kernel libraries, not framework headers, not
  another op that happens to have the same problem. Your sentinel set covers *this* op — a change
  to anything shared is a change to ops whose tests you are not running, so your before/after check
  would be silently blind to it.
- **Nothing outside the program factory and its kernels.** The device-operation class — `validate`,
  `invoke`, `compute_output_specs`, attribute parsing, dtype checks — is the op's contract with its
  callers, not Metal 2.0 surface. Editing it makes the diff something other than a post-port pass, and
  a reviewer expecting one will not be looking there.
- **No opportunistic cleanup.** Your diff may contain exactly two things: the transformation, and
  repairs to what the transformation broke. Nothing else — not a rename you prefer, not a
  simplification you can see, not a comment you would word differently. These passes are cheap to
  review precisely because a reviewer can confirm the whole diff is one known transformation; every
  unrelated change spends that. Those observations are genuinely wanted — put them in your report,
  where they cost nothing and get read.

  The second category is usually empty, and it has a test: **was it true before your change and
  false after?** A comment describing the construct you just replaced, a name that now describes
  nothing, a guard asserting what your change made untrue — you are the reason each of those is
  wrong, so leaving it behind is shipping a defect you introduced rather than respecting scope.
  Hold the two cases side by side: *a comment you would word differently* is out of scope, and *a
  comment your change made untrue* is part of your change. If you cannot answer the test with a
  plain yes, the answer is no, and it goes in the report instead.

  **Report each such repair on its own line**, per [Step 5](#step-5--report). Naming them one by one
  is also the check on this permission: if you find yourself writing a justification you would not
  want read back to you, that repair was cleanup wearing a better hat.

**`ttnn/cpp/ttnn/operations/experimental/quasar/` is not evidence of anything.** Nothing from that
directory may enter your work: do not cite it, copy a construct or a name from it, or offer it as
proof that something is legal, idiomatic, or portable — not in the code, not in the report. It holds
quick, deliberately rough ports written to unblock hardware bring-up; it does not represent good
practice and was never reviewed as though it did.

This matters more than it sounds, because it is also one of the largest bodies of Metal 2.0 kernel
code in the tree. A search for precedent lands there first, everything in it compiles and ships, and
it therefore reads as authoritative — including where it contradicts the recipe you are following. If
a recipe and a file under `experimental/quasar/` disagree, the recipe wins and the disagreement goes
in your report. The practice that keeps you clear: don't go looking there, and close the file if a
search drops you in one.

## Step 4 — Re-verify

Rebuild and run **the same sentinel set** — not a subset, not a filter narrowed to what you
touched. Rebuilds are incremental, so this is fast.

Every sentinel that passed in Step 1 must pass now. Beware the false green: a `--gtest_filter`
matching no tests reports success with zero tests run, so confirm the run actually selected the
cases you meant.

**On a regression: revert, then report. Do not patch forward.** Every fix leaves the behaviour your
sentinels exercise unchanged — that is what makes them a valid check — so a newly failing test
means the transformation was misapplied, not that it needs an adjustment. Patching forward is how a
three-line pass becomes a diff nobody can review, and it usually buries the evidence of what actually
went wrong. Revert to the baseline and report the site and the failure; that report is worth more
than a rescued diff.

## Step 5 — Report

A single pass produces **no committed document** — your invoker is right there, and a file per fix per
op would litter the tree faster than anyone would read it. Report in the session, in this shape:

- **Outcome** — `APPLIED` (with the site count), `NO SITES FOUND`, or `STOPPED` (with why).
- **Sites** — `file:line` for each, one line each.
- **Verification** — the sentinel set, green before and green after. State it plainly; do not imply
  it by silence.
- **Repaired because the change falsified it** — one line each, per the scope limits in
  [Step 3](#step-3--apply): the comment, name, or guard, and what your change made untrue about it.
  Omit the heading entirely if there were none. These are the only lines in your diff a reviewer
  cannot check mechanically, so do not leave them to be discovered.
- **Noticed, not done** — everything you saw and correctly left alone: other fixes' targets,
  oddities, things that looked wrong but were out of scope. This is the most valuable section you
  write. It is read by the op's owners and by the people maintaining these recipes, and it is the
  only channel this pass has — nothing else you produce carries an observation out of the diff.
  Write it for someone who will act on it, because someone will.

Leave the change uncommitted — committing clears the working-tree diff from a reviewer's editor
before they have seen it. **Both defaults flip in a batch:** every pass is committed and the report
goes to a file, in the shape above. See [If you are running a batch](#if-you-are-running-a-batch).

---

## When the fix doesn't fit

Stop and report. This is a **success-tier outcome**: a grounded stop is a complete deliverable and
is worth more than a forced fix.

You will find things these recipes did not anticipate. That is expected — they are written against
the ops that existed when they were written, and the corpus is large and irregular. It reflects
neither a defect in the recipe nor a failure on your part, and the judgement of *where the
prescribed transformation stops applying* is the part of this work that most needs a reader who is
actually thinking. Exercise it, and say what you saw.

Stop when a site doesn't match the recipe's shape and you would have to improvise the
transformation; when the fix would require touching code outside the op directory; when it would
change observable behaviour; or when the recipe's guidance and the code in front of you genuinely
disagree.

Report the site, what the recipe expected, what you actually found, and what you would have had to
invent in order to proceed. That last part is the valuable one — it is the signal that improves the
recipe, and these recipes are maintained on exactly that feedback.

**Do not** apply a partial fix to make the pass look complete, and do not widen the fix to
accommodate the site. Either one produces a diff whose reviewer cannot tell what was intended.

---

## If you are running a batch

*Skip this section if you were called for a single fix.*

A batch is **several fixes over one op**, run back-to-back. The procedure above is the unit of work,
run once per fix.

**Still one op.** If your invoker named several, run the one named first and report that the others
need their own runs. Each pass reads this op's factory and kernels closely, and a second op does not
add work so much as degrade every pass after the first.

**A batch is style passes only**, unless your invoker explicitly says otherwise. A semantic pass is
bounded partly by its launcher's judgement (see [Style passes and semantic
passes](#style-passes-and-semantic-passes)), and a batch is precisely the shape where that person is
not there to be asked. So if the set you were handed includes a recipe from `semantic/`, run the
style ones, leave that one untouched, and say so in your report — it wants a run of its own with
someone at the keyboard. This restricts *batching*, not the recipes: any single pass, style or
semantic, is perfectly fine run on its own.

**Assume nobody is watching.** A batch gets launched and left, so nothing you say in the session is
guaranteed to be read. That is why it reports to a file — see [What a batch
produces](#what-a-batch-produces).

**Start from a clean working tree, with the op's port already committed.** Reverting one pass without
disturbing anything else is only well-defined if nothing uncommitted was sitting underneath. If the
tree is dirty when you arrive, stop and report it rather than guessing whose work it is.

Then, for the run itself:

- **Each pass keeps its own baseline → apply → verify cycle.** Don't batch the applications and
  verify once at the end. If three fixes go in and one test goes red, a single verification tells
  you nothing about which one did it — and you have lost precisely the attribution that makes these
  passes cheap. Pass *N*'s baseline is simply pass *N−1*'s verified end state, so this costs one
  build per pass, not two.
- **Commit each applied pass on its own**, as soon as that pass's own sentinels are green. This is
  what keeps it separately revertible: when pass 4 regresses, you revert pass 4, not the day's work.
  Give the commit the fix's name and the [Step 5](#step-5--report) headline, so `git log` carries the
  outcome and the sites:

  ```
  metal2(post-port): <fix name> on <op>

  Outcome:  APPLIED (<n> sites)
  Sites:    <file:line, one per line>
  ```

  A pass that found no sites, or that stopped, has nothing to commit; it is recorded in the report
  instead.
- **Some fixes have a required order**, and the fix recipes say so where it applies. Honour it. If
  two fixes touch the same construct and neither recipe states an order, run them in separate
  passes and report the interaction rather than choosing one yourself.
- **A regression reverts that pass, not the run.** Do what [Step 4](#step-4--re-verify) says — revert
  to the pass's baseline, record the site and the failure, don't patch forward — and then carry on to
  the next fix.

A pass that stops for cause, per [When the fix doesn't fit](#when-the-fix-doesnt-fit), does not stop
the batch either. Record it and carry on to the next one — a stop is a result, and the remaining
fixes are independent of it.

### What a batch produces

**One `METAL2_POSTPORT_REPORT.md` in the op directory**, carrying a section per pass in the
[Step 5](#step-5--report) shape — for **every** pass, including the ones that found no sites and the
ones that stopped. Where a fix recipe asks you to raise a feature request, put it in verbatim, under
its own heading, so it can be lifted out and filed without being rewritten. Commit the file once, at
the end of the run.

This file replaces the in-session report, and it is not a formality: unattended, it is the only route
by which anything you noticed reaches the op's owners or the people maintaining these recipes. An
observation left in the session dies there.

Then close with a short summary to your invoker: one line per pass, plus anything you think they need
to read before they look at the diffs. They have the commits and the report; what they want from you
is the shape of the run.
