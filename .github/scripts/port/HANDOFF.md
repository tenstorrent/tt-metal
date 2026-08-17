<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Porting harness: handoff

Everything about *why* a given rule exists is written where the rule is -- in `port-op.md`'s comments
and in each script's docstrings. This file is for the things that live in no single file: the state of
the experiment, the sharp edges of operating it, and what is not yet known.

## State

**`untilize` is finished, and not by this harness.** It merged on 2026-08-14 as
[#50178](https://github.com/tenstorrent/tt-metal/pull/50178), which the generator's own
`history/vision.md` records as the original Python orchestrator's `--rescope` re-drive. So the op this
harness spent a week on is now in `main`, `ebanerjee/port-untilize` is retired, and the 76 cases it
still failed are not worth another minute -- see the pin section below for what they were actually
telling us.

**The target is now `tilize`, on `ebanerjee/port-tilize`, with the harness on
`ebanerjee/port-harness`.** Both branch off current `main`; the harness has never been merged and does
not need to be. `tilize` was chosen over the simpler `move` because it is `untilize`'s mirror, so the
merged port is the closest structural reference that will ever exist, and because it already has an
open PR from the original pipeline, [#51919](https://github.com/tenstorrent/tt-metal/pull/51919). That
PR is an independent oracle: when this harness produces a `tilize` port, the two can be diffed, which
is a far better measure of the harness than its own pass/fail.

Every manifest op except `move` carries a `ported_codegen_commit`, which the manifest contract says is
stamped only when an op ships through phase 8. So the original pipeline has shipped all of them, and
this harness is not racing to port ops nobody has ported -- it is a reimplementation of that pipeline
as a GitHub Agentic Workflow, and comparing outputs is available for every op it will ever attempt.

Keep in mind when reading everything below: **the generator token's SAML SSO authorization lapses.**
Every mode checks out `tenstorrent/tt-dm-codegen`, and when it lapses that checkout fails with
`The 'tenstorrent' organization has enabled or enforced SAML SSO` and HTTP 403. Nothing in the harness
can route around it and no run of any kind gets past it; `CODEGEN_REPO_TOKEN` has to be re-authorized
at the SSO URL GitHub prints in the error. It is not a scope problem and not something the pipeline can
detect in advance. It cost run
[31736836951](https://github.com/tenstorrent/tt-metal/actions/runs/31736836951) a day.

The harness has produced one real result: a draft PR porting `ttnn.pad`'s codegen builder, from the
`ebanerjee/port-op-dryrun` branch. Everything after that PR was hardening, driven by what was wrong
with it -- a sample that could not contain the failures it was supposed to catch, a routing check that
could not fail, an asserted rather than measured demotion set, and a verdict policy that threw the
work away when the wall clock was noisy.

`untilize` is the second op and the first that is not `pad`. It is the first exercise of the hardened
harness, and it also moved the job off the CIv1 pool the `pad` PR ran on. Both moves at once is why
the recent history is a run of failures: four of them, three ours and one not, each hiding the next.
The harness fixes are described below; the one that is not ours turned out to be unfixable from this
side, and the pipeline is now split in two because of it.

As of 2026-08-11 that run of failures is over. Both halves of the split pipeline work against real
hardware: the launcher pushes, the workflow builds and measures on CIv2, and the results come back.

On 2026-08-12 the agent drove it for the first time. It read the manifest, wrote the whole
`untilize` port -- roughly 2,400 lines across 17 files -- adopted the generated routing test the
baseline had sent back, and correctly chose a compile before spending device time. Then it could
not compile anything, for a reason that had nothing to do with the port: **the MCP gateway cancels a
tool call after 60 seconds by default**, and `build` is minutes. It died at 1m00s four times, the
agent reasonably read four identical infrastructure failures as infrastructure, reported through
`missing_tool` and declined to open a PR on unverified code. Every one of those judgements was
right. Four CIv2 builds were left running with nobody reading them.

That is fixed, and the fix reshaped the tool surface -- see below. The one build that did finish
compiled the agent's code and failed on ordinary first-attempt errors: three redefinitions in
`untilize_codegen_supported.cpp`, two `-Werror` unused variables, one bad overload. That is what the
second iteration is for, and it is the first evidence that the build path works on agent-written
code rather than on a clean tree.

**Run 3, 2026-08-12, is the first one that worked as designed.** One build dispatch rather than
four; a `wait` that came back as content; a build failure the agent actually read; an edit; and a
second build that **passed** -- tree `7fb4d2d88099`, the first agent-written port of this op that
compiles. It reached that in 38 minutes of agent time across two build cycles.

It did waste one cycle on the way, and that waste is now scaffolded against rather than written
down: after the first build failed it dispatched a `verify` on the byte-identical tree, which builds
before it measures and so failed in the same place having queued for a card. `refuse_pointless_
dispatch` now knows how the last build of a tree turned out and blocks exactly that.

**`verify` then ran end to end for the first time, and returned `blocked` for a reason that was
ours.** Run 31626314731 went green through build, wheel and the device job, so every step of the
mode works; the verdict it produced was `changes outside the port's own files`, naming
`ttm_any.tar.zst` and `ttnn-...whl`. Neither is the agent's. The measure job downloaded both into
`docker-job`, which *is* the checkout, so `check_write_paths` counted the job's own inputs as the
agent writing where it was not allowed to. The extracted build escaped notice only because
`build/` is gitignored and the two archives are not.

This was unwinnable from the agent's side, and that is the part worth remembering: the files appear
on the far side, after the snapshot, so no edit to the port could have cleared them. It then proved
itself -- the agent dispatched a second `verify` on the same tree and got the same verdict naming the
same two files, the wheel's version string carrying the new scratch commit each time, which is what
identifies them as the job's own downloads rather than anything in the tree. `verify` is exempt from
`refuse_pointless_dispatch` because measurement is noisy, so nothing bounded that loop but the run's
own budget: roughly twenty minutes of a card per turn, reported to the agent as its own fault. What
the agent did with it was right, for the third time in three runs -- it declined to open a PR on a
port it could not verify. The
archives now land in `port-inputs`, beside the checkout, for the same reason the generator does;
`selftest.py` asserts no `download-artifact` step targets the checkout, and that check was confirmed
to fail when the old path is restored. **No band has produced a number yet**, because the guard runs
before the measurement: correctness and perf on this port remain unmeasured.

One thing found on the way and worth fixing properly: the wheel install globs `./*.whl` and runs
*before* the guard, so a wheel planted in the snapshot would have been installed and only reported
afterwards. Globbing from `port-inputs` closes it for now, since the agent's snapshot cannot reach
that directory, but the ordering is the real defect and the guard should run before anything from
the tree is executed.

**With that fixed, the bands ran, and the harness returned its first graded verdict on a real port:
`back-to-translate`.** Dispatched by hand from a laptop on the agent's compiling tree, over the fixed
base -- run 31630655137, about 20 minutes. Every step of `verify` now has evidence behind it:
`report_verify`, `gate.json`, the correctness band over all 208 cases and the wall and device bands
over 24. The numbers, for the record:

| | |
| --- | --- |
| Correctness | 112 of 112 in-scope cases fail; 96 out-of-scope pass |
| Routing | **0 violations, 0 unverified** -- the predicate is right |
| Golden | `native`, as expected for this op |
| Wall clock | median ratio 0.787 over 24 cases, one regression (`codegen_untilize[206]`) |
| Device | `device_vs_native` 2.09 at its best |
| Demotions | none |

The routing result is the one to keep. `supported_by_codegen()` had to reject non-tile-aligned
`bfloat8_b` while accepting non-tile-aligned `bfloat16`, which is the genuinely hard part of this
port and the thing the emitted test exists to catch -- and the agent got it exactly right across all
96 cases.

Every in-scope failure was one cause, and it was one missing file: `writer_untilize_interleaved.cpp`
includes `rm_shard_split.h`, which lives in the generator at `common/templates/` and was never
vendored into the port. The agent did this correctly for `sequencers.h` -- it sits beside the kernels
and resolves -- and missed the second one. `untilize/codegen/kernels/*.h` was already in the CMake
glob, so vendoring the header was the whole fix: copied verbatim from the generator at the pinned
`codegen_agentic_port` ref, with the four-line SPDX header the sibling carries.

**Re-verified on that tree -- run 31635005308 -- and the harness is now reporting a real defect in the
port rather than one of its own.** Still `back-to-translate`, still 112 of 112 in-scope, but the
missing-file error is gone and what replaced it is genuine translation work:

- `writer_untilize_interleaved.cpp` reads `get_compile_time_arg_val` at indices 5 and 6 while the
  program factory passes five, so the static assertion fails with `(5 < 5)` and `(6 < 5)`. The
  kernel's compile-time argument contract and the factory that fills it disagree.
- `ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(...)` matches no overload in
  the header just vendored, so the call site and the shared helper's signature disagree too.

That is the pipeline working as designed: a true negative about the port, not a false positive from
the harness. Both remaining defects are the agent's to fix and are the natural content of a fourth
run. Two things are settled by getting here, though: the wall band came back **clean** on this tree
-- median ratio 1.009 over 24 cases, no regressions, aggregate OK, against 0.787 and one regression
before -- and `device_vs_native` sits at 2.12. Neither number should be read as the port's
performance while nothing in scope executes; they are evidence that the bands themselves grade
sanely, which is what was untested.

**The lesson worth carrying to the third op: a green build says nothing about kernel includes.**
Kernels are JIT-compiled on the device at first use, so a missing kernel header cannot fail the CI
build -- it failed all 112 in-scope cases at runtime instead, on a tree that had compiled cleanly
twice. This is the first defect the build/verify split cannot shift left, and it is a good candidate
for the build-facts list in the prompt: shared headers under the generator's `common/templates/` must
be vendored beside the kernels that include them.

The second run, the same afternoon, got further and failed for a sibling reason. The agent wrote the
port, started exactly one build -- the start/collect split working -- and then re-dispatched a
byte-identical tree rather than reading the compiler diagnostics, because a non-zero exit reaches it
as a broken tool rather than as an answer. Cancelled once that was diagnosed. Both failures are the
same shape: the harness was speaking in exit codes and the agent only hears content. Both sections
below exist because of it.

## The pipeline is two workflows, and neither can do the other's half

Read this before touching either file, because each half looks incomplete on its own.

`port-op.md` holds the agent and runs on `ubuntu-latest`. That is the only place
`api.githubcopilot.com` is reachable. It has four cores and no card, so it builds nothing: it checks
out, scaffolds, and then every build and every measurement goes out to the other workflow.

`port-measure.yaml` is where the compiler and the card are. It takes a `mode` -- `baseline`, `build`
or `verify` -- calls `build-artifact.yaml` on the CPU pool with a warm Garage ccache, and for the two
modes that need silicon runs a device job on the CIv2 N150 pool. It always uploads a results
artifact, because a run that fails is when the numbers matter most.

`dispatch.py` is the seam. It snapshots the working tree into one commit on top of the base, pushes
it to `port-op-scratch/<op>-<mode>-<run id>-<uuid>`, waits, brings the artifact home and deletes the
ref. Its docstring covers the four non-obvious parts.

### Nothing here is dispatched, and nothing needs to be on `main`

The name `dispatch.py` is a small lie, and the mechanism is worth understanding before editing
either workflow, because the obvious design does not work and this one is not obvious.

`port-measure.yaml` triggers on pushes to `port-op-scratch/**`. The launcher has to push the code
under test regardless, so that push doubles as the trigger, and it carries the payload in the commit
message. Nothing about this pipeline needs to reach `main` to work, including `port-measure.yaml`
itself.

That last sentence used to be justified by a belief that `workflow_dispatch` only fires for workflow
files already on the default branch. **That belief was wrong, and it is worth knowing why it looked
right.** Run [31744036323](https://github.com/tenstorrent/tt-metal/actions/runs/31744036323) appeared
to run a stale definition from a branch, and `main` has no port workflow at all, so dispatch seemed to
be ignoring the ref. What actually happened is in its `headSha`: `3afb236207d`, the commit *before* the
fix under test, created 15 seconds after that fix was committed. The dispatch raced the push, GitHub
resolved the branch to the tip it knew, and an entire agent run was spent proving the old code still
failed. `workflow_dispatch --ref <branch>` does run the file from that branch. Confirm
`git rev-parse origin/<branch>` matches local HEAD before dispatching and this cannot recur.

Three things follow, and each has bitten or would have:

- **The push must be made with a PAT.** GitHub deliberately refuses to start workflow runs from
  pushes made with `GITHUB_TOKEN`, so that runs cannot trigger runs. Here the push *is* the trigger,
  so that suppression would mean nothing happens at all -- the launcher would wait five minutes and
  report that no run appeared. This, not scope, is why a PAT is involved at all.
- **Parameters ride in the commit message, as JSON.** A push carries no inputs. The message is the
  only free-form channel that arrives in the event payload -- so the `resolve` job reads it with no
  checkout, in seconds -- and that leaves nothing in the tree. A params *file* would be seen by
  `gate.py`'s write-path guard, which diffs the checkout against the base commit, and would read as
  the agent writing where it was not allowed to. `resolve` validates every field, because they are
  all interpolated into shell commands on the device runner; `selftest.py` runs that same validator,
  lifted out of the YAML, against both a real launcher message and a list of injection attempts.
- **The run is found by its head SHA**, not by matching a name. The scratch commit is unique to the
  call, so there is no race with another port in flight.

The one remaining cost of `port-measure.yaml` being off `main` is that it cannot be started from the
GitHub UI. To start one by hand, push a scratch ref whose commit message is the JSON payload:

    git commit-tree HEAD^{tree} -p HEAD -m '{"mode":"build","op":"untilize"}' \
      | xargs -I{} git push origin {}:refs/heads/port-op-scratch/manual-$RANDOM

### The push credential, and why it is the sharpest edge here

The launcher takes `PORT_PUSH_TOKEN` if it exists and falls back to `CODEGEN_REPO_TOKEN`, which is
what is actually configured today: a classic PAT with `repo`, already SSO-authorised for the org,
already used by the measure job to check out the generator. Nothing needs provisioning to run.

It has to be *a* PAT for the trigger reason above, and that rules out the otherwise attractive
alternative: gh-aw's strict mode rejects any write permission on the agent job, and `strict: false`
would let `github.token` push -- a token that expires with the run and cannot touch
`.github/workflows/**`, both real advantages -- but its pushes would not start the measurement.

`CODEGEN_REPO_TOKEN` is far broader than the job needs. `repo` reaches every repository its owner
can, where the launcher only ever pushes one branch to this one and reads runs back. Two things
follow, and the second is the one that is easy to miss:

- **Provision `PORT_PUSH_TOKEN` when convenient** and the fallback stops being used with no code
  change. Fine-grained, `tenstorrent/tt-metal` only, `contents: write` plus `actions: read`.
- **The exposure that matters is not the push, it is the trigger.** The workflow that runs is the
  copy in the pushed commit. An agent edit to `port-measure.yaml` would therefore not be a proposal
  -- it would be the next thing to execute, on CIv2, with `CODEGEN_REPO_TOKEN` in scope. The same
  argument covers `gate.py`, which decides whether the port is any good and also travels in the
  scratch commit. A narrow token makes most of this GitHub's problem, since it refuses pushes
  touching `.github/workflows/**` without an explicit workflow scope; a classic `repo` PAT may
  carry that scope, so `dispatch.py` refuses to push any snapshot touching `.github/` at all,
  whatever the credential. `gate.py`'s write-path check covers the same ground from the other side,
  but only at verify time -- which is after the modified workflow would already have run.

The token still never enters the agent's sandbox, by the pre-step route described further down;
that part is unchanged and is what makes any of this defensible.

Two consequences of the split worth internalising:

- **Everything is slow now.** A `build` is the CIv2 build, so 15-25 minutes plus queue; a `verify`
  adds the device job for 35-45. Against a 350-minute job that is six to eight tool calls for the
  whole port, and the `pad` run spent six on performance re-checks alone. The prompt tells the agent
  to batch its edits; if a run runs out of budget, that instruction is the first thing to sharpen.
- **Every file involved comes from the scratch commit**, harness scripts and `port-measure.yaml`
  alike. A fix to `gate.py`, or to the measure workflow itself, takes effect on the very next
  dispatch from the working branch, with no merge and no PR.

## `pad` was load-bearing in ways nobody intended

Five separate defects have turned up in moving from `pad` to `untilize`, all of them things `pad`
happens not to do. They are fixed, but the pattern is the point: **the harness was implicitly
specialised to one op's shape**, in the manifest, in the CMake files it edits, and in the imports its
sweep module needs. Expect the third op to find its own; when it does, prefer widening the mechanism
to what the field or the file actually permits over special-casing the op.

| What `untilize` does that `pad` does not | What broke |
| --- | --- |
| `sweep_suite` names two suites, not one | `module.parameters[suite]` with a list: `TypeError: unhashable type: 'list'`, before any step ran |
| Declares `port_scope` | Nothing read it. A large slice of cases the manifest puts *outside* the ported builder would have been graded as in-scope |
| A kwarg holds a live `ttnn` object (`memory_config`) | The ledger's `json.dumps(..., default=str)` flattened it to `"MemoryConfig(...)"`, and every `ttnn.untilize(x, memory_config=...)` built from it would raise |
| Has explicit `untilize/device/kernels/...` entries in `target_sources`, as well as a glob | `register_kernel_globs` anchored on the last match anywhere in the file and wrote `untilize/codegen/kernels/*.cpp` into the FILES list, where CMake resolves literal paths. Configure failed; `verify()` reported no errors because the string was present *somewhere* |
| Its sweep module reuses the upstream suite (`from sweeps.untilize import ...`) | `tests/sweep_framework` was not on `PYTHONPATH`, so the module cannot import. `pad`'s sweep is self-contained |
| Its native path is *two* device operations (`UntilizeDeviceOperation` on a tile-aligned input, `UntilizeWithUnpaddingDeviceOperation` otherwise) | The profiler attribution check required one op code per leg and declared the run inconclusive. The real invariant is one op code per *case* and leg -- a leg spanning several across cases is normal, and a demoted case deliberately puts native's op code on the ported leg |
| Its generator injects kernel templates under a *different* op's directory (`ops/untilize/builder.py` names `untilize_with_unpadding/device/kernels/codegen_templates`) | `check_write_paths` counted them as writes outside the port and would have blocked every `verify`. They are the harness's own prototype leg dirtying the tree, so they are excluded |

Three of those were found only by a real run, and one (the CMake glob) had a green scaffold step in
front of it. When adding a check, prefer asserting *where* something landed over asserting that it
exists, and prefer the invariant that actually protects the conclusion over the strictest one that
happens to hold for the op in front of you.

`selftest.py` covers all of the above that can be checked without a device -- against a stubbed `ttnn`
and a stubbed sweep module -- and runs on a laptop in under a second. Run it after touching
`ledger.py`, `scaffold.py`, `strata.py`, or `dispatch.py`. It is the only test the harness has.

It also covers what `dispatch.py` does to the working tree, which is worth calling out because that
is now the most consequential thing in the pipeline that produces no visible output: the snapshot has
to carry the untracked scaffolded stubs and the tracked edits, exclude the generator checkout and the
build directory, sit exactly one commit above the base so a depth-1 fetch finds it, and leave the
agent's HEAD and index alone. Every one of those is a silent wrong answer 25 minutes later if it
breaks, and a second to check here.

## What the `untilize` run has established so far

Worth knowing, because it means these paths are no longer speculative:

- The ledger expands to 112 in-scope and 96 out-of-scope cases. The 96 are `port_scope` doing its
  job: non-tile-aligned `bfloat8_b`, which would otherwise have been force-graded as in-scope.
- Stratification picks `dtype` and `kwargs.memory_config` as axes, four strata, no axis dropped.
- The emitted routing test renders live `MemoryConfig` values as `ttnn.DRAM_MEMORY_CONFIG`, is valid
  Python, and covers all 96 cases grouped under the condition that rejects them.
- `ttnn.untilize` has no golden, so the golden check warns and passes, as designed.
- A cold build on CIv2 takes about 12 minutes, so a failed run costs roughly 25 minutes to reach the
  same point again. It is worth reading ahead for the next failure rather than fixing one at a time.

### A merged port can break the next port, if one name is a suffix of the other

`untilize` merging did break the pipeline, in a way worth remembering because the class is broader than
the instance. `scaffold.py` asked whether a CMake list already carried an entry with `entry in text`.
`tilize/codegen/kernels/*.cpp` is a substring of `untilize/codegen/kernels/*.cpp`, which main gained
hours earlier, so scaffolding `tilize` added no globs, reported `kernel_globs_added: []` with no error,
and then failed its own verification -- which looks at the start of a line and was right.

Entries are now compared as whole lines. The lesson for anything else that grows a check like this: op
names nest, and `tilize`/`untilize` is not the only pair -- anything ending in `pad` collides with `pad`.
The old fixture could not have caught it, because it carried no codegen globs for anything to be a
prefix of; the replacement mirrors `main` as it actually stands.

Worth noting what worked while that failed. The publish post-step committed the partial scaffold to the
branch anyway and chained attempt 2, so the bounded chain behaved as designed on its first real outing,
including on a failure it had never been shown.

What is *not* true, and was written here first: that an ungraded attempt is allowed once and no more.
`should_chain` returns `args.prev_failing < 0` when the current attempt produced no count, and a branch
that has never graded carries `-1` forever, so that test keeps passing and ungraded attempts chain all
the way to the six-attempt cap. The no-progress stop only starts biting once some attempt has produced a
real failing count for the next one to fail to beat. An op that cannot compile therefore burns the full
cap -- roughly six agent-hours on a card -- without ever producing the signal the stop was built to read.
That is the argument for making a compile failure cheap rather than for tightening the cap.

## The first real `tilize` run, and the deadlock it exposed

Run [31823944470](https://github.com/tenstorrent/tt-metal/actions/runs/31823944470) is the best output
this harness has produced and it is also how the resume path was found to be broken. Worth reading in
order, because each step looked fine at the time.

Attempt 1 wrote a whole port -- 2,457 lines over 20 files on `ebanerjee/port-tilize` at `b580709de46`:
six codegen components, eight vendored kernels, CMake registration, the generated routing test, and the
`implementation` kwarg wired through `tilize.cpp`, `.hpp` and `_nanobind.cpp`. It got two builds in
seventy minutes. The first reported three errors; the second fixed one of them and left the other two
at the same line and column, which bought nothing and spent fourteen minutes. Then the job ended,
publish committed the work and chained attempt 2 unprompted, which is exactly the design.

Attempt 2 [31829482825](https://github.com/tenstorrent/tt-metal/actions/runs/31829482825) then died in
a pre-step, before its agent existed, and this is the part that matters:

**A resumed branch whose port does not compile could not be resumed at all.** The baseline builds the
code already on the branch in order to measure what that code fails. Attempt 1 left two compile errors,
so the baseline build failed, so `report_baseline` returned 1, so the step failed and the job ended --
and attempts 3 through 6 would each have died in the same step on the same two errors, with no agent
ever handed the chance to fix them. The one state a half-finished port is most likely to be in was the
one state the resume path could not accept.

It now starts the agent anyway when the compiler is what objected, handing it the diagnostics as its
work list. Gated on the compiler specifically: a baseline that failed with no diagnostics failed for
some other reason -- no card, a bad checkout, a broken launcher -- and starting an agent against that
spends a budget on something no edit to the port can reach. A fresh port whose *scaffold* will not
compile is a harness bug and still stops the run.

### The agent could not see anything the run learned before it started

Found while fixing the above, and older than it. The baseline is a pre-step, so its output went to the
job log and nowhere an agent can read -- while the prompt claimed the agent would find it "in your
first tool output". Nobody noticed because a fresh port does not need it. A resumed port does: the
measured list of cases the existing port fails is the entire reason to resume rather than start over,
and no agent had ever seen one.

There is now a `port-brief.md` in the worktree, written before the agent starts and the first thing the
prompt tells it to read: the baseline, the resume work list, inherited compile errors, or the fact that
the tree does not compile. It is in `.git/info/exclude`, so `git add -A` leaves it out of the published
commit -- it informs the port without becoming part of it, and the write-path guard is never asked
about it.

Two smaller things in the same area. Diagnostics now survive into the next attempt, carried between two
fences in the commit message, so a resumed run does not spend its first build rediscovering what the
last one already knew. And when a build reports an error the previous build also reported, unchanged,
the tool says so in as many words rather than leaving it in a list to be noticed.

A note on those fences: they were `--- ... ---` first. The sed range matching them lives in
`port-op.md`'s YAML frontmatter, `---` is how frontmatter ends, and writing it there truncated the
frontmatter for every reader of that file. The first symptom was `selftest.py` losing the `mcp-scripts`
key three hundred lines from anything related. They are `=== ... ===` now, and a check asserts no line
inside that frontmatter can be mistaken for its end.

### What the fixed resume actually did, and what it found

Run [31832493570](https://github.com/tenstorrent/tt-metal/actions/runs/31832493570), the first resume
after the deadlock fix, is the proof the loop works and the first graded verdict `tilize` has ever had.

The baseline build failed exactly as it had for attempt 2 -- same two errors, same step. This time the
step reported success, the agent started, read the errors out of `port-brief.md`, fixed them, and its
first build passed. Roughly fifteen minutes from job start to a compiling port, on the branch that
forty minutes earlier could not be resumed at all. It then ran three builds and two verifies inside its
budget, which is the throughput the round trip allows.

Both verifies graded 170 of 170 in-scope cases failing, with zero prototype gaps -- the count did not
move between them. Two causes, and both are host-side translation, not kernels:

- **Nineteen cases: the kernels want *named* compile-time args and the program factory passes
  positional ones.** `reader_stick_interleaved_unified.cpp` calls
  `get_named_compile_time_arg_val("elem_size")`, `"tile_height"` and more, so `get_named_ct_arg`
  reaches `__builtin_unreachable()` during constexpr evaluation and the ncrisc build fails with
  `Failed to generate binaries`. This is the argument-contract lesson again in a new costume: named CT
  args are a newer generator API than anything merged `untilize` used, so there is no shipped port to
  copy the shape from and the builder under `.codegen/ops/tilize/spec.py` is the only source.
- **Six cases: `supported_by_codegen()` rejects configurations it has to accept.** `tilize.yaml`
  carries no `port_scope`, so port scope equals codegen scope and *every* ledger case is in scope. A
  `TT_FATAL` at `tilize.cpp:103` on an in-scope case is the gate being stricter than the manifest.

Worth noting what this unlocks procedurally: a graded count now exists on this branch, so the
no-progress stop can finally engage. Until this run every attempt published `-1` and the chain's only
bound was the six-attempt cap.

### What is still expensive, and the number to beat

The binding constraint is that a compile error costs a round trip to CIv2: fourteen minutes measured,
of which the build is about eight and the rest is queueing. The agent is using a card-backed CI job as
a syntax checker, and it holds an N150 doing no device work while it waits.

The cheap version of the fix is real and specific: configure once with
`CMAKE_EXPORT_COMPILE_COMMANDS` -- the `clang-tidy` preset already sets it -- then check the port's own
six translation units with `-fsyntax-only` and the exact flags cmake recorded. Seconds per file, no
card, and it catches every error both of attempt 1's builds were spent on. Whether the bare N150 host
can do it is unknown, because tt-metal's CI builds inside a container, so there is now a
`continue-on-error` step that records what the host has -- cmake, ninja, clang++, ccache, docker,
cores, disk. Read it off the next run before designing against it.

## The generator withdrew the implementation selector, and that is a different kind of drift

On 2026-08-17 the generator landed [#403](https://github.com/tenstorrent/tt-dm-codegen/pull/403) as
`f744aefb8`, and it removed the thing every leg of this harness used to select an implementation. A
ported op's public entry now keeps the exact signature it had before the port -- no `implementation`
kwarg, no selector enum, no `parse_implementation` -- because which prim serves a call is an internal
decision rather than something a caller expresses. The choice is still checkable through two
verification-only entries, `<op>_force_native` and `<op>_force_codegen`, bound under
`ttnn._ttnn.operations.<family>` and deliberately never republished as `ttnn.*` ops.

Worth knowing that #50178 -- merged `untilize` -- shipped the selector to review as a hedge and had
to remove the whole surface in a follow-up. `origin/main` now carries `untilize_force.hpp`, the
forced pair in `untilize.cpp`, and their `mod.def` bindings, which makes it the authoritative
in-tree reference for this shape. Read it rather than reconstructing the rules from the guide.

Two details in it are easy to get wrong and expensive to get wrong. `force_codegen` must `TT_FATAL`
outside the support scope rather than fall back, because a forced leg that quietly serves native
turns every bit-exactness and performance number gathered through it into a native-vs-native
comparison reported as agreement. And it must not consult `is_demoted`, which is the routed entry's
business alone, so that demoted-but-correct cases are still measurable.

### What this cost on our side, and what it did not

`measure.py` invoked all three legs through the kwarg, so every band was unrunnable. It now resolves
the two forced entries once from the manifest's `call_parity` and calls the plain public entry for
the routed leg. A missing `force_codegen` is fatal with a message naming the contract; a missing
`force_native` degrades to the public entry, because the native baseline is measured on clean main
where no port exists and the public entry *is* native there. The emitted routing test had the same
problem and took the same fix: its reference leg calls the forced-native entry, because using the
public entry for both sides would compare the router against itself and pass wherever it routed.

The port on `origin/ebanerjee/port-tilize` still exposes the old surface -- `tilize.cpp` takes a
`const std::string& implementation` and the nanobind file advertises it. That is now the whole of
what stands between that branch and a re-baseline, and it is deliberately still there; see the note
on update mode below.

### The classifier has to judge the shape, not the field

The first version of `drift.py` treated any `call_parity` change as harness-level and refused. That
is wrong in a way worth recording, because it is the difference between update mode working and
update mode being permanently stuck: this field is *expected* to move, and a change to it is only
beyond an agent when the harness cannot invoke the new shape at all. Once `measure.py` and the
emitter understood forced entries, the same drift became ordinary transliteration work -- the port
has to bind two entries it does not have.

So `call_parity` is checked rather than bucketed. A contract naming the routed entry and both forced
legs, with no field nothing here reads, is `agent` work. An entry that still carries a selector, a
missing leg, a third leg, or a non-mapping is `refuse`. Today's drift on `tilize` reads as `update`
because of that check and read as `refuse` before it, and nothing about the manifest changed in
between.

`invalid_selector_message` is classified as noise for a related reason: it was the error text the
selector rejected bad values with, nothing in this harness ever read it, and letting its removal
refuse a run would stop update mode over a field with no reader.

### A generated file is the most likely place to leak a private name

The same commit banned naming the private repository, its manifests, its ledger and its phase
numbers in anything landed into tt-metal, on the grounds that a public comment pointing at a
repository the reader cannot open is worse than no comment. The emitted routing test is the one file
this harness writes wholly by machine, which makes it the one nobody proofreads.

It was carrying four such names. The header said `AUTO-GENERATED`, named a script no tt-metal reader
can run, and explained itself in terms of a coverage ledger and an `invalidate_vector` that exist
nowhere in tt-metal. Less obviously, every case comment came from a ledger reason string prefixed
with the name of the manifest field that rejected it, so the file grouped its cases under
`port_scope: ...`. Those reasons now state the constraint -- "bfloat8_b is not served by the codegen
path" -- which reads better internally too.

`scaffold.render_routing_test` now refuses to emit a file containing any of them, which is how the
`port_scope` leak was found: the guard was written from the checklist and fired on the first
realistic fixture it saw. `codegen` alone is deliberately not on the list, because `prim::<op>_codegen`
and `codegen/` are real tt-metal names; what cannot ship is prose about a private generator.

### Kernel templates moved out from under the ports

`f895be71b` moved the shared kernel templates to a shared location, and `origin/main` now globs
`common/kernels/codegen/*`. Nothing has been done about this yet, and it is the likely first real
exercise of the re-vendor policy: a port that vendored its own copy of a now-shared template has a
stale duplicate rather than a compile error, which is the failure mode that is invisible until the
template's contents diverge.

One operational note that cost some confusion: the local `main` in this checkout was a week stale, so
`untilize`'s merged port and its forced entries appeared not to exist. Fetch before concluding
anything about what has landed.

## Update mode is one workflow, not two

Decided 2026-08-17. `port-op.md` serves the full port, the resume and the update, and the only thing
that distinguishes an update is that someone set a target generator, a PR number or an intent.

The reasoning is that the differences are small and local while the agreement is large. An update run
needs the credential pre-step, the proxy fix, both generator checkouts, category derivation, resume
detection, chain-state parsing, the scaffold, the baseline, all three launcher tools, publish and its
chaining, the artifact preservation and the scratch-ref sweep -- all unchanged. `port-op.md` is 766
lines of frontmatter to 378 of prompt, and the frontmatter is the part a second workflow would have to
duplicate or import. Against that, update mode adds a PR-to-branch resolution, a review fetch, a
target ref and an intent string.

There is also a mechanical reason: the chain re-dispatches itself by workflow filename
(`PORT_OP_WORKFLOW` in `dispatch.py`) and `workflow_dispatch` resolves the definition from the ref it
is handed, so two workflows would mean every chained attempt has to know which of the two it is.

And it follows the grain already set by resume, which is *detected, not declared* -- a tracked
`codegen/` directory is what sets `PORT_RESUME`, and nobody passes a mode flag. Setting a target
generator is the equivalent declaration for an update.

The cost lands on the prompt, because gh-aw markdown has no conditionals and mode-conditional prose is
where an agent gets confused. That is paid with `port-brief.md`: it already carries what varies per run
(baseline, resume work list, inherited diagnostics, drift report) and now also carries the intent and
the review. The workflow prompt keeps only what is true of every run, plus a short passage saying that
a brief with an intent or a review section means the port already passes and the job is the smallest
change that answers what was asked.

### What the three inputs are, and what they are not

`pr`, `codegen-target` and `intent`, all optional and all blank for a port or a resume. They are
statements of *intent*, which is the line that keeps the "no inputs about previous runs" rule intact:
the attempt number, the failing counts and the generator pin are still read off the branch's own commit
trailers, and a person updating a port never has to know what the last run was told.

`pr` resolves to its head branch before the checkout, because everything downstream works in branches
-- publish pushes to one, the chain re-dispatches with one, the write-path guard diffs against one's
HEAD -- and a pull request is a view of a branch rather than a separate kind of thing. A closed PR and
a fork PR are both refused: the fork case matters because publish must be able to push, and the whole
design rests on work being preserved by the job ending rather than by the agent asking.

The review is fetched from three endpoints, because a review is spread across three and a reviewer's
actual objection lands in whichever box they were typing in: inline diff comments carry file and line,
review bodies carry the verdict, issue comments carry the conversation. Inline comments whose `line` is
null are dropped as outdated -- a demand that was already met reads exactly like one that was not.

Worth being explicit about the security shape, because this is the first untrusted text the pipeline
feeds an agent. Anyone who can comment on a public pull request can put words in front of an agent that
can edit this repository. So it is data at every step: `jq` reads and writes it, no shell ever expands
it, and the brief introduces it as quotation -- comments can be wrong, already answered, or asking for
things the manifest forbids, and where one conflicts with the agent's rules the rules win and the
disagreement goes in the PR. Treating review text as instructions-by-default would make the comment box
a way to drive this pipeline.

### The repin, and the contract it creates

Two halves, and they only make sense together: moving a port to a new generator is the dangerous
operation, and the regression check is what makes it survivable.

**The repin is a directory swap.** After the drift comparison has run and not refused, `.codegen-target`
replaces `.codegen` and `PORT_CODEGEN_REF` is updated to the target's sha. From that point the target
*is* the generator for the run -- kernels are vendored from it, the prototype leg that grades
correctness is built from it, and the `Port-generator` trailer published at the end names it, so the
next attempt inherits it as its pin with no further work. Recording the repin is therefore free: the
trailer already reads `--codegen-ref`, which already reads `PORT_CODEGEN_REF`.

A swap rather than a second path threaded through every consumer, because `.codegen` is what
`scaffold.py` vendors from, what the manifest is read out of, what the dispatch params name, and what
the measure job checks out. A run with two generator trees in play is a run where any one of those could
be reading the wrong one. `PORT_CODEGEN_REF` has to move with the tree because the resolve step asserts
the two agree, which is exactly the check that would otherwise trip. A target that resolves to the
commit the port is already pinned to is not an error -- naming a branch that has not moved is the
ordinary way to discover it has not -- and repins nothing.

**Green-on-entry is measured, not assumed.** `gate.py`'s correctness band now emits `passing_ids`, the
complete and deliberately untruncated list of in-scope cases a tree got right. The baseline records that
set on arrival, before the agent exists, because that is the only moment it is true: re-measuring later
measures the agent's code, and the previous run's artifact may have expired or may never have existed
for a hand-written branch.

Every later `verify` diffs its passing set against the recorded one. Anything lost is printed *above*
the verdict, because an agent reads top to bottom and acts on the first thing it sees, and the call
never exits as a win -- a win is what makes the agent open a pull request. `publish` carries it too, as
a `Port-regressed` trailer written only when non-zero, and the job summary leads with it and withholds
the `gh pr create` invitation.

The reason a count could never have caught this: a port can fix three cases and break two, and the
failure count is identical before and after. Only the identities give it away. Chaining is the one place
where a regression argues for *continuing* -- stopping would leave the branch worse than it was found,
so the next attempt inherits the list and its first job is to undo the damage, bounded as ever by the
attempt cap.

## Four ways the harness misreported its own state, all found in one run

`tilize` attempt 3 is the case study. It inherited a bit-exact port, spent about two hours, and
produced nothing: no commit, no pull request, no statement. None of that was the port's fault, and all
four causes were in the reporting rather than in the porting.

**A count belongs to the tree it was measured on.** Attempt 2 stamped `Port-failing: 52` onto a tree
whose real answer was zero, because the agent kept editing after its last verify and `publish` read
the newest verdict it could find. The next run reads that number off the branch to judge whether the
attempt before it made progress, so a stale count can stop a chain that was working or continue one
that was not. `publish` now compares the worktree's tree hash against the tree the last verify
recorded, and publishes `-1` -- which already meant "no graded count" -- when they differ. Measuring
again is not an option there: the agent has ended and the card is gone, so the choice is between the
truth and a plausible number.

**Progress changes axis when correctness goes green.** The no-progress stop compared failing counts, and
a correct port's failing count is zero and cannot fall, so every attempt after the first green one read
as making no progress. `tilize` -- 170 of 170 bit-exact, 19 of 24 configurations too slow -- is half a
port reported as a dead end. There is now a `Port-slow` trailer alongside `Port-failing`, and once
`failing` is zero the stop judges the slow count instead. A correctness-green attempt that has never
managed to grade performance gets exactly one more try, not the whole cap.

**A pull request is made from a diff, so a run that changes nothing cannot open one.** That is the
ordinary ending of a resumed run whose port is already correct, and the prompt had made it illegal:
open a PR whenever correctness passes, use the no-op only when it fails. Attempt 3 called
`create_pull_request` five times, was told `failed to generate patch` five times, and stopped. The
prompt now routes an empty `git status` to the no-op with the finding in its message -- which is
published as an issue, so it needs no diff -- and says outright not to retry an output that already
refused. `publish` also writes a job summary in both cases, including the `gh pr create` command for a
branch whose port is correct and unreviewed. The pipeline could run that command itself, but whether a
measured negative result is worth someone's time is a judgement, and it belongs to the person whose
name goes on the PR.

**Two measurements settle noise; four do not.** `refuse_pointless_dispatch` deliberately allowed
re-verifying a tree that built, on the grounds that measurement is noisy in a way compilation is not.
Attempt 3 measured one unchanged tree four times at roughly forty minutes each. The allowance is now
two per tree *per band* -- per band because `correctness` then `performance` is the ordinary way
through a run rather than a repeat, and an edit resets the count because the next verify is measuring
something new.

## Two things the ledger and the generator were quietly getting wrong

Both were found on 2026-08-14 while sizing `tilize` before spending device time on it, and both had
been silently wrong for the whole `untilize` effort.

### A port is a transliteration of one generator commit, and the pin has to say which

`PORT_CODEGEN_REF` named `codegen_agentic_port`, a branch. Read fresh on every attempt, a resumed run
vendors kernels from a generator newer than the host code the previous attempt was written against.

This is what the `untilize` failures were. `untilize.yaml` records `ported_codegen_commit: 7f2930ff`,
and main's merged `writer_untilize_interleaved.cpp` reads three compile-time args and includes no
`rm_shard_split.h`. The same template on today's branch takes a shard-split contract wanting that
header and two further arguments. The port had one foot in each, which compiles, routes, and writes
uncorrelated data on 76 of 112 cases -- and for a day it read as a kernel bug, because nothing in the
harness could say which generator the port was a transliteration of.

The pin now travels in the branch's own commit trailer, read before the generator checkout so the
checkout takes it. Attempt 1 has no trailer and establishes the pin by recording what its floating ref
resolved to. Only a full sha counts, so a branch published by the older harness -- whose trailer holds
the branch *name* -- correctly reads as unpinned rather than silently resolving to today's tip.

### The sweep grid was a different grid every time it was expanded

Upstream suites build their shapes with `gen_shapes(start, end, interval, num_samples)`, which draws
each shape with `random.randint`, unseeded, while `parameters` is evaluated at import. The *count* is
stable, because the try budget is ten times the sample count. The shapes are not, and `case_id` is the
point's position in the grid.

So two expansions agreed on every identifier and disagreed about which tensor each one named, and three
things were standing on that:

- The prototype pass set and the correctness band are separate `measure.py` processes in the same job,
  neither handed a ledger. `codegen_untilize[57]` passing in the pass set excused a port failure on a
  different shape entirely -- a confident, wrong attribution, in the one place whose comment promises
  the only safe way to be wrong is too harsh.
- The no-progress chain stop compared this attempt's failing count against the last one's, which were
  two counts over two different case sets.
- `measure.prototype_key`'s `ledger_sig` hashes the case ids, so it could not notice any of it.

`ledger.py` now seeds per op before the import. The existing "expansion is deterministic" test could
never have caught this, because its fake grid holds literal shapes; the replacement redraws on every
access the way an import does, and was checked to fail when the seed is removed.

### While in there: the input memory config is not an axis anything measures

`_signature` deduplicates on `[shape, dtype, layout, kwargs]`, and `kwargs` comes from the manifest's
`vector_map`, which for both `untilize` and `tilize` maps only the *output* memory config. The sweeps
vary `input_a_memory_config` across DRAM and L1, so the ledger collapses that axis by half -- correctly,
because `measure.make_input` passes no `memory_config` at all and every input lands in the default. The
collapse is honest; the gap behind it is real. **No op is measured with an L1-resident input**, and the
DRAM/L1 columns in every perf table so far are output configs. Fixing it means teaching `make_input`
the input config and roughly doubling the ledger, which is why it is written down rather than done.

It also means `tilize`'s ledger is about 166 cases -- 96 nightly, 16 `codegen_dtype`, 54
`broaden_suite` -- against `untilize`'s 184, and not the 332 the raw grid suggests. So no cap is needed,
and capping would have been the wrong instinct anyway: the correctness and prototype bands are uncapped
on purpose, and a partial prototype pass set cannot attribute a failure to the generator.

### What the first `tilize` baseline established

Run [31823944470](https://github.com/tenstorrent/tt-metal/actions/runs/31823944470), measured on CIv2 by
[31824246253](https://github.com/tenstorrent/tt-metal/actions/runs/31824246253):

- **170 in-scope cases and zero out-of-scope ones.** The estimate from the grid was about 166, so the
  arithmetic holds. The zero is `tilize.yaml` having no `port_scope` block, which `ledger.py` reads as
  "port scope equals codegen scope". It follows that the generated routing test has nothing to assert
  and the correctness band's routing check has nothing to check -- unlike `untilize`, where 96
  out-of-scope cases were most of what that test covered. Do not read a green routing check on this op
  as evidence the routing check works.
- **The prototype passes 170 of 170.** So the parity bar is the strict one, with nothing excused, and
  the prototype leg is now exercised end to end on an op it was not written against.
- Stratification picked its axes with none dropped.
- The baseline's own port-side legs fail with `ttnn.tilize has no implementation kwarg in this build`,
  which is the correct thing to hand an agent that has not written anything yet.

## Checkout was the slowest thing in the pipeline, and it is not our fault

`build-artifact.yaml` checks out with `fetch-depth: 500` and `fetch-tags: true`. On a repo with ~1300
tags that means fetching 500 commits of trees and blobs per tag: measured off-CI at over 3GB and more
than 6 minutes without finishing, against `--filter=tree:0` at 34MB and 3 seconds for the same commit.

That has always been wasteful, but it only started failing because GitHub's fetch throughput drifted:
the same checkout on `main` went from about 1.5 minutes to about 9 minutes over two days, and the
first `baseline` probe hit the 20 minute step timeout on both build jobs. A plain rerun then passed in
7m48s, so this is a slow-afternoon problem sitting on top of a fetch that was always too expensive --
not the scratch ref, which the server serves normally.

`port-measure.yaml` therefore passes `checkout-filter: "tree:0"` to `build-artifact`. That input
already existed and reaches both build jobs, the release build directly and the wheel build through
`wheels.yaml`, so no shared workflow was modified. The filter keeps every tag and the full worktree at
the target commit and defers only historical trees, so the wheel version is unchanged: `git describe
--abbrev=10 --first-parent`, the command in `cmake/version.cmake`, returns the identical string under
`tree:0` and under a full clone. That equivalence was checked directly rather than assumed.

Measured in production across the next two runs: checkout went from 7m48s to **32 seconds**, twice,
and a whole `build` round trip now costs 8.5 minutes against the 57 the first baseline took. The debs
came out stamped `0.77.0~dev20260810+44.964eda5eba`, which is the same string a full clone produces.

Nothing else in the repo uses `checkout-filter`, so we are the first, and a partial clone does fetch
lazily if something later walks history. Nothing in this build does. If a future step starts reading
old trees it will silently refetch them and the win will quietly disappear.

## A non-zero exit is not a result, it is a broken tool -- and the agent believes it

Read this with the section below it; they are two halves of one lesson, and between them they
account for both failed agentic runs.

gh-aw runs an mcp-scripts handler through `execFile` and **rejects the promise on any non-zero
exit**. What reaches the agent is not our carefully worded report but:

    Command failed: /…/wait.sh (exit code: 4)
    stdout:
    build is still running after 8 minutes… Nothing is wrong; it is not finished.

The frame beats the text. The agent is told the tool broke, and the reasonable response to a broken
tool is to call it again -- which is precisely what it did, twice, in two different ways:

- **2026-08-12, run 1.** The gateway cancelled `build` at 60s. Four retries, four stranded CIv2
  builds. That one really was a broken tool, and the agent's reading was correct.
- **2026-08-12, run 2.** `build` returned a genuine compiler diagnostic and exited 1; `wait` said
  "still going" and exited 4. The agent re-dispatched a **byte-identical tree** twenty seconds
  later -- same tree object, `9d7c71de75e9`, confirmed by diffing the two scratch commits. It never
  acted on the diagnostics, because as far as it could tell there weren't any.

So the exit code is no longer allowed to carry meaning to the agent. `--as-tool`, which all three
tools pass, makes any *delivered* answer exit 0 and puts everything in the text. Three consequences
worth preserving:

- **A refusal is an answer.** `Refusal` exists as a distinct exception for this: "you changed the
  pipeline" and "that tree is unchanged" tell the agent what to do, so they exit 0 under the flag.
  A genuine harness failure -- no credential, push rejected -- keeps its non-zero exit, because
  that one *is* a broken tool and should look like one.
- **Workflow steps are unaffected.** They consume exit codes properly and do not pass the flag.
  `selftest.py` checks both halves.
- **The reports now say which kind of thing they are** in their first line: `BUILD PASSED`,
  `BUILD FAILED -- the compiler rejected your code. The tool worked`, `STILL RUNNING -- this is not
  a result and not a failure`. That phrasing is doing real work; do not trim it for brevity.

`refuse_unchanged_build` backs this with scaffolding rather than words: a second `build` of an
unchanged tree is refused outright, since compilation is deterministic and the answer cannot differ.
`verify` is deliberately exempt, because measurement is noisy and re-measuring can be legitimate.

## A tool call may not last ten minutes, so the launcher cannot wait inside one

This is the constraint that shapes the agent's tool surface, and it is not negotiable from here.

The MCP gateway cancels a tool call when its own deadline passes. That deadline defaults to **60
seconds**, is set by `engine.mcp.tool-timeout`, and gh-aw refuses to compile a value above **600s**
-- the error names the ceiling, so this is a deliberate limit rather than something to be talked
around. The per-tool `timeout:` values in `port-op.md` are a different thing entirely: those bound
the handler process, this bounds the request. Setting the per-tool value to 5400 did nothing,
because the gateway had already given up ninety times over.

A build is 8.5 minutes on a good day and a verify is closer to forty, so no ceiling that gh-aw
permits is enough. Waiting inside the call was never going to work.

So `dispatch.py` splits: `--start` pushes and returns a handle, and `--wait HANDLE` blocks for up to
`WAIT_BUDGET` (420s) before returning "still going". The agent sees three tools -- `build`, `verify`,
`wait` -- and a build costs two or three `wait` calls, a verify five or six. Three things about this
are worth keeping in mind if you touch it:

- **"Still going" must never read like a verdict.** `--wait` exits **4** for this, distinct from
  every gate exit code, and says in words that nothing is wrong. An agent that reads "not finished"
  as "your port is broken" will start editing code that compiles.
- **The budget has to leave room for what happens after the run ends.** Collecting means downloading
  an artifact and laying it over the tree. 420s of polling inside a 540s tool timeout inside the
  gateway's 600s is the arithmetic; if you raise one, raise them in that order.
- **Starting cancels whatever it supersedes.** `retire_inflight` exists because the failure above
  left four builds on CIv2 at once. A dispatch nobody is holding a handle to is a card nobody is
  reading, and the far side has no way to notice the near side stopped caring.

Verified end to end against real infrastructure on 2026-08-12, not just in `selftest.py`: start
returned a handle in 28 seconds, two `wait` calls reported the run still going with cumulative age
accounted correctly across calls, the third collected `build OK` at 8.7 minutes, and both the
scratch ref and the job record were gone afterwards.

## The one thing CIv2 could not do, and now can

**Fixed on 2026-08-13.** [tt-flux#748](https://github.com/tenstorrent/tt-flux/pull/748) added one line
to `apps/restricted-proxy/configmap/squid.conf`, and a throwaway probe on
`tt-ubuntu-2204-N150-viommu-stable` confirmed the door is open: `CONNECT api.githubcopilot.com:443`
now tunnels and the far side answers, where before squid answered 403 without the request ever
leaving the cluster. The API's own reply to an unauthenticated probe is a 404, which is the point --
a 404 is the service talking, and a 403 was the proxy refusing to let us talk to it.

**The agent job moved onto `tt-ubuntu-2204-N150-viommu-stable` on 2026-08-13.** Half the collapse, and
the half that costs rather than saves: build and measurement are still dispatched into
`port-measure.yaml`, because `build-artifact.yaml` is a reusable workflow and a step cannot call one,
so the job now holds a card-bearing runner for hours while doing no card work. It was taken anyway
because being local to the hardware is the precondition for the other half. Until a local incremental
build replaces the round trip, this trades an N150 for nothing, and that is the right thing to fix next.

So the reason for the two-workflow split is gone, and the collapse it was blocking is now the plan:
one in-cluster job with a local incremental build, which is both simpler and far faster per cycle.
The hosted runner was never a good host for this -- four cores, and the probe watched its build rate
collapse from 145 targets a minute to 10-15 as it crossed into the ttnn unity builds -- so the split
was paying for egress with compute. In-cluster stops paying either way.

Two things do not follow from this and must not be assumed. AWF's own allowlist still governs what
the agent may reach; the cluster proxy was only ever the second gate in series. And the agent still
runs inside AWF's container, whose mount set is `RUNNER_TEMP/gh-aw`, the workspace, the tool cache
and `/tmp/gh-aw` -- nothing there passes `--device /dev/tenstorrent`, so being in-cluster does not by
itself put a card in the agent's hands.

### What a full in-cluster agent run established

Run [31737747751](https://github.com/tenstorrent/tt-metal/actions/runs/31737747751) put a whole gh-aw
agent on `tt-ubuntu-2204-N150-viommu-stable` and it worked: the agent started, reached the model API,
reasoned, and produced output. That is the collapse's central premise, no longer inferred from a
`curl`.

The runner host is a good host. It is **not containerized**, runs as `ubuntu` with `HOME=/home/ubuntu`,
`/dev/tenstorrent/0` is present and mode `crw-rw-rw-`, and docker 29.7.2 answers. So a step on that
host can build in a container and talk to the card.

**Where an `mcp-scripts` handler executes is settled, and not by the probe.** gh-aw starts the handler
server as an ordinary step on the runner host -- its log says
`Working directory: /opt/runner/_work/_temp/gh-aw/mcp-scripts` -- and handlers are child processes of
it, so they run on the host, outside AWF's container. The existing pipeline already depends on this and
has for every successful run: `dispatch.py` reads its push token from `$HOME/.port-dispatch/token` on
every tool call, and `$HOME` is in none of AWF's mounts. A handler that ran inside the sandbox could
never have read it. That is also what makes `$HOME/.port-harness` a safe place to keep the launcher
away from the agent.

Two things this cost, both worth not repeating:

- **CIv2 proxies loopback, and that breaks gh-aw's readiness check.** The first attempt died with
  `MCP Scripts server failed to start after 10 seconds` while the server's own log said
  `Server is ready to accept requests`. CIv2 exports `http_proxy`/`https_proxy` to every step, and its
  `no_proxy` names three cluster services and not `localhost`, so the readiness poll to
  `localhost:3000` was sent to the restricted proxy, which will not tunnel to loopback. Any workflow
  moved onto this pool that talks to itself over HTTP hits this. Fix it in a pre-step, not a
  workflow-level `env:` block: Actions env keys are case-insensitive, so declaring both `no_proxy` and
  `NO_PROXY` is a duplicate-key parse error, and declaring one leaves the runner's value for the other
  spelling in place -- and both are needed, because curl reads the lowercase one and node's proxy
  agents read the uppercase one.
- **Do not ask an agent to probe its own sandbox boundary.** The probe's prompt asked it to read a
  marker in `$HOME`, compare `hostname` and `pwd` against a handler's, and report device and docker
  reachability. It refused, correctly and at some length, as reconnaissance toward sandbox escape and
  credential access -- gh-aw's own system prompt forbids exactly that, and a friendly framing does not
  and should not change the answer. Facts about the boundary have to come from ordinary workflow steps,
  which is where all of the ones above came from.

The diagnosis below is kept as the record of what was wrong, because it took three misleading layers
to reach and the reasoning is worth not repeating.

The job ran on CIv2 end to end up to the agent itself: the runner hosts gh-aw's servers, the
build succeeds, the baseline passes, and the agent process starts. Then every model request fails,
and squid's access log says why in one line, repeated 52 times:

    172.30.0.30 api.githubcopilot.com:443 10.43.242.248:3128 CONNECT 403 TCP_TUNNEL:FIRSTUP_PARENT

CIv2 has no direct egress; everything leaves through `proxy.restricted-proxy.svc.cluster.local`.
AWF handles that correctly and without being told -- its squid carries
`cache_peer proxy.restricted-proxy... parent 3128` and `never_direct allow all`, so the agent's
traffic is offered to the cluster proxy exactly as intended. The cluster proxy answers 403:
`api.githubcopilot.com` is not on its allowlist. That host is the only upstream AWF ever asks for,
so there is nothing to work around in the workflow, and no amount of configuration on this side
changes a decision made in the proxy's allowlist.

This is why the CIv1 cloud-VM pool worked: direct egress, no allowlist in the path. It is also why
the earlier failures were so confusing -- three separate layers (missing `netstat`, proxied
loopback, unpriced model) each had to be cleared before the real one became visible, and each
presented as a silent connection that did not happen.

**The ask for infra was one line**, and what landed was `acl toallow dstdomain .githubcopilot.com` --
a domain entry rather than the single host that was asked for, so the four hosts gh-aw's own allowlist
names are covered too: `api.business.githubcopilot.com`, `api.enterprise.githubcopilot.com`,
`api.individual.githubcopilot.com`, and `telemetry.enterprise.githubcopilot.com`. Nothing here has to
change if a later gh-aw feature reaches for one of them.

## A secret in an mcp-scripts tool is not hidden from the agent

Worth knowing before anyone writes another gh-aw tool that needs a credential, because the natural
way to do it is wrong and nothing warns you at runtime. The plan for the dispatch pipeline assumed
the host-side tool scripts were outside the agent's reach and only needed checking; they are not.

`${{ secrets.X }}` inside an `mcp-scripts` `run:` block is interpolated **verbatim** into the
generated handler at `${RUNNER_TEMP}/gh-aw/mcp-scripts/<tool>.sh`. There is no hoisting into an env
var. And awf mounts that whole directory into the sandbox read-only, twice:

    --mount "${RUNNER_TEMP}/gh-aw:${RUNNER_TEMP}/gh-aw:ro"
    --mount "${RUNNER_TEMP}/gh-aw:/host${RUNNER_TEMP}/gh-aw:ro"

So the agent reads the secret with `cat`. The env route is no better: awf is invoked with
`--env-all` and a fixed five-name exclusion list, so anything in the agent step's environment --
which includes everything any earlier step wrote to `$GITHUB_ENV` -- is inside the sandbox too. The
plan's fallback of a short-lived App token does not help either, since minting one requires a
private key that would be inlined exactly the same way.

What does work, and is what `dispatch.py` relies on: a **pre-step** writes the credential to a path
that is in none of awf's mounts, and the tool script names only that path. `pre-steps` are ordinary
workflow steps, so a step-level `env:` there stays step-scoped and never reaches the agent step's
environment. `$HOME/.port-dispatch/token` is outside `${RUNNER_TEMP}/gh-aw`, `$GITHUB_WORKSPACE`,
`$RUNNER_TOOL_CACHE` and `/tmp/gh-aw`, which is the full mount set.

Verified by compiling a scratch workflow both ways and reading the lock file, not by reasoning about
the docs. If gh-aw's generator changes, re-run that check before trusting this. The compiled
`port-op.lock.yml` should name the push credential in exactly three places -- the pre-step, gh-aw's
own log-redaction step, and the cleanup post-step -- and nowhere near the agent step or the handler
files. That grep is the check.

## What is *not* verified

**The agent now runs, and results now reach it.** By the end of 2026-08-12 it had collected a failed
build, a "still going", a passing build and a `blocked` verdict, and acted correctly on the first
three -- it read the compiler diagnostics, edited, and rebuilt to green. So the collect paths and the
start-and-collect split are proven from inside the sandbox, not just from a laptop.

What is still unexercised is everything *downstream of a graded verdict*: a pass, a demotion, a PR
body built from real numbers. No run has got there, because the write-path guard blocked the only
`verify` that ever reached the card.

What *is* checked, beyond `selftest.py`: both workflow files pass `actionlint`. The token boundary
was established by compiling and reading the lock file. The whole parameter round trip -- the JSON
the launcher puts in a commit message, through the validator lifted out of `port-measure.yaml` --
runs in `selftest.py`, along with eight injection attempts that must be rejected. And the agent
job's entire local half was rehearsed offline, in a throwaway worktree against a venv holding
nothing but PyYAML: `scaffold.py` ran clean, the kernel globs landed in the `file(GLOB_RECURSE)`
block rather than `target_sources`, and `commit_worktree` snapshotted all thirteen files --
untracked kernels and tracked CMake edits together -- while leaving HEAD and the index where the
agent left them. Reproduce it the same way if any of those change; it costs seconds and no cluster.

The trigger itself is no longer among the unknowns. Probed live on 2026-08-11, before the secret
existed, by pushing a scratch ref by hand from a laptop -- a user credential behaves the same way
the launcher's PAT will, so this needed nothing provisioned:

    git push origin "$(git commit-tree 'HEAD^{tree}' -p HEAD \
      -m '{"mode":"build","op":"untilize"}'):refs/heads/port-op-scratch/probe-1"

[Run 31508403789](https://github.com/tenstorrent/tt-metal/actions/runs/31508403789) started from that
push within seconds, from a workflow file that has never been on `main`. `resolve` parsed the
message, filled every default, and `check-harbor` and `build-artifact/parse-platform` both went
green on `needs.resolve.outputs.*`. Cancelled once the docker job started, since the remaining 20
minutes of CPU pool would have proved nothing further. Two minutes of hosted runner, no card, no
credits -- do this again after any change to the trigger, the message format or the resolve job.

### The launcher is proven, and here is what proved it

Run `dispatch.py` from a laptop against a normal checkout. It needs only `GITHUB_REPOSITORY` and a
token file, so it does not need the agent, the sandbox or a workflow to exercise it:

    GITHUB_REPOSITORY=tenstorrent/tt-metal \
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0= \
    python3 .github/scripts/port/dispatch.py --mode build --op untilize --repo-path .

Neutralising `credential.helper` matters: without it a developer's keychain answers the push and
`GIT_ASKPASS` -- the path that actually runs in CI -- is never taken. A `gh` OAuth token stands in
for `CODEGEN_REPO_TOKEN` faithfully, since it carries the same `repo` and `workflow` scope, and
locally uninitialised submodules reproduce the agent job's `submodules: false` exactly.

[Build 31548702163](https://github.com/tenstorrent/tt-metal/actions/runs/31548702163) went green in
8.5 minutes and [baseline 31549316125](https://github.com/tenstorrent/tt-metal/actions/runs/31549316125)
in 19. Between them they covered the whole round trip: `commit_worktree`, the `refuse_pipeline_edits`
guard, the `GIT_ASKPASS` push, `find_run` by head SHA, the heartbeat, both verdict reports, artifact
download and unzip, `adopt_workspace` laying the 259-line routing test into the checkout at its
repo-relative path, and scratch-ref deletion in the `finally`. HEAD never moved and the index was
never touched. `measure` correctly skipped itself in `build` mode.

Two things that were guesses are now settled: the `/codegen` volume mounts and is writable, and
`uv pip install graphviz pyyaml` works in a job that is not the long-lived `portdev` container.

What that leaves genuinely untested:

- **A passing verdict.** The bands now run and grade: `verify` returned `back-to-translate` on a real
  port on 2026-08-12, with real numbers behind every band. What no run has produced is a `win` -- so
  the demotion path, the per-stratum grading under a passing correctness band, and the PR body built
  from real figures are all still unexercised. They stay that way until an in-scope case executes.
- **The agent acting on a result.** It has now been handed two -- a failed build and a "still
  going" -- and misread both, because both arrived framed as tool errors. Whether it reads them
  correctly now that they arrive as content is the single open question, and the cheapest thing to
  watch: it shows up within about forty minutes of a run starting, at the first `wait` return.
- **The token pre-step under a real agent.** It ran and the push worked, so the boundary holds in
  practice, but the credential is `CODEGEN_REPO_TOKEN` and now carries `workflow` scope as well as
  `repo`. `refuse_pipeline_edits` is the only thing standing between that scope and a snapshot that
  edits CI. Do not weaken it.
- **Scale.** Both runs sampled 24 cases. The correctness band over all 208 is uncapped by design --
  the routing claim only means something across the whole out-of-scope set -- so if anything times
  out, it is still that.
- **No golden.** Confirmed live rather than assumed: the golden check emitted its warning and fell
  back to native, so correctness here means *matches native output*, not *matches torch*. A port that
  faithfully reproduces a native bug will pass, and the PR body should not be read as claiming more.
- **The deliverable contract, demotion logic and per-stratum grading**, which have still only been
  exercised against synthetic fixtures.

### How to run the dry run

Cheapest first, so each failure costs the least it can. Steps 1 to 4 are done.

1. ~~A manual scratch push with `mode: build`.~~ Done: trigger, `resolve`, front of `build-artifact`.
2. ~~A `baseline` on a card.~~ Done, twice -- once by hand and once through the launcher.
3. ~~`dispatch.py` end to end from a laptop.~~ Done: `build` in 8.5 minutes, `baseline` in 19.
4. ~~`--start` and `--wait` end to end from a laptop.~~ Done 2026-08-12: handle in 28 seconds, two
   "still going" replies, collected `build OK` at 8.7 minutes, ref and job record both swept.
5. ~~Push to `ebanerjee/port-op-dryrun` to trigger `port-op` itself, and watch the first `wait`.~~
   Done 2026-08-12, run 3: the agent read exit 4 as "call again", read a build failure, edited, and
   reached a compiling tree. `verify` ran on it and blocked on the write-path guard.
6. ~~A `verify` that reaches the bands.~~ Done 2026-08-12, run 31630655137: `back-to-translate` in
   about 20 minutes, routing clean, all 112 in-scope cases failing on one missing kernel header.
7. ~~Vendor `rm_shard_split.h` and re-verify.~~ Done 2026-08-12, run 31635005308: the header
   resolves, the wall band came back clean, and the remaining failures are the port's own.
8. Next: the two translation defects above -- the compile-time argument count and the
   `noc_write_row_split` signature. This is the first time the harness has handed back a defect that
   is purely the agent's to fix, so it is also the cleanest test of the loop the whole pipeline
   exists to close. Nothing in the harness is known to be in the way.

The agent's compiling tree is preserved locally at tag `port-run3-compiling-tree` (commit
`42487b776557`), independent of the run and its swept scratch refs. Every port file in it was
hash-compared against the worktree before the verify above, so it is a faithful copy of what the
agent wrote.

Measured per-cycle wall times, which the budget argument rests on: a `build` is 8.5 minutes and a
`baseline` 19, both well inside the 20 and 35-45 the design assumed. A `verify` is still unmeasured,
but it is a `baseline` plus the correctness band over 208 cases, so 19 minutes is its floor.

## The hard part of the `untilize` port itself

`supported_by_codegen()` has to reject non-tile-aligned `bfloat8_b` while *accepting* non-tile-aligned
`bfloat16`. The asymmetry is real and comes from the manifest: bf16 non-aligned is in scope because
the port transliterates `build_untilize_with_unpadding`, whereas bf8_b non-aligned only reaches the
generic through a `typecast` step that lives in no ported builder. A predicate that gets this
backwards, or that simplifies it to "tile-aligned only", will fail the routing test rather than
silently passing -- that is what the emitted test is for.

## Operating notes

- **`gh aw compile` rewrites `.gitattributes`**, silently dropping `merge=ours`. Run
  `git checkout -- .gitattributes` after every compile. This has bitten twice.
- **The lock file is what runs.** Editing `port-op.md` without `gh aw compile port-op` changes
  nothing.
- **`push:` triggers carry no inputs**, so every input falls back to a default duplicated inline
  throughout `port-op.md` (~18 occurrences of `|| 'untilize'`). Changing the op on the dryrun branch
  means changing all of them. This goes away when the workflow reaches the default branch and
  `workflow_dispatch` becomes usable; the `push:` trigger should be deleted then.
- **ccache is no longer this pipeline's problem.** Builds go through `build-artifact.yaml` on the
  CPU pool, which already has a warm Garage cache. The old in-job probe and its cold-build warning
  are gone with the container.
- **Hugepages are mounted conditionally**, because the viommu runners do not all expose
  `/dev/hugepages-1G`. `port-measure.yaml` inherits `test-dispatch.yaml`'s form of the conditional,
  which keys on the runner label rather than probing the path.
- **Run `actionlint` on `port-measure.yaml` after any change.** It is not covered by the gh-aw
  compile the way `port-op.md` is, and several of its failure modes -- a bad `needs` reference, an
  expression that does not resolve -- only surface as a dead run otherwise.
- **Nothing else in the repository watches `port-op-scratch/**`.** Checked, not assumed: every
  other `push` trigger in `.github/workflows` is either pinned to a branch list that excludes these
  refs or filtered to paths a scratch commit does not touch. Worth re-checking if a dispatch ever
  seems to cost more CI than it should.
- **When a gh-aw step fails on the viommu pool, suspect a missing binary before the logic.** That pool
  ships a thinner image than gh-aw assumes. `.github/scripts/port/shims/` once held stand-ins for
  `curl` and `netstat` for exactly this reason; it was deleted once nothing referenced it, and is
  recoverable from this branch's history if the pipeline ever collapses back into one in-cluster job.
  The `netstat` case is worth remembering as a near miss: the missing binary was the visible symptom,
  but the actual fault was `no_proxy` omitting the loopback addresses, so the shim would have papered
  over it.

## If the run fails

The generated code survives a failed or PR-less run: the `Preserve the port whatever the verdict`
and `Upload the port` steps capture the workspace diff and the port's own files as an artifact
regardless of the verdict. Recovering the agent's work does not depend on it having opened a PR --
that was added precisely because a run once produced good code and threw it away.

Failures are now spread over two runs, and the agent's job log will only ever show the launcher's
side of it. Every dispatch prints the `port-measure.yaml` run URL before it starts waiting; that is
the run with the real error in it, and the launcher's own message is a summary of it at best.

Triage order, cheapest first:

1. Read the `port-measure.yaml` run, not the agent's. The agent job is a client.
2. Did `ledger.py` report sane counts? Zero in-scope cases means the suite names or the declared
   `scope` in `axes/<op>.yaml` disagree with the sweep module.
3. Did `Emit the routing test` succeed? A `TypeError: cannot render ...` there means a kwarg holds a
   `ttnn` object that is not reachable as a module-level constant, and `_ttnn_constant_name` needs to
   learn it.
4. Download the port artifact before re-running anything. It carries `tracked.diff` -- the fastest
   way to see what the scaffold did to the tree, and how the CMake glob misplacement was diagnosed
   from a run that never reached the agent -- plus a copy of every dispatch's results.
5. Check for leftover `port-op-scratch/*` branches. The launcher deletes its own and a post-step
   sweeps the rest, but a hard-killed job can still strand one.

## The manifest is gone, and it was never ours

Every per-op fact the harness needed used to come from `agentic_port/manifests/<op>.yaml` in
tt-dm-codegen. The question that ended that arrangement was simply where those files come from, and
the answer is in their own schema: "Written by the **classify** stage. Read by the **translate**
stage, the **review** stage, and the **correctness** and **performance** gates." The directory beside
them holds `phase1_intake` through `phase8_conclude`.

That is a second agentic porting pipeline, running now, in the generator repository. It shipped
`untilize` as #50178 and has `tilize` open as #51919. The manifest is the scratch pad its phase 2
writes for its phase 4, and we were reading over its shoulder — a file nobody maintains on our
behalf, on a non-default branch, with nothing to tell us when it changes.

### The drift was already there

`tilize.yaml` lists eight `kernel_paths`. `ops/tilize/spec.py` selects kernels conditionally:

    "compute_tilize_typecast.cpp" if numeric_fuse else "compute_tilize.cpp"
    "writer_blockfloat_rne.cpp" if canonical_blockfloat else "writer_tilize_interleaved.cpp"

Neither `compute_tilize_typecast.cpp` nor `writer_blockfloat_rne.cpp` is in the list. Both exist. So
a `tilize` port taking the typecast or blockfloat path would reference a kernel that was never
vendored — which is not a hypothetical, because it already happened to `untilize`: `rm_shard_split.h`
was missing for exactly this reason and 112 in-scope cases wrote uncorrelated data against a green
build. We fixed that by copying the header in by hand, which fixed the case and not the cause.

The clincher is what the builder does about kernels: `ensure_symlink(_OP_TEMPLATES, ctx.kernel_dir,
shared_template_dir=_SHARED_TEMPLATES)`. It exposes two whole directories and never consults a list.
The generator's own answer to "which kernels does this op need" was never a list of eight paths.

### What replaced it

`discover.py`, which resolves the same facts by looking rather than reading, and fails at second zero
when a resolution comes back empty:

- **Category** from the operations tree. `untilize` exists twice, and the manifest's `native_entry`
  was the tiebreaker — but that field only ever encoded mainline versus experimental, which the
  directory layout already says. Real ambiguity now asks for `--category` instead of being guessed.
- **Builder** at `ops/<op>/spec.py`, with a search behind it. Six of seven manifests match the
  convention; `move` is built by `ops/identity/spec.py`, so convention alone would have been wrong.
- **Kernels** from the op's template directory, plus filenames quoted in builder source, plus the
  transitive `#include` closure. The closure is the part a list structurally cannot express, and it
  finds `rm_shard_split.h` for the same reason a compiler would.
- **Forced entries** built from the op and category. Conventional since the `implementation=` kwarg
  was removed, and checkable — `measure.py` resolves the path and refuses when it is unbound, which a
  declaration naming a nonexistent symbol cannot do. That failure mode matters: a `force_codegen` that
  silently falls back measures native against native and reads as perfect agreement.
- **Sweep module** at `common.sweeps.codegen_<op>`, which is 7 of 7.

Six fields turned out to be read nowhere outside the drift classifier — `cache_key_fields`, `tier`,
`layout_split`, `model_traced_module`, `upstream_sweep_module`, `ported_scope_sig` — because they
serve the other pipeline's translate stage, not ours. `cases` is labelled "documentation only" in its
own schema. And `coverage` powered a single warning.

### What is still written down, and where

`axes/<op>.yaml`, in this repository, beside the code that reads it. Six lines an op:

    shape: input_shape
    dtype: input_a_dtype
    layout: input_a_layout
    kwargs: {memory_config: output_memory_config}
    suites: [nightly, codegen_dtype, broaden_suite]

This one cannot be derived. A sweep names its parameters however it likes — `dtype` in
`codegen_move`, `input_a_dtype` in `codegen_tilize` — and sometimes buries them in a bundled
parameter, so `ri_specs.shape` has to descend into a value. Nothing about the tree recovers that.

Three optional blocks carry the narrowing that four of the seven ops do not need at all: `scope`
(only `untilize`), `ungradeable_reasons` (`pad`, `permute`) and `bad_golden` (`pad`). `tilize`
declares none, so the op in flight migrated with nothing to carry.

`scope` is worth a note, because it is closer to redundant than it looks. `gate.py` already grades
against the prototype leg: a case the generator itself fails is excused as a prototype gap, not
counted as failing. So the cases `scope` describes are already excused whether or not anything
declares them. What it still buys is the routing assertion and a human-readable reason for the
emitted test — neither of which a measurement produces on its own. Deriving scope entirely from the
prototype is a real option and a separate piece of work; it would move scope from ledger time to
grade time, which is a change to what the ledger *is*.

### The classifier had inherited the same bug

`drift.py` compared two manifests, so it could only report drift somebody had remembered to write
down — the exact staleness it exists to catch. `tilize.yaml` proves it: a commit changing either
unlisted kernel would have classified as nothing to do.

It compares generator trees now, through the same `discover.py` walk that `scaffold.py` vendors from,
so a drift report and a vendoring pass cannot disagree about what the port consists of. Kernels are
keyed by basename, because a basename is the identity a vendored copy has — which lets the
`f895be71b` shared-template move be reported as a move rather than as one deletion and one addition.
`blocking` changed meaning too: it is now "the harness cannot describe the op at the target at all",
which is a real stop, rather than "a manifest key nobody recognised".

`GitSource` grew a `tree()` that extracts a commit with `git archive` into a temp directory, because
the kernel walk needs to list directories and not just read named paths.

### What is not yet done

The three-way re-vendor. On a repin we can tell an agent's fix from a generator change by comparing
the vendored copy against the template as it stood at the *old* pin, and overwrite only where the copy
was never touched. Right now `scaffold.py --resume` copies what is absent, keeps what differs, and
reports both, which is the safe two-way behaviour: silence would hide generator changes and
overwriting would discard verified fixes. Getting the third way needs the workflow to keep the old
checkout instead of deleting it during the pin swap.

## Why the gates are strict, and what the accepted risk actually is

Condensed from the working notes this harness grew out of, which were deleted once they were
superseded. The original threat model and decision record are recoverable at
`agentic-port-gh-aw/NOTES.md` §10–11 in this branch's history; what follows is the part that still
governs the code.

### The exposure is inherent, and worth stating plainly

You cannot port C++ ops without compiling and executing agent-authored C++. The agent writes a
program factory, it links into `ttnn`, and a test loads `ttnn` and runs it — at which point
agent-authored code is a host process with device access. Two things make that broader than it looks:
`sources.cmake` lives inside each op directory, so any write allowlist scoped to the op necessarily
includes it, and CMake has `execute_process()`. Build time is in scope, not only test time.

So the security argument was never "the agent is sandboxed." It is that an on-card CI job is already
an untrusted-code execution context — every card job in tt-metal compiles and runs arbitrary
repository code, because that is what hardware testing is — and this workflow introduces no new
capability class. What it introduces is a new *principal* deciding what that code does. The danger is
therefore set by the trigger surface, which is why the trigger is `workflow_dispatch` and nothing
else.

### The decision, and what it changed

The accepted posture (Evan, 2026-08-07) is that the agent's output is trusted enough: anyone able to
run these workflows could already do equivalent damage deliberately, and the ops are code-generated to
begin with. That was explicitly not a gate on building the thing.

What it changed is *why* the surviving rules exist. Most of them stopped being security controls and
became **result-integrity** controls, and that reframing is the reason to keep them strict:

- **The write-path allowlist and the generated routing test matter more under this posture, not
  less.** The threat is no longer an attacker, it is reward hacking. An agent that can edit a test or
  add one of its own is an agent that can make its own numbers come out right, and the entire product
  of this pipeline is a performance and correctness *claim*. An unfalsifiable claim is worse than no
  pipeline. This is the one place worth being unreasonable, and it is why `gate.py` recomputes the
  diff against the base SHA on every call and re-renders the routing test rather than trusting it.
- **Budgets and timeouts stay**, as cost and availability controls. Card time is contended and a
  confused agent can burn a runner for an hour. The attempt cap, the no-progress stop and the
  two-verifies-per-tree cap are all this rule.
- **`noop` as a declared safe-output is a functional requirement, not a control.** Without it the
  agent feels obliged to make some safe-output call and files junk issues — observed twice in probes,
  and again this month when a diff-free run retried `create_pull_request`.

Two rules the record proposed dropping were reinstated by events. Staging the enforcement outside the
workspace was called redundant; it is now how the harness works, because the scripts live in
`$HOME/.port-harness` precisely so a port branch cannot edit the thing grading it. And
`--network none` was dropped deliberately, since ttnn tests fetch weights and pip dependencies, so
build and test run in a container for reproducibility rather than for isolation.

### The threat the notes anticipated and update mode made real

T9 in the original table was prompt injection from untrusted trigger content, mitigated at the time by
allowing human triggers only. Update mode reintroduces the substance of it by a side door: it reads
review comments off a pull request, which is text anyone able to comment can write, and puts it in
front of the agent. The trigger is still human, so a person chooses to start the run — but the
*content* is not. That is why `dispatch.py` frames collected review text in `port-brief.md` as
quotation rather than as instruction, and it is the reason to be careful about ever widening what
update mode ingests.

## Two ways a port branch quietly diverges from the harness that is supposed to grade it

Both found while setting up the first resume run on `tilize`, and both bite a human doing something
entirely reasonable to a port branch.

### A resumed port runs the harness its own branch carries, not the one you dispatched

`Place the launcher outside the sandbox` copies `.github/scripts/port/*.py` out of the workspace, and
it runs *after* `Checkout tt-metal`, which checks out `PORT_BRANCH`. So on a resume the launcher comes
from the port branch. For a fresh port the two are the same thing, because `PORT_BRANCH` is empty and
the checkout takes the triggering ref, which is why this went unnoticed.

The consequence is that a port branch freezes the harness as of its branch point, and no amount of
dispatching from a newer ref changes that. `ebanerjee/port-tilize` was cut before the manifest removal,
so it had no `discover.py`, while the workflow definition arriving from the dispatch ref had already
been rewritten to call `discover.py --category-only`. That combination fails in the resolve step,
before an agent exists, and the error says nothing about branches.

Merging the harness into the port branch is the fix that keeps the reproducibility property: a port
and the harness that graded it travel together, which is worth something when a verdict has to be
explained months later. The alternative -- staging the launcher from the dispatch ref -- would make
harness fixes reach every port branch at once and would also mean a port could be re-graded by a
harness it has never seen. That is a real choice, not an oversight, but it should be made deliberately
rather than by whichever step order happened to be written first.

### The generator pin lives on the tip commit and nowhere else

`Pin the generator to the commit this port was written against` reads the trailer with `git log -1`
against a `fetch-depth: 1` checkout. One commit, no history. So **any** commit pushed onto a port
branch that is not written by `dispatch.py publish` drops the pin, and the next run silently falls
back to the floating generator branch.

That is not a small silent failure. When this was hit, the floating branch was 68 commits ahead of the
pin, and vendoring kernels from it against host code written for `dcc8e35bf` is the untilize
`rm_shard_split.h` failure exactly: 112 cases writing uncorrelated data, presenting as a kernel bug
rather than as a pin problem. A merge commit is the likeliest way in, because merging the harness onto
a port branch is now a thing we do on purpose.

Until the read walks history -- which needs a deeper fetch, so it is not a one-line change -- any
hand-made commit on a port branch has to carry the trailer forward explicitly. Checking that
`git log -1 --format='%(trailers:key=Port-generator,valueonly=true,unfold=true)'` still returns forty
hex characters before pushing is the whole test.
