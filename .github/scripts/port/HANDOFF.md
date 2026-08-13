<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Porting harness: handoff

Everything about *why* a given rule exists is written where the rule is -- in `port-op.md`'s comments
and in each script's docstrings. This file is for the things that live in no single file: the state of
the experiment, the sharp edges of operating it, and what is not yet known.

## State

> **Blocked as of 2026-08-13: the generator token needs SAML SSO re-authorization.** Every mode --
> baseline, build and verify -- checks out `tenstorrent/tt-dm-codegen`, and that checkout now fails:
>
>     remote: The 'tenstorrent' organization has enabled or enforced SAML SSO.
>     fatal: unable to access 'https://github.com/tenstorrent/tt-dm-codegen/': HTTP 403
>
> Nothing in the harness can route around it and no run of any kind will get past it. `CODEGEN_REPO_TOKEN`
> has to be authorized for the `tenstorrent` organization at the SSO URL GitHub prints in that error, or
> reissued and authorized. It is not a scope problem and not a rotation the pipeline can detect in
> advance. First seen in run
> [31736836951](https://github.com/tenstorrent/tt-metal/actions/runs/31736836951), which was verifying
> two hand-fixed translation defects in `untilize` and would otherwise have been the first graded
> verdict on real translation work.

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
under test regardless, so that push doubles as the trigger. The reason it is not `workflow_dispatch`
is that `workflow_dispatch` only fires for workflow files that are **already on the default branch**
-- so the dispatch design could not be exercised at all without first merging it, which is a poor
way to prototype a pipeline. A `push` event has no such rule: it runs the workflow exactly as it
exists in the commit that was pushed. Nothing about this pipeline needs to reach `main` to work,
including `port-measure.yaml` itself.

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
- **`.github/scripts/port/shims/` is now unused.** It held stand-ins for `curl` and `netstat`, which
  the viommu image ships and gh-aw assumes; that mattered only while the agent ran in-cluster. Kept
  rather than deleted, because it is exactly what would be needed again if the proxy allowlist ever
  changes and the pipeline collapses back into one in-cluster job. The general lesson from it still
  applies to that pool: when a gh-aw step fails there, suspect a missing binary before the logic.

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
2. Did `ledger.py` report sane counts? Zero in-scope cases means the manifest, the suite names, or
   `port_scope` disagree with the sweep module.
3. Did `Emit the routing test` succeed? A `TypeError: cannot render ...` there means a kwarg holds a
   `ttnn` object that is not reachable as a module-level constant, and `_ttnn_constant_name` needs to
   learn it.
4. Download the port artifact before re-running anything. It carries `tracked.diff` -- the fastest
   way to see what the scaffold did to the tree, and how the CMake glob misplacement was diagnosed
   from a run that never reached the agent -- plus a copy of every dispatch's results.
5. Check for leftover `port-op-scratch/*` branches. The launcher deletes its own and a post-step
   sweeps the rest, but a hard-killed job can still strand one.
