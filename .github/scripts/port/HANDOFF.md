<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Porting harness: handoff

Everything about *why* a given rule exists is written where the rule is -- in `port-op.md`'s comments
and in each script's docstrings. This file is for the things that live in no single file: the state of
the experiment, the sharp edges of operating it, and what is not yet known.

## State

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
  report that no run appeared. This, not scope, is why `PORT_PUSH_TOKEN` exists.
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

### Before the next run, one thing must exist

**A `PORT_PUSH_TOKEN` repository secret**: a fine-grained PAT on `tenstorrent/tt-metal` with
`contents: write` (push the scratch ref) and `actions: read` (read back the run, its artifacts and
its logs). The pre-step fails loudly if it is missing. As of writing, `gh secret list` shows only
`CODEGEN_REPO_TOKEN`, so this has not been done.

It has to be a PAT for the trigger reason above. Note that this rules out the otherwise attractive
fallback: gh-aw's strict mode rejects *any* write permission on the agent job, and turning it off
with `strict: false` would let `github.token` push -- a token that expires with the run and cannot
touch `.github/workflows/**`, both real advantages over a PAT -- but its pushes would not start the
measurement, so it is not an option here whatever the permissions say.

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

## The one thing CIv2 cannot do

Routed around rather than fixed, by the split described at the top. The diagnosis is kept because it
is what rules out running the agent in-cluster at all, and because the infra ask at the end of it is
still the shorter fix if it ever becomes available -- it would collapse the two workflows back into
one job with a local incremental build, which is both simpler and far faster per cycle.

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

**The ask for infra is one line:** allow `api.githubcopilot.com:443` through `restricted-proxy` for
the `tt-ubuntu-2204-N150-viommu-stable` pool. Worth requesting alongside it, since gh-aw's own
allowlist names them and a later feature may reach for one:
`api.business.githubcopilot.com`, `api.enterprise.githubcopilot.com`,
`api.individual.githubcopilot.com`, and `telemetry.enterprise.githubcopilot.com`.

Note that AWF is *not* being bypassed by this request. Its own allowlist still governs what the
agent may reach; the cluster proxy is a second gate in series, and this only stops the two gates
from disagreeing about the one host the agent cannot work without.

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
`port-op.lock.yml` should mention `PORT_PUSH_TOKEN` in exactly three places -- the pre-step, gh-aw's
own log-redaction step, and the cleanup post-step -- and nowhere near the agent step or the handler
files. That grep is the check.

## What is *not* verified

**No dispatch has ever left the ground.** The token above was outstanding when this was written, so
nothing that needs GitHub or a card has run even once.

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

The untested surface, roughly in the order a first run will meet it:

- **The rest of the round trip.** Find-the-run-by-SHA, poll, artifact download, scratch-ref cleanup.
- **The `/codegen` volume in the device job.** It is mounted from a checkout path the way
  `test-dispatch.yaml` mounts `docker-job`, but Docker creates the host directory before the
  checkout step runs, so an image that does not run as root could fail to write into it.
- **Whether the harness dependencies install.** `uv pip install graphviz pyyaml` in a job that is
  not the long-lived `portdev` container. The in-container version needed `--python` to find the
  right interpreter; this one follows `test-dispatch.yaml`'s plain form instead.

And, unchanged from before the split, the things a device was always going to have to settle:

- **Scale.** 208 cases against `pad`'s 196, so comparable, but the correctness band calls native for
  every case here because there is no golden to compare against instead. It is uncapped by deliberate
  design -- the routing claim only means something over the whole out-of-scope set -- so if anything
  times out, it is this.
- **No golden.** `ttnn.untilize` has no golden function registered, so correctness means *matches
  native output*, not *matches torch*. A port that faithfully reproduces a native bug will pass.
  `pad` had a real torch golden, so this weaker oracle is new, and the PR body should not be read as
  claiming more than it measures.
- **Everything from the agent onward.** The agent has never reached a model, so it has never taken a
  single action. The deliverable contract, the demotion logic and the per-stratum grading have still
  only been exercised against synthetic fixtures.

### How to run the dry run once it is unblocked

Cheapest first, so each failure costs the least it can. Step 1 is already done, above.

1. ~~A manual scratch push with `mode: build`.~~ Done: the trigger, `resolve` and the front of
   `build-artifact.yaml` are proven.
2. A manual scratch push with `"mode":"baseline"` from a scaffolded tree, letting the build run to
   completion this time. Exercises the device job, the `/codegen` mount, the wheel install and the
   results artifact -- the whole half that has never executed. Still no agent and no credits.
3. Push the branch to `ebanerjee/port-op-dryrun` to trigger `port-op` itself, which runs step 2 as
   its own pre-agent step and then hands over to the agent.

Record the per-cycle wall times from step 3 here afterwards. The design assumed 20 minutes for a
build and 35-45 for a verify, and the whole budget argument rests on those two numbers.

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
