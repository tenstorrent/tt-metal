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
harness, and it also moved the job to CIv2, which the `pad` PR never ran on. Both moves at once is
why the recent history is a run of failures: four of them, three ours and one not, each hiding the
next. The harness fixes are described below; the one that is not ours is the last section before
*What is not verified*, and it is currently what blocks a `untilize` PR.

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
`ledger.py`, `scaffold.py`, or `strata.py`. It is the only test the harness has.

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

## The one thing CIv2 cannot do yet, and it is not ours to fix

The job now runs on CIv2 end to end up to the agent itself: the runner hosts gh-aw's servers, the
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

## What is *not* verified about this run

None of the following could be checked without a device, so treat them as the likely failure points:

- **Scale.** 208 cases against `pad`'s 196, so comparable, but the correctness band calls native for
  every case here because there is no golden to compare against instead. It is uncapped by deliberate
  design -- the routing claim only means something over the whole out-of-scope set -- so if anything
  times out, it is this.
- **No golden.** `ttnn.untilize` has no golden function registered, so correctness means *matches
  native output*, not *matches torch*. A port that faithfully reproduces a native bug will pass.
  `pad` had a real torch golden, so this weaker oracle is new, and the PR body should not be read as
  claiming more than it measures.
- **Everything from the agent onward.** The agent process now starts, but it cannot reach a model
  (see above), so it has never taken a single action. The deliverable contract, the demotion logic
  and the per-stratum grading have still only been exercised against synthetic fixtures.

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
- **ccache is still cold.** Garage is unreachable from the runner, so the probe warns and the build
  proceeds from scratch. Moving to CIv2 was about device access, not build speed; whether the
  in-cluster egress path makes Garage reachable is untested. Budget for a full build.
- **Hugepages are mounted conditionally**, because the viommu runners do not all expose
  `/dev/hugepages-1G`.
- **The viommu image ships neither `curl` nor `netstat`, and gh-aw assumes both.** It decides whether
  its MCP servers came up with `curl -s -f http://localhost:$PORT/health >/dev/null 2>&1`, ten times,
  then prints `netstat` as the diagnostic. With curl absent, every attempt failed silently while the
  server sat listening, and the only visible error was the diagnostic's own
  `netstat: command not found` -- which is misleading, because netstat was never what failed.
  `.github/scripts/port/shims/` holds stand-ins over `ss` and `python3`, linked onto `PATH` only when
  the real tool is missing. Shims rather than `apt-get install`, because egress goes through the
  restricted proxy and neither answer needs the network.

  Two general lessons for this runner pool. A silent probe in someone else's step is the worst thing
  to debug, so the shim step verifies itself against a server it starts. And the image is leaner than
  the CIv1 pool in ways that surface late, after the build is paid for -- when a gh-aw step fails
  here, suspect a missing binary before suspecting the logic.

## If the run fails

The generated code survives a failed or PR-less run: the `Preserve the port whatever the verdict`
and `Upload the port` steps capture the workspace diff and the port's own files as an artifact
regardless of the verdict. Recovering the agent's work does not depend on it having opened a PR --
that was added precisely because a run once produced good code and threw it away.

Triage order, cheapest first:

1. Did the job land on a `tt-ubuntu-2204-N150-viommu-stable` runner at all?
2. Did `ledger.py` report sane counts? Zero in-scope cases means the manifest, the suite names, or
   `port_scope` disagree with the sweep module.
3. Did `Emit the routing test` succeed? A `TypeError: cannot render ...` there means a kwarg holds a
   `ttnn` object that is not reachable as a module-level constant, and `_ttnn_constant_name` needs to
   learn it.
4. Download the port artifact before re-running anything. `tracked.diff` in it is the fastest way to
   see what the scaffold did to the tree -- it is how the CMake glob misplacement was diagnosed, from
   a run that never reached the agent.
