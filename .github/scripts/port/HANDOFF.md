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

`untilize` is the second op and the first that is not `pad`. It is running now as the first exercise
of the hardened harness.

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

Two of those five were found only by a real run, and one of them (the CMake glob) had a green
scaffold step in front of it. When adding a check, prefer asserting *where* something landed over
asserting that it exists.

`selftest.py` covers all of the above that can be checked without a device -- against a stubbed `ttnn`
and a stubbed sweep module -- and runs on a laptop in under a second. Run it after touching
`ledger.py`, `scaffold.py`, or `strata.py`. It is the only test the harness has.

## What is *not* verified about this run

None of the following could be checked without a device, so treat them as the likely failure points:

- **Scale.** `pad`'s ledger is 196 cases. `untilize`'s is roughly twice that, because its `nightly`
  grid uses a `gen_shapes` interval of 1 and the suite union adds more. The correctness band is
  uncapped by deliberate design -- the routing claim only means something over the whole out-of-scope
  set -- so this is the first time that choice has been stressed. If anything times out, it is this.
- **No golden.** `ttnn.untilize` has no golden function registered, so `resolve_golden` falls back to
  native and the "Check the golden against native" step emits a warning and passes trivially.
  Correctness for this port therefore means *matches native output*, not *matches torch*. A port that
  faithfully reproduces a native bug will pass. `pad` had a real torch golden, so this weaker oracle
  is new, and the PR body should not be read as claiming more than it measures.
- **Case counts, timings, and stratum labels** are all as-yet unobserved for this op.

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
