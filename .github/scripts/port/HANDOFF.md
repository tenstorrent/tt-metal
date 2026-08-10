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

Three separate defects were found by reading the `untilize` manifest before pushing, all of them
things `pad` happens not to do. They are fixed, but the pattern is the point: **the harness was
implicitly specialised to one op's manifest shape**, and each new op is likely to find more of this.
Expect the third op to find its own.

| What `untilize` does that `pad` does not | What broke |
| --- | --- |
| `sweep_suite` names two suites, not one | `module.parameters[suite]` with a list: `TypeError: unhashable type: 'list'`, before any step ran |
| Declares `port_scope` | Nothing read it. A large slice of cases the manifest puts *outside* the ported builder would have been graded as in-scope |
| A kwarg holds a live `ttnn` object (`memory_config`) | The ledger's `json.dumps(..., default=str)` flattened it to `"MemoryConfig(...)"`, and every `ttnn.untilize(x, memory_config=...)` built from it would raise |

`selftest.py` now covers all three against a stubbed `ttnn` and a stubbed sweep module. It runs on a
laptop in under a second and needs no device; run it after touching `ledger.py`, `scaffold.py`, or
`strata.py`. It is the only test the harness has.

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
4. Download the port artifact before re-running anything.
