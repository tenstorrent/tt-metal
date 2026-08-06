# `.github/workflows/` Overview

This folder has 200+ workflow files. This doc is a map, not a per-file reference — it groups
workflows by what they're for and points you at the right one. For how to add/change a *test* or
build a *new pipeline*, see [`tests/pipeline_reorg/README.md`](../../tests/pipeline_reorg/README.md)
first; this doc is about the workflow layer itself. Written for humans and AI agents.

## Naming conventions

| Pattern | Meaning |
|---------|---------|
| `<name>.yaml` | Entry point — triggers on `schedule`/`workflow_dispatch` (and sometimes `pull_request`/`merge_group` for gates). This is what shows up in the Actions UI "Run workflow" list. |
| `<name>-impl.yaml` | Reusable `workflow_call` workflow that does the actual work for one `<name>.yaml`. Not meant to be run directly. |
| `_<name>.yaml` | Reusable-only helper, shared across multiple pipelines (not tied to one `<name>.yaml`). E.g. `_produce-data.yaml`, `_auto-retry-post-commit.yaml`. |
| `<name>-wrapper.yaml` | Thin wrapper adapting a shared reusable workflow for one specific caller (e.g. `build-wrapper.yaml`, `docs-latest-public-wrapper.yaml`). |
| `pipeline-select-*.yaml` | Manual-dispatch convenience UI that fans out to several already-reorged pipelines from one form (e.g. `pipeline-select-galaxy.yaml`). Not a pipeline itself. |
| `zzz <name>` (workflow `name:` field, not filename) | Reusable-only workflow (e.g. `basic.yaml`, `smoke.yaml`) name-prefixed so it sorts to the bottom of the Actions UI, away from real dispatchable pipelines. |

A parent/impl pair can also be **shared by multiple entry points** rather than 1:1 — e.g.
`sanity-tests.yaml` (workflow_call only) is called by both `sanity-tests-pr.yaml` (fork-PR trigger)
and `sanity-tests-debug.yaml` (nightly, with watcher/LLK-assert options). Don't assume every
`-impl.yaml`-shaped file has exactly one caller.

## Gates: PR gate & merge gate

The two mandatory checkpoints every change passes through — grouped together here because they're
designed as a pair and share most of their reusable building blocks:

- **[`pr-gate.yaml`](pr-gate.yaml)** — runs on every PR push. Must stay fast (target: end-to-end
  under 5 minutes, per the comment at the top of the file). Calls `smoke.yaml` (reusable, name
  `zzz Smoke tests`) plus a handful of team-specific smoke jobs (`runtime-smoke-tests`,
  `ttnn-smoke-tests`, `llk-smoke-tests`, …).
- **[`merge-gate.yaml`](merge-gate.yaml)** — runs when a PR enters the merge queue (`merge_group`).
  Slightly heavier than PR gate; calls `basic.yaml` (reusable, name `zzz Basic tests`) plus
  team-specific basic/merge-gate jobs (`ttnn-merge-gate-tests`, `train-merge-gate-tests`,
  `llk-unit-tests`, `scaleout-unit-tests`, …).
- Both terminate in a `workflow-status` job — the single required check branch protection actually
  looks at, aggregating every other job's result. If you add a job to either gate, wire it into
  `workflow-status`'s inputs or it won't actually block anything.
- Both are wired into the time-budget system (see `tests/pipeline_reorg/README.md`), but coverage
  is incomplete — see that doc's "common pitfalls" for the current gap.

Adding a test to an *existing* gate suite is the "add or change a test" path in
`tests/pipeline_reorg/README.md`. Adding a whole new job to `pr-gate.yaml`/`merge-gate.yaml`
itself (not just a test-yaml entry) needs metal-infra review — these two files are the most
blast-radius-sensitive workflows in the repo.

## Shared pipelines used across teams

Not gates, not team-owned — infrastructure-owned pipelines that every team relies on:

- **Package and release** — [`package-and-release.yaml`](package-and-release.yaml) orchestrates
  versioning, tagging, and publishing; calls [`release-build-test-publish.yaml`](release-build-test-publish.yaml),
  [`release-verify-or-create-tag.yaml`](release-verify-or-create-tag.yaml), and
  [`publish-release-image.yaml`](publish-release-image.yaml)/`-wrapper.yaml`.
  [`release-cleanup.yaml`](release-cleanup.yaml) handles teardown. Release-specific test selection
  lives in `tests/pipeline_reorg/release_tests.yaml`, not inlined here.
- **L2 nightly** — [`tt-metal-l2-nightly.yaml`](tt-metal-l2-nightly.yaml) is the broad nightly
  catch-all across wormhole/blackhole, cpp unit tests, LLK unit tests, and DIDT tests, toggled via
  `workflow_dispatch` booleans. Still mid-migration to per-team reorged pipelines — see MINFRA-408
  for status before assuming everything here is reorg-format.
- **Sanity** — [`sanity-tests.yaml`](sanity-tests.yaml) (the renamed APC) is the cross-team
  post-commit "if this is broken, Metal is broken" suite: toggles TTNN sanity, T3000 APC-fast, ops
  unit, and more via `workflow_dispatch` inputs. It's `workflow_call`-only; the actual entry points
  are [`sanity-tests-pr.yaml`](sanity-tests-pr.yaml) (fork PRs) and
  [`sanity-tests-debug.yaml`](sanity-tests-debug.yaml) (nightly cron, with debug/watcher options).

## Team-specific pipelines

Everything else is one team's pipeline suite, generally following the reorg format
(`<name>.yaml` + `<name>-impl.yaml` + `tests/pipeline_reorg/<name>_tests.yaml`). Filename prefix
tells you the team:

| Prefix / pattern | Team | Example entry points |
|---|---|---|
| `models-*`, `single-card-demo-*`, `demo-sp-*`, `blaze-models-*`, `perf-models*`, `perf-device-models*` | Models | `models-t1-e2e-tests.yaml`, `single-card-demo-tests.yaml` |
| `ttnn-*`, `ops-*` | TTNN / Ops | `ttnn-run-sweeps.yaml`, `ttnn-model-trace-sweep-validation.yaml` |
| `llk-*` | LLK | `llk-e2e.yaml`, `llk-perf.yaml`, `llk-ttsim-weekly.yaml` |
| `runtime-*` | Runtime | `runtime-unit-tests.yaml`, `runtime-perf-tests.yaml` |
| `galaxy-*` | Scaleout (WH Galaxy) | `galaxy-e2e-tests.yaml`, `galaxy-demo-tests.yaml` |
| `blackhole-*` | Blackhole | `blackhole-e2e-tests.yaml`, `blackhole-sanity-tests.yaml` |
| `t3000-*` / `t3k-*` | T3000 | `t3000-e2e-tests.yaml`, `t3000-fast-tests.yaml` |
| `tt-train-*`, `train-*` | tt-train | `tt-train-tests.yaml` (one impl, multiple categories — see Naming conventions) |
| `tm-*`, `fabric-*` | Fabric / Scaleout | `tm-fabric-tests.yaml`, `fabric-build-and-unit-tests.yaml` |
| `triage-*` | Triage | `triage-tests.yaml` |
| `syseng-*`, `*didt*` | Syseng | `syseng-didt-tests-impl.yaml` |
| `umd-*` | UMD | `umd-sanity-tests-impl.yaml` |
| `vllm-*` | vLLM / Models | `vllm-nightly-tests.yaml` |
| `metal-run-microbenchmarks*` | Runtime | `metal-run-microbenchmarks.yaml` |

**Not everything here is reorg-format yet.** `conda-post-commit.yaml`, `test-dispatch.yaml`,
`upstream-tests.yaml`, `fabric-multihost-exabox.yaml`, and `galaxy-multi-user-isolation-tests.yaml`
are examples of pipelines still pending migration (see the candidate list in
[MINFRA-415](https://tenstorrent.atlassian.net/browse/MINFRA-415)) — don't copy their shape for a
new pipeline. And per `tests/pipeline_reorg/README.md`, **you generally shouldn't need to create a
new team pipeline** — file a request against [MINFRA](https://tenstorrent.atlassian.net/browse/MINFRA)
first.

## Infra pipelines

Not test pipelines at all — repo/CI operations tooling, owned by metal-infra:

- **Build plumbing**: [`build-artifact.yaml`](build-artifact.yaml), `check-harbor.yaml`,
  `build-docker-*.yaml`, `resolve-docker-pull-refs.yaml` — shared by nearly every pipeline above to
  produce/resolve the docker image and build artifact a test job runs against.
- **CI health / triage**: `aggregate-workflow-data.yaml` (aggregates run data into the CI health
  report, every 10 min), `triage-ci.yaml` (opens GitHub issues for persistent failures — see
  "Auto-Triage Info" on Confluence for the full lifecycle), `runner-failure-report.yaml` /
  `runner-failure-scan.yaml`, `grouping-ci-failures.yaml`, `ci-digest.yaml`.
- **Data pipeline**: `_produce-data.yaml` uploads workflow results to Superset/Snowflake — see
  "Metal Infra Data Pipeline" on Confluence.
- **PR/repo automation**: `auto-approve.yaml` (self-bump dependency PRs), `pr-description-inject-branch-name.yaml`,
  `mirror-fork-branch.yaml`, `remove-stale-branches.yaml`, `set-opened-on.yaml`,
  `on-community-issue.yaml`, `notify-slack-on-mention.yaml`.
- **Dispatch conveniences**: `pipeline-select.yaml` / `pipeline-select-galaxy.yaml` /
  `pipeline-select-t3k.yaml` / `pipeline-select-profiler.yaml` — one-click multi-pipeline dispatch
  forms, not pipelines themselves (see Naming conventions).
- **Everything else** (docs publishing, code coverage, static analysis, AI/Copilot tooling,
  wheels/packaging tests) generally isn't part of the test-pipeline system this doc is about — if
  you're not sure whether a workflow is in scope for the reorg format, ask in `#tt-metal-pipelines`
  before treating it as a template.

## Reference

- Test/pipeline authoring: [`tests/pipeline_reorg/README.md`](../../tests/pipeline_reorg/README.md)
- PR review rules (human + AI): [`.github/instructions/ci-cd.instructions.md`](../instructions/ci-cd.instructions.md)
- Migration tracking: [MINFRA-408](https://tenstorrent.atlassian.net/browse/MINFRA-408) epic
- Help: Slack `#tt-metal-pipelines`, `@metalinfra`
