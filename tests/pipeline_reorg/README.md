# Pipeline Reorg Format

This folder is the **test registry** half of tt-metal's "pipeline reorg" CI format. If you're
adding a test, moving a test between pipelines, creating a brand-new CI pipeline, or reviewing a
PR that touches `.github/workflows/` or `tests/pipeline_reorg/`, read this first.

It's written for humans and for AI coding agents — if you're an agent picking up a task like "add
a CI job for my new op" or "put this test in the reorg format," this doc plus the linked examples
should be everything you need without having to reverse-engineer other pipelines.

## Why this exists

Before the reorg, pipelines were ad-hoc: test commands were inlined directly in workflow YAML,
there was no consistent way to tell how much machine time a team was using, and it wasn't clear
which pipeline a new test belonged in. See the original design doc for the full problem statement:
[Proposed pipeline and test organization changes](https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/1396506680)
(Confluence, MI6 space). Migration status/tracking lives under the
[MINFRA-408](https://tenstorrent.atlassian.net/browse/MINFRA-408) epic. **Not every pipeline in
this repo has been migrated yet** — see `.github/instructions/ci-cd.instructions.md` for how to
spot a pre-reorg pipeline.

Questions, or something in this doc looks stale? Ask in Slack `#tt-metal-pipelines` or ping
`@metalinfra`.

## The format, in one picture

A fully-reorged pipeline is **three files + two registry entries**:

```
.github/workflows/<name>.yaml          # 1. thin parent workflow
.github/workflows/<name>-impl.yaml     # 2. reusable impl workflow
tests/pipeline_reorg/<name>_tests.yaml # 3. test registry (this folder)
.github/time_budget.yaml               # 4. time-budget entry: team -> pipeline -> sku -> minutes
.github/sku_config.yaml                # 5. SKU -> runner label mapping (only needed if you introduce a new SKU)
```

1. **Parent workflow** (`.github/workflows/<name>.yaml`) — thin. Handles `workflow_dispatch`
   inputs, triggers the build (`build-artifact.yaml`), and calls the impl workflow with
   `enabled-skus` computed from the inputs. See
   [`galaxy-e2e-tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/galaxy-e2e-tests.yaml)
   for a minimal example.
2. **Impl workflow** (`.github/workflows/<name>-impl.yaml`) — a reusable `workflow_call` workflow
   that does the real work:
   - `load-test-matrix` job: checks out `tests/pipeline_reorg/` (sparse), runs
     `verify_time_budget.py` (fails fast if the registry's timeouts exceed the team's budget), then
     `prepare_test_matrix.py` (expands the registry into a flat GHA matrix filtered to
     `enabled-skus`).
   - A second job that runs the actual matrix (`strategy.matrix.test-group`), one GHA job per
     `(test, sku)` pair, running `${{ matrix.test-group.cmd }}` on `${{ matrix.test-group.runs_on }}`
     with `timeout-minutes: ${{ matrix.test-group.timeout }}`.
   - See
     [`galaxy-e2e-tests-impl.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/galaxy-e2e-tests-impl.yaml)
     for the minimal version, or
     [`tt-train-tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/tt-train-tests.yaml)
     for a fancier one that adds `subcategories` filtering and a `timeout-minutes-override` escape
     hatch.
3. **Test yaml** (`tests/pipeline_reorg/<name>_tests.yaml`, this folder) — the dev-maintained
   registry of what actually runs. This is almost always the **only file you need to touch** to
   add, remove, or retime a test — you should not need to edit workflow YAML at all for routine
   test changes.

This split matters for ownership: workflow YAML (`.github/workflows/`) is metal-infra-maintained
plumbing; test yaml (`tests/pipeline_reorg/`) is dev-maintained content. Keep changes in your PR
scoped accordingly — if you're "just adding a test," you should have a one-file diff.

## Test yaml schema

Each file in this folder is a YAML list. Every entry (one entry = one logical test, which may
expand into multiple GHA jobs if it targets multiple SKUs) requires:

| Key | Type | Purpose |
|-----|------|---------|
| `name` | string | Display name for the GHA job. The SKU is auto-appended (`"{name} [{sku}]"`) so you don't need to include it yourself. |
| `cmd` | string | The exact shell command(s) to run. Multi-line block scalars (`\|`) are fine. Must not depend on env vars the impl workflow doesn't set — inline what you need. |
| `skus` | mapping | `{ <sku_name>: { timeout: <minutes>, ... } }`. A test can list multiple SKUs; each expands into its own matrix entry. `sku_name` must exist in `.github/sku_config.yaml`. |
| `skus.<sku>.timeout` | int | Per-job GHA timeout in minutes, **for this SKU**. This is also what `verify_time_budget.py` sums against the team's budget — treat it as the source of truth for machine time, not a guess. |
| `owner_id` | string | Slack member ID (starts with `U`) to notify on failure. Get yours: Slack profile photo → View profile → ⋯ → Copy member ID. Add a `# Name` comment next to it for human readability. |
| `team` | string | Owning team. Must match a top-level key in `.github/time_budget.yaml`. |

Commonly-used optional keys (the format explicitly allows arbitrary extra keys per pipeline — add
what your pipeline needs):

| Key | Purpose |
|-----|---------|
| `id` | Stable slug used for `workflow_dispatch` test-selection / `jq` filtering (e.g. `pipeline-select-*.yaml`). Add this if your parent workflow lets users run a single test by name. |
| `model` | Model identifier, for pipelines with per-model `workflow_dispatch` selection (see `models_*_tests.yaml`). |
| `skus.<sku>.tier` | Model tier (1/2/3) for tiered Models pipelines — lets the impl workflow filter by tier and look up budgets under `<workflow>_tier<N>` instead of the plain key. See `models_e2e_tests.yaml` + `models-e2e-tests-impl.yaml`. |
| `category` / `subcategories` | Coarser grouping than `id`, for pipelines that filter by category rather than by individual test (see `tt-train-tests.yaml`). |
| `arch` | Architecture tag, when a test's routing depends on it beyond what `sku` already encodes. |
| `{placeholder}` in `cmd` | Any other top-level key on the same entry can be referenced as `{key}` inside `cmd` and will be substituted (e.g. per-SKU cache paths). See `prepare_test_matrix.py::substitute_cmd_placeholders`. |

Minimal example:

```yaml
- name: my_model_tests
  cmd: pytest models/demos/my_model/tests/
  skus:
    wh_n150_civ2:
      timeout: 10
  owner_id: U01234ABCDE # Your Name
  team: models
```

Real examples worth reading (in rough order of complexity):
[`fabric_merge_gate_tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/tests/pipeline_reorg/fabric_merge_gate_tests.yaml) →
[`galaxy_e2e_tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/tests/pipeline_reorg/galaxy_e2e_tests.yaml) →
[`models_unit_tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/tests/pipeline_reorg/models_unit_tests.yaml) (multi-SKU + tier).

## Adding or changing a test (the common case)

To add, remove, retime, or reassign ownership of a test **in an already-reorged pipeline**, you
only need to edit the relevant `tests/pipeline_reorg/<name>_tests.yaml`:

1. Find the right file for your test's team + pipeline level (see "Which pipeline level?" below).
2. Add/edit an entry per the schema above.
3. Check whether the sum of your team's timeouts for that `(team, pipeline, sku)` triple still
   fits the budget in `.github/time_budget.yaml` (`<team>.<pipeline_key>.<sku>`). If not, bump the
   budget in the same PR — see "Time budgets" below. `verify_time_budget.py` runs on every
   pipeline invocation, so a mismatch fails CI, not just review.
4. No workflow YAML changes needed. If you find yourself editing `.github/workflows/*-impl.yaml`
   just to add a test, something's wrong — impl workflows should be generic over the registry.

## Adding a brand-new pipeline

If no existing pipeline fits (new team, new testing level, new architecture), create all three
artifacts:

1. **Test yaml**: `tests/pipeline_reorg/<name>_tests.yaml` with your test entries.
2. **Impl workflow**: `.github/workflows/<name>-impl.yaml`, `workflow_call`-triggered, following
   the `load-test-matrix` → `verify_time_budget.py` → `prepare_test_matrix.py` → matrix-execution
   shape above. Copy the simplest existing impl workflow that's close to your shape rather than
   writing from scratch.
3. **Parent workflow**: `.github/workflows/<name>.yaml`, thin, calling your impl workflow.
4. **Time budget**: add a `team -> pipeline -> sku: minutes` entry to `.github/time_budget.yaml`.
   The `pipeline` key here is the `workflow_name` argument you pass to `verify_time_budget.py` in
   your impl workflow (by convention this matches the testing level — `unit`, `e2e`, `sanity`,
   etc. — not the file name; see `query_time_budget.py`'s docstring for why the two can differ).
5. **SKU**: only if you need a machine configuration that doesn't already exist in
   `.github/sku_config.yaml` — check there first, most SKUs you need already exist.

Then follow the naming/trigger rules below, and get one parent + one impl workflow reviewed by
metal-infra (see `.github/instructions/ci-cd.instructions.md`, which the AI PR reviewer also reads).

### Naming and trigger rules

From the reorg design (`MINFRA-415`) and the CI/CD review guide:

- Exactly **one parent + one impl workflow** per product/team pipeline. Anything else beyond that
  pair must be shared infra workflows (reusable actions), not a second bespoke pipeline.
- Name pipelines by **intent/level** (`smoke`, `sanity`, `unit`, `integration`, `e2e`, `perf`,
  `stress`, `sweep`), not by frequency (`nightly`, `weekly`) or architecture (`galaxy`, `t3k`) where
  avoidable — frequency and architecture can change independently of what the pipeline tests.
  (Note: a lot of already-reorged pipelines still use frequency/arch names; that's a known,
  explicitly out-of-scope-for-now cleanup, not something to imitate in new pipelines.)
- Pipelines should trigger only on `schedule` and/or `workflow_dispatch`, **not** on `push` or
  `pull_request`, unless there's a specific justification (comment it in the workflow).
- A pipeline should not trigger on PR at all unless it's invoked *by* `pr-gate.yaml` or
  `merge-gate.yaml`, or there's an explicit, commented exception.
- Every pipeline should upload results to Superset (ask `#tt-metal-infra` / William if unsure how)
  and send Slack failure notifications (ask Evan) — check that both are wired up, don't assume
  they come for free.

## Pipeline levels (which one does my test belong in?)

| Level | Intent | Expected runtime | Frequency |
|-------|--------|-------------------|-----------|
| `smoke` | Bare minimum sanity on merge attempt | Seconds, < 1 min/test | Every merge-queue attempt |
| `sanity` | Happy-path coverage — if broken, Metal is broken | Short, < 5 min/test | Post-commit / per-commit |
| `unit` | Single component (one op, one model module) | Short | On schedule |
| `integration` | Multiple components together | Medium | On schedule |
| `e2e` | Full end-to-end system | Long | On schedule |
| `perf` / `performance` | Performance measurement, perf-mode enabled | Variable | On schedule |
| `stress` | Repeated execution, no perf assertion, long-running | Long | Infrequent |
| `sweep` | Parameter sweep/schmoo across a large space | Very long | Infrequent |

Models-team pipelines additionally split by tier (1/2/3) within `unit`/`e2e`/`device_perf`/etc. —
see `models_*_tests.yaml` and the `tier` key above.

If you're unsure which level a test belongs in, default to the shortest/most-frequent level whose
runtime budget fits it — over-placing tests in `sanity`/`smoke` slows down everyone's merge; under-
placing a slow test there will blow the time budget check.

## Time budgets

`.github/time_budget.yaml` caps, per `team -> pipeline -> sku`, the total minutes that team's tests
may consume for that pipeline/SKU combination. `verify_time_budget.py` sums the `timeout` values
across all entries in a `tests/pipeline_reorg/*.yaml` file (grouped by `team` + `sku`) and fails
the job if the sum exceeds budget — this runs as the first step of every impl workflow, before any
hardware is touched, so a budget overrun fails fast and cheaply.

- **Bumping a budget**: increase the number in `time_budget.yaml`. Per `ci-cd.instructions.md`,
  **always include a comment explaining the new number and which job(s) drove it** — reviewers
  (and the AI reviewer) will flag an unexplained budget change.
- **Tiers**: if your pipeline passes a `tier` to `verify_time_budget.py`, the budget is looked up
  under `<pipeline>_tier<N>` instead of the plain `<pipeline>` key (see `models_e2e_tests.yaml`).
- **Per-test ceiling**: some pipelines (e.g. smoke/merge-gate) also pass `--max-per-test-timeout`
  to enforce that no single test entry exceeds a hard ceiling, separately from the team-wide sum.
- **Known gap — read before assuming budget checks always run pre-merge**: not every registry is
  wired into `pr-gate.yaml` / `merge-gate.yaml`. Pipelines that only run on `schedule` or manual
  `workflow_dispatch` do **not** get `verify_time_budget.py` run against them pre-merge — a budget
  mismatch there won't be caught until the next scheduled run fails (this happened in production:
  a budget bump was validated on a branch, then dropped by a later rebase, and the mismatch wasn't
  caught for ~10 hours — see `MINFRA-1303`). If you're adding tests to a schedule-only pipeline,
  don't rely on CI to catch a budget mismatch before merge — double check the sum yourself, or run
  the pipeline manually via `workflow_dispatch` first (some team playbooks, e.g.
  `models/MIGRATING_TO_TIERED_CI.md`, call this out explicitly).
- A universal pre-merge budget-consistency check (independent of which pipeline consumes a given
  registry) is tracked but not yet built — see `MINFRA-1303`. Until it lands, the above gap is real.

## SKUs

`.github/sku_config.yaml` is the single source of truth mapping a logical SKU name (what you write
in a test yaml's `skus:` key) to concrete GHA `runs_on` runner labels. Check here before assuming
you need a new SKU — most machine configurations already have an entry.

- **`merge_queue_sku`**: some logical SKUs (e.g. `wh_n150_civ2`) alias to a different, higher-
  priority concrete SKU when the triggering event is `merge_group` (merge-queue runs get priority
  runners). `prepare_test_matrix.py --event merge_group` handles this rewrite automatically — you
  don't do anything special in the test yaml, just use the logical SKU name.
- **Renaming a SKU is not "just find and replace"**: if you rename or remove a SKU in
  `sku_config.yaml`, every `tests/pipeline_reorg/*.yaml` and `time_budget.yaml` entry that
  references the old name will silently stop matching — the test simply drops out of the matrix
  with **no error**, since an unmatched SKU key is just absent from `enabled_skus`, not invalid.
  This has caused a real incident (a full demo test group silently produced zero jobs after a SKU
  rename touched `sku_config.yaml` but missed the corresponding `tests/pipeline_reorg/` and
  `time_budget.yaml` references). When renaming a SKU, grep all three files
  (`sku_config.yaml`, every `tests/pipeline_reorg/*.yaml`, `time_budget.yaml`) in the same PR.

## Ownership (`owner_id`) and failure notification

`owner_id` in a test entry is the primary, preferred way to assign ownership for Slack failure
pings — see `tests/pipeline_reorg/*.yaml` everywhere. A background job
(`.github/actions/analyze-workflow-data/update-owners-from-pipeline.js`) periodically syncs these
into `.github/actions/analyze-workflow-data/owners.json` automatically, so you should **not** need
to hand-edit `owners.json` for a job that's already in `pipeline_reorg` format — only pre-reorg
jobs still use `owners.json` directly (substring-matched against the full `workflow / job` name).
If a test has no `owner_id`, it shows up as unowned in CI health reporting and nobody gets pinged
on failure — always set one.

## Common pitfalls

These are real incidents, not hypotheticals — check for them before merging a pipeline_reorg
change:

- **Silent SKU-name drift** (see "SKUs" above) — renaming a SKU without updating every consumer
  drops tests from the matrix with no error, not a failure.
- **Budget bumps dropped by a rebase** — validate your final rebased diff, not just what you tested
  earlier on the branch, especially for schedule-only pipelines where nothing re-checks pre-merge
  (see "Time budgets" above, `MINFRA-1303`).
- **Multi-SKU tests can't override `cmd` per SKU** — a test's `cmd` is shared across every SKU in
  its `skus:` map; if you need different commands/filters per machine, split into separate entries
  rather than trying to conditionally branch inside one `cmd`.
- **A SKU with zero matching tests doesn't fail loudly in every path** — `prepare_test_matrix.py`
  errors if *no* SKU produces any tests, but an individual SKU with no matches for a given test
  selection can look like a silent skip in the matrix. If a job you expected to run didn't show up
  in the Actions UI, check the `id`/`sku` spelling in the test yaml against what the parent
  workflow actually enabled, before assuming infra is broken.
- **Don't hand-edit `owners.json` for reorged jobs** — it gets overwritten by the sync script; edit
  `owner_id` in the test yaml instead (see "Ownership" above).

## Reference

- Design doc: [Proposed pipeline and test organization changes](https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/1396506680) (Confluence, MI6)
- Migration tracking: [MINFRA-408](https://tenstorrent.atlassian.net/browse/MINFRA-408) (epic), [MINFRA-415](https://tenstorrent.atlassian.net/browse/MINFRA-415) (candidate pipeline list)
- AI/human PR review rules: [`.github/instructions/ci-cd.instructions.md`](../../.github/instructions/ci-cd.instructions.md)
- Job ownership: [How To Update Job Ownership](https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/2502394033) (Confluence)
- Time budgets: [`.github/time_budget.yaml`](../../.github/time_budget.yaml)
- SKU configs: [`.github/sku_config.yaml`](../../.github/sku_config.yaml)
- Time budget tooling: [`.github/scripts/utils/verify_time_budget.py`](../../.github/scripts/utils/verify_time_budget.py), [`.github/scripts/utils/prepare_test_matrix.py`](../../.github/scripts/utils/prepare_test_matrix.py), [`.github/scripts/utils/query_time_budget.py`](../../.github/scripts/utils/query_time_budget.py)
- Questions / help: Slack `#tt-metal-pipelines`, or `@metalinfra`
