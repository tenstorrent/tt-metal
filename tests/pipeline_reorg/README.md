# Pipeline Reorg Format

This folder is the test registry for tt-metal's CI "pipeline reorg" format. Read this before
adding/changing a test, creating a new pipeline, or reviewing a PR that touches
`.github/workflows/` or `tests/pipeline_reorg/` — written for humans and AI agents alike.

Background/rationale: [design doc](https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/1396506680).
Migration status: [MINFRA-408](https://tenstorrent.atlassian.net/browse/MINFRA-408) (not every
pipeline is migrated yet — see `.github/instructions/ci-cd.instructions.md` to spot a pre-reorg
one). Questions: Slack `#tt-metal-pipelines` or `@metalinfra`.

## Format: 3 files + 2 registries

```
.github/workflows/<name>.yaml          # thin parent workflow
.github/workflows/<name>-impl.yaml     # reusable impl workflow (does the work)
tests/pipeline_reorg/<name>_tests.yaml # test registry (this folder, dev-owned)
.github/time_budget.yaml               # team -> pipeline -> sku -> minutes budget
.github/sku_config.yaml                # sku -> runner label mapping
```

- **Parent** (`<name>.yaml`) — `workflow_dispatch` inputs, triggers the build, calls impl with
  `enabled-skus`. Minimal example: [`galaxy-e2e-tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/galaxy-e2e-tests.yaml).
- **Impl** (`<name>-impl.yaml`, `workflow_call`) — `load-test-matrix` job runs
  `verify_time_budget.py` then `prepare_test_matrix.py` to expand the registry into a GHA matrix;
  a second job runs `${{ matrix.test-group.cmd }}` per `(test, sku)`. Minimal:
  [`galaxy-e2e-tests-impl.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/galaxy-e2e-tests-impl.yaml);
  fancier (category filtering, timeout override): [`tt-train-tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/tt-train-tests.yaml).
- **Test yaml** (this folder) — dev-owned. **The only file you should need to touch for a routine
  test change.**

Workflow YAML is infra-owned plumbing; test yaml is dev-owned content — if your PR is "just
adding a test" and it touches `.github/workflows/`, something's off.

This is the standard shape, not a rule to follow to the letter for every file — reuse is fine
where it avoids near-duplicate workflows. The most common exception is a **shared impl workflow**:
[`tt-train-tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/tt-train-tests.yaml)
is one impl workflow covering `merge_gate`/`unit`/`perf` (via a `test-category` input) instead of
three nearly-identical ones. Only duplicate an impl workflow when the matrix/SKU logic actually
diverges enough that sharing would add more branching than it saves.

## Add or change a test (common case)

1. Edit the relevant `tests/pipeline_reorg/<name>_tests.yaml` — schema below.
2. Check your `(team, pipeline, sku)` timeout sum still fits `.github/time_budget.yaml`. If not,
   bump it in the same PR (see Time budgets).
3. Nothing else. No workflow YAML changes needed.

## Create a new pipeline

**Most devs shouldn't need this.** A new pipeline means a new slice of dedicated machine time and
ongoing infra maintenance, so check the "Add or change a test" path and the existing pipeline list
first. If you genuinely think you need a new one, file a request in the
[MINFRA](https://tenstorrent.atlassian.net/browse/MINFRA) Jira project (under the
[MINFRA-408](https://tenstorrent.atlassian.net/browse/MINFRA-408) epic) before building anything —
metal-infra needs to weigh in on machine time and ownership, and may already have a better fit.

Once that's agreed, create all three artifacts, then a `time_budget.yaml` entry
(`team -> pipeline -> sku: minutes`, where `pipeline` is the `workflow_name` you pass to
`verify_time_budget.py`, matching the testing level — see Pipeline levels). Only touch
`sku_config.yaml` if you need a machine config that doesn't already exist there. Copy the closest
existing impl workflow rather than writing from scratch.

Rules (from `MINFRA-415` / `ci-cd.instructions.md`):

- Exactly one parent + one impl workflow per pipeline; anything else must be shared infra actions.
- Name by intent/level (`smoke`, `sanity`, `unit`, `integration`, `e2e`, `perf`, `stress`, `sweep`),
  not frequency or architecture, where avoidable.
- Trigger only on `schedule`/`workflow_dispatch` — not `push`/`pull_request` — unless justified in
  a comment. Don't trigger on PR unless called by `pr-gate.yaml`/`merge-gate.yaml`.
- Wire up Superset upload and Slack failure notification — don't assume they come for free.
- Get it reviewed by metal-infra (`.github/instructions/ci-cd.instructions.md` — the AI reviewer
  reads this too).

## Test yaml schema

Each file is a YAML list; one entry = one logical test, expanded per SKU. Required:

| Key | Purpose |
|-----|---------|
| `name` | Display name (SKU auto-appended, don't include it yourself). |
| `cmd` | Exact shell command(s) to run. Inline any env vars you need. |
| `skus` | `{ <sku_name>: { timeout: <minutes> } }` — one or more SKUs, each expands to its own matrix entry. `sku_name` must exist in `.github/sku_config.yaml`. |
| `owner_id` | Slack member ID (starts with `U`) pinged on failure. Get yours: profile photo → View profile → ⋯ → Copy member ID. Add `# Name` for readability. |
| `team` | Must match a top-level key in `.github/time_budget.yaml`. |

Common optional keys (arbitrary extra keys are allowed if your pipeline needs them):

| Key | Purpose |
|-----|---------|
| `id` | Stable slug for `workflow_dispatch` single-test selection (`pipeline-select-*.yaml`). |
| `model` | Model identifier for per-model dispatch selection (`models_*_tests.yaml`). |
| `skus.<sku>.tier` | Model tier (1/2/3); budget looked up under `<pipeline>_tier<N>`. See `models_e2e_tests.yaml`. |
| `category` / `subcategories` | Coarser grouping than `id` (`tt-train-tests.yaml`). |
| `{key}` in `cmd` | Any other key on the entry is substituted into `cmd` as `{key}` (e.g. per-SKU cache paths). |

**These tables aren't exhaustive for every pipeline.** `prepare_test_matrix.py` passes through any
key it doesn't recognize, and an impl workflow can read `matrix.test-group.<key>` directly — so
individual pipelines grow their own extra keys, parsed only by that pipeline's impl workflow.
Check the impl workflow (and existing entries in the same test yaml) before assuming a key is
either required everywhere or unsupported. Examples already in the repo: `coverage: false` in
`llk_pr_gate_tests.yaml` (opts a job out of coverage aggregation); `weights-cache-mode` on
Blackhole demo SKUs (picks which cache volume `blackhole-demo-tests-impl.yaml` mounts); `arch`
read directly in `ops-unit-tests-impl.yaml` to set `ARCH_NAME`/perf-throttle env vars; SKU names
prefixed `sim_`, special-cased in `ttnn-smoke-tests-impl.yaml` to route to a simulator runner
instead of real hardware.

```yaml
- name: my_model_tests
  cmd: pytest models/demos/my_model/tests/
  skus:
    wh_n150_civ2:
      timeout: 10
  owner_id: U01234ABCDE # Your Name
  team: models
```

Examples by complexity: [`fabric_merge_gate_tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/tests/pipeline_reorg/fabric_merge_gate_tests.yaml) →
[`galaxy_e2e_tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/tests/pipeline_reorg/galaxy_e2e_tests.yaml) →
[`models_unit_tests.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/tests/pipeline_reorg/models_unit_tests.yaml) (multi-SKU + tier).

## Pipeline levels

| Level | Intent | Runtime | Frequency |
|-------|--------|---------|-----------|
| `smoke` | Bare minimum, merge attempt | Seconds | Every merge-queue attempt |
| `sanity` | Happy-path — if broken, Metal is broken | < 5 min/test | Post-commit |
| `unit` | Single component | Short | Scheduled |
| `integration` | Multiple components | Medium | Scheduled |
| `e2e` | Full end-to-end system | Long | Scheduled |
| `perf` | Performance measurement, perf-mode | Variable | Scheduled |
| `stress` | Repeated run, no perf assertion | Long | Infrequent |
| `sweep` | Parameter sweep | Very long | Infrequent |

Models pipelines additionally split by tier (1/2/3) within a level — see `tier` key above. When
unsure, pick the shortest/most-frequent level your runtime actually fits.

## Time budgets & SKUs

`.github/time_budget.yaml`: `team -> pipeline -> sku -> minutes`. `verify_time_budget.py` sums your
registry's `timeout`s per `(team, sku)` and fails before any hardware runs if you're over. Bump the
number with a comment explaining why — unexplained bumps get flagged in review. A `tier` input
looks the budget up under `<pipeline>_tier<N>`; some pipelines also enforce a hard per-test ceiling
via `--max-per-test-timeout`.

`.github/sku_config.yaml`: logical SKU → `runs_on` runner labels. Check here before assuming you
need a new SKU. `merge_queue_sku` auto-reroutes a logical SKU to a priority runner on
`merge_group` events — no test-yaml changes needed for that.

## Common pitfalls

Real incidents, not hypotheticals:

- **Budget bumps not checked pre-merge for schedule-only pipelines.** Only pipelines wired into
  `pr-gate.yaml`/`merge-gate.yaml` run `verify_time_budget.py` before merge. A schedule-only
  pipeline's mismatch surfaces only on its next scheduled run — e.g. a budget bump validated on a
  branch, then dropped by a later rebase, went undetected for ~10 hours (`MINFRA-1303`, fix
  pending). Double-check your final diff yourself; don't rely on CI here.
- **Renaming a SKU silently drops tests, doesn't fail.** An unmatched SKU key just disappears from
  `enabled_skus` — no error. Grep `sku_config.yaml`, every `tests/pipeline_reorg/*.yaml`, and
  `time_budget.yaml` together when renaming one.
- **`cmd` is shared across all SKUs on one entry.** Need a different command per machine? Split
  into separate entries, don't branch inside `cmd`.
- **Don't hand-edit `owners.json`.** For reorged jobs, ownership comes from `owner_id` in the test
  yaml and is synced automatically; manual `owners.json` edits get overwritten. No `owner_id` means
  nobody gets pinged on failure.

## Reference

- [Design doc](https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/1396506680) · [MINFRA-408](https://tenstorrent.atlassian.net/browse/MINFRA-408) epic · [MINFRA-415](https://tenstorrent.atlassian.net/browse/MINFRA-415) candidate list
- [`.github/instructions/ci-cd.instructions.md`](../../.github/instructions/ci-cd.instructions.md) — PR review rules (human + AI)
- [How To Update Job Ownership](https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/2502394033)
- [`.github/time_budget.yaml`](../../.github/time_budget.yaml) · [`.github/sku_config.yaml`](../../.github/sku_config.yaml)
- [`verify_time_budget.py`](../../.github/scripts/utils/verify_time_budget.py) · [`prepare_test_matrix.py`](../../.github/scripts/utils/prepare_test_matrix.py) · [`query_time_budget.py`](../../.github/scripts/utils/query_time_budget.py)
- Help: Slack `#tt-metal-pipelines`, `@metalinfra`
