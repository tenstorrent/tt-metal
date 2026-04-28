# CI Time Budget System

This directory contains the configuration files that define how CI machine-hours are
budgeted across teams. The enforcement script (`scripts/budget_check.py`) reads these
files on every PR that touches pipelines, workflow files, or this directory itself.

---

## Directory contents

| File | Purpose |
|------|---------|
| `pools.yaml` | Weekly machine-hours per hardware SKU — actual Snowflake observed usage for tt-metal |
| `allocations.yaml` | Fractional share of each pool allocated to each team (**active — used by the script**) |
| `allocations_usage_based_v2.yaml` | Usage-derived allocations, pipeline_reorg + all legacy (bucket-2) workflows — reference |
| `allocations_usage_based_timeout_x_snowflake_rpw.yaml` | Hybrid: YAML timeout × Snowflake runs/week — worst-case ceiling at real cadence — reference |
| `runs_per_week.yaml` | How many times each pipeline (or test YAML) runs per week |
| `BUDGET_STATUS.md` | Auto-generated status snapshot; do not edit manually |
| `README.md` | This file |

---

## How pools.yaml was decided

`pools.yaml` lists the **actual observed weekly machine-hours** tt-metal jobs consumed on
each hardware SKU. These are the denominators in all budget calculations.

### Key format

Keys match `.github/sku_config.yaml` exactly, which is the single source of truth for
SKU → runner label mapping. CIv1 and CIv2 SKUs are separated by comments within a flat
structure (no nesting). There are no key collisions — CIv2 keys carry a `_civ2` suffix.

### Values — Snowflake actuals (tt-metal only)

```
Formula: SUM(JOB_EXECUTION_TIME) / 3600 / 28 * 7
Source:  TTDATASF.SW_TEST.CICD_JOB + CICD_PIPELINE
Filter:  PROJECT = 'tt-metal', last 28 days
```

SKU labels are matched to sku_config entries using ILIKE patterns derived from each
SKU's `runs_on` array. CIv2 labels are single strings (e.g. `tt-ubuntu-2204-N150-viommu-stable`);
CIv1 labels are comma-separated tag sets (e.g. `arch-wormhole_b0,config-t3000,in-service`).

CIv2 machines are shared across repos and orgs. Using tt-metal-only observed hours as
the denominator is intentionally conservative — it does not claim capacity beyond what
tt-metal actually consumed.

~3,800 h/wk is untracked: legacy multi-host runner labels (`multi-host-galaxy`,
`multi-host-glx-4x`, `multi-host-t3000`, bare hostnames like `g04glx03`) that have no
entry in sku_config.

### Notable observed values (April 2026)

```
CIv2:  wh_n300_civ2 5,035 h/wk   bh_p150b_civ2 2,411   wh_n150_civ2 1,053
CIv1:  wh_n300     12,704 h/wk   wh_galaxy_perf 2,404  wh_llmbox    1,152
```

`wh_n300` (CIv1 cloud VMs) dominates — it carries the bulk of ttnn-unit-tests and
fd-nightly. `wh_n300_civ2` (bare-metal CIv2) is a separate pool.

### Updating pools.yaml

Re-run the Snowflake query documented in the header of
`allocations_usage_based_timeout_x_snowflake_rpw.yaml`, replacing `COUNT(DISTINCT CICD_PIPELINE_ID)`
with `SUM(JOB_EXECUTION_TIME)/3600`. Update values when the hardware farm changes
significantly or after major pipeline restructuring.

> **Note**: `budget_check.py` currently resolves pool keys using a `RUNNER_SKU_MAP`
> dict with aggregated names (`N150`, `N300`, etc.). It needs to be updated to look up
> per-SKU keys from `sku_config.yaml` before this new format is enforced.

---

## How allocations.yaml was decided

`allocations.yaml` gives the **fraction of each pool** allocated to each team.
Values per SKU must sum to ≤ 1.0; the remainder is unallocated headroom.

### Version 1 — manual estimates (current active file)

The original `allocations.yaml` was hand-authored based on rough prior knowledge of
which teams own which hardware-intensive test suites:

- **N150 / N300**: models team runs the heavy fast-dispatch nightly suites; runtime
  owns metalium/dispatch tests; LLK team runs kernel validation.
- **p150b / p300b**: LLK team was allocated a large share based on their planned BH
  kernel work; models and runtime split the remainder.
- **LLMBOX**: mostly models (large model inference) with runtime taking T3000 dispatch.
- **galaxy**: heavily models (demo + perf), with some runtime.

These fractions were set before actual usage data was available. They are intentionally
conservative so that no team can monopolise a pool and starve others.

### Usage-based: all pipelines (allocations_usage_based_v2.yaml)

Adds all legacy bucket-2 workflows on top of pipeline_reorg. This is the most
complete picture of actual hardware consumption. See the methodology comment in
that file for the full Snowflake query and attribution logic.

| | v1 (manual) | all pipelines (v2) | timeout × sf_rpw |
|-|------------|-------------------|-----------------|
| Source | Human estimates | Snowflake actuals | YAML timeout + Snowflake rpw |
| Coverage | pipeline_reorg only | pipeline_reorg + bucket-2 | pipeline_reorg + bucket-2 |
| Cost model | `timeout × manual rpw` | actual wall-clock | `timeout × Snowflake rpw` |
| Teams | infra, llk, models, runtime | models, ttnn, runtime, scaleout, shield, llk | models, ttnn, runtime, scaleout, shield, llk |
| Pessimism | High (worst-case timeout, guessed rpw) | Low (actual times) | High (worst-case timeout, real cadence) |

Key difference: bucket-2 workflows shift N300 heavily toward models (fd-nightly alone
adds 8,200 h/wk) and N150/p150b toward runtime vs pipeline_reorg-only attribution.

```
all pipelines summary:
N300:   models 70%, ttnn 16%, runtime 14%
N150:   models 30%, ttnn 23%, runtime 41%, llk 6%
p150b:  runtime 64%, ttnn 24%, models 8%, llk 3%, scaleout 1%
p300b:  runtime 85%, models 15%
LLMBOX: runtime 81%, models 17%, ttnn 2%
galaxy: models 67%, ttnn 19%, scaleout 14%
```

To promote either file to active: update the `allocations` load path in
`budget_check.py` (line 545) and update `pools.yaml` with measured throughput.

---

## How runs_per_week.yaml was decided

`runs_per_week.yaml` maps pipeline names to estimated weekly run counts.
These are used by the script to convert a per-run timeout cost into weekly machine-hours:

```
weekly_hours = timeout_minutes / 60 × runs_per_week
```

Current values:

| Pipeline | Runs/week | Basis |
|---------|-----------|-------|
| `pr-gate.yaml` | 256 | ~51 PRs/day × 5 weekdays |
| `merge-gate.yaml` | 49 | ~10 merges/day × 5 weekdays |
| `post-commit.yaml` | 49 | same cadence as merge-gate |

For **pipeline_reorg test YAMLs** the key is the YAML filename (e.g.
`galaxy_e2e_tests.yaml: 10`). Currently these are commented out pending the
pipeline_reorg migration completing.

Update this file whenever pipeline frequency changes (e.g. after gate restructuring
or splitting a nightly into multiple cadences).

---

## How budget_check.py works

### Cost model

The script calculates *projected weekly machine-hours* per `(team, sku)` pair using
worst-case timeouts, not actual execution times:

```
cost = (timeout_minutes / 60) × runs_per_week
```

This is pessimistic by design — if every job runs to its full timeout, does this
team's allocation cover it? The v2 allocation file uses actuals instead, but the
enforcement script still uses timeouts for fast, stateless CI checks.

### Data sources

**pipeline_reorg tests** (`tests/pipeline_reorg/*.yaml`)

Each file contains a list of test entries. Each entry is one parallel job (one machine).
Fields used: `team:`, `skus:` (dict of sku_name → `{timeout, tier}`). Each entry is
one parallel job running on one machine.

```yaml
# Example: tests/pipeline_reorg/runtime_unit_tests.yaml
- name: runtime_fast_dispatch_on_eth
  cmd: ./build/test/tt_metal/unit_tests_dispatch ...
  skus:
    wh_n150_civ2:
      timeout: 10
    bh_p150b_civ2:
      timeout: 10
  team: runtime
```

**Legacy workflow files** (`.github/workflows/*.yaml`)

Workflows that use `prepare_test_matrix` / `prepare-test-matrix` are treated as
pipeline_reorg wrappers and skipped (their costs are already captured via the test
YAMLs above). All other workflows are parsed for direct `runs-on` + `timeout-minutes`
fields on each job. Team ownership is resolved from `CODEOWNERS`.

> **Limitation**: bucket-2 legacy workflows use `uses:` (reusable workflows) with
> `with: {runs_on: ..., timeout_minutes: ...}` parameters. Static YAML parsing
> yields nothing for these jobs — their `runs-on` / `timeout-minutes` values are
> not on the job dict itself. As a result, the script currently under-counts cost
> for these workflows. The usage-based v2 allocations file was produced from
> Snowflake data specifically to compensate for this gap.

### Runner label → SKU mapping

```python
RUNNER_SKU_MAP = {
    "n150":      "N150",
    "n300":      "N300",
    "p150b":     "p150b",
    "p300b":     "p300b",
    "llmbox":    "LLMBOX",
    "galaxy":    "galaxy",
    "blackhole": "p150b",   # alias
}
```

> **Pending update**: this map uses aggregated pool names (`N150`, `N300`, …) which no
> longer match pools.yaml keys. It needs to be rewritten to resolve runner labels to
> individual sku_config SKU names (e.g. `wh_n150_civ2`, `wh_n300`) using the
> `runs_on` arrays in `.github/sku_config.yaml`.

Substring match against the runner label string. CPU-only runners
(`ubuntu-latest`, `large-stable`, etc.) do not match and are silently skipped.

### Budget limit formula

```
limit(team, sku) = pools[sku] × allocations[sku][team]
```

A team is over budget if their projected weekly cost exceeds this limit.

### Modes

| Flag | What it does |
|------|-------------|
| `--report` | Full per-team-per-SKU report against current state |
| `--diff BASE_REF` | Shows only the delta introduced by the PR; fails if post-merge total exceeds limit |
| `--update-status` | Regenerates `BUDGET_STATUS.md` (always runs in CI, even on failure) |

Exit code is 1 if any team is over budget; 0 otherwise.

### Warning vs. blocking

- **OVER** (>100% of limit) — script exits 1, PR is blocked.
- **WARN** (>85% of limit) — logged but does not block.
- Workflows with no CODEOWNERS match → attributed to `unattributed`; cost is
  reported but does not count against any team's limit (warning only).

---

## CI workflow (.github/workflows/budget-check.yaml)

The workflow fires on:
- Pull requests touching `tests/pipeline_reorg/**`, `.github/workflows/**`, or
  `.github/time_budgets/**`
- Manual `workflow_dispatch`

Steps:
1. **Checkout** with full history (`fetch-depth: 0`) so `git diff` against the PR
   base is available.
2. **Install deps**: `pip install pyyaml`
3. **Check budget delta**: `budget_check.py --diff <base_sha>` — fails the PR if
   the change pushes any team over their limit.
4. **Regenerate BUDGET_STATUS.md**: runs `--update-status` regardless of step 3
   outcome (`if: always()`).
5. **Upload artifact**: the generated `BUDGET_STATUS.md` is attached to the workflow
   run so reviewers can inspect the full breakdown without re-running locally.

---

## Updating these files

### Adding a new hardware SKU

1. Add it to `sku_config.yaml` with its `runs_on` labels.
2. Add it to `pools.yaml` using the same key, with its observed weekly hours from
   Snowflake (`SUM(JOB_EXECUTION_TIME)/3600/28*7`, filtered to `PROJECT = 'tt-metal'`).
3. Add team fractions to `allocations.yaml` (must sum to ≤ 1.0).
4. Add a runner label pattern to `RUNNER_SKU_MAP` in `budget_check.py`.

### Adding a new team

Add the team key to the relevant SKU entries in `allocations.yaml` (reduce other
teams' fractions to stay ≤ 1.0).

### Changing pipeline frequency

Update the appropriate entry in `runs_per_week.yaml`. For a new pipeline_reorg YAML,
add a new key using the YAML filename (with `.yaml` extension).

### Hybrid: timeout × Snowflake runs/week (allocations_usage_based_timeout_x_snowflake_rpw.yaml)

Answers the question: *if every job ran to its configured timeout at the actual observed cadence, how would the pool be split?*

- **Timeout** — taken from `tests/pipeline_reorg/*.yaml` (new `skus:` dict format) and from static analysis of legacy impl files in `.github/workflows/`.
- **Runs/week** — `COUNT(DISTINCT CICD_PIPELINE_ID) / 28 * 7` from Snowflake, summing both main-branch and PR-branch runs.
- **Team ownership** — read directly from each pipeline_reorg YAML's `team:` field (authoritative); legacy bucket-2 workflows attributed from `CODEOWNERS`.

Use this file when you want a *conservative, cadence-anchored* limit rather than an actuals-based one. It will flag teams whose timeouts are padded far beyond actual run times.

```
N150:   runtime 37%, ttnn 36%, llk 20%, models 7%
N300:   ttnn 54%, runtime 35%, models 11%
p150b:  runtime 46%, ttnn 20%, llk 18%, scaleout 8%, models 8%
p300b:  runtime 88%, models 12%
LLMBOX: models 64%, scaleout 24%, ttnn 10%, shield 2%
galaxy: scaleout 48%, models 26%, ttnn 26%
```

Notable corrections vs. earlier estimates (April 2026 revision):
- **llk N150/p150b**: raised from 6%/3% → 20%/18% after accounting for actual
  `llk-test-wormhole`/`llk-test-blackhole` rpw (317.75/wk each).
- **LLMBOX runtime**: dropped from 7% → 0%. t3k_*.yaml `team:` fields assign those
  jobs to models, scaleout, ttnn, and shield — not runtime. Historical Snowflake
  attribution was wrong.
- **galaxy scaleout**: raised from 17% → 48% after adding `multi-host-glx-2x/4x`
  (165+51 h/wk) which was missing from earlier estimates.

### Promoting v2 allocations

When the usage-based allocations are ready to become authoritative:
1. Replace (or rename) `allocations.yaml` with the content from
   `allocations_usage_based_v2.yaml`.
2. Verify `pools.yaml` reflects current Snowflake actuals (already updated April 2026).
3. Update `budget_check.py` to resolve runner labels to per-SKU sku_config keys
   instead of the aggregated `RUNNER_SKU_MAP`.
4. Remove the now-superseded file to avoid confusion.
