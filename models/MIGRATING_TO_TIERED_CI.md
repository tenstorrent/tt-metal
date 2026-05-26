# Migrating a model to the 3-tier Models CI

This guide walks model owners through porting an existing model from the
legacy CI pipelines (single-card, T3000, Galaxy, etc.) onto the new
**3-tier Models CI**.

It is a checklist plus the reasoning behind each step. If you only want
to know *what* to edit, skim the [step-by-step checklist](#step-by-step-checklist).
If you want to know *why* the system is shaped this way, read the
[overview](#overview) first.

For the list of models part of the 3-tier CI infra, see
[`models/model_ci_tiers.md`](./model_ci_tiers.md).

---

## Overview

The 3-tier Models CI is organised into **six scheduled pipelines** —
three tiers (1, 2, 3) × two test types (e2e, unit). Each runs on a
daily cron on `main` and is also dispatchable manually:

| Pipeline | Workflow | Filterable list |
|---|---|---|
| (Tier 1) Models e2e | `.github/workflows/models-t1-e2e-tests.yaml` | `workflow_dispatch.inputs.model` |
| (Tier 1) Models unit | `.github/workflows/models-t1-unit-tests.yaml` | `workflow_dispatch.inputs.model` |
| (Tier 2) Models e2e | `.github/workflows/models-t2-e2e-tests.yaml` | `workflow_dispatch.inputs.model` |
| (Tier 2) Models unit | `.github/workflows/models-t2-unit-tests.yaml` | `workflow_dispatch.inputs.model` |
| (Tier 3) Models e2e | `.github/workflows/models-t3-e2e-tests.yaml` | `workflow_dispatch.inputs.model` |
| (Tier 3) Models unit | `.github/workflows/models-t3-unit-tests.yaml` | `workflow_dispatch.inputs.model` |

The actual job matrix is *not* in the workflow files — it lives in two
central registries:

- `tests/pipeline_reorg/models_e2e_tests.yaml`
- `tests/pipeline_reorg/models_unit_tests.yaml`

Each entry in those registries declares which SKUs it runs on and which
tier it belongs to. The workflows read the registry at runtime and
dispatch jobs whose `tier` matches the workflow.

### Additional pipelines (manual today, scheduled later)

In addition to the six scheduled pipelines above, the same 3-tier
infrastructure already hosts **sweep** and **device-perf** pipelines.
These exist on `main` and follow the **same registry-driven approach**
as e2e / unit, but are **not yet on a daily cron** — they're only
runnable via `workflow_dispatch`.

**You can register tests in them today**, following the same steps as
the `e2e` and `unit` pipelines. Automated schedules will be enabled
once these registries have tests.

| Pipeline | Workflow | Registry |
|---|---|---|
| (Tier 1) Models sweep | [`models-t1-sweep-tests.yaml`](https://github.com/tenstorrent/tt-metal/actions/workflows/models-t1-sweep-tests.yaml) | `tests/pipeline_reorg/models_sweep_tests.yaml` |
| (Tier 2) Models sweep | [`models-t2-sweep-tests.yaml`](https://github.com/tenstorrent/tt-metal/actions/workflows/models-t2-sweep-tests.yaml) | `tests/pipeline_reorg/models_sweep_tests.yaml` |
| (Tier 1) Models device-perf | [`models-t1-device-perf-tests.yaml`](https://github.com/tenstorrent/tt-metal/actions/workflows/models-t1-device-perf-tests.yaml) | `tests/pipeline_reorg/models_device_perf_tests.yaml` |

Entry shape, env conventions, naming, owner_id rules, and the
`workflow_dispatch.inputs.model` enum requirement are all identical to
e2e / unit — see the rest of this guide. The only difference is the
registry filename and the corresponding workflow file you edit.

### Which tier should I assign to my model?

The tier (1, 2, or 3) is a product/leadership decision, set by the model
lead. See [Tier assignment](#tier-assignment) below for the criteria.

---

## Step-by-step checklist

When porting `<your-model>`:

- **1. Get a tier assignment** from the model lead (1, 2, or 3).
- **2. Add the model to `models/model_ci_tiers.md`** under the
  appropriate tier table, with the SKUs it runs on.
- **3. Migrate existing tests** off the old single-card / T3000 / Galaxy
  pipelines (`t3k_*_tests.yaml`, `galaxy_*_tests.yaml`,
  blackhole-specific demo files, etc.) into the matching tiered
  registry:
  - `tests/pipeline_reorg/models_e2e_tests.yaml` for end-to-end demos.
  - `tests/pipeline_reorg/models_unit_tests.yaml` for module
    correctness tests.
  - `tests/pipeline_reorg/models_sweep_tests.yaml` for sweep tests
    (Tier 1 / 2 — manual-only today, scheduled later).
  - `tests/pipeline_reorg/models_device_perf_tests.yaml` for
    device-perf tests (Tier 1 — manual-only today, scheduled later).
- **4. Use the standard cache + weights paths** (see
  [Standard env conventions](#standard-env-conventions)).
- **5. Use the HuggingFace name** for `HF_MODEL` and the model
  identifier for the registry's `model:` and the workflow filter
  enum (see [Naming](#naming)).
- **6. Add the model to the workflow's `workflow_dispatch.inputs.model`
  enum** for every test type you registered (e2e / unit / sweep /
  device-perf). Example: Tier 2 e2e →
  `.github/workflows/models-t2-e2e-tests.yaml`. This enables model
  filtering on manual dispatch.
- **7. Set tight test timeouts** in the registry yamls
  (`tests/pipeline_reorg/models_*_tests.yaml`). Use the measured test
  time plus a reasonable margin for runner variance (~15%) —
  generous timeouts count against the shared CI budget.
- **8. Verify the time-budget table** in `.github/time_budget.yaml`
  covers the SKUs your tests need. When porting from a legacy pipeline
  this typically means *moving* budget from the legacy pipeline's
  section to the tiered pipeline's section, not adding net new budget,
  unless new tests were added.
- **9. Set a valid Slack `owner_id` and `team`** on every entry.
- **10. Bake performance / accuracy assertions into the test itself**
  (e.g. `assert pcc > 0.99`, `assert tokens_per_second >= …`) so
  regressions fail the run. Tier 3 models are exempt from perf
  targets, but accuracy assertions still apply. Once the centralized
  targets YAML
  ([#42671](https://github.com/tenstorrent/tt-metal/issues/42671))
  ships, those numbers will migrate there. See
  [Performance / accuracy targets](#performance--accuracy-targets).
- **11. Run the pipeline manually** (`workflow_dispatch`) end-to-end
  before merging. Schedule will pick it up automatically afterwards.
  The easiest way to do a one-off run for your model is via the
  [`all-model-tests`](https://github.com/tenstorrent/tt-metal/actions/workflows/all-model-tests.yaml)
  pipeline — pick the `tier` (1/2/3) and `type` (e2e/unit/sweep/device-perf)
  and filter by the new `model:` identifier you registered. This dispatches
  only your model's job rather than the whole tier matrix, so iteration
  on a flaky test is much cheaper.

---

## Tier assignment

The model lead — typically the team's tech lead or the model's owner
of record — decides the tier. Considerations:

- **Tier 1** if the model is a release-blocker, on a customer
  roadmap, or actively driving framework optimisation work.
- **Tier 2** if the model is supported and has investment, but a
  failure does not block a release.
- **Tier 3** if the model is "kept alive" — we do not want it to
  silently rot, but it is not a primary focus.

Once decided, record it in:
- `models/model_ci_tiers.md` (under the matching tier table).
- The `tier:` field of every SKU entry for that model in the registry
  YAMLs.

---

## Updating `models/model_ci_tiers.md`

[`models/model_ci_tiers.md`](./model_ci_tiers.md) lists every
supported model by tier with the SKU(s) it runs on.
Every model added to the tiered CI **must** also appear here.

Add your model under the table for its tier with the SKU column
matching the SKUs declared in the registry YAML — e.g.:

```markdown
| Qwen3-Embedding-4B | N150 |
```

Use **N150 / N300 / T3000 / WH LoudBox / WH Galaxy** in the SKU column
(human-readable hardware names, matching neighbouring rows), not the
internal `wh_n150` / `wh_llmbox_perf` SKU keys.

> **Note:** the 3-tier CI is wormhole-only today. **Blackhole SKUs**
> (`bh_p100`, `bh_p150`, `bh_loudbox`, `bh_quietbox`, …) will be
> added to the tiered registries / workflows / `model_ci_tiers.md`
> at a later stage. Until then, keep Blackhole tests in their existing
> location and don't pre-emptively add BH rows to the tiered index.

---

## Migrating tests off legacy pipelines

If your model is currently running in any of:

- `tests/pipeline_reorg/t3k_e2e_tests.yaml`,
  `t3k_unit_tests.yaml`, `t3k_demo_tests.yaml`, `t3k_perf_tests.yaml`,
  `t3k_integration_tests.yaml`
- `tests/pipeline_reorg/galaxy_*_tests.yaml`
- single-card / standalone workflow files
- or any other relevant pipelines

…remove the entry from the legacy file and re-create it in the
appropriate tiered registry:

- `tests/pipeline_reorg/models_e2e_tests.yaml` — end-to-end demos.
- `tests/pipeline_reorg/models_unit_tests.yaml` — module correctness.
- `tests/pipeline_reorg/models_sweep_tests.yaml` — sweep tests
  (Tier 1 / 2; manual-only today, scheduled later).
- `tests/pipeline_reorg/models_device_perf_tests.yaml` — device-perf
  tests (Tier 1; manual-only today, scheduled later).

**Do not leave a duplicate entry in the legacy file.** Two entries
means the test runs twice on every nightly cron, which wastes hardware
and produces conflicting signal.

### Registry entry format

Use the same shape as the existing entries. Minimal e2e example:

```yaml
- name: <Display Name> e2e tests
  cmd: |
    export HF_MODEL=<HF/Org>/<HF-Name> TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/<HF/Org>/<HF-Name>
    pytest --timeout <secs> <path/to/demo_or_test.py> -k "<selector>"
  model: <model-identifier>
  skus:
    wh_n150:
      timeout: <minutes>
      tier: <1|2|3>
  owner_id: U03XXXXXXXX # <Owner Name>
  team: models
```

Minimal unit example:

```yaml
- name: <Display Name> unit tests
  cmd: |
    export HF_MODEL=<HF/Org>/<HF-Name> TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/<HF/Org>/<HF-Name>
    pytest --timeout <secs> <path/to/unit_test.py>
  model: <model-identifier>
  skus:
    wh_n150:
      timeout: <minutes>
      tier: <1|2|3>
  owner_id: U03XXXXXXXX # <Owner Name>
  team: models
```

A model can run on multiple SKUs by adding more entries under `skus:`,
each with its own `timeout` and `tier`.

### Test conventions

Match the standard already used by neighbouring models:

- **e2e tests** are end-to-end demos under `models/demos/...` or
  `models/tt_transformers/demo/...`. They exercise the full prefill +
  decode loop (or full prefill for embedding models) and assert
  performance / token-matching against a reference.
- **Unit tests** cover module-level correctness — attention,
  decoder, MLP, prefill paths — with PCC checks against a CPU
  reference. They live alongside the model code in
  `models/tt_transformers/tests/...` or the model's own `tests/`
  directory.
- Tests must be **non-interactive** (no prompts), **deterministic**
  (seed where required), and **finish well within the declared
  `timeout`** with margin for runner variance.

---

## Standard env conventions

All tiered CI jobs should use the same shared paths so weights and
cache are re-usable across runs. Exceptions need approval from the
CI owners with a clear justification. Use these in your `cmd:` block:

| Env var | Value | Purpose |
|---|---|---|
| `HF_HOME` | `/mnt/MLPerf/huggingface` | HuggingFace download cache (weights, tokenizer). Pre-populated on the runners. |
| `HF_MODEL` | `<HF/Org>/<HF-Name>` | The HuggingFace checkpoint id, e.g. `meta-llama/Llama-3.3-70B-Instruct`. |
| `TT_CACHE_PATH` | `/mnt/MLPerf/huggingface/tt_cache/<HF/Org>/<HF-Name>` | TT-side compiled-kernel / weight-conversion cache. The path must mirror `HF_MODEL` so caches don't collide between models. |
| `MESH_DEVICE` (only when needed) | `T3K` / `TG` / `N300` / etc. | When a test needs a non-default mesh shape. Most single-device runs can omit this. |

---

## Naming

### `HF_MODEL`

Always the **HuggingFace canonical name**, e.g.:
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen3-32B`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Do **not** invent local aliases. The HF name is what `HF_HOME` looks
up, what the tokenizer pulls, and what `TT_CACHE_PATH` mirrors.

### `model:` identifier (registry + workflow filter)

A short kebab-case version of the HF model name, e.g.:

| HF name | `model:` identifier |
|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | `llama3.3-70b` |
| `Qwen/Qwen3-32B` | `qwen3-32b` |
| `Qwen/Qwen3-Embedding-4B` | `qwen3-embedding-4b` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistral-7b` |

This identifier is used in two places and **must match exactly in
both**:

1. The `model:` field in the registry YAML entry.
2. The `workflow_dispatch.inputs.model` enum in the matching tier
   workflow file (e.g. `.github/workflows/models-t1-e2e-tests.yaml`).

### Adding to the workflow filter list

Open the workflow for the tier + type your model belongs to (e.g. for
a Tier 3 e2e model, edit
`.github/workflows/models-t3-e2e-tests.yaml`), and add the identifier
to the `model` input enum:

```yaml
on:
  workflow_dispatch:
    inputs:
      model:
        type: choice
        options:
          - all
          - llama3.3-70b
          - qwen3-32b
          - <your-model>           # add here
```

If you skip this step, the registry entry will still run on the daily
cron, but **`workflow_dispatch` users will not be able to select your
model** from the dropdown — only run "all" or another model.

If your model runs on multiple test types (any combination of e2e,
unit, sweep, device-perf), it must be added to **every** matching
workflow file's `model` enum:
- `.github/workflows/models-tN-e2e-tests.yaml`
- `.github/workflows/models-tN-unit-tests.yaml`
- `.github/workflows/models-tN-sweep-tests.yaml` (Tier 1 / 2 only)
- `.github/workflows/models-t1-device-perf-tests.yaml` (Tier 1 only)

---

## Time budgets

`.github/time_budget.yaml` declares the **maximum allowed total
runtime** per pipeline + SKU. Per-job timeouts in the registry must
sum to within that budget, otherwise the pipeline cannot fit on the
allocated hardware in a single nightly slot.

(For porting-specific guidance on moving budget from a legacy pipeline
to the tiered one, see [step 8](#step-by-step-checklist) of the
checklist.)

If your new entries push a SKU over budget, either:
- Tighten your `timeout:` to the minimum that holds with margin, or
- Negotiate a larger budget with the CI owners (and document why in
  the PR description).

A common pitfall: choosing a 30-minute timeout for a 5-minute job
because "it's safer". Don't — it counts against the shared budget.

---

## Owner and team fields

```yaml
owner_id: U03PUAKE719 # Miguel Tairum Cruz
team: models
```

- `owner_id` must be a **Slack user ID**. These start with `U…`. IDs
  starting with `D…` are DM channel IDs, not user IDs, and will not
  route notifications correctly.
- `team` is currently `models` for all entries in the registry. If you
  want a different team to own the failure pings, confirm with the
  CI owners that the new team value is wired into the notification
  routing before using it.
- The trailing `# <Owner Name>` comment is required so reviewers can
  read the file without looking up Slack IDs.

---

## Performance / accuracy targets

A **centralized targets YAML** for performance and accuracy is
planned — see [issue #42671](https://github.com/tenstorrent/tt-metal/issues/42671).

When that lands, every tiered model will be expected to declare:

- **Accuracy** target (e.g. minimum PCC, minimum eval score, expected
  generation token-matching).
- **Performance** target (e.g. minimum tokens/s, maximum prefill latency)
  per SKU — **Tier 3 models are exempt from perf targets.**

For now:
- Bake the assertions into your test (e.g. `assert pcc > 0.99`,
  `assert tokens_per_second >= …`) where applicable.
- Note your target numbers in the PR description that adds the model
  to tiered CI, so they can be migrated into the centralized YAML
  cleanly when it ships.
- Track the issue (#42671) and migrate when it's available — do
  **not** invent a parallel one-off targets file in the meantime.

---

## Verifying before merge

1. Run the matching pipeline manually via `workflow_dispatch` with
   your model selected and confirm a green run.
2. Run with `model: all` to confirm you didn't break neighbouring
   entries (e.g. by colliding on `TT_CACHE_PATH` or breaking the
   YAML schema).
3. Confirm `models/model_ci_tiers.md` renders correctly in the GitHub
   preview.
4. Confirm there are no lingering entries for your model in any
   legacy YAML.

After merge, the next scheduled cron run on `main` will include your
model. Check the run from the Actions tab to confirm the model appears
and is green on the next nightly cycle.

---

## Quick reference: files you will touch

To verify before merge, dispatch the
[`all-model-tests`](https://github.com/tenstorrent/tt-metal/actions/workflows/all-model-tests.yaml)
pipeline filtered to your tier + type + the new `model:` identifier
(see step 11).

| File | What changes |
|---|---|
| `models/model_ci_tiers.md` | Add a row under the matching tier table. |
| `tests/pipeline_reorg/models_e2e_tests.yaml` | Add e2e job entry/entries. |
| `tests/pipeline_reorg/models_unit_tests.yaml` | Add unit job entry/entries. |
| `tests/pipeline_reorg/models_sweep_tests.yaml` | Add sweep entries (Tier 1/2; manual-only today). |
| `tests/pipeline_reorg/models_device_perf_tests.yaml` | Add device-perf entries (Tier 1; manual-only today). |
| `.github/workflows/models-tN-e2e-tests.yaml` | Add `model` to the `workflow_dispatch` input enum. |
| `.github/workflows/models-tN-unit-tests.yaml` | Same, if you registered a unit test. |
| `.github/workflows/models-tN-sweep-tests.yaml` | Same, if you registered a sweep test (Tier 1/2). |
| `.github/workflows/models-t1-device-perf-tests.yaml` | Same, if you registered a device-perf test. |
| `.github/time_budget.yaml` | Verify or extend the budget for the SKU you target. |
| Legacy `t3k_*` / `galaxy_*` YAMLs | **Remove** any old entries for this model. (Blackhole legacy yamls stay untouched until BH support lands in the 3-tier CI.) |
