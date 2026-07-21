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
- **10. Declare performance / accuracy targets** in
  [`models/model_targets.yaml`](./model_targets.yaml) for every (model,
  SKU, batch_size) combination your test runs. The CI verifier checks
  the test's benchmark payload against these on every run and fails on
  drift. Tier 3 models are exempt from perf targets, but accuracy
  targets still apply. See
  [Performance / accuracy targets](#performance--accuracy-targets) for
  the schema and SKU-aliasing rules. Ongoing standardization is
  tracked in [#42671](https://github.com/tenstorrent/tt-metal/issues/42671).
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

Use the **human-readable hardware names** in the SKU column, matching
neighbouring rows — not the internal `wh_n150` / `bh_quietbox_2` SKU
keys. Common values:

| Display name      | Internal SKU key            |
|---|---|
| WH N150           | `wh_n150`                   |
| WH N300           | `wh_n300`                   |
| WH LLMBox         | `wh_llmbox` / `wh_llmbox_perf` |
| WH Galaxy         | `wh_galaxy` / `wh_galaxy_perf` |
| BH P150           | `bh_p150`                   |
| BH P300           | `bh_p300` (LFC-mode — see [Blackhole weight-cache modes](#blackhole-weight-cache-modes)) |
| BH QuietBox 2     | `bh_quietbox_2` (2× P300)   |
| BH Galaxy         | `bh_galaxy` / `bh_galaxy_perf` |

> **Blackhole status:** BH SKUs are first-class citizens of the 3-tier
> CI as of June 2026 (see PR #45300). Add BH rows to
> `models/model_ci_tiers.md`, the registry yamls, and the relevant
> workflow `model:` dropdowns the same way you'd add WH rows — but
> read [Blackhole weight-cache modes](#blackhole-weight-cache-modes)
> first if your model targets BH P300 (LFC) or any of the local-disk
> runners.

---

## Migrating tests off legacy pipelines

If your model is currently running in any of:

- `tests/pipeline_reorg/t3k_e2e_tests.yaml`,
  `t3k_unit_tests.yaml`, `t3k_demo_tests.yaml`, `t3k_perf_tests.yaml`,
  `t3k_integration_tests.yaml`
- `tests/pipeline_reorg/galaxy_*_tests.yaml`
- `tests/pipeline_reorg/blackhole_demo_tests.yaml`
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

The same `/mnt/MLPerf/...` paths work transparently on Blackhole runners
too — the per-job impl yaml mounts the right host path under
`/mnt/MLPerf/huggingface` based on the SKU's cache mode. See
[Blackhole weight-cache modes](#blackhole-weight-cache-modes) below for
the LFC exception (BH P300).

---

## Blackhole weight-cache modes

Different BH runners host their model-weight cache in different places.
The mapping lives as a `weights-cache-mode` field on each SKU entry in
[`.github/sku_config.yaml`](../.github/sku_config.yaml)
and is consumed by [`models-e2e-tests-impl.yaml`](../.github/workflows/models-e2e-tests-impl.yaml)
/ [`models-unit-tests-impl.yaml`](../.github/workflows/models-unit-tests-impl.yaml)
when each tier job spins up its container.

| `weights-cache-mode` | Host source mounted at `/mnt/MLPerf/huggingface` | Used by SKUs |
|---|---|---|
| `cloud-mlperf` | `/mnt/MLPerf/huggingface` (shared NFS) | `bh_p150`, `bh_loudbox`, `bh_deskbox`, BH Galaxy SKUs, all WH SKUs |
| `local-disk` | `/localdev/blackhole_demos/huggingface_data` (per-runner) | `bh_quietbox_2`, `bh_llmbox`, `bh_p150_perf` |
| `lfc` | (no mount — weights fetched at job start) | `bh_p300`, `bh_p150b_civ2` |

For most BH SKUs you don't need to do anything special — the standard
`HF_MODEL=<org>/<name>` + `TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/<org>/<name>`
pattern works because the impl yaml maps the host cache into the
canonical path.

**Exception: LFC-mode SKUs (`bh_p300`).** The cache is not pre-populated
on these runners; the test job must pull the weights itself at the
start of each run. Bundle the `wget` into the test's `cmd:` (not into
the workflow yaml — keep SKU-specific behavior co-located with the
test definition). Use `/mnt/MLPerf/huggingface/...` paths inside the
cmd; the impl yaml mounts the per-runner LFC cache at that path so
the wget output persists across jobs.

Reference entry (see
[`tests/pipeline_reorg/models_e2e_tests.yaml`](../tests/pipeline_reorg/models_e2e_tests.yaml)
for the live version):

```yaml
- name: Llama 3.1-8B data-parallel e2e tests (P300 LFC)
  cmd: |
    set -e
    mkdir -p /mnt/MLPerf/huggingface/meta-llama
    wget -r -nH -x --cut-dirs=5 -np --progress=dot:giga -R "index.html*" \
      -P /mnt/MLPerf/huggingface/meta-llama \
      http://yyz2-lfcache564.yyz2.tenstorrent.com/mldata/model_checkpoints/pytorch/huggingface/meta-llama/Llama-3.1-8B-Instruct/
    export HF_MODEL=/mnt/MLPerf/huggingface/meta-llama/Llama-3.1-8B-Instruct
    export TT_CACHE_PATH=/mnt/MLPerf/huggingface/meta-llama/Llama-3.1-8B-Instruct
    pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "<your selector>"
  model: llama3.1-8b-dp
  skus:
    bh_p300:
      timeout: 25
      tier: 2
```

Note that the LFC entry uses **file-path** `HF_MODEL` (the local
checkpoint directory) rather than the `<org>/<name>` hub id, because
the LFC cache is laid out as a flat checkpoint tree, not the HF hub
cache structure.

If you add a new BH SKU that isn't covered above, add a `weights-cache-mode`
field to its entry in `.github/sku_config.yaml` before adding tier yaml
entries for it — otherwise the impl yaml will fall back to the default mount
and the job will fail at container start on hosts without `/mnt/MLPerf`.

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

### Adding a brand-new SKU to the dropdowns

Each tier workflow also exposes a `sku:` dispatch input listing every
SKU that pipeline accepts. If you add a tier yaml entry that targets a
SKU not already in the dropdown of that workflow, also add it to:
- The `sku:` input `options:` list (e.g. `- "bh_quietbox_2 (BH QB2)"`).
- The `ALL_SKUS=` shell variable in the workflow's `resolve-skus` step.
- The corresponding lists in
  [`.github/workflows/all-model-tests.yaml`](../.github/workflows/all-model-tests.yaml)
  — `BH_SKUS` / `WH_SKUS`, `VALID_SKUS`, the `description:` string,
  and the `::error::` message. The `wh_all` and `bh_all` expansion
  macros in that wrapper drive what runs on a full-matrix dispatch.

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

Targets live in [`models/model_targets.yaml`](./model_targets.yaml).
The CI verifier loads this file at the end of every job, looks up the
entry that matches the running (model, SKU, batch_size, seq_len), and
diffs the measured benchmark payload against it. Drift in either
direction fails the job — too-slow flags a regression, too-fast flags
an outdated target.

Every tiered model is expected to declare:

- **Accuracy** target (top1 / top5 token-matching against a reference).
- **Performance** target (`prefill_time_to_token` in seconds,
  `decode_t/s/u`, `decode_t/s`, with a `decode_tolerance` for slack).

**Tier 3 models are exempt from perf targets** (the perf fields can be
omitted or set to `{}`), but accuracy targets still apply.

### Entry schema

```yaml
targets:
  <model-key>:                       # short kebab-case; matches `model:` in registry yaml
    aliases: ["<HF-Name>", "<HF/Org>/<HF-Name>"]   # what the resolver matches against
    skus:
      <sku-key>:
        entries:
          - batch_size: 32           # optional; omit for a generic entry
            seq_len: 1024            # optional; omit for a generic entry
            status: active           # active | TODO
            perf:
              prefill_time_to_token: 0.136   # seconds
              decode_t/s/u: 16.30
              decode_t/s: 521.6              # = batch_size * decode_t/s/u
              decode_tolerance: 1.15         # 15% slack on decode_t/s/u
            accuracy:
              top1: 96.0                     # percent
              top5: 100.0
            owner_id: U03XXXXXXXX
            team: models
```

The resolver picks the entry whose `batch_size` and `seq_len` best
match the caller. Generic entries (no `batch_size` / `seq_len`) match
any call but lose to a more specific entry when both exist.

### SKU naming in the targets yaml

The resolver normalizes the SKU input via aliases declared in
[`models/demos/utils/model_targets.py`](./demos/utils/model_targets.py).
The notable one for BH: `p300x2`, `p150x4`, and `bh_quietbox_2` all
resolve to the same canonical entry — because `determine_device_name`
labels 2× P300 hardware as `"P150x4"` today even though it's really a
P300x2. Use whichever SKU key you prefer in the yaml; the resolver
finds it from any alias.

### In-test perf checks

In addition to the centralized YAML, `tt_transformers` demos that
ship a hardcoded `ci_target_ttft` / `ci_target_decode_tok_s_u` dict
inside `simple_text_demo.py` keep using those for an in-test
`verify_perf` call. When you add or update a model in
`model_targets.yaml`, add the matching `<device_name>_<model_name>`
entry in those dicts too so both checks agree. The keys use the value
`determine_device_name` returns (e.g. `P150x4_Llama-3.3-70B`).

### When you don't have numbers yet

If your model is new and you haven't measured perf or accuracy yet,
add a `status: TODO` entry with empty `perf: {}` / `accuracy: {}` and
populate it after the first CI run. The resolver skips TODO entries
by default so the test doesn't fail on missing numbers, but the
entry is on the books so it's not forgotten.

### Trace region sizes

Trace buffer sizes live in [`models/model_trace_region_sizes.yaml`](./model_trace_region_sizes.yaml).
Add a `(model, SKU)` block with `trace_region_size: <bytes>` whenever a
demo or test needs a specific reserved trace region. Unconfigured `(model,
SKU)` pairs are **not** an error: [`resolve_trace_region_size`](./demos/utils/trace_region_sizes.py)
logs an info message and falls back to `TRACE_REGION_SIZE_DYNAMIC` (`0`,
dynamic allocation). Add an explicit entry when a model needs a fixed
reserved size rather than dynamic allocation.

- **Model keys** — same short kebab-case + `aliases` convention as `model_targets.yaml`.
- **SKU keys** — canonical names (`wh_n150`, `wh_llmbox_perf`, `bh_p150`, …); legacy labels like `T3K` / `P150x4` / `wh_llmbox` / `bh_galaxy` resolve via `normalize_sku` in [`model_targets.py`](./demos/utils/model_targets.py).
- **`tt_transformers`** — `get_supported_trace_region_size` in [`demo/trace_region_config.py`](./tt_transformers/demo/trace_region_config.py) loads from the YAML automatically when `HF_MODEL` is set (root [`conftest.py`](../conftest.py) applies the override on `mesh_device`).
- **Other demos** — call `resolve_trace_region_size(model_name, get_current_device_sku_name())` from [`demos/utils/trace_region_sizes.py`](./demos/utils/trace_region_sizes.py).

#### Galaxy and other bypass demos

Galaxy e2e/sweep jobs use **different** trace sizes than the matching
`tt_transformers` model on the same SKU. Do not reuse the tt_transformers
key — add a separate model block, e.g.:

| Model key | SKU | Notes |
|---|---|---|
| `llama3.3-70b-galaxy` | `wh_galaxy_perf` | Full e2e / prefix-caching (216 580 672 bytes) |
| `llama3.3-70b-galaxy-decode` | `wh_galaxy_perf` | Decode-only benchmarks (23 887 872 bytes) |
| `llama3.3-70b-galaxy-qwen` | `wh_galaxy_perf` | Qwen-on-galaxy stack |
| `qwen3-32b-galaxy` | `wh_galaxy_perf` | Galaxy e2e (102 000 000 bytes) |
| `qwen3-32b-galaxy-decode` | `wh_galaxy_perf` | Galaxy decode benchmarks |

For pytest demos that open a device via the shared `device_params` /
`mesh_device` fixtures, pass the YAML model key through parametrize instead
of hardcoding bytes:

```python
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ..., TRACE_MODEL_KEY_PARAM: "llama3.3-70b-galaxy"}],
    indirect=True,
)
```

The `mesh_device` fixture pops `TRACE_MODEL_KEY_PARAM` and resolves it to
`trace_region_size` at device-open time, using the SKU of the **logical
submesh** actually opened (derived from the mesh shape / `data_parallel` /
`MESH_DEVICE`) — not the physical cluster. This matters for runs that open a
sub-slice of a larger machine (e.g. a `1x4` slice of a Galaxy, or
`MESH_DEVICE=N300` on a T3K).

For demos that open a device directly (no shared fixture), use
`build_trace_device_params(model_key)`:

```python
from models.demos.utils.trace_region_sizes import build_trace_device_params

device_params = build_trace_device_params("deepseek-v3")
```

#### Dynamic allocation (`trace_region_size: 0`)

Dynamic allocation lets the runtime size trace buffers at launch instead of
reserving a fixed region. It is the **default** for any `(model, SKU)` pair
not present in the YAML (resolution logs an info message and returns
`TRACE_REGION_SIZE_DYNAMIC`). A model can also opt in explicitly by setting
`trace_region_size: 0` (see `deepseek-v3`); use the named constant
`TRACE_REGION_SIZE_DYNAMIC` from `trace_region_sizes.py` when referencing the
value in code or comments. Do **not** assign `trace_region_size = …` in demo
code — always go through the resolver or `build_trace_device_params`.

#### CI coverage test

[`models/tt_transformers/tests/test_trace_region_sizes.py`](./tt_transformers/tests/test_trace_region_sizes.py)
checks that every tiered CI job that sets `HF_MODEL` (including per-SKU
`hf_model` placeholders in device-perf entries) resolves to a valid size —
either an explicit YAML entry or the dynamic-allocation fallback (`0`). Run
locally (without hardware):

```bash
pytest models/tt_transformers/tests/test_trace_region_sizes.py --noconftest -v
```

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
| Legacy `t3k_*` / `galaxy_*` / `blackhole_demo_tests.yaml` | **Remove** any old entries for this model — duplicate scheduling wastes runners and produces conflicting signal. |
| `models/model_targets.yaml` | Add (model, SKU, batch_size) entries with perf + accuracy targets. |
| `.github/sku_config.yaml` | Only if you're adding a brand-new BH SKU — add a `weights-cache-mode` field to its SKU entry before adding tier yaml entries. |
