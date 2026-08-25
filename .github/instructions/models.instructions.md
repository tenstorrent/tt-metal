---
description: 'PR review rules for model implementations and demos'
applyTo: 'models/**'
excludeAgent: "cloud-agent"
---

# Models Review

## 🔴 CRITICAL

- **TTNN path must be exercised**: if a model claims TTNN acceleration, verify the forward path actually executes through TTNN ops — not fall-through to PyTorch. A forward method that returns `pytorch_model(input)` while wrapping a TTNN class is a non-functional port.
- **Weight fidelity (conversion path only)**: in the *model conversion / weight-loading* path, preserve the original HuggingFace/reference weights — flag code that initializes new random weights where it should copy from the source model's state dict, and require converted models to match the reference output within stated PCC tolerance. This does **not** apply to unit tests, which legitimately generate random weights/inputs to exercise a module in isolation; only flag random weights when they stand in for weights that should have been loaded from the reference checkpoint.
- **Weight-prep callback contract (models with a `prepare.py`)**: *if* the model already uses a weight-preparation callback pattern, keep weight folding, reshaping, and preprocessing *inside* that callback (matching the existing attn_norm/q_norm style), not injected above it — bypassing it breaks demos and tests that rely on the HF state dict keys. Today the only model with this contract is DeepSeek-V3 B1 (`models/demos/deepseek_v3_b1/weights/prepare.py`, tested by `models/demos/deepseek_v3_b1/tests/unit_tests/test_prepare_weights.py`); use that as the reference example. Do **not** ask a model that has no such callback to adopt one.

## 🟡 IMPORTANT

- **No hardcoded SKU or device topology**: use `ttnn.cluster.get_cluster_type()` to determine the running hardware. Do not lock available SKUs in model code — validate against the centralized perf-targets file instead.
- **No hardcoded inputs derivable at runtime**: sequence lengths, batch sizes, and input shapes should come from the actual inputs or config, not be hardcoded constants that silently drift out of sync.
- **Performance targets in centralized YAML**: perf/accuracy numbers belong in the centralized targets file, not scattered across individual model demos or README tables. Duplicated numbers go stale.
- **CI model tier registration**: new models must be added to the appropriate CI tier by following the full procedure in [`models/MIGRATING_TO_TIERED_CI.md`](../../models/MIGRATING_TO_TIERED_CI.md) (registry YAMLs under `tests/pipeline_reorg/`, the `workflow_dispatch` model enums, `models/model_ci_tiers.md`, `models/model_targets.yaml`, and `.github/time_budget.yaml`). A model without CI coverage will silently regress. See the [Tiered CI registration checklist](#tiered-ci-registration) below.
- **No large binary/data files**: do not commit CSVs, weight checkpoints, large tensors, or sweep result files to the repo. Use external storage or generate test data programmatically.
- **Don't modify deprecated files**: check whether a model demo or test file is deprecated before editing. Stale files (e.g., old `demo_t3000.py`, legacy perf tables) should not accumulate new changes.
- **Magic numbers need provenance**: trace region sizes, tolerance thresholds, timeout values, and block sizes must have a comment explaining their derivation or a reference to the source.
- **Time budgets**: CI timeout values must be calibrated to actual measured runtime plus a margin for runner variance (~15%, per [`models/MIGRATING_TO_TIERED_CI.md`](../../models/MIGRATING_TO_TIERED_CI.md#step-by-step-checklist)). Don't guess; measure and document. Generous timeouts count against the shared CI budget.

## 🟢 SUGGESTION

- Prefer fused TTNN ops (`ttnn.lerp`, `ttnn.rms_norm`) over manually decomposed equivalents. If a fused version isn't used, open an issue documenting the accuracy/perf delta.
- Let pipeline configs derive parallel/sharding settings from architecture and mesh shape rather than hardcoding per-test.
- Don't silently mask errors: clamping vocab indices, swallowing exceptions, or zeroing invalid outputs hides real bugs. Raise or log explicitly.
- Link README performance claims to the centralized source of truth rather than duplicating numbers that will go stale.
- Keep tests focused on a single concern — don't duplicate coverage across test files. If sampling is tested in `test_sampling.py`, the vLLM generator test should focus on the pass-through boundary.
- Use the repo-standard random-data helpers for test setup rather than hand-rolling per-model RNG, to keep test setup consistent across models. `models/**` is overwhelmingly Python, so prefer the shared Python test fixtures/utilities used by neighbouring model tests; only reach for C++ helpers (`tt::test_utils` in `tests/tt_metal/test_utils/`, or `ttml::test_utils` when actually editing `tt-train/**` C++ code) for C++ tests.
- When a model has a weight-preparation callback (see the DeepSeek-V3 B1 example above), add tests for new weight-prep logic alongside it — e.g. `models/demos/deepseek_v3_b1/tests/unit_tests/test_prepare_weights.py`.

## Review Checklist

- [ ] TTNN forward path is actually exercised (not PyTorch fallthrough)
- [ ] Conversion path transfers weights from the reference model (random init only where the test legitimately needs it)
- [ ] If the model has a weight-prep callback, preprocessing stays inside it (DeepSeek-V3 B1 pattern)
- [ ] No hardcoded SKU — uses `ttnn.cluster.get_cluster_type()` or centralized targets
- [ ] No hardcoded seq lengths / batch sizes derivable from inputs
- [ ] Performance targets in centralized YAML (not README or inline)
- [ ] Model registered in the tiered CI (see checklist below)
- [ ] No large CSV/data/weight files committed
- [ ] Magic numbers documented with provenance comment
- [ ] Time budgets calibrated from actual CI measurements (~15% variance margin)

## Tiered CI registration

When a PR adds or ports a model, confirm it is wired into the 3-tier Models
CI. These mirror the [step-by-step checklist in
`models/MIGRATING_TO_TIERED_CI.md`](../../models/MIGRATING_TO_TIERED_CI.md#step-by-step-checklist)
— flag any that are missing:

- [ ] Tier (1/2/3) assigned by the model lead
- [ ] Model row added to [`models/model_ci_tiers.md`](../../models/model_ci_tiers.md) under the matching tier table, with its SKU(s)
- [ ] Tests registered in the tiered registries under `tests/pipeline_reorg/` (`models_e2e_tests.yaml` / `models_unit_tests.yaml`, plus `models_sweep_tests.yaml` / `models_device_perf_tests.yaml` where applicable)
- [ ] Any legacy `t3k_*` / `galaxy_*` / `blackhole_demo_tests.yaml` entries for this model **removed** (no duplicate scheduling)
- [ ] `model:` identifier added to the matching workflow `workflow_dispatch.inputs.model` enum(s) (`.github/workflows/models-tN-*-tests.yaml`)
- [ ] Per-job `timeout:` set from measured runtime + ~15% margin, fitting within [`.github/time_budget.yaml`](../../.github/time_budget.yaml)
- [ ] Standard env conventions used (`HF_MODEL`, `TT_CACHE_PATH`, `HF_HOME`)
- [ ] Valid Slack `owner_id` (starts with `U…`) and `team` on every entry
- [ ] Perf/accuracy targets declared in [`models/model_targets.yaml`](../../models/model_targets.yaml) (Tier 3 exempt from perf, accuracy still required)
- [ ] Pipeline run manually via `workflow_dispatch` (e.g. `all-model-tests`) before merge
