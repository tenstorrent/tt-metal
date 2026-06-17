---
description: 'PR review rules for model implementations and demos'
applyTo: 'models/**'
excludeAgent: "cloud-agent"
---

# Models Review

## 🔴 CRITICAL

- **TTNN path must be exercised**: if a model claims TTNN acceleration, verify the forward path actually executes through TTNN ops — not fall-through to PyTorch. A forward method that returns `pytorch_model(input)` while wrapping a TTNN class is a non-functional port.
- **Weight fidelity**: model conversion must preserve original HuggingFace/reference weights. Flag any code that initializes new random weights instead of copying from the source model's state dict. Converted models must match the reference output within stated PCC tolerance.
- **`prepare.py` API contract**: weight folding, reshaping, and preprocessing must happen *inside* the `prepare.py` callback pattern (matching existing attn_norm/q_norm style), not injected above it. Breaking this contract breaks demos and tests that rely on the HF state dict keys.

## 🟡 IMPORTANT

- **No hardcoded SKU or device topology**: use `ttnn.cluster.get_cluster_type()` to determine the running hardware. Do not lock available SKUs in model code — validate against the centralized perf-targets file instead.
- **No hardcoded inputs derivable at runtime**: sequence lengths, batch sizes, and input shapes should come from the actual inputs or config, not be hardcoded constants that silently drift out of sync.
- **Performance targets in centralized YAML**: perf/accuracy numbers belong in the centralized targets file, not scattered across individual model demos or README tables. Duplicated numbers go stale.
- **CI model tier registration**: new models must be added to the appropriate CI tier (see `models/model_ci_tiers.md` and the tiered pipeline YAMLs). A model without CI coverage will silently regress.
- **No large binary/data files**: do not commit CSVs, weight checkpoints, large tensors, or sweep result files to the repo. Use external storage or generate test data programmatically.
- **Don't modify deprecated files**: check whether a model demo or test file is deprecated before editing. Stale files (e.g., old `demo_t3000.py`, legacy perf tables) should not accumulate new changes.
- **Magic numbers need provenance**: trace region sizes, tolerance thresholds, timeout values, and block sizes must have a comment explaining their derivation or a reference to the source.
- **Time budgets**: CI timeout values must be calibrated to actual measured runtime + variability buffer (typically +25%). Don't guess; measure and document.

## 🟢 SUGGESTION

- Prefer fused TTNN ops (`ttnn.lerp`, `ttnn.rms_norm`) over manually decomposed equivalents. If a fused version isn't used, open an issue documenting the accuracy/perf delta.
- Let pipeline configs derive parallel/sharding settings from architecture and mesh shape rather than hardcoding per-test.
- Don't silently mask errors: clamping vocab indices, swallowing exceptions, or zeroing invalid outputs hides real bugs. Raise or log explicitly.
- Link README performance claims to the centralized source of truth rather than duplicating numbers that will go stale.
- Keep tests focused on a single concern — don't duplicate coverage across test files. If sampling is tested in `test_sampling.py`, the vLLM generator test should focus on the pass-through boundary.
- Use `ttml::test_utils` (or Python equivalents) for random data generation to keep test setup consistent across models.
- Add tests for new weight preparation logic in `test_prepare_weights.py`.

## Review Checklist

- [ ] TTNN forward path is actually exercised (not PyTorch fallthrough)
- [ ] Weights correctly transferred from reference model (no random init)
- [ ] Weight preprocessing follows `prepare.py` callback pattern
- [ ] No hardcoded SKU — uses `ttnn.cluster.get_cluster_type()` or centralized targets
- [ ] No hardcoded seq lengths / batch sizes derivable from inputs
- [ ] Performance targets in centralized YAML (not README or inline)
- [ ] Model added to appropriate CI tier
- [ ] No large CSV/data/weight files committed
- [ ] Magic numbers documented with provenance comment
- [ ] Time budgets calibrated from actual CI measurements
