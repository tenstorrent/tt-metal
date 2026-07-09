---
name: models-ci
description: Add a model to the models CI tier tests.
---

## DiffusionGemma adaptation

Load `diffusion-gemma` first. The representative units are bidirectional SDPA,
three-phase KV, denoise decisions, and canvas sampling—not autoregressive
decoder kinds. The device RUN regression is
`models/experimental/diffusion_gemma/tests/test_device_text_demo_run.py`, and
QB2 entries belong in `tests/pipeline_reorg/blackhole_e2e_tests.yaml`.
Preserve checkpoint-optional skips and do not generate/read TTNN cache files in
CI.

Read `models/model_ci_tiers.md` and follow the instructions to add the user's named model to their requested tier. If the user did not specify a tier, default to Tier 3. For DiffusionGemma, select the device RUN regression plus a representative diffusion-specific unit such as denoise decisions, bidirectional SDPA, three-phase KV, or canvas sampling; do not select autoregressive decoder-layer kinds. For ordinary autoregressive LLMs, choose each implemented decoder-layer kind. For other models, use your judgement to choose the most critical module.

Add the model to the appropriate pipeline manifests. For DiffusionGemma this
includes `tests/pipeline_reorg/blackhole_e2e_tests.yaml`; use the generic
models manifests only where the selected test tier actually runs.

Update `models/model_ci_tiers.md` with the new model and tier.

CI runners do not have write access so make sure your tests do not generate or read a TTNN cache file - they should convert from the huggingface weights each time.

Use `gh` to run your new CI jobs for the new model and tier. Wait for them to complete and verify that they pass; repeat runs blocked by infrastructure flakiness and investigate and fix any other issues or problems.

Goal completion criteria:
- [ ] The checklist in `models/MIGRATING_TO_TIERED_CI.md` is fully-completed.
- [ ] The CI jobs for the new model and tier are passing.
- [ ] The model is added to the `models/model_ci_tiers.md` file.
- [ ] The model is added to every applicable pipeline manifest, including the
      Blackhole QB2 manifest for DiffusionGemma.
