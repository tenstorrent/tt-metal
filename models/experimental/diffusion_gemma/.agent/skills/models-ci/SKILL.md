---
name: models-ci
description: Add a model to the models CI tier tests.
---

Read `models/model_ci_tiers.md` and follow the instructions to add the user's named model to their requested tier. If the user did not specify a tier, default to Tier 3 and for LLMs choose each kind of decoder layer implemented in the model as the unit test to be added. For other kinds of models use your judgement to choose the most critical module to test.

Add your model to the pipeline filters in the workflow files (`tests/pipeline_reorg/models_e2e_tests.yaml` and `tests/pipeline_reorg/models_unit_tests.yaml`).

Update `models/model_ci_tiers.md` with the new model and tier.

CI runners do not have write access so make sure your tests do not generate or read a TTNN cache file - they should convert from the huggingface weights each time.

Use `gh` to run your new CI jobs for the new model and tier. Wait for them to complete and verify that they pass; repeat runs blocked by infrastructure flakiness and investigate and fix any other issues or problems.

Goal completion criteria:
- [ ] The checklist in `models/MIGRATING_TO_TIERED_CI.md` is fully-completed.
- [ ] The CI jobs for the new model and tier are passing.
- [ ] The model is added to the `models/model_ci_tiers.md` file.
- [ ] The model is added to the pipeline filters in the workflow files (`tests/pipeline_reorg/models_e2e_tests.yaml` and `tests/pipeline_reorg/models_unit_tests.yaml`).
