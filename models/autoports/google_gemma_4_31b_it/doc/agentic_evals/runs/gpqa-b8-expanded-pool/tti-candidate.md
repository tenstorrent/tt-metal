# Held C8/B8 GPQA20 configuration

Local source-only TTI commit `7ad6cf50d160a2e9cbe007fc4475af3c8f8bf365`, parent `af1ef17abfecb059da13d0a1287c00eace0f210e`, on isolated branch `prep/gpqa20-c8-expanded` at `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa20-c8-expanded`. Worktree clean. No push or dispatch; published normal40 branch unchanged. Scored use is conditional on coordinator acceptance of local expanded-pool capacity and exactness.

Production delta from parent: frozen first40 reduced to original frozen20, GPQA planned clients4→8, and model-entry env `GEMMA4_B8_KV_POOL_IDS="20480"`. B8/max_num_seqs8, context262144, output126976, timeout14400, thinking/sampling/revisions and accepted request/response logging remain unchanged. SWE/Terminal stay disabled for nightly. The generated route remains stock EVALS_COMMON.

AST comparison of the entire model EvalConfig against repaired20 TTI `37ee508969c730320c01259fafe340e6c2c14d4e` passed after normalizing only max_concurrent8→4: exact cohort ordering, all remaining task settings and limits match. Parsed whole model YAML equals parent after removing only the new scoped env. No raw tasks or answers are copied here.

Validation:36 focused tests passed; all repository hooks passed (ruff, formatter, full pytest hook, copyright). Hook log is private `bringup/artifacts/gpqa-c8-expanded-tti-hooks.log`; staged diff check passed. Five committed files are the eval config, model YAML and three focused test files. One stale40 test expectation was corrected during preparation, with final rerun passing.

This configuration requires the new model-owned helper in execution commit `3a7704d6` (or its coordinator-published equivalent); the old ab273 image does not contain that helper. Do not dispatch with an old immutable image merely because the TTI delta is configuration-only. Publication/image binding belongs to the coordinator after local confirmation.
