# Terminal smoke retry registration result

Two-line TTI model-config change on da219dd56a241bb5314ae1da4b87104b84b0da15: SWE CI_NIGHTLY limit 2 to 0; Terminal CI_NIGHTLY limit 0 to 2. GPQA remains disabled. Frozen Terminal task names remain terminal-bench/llm-inference-batching-scheduler and terminal-bench/mailman. Native parser flags, all revision pins and every serving/eval limit remain unchanged.

Confirmation: exact active limit map and exact two task names asserted; 23 focused model-registration tests passed. Full pre-commit hooks passed (Ruff, format, pytest, copyright), with inherited tt_metal_home and TT_METAL_HOME removed. Command: `PRE_COMMIT_HOME=/tmp/gemma4-suite-precommit env -u tt_metal_home -u TT_METAL_HOME pre-commit run --files reference_config/evals/eval_config.py`. Log: bringup/artifacts/stage12-terminal-retry-hooks.log.

No commit, publication, dispatch or device work. This enables a retry candidate only; it does not assert fabric recovery or valid trajectories. Device hours zero.
