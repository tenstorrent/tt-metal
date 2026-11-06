# Suite registration result

Change kept for coordinator integration: the fresh autoport model's eval entry selects exactly r1_gpqa_diamond, swe_bench_verified, terminal_bench_2_1. Both smoke-test and ci-nightly currently select the frozen two-task smoke cohorts; this is the smoke publication, not a claim that the nightly rung passed. GPQA uses doc_ids [152,189]. Harbor uses explicit maps for both modes and two concurrent trials. CPU and memory overrides remain absent so task definitions supply budgets.

Pinned GPQA protocol retained: thinking true, temperature 1, top_p .95, top_k 20, request timeout 14400 seconds, output cap 126976. Declared context 262144 reaches the common client as 262080 due to the stock harness's 64-token safety reserve. Server context is not reduced by this registration. Existing GPQA published 84.3 and GPU 83.33 references remain metadata; subset scores require uncertainty reporting and do not establish agreement with a full-suite reference.

Tool defaults are newly registered conservative values, not an existing model release measurement: agent 21600 seconds, request 3600 seconds, output 32768, temperature 1, top_p .95, top_k20, thinking true. Dataset/agent pairs are swebench-verified/mini-swe-agent (harness constant 2.2.8) and terminal-bench/terminal-bench-2-1/terminus-2. Their reference scores are unavailable, not fabricated.

Validation: `python -m pytest -q tests/workflows/test_gemma4_autoport_smoke_suites.py` in the separate TTI checkout: 7 passed. Tests call stock common command and Harbor config builders; they verify exact lists, client concurrency, transmitted protocol and absence of resource overrides. No device, sandbox, dataset download, score or runtime trajectory was exercised. Runtime identity/trajectory audits remain necessary. No generic harness/plugin code changed.

Pre-commit copyright check passed. Ruff/pytest hooks could not activate the missing `.pre-commit/bin/activate`; default caches were also inaccessible, so copyright used `/tmp/gemma4-suite-precommit`. Direct focused pytest succeeded; ruff is not installed on current PATH. Coordinator should run hooks in its established environment before publication.

Predicted device hours 0; actual device hours 0. Confirmation satisfied at host configuration scope; live validity remains open. No publish/dispatch/commit performed by this effort.

## Host regression follow-up

Updated three existing host test files in TTI. Generic document-sharding tests now use the unchanged 26B LiveCodeBench configured-doc fixture, preserving all exact partition/merge/wrong-set rejection checks. The existing 26B/12B model-card release checks remain; a dedicated 31B assertion now verifies its exact three-suite smoke catalog, retained GPQA references, frozen IDs, and informational `should` priorities. Thinking checks cover the fresh GPQA/SWE/Terminal request paths.

Validation: `.pre-commit/bin/pytest -q tests/test_eval_shard.py tests/workflows/test_eval_config_mode_reference.py tests/workflows/test_eval_config_quetzal_refs.py`: 217 passed, 1 skipped (existing skip); one collection warning for an enum class. Ruff format/check passed on all three files. No device work, no commits.

The unrelated CLI-default test failure is inherited `tt_metal_home` from the running environment (actual execution checkout path rather than None), not catalog registration. No CLI implementation or test changed; coordinator will rerun full hooks with those environment variables removed. The separate Harbor digest test is outside this effort's ownership and is handled by the coordinator.
