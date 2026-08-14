# Mistral Small 24B Instruct 2501 — TTI release handoff

## Status

- Final classification: `release-workflow-fail` and `readiness-fail`.
- The native v9 release workflow exited 1. Its immutable native report is `native_release_report.md` / `.json` and remains authoritative: two accuracy rows failed and the benchmark task exited before producing a block.
- The benchmark harness defect was repaired afterward and the preserved acceptance-target run was regraded through the native TTI parser, target checker, acceptance checker, and report generator. `benchmark_report.md` / `.json` is a supplemental component report only; it records Benchmarks `PASS` (1/1) and does not convert the release workflow to PASS.
- No `known_issues` masks or release waivers are declared. The remaining unwaived quality gaps are IFEval 75.6635% versus 78.755% and GPQA flexible-extract 38.8889% versus 40.3% (35/90, two answers short).

## Scope and implementation

- Target implementation: `models/autoports/mistralai_mistral_small_24b_instruct_2501`; the copied spec's `impl.code_path` matches it.
- Server selector: `TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport`.
- HF model: `mistralai/Mistral-Small-24B-Instruct-2501`.
- Server mode: external OpenAI-compatible autoport vLLM API on P300X2; no Docker and no stock `models/tt_transformers` or `models/demos` implementation.
- Serving context remained 32768 with block size 32, fixed batch/max sequences 32, `max_num_batched_tokens=32768`, engine seed 9472, device sampling `all`, a 200,000,000-byte trace region, and `FABRIC_1D`.

## Provenance

- `runtime_model_spec_v9_immutable.json` is the exact runtime spec written by TTI for the v9 invocation. It records tt-metal base `1529e332a1c37937a682ba04b77e7dc3418f2589` and vLLM base `6bd775d4f3a41d09d3ed03c40b45b5f9621fff9e`.
- The launch checkout contained uncommitted changes. Logs demonstrate the launch configuration and observed behavior, but do not cryptographically attribute v9 to later vLLM commit `0e5e6495ac0a39e7c16a925140547cfb4a2e3030` or TTI commit `3e933fbbaaf71fd4859b017f8f08570e39834c09`.
- The copied `release_spec.json` is the current corrected source spec, not an immutable record of every byte used at launch. It contains no quality masks. Use the immutable runtime spec for run provenance.

## Workflow and CI-subset evidence

The v9 release command used workflow `release`, tool `vllm`, device `p300x2`, external server `http://127.0.0.1:8000`, `--limit-samples-mode ci-nightly`, `--skip-system-sw-validation`, and `--disable-trace-capture`; the complete normalized arguments are in `runtime_model_spec_v9_immutable.json` and the client log.

- IFEval used the CI-nightly 0.2 subset: 109/109 samples completed operationally, score 75.6635%, below the configured 78.755% threshold.
- GPQA used the CI-nightly 0.2 subset: 90/90 samples completed through 13 preemptions, score 38.8889%, below the configured 40.3% threshold. There were no request, transport, page, slot, retry, EngineCore, or fatal errors in that run.
- These are CI-subset scores compared with the workflow's configured thresholds; they are not unrestricted full-dataset accuracy claims.
- Measured v9 durations were approximately 18m34s for the 0.2 IFEval subset, 73m05s for the 0.2 GPQA subset, and 19m53s for spec tests. A simple five-times projection for unrestricted evals plus the measured spec run is about 7h58m before setup/report overhead. That exceeded the three-hour release watchdog/reservation window, so `ci-nightly` was used; the projection is scheduling guidance, not performance evidence.
- Spec/API conformance passed its two report blocks and all 22 parametrized vLLM chat-completion cases.

The original benchmark task failed before requests because the benchmark environment's cached Transformers 5 Mistral tokenizer lacked `is_fast`. After the compatibility fix, the exact acceptance-target point completed 8/8 requests with zero failures: mean TTFT 1272.74 ms, mean TPOT 19.19 ms, and decode throughput 34.50 tokens/s. The corrected grader selects the strictest configured measurable tier, so a spec with only functional targets is explicitly PASS rather than NA.

## Prompt and qualitative evidence

- Both evals used the HF-declared instruct chat template.
- GPQA used five official demonstrations and a held-out question. `gpqa_prompt_format_v2.json` records the inspectable seed-42 prompt construction metadata; the scored target remains the held-out item.
- `vllm_qualitative_prompt_format.json`, `vllm_qualitative_verdict.md`, `vllm_non_aligned_prompt_check.json`, and `vllm_adapter_unit_tests.log` are small, inspectable evidence copied from the completed optimized-vLLM stage. They are supporting prompt/adapter checks, not replacements for release accuracy gates.
- No raw eval sample JSONL or generated completion corpus is copied here.

## Trace warning classification

`server_release_v9.log` contains one allocator warning at startup: device buffers were allocated while a trace was active. AutoDebug identified the first post-capture KV-cache reset as the unsupported ordering: its first `ttnn.fill` program compilation occurred after model and sampling traces had been retained. Commit `993aabf2f73b911062c3bb59412af0b5fdb3e456` compiles the in-place reset in the eager pre-capture phase and retains the post-capture reset as a program-cache hit. On the exact committed server, `server_tracefix_smoke.log` contains zero active-trace allocator warnings; health, models, deterministic chat, and a one-request non-page-aligned benchmark all returned HTTP 200. This closes the allocator-warning defect without claiming that the old allocation order was safe.

The Mistral regex warning remains during the separate APIServer processor/chat-template warmup probe. It does not describe the registered live request tokenizer: `tokenizer_regex_ab.json` compares exact preserved IFEval and GPQA prompt messages, and all six live-plugin token sequences exactly match HF `fix_mistral_regex=true` while all six differ from `false`. The warning is noisy, but the live-tokenizer-bug hypothesis is refuted and does not justify rerunning the quality evaluations.

## Fixes and verification

- Production fixes retained: request-slot release; draining pending async decode token/position feedback before preemption repack; one lookahead KV token for async scheduling; cached Mistral tokenizer `is_fast` compatibility without tokenizer/template substitution; and GPQA few-shot/parser plus flexible-score-key corrections.
- Remediation removes the invalid IFEval/GPQA masks and fixes functional-only benchmark grading, with a regression test.
- Trace remediation precompiles the in-place KV-cache reset before trace capture. A focused event-order regression records reset -> capture -> reset; the complete autoport full-model test file passed 9 tests with 4 hardware-gated skips, and the nested plugin tokenizer suite passed all 6 tests.
- Focused host tests cover scheduler/preemption, tokenizer adaptation, GPQA/config, report acceptance, target grading, and model-spec parsing. Exact remediation output is recorded in `remediation_tests.log`.
- AutoFix final status remains `release-workflow-fail` / `readiness-fail`: the two mandatory quality gates remain below threshold, no waiver is valid, and the warning investigations found no surviving mechanism that predicts a score increase. A repeated full IFEval/GPQA run was therefore not presented as a fix or used to mask the failure.

## Cleanup

- Server/client processes and the `autoport-vllm-mistral-tti` tmux session were stopped; the TTI `.env` was removed.
- The exact-server AutoFix smoke was stopped with no residual API-server or EngineCore process. `tt-smi -ls --local`, reset, and a final `tt-smi -ls --local` completed successfully; all four Blackhole p300c UMD chips were visible and resettable afterward, and a fresh `MeshShape(1, 4)` opened and closed successfully.

## Artifact inventory

- Native outcome: `native_release_report.md`, `native_release_report.json`, `runtime_model_spec_v9_immutable.json`.
- Pre-v9 ordering evidence: `pre_v9_benchmark_8of8.log` and its byte-exact TTI-written `pre_v9_benchmark_8of8_runtime_spec.json`. This was the benchmark portion of an earlier release attempt, not the later v9 workflow: it records 8/8 requests, zero failures, and predates v9.
- Supplemental benchmark: `benchmark_report.md`, `benchmark_report.json`, `benchmark_raw_v9_fixed.json`, `benchmark_smoke_v9_fixed.json`.
- Aggregate eval evidence: `ifeval_v9_results.json`, `gpqa_v9_results.json`, `eval_v9_aggregate_metadata.json`.
- AutoFix evidence: `tokenizer_regex_ab.json`, `server_tracefix_smoke.log`, `chat_tracefix_smoke.json`, `benchmark_tracefix_smoke.log`, and `benchmark_tracefix_smoke.json`.
- Prompt/qualitative evidence: `gpqa_prompt_format_v2.json`, `gpqa_v5_enginecore_failure_metadata.json`, `vllm_qualitative_prompt_format.json`, `vllm_qualitative_verdict.md`, `vllm_non_aligned_prompt_check.json`, `vllm_adapter_unit_tests.log`.
- Configuration and logs: `release_spec.json`, `client_release_v9.log`, `server_release_v9.log`, `remediation_tests.log`.
- Excluded: raw eval sample JSONL, caches, weights, tensor dumps, profiler bulk, `.env`, and secrets.
