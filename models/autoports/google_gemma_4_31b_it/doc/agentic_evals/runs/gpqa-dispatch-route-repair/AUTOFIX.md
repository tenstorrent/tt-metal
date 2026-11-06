# AutoFix: GPQA dispatch route

Diagnosis verified against both candidate source trees and the private C4 failure log: GPQA uses EVALS_COMMON, whereas workflow=agentic selects only EVALS_AGENTIC; its two remaining suites have nightly limit0 and therefore produce no result blocks. This is a coordinator dispatch error, not a model/runtime defect.

Repair: change only the submitted workflow input from `agentic` to `evals` for GPQA-only runs. Keep C4 ten on TTI82da396ef63bbe95cedbe254bb5433d905cd0491 and C2 twenty on TTI82569bf17497585555a5710929a8e2f78ea47f1b, together with their already selected immutable tt-metal SHA, all other dispatch flags, frozen cohorts, limits, concurrency, pins and sampling. No source commit is needed. Coordinator owns corrected dispatch and preserves failed attempt IDs42232104462 and42233126698 with their separate device/queue costs.

## Confirmation

Bundled AutoDebug completed successfully in a fresh personal CODEX_HOME session, gpt-6-astra/medium, explicit previously authorized child sandbox skip; report is AUTODEBUG.md here. API-key environment variables were removed. Report verified against source selectors, command construction and failure log lines1704–1711. Root reports untouched.

`check_route.py` run with each candidate's host test interpreter from its worktree:

- C4: `.pre-commit/bin/python <effort>/check_route.py 10 4`
- C2: `.pre-commit/bin/python <effort>/check_route.py 20 2`

Both pass, recorded in c4-10-check.json and c2-20-check.json. Actual agentic selector returns only zero-sized SWE/Terminal; actual standard selector returns GPQA; built local-chat-completions command contains exact frozen IDs, intended concurrency, timeout14400 and HF revision. Host-only command construction, no evaluation or device execution.

Focused existing tests on C4 source: `.pre-commit/bin/python -m pytest tests/test_module/llm_tests/test_agentic_eval_tests.py tests/test_module/llm_tests/test_llm_eval_tests.py tests/workflows/test_gemma4_autoport_smoke_suites.py -q` —198 passed, one existing TestStatus collection warning,1.02s (focused-tests.log).

Change status: dispatch correction accepted as the minimum host-confirmed repair; actual corrected submission remains coordinator-owned. Problem status: routing cause solved; live eval validity remains unverified until corrected runs finish. No score, trajectory, gated dataset access, provisioning or runtime fit claim follows from these checks. No source/runtime/cohort edits, hardware commands, push or dispatch performed. Effort device hours0; saving0; validity unblocks GPQA. Failed-run device costs are outside this host repair and must remain in coordinator queue accounting.
