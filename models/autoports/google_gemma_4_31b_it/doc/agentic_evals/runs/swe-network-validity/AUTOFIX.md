# SWE task environment activation repair

The failing contract is that installed mini-swe-agent commands use the official task image's prepared `testbed` environment. The two quarantined trajectories instead used base Python under `/opt/miniconda3/lib/python3.11` and downloaded dependencies. This establishes an activation mismatch; it does not establish that activation alone prevents every future model download.

## Supported repair

Isolated TTI branch `fix/swe-agent-environment-activation`, base `e7643650635b2ab4e5f9662de26b1ae04433f031`. In `llm_module/agentic/harbor.py::_get_agent_env`, default agent-scoped `BASH_ENV=/root/.bashrc` for `mini-swe-agent` with `swebench-verified`, Docker, and no custom agent import path. Preserve explicit overrides and all other routes. No PATH value is constructed or copied. No task, sandbox, network policy, selector, model implementation, Harbor source, or mini source change.

Rejected alternatives: mini LocalEnvironment has no configurable executable/prefix and uses subprocess shell=True; BASH_ENV set only inside mini therefore cannot activate its `/bin/sh`. Static PATH replacement would discard task state. External Harbor launcher changes are unnecessary because supported agent environment configuration reaches the existing Bash launcher.

## Source chain

Pinned Harbor e76c762fca5267dfb36e1f900f0a7724ed72e735: `Trial._run_agent_phase` uses `scoped_exec_env(self.agent.extra_env)`; Docker `_service_exec` merges this environment and Unix `exec_shell_args` returns `bash -c`. Agent setup is also scoped. Mini launcher runs under this Bash and retains inherited PATH while adding its tool bin directory. mini2.2.8 LocalEnvironment subsequently merges os.environ with task-neutral config.env and executes child shells.

Official image registry config ENV places base miniconda first. Both image startup files were extracted read-only from these layer blobs:

- Django: sha256:57d8c8824f066b69f765fab96e977938bf8e785700abf524bcc68ca7b1740e6a
- Pylint: sha256:2c8aada2abc2881b787ad9f651fe144c3c85b2805d5ae988c83b96d6c3c01ece

Each exact file is `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed` followed by newline, with no noninteractive early return. Files and registry manifest/config provenance are adjacent. Harbor's mini installer appends a user-local PATH export; it does not replace activation. Current registry digests are not claimed to prove the historical CI-pulled digest. No task solution/test files were inspected or altered.

## Confirmation

Seven focused regression tests pass: emitted agent-only env, no PATH replacement, explicit override preservation, four unaffected routes, and real Bash startup/Python subprocess/shell child inheritance using exact startup contents and substituted image conda implementation. This confirms Bash semantics and TTI serialization; it is not a real-image conda/package test. Full hook outcome recorded with final patch below by coordinator handoff.

No hardware/device hours. Frozen smoke cohort, clients, release caps, and pins remain unchanged. Actual installed package inventory, inference behavior, score validity, and absence of downloads require the next normal same-two-task smoke. The original two results remain quarantined.

## Final host result

Local repair commit `91e38023e0561504443361df8c8a21ed88c58b16` in `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-swe-environment-activation`; patch `swe-agent-environment-activation.patch` adjacent. Focused tests: 7 passed. Full hooks: ruff, ruff-format, pytest, copyright all passed (`bringup/artifacts/stage12-swe-activation-hooks-final.log`). Initial full run had one relevant exact-dict expectation failure with 3771 passing and 15 skipped; updating that existing endpoint test to include BASH_ENV resolved it. No source changes beyond the ten-line default and tests. `git diff --check` passed. Hooks were completed before the local commit; commit bypassed duplicate hook invocation only. No push or active publication edits.

Apply source with `git cherry-pick 91e38023e0561504443361df8c8a21ed88c58b16` from the shared TTI repository, or `git apply swe-agent-environment-activation.patch` against base e764365. Before normal smoke rerun, coordinator selects the original frozen SWE2 configuration; the repair commit itself leaves the current GPQA configuration untouched. Actual and predicted device cost for this effort are zero; no speed credit is claimed.
