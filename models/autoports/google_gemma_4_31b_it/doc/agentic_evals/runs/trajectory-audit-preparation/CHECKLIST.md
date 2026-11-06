# First real smoke trajectory audit

Purpose: shorten feedback after collection. Host-only, zero device hours; this is an inspection checklist, not a replacement runner, selector, network policy or score gate. Apply to actual collected files once, then record that run's audit beside its results.

## 1. Identify what actually ran

Read run manifest, CI immutable SHAs, built image digest, emitted runtime_model_specs JSON and server startup/autoport_registration event. Require the fresh adapter module, full60layers, pinned HF/tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475,262144context, actual batch/max_num_seqs and planned_clients. Compare task names to the frozen cohorts, not only summary counts. Capture resolved package/source versions: TTI workflows/workflow_venvs.py selects Harbor ref tt-inference-server, which moves; collected checkout SHA takes precedence over the locally inspected Harbor e76c762fca5267dfb36e1f900f0a7724ed72e735. mini-swe-agent is2.2.8 (58ad92fc1385be3f8f2326b7a42320bb3655e169). Terminal agent is terminus-2; record dataset version/checksum from trial config/task_id.

## 2. Find raw evidence before reading aggregate accuracy

TTI llm_module/agentic/harbor.py:571 writes under config.jobs_dir/task_name, with top-level result.json. Locate each trial's config.json, result.json, agent/trajectory.json, raw agent logs and verifier output. Harbor models/trial/paths.py:289 returns **result.json** despite a stale directory-docstring saying results.json. mini_swe_agent.py:794-799 also saves agent/mini-swe-agent.trajectory.json; compare it with converted ATIF when timing or protocol fields are missing. Terminus may produce continuation/summarization trajectories: include all segments and avoid double-counting copied context.

Actual TrialResult fields (Harbor models/trial/result.py): task_name, trial_name, task_id, task_checksum, config, agent_info(name/version/model_info), agent_result, verifier_result, exception_info, started_at, finished_at, agent_execution, verifier and step_results. ATIF models/trajectories/step.py exposes timestamp, source, message, reasoning_content, tool_calls, observation, metrics, is_copied_context and llm_call_count. Missing fields remain unknown, not zero.

## 3. Validate protocol, limits and task resources

SWE mini2.2.8 requires native bash tool_calls with JSON command; inspect real actions and tool-result roundtrips, not only HTTP200. Terminal JSON parser requires analysis/plan/commands; warnings about fenced content alone do not invalidate a parsed trajectory. GPQA is chat/text lm-eval with thinking enabled, not a Harbor tool agent.

Compare collected trial config and requests to the frozen release limits in serving_config.json and dispatched TTI eval_config entry. Current smoke intent: SWE/TB agent21600s, model32768output tokens and3600s request timeout; GPQA126976output and14400s request timeout. Verify these in actual artifacts rather than assume they propagated. Task CPU/memory/storage must match each task definition. Distinguish infrastructure exceptions from ordinary task failure. A looping generation that reaches a limit is a scored failure and a ranked effort; a progressing task cut off requires limits correction and rerun. Keep finish reasons and timeouts visible.

## 4. Quarantine contaminated trajectories

Inspect every actual agent tool call and observation for external network access or fetched fixes: curl/wget, requests/urllib, git fetch/pull/clone, package/download commands, browser/API calls, remote URLs, patch/commit retrieval and encoded/subprocess equivalents. Inspect resulting output and task files to distinguish a successful access from a refused attempt. Mere allowlist/no-network configuration is not proof: Harbor applies task-declared network behavior (TTI harbor.py:134+), not a stage-wide sandbox guarantee. A trajectory that reached the network or fetched its solution is quarantined, not included in valid scores. Record the task, step and evidence; keep raw harness aggregates separately and never silently convert quarantine into success/failure. Task setup and verifier traffic are separate from agent traffic. Never change the stock network policy to make an audit pass.

## 5. Reconcile scores and intervals

TTI llm_module/parsers/agentic.py:168 documents n_trials=rewarded trials, n_errors=exceptions, n_total_trials=job trials; a trial can be both rewarded and errored. Reconcile each verifier reward with actual task coverage before using mean/reward/pass_at_1. Report attempted, scored-valid, quarantined, infra-error and loop-failure counts explicitly. GPQA uses lm-eval per-sample exact_match and completed sample IDs. For binary valid scores, compute95% Wilson interval (z1.96) so n2 and boundary scores do not misleadingly have zero uncertainty. State denominator and subset scope; two tasks cannot establish parity with full-suite references. Report published/GPU references and protocol differences; scores are not stage gates.

## 6. Measure run behavior without inventing phases

Use job started_at/finished_at for wall time and trial agent_execution intervals for aggregate agent time. ATIF metrics prompt_tokens/completion_tokens/cached_tokens and raw per-call traces support token totals and cache ratios. Step timestamp gaps include tool/container/scheduling time: do not label them pure generation. Use paired request send/response timestamps where present; absent TTFT/prefill split stays unknown. Include slowest trial and actual simultaneous conversations, all summaries/continuations, limits and retries. Rank errors/loops/idle gaps first as validity issues; update next-run projections from measured aggregate wall and agent time. Do not profile serving.

Source roots: TTI /home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12; inspected Harbor bringup/artifacts/parser-protocol-harbor/src/harbor; mini bringup/artifacts/parser-protocol-mini-swe/src/minisweagent. Confirm fetched CI versions before applying this source map to changed schemas.
