# AutoDebug: SWE nightly network reach

The evidence supports **an agent-selected probe of the issue's reproduction URL**, not a new generic harness or activation defect. Quarantine `pydata__xarray-4966__58JAeKK` from run `42210467509`: its first action issued `curl -I` to the external dataset URL and received HTTP headers. The recorded verifier reward of `1.0` is not a valid scored success under the stage's strict network-reach rule.

No source change is supported by this diagnosis. Recommend a measured stop for SWE advancement at this nightly rung, retaining the previously banked `functional_at: 2`. This report does not change coordinator records or operate a running job.

## Scope and revision consistency

Inspection performed on 2026-11-06. Only local artifacts and source were read; cheap checks parsed JSON/YAML and compared strings. No target execution, network requests, device commands, additional agents, custom prompts, task replacement, network-policy changes, or source edits were performed. No verifier solution patches were inspected. Raw reward/timing metadata came from trial `result.json`, not verifier implementation or solution contents.

Evidence is under `../nightly-swe-observable/`. The relevant collected job is `swe_bench_verified_20261106T124213`, trial `pydata__xarray-4966__58JAeKK`.

| Component | Evidence-consistent revision |
|---|---|
| TTI used by the run | `4b94cf3ba86d30a6816149872e59cc34438748ae`, from the dispatch manifest |
| TTI prepared checkout | `b4662344c0119e2c1fe7840e9ca954324a2f0afe` |
| Harbor | `e76c762fca5267dfb36e1f900f0a7724ed72e735`, version `0.25.0`; collected job lock and runtime model spec agree with the prepared source HEAD |
| mini-swe-agent | `2.2.8`; prepared source HEAD `58ad92fc1385be3f8f2326b7a42320bb3655e169`, exact tag `v2.2.8`; native trajectory and startup log both report `2.2.8` |
| Task dataset | `harbor-datasets@86723674f04e4209ac479d0fb75d9d9f44b4377e`, from trial config |
| TT model | `439c22b0e991e5af626efafa7224edf28f84795b`, from dispatch manifest and runtime spec |

The TTI checkout is newer than the failing run. A local Git comparison shows **no changes** between those revisions in `llm_module/agentic/harbor.py` or `tests/llm_module/test_harbor_swe_activation.py`. Later eval-config changes add native token retention and change suite selection; they are not attributed to this run. Actual collected job/trial configs and the native trajectory determine the settings below.

## Direct observations

All native message indices below are **zero-based**. The corresponding first agent step in normalized ATIF is `step_id: 3`.

1. Native message 0 is the stock system prompt. Message 1 contains the issue description, including the exact dataset URL used by the subsequent command and examples of opening it with `netcdf4` and `pydap`. The URL was already available to the model before any action.
2. Message 2, the first assistant action, says: “I will start by checking if I have internet access to download the test file mentioned in the issue.” Its structured `bash` call is:

   ```text
   curl -I https://observations.ipsl.fr/thredds/dodsC/EUREC4A/PRODUCTS/testdata/netcdf_testfiles/test_NC_BYTE_neg.nc
   ```

   Its timestamp is `2026-11-06T12:42:52.856637Z`. The retained API response's `choices[0].message.tool_calls`, native `tool_calls`, and parsed `extra.actions` all contain that same command. This is not a command invented by trajectory normalization.
3. Message 3 has the matching `tool_call_id`, `chatcmpl-tool-95f77769381ea017`, timestamp `2026-11-06T12:42:54.226918Z`, `returncode: 0`, and empty `exception_info`. Both rendered content and `extra.raw_output` contain:

   ```text
   HTTP/1.1 400 400
   Date: Fri, 06 Nov 2026 12:42:54 GMT
   Server: Microsoft-IIS/10.0
   Content-Description: dods-error
   Content-Type: text/plain; charset=UTF-8
   Content-Length: 65
   ```

   This supports completed HTTP network interaction. It does not prove successful dataset retrieval, receipt of a response body, or retrieval of a fix. A HEAD request can receive an HTTP error while curl returns zero without `--fail`; neither the status nor the exit code erases the network reach. The artifact does not identify whether an intermediary or the origin generated the headers.
4. There is **no preceding tool failure**. The next command searches local xarray source. The first local Python reproduction appears at message 16; its message-17 failure is a `Variable` dimension mismatch, after successful NumPy/xarray imports. Message 19 runs the corrected local reproduction and observes the unsigned-byte behavior. Later import-time errors at messages 43 and 49 are `IndentationError` in the agent-edited file, not missing-package errors.
5. The trajectory has 67 messages, 32 API calls, and exit status `Submitted`. Trial metadata records reward `1.0` and no trial exception. These completion facts do not supersede the validity rule.

The native evidence starts at [message 1][native-user], [message 2][native-action], and [message 3][native-observation]. [Trial metadata][trial-result] supplies completion/reward data.

## Effective configuration and causal path

The generated Harbor config, saved job config, and saved trial config all contain `agents[0].env.BASH_ENV` or `agent.env.BASH_ENV` set to `/root/.bashrc`. They select the built-in `mini-swe-agent`, Docker, version `2.2.8`, agent timeout 21,600 seconds, request timeout 3,600 seconds, and output cap 32,768. Inline mini overrides contain only `model` settings: temperature `1.0`, top-p `0.95`, request top-k `20`, thinking enabled, and connection/accounting settings. The request's top-k is the observed eval setting, distinct from the server's operator-authorized default top-k `32`; it is not evidence of a network defect.

The native effective configuration records `InteractiveAgent`, mode `yolo`, `LocalEnvironment`, and `LitellmModel`. A static equality check confirms its `system_template` and `instance_template` exactly equal pinned `config/mini.yaml`. The template recommends creating a reproduction script; it does not specifically direct a network fetch or impose the stage's audit rule. No stage-owned prompt override is present in the saved inline config.

The source chain explains both the command and environment without an additional defect:

| Boundary | Source fact and implication |
|---|---|
| TTI → Harbor config | [`harbor.py:273`][tti-env] defaults `BASH_ENV` for built-in Docker SWE mini runs; lines 344–355 put it in agent `env`. Explicit overrides survive. |
| Agent config → execution scope | Harbor `agents/factory.py:290–329` resolves `config.env` into `extra_env`; [`trial.py:716`][harbor-run] carries it into `scoped_exec_env` around `run`. |
| Scope → container shell | Harbor `environments/base.py:584–625` merges scoped environment last. `docker/docker.py:1446–1521` forwards it through Docker `-e`; `docker_unix.py:307–309` wraps the launch in `bash -c`. This is the outer Bash activation boundary. |
| Harbor → mini | [`mini_swe_agent.py:999`][harbor-mini] prepends stock `-c mini` and launches `mini-swe-agent --yolo --task=... --exit-immediately`. The task is shell-quoted at line 863. Startup logs show `mini` followed by model overrides. |
| mini → model → action | `run/mini.py:70–102` merges configs and runs the task. `agents/default.py:77–84` renders system/user messages. [`litellm_model.py:80`][mini-model] retains the returned response; `models/utils/actions_toolcall.py:30–66` parses its JSON command without synthesizing a replacement. |
| Action → subprocess → observation | `agents/interactive.py:98–112` executes the parsed action; yolo mode skips confirmation. [`environments/local.py:23`][mini-local] runs the command with `env=os.environ \| self.config.env`, waits via `subprocess.run`, and records output/return code. `actions_toolcall.py:69–103` attaches the action's ID to the tool observation. |

Mini's local subprocess uses `shell=True` without explicitly selecting Bash. The repair does **not** depend on every child shell sourcing `BASH_ENV`: the outer Harbor Bash sources the startup file before launching mini, whose child inherits the resulting environment. Also, native `info.config.environment.env` serializes only mini's explicit environment overrides (`local.py:71–78`), not all inherited `os.environ`. The absence of `BASH_ENV` from that small native mapping is therefore not evidence it was dropped.

The existing [activation tests][activation-tests] cover config scoping, explicit overrides, route restrictions, and outer-Bash-to-child inheritance. They were inspected, not rerun. This trial does not include an independent dump of its `.bashrc`, `PATH`, `CONDA_PREFIX`, or interpreter executable, so full activation of every dependency is not independently proven here. Nevertheless, an activation failure cannot explain a **reaction to a dependency error before the first action**: no such observation exists. Successful subsequent local imports further weaken that hypothesis.

## Candidate adjudication

| Candidate | Evidence/preconditions | Finding and smallest implied intervention |
|---|---|---|
| Agent probes the reproduction URL supplied in the issue | Exact URL in the initial user message; first structured action and matching response; local inspection follows | **Supported.** Quarantine the trajectory and record the validity stop. No harness patch follows. |
| Prior activation defect caused dependency-recovery downloads | Would require missing activation and a relevant dependency failure motivating this action | **Unsupported for this event.** Repair is configured and source propagation is intact; curl precedes all tool feedback. No additional activation correction is justified. |
| Harness/parser fabricated curl or mispaired another command's output | Would require disagreement among raw API response, parsed action, or tool ID | **Refuted by available records.** Command and native ID match; ATIF preserves command and observation. No parser correction is justified. |
| Missing stock mini config caused an unintended prompt | Would require absent base config or different effective templates | **Refuted.** Startup explicitly loads `mini`; both native templates equal the pinned stock YAML. |
| HTTP 400 or HEAD means no disqualifying network reach | Would require a successful body/fix download to trigger the rule | **Contradicted by the contract.** The rule includes network reach itself; headers are sufficient evidence. |
| Open egress is a harness defect to repair | Would require a violated harness isolation promise applicable to this run | **Not established.** This stage expressly relies on trajectory audit and forbids a stage-owned network policy. Do not turn an undesirable model action into an unsupported isolation fix. |

“Agent-selected reproduction” describes the observed served-agent behavior. This investigation does not establish that HF would make the same choice, that TT numerical behavior is identical to HF, or why the model preferred this action probabilistically. Those claims need a separate controlled comparison and are not necessary to classify this trajectory.

## Disposition and measured-stop reason

The [agentic-serving Harness contract][network-rule] says a task whose trajectory “fetched its fix or reached the network is quarantined, not scored,” and prohibits stage-owned wrappers, task selectors, and network policy. Its rung gate requires a validly scored current rung before advancing.

Record this trial as **quarantined: external HTTP HEAD request to the issue's reproduction dataset in the first agent action**. Preserve its raw reward separately; do not count it as either a clean success or a normal model-task failure. Do not claim a valid five-task nightly by silently dropping or replacing xarray. This report does not independently audit the other four nightly trajectories.

Recommended coordinator disposition: stop SWE advancement on this evidence and retain `functional_at: 2`, the already recorded valid smoke rung. `estimate.json` and `runs/smoke2-swe-activation-repair/audit.json` record that prior banked rung; those previous audit conclusions were read as context, not re-audited here. The measured stop is a **validity failure with no supported generic remedy identified**, not exhaustion of the 48-hour stage budget, a timeout, or proof that the task cannot be solved locally. The agent itself later constructs a local reproduction.

A new network block, task rewrite/replacement, special prompt, or speculative activation patch would not be a supported fix for the traced cause and falls outside this investigation. No retry-until-clean recommendation is made. Preserve the failed attempt as evidence and retain the current generic activation repair.

## Other potential issues and evidence limits

- The initial user content repeats the issue title/body twice. Stock mini's template contains one `{{task}}` insertion; the observed duplication alone does not locate its origin or prove it caused the network choice. A single copy already supplies the same URL. No prompt/data correction is justified here without examining the exact task instruction provenance; no task instruction was changed.
- The observable run's dispatch manifest still says `in_progress`, while the collected xarray trial is complete. Use per-trial result and native trajectory for this finding; do not infer current remote job status from that stale manifest.
- No packet capture, DNS/TLS trace, or response body is available. The evidence supports network interaction, not the identity of the responder or successful file/fix retrieval.
- The skill's broad network rule can quarantine legitimate reproduction-data probes as well as solution retrieval. That is the stated acceptance rule, not a source-code contradiction or an exception inferred by this investigator.

## Report review and static validation

After drafting, the headline claims were rechecked against the saved artifacts and cited source boundaries. The review distinguishes configured activation from independently observed interpreter identity, actual HTTP interaction from successful retrieval, and a valid previous smoke from an invalid current nightly. No candidate lacking its necessary preconditions remains a headline bug.

Static checks passed for exact API/native/parsed-command equality; native tool-call ID pairing; equality of formatted versus raw tool output; ATIF command/observation correspondence; stock-template equality; and `BASH_ENV`, version, and model-only overrides in all three saved configs. No target code was imported or executed by these checks.

Artifact SHA-256 values:

| Trial artifact | SHA-256 |
|---|---|
| `agent/mini-swe-agent.trajectory.json` | `51a490c6b5b1c694b0aee64771ffde3d189cd28fa601c1fc80a66e7d5a5517e6` |
| `agent/trajectory.json` | `43d96932fa338df6db0da8a1ee32908d7b50ca5879c33c3a1232988841717068` |
| `config.json` | `3b000e60020020763ee218fbef8b43c9c7a7163421c33da81badb82a7642c828` |
| `result.json` | `599cdb6d28d5a27a425e46babb0239e9ad698eb06a4c16d0a4d1dc723b943e50` |

[native-user]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-swe-observable/collected/workflow_logs_agentic_gemma-4-31B-it_bh-qb-ge_gemma4-31b-autoport-v3/reports_output/agentic/google__gemma-4-31B-it_p300x2_agentic/eval_google__gemma-4-31B-it/agentic/swe_bench_verified_20261106T124213/pydata__xarray-4966__58JAeKK/agent/mini-swe-agent.trajectory.json:69
[native-action]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-swe-observable/collected/workflow_logs_agentic_gemma-4-31B-it_bh-qb-ge_gemma4-31b-autoport-v3/reports_output/agentic/google__gemma-4-31B-it_p300x2_agentic/eval_google__gemma-4-31B-it/agentic/swe_bench_verified_20261106T124213/pydata__xarray-4966__58JAeKK/agent/mini-swe-agent.trajectory.json:73
[native-observation]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-swe-observable/collected/workflow_logs_agentic_gemma-4-31B-it_bh-qb-ge_gemma4-31b-autoport-v3/reports_output/agentic/google__gemma-4-31B-it_p300x2_agentic/eval_google__gemma-4-31B-it/agentic/swe_bench_verified_20261106T124213/pydata__xarray-4966__58JAeKK/agent/mini-swe-agent.trajectory.json:158
[trial-result]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-swe-observable/collected/workflow_logs_agentic_gemma-4-31B-it_bh-qb-ge_gemma4-31b-autoport-v3/reports_output/agentic/google__gemma-4-31B-it_p300x2_agentic/eval_google__gemma-4-31B-it/agentic/swe_bench_verified_20261106T124213/pydata__xarray-4966__58JAeKK/result.json
[tti-env]: /home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/llm_module/agentic/harbor.py:273
[harbor-run]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/trial/trial.py:716
[harbor-mini]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/mini_swe_agent.py:999
[mini-model]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models/litellm_model.py:80
[mini-local]: /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/environments/local.py:23
[activation-tests]: /home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/tests/llm_module/test_harbor_swe_activation.py:37
[network-rule]: /home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/agentic-serving/SKILL.md:175
