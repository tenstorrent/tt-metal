# AutoDebug: malformed mini-swe response retention

Investigated 2026-11-06. Inspection only; no source changes, installations, agent/model launches, hardware commands, credential access, or publication. The AutoDebug launcher was not invoked again. Two inspection subagents independently checked the retention boundary and dependency selection; their conclusions were checked against source.

## Finding

**Confirmed: mini-swe-agent 2.2.8 discards the completed LiteLLM response when action parsing raises `FormatError`.** The parser supplies feedback, and the agent saves that feedback, but the full response never reaches serializable agent state. This is an upstream mini-swe observability defect, independent of the model producing a malformed action. It explains absent raw response metadata; it does **not** explain why a response is malformed or prove that the currently running long CI decode is a loop.

The smallest behavior-preserving repair is in upstream `LitellmModel.query`: catch `FormatError` specifically around action parsing, retain the response dump under the existing feedback message's `extra`, and re-raise the same exception. A dedicated key such as `extra.failed_response` preserves existing Harbor usage accounting as well as request/action behavior. No custom agent, monkeypatch, or model-specific parser workaround is needed.

**Consumption requires an external dependency change.** The inspected Harbor supports installing a published mini-swe package version through `agent_kwargs.version`. It does not support choosing an unreleased mini-swe Git SHA in its installer. A fixed mini-swe release plus a TTI version repin is the smallest supported route. An installer-managed unreleased-SHA route would require a generic Harbor installer/options change and a new Harbor pin. Neither change was made here.

## Evidence boundaries and revisions

The prepared source checkouts were clean when inspected:

| Name used below | Checkout | HEAD |
| --- | --- | --- |
| M | `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe` | `58ad92fc1385be3f8f2326b7a42320bb3655e169`, tag `v2.2.8` |
| H | `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor` | `e76c762fca5267dfb36e1f900f0a7724ed72e735` |
| T | `/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12` | `f12b2682cb03e84555ba122dd647262ae399815f` |

The startup run contract and goal-12 inputs were read. Their original release-spec anchor is distinct from this prepared TTI HEAD. Findings about dependency selection below apply to these prepared revisions, not an assumed older Harbor install template or the entire live CI environment.

Direct source observations are separated below from a historical trajectory observation and proposed changes. No missing response has been reconstructed, and no current CI request has been paired with a token trace.

## Exact error and serialization path

1. `DefaultAgent.step()` evaluates `self.query()` before `execute_actions`. `DefaultAgent.query()` checks limits, increments `n_calls`, and calls `model.query(self.messages)`. The instance-cost update and assistant-message append occur only after that call returns. See [M default.py:99](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/agents/default.py:99), lines 99–122.

2. `LitellmModel.query()` finishes the `_query` retry loop, calculates cost, updates global model statistics, and dumps the selected assistant message at line 86. It then builds `message["extra"]`. The first dictionary value calls `_parse_actions(response)` at line 88; the whole-response `response.model_dump()` is the next value at line 89. If parsing raises, that second dump is never evaluated, the dictionary assignment never completes, and the selected-message dump remains only a local variable. See [M litellm_model.py:80](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models/litellm_model.py:80), lines 80–93.

3. The selected tool parser raises for no calls, invalid JSON arguments, an unknown tool, a non-object decoded argument payload, or an object missing the `command` key. It does not validate the type of the `command` value. Its payload is one `role="user"` feedback message with rendered error content and `extra.interrupt_type="FormatError"`. It contains no raw response. Even a valid first tool call followed by an invalid call causes the entire parse to raise before any actions reach execution. See [M actions_toolcall.py:30](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models/utils/actions_toolcall.py:30), lines 30–66.

4. `FormatError` subclasses `InterruptAgentFlow`; the latter stores only the supplied `messages` as its explicit payload. It does not carry `response`. See [M exceptions.py:1](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/exceptions.py:1), lines 1–22.

5. Unwinding skips `DefaultAgent.query()`'s cost update/assistant append and skips `execute_actions()`. `DefaultAgent.run()` catches `InterruptAgentFlow`, appends `e.messages`, and calls `save()` in `finally`. The last message is a user message, so the normal next iteration begins. This is agent feedback followed by a new query, **not** a parser retry inside LiteLLM's retry loop. See [M default.py:77](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/agents/default.py:77), lines 77–97; the retry boundary is at `litellm_model.py:81–83`.

6. `DefaultAgent.serialize()` includes `self.messages`, counters, exit/submission fields, and merged model/environment serialization. `save()` JSON-serializes this data. `LitellmModel.serialize()` contributes configuration only; it has no last-response store. Neither exception locals nor traceback locals are serialized. Thus moving the dump earlier without attaching it to durable state cannot fix retention. See [M default.py:124](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/agents/default.py:124), lines 124–154, and `litellm_model.py:139–147`.

The generic text-based subclass inherits the same `query()`. Its regex parser already saves the **content string** as `extra.model_response`, alongside `n_actions`, on zero/multiple actions. It still loses the full response and token metadata. Therefore “all malformed text is lost” would be too broad. See [M litellm_textbased_model.py:16](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models/litellm_textbased_model.py:16) and [M actions_text.py:15](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models/utils/actions_text.py:15), lines 15–33.

Harbor actually launches the `mini-swe-agent` CLI with `--yolo`, `--exit-immediately`, an output path, and `-c mini` plus overrides. The CLI defaults to `InteractiveAgent`, which inherits this run/save path. In yolo mode its query delegates to `DefaultAgent.query`; it does not catch `FormatError`. Thus the DefaultAgent diagnosis applies to this route without falsely claiming Harbor directly instantiates DefaultAgent. See [H mini_swe_agent.py:999](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/mini_swe_agent.py:999), lines 999–1016; [M run/mini.py:99](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/run/mini.py:99); [M interactive.py:57](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/agents/interactive.py:57), lines 57–87.

## Historical artifact corroboration, with limits

The collected [Django native trajectory](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/smoke2-swe-native-parser/collected/workflow_logs_agentic_gemma-4-31B-it_bh-qb-ge_gemma4-31b-autoport-v3/reports_output/agentic/google__gemma-4-31B-it_p300x2_agentic/eval_google__gemma-4-31B-it/agentic/swe_bench_verified_20261106T073358/django__django-11265__ohWF75G/agent/mini-swe-agent.trajectory.json) declares mini version `2.2.8`, `InteractiveAgent`, and `LitellmModel`. Its 126 model calls correspond to 115 saved assistant messages and 11 saved FormatError messages. Each error message has exactly `role`, `content`, and `extra`, with only `interrupt_type` inside `extra`. The first successful assistant message has `actions`, `cost`, `response`, and `timestamp` in `extra`.

The zero-based FormatError message indices are 170, 179, 188, 209, 212, 221, 226, 231, 234, 237, and 240. These are actual serialized indices, not inferred timestamps or token-response pairings.

That older trajectory's model kwargs do **not** request `return_token_ids`; its first successful response has `prompt_token_ids=null` and no top-level choice token-ID field. It corroborates the response-retention asymmetry, not token-ID delivery or the current CI execution. The prepared TTI configuration now does request `extra_body.return_token_ids=True` at [T eval_config.py:11990](/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/reference_config/evals/eval_config.py:11990), lines 11990–12015. Those two configurations must not be conflated.

## Minimal upstream repair contract

Change the generic mini-swe LiteLLM chat model, retaining its existing parser and agent loop:

1. Keep `_query`, retry scope, API kwargs, cost calculation, global-stat update, and the successful-response schema unchanged.
2. Isolate `_parse_actions(response)` in a `try` that catches only `FormatError`.
3. On that exception, export `response.model_dump()` into the existing feedback dictionary under `extra.failed_response`, retaining every original feedback field. Both built-in parsers on this base-query path provide exactly one feedback message. Attach the dump once, without adding a conversation message or copying partial actions into executable state.
4. Re-raise the same exception. Guard diagnostic export and validate the candidate dump with the same JSON serialization used by `save()` **before attaching it**. If export or JSON validation fails, leave the feedback unchanged and re-raise the original FormatError. This prevents diagnostic data from making the later `finally: save()` fail. Retention is necessarily conditional on a serializable dump; the check must not transform its values.
5. On success, use the parsed actions in the existing `extra` dictionary and return the same message shape. Keep changes to broader cost accounting or other provider models separate.

This is a proposed change, not an applied patch. The retention location would be `messages[i].extra.failed_response` in `mini-swe-agent.trajectory.json`.

| Invariant | Why this intervention preserves it |
| --- | --- |
| Request contents and order | `_prepare_messages_for_api` removes the entire `extra` field before Anthropic thinking-block and cache-control processing. The original feedback role/content/count/order remains identical. See [M litellm_model.py:63](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models/litellm_model.py:63), lines 63–78. |
| API kwargs and tools | Keep `_query` and `self.config.model_kwargs \| kwargs` unchanged; retain `tools=[BASH_TOOL]`. No additional request is made to recover a response. |
| Feedback and retries | Bare re-raise preserves the same FormatError payload and run-loop catch. Parsing remains outside the provider retry loop. |
| Actions and submission | The exception still prevents action execution; valid actions follow the same execution path. No malformed assistant is inserted and no partial action list is returned. Exit/submission logic remains unchanged. |
| Counters and limits | Keep `n_calls`, global stats, instance cost, and limit tests as they are. Adding diagnostics must not also repair cost-limit behavior. |
| Task scoring | Harbor's verifier executes the task test and reads reward files independently of this metadata; TTI's existing result handling remains unchanged. See [H verifier.py:255](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/verifier/verifier.py:255), lines 255–310. This preserves scoring logic, not a promise of identical stochastic outcomes or wall-clock timing. |

Why recommend `failed_response` instead of the existing success key `response`? Harbor's `_message_usage()` reads `extra.response.usage` on **every message role**, and both its context totals and ATIF totals consume that helper. Putting the snapshot at `extra.response` is request-preserving and would recover missing usage, but also changes reported usage totals and completion-weighted cost allocation. A dedicated diagnostic key avoids that unrelated behavior change. See [H mini_swe_agent.py:100](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/mini_swe_agent.py:100), lines 100–125, 152–180, 313–336, and 818–834.

The native trajectory must be retained. Harbor loads it, totals usage, and writes a separate ATIF trajectory. Its user-feedback conversion does not copy arbitrary extras; it attaches feedback as an observation to the last agent step when one exists. A first-turn format error may have no such agent step. Adding diagnostic metadata to native feedback does **not** automatically create an ATIF raw-response record. See [H mini_swe_agent.py:349](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/mini_swe_agent.py:349), lines 349–363, and 802–846. Trial output synchronization downloads agent logs before post-run context extraction at [H trial.py:1160](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/trial/trial.py:1160), lines 1160–1173.

Token-ID scope is precise: `model_dump()` retains the fields present in the LiteLLM response object. It cannot recreate IDs the server omitted or the SDK discarded, and it is not retention of raw HTTP bytes. The fix should preserve the full object dump without reconstructing IDs from text. The installed LiteLLM version and its extension-field behavior need explicit fixture evidence before claiming end-to-end ID retention: Harbor installs `litellm[mcp]` without an exact version, and mini's `pyproject.toml:38` permits a dependency range.

## Supported dependency route

The current route is concrete:

1. TTI defines `MINI_SWE_AGENT_VERSION="2.2.8"` at [T eval_config.py:356](/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/reference_config/evals/eval_config.py:356), and the Gemma SWE row puts it under `agent_kwargs.version` at lines 11997–11999.
2. `_get_agent_kwargs()` deep-copies those kwargs and only adds model defaults. JSON configuration writes them as `agents[0].kwargs`; the CLI route forwards them with `--agent-kwarg`. See [T harbor.py:183](/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/llm_module/agentic/harbor.py:183), lines 183–188, 344–352, 572–574.
3. Harbor's factory forwards kwargs to the built-in agent. `MiniSweAgent.__init__` forwards them to its base, which takes `options.version` into `_version`. See [H factory.py:315](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/factory.py:315), also lines 32–46; `mini_swe_agent.py:704–713`; [H installed/base.py:717](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/base.py:717).
4. The inline installer constructs `version_spec = f"=={self._version}"` and installs `mini-swe-agent{version_spec}` with Python 3.12. See [H mini_swe_agent.py:755](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/mini_swe_agent.py:755), lines 755–789. There is no mini install template at this revision.

A published fixed mini version can therefore be consumed through this existing route, after an explicit release/version repin. This investigation did not establish that such a release already exists. A TTI-only edit cannot repair the currently pinned `2.2.8` implementation.

For an unreleased mini commit, the present route has these limits:

- `MiniSweAgentOptions` and its installed-agent base declare `version` and `skip_install`, but no `git_ref`, repository, or package-source selector. Unknown agent kwargs are rejected (`extra="forbid"`), not silently accepted. See [H options.py:50](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/options.py:50), lines 50–90; `mini_swe_agent.py:487–505,644`; [H agents/base.py:283](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/base.py:283), lines 283–305.
- A SHA or Git URL in `version` becomes `mini-swe-agent==<value>`, not a VCS requirement. A top-level Harbor agent `version` is not the declared selector; it belongs in `kwargs`. Native `kwargs.config` is runtime YAML, written after installation, and cannot select the package source.
- `skip_install` only uses an agent already provided by the task image; it neither fetches a SHA nor validates that source. Its version mismatch path warns and records the detected version. See [H installed/base.py:1179](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-harbor/src/harbor/agents/installed/base.py:1179), lines 1179–1201. A new task image is a separate artifact change, not an existing Git selector.
- Installing mini into TTI's host venv is ineffective: Harbor installs it inside the task container. This separation is explicit in [T evals-agentic.txt:13](/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/requirements/evals-agentic.txt:13).

If an installer-managed unreleased SHA is required, a generic external Harbor options/installer change would have to introduce a validated immutable mini-source selection. The separately built task-image route with `skip_install` remains a distinct external artifact alternative. TTI **can** select a new Harbor SHA: `metadata.harbor_git_ref` requires 40 hexadecimal characters, is checked out from the fixed `tenstorrent/harbor` repository, and is verified against HEAD before editable installation. See [T workflow_venvs.py:335](/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/workflows/workflow_venvs.py:335), lines 335–385, repository at lines 46–47. The current Gemma model pin is `e76c762...` at `workflows/model_specs/dev/llm.yaml:7654`. This is a route to consume a future Harbor change, not evidence that the current pin supports mini Git sources.

## Verification and regression requirements

Performed here: read-only source/test inspection; Git revision/status checks; JSON structural inspection of existing trajectories; AST checks confirming dictionary evaluation order, absence of a FormatError handler in `query`, parsing outside retry, stripping of `extra`, saving in the run-loop `finally`, and the inline package-version installer. The dependency subagent also checked rendered requirement strings with `packaging.Requirement`: `mini-swe-agent==2.2.8` was valid, while SHA/Git-URL strings placed after `==` were invalid. These are static checks, not a runtime reproduction or a verified patch.

The existing [M test_litellm_model.py:55](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/parser-protocol-mini-swe/tests/models/test_litellm_model.py:55) only asserts that missing tool calls raise. Its fake whole-response dump is empty. It cannot detect retention loss or prove token-ID fidelity.

A future upstream patch needs paired host-only fixtures that:

- Cover no tool call, invalid JSON, unknown tool, non-object decoded arguments, a missing `command` key, valid-plus-invalid tool lists, and zero/multiple regex actions in the inherited text model.
- Preserve an exact response dump containing synthetic prompt/completion token-ID arrays, usage, finish reason, reasoning content, and unknown extension fields through the existing agent save/JSON round trip. Use the actual selected LiteLLM response type as well as a controlled response fixture to test SDK preservation separately.
- Compare every prepared request and API kwargs across malformed → valid → submitted sequences before/after; check request count, feedback role/content/order, commands executed, successful action schema, exit/submission, and limit behavior.
- Confirm no malformed or partially parsed action executes, no provider retry is added, and original FormatError feedback survives diagnostic-export or JSON-validation failure.
- Confirm existing counters/cost behavior and Harbor task reward/usage behavior remain unchanged with the dedicated diagnostic key. Verify that the native file retains diagnostics even though ATIF does not copy them.

No tests requiring model service, Docker, or hardware are needed to establish the source defect. Those environments would be needed only for a later real deployment/response-delivery check.

## Other Potential Issues and rejected explanations

| Candidate | Adjudication |
| --- | --- |
| Only move `response.model_dump()` before parsing | Insufficient: the dumped object remains local when the exception unwinds. The sibling `LitellmResponseModel.query` already dumps before parsing (`litellm_response_model.py:50–62`) and still has this escape-path problem. |
| Append the malformed assistant to `FormatError.messages` | Violates the requested intervention boundary: DefaultAgent appends it to conversation history, and API preparation strips `extra`, not entire assistant messages. Invalid tool calls/text would enter the next request. |
| Repair cost accounting at the same time | Malformed calls increment `n_calls` and global statistics but skip instance cost. This discrepancy is real at the cited source lines, but correcting it can change cost-limit termination and future requests. Keep it separate. |
| Cost/global-limit failures also lose response metadata | They can raise before parsing/dumping (`litellm_model.py:84–85`); they are not the demonstrated FormatError path and are outside this minimal repair. |
| Responses API and other providers need the same repair | The Responses API overrides `query`, so changing the base chat method does not fix it. It is a separate coverage opportunity; the selected route and historical artifact use `LitellmModel`. |
| Retention explains current slow decode/repetition | Unproven. This loss occurs after a completion returns. An in-flight long decode has not yet reached this parser boundary. Missing metadata or duration alone cannot establish repeated malformed completions, repeated token IDs, a model loop, or device failure. |

After drafting, the headline was rechecked against the parser, exception payload, run/save path, request preparation, Harbor conversion, and installer. The supported conclusion is limited to completed malformed-response retention and the external dependency boundary. The cause of malformed generation, exact current CI LiteLLM token-ID preservation, and any current decode loop remain unresolved.
