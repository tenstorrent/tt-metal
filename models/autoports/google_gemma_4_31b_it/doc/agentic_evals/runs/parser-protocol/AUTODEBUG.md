# Missing native response parsers

Failing contract: stock Harbor mini-swe-agent 2.2.8 must receive structured OpenAI `tool_calls`; thought-enabled terminal replies must expose final JSON in content separately from native reasoning.

The new autoport's resolved `vllm_args` has neither `tool_call_parser`, `enable_auto_tool_choice`, nor `reasoning_parser`. Metadata parser-name fields are not consumed by the inspected current TTI Python server launcher.

Causal chain:
- TTI `reference_config/evals/eval_config.py` selects SWE Harbor `mini-swe-agent`, version 2.2.8, not DeepSWE or the textbased model class.
- Upstream v2.2.8 `src/minisweagent/models/litellm_model.py:65-69` sends `tools=[BASH_TOOL]` through chat completions. Lines117-118 consume `message.tool_calls`. `models/utils/actions_toolcall.py:32-42` raises FormatError when absent.
- Stock vLLM0.31 `entrypoints/openai/chat_completion/protocol.py:898-900` defaults tools to tool_choice=auto. `serving.py:1025-1029` creates a content-only message when auto tools or its parser are missing. No amount of valid model call text can populate tool_calls on that branch.
- The pinned HF chat_template.jinja supports native `<|tool_call>` function syntax and `<|channel>thought` channels. vLLM0.31 `parser/gemma4.py:410+` supplies exactly these grammars; reasoning and tool adapters are registered as gemma4. This evidence is HF/generic vLLM, not old Gemma TT implementation settings.
- Terminal-Bench terminus-2 uses JSON in ordinary message.content, with required analysis/plan/commands; it does not pass native tools. Harbor `terminus_json_plain_parser.py` tolerates extra surrounding text, so absent reasoning separation is a representativeness risk, not proof every task fails.
- GPQA uses chat-completion text and thinking=true without tools. Absence of a tool parser does not block its API protocol. Reasoning parser correctly separates native thought from final answer; no quality claim is inferred.

Minimal correction: model-owned TTI vllm_args `tool_call_parser: gemma4`, `enable_auto_tool_choice: true`, `reasoning_parser: gemma4`. Keep stock plugin/runtime and harness agent protocols. A metadata-only correction would not fix actual emitted argv.
