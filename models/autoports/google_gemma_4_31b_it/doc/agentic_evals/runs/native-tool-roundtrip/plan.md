# Native tool-result roundtrip

Goal: verify second-turn API acceptance and native action parsing after a mini-swe-agent2.2.8 tool result. Coordinator owns the device; planned_clients1, batch1, max_num_seqs1. Estimate10seconds; device-hour box0.05.

request.json extends the existing generic SWE probe with its actual candidate assistant tool call, then a synthetic harmless output protocol_ok (returncode0, exception_info null). The unchanged upstream format_toolcall_observation_messages renders this through config/mini.yaml model.observation_template; LitellmModel._prepare_messages_for_api strips internal extra fields. No generated command is executed.

Source: bringup/artifacts/parser-protocol-mini-swe at58ad92fc1385be3f8f2326b7a42320bb3655e169, src/minisweagent/models/utils/actions_toolcall.py:68+, models/litellm_model.py:75-78,123-134, config/mini.yaml:112+. Previous actual response:../parser-protocol/swe-candidate-response.json.

The generation settings remain temperature1, top_p0.95, top_k20, max_tokens32768, enable_thinkingtrue. POST /v1/chat/completions once; save status/body. Confirm HTTP200 and parse returned message.tool_calls with the actual mini parser using ../parser-protocol/audit_responses.py --swe response.json. Do not execute returned actions. This establishes protocol acceptance only, not a score or quality verdict; no HF control is invented.
