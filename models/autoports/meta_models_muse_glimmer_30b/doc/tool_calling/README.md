# Tool calling for agentic coding

The bring-up shipped without tool calling. The model's card puts it first —
*"purpose-built for autonomous agentic tasks… reliable tool use"*, with **Reliable
Tool Use**, **Failure Recovery** and **Scaffold Compatibility** among its eight
capability bullets and six benchmark rows (MCP Atlas 75.5, τ3-Banking 23.5,
Gaia2 43.3, WildClawBench 47.6, SWE-Bench Pro 51.2, TerminalBench 51.7) that all
depend on it — but no stage produced a tool parser, and the release server was
never launched with tool calling enabled.

Without a parser the failure is silent rather than loud: the model emits a correct
`<atem:function_calls>` block, the server never extracts it, and the client sees
`finish_reason=stop` with markup sitting in `content`. Every agentic scaffold
breaks and nothing looks broken.

## The grammar

Quoted from the checkpoint's own chat template, which is what instructs the model:

```
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
...
</atem:invoke>
</atem:function_calls>
```

Three template statements drive the implementation, all quoted rather than inferred:

- *"The output is not expected to be valid XML and is parsed with regular
  expressions."* — so the parser is regex-based deliberately; an XML parser would
  reject output the model is licensed to produce.
- *"String and scalar parameters should be specified as is, while lists and
  objects should use JSON format."* — parameters arrive as name/value text, so the
  arguments object is reassembled and each value decoded individually.
- *"Note that spaces for string values are not stripped."* — values are preserved
  verbatim, including a trailing newline when the model closes the tag on its own
  line.

No stock vLLM parser reads this grammar. It is not the `<tool_call>` shape most
expect, and not Laguna's `poolside_v1` shape either.

## Serve with tool calling on

```bash
MESH_DEVICE=P300x2 python -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B \
  --block_size 64 --max_num_seqs 32 --max_model_len 131072 --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}}'
```

`--tool-call-parser muse_glimmer` resolves because the `vllm_ext` bundle registers
it under that name; install the bundle first:

```bash
pip install -e models/autoports/meta_models_muse_glimmer_30b/vllm_ext
```

The same bundle also registers the architecture, so no `vllm-tt-plugin` patch is
needed. Registration lands early enough because
`load_general_plugins()` runs in `EngineArgs.__post_init__`
(`arg_utils.py:757`) while `ModelConfig(...)` is built later in
`create_model_config()` (`:1598`).

## Verify

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"meta-models/Muse-Glimmer-30B",
  "messages":[{"role":"user","content":"Weather in Paris in Celsius? Use get_weather."}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "parameters":{"type":"object","properties":{"city":{"type":"string"},
    "metric":{"type":"boolean"}},"required":["city"]}}}],
  "tool_choice":"auto"}' | python3 -m json.tool
```

Expect `finish_reason: "tool_calls"` and a `tool_calls[0].function` naming
`get_weather` with `arguments` a JSON string. If instead you get
`finish_reason: "stop"` and raw `<atem:function_calls>` text in `content`, the
parser is not registered — check the bundle is installed into the same venv the
server runs from.

## What is tested, and what is not

`tests/test_tool_parser.py` — **19 tests, no device, no weights on the parse path**:

- single call; mixed value types (string / int / bool / list / object)
- parallel calls in one block; namespaced names passed through
- whitespace preserved verbatim; multi-line values keep their newline
- `"NaN"` stays a string (`json.loads` would coerce it to a float)
- prose before a block becomes `content`
- an unterminated block is **not** reported as a tool call
- streaming: content streams, a split open tag never leaks, calls emit once
- the worked example from the template itself parses to that example
- multi-turn round trip through vLLM's own `_postprocess_messages`

**Not tested:** anything on device. The parser is host-side text handling, so the
above is the whole contract — but an end-to-end `tool_choice: "auto"` request
against a live server has not been run, and should be before this is called done.

### One thing that looks like a bug and is not

The chat template raises if `tool_call.function.arguments` is a JSON string:

```
Muse Glimmer ATEM chat template requires tool_call.function.arguments to be a
dict (mapping); a JSON string cannot be parsed in the HF jinja sandbox.
```

OpenAI clients send a JSON string, so this looks like multi-turn tool use is
broken by construction. It is not: vLLM converts it in
`chat_utils._postprocess_messages` before rendering, and jinja genuinely cannot
do it (`tojson` has no inverse), which is why the template refuses rather than
guessing. Calling the tokenizer directly *does* raise — so a test that bypasses
vLLM will mislead you. `test_openai_json_string_arguments_round_trip` goes
through vLLM's conversion for that reason.
