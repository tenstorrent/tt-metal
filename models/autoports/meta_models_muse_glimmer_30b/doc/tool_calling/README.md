# Tool calling for agentic coding

Muse-Glimmer emits tool calls in its own ATEM grammar. Stock vLLM parsers do
not recognize that grammar, so a server without the model-owned parser returns
the raw markup as prose with `finish_reason: stop`. The parser in
`tt/muse_glimmer_tool_parser.py` converts the markup to OpenAI-compatible tool
calls for coding agents and other tool-using clients.

## Protocol contract

The checkpoint emits channel-framed assistant messages:

```text
<|start|>assistant to=self<|message|>reasoning<|eom|>
<|start|>assistant to=read_file<|message|>
<atem:function_calls>
<atem:invoke name="read_file">
<atem:parameter name="path">src/app.py</atem:parameter>
</atem:invoke>
</atem:function_calls><|eom|>
<|start|>assistant to=user<|message|>final answer<|eot|>
```

Channel scoping is security-relevant. Only ATEM invocations in a tool-recipient
message are dispatched. Markup quoted in `to=self` reasoning or a `to=user`
answer remains text and cannot become a tool call.

The tool and reasoning parsers together support:

- OpenAI Chat Completions in streaming and non-streaming mode.
- `tool_choice` values `auto`, `required`, and a named function.
- Multiple tool calls in emission order.
- Mixed string, scalar, list, and object arguments.
- Stable JSON argument serialization and generated streaming call IDs.
- Reasoning, calls, and final content in the same generated turn.
- Markers split across stream chunks without leaking protocol tokens.
- Recovery from damaged/truncated framing without fabricating calls.
- Bare-name normalization: `get_weather.get_weather` maps to a registered
  `get_weather`; other unmatched namespaces are never guessed.
- Multi-turn OpenAI tool results through vLLM's chat-template preprocessing.
- Separation of `to=self` analysis into `reasoning_content` without consuming
  an adjacent tool-recipient message.

The request hook forces `skip_special_tokens=false`. These channel markers are
the only reliable way to distinguish executable calls from quoted markup.
Required and named choices deliberately avoid vLLM's generic JSON-guidance
path because Muse-Glimmer generates ATEM rather than a JSON tool envelope.

## Serve with tt-model

The published package contains one image with three launch profiles. Select the
profile matching the number of Blackhole ASICs made available to the container:

| profile | mesh | visible devices | fabric | intended use |
|---|---:|---|---|---|
| `p150` | 1x1 | `0` | disabled | one interactive agent |
| `p150x2` | 1x2 | `0,1` | `FABRIC_1D_RING` | one interactive agent |
| `p150x4` | 1x4 | `0,1,2,3` | `FABRIC_1D_RING` | validated four-chip/capacity default |

```bash
tt-model serve tt-hous/muse-glimmer-30b --profile p150
tt-model serve tt-hous/muse-glimmer-30b --profile p150x2
```

`tt-model` enables both parser names and loads both model-owned plugin files
from the image. Do not add parser flags after the bundle ID.

## Direct developer serve

For a direct developer serve, load both model-owned files explicitly and use
the clean plugin revision that registers the architecture. P150 uses no fabric
configuration:

```bash
MESH_DEVICE=P150 TT_METAL_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B \
  --block-size 64 --max-num-seqs 1 --max-model-len 131072 --port 8000 \
  --enable-auto-tool-choice \
  --tool-parser-plugin models/autoports/meta_models_muse_glimmer_30b/tt/muse_glimmer_tool_parser.py \
  --tool-call-parser muse_glimmer \
  --reasoning-parser-plugin models/autoports/meta_models_muse_glimmer_30b/tt/reasoning_parser.py \
  --reasoning-parser muse_glimmer \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":400000000,"trace_mode":"decode_only"}}'
```

P150x2 uses the same command with a 1x2 mesh and the measured multi-chip
fabric/L1 settings:

```bash
MESH_DEVICE=P150x2 TT_METAL_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B \
  --block-size 64 --max-num-seqs 1 --max-model-len 131072 --port 8000 \
  --enable-auto-tool-choice \
  --tool-parser-plugin models/autoports/meta_models_muse_glimmer_30b/tt/muse_glimmer_tool_parser.py \
  --tool-call-parser muse_glimmer \
  --reasoning-parser-plugin models/autoports/meta_models_muse_glimmer_30b/tt/reasoning_parser.py \
  --reasoning-parser muse_glimmer \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":400000000,"fabric_config":"FABRIC_1D_RING","fabric_packet_payload_bytes":8192,"l1_small_size":6144,"trace_mode":"decode_only"}}'
```

Do not set `VLLM_PLUGINS`; it is an allowlist and can suppress the Tenstorrent
platform plugin.

## API smoke test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"meta-models/Muse-Glimmer-30B",
    "messages":[{"role":"user","content":"Weather in Paris in Celsius? Use get_weather."}],
    "tools":[{"type":"function","function":{
      "name":"get_weather",
      "description":"Get current weather for a city.",
      "parameters":{"type":"object","properties":{
        "city":{"type":"string"},"metric":{"type":"boolean"}
      },"required":["city"]}
    }}],
    "tool_choice":"auto",
    "max_tokens":256,
    "temperature":0
  }' | python3 -m json.tool
```

Pass criteria are `finish_reason: "tool_calls"`, a function named
`get_weather`, and `function.arguments` containing valid JSON matching the tool
schema.

## Agentic coding acceptance

After the packaged server is ready, run the bounded multi-turn harness:

```bash
python models/autoports/meta_models_muse_glimmer_30b/tests/tool_calling_harness.py \
  --base-url http://127.0.0.1:8000 \
  --model meta-models/Muse-Glimmer-30B
```

It creates a disposable project, exposes only `list_files`, `read_file`,
`write_file`, and `run_tests`, and requires the model to inspect the source,
repair a defect, obtain a passing test run, and then give a final answer. Paths
cannot escape the temporary workspace, writes cannot create unexpected files,
and the test command is fixed rather than model-controlled.

## Offline tests

The CPU suite requires no model execution and covers channel isolation,
argument decoding, parallel calls, malformed output, auto/required/named
choices, streaming boundary splits, parser registration, and a tokenizer/chat
template round trip pinned to weight revision
`f84ecc3a0ea984a4c04542a84269e3d065350a6e`.

```bash
python -m pytest -q \
  models/autoports/meta_models_muse_glimmer_30b/tests/test_tool_parser.py
```

Hardware API and coding-harness results must be recorded separately for each
published image digest; parser unit tests alone do not qualify a release.

## Tool-calling latency sweep

Run the live API sweep after each packaged profile is ready. Unlike the legacy
fixed-token generator sweep, every request carries a tool schema and must
produce a valid structured call:

```bash
python models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/tool_call_latency_sweep.py \
  --profile p150 \
  --base-url http://127.0.0.1:20000 \
  --repeats 3 \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/p150_tool_call_latency.json
```

Repeat with `--profile p150x2` and a `p150x2_...json` output path. The runner
warms each shape once, validates the returned function and JSON arguments, and
records TTFT, derived TPOT, E2E, and all raw samples.
