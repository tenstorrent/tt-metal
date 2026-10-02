# MiMo-V2.6-Flash OpenAI-compatible server

`server.py` serves the 48-layer model on a TT mesh (BH LoudBox 2x4) behind an OpenAI-style API, for agent harnesses
(Pi, opencode, anything using `openai-completions` / `@ai-sdk/openai-compatible`).

- `POST /v1/chat/completions`: `stream` true (SSE) or false; `tools` -> `tool_calls` (finish_reason `tool_calls`);
  the model's `<think>...</think>` -> `reasoning_content`, the rest -> `content`; `temperature` / `top_p` / `seed` /
  `stop` / `max_tokens` (or `max_completion_tokens`); `stream_options.include_usage`; thinking per request via
  `chat_template_kwargs: {"enable_thinking": false}` or `reasoning_effort: "none"`.
- `GET /v1/models`, `GET /health`.
  Thinking per request also via top-level `enable_thinking` (Qwen style).
- Prefix KV cache across requests: the server keeps the token list in the KV cache and re-prefills only from the chunk
  holding the first differing token (`usage.prompt_tokens_details.cached_tokens`). The template renders past assistant
  turns as `<think>{reasoning_content}</think>content`, so a harness that does not echo `reasoning_content` back
  diverges at the previous assistant turn (still reused up to there).
- Each response carries a non-standard `timings` object (prompt / cached tokens, TTFT, decode tok/s).
- One request at a time (queued); prompt + completion must fit `MIMO_SERVE_MAX_CTX` (400 otherwise).

## Start / stop (on the box)

```bash
cd /localdev/mstaletovic/tt-metal
models/demos/mimo_v2_d_p/serve/start.sh                      # defaults: 2x4, 48 layers, 64K ctx, chunk 1024, port 8000
# e.g. models/demos/mimo_v2_d_p/serve/start.sh MIMO_SERVE_MAX_CTX=32768 MIMO_SERVE_THINKING=0
grep -m1 'MIMO_SERVE: listening' generated/mimo_serve/server.log   # ready after the model load (minutes)
models/demos/mimo_v2_d_p/serve/stop.sh                       # SIGTERM: closes the mesh, frees the device lock
```

`start.sh` runs `test_serve.py::test_serve` through `scripts/run_safe_pytest.sh` (detached), so the server holds
`/tmp/tt-device.lock` while it runs (other device jobs wait), gets hang triage, and the device is reset afterwards.

Env (pass as `start.sh VAR=val ...`): `MIMO_SERVE_PORT` (8000), `MIMO_SERVE_MAX_CTX` (65536), `MIMO_SERVE_CHUNK`
(1024), `MIMO_SERVE_LAYERS` (48), `MIMO_SERVE_THINKING` (1), `MIMO_SERVE_MAX_TOKENS` (8192, default completion cap),
`MIMO_SERVE_TEMPERATURE` / `MIMO_SERVE_TOP_P` (0.6 / 0.95 when the request sets none; 0 = greedy),
`MIMO_SERVE_MODEL_ID` (mimo-v2.6-flash), `MIMO_SERVE_WARMUP` (tokens of a dummy prompt prefilled at start-up to JIT
the chunk positions), `MIMO_MESH` (2x4).

Check from the box (or through the tunnel): `python models/demos/mimo_v2_d_p/serve/client_check.py`.

## From your machine

```bash
ssh -p 49210 -N -L 8000:localhost:8000 bh-lb-17
curl -s localhost:8000/v1/models
```

### Pi (`~/.pi/agent/models.json`)

```json
{
  "providers": {
    "mimo-tt": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "compat": { "supportsDeveloperRole": false, "thinkingFormat": "qwen-chat-template", "maxTokensField": "max_tokens" },
      "models": [
        {
          "id": "mimo-v2.6-flash",
          "name": "MiMo-V2.6-Flash (TT LoudBox)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 8192,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

### opencode (`opencode.json`)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mimo-tt": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MiMo on Tenstorrent",
      "options": { "baseURL": "http://localhost:8000/v1", "apiKey": "none" },
      "models": {
        "mimo-v2.6-flash": { "name": "MiMo-V2.6-Flash", "limit": { "context": 65536, "output": 8192 } }
      }
    }
  },
  "model": "mimo-tt/mimo-v2.6-flash"
}
```

Verified on the box (Pi 0.73.1 `pi -p`, opencode 1.18.34 `opencode run`, node 22 from a tarball): both completed a
read-tool turn and answered. Pi: `PI_CODING_AGENT_DIR=<dir with models.json> pi --model mimo-tt/mimo-v2.6-flash -p "..."`.

## Performance notes

Measured (BH LoudBox 2x4, 48 layers, chunk 1024, warm JIT): ~4.1-4.6 tok/s at 0-25K context; TTFT 0.21 s for a
<1K prompt, 5.3 s for a 25K-token prompt with no cache hit vs 0.25 s when the previous turn is reused (24,576 of 25,106
tokens cached); opencode's 7.2K-token system prompt: 1.86 s first turn, 0.45 s on the tool-result turn (7,168 cached).
The first request after start-up at a new chunk position pays the JIT compile (~40 s for the very first request).

Generation re-prefills the chunk holding the newest token for every token (no decode path yet), so tok/s is that of a
`MIMO_SERVE_CHUNK`-token prefill at the current context. Agent system prompts are long; the prefix cache keeps
follow-up turns to the changed tail, but the first request of a session pays the full prefill (plus JIT compile per new
chunk position the first time the server sees it).
