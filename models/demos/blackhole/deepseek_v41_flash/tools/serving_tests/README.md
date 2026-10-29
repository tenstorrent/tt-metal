# Serving tests (run against a live server)

Plain Python 3 scripts (standard library; `Pillow` for the image ones) that talk to `http://127.0.0.1:8000`. They were run against the image of this release at 16 users / 262,144 tokens and at 4 users / long context; the results are in `RELEASE_NOTES.md` and `GETTING_STARTED.md`.

| Script | What it checks |
|---|---|
| `sanity.py` | sampled request, streaming, 8 concurrent arithmetic questions |
| `long.py` | a ~20,000-token prompt with a number to retrieve |
| `needle_probe.py` | short questions, 4 concurrent GSM8K 8-shot requests (set `GSM8K_PROMPTS` to the prompt file), then a 100,000-word haystack with the secret code at the end and at the start (the answer must be 4817) |
| `serving_api_test.py <max_model_len>` | 23 OpenAI-API checks: models, chat, completions, streaming + usage, stop, finish reasons, seeds, top-k, the logprobs / `n` policy, too-long prompts, unknown model, malformed bodies, concurrent mixed streaming, client disconnect |
| `overload_test.py <slots>` | 4 x `max_num_seqs` requests at once, two waves: every answer correct, `/health` 200 afterwards |
| `client_img.py` | text only, one image, two images, 4 image + 4 text at once, streaming |
| `client_many.py` | 16 images in one prompt, one maximum-size image, 17 images (must be refused with an OpenAI-style 400) |
| `client_stress.py` | 16 concurrent image requests, then 48 random images (number and colours) |
| `tool_protocol_probe.py [--base-url URL] [--out probe.json]` | 7 tool-protocol cases, PASS/FAIL each, exit code 1 on any FAIL (see below) |

The GSM8K 8-shot accuracy and the speed sweeps use tt-inference-server's `evals` and `benchmarks` workflows (see the guide).

## tool_protocol_probe.py

```
python3 tool_protocol_probe.py --base-url http://<server>:8000 --out probe.json     # exit 0 = all PASS, 1 = a case FAILED, 2 = server unreachable
```

Run it against every serving variant (catalog default, and each `assets/runtime_spec_*.json` launch). The prompt is inspected through the server's own `usage.prompt_tokens` (same request with and without the item under test, `max_tokens=1`), so it works with the stock jinja template and with `tokenizer-mode deepseek_v41`. The optional `/tokenize` check is reported on its own `INFO` line and never fails the run (on the deepseek_v41 renderer it can fail with a pydantic pickling error for history messages).

| Case | Guards |
|---|---|
| 1 `schema_in_prompt` | declared `tools` reach the prompt (the stock `vllm_config/chat_template.jinja` has no tools handling) |
| 2 `tools_request_parsed` | `--enable-auto-tool-choice --tool-call-parser deepseek_v41` are set and the model's call comes back as `tool_calls` (without the flags the server answers 400) |
| 3 `assistant_tool_calls_in_prompt` | an assistant message with `tool_calls` and `content: null` in the history is encoded |
| 4 `tool_result_in_prompt` | a `role: tool` message with `tool_call_id` (OpenAI format) is encoded |
| 5 `plain_chat_placement` | no thinking kwarg: the answer is in `content`, `reasoning` is empty (`default-chat-template-kwargs {"thinking": false}`) |
| 6 `thinking_true_placement` | `chat_template_kwargs {"thinking": true}`: reasoning in `reasoning`, answer in `content` (`--reasoning-parser deepseek_v41`) |
| 7 `tool_loop_marker` | a 2-turn `get_weather` loop; the tool result carries a random marker that only the result contains, the final answer must quote it |

The CPU side of the same guard is `tests/test_runtime_specs_tool_flags.py` (spec flags, and an offline render of the three message shapes).
