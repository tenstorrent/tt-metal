# The chat server

`tools/run_qwen38_chat_server.sh` starts `tools/qwen38_chat_server.py` on one 1x4 mesh; `--help` lists its flags.
This page covers what the flags do not: start-up, the request rules, `/health`, follow-up turns, lanes and MTP.

## Start-up

- **PLE table**: decode reads rows of the PLE n-gram table from the checkpoint through the host page cache. On a
  host whose RAM holds it, `tools/prewarm_ple_table.py --checkpoint DIR` reads it once before the start
  (`--report-only` prints the residency only).
- **Launcher**: prints the checkout, interpreter and `ttnn` extension, then exports the profile's mesh graph
  descriptor, the device set, the cache and log roots and `TT_METAL_HOME`. It starts the server with `--sampling` and
  `--stall-seconds 300` unless told otherwise. It takes no device locks.
- **Runtime admission** (`tools/runtime_admission.py`): the interpreter's `ttnn` and its extension must resolve
  under this checkout, or the start is refused with both paths. The checkout's head, tree, dirty flag and extension
  digest become the runtime identity, recorded in the cache namespaces, every response's `system_fingerprint` and
  `result.json`. A tree without git history needs `QWEN38_TT_METAL_SHA`.
- **BF4 expert cache** (`<cache-root>/caches/bf4-experts/`): the first start converts the routed experts of every
  MoE layer; `--prepare-only --bf4-stage-limit N` converts at most N layers per run and resumes. The cache is keyed by
  the checkpoint, the converter's sources and the mesh's DRAM bank count and order, not by the tt-metal revision.
  Every start re-packs one expert and compares it with the cache; a mismatch is refused, naming the layer, expert and
  tensor.
- **READY**: the weights load, the traces are captured, the acceptance records replay
  ([NUMERICS.md](NUMERICS.md)), then the server writes `READY` and listens.

The run directory (`<cache-root>/runs/<stamp>/`) holds `READY`, `phase-markers.jsonl`, `requests.jsonl`,
`acceptance.json` and, at shutdown, `result.json` and `STOPPED`.

## Endpoints

`POST /v1/chat/completions` (SSE stream or one JSON document), `GET /v1/models`, `GET /v1/models/<id>` (another id is
404 `model_not_found`) and `GET /health`. `/health` reports the context limit, the sampling mode and defaults, the
free DRAM after the captures, the runtime identity, the defaults (`defaults.system_prompt` is null), the
`logprobs_normalizer` and `current_request` with the seconds since its last device step; MTP, lanes and vision
servers add their own blocks. `HEAD` and `OPTIONS` are served; a request body needs `Content-Length`.

## Requests

- Fields: `messages`, `max_tokens` / `max_completion_tokens` (default and limit: the remaining context), `stream`,
  `stop`, `tools` / `tool_choice` (OpenAI shape), `enable_thinking` (default true; reasoning streams as
  `reasoning_content`), `reasoning_effort`, `thinking_budget`, `ignore_eos`, `seed`, `temperature`, `top_p`, `top_k`,
  `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` and `logprobs`. `chat_template_kwargs`
  means the same as the top-level fields; JSON `null` is an absent field.
- A request naming no sampling field is greedy: the argmax stream the acceptance replay measures. `temperature > 0`
  samples; another sampling field alone samples with the model card's profile (`/health.sampling_defaults`). A
  `--no-sampling` server refuses sampling fields unless `temperature` is 0.
- The device samples in the TAIL trace when the request's parameters are in its range, otherwise the host samples
  the same candidate row; `qwen38.sampling.sampler` names the path. `--host-sampler` keeps every request on the
  host. `logprobs` are relative to the candidate row (`logprobs_normalizer`).
- What the server cannot honour is refused with HTTP 400, never dropped: `response_format` other than `text`,
  `logit_bias`, `parallel_tool_calls: false`, `extra_body.mtp_drafts`. Unknown fields are logged.
- One request decodes at a time and up to `--queue-limit` (4) wait; the next gets HTTP 503 with `Retry-After`. A
  prompt over the context limit gets HTTP 400 `context_length_exceeded`.
- **Images**: `image_url` parts with `data:` URLs only (nothing is fetched). `detail` `auto` / `high` use the
  checkpoint's image processor; `low` caps the pixel count at 512 x 512. Video parts, images in a system message
  and unknown part types are HTTP 400; undecodable bytes are `invalid_image`. The vision tower loads at start when
  the DRAM admission allows it; otherwise the server starts text-only and image requests get `vision_unavailable`.
  `usage.prompt_tokens` counts the image tokens; `qwen38.vision` reports the grids and tower time.

## Follow-up turns and the prompt-end snapshot

The device prompt is the client's messages rendered by the chat template; the server adds no system prompt and
reasoning never re-enters the context. When the new render extends the committed ids the server prefills only the
new turn (`qwen38.prefix_reused`). Otherwise, before resetting, it tries the prompt-end snapshot: before the last
prompt token of every request the chain copies the recurrent state (GDN states and ring slots, PLE slots, QSA
staging and raw-key rings; the KV caches are positional) into a resident snapshot, and a request whose render extends
those ids restores it and prefills only the tail (`qwen38.prefix_restored`). This is what keeps thinking
conversations incremental, since the template re-renders a past think block differently from what the model
generated. A diverging history resets and prefills the whole conversation (`qwen38.reset`). `snapshot_schedule`
says how exact the restored state is ([NUMERICS.md](NUMERICS.md)).

## Hang-ups, deadlines and the watchdog

- A client hang-up is noticed at the next device step or prefill event; a queued request whose client left gives up
  its place.
- A streaming request gets its role chunk on admission and an SSE `: keepalive` comment every `--heartbeat-seconds`
  (30) through the queue and the prefill. A socket write blocked for `--socket-timeout-seconds` (60) ends the request.
  `--request-deadline-seconds` (off by default) applies inside the prefill too.
- `--stall-seconds N` fires only on zero progress of the request holding the device: every decode step, prefill
  event and admission restarts its clock. On a stall the server logs `stalled` and exits 1 without releasing the
  chain, for a supervisor to restart it. N must exceed `--socket-timeout-seconds`; 0 disables it.
- SIGTERM or `--serve-seconds` drains: the request in flight ends at its next step with `qwen38.finish` `shutdown`,
  queued and new requests get 503, then the chain is released.

## Concurrent requests: `--lanes B`

`--lanes B` (B in 2..8, B x (k + 1) <= 32; needs `--mtp`; greedy only) serves up to B requests at once through the
MTP lane chain, each lane with its own KV, GDN and PLE state. A request that finds a free lane is admitted at the next
pass boundary: its prompt prefills on the single-stream chain and its state moves into the lane. The admission runs
in segments; between two segments the driver runs a pass for the decoding lanes once the admission work since their
last pass reaches `--lanes-stall-budget` seconds (default 0.5; `0` runs a pass at every segment, `off` runs every
admission whole). Lanes do not reuse prompt prefixes. Responses carry `qwen38.lanes`; `/health.lanes` has the live
counts. At start the server checks the lanes' DRAM admission, the measured growth of the lane traces and the
start-up gate ([NUMERICS.md](NUMERICS.md)).

## MTP: `--mtp K`

The chain also holds the MTP layer and its verify, draft and commit traces. Greedy requests draft through the fused
verify; on an `--mtp --sampling` server sampled requests draft too, through the split verify, decided per pass by
exact speculative sampling on the host. `QWEN38_MTP_DEVICE_ACCEPT=1` (default off; needs the split verify) adds a
form whose acceptance runs on the device. MTP servers sample on the host, so `--device-sampler` with `--mtp` is
refused. `--mtp-gdn-anchor layer0` re-anchors layer 0's GDN state from the 1-row recurrence. The open checks the
chain's DRAM growth against the admission's estimate (`/health.mtp.admission`). `QWEN38_MTP_SAMPLED`,
`QWEN38_MTP_PLE_EARLY`, `QWEN38_MTP_DRAFTS_PER_REQUEST` and `QWEN38_MTP_MOE_ROWS` are not read; setting one refuses
the start.
