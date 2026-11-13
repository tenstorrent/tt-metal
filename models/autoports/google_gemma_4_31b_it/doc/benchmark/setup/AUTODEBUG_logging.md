# AutoDebug: native response logging buffers the benchmark stream

## Finding (high confidence, source-confirmed)

The installed native response logger destroys streaming delivery when enabled. In `python_env/lib/python3.10/site-packages/vllm/entrypoints/serve/middleware/log_response.py:146-161`, line 148 drains `response.body_iterator` completely with an async list comprehension **before** returning the response. Only afterward does it replace the iterator with a replay iterator. The content-type test occurs after draining, so SSE is not exempt. `_log_streaming_response` (95-134) replays an already-complete list; its generator does not restore live streaming.

This is an actual delivery boundary, not merely logging ahead of an independent network sender: installed Starlette `middleware/base.py:178-198` constructs the response around an internal body stream and awaits the middleware dispatch before invoking the response with the outward `send`. Thus all generation for each response finishes before that response's headers/body are forwarded by this layer.

Registration is controlled by `vllm/entrypoints/serve/middleware/register.py:57-63`. `vllm/envs.py:816-818` enables it only when the environment value lowercases to `true`. The matching benchmark server log, `models/autoports/google_gemma_4_31b_it/doc/benchmark/run/server-b32/server.log:2201,2205,2209`, contains native `log_response.py:134` streaming-started records at 20:29:20-21, supporting that this path was active during the warmup. The label “streaming_started” is emitted after buffering, not at generation start.

## Direct observations and causal interpretation

The current run's `models/autoports/google_gemma_4_31b_it/doc/benchmark/run/perf-b32-warmup.json` reports:

| Measurement | Value |
|---|---:|
| Completed / failed | 32 / 0 |
| Concurrency | 32 |
| All input / output lengths | 4096 / 128 |
| Mean TTFT | 36,734.055624 ms |
| Mean end-to-end latency | 36,739.691283 ms |
| Mean TPOT | 0.044375270 ms |
| Mean ITL | 0.046395102 ms |
| Run duration | 38.019594 s |

Its `.command.json` identifies the separate `bringup/artifacts/benchmark-perf-client/bin/vllm` client, backend `openai-chat`, streaming endpoint `/v1/chat/completions`, 32 requests and exact random lengths. The **actual client** implementation in `bringup/artifacts/benchmark-perf-client/lib/python3.10/site-packages/vllm/benchmarks/lib/endpoint_request_func.py:362,388-431` requests streaming, timestamps the first choices-bearing SSE event as TTFT, and advances the final timestamp on subsequent parsed events (including usage). Its `benchmarks/serve.py:609-610` computes TPOT as `(latency - ttft)/(output_len - 1)`.

The numbers reconcile: `(36739.691282964486 - 36734.05562365224) / 127 = 0.044375270175... ms`. Only about 5.64 ms remains after the first received event. Full buffering predicts exactly this shape: generation and waiting time move into client TTFT, followed by rapid replay/parsing of stored chunks. TPOT/ITL here measure replay delivery, not device decode cadence. TTFT is not usable as prefill latency.

Interpretation limits: source and logs establish the buffering mechanism and activation, and the JSON is consistent with it. They do not partition the 36.7 s into actual prefill, decode, queueing, warmup, and logging costs. No disabled-logger control was run; no claim is made that removing buffering eliminates all of that latency. End-to-end time and throughput still describe this instrumented run, but may themselves include logger overhead.

## Proposed model-owned intervention (not implemented)

Add a pure ASGI middleware class, for example `models.autoports.google_gemma_4_31b_it.tt.response_tee.ResponseTeeMiddleware`, and register it through the existing `--middleware` CLI option. `register.py:65-75` already imports middleware classes and calls `app.add_middleware`; no vLLM or plugin change is needed. Set `VLLM_DEBUG_LOG_API_SERVER_RESPONSE=false` explicitly in all relevant server launch environments. Adding a tee outside the enabled native logger cannot undo its buffering.

Required design:

1. Implement `__init__(app)` and async `__call__(scope, receive, send)`. Pass non-HTTP scopes straight through. Keep all response state local to the call, never on the shared middleware instance.
2. Wrap only `send`. Forward `http.response.start` and every body message immediately with `await send(message)`, preserving bytes, ordering, status, headers, `more_body`, backpressure, cancellation and exceptions. Never drain an iterator, collect SSE before forwarding, or reconstruct a response object.
3. Inspect response content type using its media type, ignoring parameters. For `text/event-stream`, forward without collecting/logging full bodies or doing per-chunk parsing. Stage 11 needs unbuffered delivery, not a complete SSE transcript.
4. For the nonstream JSON completion endpoints, collect a copy of all body fragments as they are forwarded; after successfully forwarding the final body (`more_body` absent/false), join the bytes and decode once. This handles split UTF-8 and JSON boundaries. Log the complete body, including ID, choices/message, reasoning, tool calls, finish reason and usage, without truncation. Do not emit a successful complete record for an interrupted body. Arrange the observer before any response compression, or explicitly handle encoded bodies.
5. Preserve the existing message format exactly: `logger.info("response_body={%s}", decoded_json)`. The extra outer braces are deliberate: installed skill `scripts/reply_digest.py:42,90-106` searches for `response_body={` and JSON-decodes the remainder. Existing `tt/native_log_formatter.py` provides framed complete records; route the new logger through that handler/configuration and confirm its output reaches the digest. The digest accepts plain and TTNATIVEHEX1/2 records and extracts reasoning/content/tool arguments (109-120). A streaming summary is not a substitute for a complete nonstream completion object.
6. Perform complete-body logging after forwarding the final chunk. This avoids withholding the response for logging, but synchronous logging can still occupy the event loop and affect other requests. Reuse an appropriate reliable logging sink; do not claim zero instrumentation overhead or silently drop records with an unbounded/fire-and-forget scheme.

## Secondary discrepancy

Native `_log_non_streaming_response` at 137-143 decodes only `response_body[0]`, despite the caller collecting a list of all fragments. A fragmented nonstream JSON body would be incompletely logged. This is a concrete logging limitation, not the cause of the streaming timing anomaly; ordinary one-fragment JSON responses are unaffected. The proposed tee must join all fragments. Native SSE logging also truncates content at 2048 characters (119-120), so it cannot supply full stage-12 reply evidence.

## Validation to perform after implementation

Host-only ASGI tests should gate the producer after its first body message and assert the receiver has already seen that message before the gate opens; compare every forwarded message unchanged. Test concurrent requests, non-HTTP passthrough, disconnect/send failure, fragmented JSON and UTF-8, empty final chunks, and a large nonstream completion with reasoning/tool calls. Feed actual framed output into the existing `reply_digest.responses`/`reply_text` and verify full content, ID, finish reason and usage survive. Confirm launch configuration disables native logging and installs the model-owned class.

A later authorized rerun of the same warmup is the controlled runtime check for restored token delivery. Preserve this result as affected evidence and regenerate latency measurements; do not infer corrected TTFT/TPOT from the buffered sample.

## Report review and scope

Rechecked the headline against the logger, Starlette outward-send boundary, registration condition, matching server log, exact warmup artifact, and the separate benchmark client's metric code. The installed server environment's benchmark client differs slightly in trailing-usage timestamp handling; the report uses the recorded client environment instead. No device commands, API calls, launcher invocation, runtime edits, or broader model investigation were performed. Only this report was written.
