# AutoDebug: stock response-body logging is an overlooked completion-ID source

## Verdict

The empty-visible-content gap is real for `--enable-log-outputs`, but the earlier conclusion that no self-contained stock configuration can expose those response IDs is too broad. Pinned vLLM has `VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true`: its existing middleware logs the serialized nonstreaming response, including ID and choices, independently of visible content. Model-owned metadata can enable it without a shared runtime, parser, harness, workflow, generation-limit, service or credential change.

This is a **source-supported configuration candidate, not a completed dry-run proof**. It witnesses a constructed server reply, not client receipt or parsing. If the contract requires an exact client-receipt event for every initial C identity, that stronger requirement remains unproved. No admissible shared-source patch is needed or proposed.

## Scope and provenance

Read `plan.md`, `SOURCE_FINDING.md`, `RESULT.md`, startup contracts, and the AutoDebug skill; used its already-isolated investigator path. No launcher, subagents, target execution, network, hardware, dispatch or push. Only this report was written.

References prefixed `V/` are relative to `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/vllm-server-source-v0.31.0/`: HEAD `db9527a46873454610df6dbedf79a36d6bf1a7f6`, clean status. Inspected plugin `ca7a18246e4111629ed6cb4a7d1192002749fccf:docs/install-vllm-tt.sh:42` directly with `git show`; it installs vllm 0.31.0. This verifies intended source/version, not the contents of a running CI image.

References prefixed `T/` are relative to `/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/`, clean HEAD `37ee508969c730320c01259fafe340e6c2c14d4e`. Transport findings apply to that inspected checkout; parent must verify parity with the exact proposed immutable release SHA.

## Direct source findings

1. **Existing output logger skips valid replies.** `V/vllm/entrypoints/openai/chat_completion/serving.py:1207–1238` derives `output_text` only from content/tool calls, then gates the logging call on its truthiness. A choice with empty content, reasoning and no tool calls returns normally without a Generated-response log. `max_log_len=0` truncates inside `V/vllm/entrypoints/serve/utils/request_logger.py:90–113`, after this guard, so neither increasing that limit nor DEBUG fixes the omission. `V/vllm/v1/engine/async_llm.py:730–787` and `output_processor.py:751–790` supply no ordinary success-ID log. Raising log verbosity alone is insufficient.

2. **Separate built-in logging bypasses that guard.** `V/vllm/envs.py:815–818` recognizes the literal string `true` (case insensitive; use a quoted string in metadata). `V/vllm/entrypoints/serve/middleware/register.py:57–63` registers the existing `log_response` function; launcher `app.py:54` calls this registration. `V/vllm/entrypoints/serve/middleware/log_response.py:146–161` drains the response iterator, restores its bytes for sending, and logs nonstreaming bodies through lines 137–143. That function logs the decoded body without testing content or reasoning and without `max_log_len` truncation. This is distinct from setting `VLLM_LOGGING_LEVEL=DEBUG`.

3. **Identity survives serialization.** Chat serving lines 291–297 construct the ID; lines 316–317 and 342–347 use that ID for the single-input start log. Lines 948–959 drain generation before response construction; lines 1189–1204 put the same ID and final choices in `ChatCompletionResponse`. `V/vllm/entrypoints/openai/chat_completion/api_router.py:69–73` serializes it with `JSONResponse(generator.model_dump())`. Thus the empty-visible n=1 case still has a nonempty JSON body containing its ID, choice and finish reason. Logging that body is server response evidence even though the answer text may be unusable for GPQA.

4. **Metadata transport exists.** `T/workflows/model_spec.py:729–735` merges model/device environment declarations. Runtime JSON is mounted at `T/workflows/run_docker_server.py:759–770`; `T/vllm-tt-metal/src/run_vllm_api_server.py:1004–1049` merges those declarations and writes `os.environ`. Lines 1477 and 1507–1509 set them before starting vLLM in the same process. Do not cite Docker lines 723–726 as the vLLM transport: that branch is for FORGE/MEDIA. The runtime-JSON path is the relevant one.

## Smallest proposed intervention and proof boundary

Merge into the selected autoport model entry (preserve all existing fields):

```yaml
env_vars:
  VLLM_DEBUG_LOG_API_SERVER_RESPONSE: "true"
# Under existing device_model_spec.vllm_args:
# enable_log_requests: true
# max_log_len: 0
```

Keep INFO logging. `enable_log_outputs` is optional/redundant for this witness. No custom `--middleware` import is necessary: this switch activates stock code. Leave sampling, stream=false, n=1, parser, scheduling, order and all limits unchanged.

For the frozen run, identify initial C starts using the isolated eval boundary and matching admission evidence; do not infer them from completion order or TCP ports. Match their exact IDs to serialized successful chat-response bodies with one finished choice, including empty content. Retain native client progress/error evidence separately. A timestamp on this body log occurs after generation and serialization but before sending the restored body to the client; it is not a client acknowledgement. Server logging does not impose a before-scoring barrier. Existing evidence must independently establish the required timing relationship, and it cannot be reconstructed for already-finished runs whose logging was disabled.

The switch logs full responses (including reasoning) and buffers them before delivery. Zero max-log-len does **not** redact this logger. Retain raw output only in existing private logs, exclude it from source publication, and report compact IDs/timing/counts. Additional memory, logging latency, artifact size and log truncation require ordinary-route validation; no performance neutrality claim is justified. If identity-only logging with no full response retention is mandatory, this candidate does not satisfy that extra condition.

## Alternatives and remaining uncertainty

- OTLP is source-capable: `V/vllm/v1/engine/output_processor.py:769–770,799–859` emits finished-request spans with `external_req_id`, without a visible-content guard. `V/vllm/tracing/otel.py:65–101` explicitly constructs a batched network exporter, not a flag-selected console/file exporter. No existing authorized collector/read route was supplied; focused inspection of TTI Docker/reference/workflow files found none. This is not proof that none exists elsewhere. No endpoint was contacted.
- Request-ID headers alone do not supply a retained completion event. `V/vllm/entrypoints/serve/middleware/x_request_id.py:29–37` can generate an independent UUID when the incoming header is absent, whereas engine serving `serving.py:120–129` separately selects its base ID. Do not assume the generated header equals the chat ID.
- The response logger reads only `response_body[0]`. The selected API produces an ordinary JSONResponse; local Starlette `python_env/lib/python3.10/site-packages/starlette/responses.py:165–169,183–194` sends that body as one ASGI message. This supports the selected path, but does not establish the exact CI dependency/middleware stack. A different middleware splitting the body could truncate this witness. Preserve this deployment check rather than promising universal body logging.
- The supplied reports describe two empty-visible cohort outputs and a stale-log/cancellation race. Those are reported observations, not independently replayed or identity-mapped here. Static findings prove why the output-log route misses such responses; they do not identify which initial C requests were affected, prove all first replies arrived, or resolve cancellation races.

## Report review

Rechecked the headline against the independent registration, environment transport, response serialization and logging guards. The candidate survives the reasoning-only counterexample and requires no collector. Narrowed its claim to server response construction: successful client receipt and exact before-scoring timing remain separate requirements. No claim of completed confirmation or admissible runtime-source fix is made.
