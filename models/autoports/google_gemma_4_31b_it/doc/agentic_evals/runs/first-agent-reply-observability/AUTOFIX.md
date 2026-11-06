# AutoFix: source-supported stock response logger candidate

**A configuration-only candidate survives the host checks; execution-route confirmation is pending. AutoFix is not exhausted.** Fresh independent AutoDebug found the overlooked stock `VLLM_DEBUG_LOG_API_SERVER_RESPONSE="true"` middleware. Earlier RESULT.md's no-self-contained-fix conclusion is superseded by this finding, while its ordinary-output-logger and OTLP limitations remain correct. No dry pass, device result or client-receipt proof is claimed.

Failing contract: exact firstC GPQA requests, including reasoning-only/empty-visible completions, must be shown to receive model replies before scoring on the unchanged authorized route. Prior output logging omitted these responses; aggregate progress and physical slots did not establish identity. `AUTODEBUG.md` records the fresh personal-login gpt-6-astra/medium isolated investigation. Launcher sandbox preflight initially failed because bwrap was absent before model startup; documented skip-child-sandbox retry succeeded within existing host read-only authority. No API-key fallback was used. Root reports separate model-interface diagnosis: plugin model_runner.py4098–4148/async_decode.py1343–1449 exposes no request IDs/finalization; release_request(slot) conflates finished and preempted without reason/ID; model-owned phase request_ids=None and client_emission_verified=false. Those cannot repair identity without a prohibited interface/runtime change.

## Discriminating checks

1. Ordinary enable-log-requests/enable-log-outputs/max-log-len: executed the unchanged logging AST block with synthetic objects (`python check_log_guard.py`). Nonempty visible output logged once with the exact ID; reasoning-only and empty output logged zero times. Confirms the causal `if output_text` guard without a model/server launch.
2. DEBUG success IDs: inspected exact AsyncLLM and OutputProcessor finish/removal paths. No success-ID log exists; verbosity cannot create it. A nonempty-only device test was explicitly declined because it cannot prove the failing case.
3. OTLP: finished request spans contain external_req_id even empty, but source installs only network exporters. No existing authorized collector/read path supplied, package availability unverified. No collector or service created; no profiling.
4. Response-ID headers: stock middleware can add a header, but lm-eval does not preserve it as completion identity, and auto-generated header ID is not necessarily the chat ID. Not sufficient.
5. **Stock response-body middleware:** independently verified envs.py815–818 requires string true; middleware/register.py57–63 registers built-in log_response; log_response.py137–161 logs nonstreaming JSON without testing visible content and restores body bytes. Chat response serialization retains the same start ID and finish reason. Ran `python check_response_logger.py` on unchanged AST functions with synthetic one-chunk JSON responses. Visible, reasoning-only and empty cases each logged exactly once, preserved ID/finish reason, and returned byte-identical body. Compact result: response-logger-host-check.json. This is a host source-fragment check, not actual CI middleware/HTTP/client validation.

## Minimum proposed delta, not applied

Merge only into this model's existing entry:

```json
{
  "env_vars": {"VLLM_DEBUG_LOG_API_SERVER_RESPONSE": "true"},
  "device_model_spec": {"vllm_args": {"enable_log_requests": true, "max_log_len": 0}}
}
```

This is a merge illustration, not a replacement model spec. Preserve every existing key, pin, sampling setting, concurrency, task order, output cap and timeout. INFO level suffices. The separate enable_log_outputs flag is unnecessary. TTI runtime-spec env transport and vllm_args serialization are verified in AUTODEBUG.md against TTI37ee508; verify parity for chosen release TTI. No custom middleware import, shared/plugin/runtime/harness/workflow mutation, new endpoint, service, credentials, parser or model-output change is required.

The built-in response logger **ignores max-log-len** and logs full serialized responses, including private reasoning. Keep raw bodies solely in existing private logs/artifacts; no raw content in committed evidence or telemetry. It buffers nonstreaming bytes before sending; overhead, memory, retained log size and actual CI log truncation must be measured. This candidate may increase latency and log volume; no neutrality claim. It logs after serialization and before send, so exact client receipt remains a separate confirmation. No-score/cancellation races are unchanged.

## Remaining coordinator confirmation

Coordinator must review/admit the model-entry-only delta before any live work. Existing effort box0.10deviceh,0used. Original0.04h estimate was below observed226s startup; use at least226s startup plus approximately30s simple request/cleanup (0.0711h guess) and reprice if the actual full B8/C4 launcher differs. No truncated or timed-out outstanding TT work to force budget fit. A minimal approved local standard-client check should include a native empty/reasoning-only response already available through the authorized workload, verify full same-ID JSON body, unchanged bytes/client parse and clean idle shutdown. Do not fabricate or shorten a model workload to manufacture the failing case; if a required full cap cannot fit, report the cost and keep it pending rather than launch under this box. Host checks already cover serialization branches cheaply; actual initialC/REST timing must still be confirmed on the normal frozen route.

Only after accepted live/source integration can coordinator publish the model-owned config, freeze its SHA/settings, and attempt the exact dry. Record firstC Received/Added IDs after isolated eval starts, bind all same-ID final response bodies, and separately establish client parse and pre-score stop. This work has not proved those execution conditions. Changed files are only this effort's diagnostic scripts/reports; no model entry or runtime files were edited. Preserve the old findings as history instead of rewriting once-only evidence.
