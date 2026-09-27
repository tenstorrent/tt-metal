# Request admission diagnosis

2026-09-27. Investigation was host/source-only: no calls to the live server, no device access, and no model execution.

## Root cause

The TT platform validator has a stale argument signature for installed vLLM0.26:

- Installed `vllm/v1/engine/input_processor.py:296` calls `current_platform.validate_request(processed_inputs, params)`.
- Checkout plugin `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py:836` requires `(prompt, params, processed_inputs)` after `cls`.
- Every ordinary completion request therefore raises `TypeError: TTPlatform.validate_request() missing 1 required positional argument: 'processed_inputs'` before EngineCore request submission.
- Installed `vllm/v1/engine/async_llm.py:607` catches unexpected exceptions in `generate`, logs the cause only when request logging is enabled, and raises a fresh `EngineGenerateError()` from the original cause. That exception has an empty string representation. `entrypoints/serve/utils/error_response.py` consequently produces the observed empty-message HTTP500.

This explains all supplied observations: renderer-only requests succeed; actual completions fail for both strings and token IDs; no request reaches EngineCore; the error JSON message is empty despite the underlying Python error having useful text. The nested Gemma4 configuration and adapter forward stubs are not implicated by this failure.

## Discriminating host reproduction

Called the actual TT validator with the installed caller's two-argument shape inside a fake `AsyncLLM.add_request` coroutine, then consumed the real installed `AsyncLLM.generate` method and passed its exception to the real error-response converter. This used no engine or server.

Observed output:

```text
wrapper: EngineGenerateError ''
cause: TypeError TTPlatform.validate_request() missing 1 required positional argument: 'processed_inputs'
api_body: {'error': {'message': '', 'type': 'InternalServerError', 'param': None, 'code': 500}}
```

The host reproduction exactly matches `readiness_vllm/request_error.json`.

## Minimal repair and verification

Make the third `processed_inputs` argument optional with default `None`. The validator currently uses only `params` and the platform device name, so both the old three-argument API and current two-argument API retain the same behavior. Keep the `prompt_logprobs` rejection unchanged.

Test the installed `InputProcessor.process_inputs` against the real TT platform validator and verify that it returns an `EngineCoreRequest` with unchanged prompt IDs and sampling parameters. Also test the old/new validator call shapes and prompt-logprob rejection. After these host checks, restart the owned reduced server and rerun the exact saved request. No claim about device inference or downstream request handling follows from fixing admission alone.
