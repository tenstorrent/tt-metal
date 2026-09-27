# Request admission repair

2026-09-27. Status: **host-verified**; live request retry belongs to the main stage's reduced-server validation. No server calls or device operations were performed by this investigation.

## Diagnosis and change

[AUTODEBUG_request.md](AUTODEBUG_request.md) identifies the exact blank-HTTP500 cause: installed vLLM0.26's `InputProcessor` calls `TTPlatform.validate_request(processed_inputs, params)`, but the older TT plugin required a third argument. `AsyncLLM.generate` wraps the resulting `TypeError` in an empty-message `EngineGenerateError` before the request reaches EngineCore.

Changed `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` so the third `processed_inputs` argument defaults to `None`. The existing validation body is unchanged. Both legacy `(prompt, params, processed_inputs)` and current `(processed_inputs, params)` callers retain the same sampling checks, including rejection of prompt logprobs.

Added `vllm/plugins/vllm-tt-plugin/tests/test_request_validation.py`. It exercises the real installed `InputProcessor.process_inputs` call, checks returned prompt IDs and sampling settings, tests both platform API shapes directly, and checks prompt-logprob rejection through both API shapes. The input-processor test replaces unrelated model-configuration validators with no-ops; the actual platform hook and request construction remain real.

## Verification

Before the fix, the five focused tests produced **3 failed, 2 passed**. The real input-processor test failed at installed `vllm/v1/engine/input_processor.py:296` with the diagnosed missing-argument `TypeError`. Legacy calls passed.

After the fix:

```bash
python -m pytest vllm/plugins/vllm-tt-plugin/tests/test_request_validation.py \
  -q --disable-warnings --tb=short
```

Result: **5 passed, 2 warnings, 5.39 seconds**.

The separate host reproduction passed the broken validator call through the real `AsyncLLM.generate` wrapper and error-response converter; it yielded exactly `{'error': {'message': '', 'type': 'InternalServerError', 'param': None, 'code': 500}}`, with the missing-argument `TypeError` retained as the cause. This distinguishes the repaired failure from renderer, model, cache, and device failures.

Checks passed:

```bash
python -m black --check --target-version py310 \
  vllm/plugins/vllm-tt-plugin/tests/test_request_validation.py
git -C vllm diff --check
```

No C++ or CMake changed; no build was needed. The saved request still requires a fresh reduced-server retry because an existing API-server process retains the old imported method. Passing admission does not establish successful device generation or complete the serving stage.
