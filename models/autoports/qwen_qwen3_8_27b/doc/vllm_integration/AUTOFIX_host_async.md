# AutoFix: explicit host compatibility under async decode

## Starting evidence

- Fresh source-only diagnosis: `AUTODEBUG_host_async.md`, written before
  reproduction or implementation edits.
- Actual failure: `host_async_failure.log`, 2026-09-14 00:55:21. A `min_p`
  request selected explicit host compatibility; async submission then sent
  already-materialized Torch logits into `ttnn.get_device_tensors`.
- Scope: only adapter `read_decode_output` and `process_decode_output_host`,
  plus a separate host test module. No device jobs, server operations,
  dependency installation, shared-plugin edits, or commits by this agent.

## Hypothesis experiment

**Hypothesis:** completed Torch logits need an explicitly gated identity path
at the adapter output boundary. No extra read/event/conversion is necessary.
The native TTNN token output path should remain unchanged.

Focused experiment, from the tt-metal root:

```bash
PYTHONPATH="/home/mvasiljevic/qwen38-full-rerun/vllm:$PYTHONPATH" python -m unittest models.autoports.qwen_qwen3_8_27b.tests.test_vllm_host_async -v
```

The first attempt without vLLM on `PYTHONPATH` failed at import; this environment
setup result is preserved in `host_async_import.log` and is not a code
reproduction. Adding the existing checkout path required no installation.

**Before fix:** `host_async_before.log` preserves the six-test run. Four test
methods failed, with unittest reporting three failures and three errors due to
subtests. The actual plugin `TTAsyncDecodeController.submit_decode` reproduced
the exact `get_device_tensors(... torch.Tensor)` TypeError. Native async token
handling and rejection of device output as logits were passing controls.

**Verdict: verified.**

**Retained fix:** nine added lines in the two adapter methods:

- Torch outputs in `read_decode_output` require
  `QWEN_VLLM_HOST_COMPATIBILITY=1`, then return the same tensor synchronously
  or `(same_tensor, [])` asynchronously.
- Torch logits in `process_decode_output_host(..., is_tokens=False)` use the
  same explicit guard and return the same tensor, preserving shape, values,
  and dtype.
- The existing TTNN branches are unchanged. No new sampler, host-logits
  transfer, token transfer, trace operation, or synchronization was added.

**After fix:** the same command passes **6/6 tests**, recorded in
`host_async_after.log`. The tests cover:

- Actual plugin submit/finalize around a stubbed forward returning CPU logits:
  original tensor identity retained, no TTNN conversion or event operations,
  no token-readback counter increment.
- Direct synchronous/asynchronous read and host-processing methods.
- Compatibility-disabled rejection at both boundaries.
- Device output rejected when incorrectly labeled as host logits.
- Native async read control with compatibility both off and on: one replica
  receives `.cpu(blocking=False)`, one event is recorded and waited before
  conversion, and one token-readback is counted. The mocked native payload is
  32 four-byte token IDs (128 bytes); this is an interface control, not a new
  hardware traffic measurement.

Additional checks passed:

```bash
python -m black --target-version py312 --check models/autoports/qwen_qwen3_8_27b/tests/test_vllm_host_async.py models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py
python -m py_compile models/autoports/qwen_qwen3_8_27b/tests/test_vllm_host_async.py models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py
git diff --check
```

## Final status

The reported output-boundary defect is fixed and verified on host. The parent
was notified immediately after the focused pass and owns the reduced-server
`min_p` rerun plus broader sampling checks. This investigation did not rerun
the original server test or measure device correctness, output quality, or
performance. Existing native read operations and their ordering were retained;
the tests do not replace real-silicon async validation.
