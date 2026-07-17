# AutoFix Report: vLLM parameter-conformance request timeouts

## Starting Evidence

- Diagnosis: `AUTODEBUG.md` in this directory.
- Original failure: the final4 vLLM parameter-conformance child suite reported
  six `requests.exceptions.ReadTimeout` failures with `read timeout=30` while
  the autoport server continued making progress.
- Verified intervention boundary: the affected long/model-speed-dependent
  calls in
  `tt-inference-server-v2/llm_module/test_vllm_chat_completions.py` inherited
  the API fixture's 30-second default. Existing 1024-token penalty checks
  already opted out locally with `timeout=None` and passed.

## Hypothesis Experiment

- Hypothesis: opting out of the client read deadline only at the six affected
  conformance call sites removes the harness false negative without changing
  request semantics or weakening the timeout for short tests.
- Experiment: add mocked call-recording regressions for seed reproducibility,
  32-request non-uniform seeding, logprobs, all three determinism parameter
  paths, and one short `n` request.
- Result: verified. The mocked tests prove every affected call passes
  `timeout=None`; payloads retain their original fields and values; the
  seed/determinism requests still omit `max_tokens`; non-uniform seeding still
  sends 32 requests with the original seed pattern and 50-token cap; logprobs
  retains its 100-token cap; and the short request still supplies no timeout
  override.
- Fix: add `timeout=None` only to both seed-reproducibility calls, the threaded
  non-uniform-seeding call, the logprobs call, and both determinism calls. The
  shared API-client default and all payloads are unchanged.

## Verification

Focused regression:

```text
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml \
  --rootdir . --confcutdir . \
  tt-inference-server-v2/tests/llm_module/test_vllm_chat_completions_timeouts.py -q
5 passed in 0.12s
```

Broader unit regression:

```text
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml \
  --rootdir . --confcutdir . \
  tt-inference-server-v2/tests/llm_module \
  tt-inference-server-v2/tests/test_module/llm_tests/test_vllm_param_conformance_tests.py -q
66 passed in 0.25s
```

Live isolated regression against the external autoport server:

```text
test_vllm_chat_completions.py::test_logprobs
1 passed in 65.19s (0:01:05)
```

The unchanged 100-token request ran past the former 30-second read deadline and
then passed its real logprobs-content assertions, directly confirming the
scoped timeout fix without reducing the request.

Formatting and patch checks:

- `black --check` passes for the new mocked regression file.
- `git diff --check` passes for the complete patch.
- The committed conformance source has pre-existing formatting differences
  under the locally installed Black 26.3.1 (assert-expression rewrites outside
  this fix); those unrelated lines were intentionally not reformatted.

## Final Status

The scoped harness fix is implemented, unit-verified, and isolated-live-verified. No model, generator,
adapter, context, server, benchmark/eval configuration, request payload, or
global timeout was changed. The combined release rerun remains required to
prove every real server assertion and the final release gate.
