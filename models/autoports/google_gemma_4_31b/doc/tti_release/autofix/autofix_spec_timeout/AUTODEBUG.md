# AutoDebug: final4 vLLM parameter-conformance timeouts

## Scope and starting evidence

- Diagnosis-only investigation of the six final4 spec-test failures in
  `tti_release_final4.log` and the generated release report.
- Failing child suite: `tt-inference-server-v2/llm_module/test_vllm_chat_completions.py`.
- Result: `6 failed, 15 passed in 5759.51s`; every failure is
  `requests.exceptions.ReadTimeout` from `127.0.0.1:8000` with
  `read timeout=30`.
- No implementation files were edited.

## Evidence-ranked findings

### 1. Verified root cause: a fixed 30-second harness timeout is applied to model-speed-dependent conformance requests

Confidence: **high**.

`server_tests/conftest.py:124-148` supplies the fixture used by the v2 suite.
Its `_make_request(..., timeout=30, ...)` forwards that value directly to
`requests.post`. The final4 tracebacks show the same value all the way through
Requests/urllib3 (`Timeout(connect=30, read=30)`), proving the failure is a
client-side read deadline rather than an HTTP error or assertion failure.

The six failed test instances are exactly the calls that exceed this deadline:

- `test_seed_reproducibility` (1 instance);
- `test_non_uniform_seeding` (1 instance, 32 concurrent 50-token requests);
- `test_logprobs` (1 instance, 100 output tokens);
- `test_determinism_parameters` (3 parametrizations).

The controlled contrasts agree:

- All 15 other instances passed.
- All 9 penalty instances explicitly call `api_client(..., timeout=None)` and
  passed, including 1024-token generations that are much longer than 30 seconds.
- The server log shows forward progress throughout the failing interval, including
  32 live requests for the concurrent seed test and steady generation throughput;
  it contains no engine-death traceback, HTTP 5xx, or device fault for this run.
  Requests are released after their clients time out and later requests continue.

This is a test-harness false negative. The six timeouts do not reach their
parameter assertions, so they provide no evidence of a server/model correctness
bug. The corrected live rerun must still prove those assertions.

Four failing instances intentionally omit `max_tokens`, and vLLM consequently
allows generation up to the remaining context window. That is a valid API
request and part of the suite's existing semantics, not a model defect. It also
is not needed to establish the timeout bug: the concurrent seed test explicitly
requests 50 tokens and the logprobs test explicitly requests 100, yet both still
exceed the same 30-second deadline.

## Correct scoped fix

Change only
`tt-inference-server-v2/llm_module/test_vllm_chat_completions.py`:

1. Pass `timeout=None` explicitly for the seed-reproducibility,
   non-uniform-seeding, logprobs, and determinism API calls, as the penalty tests
   already do. These are functional conformance checks, not 30-second latency
   checks. Preserve each payload exactly, including omitted `max_tokens`; do not
   cap valid request length to make the test faster.
2. Do **not** change the global `api_client` default in
   `server_tests/conftest.py`. Keeping its 30-second default protects unrelated
   short API tests from hanging; the long/model-speed-dependent suite should opt
   out locally.

No autoport model, generator, adapter, server, context contract, service port, or
benchmark/eval configuration should change for this symptom.

## Regression tests and verification

Add a cheap source/mocked unit test around the chat-conformance functions that
records fixture calls and verifies:

- every affected call passes `timeout=None`;
- non-uniform seeding still issues 32 requests with `max_tokens == 50` and the
  expected seed pattern;
- logprobs retains its 100-token payload and validates returned logprob content;
- seed-reproducibility and determinism retain their original payloads without an
  injected `max_tokens` cap;
- the shared `api_client` default remains 30 for unrelated callers.

Then run, in order:

1. The new mocked test plus the existing v2 conformance-wrapper unit tests.
2. The original focused child pytest command against the live autoport server;
   require all 21 instances to pass and inspect the parameter report for actual
   assertion results (not merely absence of timeout).
3. `run.py --workflow release --no-auth` with the same no-Docker autoport spec;
   require the combined report's spec-test and acceptance gates to pass.

## Final status

**Root cause verified; implementation fix not applied in this diagnosis-only
pass.** The evidence ranks the TTI client/test contract as the intervention
boundary. There is no evidence of a server/model bug in these six failures, but
the fixed live suite is required before declaring the six parameter behaviors
conformant.
