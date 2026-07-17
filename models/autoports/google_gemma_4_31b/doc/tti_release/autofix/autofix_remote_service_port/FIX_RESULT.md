# AutoFix result: external-server service-port propagation

## Verdict

The focused hypothesis is verified and fixed in the pinned TTI checkout at starting HEAD `f1a89cb4bf8a9931be2e7ad3f5fa4db6d9222169`. External servers now use the same canonical explicit-port-wins rule as local servers: `--server-url http://127.0.0.1 --service-port 8000` resolves to `http://127.0.0.1:8000`, while an explicit URL port remains unchanged. Standard and agentic eval readiness now consume `MediaContext.base_url`, so their probes preserve the resolved service port too.

No hardware, live-server endpoint, container, or accelerator command was used. The changes are uncommitted as requested.

## Focused implementation

- `tt-inference-server-v2/llm_module/config.py`
  - Removed the remote-only early return from `ServerConnection.url_with_port`.
  - Reused `utils.url_helpers.build_base_url` after the existing scheme normalization. This keeps one precedence rule for remote and local paths and prevents double-port construction.
- `tt-inference-server-v2/test_module/llm_tests/llm_eval_tests.py`
  - Standard-eval `RemoteOpenAIController` now receives `ctx.base_url`.
- `tt-inference-server-v2/test_module/llm_tests/agentic_eval_tests.py`
  - Agentic readiness `RemoteOpenAIController` now receives `ctx.base_url`.

Focused regression coverage was added for:

- remote and local no-port fallback (`:8000` is added);
- remote and local explicit-port preservation (`:9000` wins);
- the performance bridge's controller and driver connection;
- standard-eval remote readiness;
- agentic-eval remote readiness.

## Verification evidence

The requested neighboring CPU-only suites passed:

```text
PYTHONPATH=tt-inference-server-v2:$PWD python3 -m pytest -q \
  tt-inference-server-v2/tests/llm_module/test_config.py \
  tt-inference-server-v2/tests/llm_module/test_vllm_driver.py \
  tt-inference-server-v2/tests/test_module/test_context.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_llm_performance_tests.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_llm_eval_tests.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_agentic_eval_tests.py

81 passed, 1 warning in 0.29s
```

The warning is the pre-existing pytest collection warning for the imported enum-like `TestStatus`; there were no test failures.

The exact side-effect-free construction repro now reports:

```text
ctx.base_url=http://127.0.0.1:8000
server.url_with_port=http://127.0.0.1:8000
controller.models_url=http://127.0.0.1:8000/v1/models
```

Additional checks:

```text
python3 -m compileall -q <all changed Python files>  # passed
git diff --check                                      # passed
```

`black --check` passed for five of the seven selected files and reported two already-unformatted neighboring test files. Its diff included pre-existing formatting in both files as well as one newly added multi-context-manager block; no functional or syntax issue was reported. The focused pytest, compile, and diff checks all passed.

## Resulting process boundary

The corrected path is now:

```text
server_url=http://127.0.0.1 + service_port=8000
  -> MediaContext.base_url=http://127.0.0.1:8000
  -> ServerConnection.url_with_port=http://127.0.0.1:8000
  -> performance readiness and driver base URL use port 8000
  -> standard and agentic eval readiness use port 8000
```

An explicit URL such as `http://host:9000` remains `http://host:9000` even when `service_port=8000`.
