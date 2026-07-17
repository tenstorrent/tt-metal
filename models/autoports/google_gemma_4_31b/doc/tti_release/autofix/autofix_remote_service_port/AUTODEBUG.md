# AutoDebug: v0.18 external-server service port is dropped

## Verdict

The smoke hang is explained by a high-confidence, source-level URL-resolution regression. With the supported CLI pair `--server-url http://127.0.0.1 --service-port 8000`, v2 correctly resolves port 8000 in `MediaContext`, but the LLM performance path discards it because `ServerConnection.url_with_port` returns every `is_remote=True` URL before applying `service_port`. Readiness therefore polls `http://127.0.0.1/v1/models` (implicit port 80), and the vLLM benchmark command would target the same wrong base URL if readiness were bypassed.

The full release has two additional instances of the same contract drift: standard-eval and agentic-eval readiness pass raw `ctx.server_url` directly to `RemoteOpenAIController`, also bypassing `ctx.base_url` and port 8000. Fixing only `ServerConnection` would unblock the benchmark smoke but leave later release children pointed at port 80.

No hardware or live-server requests were made during this inspection.

## Evidence-ranked findings

### 1. Confirmed release blocker: `ServerConnection.url_with_port` suppresses the fallback port for every external server

Confidence: high. This accounts for the observed smoke hang end to end.

1. Both outer and v2 CLIs explicitly define `--server-url` as a host/base URL that is combined with `--service-port` unless the URL has its own port:
   - `run.py:154-164` says to use `--server-url` together with `--service-port`.
   - `tt-inference-server-v2/run.py:130-141` says to combine them unless the URL carries an explicit port.
2. `normalize_server_url` (`utils/url_helpers.py:28-47`) normalizes scheme and trailing slash but intentionally does not add a port.
3. Any nonempty `--server-url`, including `http://127.0.0.1`, sets remote mode (`utils/url_helpers.py:63-74`). Here “remote” means externally managed/already running; it does not mean “ignore `service_port`.”
4. `MediaContext` resolves the pair correctly:
   - `base_url` calls the canonical `build_base_url` helper (`test_module/context.py:54-56`).
   - `server_port` uses an explicit URL port when present and otherwise falls back to `service_port` (`context.py:65-69`).
   - Existing tests encode both sides of this contract: explicit `:9000` wins, while a URL without a port receives `service_port=8001` (`tests/test_module/test_context.py:55-64`).
5. `run_llm_performance` then rebuilds a connection from raw `ctx.server_url`, resolved `ctx.server_port`, and `is_remote=True` (`llm_performance_tests.py:63-70`).
6. `ServerConnection.url_with_port` immediately returns the raw host when `is_remote` is true (`llm_module/config.py:63-72`). Its explicit-port/fallback logic is unreachable in this mode.
7. The resulting URL is consumed twice:
   - `RemoteOpenAIController` receives it (`llm_performance_tests.py:75-80`) and appends `/v1/models` (`server_control.py:127-147`).
   - The vLLM driver passes it as `vllm bench serve --base-url` (`llm_module/drivers/vllm.py:85-87`).
8. A side-effect-free construction probe reproduced the exact divergence:

   ```text
   input:              server_url=http://127.0.0.1 service_port=8000 remote=True
   MediaContext:       base_url=http://127.0.0.1:8000 server_port=8000
   ServerConnection:   url_with_port=http://127.0.0.1
   readiness target:   http://127.0.0.1/v1/models
   ```

Port 80 is therefore not a symptom inference; it follows directly from standard HTTP URL semantics after the code omits `:8000`.

Introduction evidence: `git blame` attributes the early `if self.is_remote: return host` branch to `695d263d5` (“route LLM evals/benchmarks through tt-inference-server-v2,” 2026-07-08). Before that change, the property always added `service_port`; the same change also introduced the external-server performance route. The change added a remote-console test with an already explicit `https://...:443` URL, so it did not exercise the documented URL-without-port form.

### 2. Confirmed subsequent release blocker: eval readiness independently bypasses the canonical base URL

Confidence: high. This is not needed to explain the benchmark smoke hang, but it will affect later children of the requested `release` workflow.

- Standard LLM eval readiness constructs `RemoteOpenAIController(base_url=ctx.server_url)` in `test_module/llm_tests/llm_eval_tests.py:301-305`.
- Agentic readiness does the same in `agentic_eval_tests.py:92-96`.
- For this invocation, raw `ctx.server_url` is `http://127.0.0.1`, while canonical `ctx.base_url` is `http://127.0.0.1:8000`.
- `RemoteOpenAIController` has no separate `service_port` field, so it cannot recover the dropped port.
- The standard eval command itself separately receives `ctx.server_host` plus `ctx.server_port`, so the concrete defect at this boundary is the readiness probe. The agentic driver constructs a split host/port `ServerConnection`; its independent readiness probe is still wrong.

Thus a narrow change to `ServerConnection.url_with_port` is necessary for benchmarks but insufficient for a complete external-server release.

## Existing coverage and test gap

The current focused URL suites pass (`25 passed`) while missing the failing combination:

- `tests/llm_module/test_config.py` tests local/non-remote URLs without a port but never `is_remote=True` without an explicit port.
- `tests/llm_module/test_vllm_driver.py::test_remote_console_uses_base_url_and_skips_ready_check` uses `https://console.tenstorrent.com:443` and `service_port=443`; the explicit port masks the bug.
- `tests/test_module/test_context.py` already proves the intended precedence, but no integration-level assertion carries that resolved context through performance/eval readiness.

## Smallest intervention boundary

Preserve `remote_server=True`: it correctly selects external-server readiness (`/v1/models`), remote driver flags, and no local server ownership. The bad behavior is port composition, not mode selection.

The focused repair should:

1. Make `ServerConnection.url_with_port` follow the shared `build_base_url` rule for remote and local connections: an explicit URL port wins; otherwise append `service_port`. Reusing the helper avoids another local copy of the precedence rule.
2. Pass `ctx.base_url`, not raw `ctx.server_url`, to `RemoteOpenAIController` in standard and agentic eval readiness.
3. Avoid appending a second port when the URL already contains one. Existing explicit-port behavior must remain intact.

An alternative of changing remote detection is not recommended: an already-running localhost server is still external to TTI and needs remote-controller/driver behavior even though its host is loopback.

## Focused verify/refute plan

No accelerator or live endpoint is required.

1. Add a parameterized `ServerConnection` unit test covering:
   - remote `http://127.0.0.1` + 8000 -> `http://127.0.0.1:8000`;
   - remote `http://host:9000` + 8000 -> `http://host:9000`;
   - existing remote console `https://host:443` + 443 remains unchanged;
   - local behavior remains unchanged.
2. Add a performance-path test that captures the generated controller and driver connection for `ctx.server_url=http://127.0.0.1`, `service_port=8000`, `remote_server=True`; assert both readiness and driver receive `http://127.0.0.1:8000`.
3. Add standard-eval and agentic-readiness tests that monkeypatch `RemoteOpenAIController` and assert its `base_url` is `ctx.base_url`, including the URL-without-port case.
4. Run focused CPU-only suites:

   ```text
   pytest -q \
     tt-inference-server-v2/tests/llm_module/test_config.py \
     tt-inference-server-v2/tests/llm_module/test_vllm_driver.py \
     tt-inference-server-v2/tests/test_module/test_context.py \
     tt-inference-server-v2/tests/test_module/llm_tests/test_llm_eval_tests.py \
     tt-inference-server-v2/tests/test_module/llm_tests/test_agentic_eval_tests.py
   ```

5. Re-run the side-effect-free construction probe and require:

   ```text
   server.url_with_port=http://127.0.0.1:8000
   controller.models_url=http://127.0.0.1:8000/v1/models
   ```

6. Only after those isolated checks pass, retry the no-Docker TTI smoke against the already-running autoport server. A request appearing on port 8000 would validate the final process boundary; hardware is not needed to establish the source-level fix itself.

## Claim review

- The primary finding explains every reported condition: raw URL without a port, separate service port 8000, `remote_server=True`, readiness polling port 80, and no benchmark request reaching the intended server.
- Explicit-port behavior is not diagnosed as broken: construction with `http://127.0.0.1:9000` preserves 9000 today, and the proposed precedence preserves it.
- The eval finding was kept as a separate confirmed downstream blocker because it does not cause the initial smoke hang, but its raw-URL flow is concrete and would survive a benchmark-only fix.
- No speculative hardware, model, network, or authentication failure is needed to explain the observation.
