# AutoDebug: idle() parses Prometheus metadata as numeric samples

## Finding

**The unanchored regex in `tools/benchmark_server.py:47` matches metric names inside `# HELP` and `# TYPE` comments.** Line 48 then attempts `float("Number")`, producing the exact reported exception even when the actual request-count samples are zero.

Paths below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/` unless otherwise stated.

## Direct evidence

- `doc/benchmark/pilot_performance/server-selection.log:1-10` records successful readiness after 144 seconds. Lines 11-22 trace the subsequent failure through `select():224` into `idle():48`, ending with `ValueError: could not convert string to float: 'Number'`.
- `doc/benchmark/servers/b32-1794637295287193086/server.log:12730-12732` records application startup, HTTP 200 for `/health`, and HTTP 200 for `/metrics`.
- `idle():44-48` reads the entire metrics response and applies `re.escape(name) + r"(?:\{[^}]*\})?\s+(\S+)"` without excluding comments or anchoring the metric name to the start of a sample line.
- Installed vLLM source at repository-relative `python_env/lib/python3.12/site-packages/vllm/v1/metrics/loggers.py:508-522` defines both checked gauges; their documentation starts with `Number`. Installed `prometheus_client/exposition.py:336-338` emits HELP and TYPE lines before samples.

A standalone standard-library check extracted the regex suffix from the source AST, without importing or executing the target module. For this synthetic, source-consistent input:

```text
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="example",engine="0"} 0.0
```

The exact regex returns `['Number', 'gauge', '0.0']`; the conversion expression raises the reported exception on its first element. The waiting metric behaves identically. Removing only HELP would still leave `float("gauge")` failing on TYPE.

## Causal interpretation and intervention

With standard exposition as above, `select():224` fails on the first running-gauge observation, before checking the waiting gauge or reaching the second observation. This explains startup success followed by the reported exception without requiring a server or model failure. Selection aborts before configuration/identity publication. The exception has no cleanup path; the server was launched with `start_new_session=True` at line 192, consistent with it remaining running. `stop():61-62` also calls `idle()` before SIGTERM, so the same parsing defect can block an orderly stop.

The smallest correction is to restrict matching to actual sample lines for the exact metric names, excluding all comment lines. If retaining regex parsing, use multiline start anchoring and horizontal whitespace so matching cannot cross lines. Preserve the existing requirement for present samples and zero values across all series. Suggested regression cases: HELP plus TYPE with zero samples; a nonzero series; missing samples; multiple labeled series.

## Limits and review

Rechecked the headline against the traceback, definitions, regex behavior, and caller order; no additional cause is needed to explain this exception. The failing response body was not captured in the inspected artifacts, so its exact contents and actual request counts remain unverified. The synthetic check proves the parser defect, not live idleness. No target execution, server requests, source changes, signals, or device operations were performed; only this report was written.
