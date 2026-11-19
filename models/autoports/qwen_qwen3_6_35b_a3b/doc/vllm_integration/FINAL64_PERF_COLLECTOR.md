# Final64 performance collector

`run_final64_perf.py` is now executing for final acceptance. The parent first starts an otherwise idle final-code server with `QWEN_VLLM_AUDIT=0`, `TT_METAL_TRACE_ALLOC_TRACKING=0`, selected precision, capacity32, context262144, retained B2/B32 model traces and native32/native64 sampling traces. Watcher/profiler must be off. Keep the adapter's compact `QWEN_DECODE` route ledger enabled.

From tt-metal after activating the normal Python/lane environment:

```bash
export PYTHONPATH="$TT_MODEL_BRINGUP_ROOT/runtime:$PYTHONPATH"
python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/run_final64_perf.py \
  --server-log ACTUAL_FINAL64_PERFORMANCE_SERVER_LOG \
  --server-url http://localhost:8000 \
  --overall-timeout 3600 \
  --execute
```

Omit `--execute` for a plan with no HTTP, process execution, output mutation, or server-log access. The fresh default output root is `readiness_vllm/final64_perf/`; an existing root is rejected. An alternate fresh `--output-dir` is supported. No retry/resume or artifact promotion is built in.

The first job copies the baseline `wrapup_execution.json` planned `benchmarks` argv, changing only the Python executable, attached server URL/log and adding the new output directory. It runs the shared `run_vllm_server --stages benchmark,prefill_burst` exactly once, preserving the canonical defaults:

| Output | Input/output tokens | Requests / concurrency | Role |
| --- | --- | --- | --- |
| `vllm_benchmark.json` | 128/128 | 1/1 | Primary single-user serving |
| `vllm_ci_serving_benchmark.json` | 100/100 | 32/unset | Secondary CI serving burst |
| `vllm_prefill_burst_benchmark.json` | 128/32 | 32/32 | Prefill burst |

All use explicit greedy sampling and ignore EOS. Capacity remains32; the primary benchmark has concurrency1. CI's unset concurrency is the existing canonical admission shape, not an explicit concurrency override. The collector checks exact normalized workload shapes, completed request counts, no missing output tokens, and local raw result files. Normalized files retain the shared raw TTFT/TPOT/ITL statistics, throughput, TPOT-derived t/s/u and raw benchmark commands.

The second job runs the existing `sampling_debug/run_matched_b2_http.py` once with saved `serving_overhead/baseline.prompt.json` and `baseline.json`, one warmup and two measured repetitions. It verifies the exact128 prompt/output IDs using the helper's unchanged assertions and saves results/SSE data under `final64_perf/`. This matched fixture supplements the primary canonical benchmark. Its two samples provide descriptive percentiles; native timings remain a lower bound with different measurement boundaries.

`performance_execution.json` records source command provenance, actual effective argv, UTC/elapsed times, raw exits, log byte offsets and copied slices, device/host decode counts, error lines, idle observations, normalized results and baseline path/size/mtime_ns snapshots. It obtains the actual serving engine PID from the adapter initialization event, reads only selected flags from `/proc/PID/environ`, requires tracking disabled (unset or0) and audit0/default-off, and records the process command. Two zero-running/zero-waiting metric scrapes establish idle before and after each job. Detailed per-forward audit activity, any host decode, missing device route evidence, failed/incomplete output, process/HTTP failure or failure to become idle stops subsequent work.

The shared benchmark job is bounded at1800 seconds and the matched HTTP job at900 seconds; the independent3600-second overall budget bounds subsequent work. GNU timeout affects only collector clients and their children, never the separately started server. The collector compares path/size/mtime_ns snapshots of the existing native baseline and top-level canonical `vllm*.json` artifacts before/after; it never overwrites or promotes them. It leaves a successful server idle and sends no server signal/shutdown. The parent reviews results and owns shutdown and any later canonical promotion.

Validation during preparation: Python compile, dry plan using a nonexistent server log, CPU checks of baseline command equivalence/workload validation, and pre-commit. No HTTP, server process inspection, benchmark, or TT operation was performed.

The collector does not calculate individual artifact content digests. Logs and the execution manifest retain evidence paths, raw results, metrics and stat-based mutation checks. After this change, Python compilation was rerun; no collector execution occurred.

Preflight correction2026-11-17: the launch requested tracking0, but the engine environment presented it unset. `tt_metal/llrt/rtoptions.cpp:304,437` enables tracking only for a value beginning with1, so unset also proves disabled. The first collector stopped before any job/request (`final64_perf_preflight_attempt/performance_execution.json`); its artifact is preserved. The corrected preflight accepts unset/0 and rejects enabled values; benchmark commands are unchanged. No server restart or benchmark repeat was needed.
