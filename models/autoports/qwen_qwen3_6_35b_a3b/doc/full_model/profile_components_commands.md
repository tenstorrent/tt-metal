# Reduced full-model component profile

Final wrap-up: v4 capture, native conversion, all component reports and execution/metadata validation **PASS**; see `performance/profile_reduced_v4_drains/component_summary.json` and its `commands.log`. All40 isolated performance also **PASS** in `performance/lifetime_fixed.json`; reduced Watcher **PASS** in `watcher/summary.json`. No further runs are authorized by the capped plan. The command blocks below preserve the earlier proposed invocations; actual executed commands are recorded in the work log and corresponding logs.

Native profile attempts
`performance/profile_reduced_v1` and `profile_reduced_v2` failed during preparation
with host OOM; the separate Tracy source-location overflow is documented in
`AUTODEBUG_PROFILE.md`. Preserve both directories and their `failure.json` files,
which supersede stale intermediate RUNNING records. The C++ repair making the
real-time Tracy handler honor `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY` has been
built and committed as `958dbadc5c4eb4b6b6ce73f792a0b862ba8f9e35`. V3 subsequently failed marker-loss acceptance; v4 with eager/capture-boundary drains passed. See `AUTOFIX_PROFILE.md`.

Run each hardware command separately, after the active context run has finished
and ownership checks show the assigned mesh is idle. Do not rebuild or replace
runtime libraries under a live workload.

## Fresh native capture

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-35b-a3b/lane-env.sh
TT_METAL_PROFILER_CPP_POST_PROCESS=1 bash \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_full_model_components.sh \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/profile_reduced_v3_midrun
```

The destination must not exist. The collector checks device ownership, preserves
native Tracy/device files and exact commands, bounds capture and conversion to
3600 seconds each, and bounds each report/summary command to 300 seconds. It
performs no retries or resets. It disables watcher and allocation tracking;
those belong to separate correctness runs.

The collector enables device profiling, op metadata, trace tracking, the
5000-program buffer and **`TT_METAL_PROFILER_MID_RUN_DUMP=1`**, while disabling
push-to-Tracy program lanes. Device files and cached op metadata remain enabled.
The command explicitly enables C++ postprocessing; Tracy normally enables it
for this invocation too. The validator requires both mid-run dumping and C++
postprocessing to be recorded as `1` in `performance.json`.

Mid-run dumps write/analyze device records and release accumulated host markers;
merely reading the device ring did not prevent the earlier host OOM. Do not
combine this path with trace-only device profiling or dispatch-core profiling.
Mid-run dumping reduces device-to-device timestamp synchronization accuracy:
use per-rank duration/work attribution, not precise inter-chip CCL skew or
sensitive cross-device event ordering.

The shell rejects instrumentation failures, dropped markers and partial-report
warnings before conversion, even if Tracy exits zero. It preserves nonzero
workload/timeout status. This check runs after capture exits; it is not a live
abort monitor. CPU conversion/summary also inspect capture and postprocess logs
for malformed or missing cached metadata, discarded messages and device ops
without matching host records. A partial native capture is not accepted.

## Workload and acceptance

The profile uses real layers 0/3, full embedding/final norm/LM head, the selected
split greedy sampler, batch 1, context 1024 and the 128-token AIME prefix. Each
component has 16 fenced replays:

| Signpost pair | Measured work |
|---|---|
| `PERF_TOKEN_OUT` / `_END` | Model and sampling traces, resident token feedback, positions 128–143 |
| `PERF_MODEL_ONLY` / `_END` | Model trace over a fixed input token, positions 128–143; not teacher forcing |
| `PERF_SAMPLER_ONLY` / `_END` | Sampling trace over fixed real decode logits |

Profiler dumps are outside each timed sample. Synchronizations and instrumentation
remain inside the fenced diagnostic boundary; these timings are not headline
throughput. Distinct input regimes mean the separate controls are not an additive
decomposition of autoregressive latency.

`profile_full_model_components.py` reuses stage-5 native compaction helpers and
unchanged Tracy `process_ops`. Acceptance requires a completed PASS workload,
unchanged program-cache entries, the expected trace IDs, four ranks, and 16
matching replay sessions per component/rank. It compares converted execution
UIDs against native `raw/.logs/cpp_device_perf_report.csv`, rejecting duplicates,
missing or extra executions, including operators missing from every converted
replay. It also requires identical complete operation signatures across replays.

For every selected `(device, program hash)`, native JSON from
`raw/.logs/tracy_ops_data.csv` must supply consistent cached metadata. Validation
checks complete, untruncated operands, shapes, dtype, layout and storage/memory
against converted rows, plus the full native kernel inventory. Every compute
kernel requires source, name and math fidelity. **An empty fidelity set is valid
only when native metadata explicitly lists zero compute kernels and a nonempty
data-movement kernel list.** An operation name or blank CSV cell is not an
exemption. Missing or ambiguous cache metadata fails acceptance.

Outputs retain their distinct purposes:

- `ops.csv`: native converted operation rows, unchanged by summary attribution.
- `native_program_metadata.json`: all validated operands and kernel records,
  including the complete compute-fidelity set.
- `component_summary.json`: execution coverage, source hashes, artifact paths,
  per-rank model/sampler attribution and instrumented host timing.
- `*_op_groups.csv`: operation groups with validated shapes, dtypes, memory,
  program identity and the full compute-fidelity set.
- `*_perf_report.{csv,txt}`: `tt-perf-report` tables for each signposted component.

Kernel-duration sums describe work, not elapsed latency. Four parallel ranks
must not be summed into latency. The summary reports the rank with the largest
combined model/sampler kernel sum and its sampler share; concurrency prevents
interpreting that share as an exact wall-time fraction. Reduced-layer attribution
cannot substitute for all-40-layer token-out performance or its sampler share.

Historical CPU-summary invocation (not a request to repeat it under the cap); v4 summary has already completed:

```bash
timeout 300 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.profile_full_model_components \
  --mode summarize \
  --artifact models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/profile_reduced_v3_midrun
```

## Separate unprofiled teacher forcing

The supplemental `measurements.teacher_forcing_logits_only` path passed the final all40 run:99steps at24.600471626ms,40.649627178t/s/u. See `performance/lifetime_fixed.json`; the exact executed invocation is in `work_log.md`. The same unprofiled performance harness runs the existing
128-input/128-output token-out measurements and then this separate measurement;
there is no teacher-only switch. Historical proposed command:

```bash
env -u TT_METAL_WATCHER -u TTNN_OP_PROFILER \
  -u TT_METAL_PROFILER_TRACE_TRACKING \
  TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance \
  --context 1024 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/final_source_with_teacher.json
```

This all-40-layer measurement consumes 99 actual HF continuation tokens after
the full 161-token reference prompt. It reuses the existing model trace, retains
distinct UINT32 host inputs through the final synchronization, and advances
device positions from 161 to 260. Timing includes forced HF token uploads, model
trace enqueues, audit overhead and final synchronization. It excludes prefill,
reset, input preparation, initial position upload, sampling, generated-token
feedback and per-token readbacks. Final input/position checks occur after timing.
Context must be at least 260. Reference provenance comes from the exact-match
AIME cache loader. This is distinct from both fixed-token `model_only` and the
packaged teacher-forcing gate, which still executes sampling before forcing the
next input.

## Separate watcher replay

This is a reduced real-layer replay diagnostic, not a performance run. Keep
watcher logs in a fresh isolated `TT_METAL_LOGS_PATH`, retain allocation tracking
and program-cache checks, and disable profiling:

```bash
qwen_watcher_artifact="$PWD/models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/watcher_replay_v1"
mkdir "$qwen_watcher_artifact"
mkdir "$qwen_watcher_artifact/runtime"
env -u TTNN_OP_PROFILER -u TT_METAL_PROFILER_TRACE_TRACKING \
  TT_METAL_DEVICE_PROFILER=0 TT_METAL_WATCHER=10 \
  TT_METAL_LOGS_PATH="$qwen_watcher_artifact/runtime" \
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_generator_replay_probe \
  --section all --context 1024 --steps 4 \
  --output "$qwen_watcher_artifact/result.json" \
  > "$qwen_watcher_artifact/run.log" 2>&1
```

Preserve the result, exact invocation and watcher files. Inspect watcher findings as
well as process status; file presence or a successful open/close is not evidence
that replay checks passed. The actual completed Watcher invocation used the artifact paths in `watcher/summary.json`, with90clean dumps across4chips; the historical example above is not its exact command. This documentation update ran no hardware commands.
