# Bounded single-prefill profiler capture

Use the new model-local `tests/profile_single_prefill_fused_snapshot.py` for a **first-call/cold full-prefill operation table**, followed by the unchanged warmed traced-decode capture. This is a source-supported capture-size hypothesis, not a validated profiler repair. The helper does not claim warmed prefill timing. The separate clean benchmark continues to supply warmed prefill/decode latency; parent reported 8cda S256 warmed prefill 1055.736 ms versus 700c 2370.056 ms and traced decode 2.07353 ms. Those parent results are context only, not measurements made by this source-only task.

The parent explicitly selected this cold-table strategy after the distinction was reported. The alternative of silently describing the first call as warmed is rejected. No quality threshold, source-location limit, trace completeness requirement, or report-overlap gate is weakened.

## Why the previous controls are insufficient

- `run_decoder.py` under `profile=True` already runs exactly **two** full prefills: first correctness/compile, then reset plus `PERF_PREFILL` measurement. `--prefill-repeats` and `--decode-repeats` affect only its non-profile branch; decreasing them cannot shrink a profile capture.
- Native `RealtimeProfilerTracyHandler::make_marker` (`tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp:39`) creates `Program_<runtime_id>` names, and `HandleRecord` pushes them directly at lines 212–254. Its construction in `distributed/realtime_profiler_manager.cpp:909` does not consult the legacy push-disable option.
- `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1` only disables the legacy per-core producer at `impl/profiler/profiler.cpp:2663`. It is retained in the proposed command to avoid that additional producer, but it does **not** bound realtime names by itself.
- `TracyWorker.cpp:3496` and `:3893` enforce signed-16-bit source-location capacity. Repeated drains free device marker capacity, not the capture's accumulated location table. Raising `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` likewise does not raise that table limit.
- `TT_METAL_STREAMING_PROFILER` makes realtime eligibility stand down, but selects a different profiler/data path. It is not an ordinary off switch for this validated report workflow and is not proposed here. `tracy.Profiler.disable()` disables Python tracing hooks, not native realtime emission. `SuppressEmit` is an internal suppression mechanism, not a supported reversible Python warmup/capture control.

## Is fused matmul alone likely enough?

The rejected S64 capture contains **27,513 rows before `PERF_PREFILL`**, a complete initial correctness prefill plus setup. It contains **768 corrected per-K GenericOps**, matching eight matrix K lengths `[128,64,64,128,64,128,128,64]`. Most consecutive FMA calls have 11 operations between them, including slice/untilize/tilize, broadcast, and sharding adapters. The new fused matmul executes one GenericOp for each matrix.

The retained CSV's first-through-last-FMA spans account for 8,304 rows across the eight matrices. Replacing those spans with eight calls gives a **19,217-row estimate** for that setup plus first prefill, before further savings from removed matrix initialization/boundary operations. Thus two comparable eager prefills remain approximately 38K operations and are still risky; the reduced graph alone is not a convincing repair for the old two-pass capture. The large remaining section is the explicit triangular-state update, including its prescribed reductions; no arithmetic is removed by this task.

This is an operation-count estimate, **not** a measured internal Tracy location count or formal upper bound. It does not add new compiler/static locations and decode work, or subtract every removed boundary adapter. Realtime names and source payloads, setup, and cache behavior still require a real capture to establish feasibility. The single-prefill strategy removes one complete ~19K-op invocation and therefore has substantially more headroom. Native trace dispatch uses `trace_node.program_runtime_id` (`impl/program/dispatch.cpp:2973`); trace replays reuse that captured ID, and Tracy deduplicates identical source payloads. The required 20 decode sessions do not imply 20 newly named copies of every operation. All 20 remain mandatory.

The exact retained-CSV census and limitations are in `v6_profile_fused_matmul_location_estimate.json`. The old capture remains rejected and is never used for phase timing or top-operation conclusions.

## What the new helper changes

The new helper statically copies the original harness and applies only these profiling changes:

1. Put `PERF_PREFILL` and `PERF_PREFILL_END` around the **first complete** `dec.prefill_forward`, with synchronization and drains. Retain the old profiling-only drain after each existing prefill chunk; no plan, sequence, shape, or tensor path changes.
2. Reuse that same full output for every original initial prefill PCC/contribution/state check. Replace the second profile-only full prefill with the original additional PCC check on the completed output and explicit cold-regime metadata.
3. Leave the entire state-snapshot-through-device-close AST identical: eager decode correctness, trace capture, three deterministic correctness replays, one warm replay, **16 measured replays**, all decode checks, runtime audit, trace release, and final device close.

All four signpost names remain exact, one-argument `signpost` calls. Host readback is outside measured windows. The wrapper still hashes the selected decoder source, records its own hash, loads the explicit candidate, and restores its temporary chunk-drain wrapper. It records `profile_warmed_latency_claim=false` and `profile_prefill_passes=1` in output/provenance. No runtime AST or source transformation is needed in this helper. No existing helper, snapshot, production/native/compiler source was changed.

## Parent command and artifacts

Prepared command, **not executed by this agent**:

```bash
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_compact_single_prefill_s64.sh
```

It targets the frozen compact/direct-coefficient/fused-matmul candidate `v6_corrected_fma_full_v5_direct_coeff_compact_silu_fused_matmul_integration.py.snapshot` (SHA256 `5ff93fd27348d9e49b6c09da6fd77a362243c7c575b4b92431232638247de242`), layer 0, full logical S64/context512, in a fresh `v6_profile_compact_single_prefill_s64` directory. It uses the existing Tracy CLI with `--no-op-info-cache`, retains native device CSV dumps and realtime profiling, disables only legacy device GUI push, and removes watcher/allocation-tracking environment switches. The script refuses to overwrite existing evidence. Other representative layers can use the same helper with separate fresh output directories.

Frozen helper SHA256: `c16d7b30bf83c4f2c2a4e8837562c15eb41ac96623255198dba034201e7d2617`.

Artifacts:

- `tests/profile_single_prefill_fused_snapshot.py` and `v6_single_prefill_profile_helper.py.snapshot`: exact new static helper.
- `v6_single_prefill_profile_source_proof.json`: original harness/wrapper hashes, full original-run AST restoration, unchanged decode-tail proof and regime/trace contract.
- `v6_single_prefill_profile_builder.py.snapshot`: CPU-only generator used to prepare the helper; this generator is not part of hardware runtime.
- `v6_profile_fused_matmul_location_estimate.json`: source-hashed old CSV census, matrix intervals and estimate limitations.
- `v6_profile_compact_single_prefill_s64.sh`: exact parent-owned capture command.

CPU validation completed: Python compile, exact original-run AST restoration before formatting, exact decode-tail AST comparison after formatting, four exact signpost calls, and `bash -n`. No hardware was opened.

Acceptance remains unchanged: complete start/end windows, **20 identical total decode sessions and 16 measured**, positive durations, no device drops, no instrumentation failure, zero report overlap, actual fused source hash, all original correctness/state gates, clean device close, then prefill/decode tt-perf-report tables and CSV. Do not publish a partial table if the hypothesis fails. Label accepted prefill tables first-call/cold and keep them separate from clean warmed latency.

Untried fallback: capture/replay a complete prefill to reuse runtime IDs could retain a warmed prefill window, but requires separate trace-safety/allocation validation and report logic distinguishing prefill and decode trace IDs. It is not implemented or claimed correct here. Splitting an eager logical prefill into independently reported fragments would not satisfy the full measured-window contract and is not proposed.
