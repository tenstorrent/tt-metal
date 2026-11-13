# AutoDebug: optional grid timeout can abort mandatory native profiles and orphan its client

## Scope and conclusion

Inspection only: read source, Git revision 262d3a3c, existing JSON/log evidence, and `/proc` metadata. No device commands, server/client execution, HTTP probes, resets, or process signals were issued. Only this report was written. The previously present root report concerned response logging; its archived counterpart already exists under `doc/benchmark/setup/`.

**High confidence:** the old grid admits rows without checking whether they fit, then applies its deadline to a wrapper rather than the actual benchmark client. A timeout kills that wrapper, can leave its client running, and fails the native runner before B8/B1. The newer source does not repair the running grid. There is no genuine native resume API. Preserve all completed evidence; allow natural completion if possible, otherwise use a fresh native headline-only attempt within the remaining original deadlines. Do not manufacture completion markers or merge native summaries.

The exact future timeout is a forecast, not an observed failure. At the read-only snapshot (2026-11-13 21:21:39 UTC), PID 4818 and PID 5627 were alive and `run/summary.json` did not exist.

Paths below are relative to `models/autoports/google_gemma_4_31b_it/` unless stated otherwise. `runtime/benchmark_stage/` means the installed package `/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/runtime/benchmark_stage/`.

## Direct evidence and clock ledger

| Boundary | Evidence / value |
|---|---|
| Original stage start | `doc/benchmark/stage_timing.json`: monotonic 1545199.297087911; 20:24:36.917 UTC |
| Running old grid deadline | start + 4800 = monotonic 1549999.297087911; 21:44:36.917 UTC |
| Conservative original native start | `setup/runner_clock.json`: PID 4818 start 1545702.56; approximately 20:33:00.180 UTC |
| Conservative original native end | 1545702.56 + 5400 = 1551102.56; approximately 22:03:00.180 UTC |
| Outer stage end | 1555999.297087911; 23:24:36.917 UTC (3 hours) |

The native implementation starts its timer inside `run()` (run.py:62,82-86), slightly after process start. Use the recorded process-start boundary conservatively; do not claim it is the exact internal timestamp. The gap between the old grid deadline and conservative native end is about 1103 seconds, not a fresh 90 minutes.

`run/run_config.json` requests tasks `[]`, batches `[1,8,32]`, 5400 seconds, the eval client wrapper, server selector, and grid as roofline command. `/proc/5627/stat` confirmed parent 4818 and process group/session 5627; its command was the B32 collect hook.

`run/roofline-b32-collect.log` showed completed seeded rows through the maximum-context row, then the unseeded short full-batch row. The seeded maximum-context raw measurement has 2 completed / 0 failed, input lengths 262016, output lengths 128, duration 502.342607 seconds, and mean end-to-end latency 251.169875 seconds. Warmup plus measurement durations total 753.475831 seconds. The enclosing row consumed about 775.34 seconds according to consecutive grid log remaining-time values. Thus “three calls per row” means three requests (one warmup plus two sequential measured requests), executed by two wrapper invocations, not three subprocess invocations.

The long unseeded row has the same concurrency/request counts. Similar latency is plausible but unverified. Several earlier unseeded rows still remain. The recorded rates support the reported overrun risk; they do not prove its exact arrival time. If the deadline is reached **between** rows, the old code returns normally instead, and the native runner can still proceed.

## Supported causal chain

### 1. Optional work sits on the mandatory critical path without row admission

At **262d3a3c**, `tools/benchmark_grid.py:37-49` collects the headline, reads timing and plan once, sets stage-start + 4800, then checks only whether the deadline has already passed. It does not reserve a complete row or subsequent profile time. Lines 60-93 give each warmup/measured wrapper `timeout=max(60, deadline-now)`. A row can start with insufficient time; the second invocation can even start after the nominal deadline with a new 60-second allowance. This is neither a safe row boundary nor a strict absolute cutoff.

`runtime/benchmark_stage/run.py:141-173` completes B32 headline warmup/measurement, then synchronously waits for the collect hook (which runs the whole optional B32 grid) before selecting B8, then B1. This explains why valid B32 headline results coexist with missing mandatory later profiles.

### 2. Timeout ownership stops one level too early

Process topology:

```
native runner 4818
  grid 5627                      new session/group via run.command()
    benchmark_eval_client.py     direct subprocess.run child
      actual benchmark client    wrapper's subprocess.run child

owned server launcher            separate session; HTTP consumer
  readiness/API/engine processes
```

The old grid's timed `subprocess.run` owns the wrapper only. Python 3.10 `subprocess.py:503-519` kills and waits for that direct process on TimeoutExpired; it does not recursively kill descendants. `tools/benchmark_eval_client.py:85` starts the actual client without timeout or child cleanup. The wrapper normally writes its measured window end only after that client returns (`:87-90`). Killing the wrapper can therefore leave a live client and an absent window-end artifact. Even if that surviving client later saves raw JSON, it does not recreate the wrapper's missing completion-window evidence.

The grid does not catch TimeoutExpired and exits nonzero. Native `run.command()` then follows its ordinary nonzero-exit branch (`run.py:53-54`), which raises **outside** the try/except around `process.wait()`. Consequently its group cleanup at lines 27-52 does not run for this failure. The native runner marks the summary failed and writes its report (`:176-200`), skipping later profiles. This is the key distinction: native timeout/interrupt does trigger group cleanup; a grid that exits nonzero first does not.

The server is separately launched with `start_new_session=True` (`benchmark_server.py:360-363`). Neither client-group termination nor client exit establishes server quiescence or cancels all accepted HTTP work. Actual survival and completion at failure time remain runtime-dependent; source proves the ownership gap, not an already-observed orphan.

### 3. Live source/config edits cannot redirect this invocation

The old grid reads its plan and deadline once. Editing source, timing JSON, or plan now cannot change those in-memory values. Its `stop_after_row` check is only at entry (`old :35-36`), despite the name; it raises an error rather than gracefully completing. Creating that file now would not stop the active loop and would sabotage later collect hooks. Changing `run_config.json` also cannot replace the runner's already-loaded command list.

Current grid source adds estimated-row deferral and `--grid-only`/`--deadline-seconds` (`:30-31,44,59-76`), plus completed-row reuse. Those apply to **new** processes. The override is an offset from the original stage start, not a duration from invocation. It is capped at outer deadline minus 900 seconds, but does not itself know the native runner deadline. The underlying direct-child timeout remains (`:113-120`), so predictions reduce risk without fixing ownership.

## Minimal safe continuation and fallback

These are recommendations to the coordinator, not actions performed here.

1. Preserve the original clock files, runner configuration, launch identity, row predictions, raw results, phase windows, roofline evidence, logs, and failed/partial directories. Do not change clocks, write synthetic `complete.json`, mark budget omissions as hardware capacity failures, or commit runtime changes while an existing server must pass provenance checks. `benchmark_server.py:198-210` checks source hashes and launch HEAD.

2. There is no source-only hot repair for PID 5627. Within this investigation's no-signals constraint, let it reach its natural result and inspect retained evidence. A normal return permits the original native runner to continue. After the old grid cutoff, later default grid hooks still collect their headline first, then skip grid work at the expired cutoff. This is a legitimate path to native completion if B8/B1 startup, measurements, accounting and reporting fit before the original native end. Do not run a supplementary client concurrently or reset its shared observer.

3. If the grid fails, first establish that any actual client descendant has finished and the server has no outstanding requests before any continuation. Preserve the timed-out row as incomplete; raw JSON alone is insufficient. A later device-owning coordinator can verify client ownership, idle server metrics, and healthy observer state using the existing workflow. This investigator did not perform those checks. Do not force a server transition to solve a client timeout: `benchmark_server.py:125-153` refuses unowned devices, nonidle shutdown and ambiguous ownership. It signals only the identified readiness driver after three idle checks. Launcher disappearance with device owners remaining is a recovery condition, not permission to reset.

4. **There is no native resume:** `run.py:61` requires a nonexistent output directory, builds a fresh summary (`:87`), and reruns all configured profiles. Do not stitch B32 from a failed run into an invented “completed” native summary or run B8/B1 alone and claim all three native profiles. Standalone supplementary rows can preserve useful grid coverage, but do not replace native completion.

5. **Evidence-preserving headline-only fallback:** after old runner/client completion and safe server closure, archive the failed attempt intact outside the canonical `run` path, retain its original identity and absolute-path provenance, then launch a fresh native attempt at canonical `doc/benchmark/run`. Use all `[1,8,32]`, `tasks: []`, the existing client/server commands, and set `roofline_command` to `tools/collect_benchmark_roofline.py` directly. Its CLI implements the required `check`/`collect` interface (`:243-250`), so all readiness probes, actual headline measurements and native validations remain genuine. Do not merely rename a live server directory: owned-server state and phase paths still reference it. Close the owned server safely before moving its evidence.

6. Preserve the **original** 90-minute boundary across that fallback. Set its budget to no more than `min(5400, original_native_end - now, outer_stage_end - now)`, less an explicit reporting/closure reserve. Do not assign a fresh 5400 seconds just because the new runner allows it. Its fresh elapsed counter alone does not prove compliance with the original absolute boundary; retain attempt clocks and total elapsed evidence. A new output path does not reuse the B32 server: phase-path identity differs (`benchmark_server.py:288-300`), so budget for B32 reload plus B8 and B1. The observed B32 startup was 236.29 seconds, but this is not a bound on future loads. If the remaining time cannot support the complete native attempt, record incomplete stage evidence rather than extend either cap.

7. Only after mandatory native completion, use remaining outer-stage time for serial grid-only supplements, with full row/closure estimates based on observed timings and the unchanged stage clock. Current `--deadline-seconds` must be chosen against that absolute clock; its built-in 900-second reserve is not an execution guarantee. Use new row destinations to retain failed attempts. Preserve validated completed rows without rerunning them; both grid reuse and `report_benchmark_v5.py:34-45` recognize `run/grid` and `run/supplement-*/grid`. If carrying archived rows into this discovery layout, copy complete evidence bundles byte-for-byte, retain originals and record provenance; do not rewrite hashed server identities to pretend they came from the new attempt.

8. Require real native summary/checks before declaring completion. Current `benchmark_stage/check.py:45-76` supports `tasks: []`, requires completed status and elapsed below the configured cap, and checks matching concurrency sets. For three profiles the cap is 5400 seconds. Historical nonempty-accuracy/3600-second waivers do not apply to this installed checker. Its canonical evidence location is `doc/benchmark/run`; retain a genuine native report/config/summary there and refresh surrounding identity/report artifacts consistently. Grid reports alone are insufficient.

## Other potential issues / limits

- A new row estimate is not a proof of completion before deadline. Imports, tokenization, observer exchange and actual long-context execution must fit as well. The current max-context estimate is approximately 789 seconds plus a 60-second admission margin, versus the observed roughly 775-second enclosing seeded row; this specific estimate is not evidence of an underestimate. Historical source had no estimate at all.
- Server lifetime is independently bounded by `tests/run_serving.sh:14` (`timeout 7200`). The original B32 server cannot be assumed available for arbitrary late supplements. That is a separate lifecycle limit, not the cause of the earlier grid deadline.
- Completion-marker reuse checks existence, not content equivalence. Reuse is justified here only after matching row parameters, runtime identity, successful lengths/counts and phase evidence. It must not import v3/v4 results.
- No hardware/model correctness claim is implicated by this investigation. Longer-than-predicted host-observed latency is supported; its device-level cause is outside scope.

## Claim review

Rechecked the historical grid, current diff, wrapper child launch, Python timeout implementation, native nonzero versus timeout branches, fresh-output requirement, server isolation/idle/provenance guards, direct collector interface and current checker. The headline findings survive those checks. The future timeout and orphan's eventual behavior remain conditional. A normal between-row deadline return would falsify the predicted failure for this attempt without removing the demonstrated admission and child-ownership defects. No source changes or reproductions were performed.
