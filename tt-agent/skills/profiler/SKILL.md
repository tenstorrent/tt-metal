---
name: profiler
description: "Profile a TT kernel or op with Tracy + tt-perf-report — runs through tt:run (MCP), locates the CSV, interprets device timing, produces a structured profile note. Use for device-time breakdowns and bottleneck identification."
metadata:
  layer: tool
---

# TT Profiler

## Purpose

Own the full profiling cycle for a single target. Compose the `python -m tracy
-p -r -v -m <cmd>` invocation, delegate execution to `tt:run` (which routes to
tt-device-mcp), locate the resulting CSV under
`$TT_METAL_HOME/generated/profiler/reports/<timestamp>/`, run `tt-perf-report`,
and write a structured note with device timing and a bottleneck read.

Pure measurement — never modifies code, never iterates, never hypothesizes
fixes. Called by `tt:optimizer` once per iteration; invokable directly by a
developer who wants a one-shot profile.

**Why `layer: tool`:** pipeline-bound to profiling, same shape as `tt:run`. Not
cross-cutting — it is only meaningful as part of a measure-then-act loop.

## When to Invoke

- "Profile this test / kernel / op"
- "What's the device time for X?"
- "Where's the bottleneck in <test>?"
- Called by `tt:optimizer` each iteration (baseline and trials)

## Pipeline

```
resolve target → compose tracy cmd → dispatch via tt:run → find CSV → tt-perf-report → interpret → note
```

1. **Resolve target**: the caller hands over a pytest path (+ optional `-k`
   filter) or a binary target. Confirm the file exists.
2. **Compose**: build the command per `knowledge/recipes/tt-metal/profiler.md`.
   Quote the wrapped pytest invocation. Surface any conflicting env vars
   (`TT_METAL_DPRINT_CORES`, `TT_METAL_WATCHER`, `TTNN_CONFIG_PATH`) to the
   caller before running — profiling fails silently if these are set.
3. **Dispatch**: hand the composed command to `tt:run`. tt:run routes to
   tt-device-mcp (profiling runs on-device and must not go through Bash).
4. **Find CSV**: after tt:run returns, locate the most recent subdir under
   `$TT_METAL_HOME/generated/profiler/reports/` by mtime. Read
   `ops_perf_results_<timestamp>.csv` from that subdir. If missing, stop and
   report the profile failed (common causes: device profiler was disabled, an
   env conflict silently dropped data, or the wrapped command exited non-zero
   before the device closed).
5. **Analyze**: run `tt-perf-report <csv>`. If it crashes on a custom op, retry
   with `--no-stacked`. Capture the full rendered report.
6. **Interpret**: load `interpretation.md` to read the CSV columns, rank ops
   by `DEVICE FW DURATION [ns]`, and tag each top op with a bottleneck
   classification (reader-bound / compute-bound / writer-bound / NOC-stall).
7. **Note**: write `~/.tt-agent/notes/profile-<scope>-<YYYY-MM-DD-HHMMSS>.md`
   (format below). Return the note path + top-3 ops to the caller.

## Scope Naming

`<scope>` is a short slug derived from the target — op name, test name, or
kernel name. Examples: `matmul-2d`, `bert-tiny-demo`, `flash-attention`.

## Note Format

```markdown
# Profile: <scope>

**Date:** YYYY-MM-DD HH:MM:SS
**Repo:** tt-metal @ <short-sha>
**Target:** <test path> -k <filter>
**CSV:** <absolute path to ops_perf_results_*.csv>
**Tracy trace:** <absolute path to .tracy file>

## Top Ops by Device FW Duration

| Rank | Op Code | Device FW [ns] | Cores | Math Fidelity | Bottleneck tag |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... |

## Per-RISC Breakdown — Top Op

| RISC | Duration [ns] | % of op |
|---|---|---|
| BRISC (reader) | ... | ... |
| NCRISC (writer) | ... | ... |
| TRISC0/1/2 (compute) | ... | ... |

## Bottleneck Read

<1-3 sentences. What dominates device time, which processor, any obvious
under-utilization (low core count, low math fidelity setting, serialized NOC).
No prescriptions.>

## Raw tt-perf-report Output

<paste the full rendered report>
```

## Caller Contract

- **Input**: target (pytest path + filter, or binary) and scope slug.
- **Output**: path to the profile note, plus top-3 op codes and their device
  durations inline for the caller's convenience.
- **Failure**: if profiling data cannot be collected, return an error message
  naming the likely cause. Do not fabricate numbers.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Tracy invocation, output paths, tt-perf-report CLI | `knowledge/recipes/tt-metal/profiler.md` |
| Reading CSV columns, bottleneck classification | `interpretation.md` |
| MCP routing, env handoff | invoke `tt:run` |
| Unfamiliar op type or perf idiom | invoke `tt:learn("<op> profiling patterns")` |
