---
name: tt-dit-benchmark-profile
description: >-
  Use when someone needs numbers — or an explanation of numbers — about
  Tenstorrent device performance for a models/tt_dit component (LTX, Wan, Flux,
  Mochi, SD3.5, Qwen-Image, Ideogram, MiniMax-H3 and friends), rather than a code
  change. Three situations: (1) Measuring — profile, benchmark or time a model,
  layer or op; per-op or per-layer breakdown; find the bottleneck; capture a
  baseline before optimizing. (2) Reading a profile already in hand — Tracy
  output, ops_perf_results CSV, device zones, op-to-op gap, tt-perf-report
  claims, what a surprising hot op means (tilize/untilize, reshard, collective),
  or durations that look wrong, zero or missing. (3) Investigating a slowdown — a
  test, layer or mesh slower than a previous run or another config, and you need
  to locate what changed. Pure measurement and interpretation; it produces
  numbers and hands them off, never edits code or applies fixes.
---

# TT-DiT Profiling

**Pure measurement. Never modify code, never iterate, never propose fixes.**
Hand numbers to `tt-dit-performance`. An agent that measures and hopes for a result
at the same time will find the result.

## The metric

**Warm device time per inference step, at a stated mesh shape and input shape.**

- **Warm** — program cache populated, weights uploaded. Cold measures compilation.
- **Device time**, not wall clock. Wall clock on an unwarmed pipeline is
  dominated by host work.
- **Per step** — a per-invocation number plus a work-unit count, so full-clip
  time is a multiplication, not another measurement.
- **Stated shapes** — a number without mesh shape and input shape is not
  comparable to anything.

**Weight upload is one-time construction cost and is never counted.** It is
often the largest thing in an unfiltered profile and has produced wrong
conclusions here more than once.

## Pipeline

```
baseline → capture → find CSV → establish warm window → rank → classify vs known fast paths → journal
```

| Stage | Non-negotiable |
|---|---|
| **Baseline first** | A campaign without a fixed, recorded "before" produces numbers nobody can check. `benchmark-and-profile.md` has what to pin, the gate that catches invalid runs, and how to record it as a regression test |
| **Warm window before any aggregate** | Skipping it is the single most common way to produce a confidently wrong profile — `reading-profiles.md` § "The op-to-op gap" |
| **Classify against `existing-fast-paths.md`** | ttnn already ships DiT-specific fused ops for distributed norms, matmul+collective, AdaLN modulation, head reshaping, halo padding and CCL/compute overlap. A hot spot is far more likely to be a fast path that isn't engaged than one that doesn't exist |
| **Timeout-gate every capture** | `../shared/device-hangs.md`. Profiling runs hang like any other, and a wedged one leaves a dirty device *and* no data |

## Scope the capture — always one layer or smaller

**Profile a single layer, a single block, or a single op. Never the whole
model** — this is the default, not a fallback for when the whole model won't fit.

A small scope is a *better* profile: the per-op ranking within one repeated unit
generalizes to the stack, and a capture you can read end to end is one you can be
honest about. Profile the unit, multiply by the count.

It also keeps you inside Tracy's ~1000-op-per-device buffer — a video VAE encoder
can emit ~550 ops per iteration, so two iterations overflow and report generation
fails with `Device data missing: Op <id> …`, which looks like a tool bug and is
buffer overflow. **Only if you genuinely cannot reduce the op count** should you
flush mid-run (`tracy-capture.md`); reaching for the dump first means the scope is
wrong.

## Output

```markdown
**Target:** <pytest path -k filter> · mesh <shape> · input <shape> · `<sha>`
**Window:** <signpost / last N ops / iteration 2>

| Op | n | Device FW | % of scope | Cores | Fidelity | Bound |
|---|---|---|---|---|---|---|

**Op-to-op gap:** median <x> µs, mean <y> µs, share of window <z>%
**Bottleneck:** <op + why>
**CSV:** <path>
```

The gap line is mandatory even when boring. Its absence is what let a wrong
conclusion stand for two amendments.

Report only what you observed. If the capture failed, name the failure and the
likely cause; never estimate a number and present it as measured.

| Sub-task | Load |
|---|---|
| Establishing / comparing a baseline, merge checklist | `benchmark-and-profile.md` |
| Capture mechanics, signposts, zones, buffer limits | `tracy-capture.md` |
| Warm window, gap distribution, bound classes, peaks | `reading-profiles.md` |
| **Classifying a hot spot — does a fused op already exist** | **`existing-fast-paths.md`** |
| Run wedged, or profiling a suspected hang | `../shared/device-hangs.md` |
