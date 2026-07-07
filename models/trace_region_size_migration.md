# trace_region_size dynamic allocation migration (#47574)

**Status: partial migration (phase 1).** Default is dynamic allocation
(`trace_region_size=0`) for unconfigured `(model, SKU)` pairs. Multi-trace LLM
models keep fixed sizes from `main` with `TODO(#48869)` until the runtime
validates high-water marks across all live traces ([#48869](https://github.com/tenstorrent/tt-metal/issues/48869)).

## Phase 1 scope

| Category | Action | Reason |
|---|---|---|
| Single-trace / safe models | Dynamic (`0`) | No simultaneous trace overlap risk |
| Multi-trace LLM stack | Fixed from `main` + `TODO(#48869)` | Prefill + decode traces coexist |
| `unet-3d` / `wh_n150` | Fixed infra exception | Small single-device UNet trace region |
| Galaxy e2e/decode keys | Fixed from `main` + `TODO(#48869)` | Large trace budget + multi-trace |
| `deepseek-v3`, `informer` | Dynamic (no YAML entry) | Reference dynamic model; single-trace (informer) |
| `llama3.1-8b` / `wh_n150` | Dynamic (no YAML entry) | Already `0` on `main` (#48636) |

## Blocked on #48869

When `trace_region_size=0`, the runtime allocates trace buffers top-down in DRAM
and tracks a per-capture high-water mark (HWM). Validation in
[`mesh_trace.cpp`](../tt_metal/distributed/mesh_trace.cpp) only checks the
**current** trace's HWM, not HWMs of other live traces. Multiple simultaneous
traces (prefill + decode, warmup + main, spec-decode) can silently corrupt each
other's buffers on replay.

**Runtime fix (phase 2):** store per-trace HWM in `MeshTraceBuffer`, validate
new allocations against `max(HWM)` of all unreleased traces.

Affected model families: `tt_transformers` generator/executor, Galaxy generator,
Gemma4 spec-decode, and all entries in YAML marked `TODO(#48869)`.

## CI validation (PR #49072)

Source: [All Model Tests run 28667100396](https://github.com/tenstorrent/tt-metal/actions/runs/28667100396)
(e2e, 2026-07-03). Unit runs: `28449218507`, `28521068573`, `28582288721`.

| Model / SKU | Phase 1 status | CI result |
|---|---|---|
| deepseek-v3 / all Galaxy-T3K SKUs | dynamic (default) | pass |
| llama3.1-8b / wh_n150 | dynamic (default) | pass |
| llama3.1-8b / wh_llmbox_perf | fixed (TODO #48869) | pass |
| Qwen3.6-27B / p300x2 | dynamic (default) | pass |
| Qwen3-VL-32B / wh_llmbox_perf | dynamic (default) | pass |
| Shallow UNet / wh_n150 | fixed (infra) | pass |
| Shallow UNet / wh_llmbox_perf | dynamic (default) | pass |
| Gemma-4-* / bh_quietbox_2 | fixed (TODO #48869) | pass (PCC issues unrelated) |
| Galaxy e2e/decode | fixed (TODO #48869) | pass |
| informer / wh_n150 | dynamic (default) | not in tiered CI |

**CI verdict (mtairum, PR #49072):** zero OOM / trace-region allocation errors
across 21 e2e failures. This does **not** prove safety for multi-trace models
under #48869 (silent corruption, not `TT_FATAL`).

## Fixed-size exceptions in YAML

| Model key | SKU | Size (bytes) | Reason |
|---|---|---|---|
| Multi-trace LLMs (see YAML) | various | from `main` | `TODO(#48869)` |
| `unet-3d` | `wh_n150` | 679936 | Small UNet trace region |
| `llama3.3-70b-galaxy` | `wh_galaxy_perf` | 216580672 | Galaxy e2e + #48869 |
| `llama3.3-70b-galaxy-decode` | `wh_galaxy_perf` | 23887872 | Galaxy decode + #48869 |
| `llama3.3-70b-galaxy-qwen` | `wh_galaxy_perf` | 184915840 | Qwen-on-galaxy + #48869 |
| `qwen3-32b-galaxy` | `wh_galaxy_perf` | 102000000 | Galaxy e2e + #48869 |
| `qwen3-32b-galaxy-decode` | `wh_galaxy_perf` | 12726272 | Galaxy decode + #48869 |

## Phase 2 (after #48869 fix)

1. Remove `TODO(#48869)` blocks from [`model_trace_region_sizes.yaml`](./model_trace_region_sizes.yaml).
2. Re-run **All Model Tests** (`workflow_dispatch`): unit tiers 1–3, e2e traced
   paths, device perf.
3. Keep only infra exceptions that still regress (likely `unet-3d` / `wh_n150`).

Pass criteria: no `TT_FATAL` trace buffer overlap / OOM; no silent correctness
regression on multi-trace models.

## Rollback

If a `(model, SKU)` pair regresses, re-add an explicit `trace_region_size: <bytes>`
entry under a model block in `model_trace_region_sizes.yaml` with a comment
explaining why dynamic allocation failed.
