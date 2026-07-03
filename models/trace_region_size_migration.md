# trace_region_size dynamic allocation migration (#47574)

Migration completed: all CI models now use dynamic trace buffer allocation
(`trace_region_size=0`) unless listed as a fixed-size exception in
[`model_trace_region_sizes.yaml`](./model_trace_region_sizes.yaml).

## CI validation (pre-migration analysis)

Source: unit-test runs with Watcher diagnostics (`28449218507`, `28521068573`,
`28582288721`). Watcher-related failures are unrelated to trace region sizing.

| Model / SKU | Pre-migration | Post-migration | CI unit result (pre-Watcher) |
|---|---|---|---|
| deepseek-v3 / all Galaxy-T3K SKUs | explicit `0` | dynamic (default) | pass |
| llama3.1-8b / wh_n150 | explicit `0` | dynamic (default) | pass |
| Qwen3.6-27B / p300x2 | unconfigured → `0` | dynamic (default) | pass |
| Qwen3-VL-32B / wh_llmbox_perf | alias gap → `0` | dynamic (default) | pass |
| Shallow UNet / wh_llmbox_perf | YAML `679936` (N150 only) | exception kept | pass |
| Gemma-4-* / bh_quietbox_2 | fixed sizes | dynamic (default) | pass |
| Most tt_transformers LLMs | fixed 10M–250M | dynamic (default) | validate via CI |

## Fixed-size exceptions (kept in YAML)

| Model key | SKU | Size (bytes) | Reason |
|---|---|---|---|
| `unet-3d` | `wh_n150` | 679936 | Small single-device UNet trace |
| `llama3.3-70b-galaxy` | `wh_galaxy_perf` | 216580672 | Galaxy full e2e / prefix-caching |
| `llama3.3-70b-galaxy-decode` | `wh_galaxy_perf` | 23887872 | Galaxy decode benchmarks |
| `llama3.3-70b-galaxy-qwen` | `wh_galaxy_perf` | 184915840 | Qwen-on-galaxy stack |
| `qwen3-32b-galaxy` | `wh_galaxy_perf` | 102000000 | Galaxy e2e |
| `qwen3-32b-galaxy-decode` | `wh_galaxy_perf` | 12726272 | Galaxy decode benchmarks |

## Recommended CI validation after merge

Run **All Model Tests** (`workflow_dispatch`) on branch without Watcher enabled:

1. **Unit (tiers 1–3):** `run-unit-tests=true`, `model=all`, `skus=all`
2. **E2E traced paths:** `run-e2e-tests=true`, filter by `llama3.1-8b`, `gemma`, `qwen36`, `shallow-unet`
3. **Device perf:** jobs in `models_device_perf_tests.yaml` with trace capture

Pass criteria: no `TT_FATAL` trace buffer overlap / OOM during trace capture; perf
within existing CI tolerances.

## Rollback

If a `(model, SKU)` pair regresses, re-add an explicit `trace_region_size: <bytes>`
entry under a model block in `model_trace_region_sizes.yaml` with a comment
explaining why dynamic allocation failed.
