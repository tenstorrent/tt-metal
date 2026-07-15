# DiffusionGemma pure prefill speed — 64K build

Date: 2026-07-13
Device: QB2 / P150x4 / 4× Blackhole / TP=4
Model: full 30-layer `google/diffusiongemma-26B-A4B-it`
Build: one model instance with `max_seq_len=65536`
Mode: prefill only; no denoise, sampling, commit, or output block

| context L | build | prefill | prefill tok/s |
|---:|---:|---:|---:|
| 1,024 | 65,536 | 0.69 s | 1,473.9 |
| 4,096 | 65,536 | 1.27 s | 3,213.2 |
| 16,384 | 65,536 | 43.17 s | 379.5 |
| 32,768 | 65,536 | 96.54 s | 339.4 |
| 65,536 | 65,536 | 235.44 s (3.92 min) | 278.4 |

The run built once in 21.37 s and used 17.33 GiB DRAM/chip after build. Each row is one synchronized
first execution of that exact prompt shape, so shape-specific first-use compilation is included.

Artifacts:

- `context_window_prefill_only_20260713_msl65536.json`
- `prefill_speed_64k_build_20260713.png`
