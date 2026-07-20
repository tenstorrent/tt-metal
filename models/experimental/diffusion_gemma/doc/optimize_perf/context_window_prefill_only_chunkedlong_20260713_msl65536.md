# DiffusionGemma pure prefill speed — 64K build, current chunked-ragged path

Date: 2026-07-13
Device: QB2 / P150x4 / 4× Blackhole / TP=4
Model: full 30-layer `google/diffusiongemma-26B-A4B-it`
Build: one model instance with `max_seq_len=65536`
Mode: prefill only; no denoise, sampling, commit, or output block
Source revision: `233b88276ab`
Prefill path: `DG_PREFILL_RAGGED_LONG=1` (current default), 4096-token ragged slices

| context L | build | prefill | prefill tok/s |
|---:|---:|---:|---:|
| 1,024 | 65,536 | 0.78 s | 1,309.4 |
| 4,096 | 65,536 | 1.37 s | 2,986.6 |
| 8,192 | 65,536 | 4.15 s | 1,972.6 |
| 16,384 | 65,536 | 5.55 s | 2,950.2 |
| 32,768 | 65,536 | 10.84 s | 3,021.5 |
| 65,536 | 65,536 | 35.58 s | 1,841.8 |

The model built once in 21.48 s and used 17.33 GiB DRAM/chip after build. Each row is one
synchronized first execution of that prompt shape, so shape-specific first-use compilation is
included. The 8K row is the first multi-chunk shape and pays first-use costs for that path.

This is the current pure-prefill table for a 64K build. `tt/prefill_moe.py` defaults
`DG_PREFILL_RAGGED_LONG` on, so every multi-token sequence uses ragged top-8 expert execution;
sequences above 4096 are split into 4096-token slices. The 64K slowdown is from the separate
long-context attention-chunking path above 32K, not a dense-MoE fallback.

The similarly named `context_window_prefill_only_20260713_msl65536.{json,md}` artifact at
`ec5b64b4891` is the historical pre-fix control. It records the old all-128-expert fallback and must
not be quoted as current default performance.

Artifacts:

- `context_window_prefill_only_chunkedlong_20260713_msl65536.json`
- `prefill_speed_64k_build_chunkedlong_20260713.png`
- `chunked_ragged_prefill.md`
- `chunked_ragged_prefill_bitident.json`
