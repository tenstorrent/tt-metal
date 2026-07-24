# REAP full ISL×batch sweep — old vs new (chunked)

**Mesh:** 8×4 · **Model:** `cerebras/GLM-4.7-REAP-218B-A32B`
**New path:** batched + `PCM=1` / `CHUNK=4096` + `BATCHED_PREFILL_MAX_TOKENS=131072` + attn `rs_ag`
**Old path:** pre-chunk PCM0/CHUNK0 (validation pairs) or Jul17 `batched_prefill_smoke` matrix
**Sweep wall:** 2026-07-21 ~03:57–04:43 UTC (~46 min) · **17/20 ok · 3 OOM · 0 timeout**

Artifacts:
- New CSV: `reap_full_new_summary.csv`
- Old inventory: `old_baseline_inventory.csv`
- Runner: `run_reap_full_isl_batch_sweep.sh`
- Graphs: `reap_full_prefill_vs_isl_by_batch.png`, `reap_full_speedup_heatmap.png`, `reap_full_prefill_vs_T.png`

---

## Prefill seconds (old vs new) + speedup

| ISL | B | T | Old prefill_s | New prefill_s | Speedup | New status | Old source |
|----:|--:|--:|--------------:|--------------:|--------:|:-----------|:-----------|
| 128 | 4 | 512 | 22.370 | 3.982 | **5.62×** | ok | validation PCM0 (2026-07-21) |
| 128 | 8 | 1024 | 50.198 | 8.988 | **5.58×** | ok | Jul17 matrix |
| 128 | 16 | 2048 | 177.044 | 14.828 | **11.94×** | ok | Jul17 matrix |
| 128 | 32 | 4096 | 692.724 | 20.359 | **34.03×** | ok | Jul17 matrix |
| 512 | 4 | 2048 | 173.928 | 10.988 | **15.83×** | ok | Jul17 matrix |
| 512 | 8 | 4096 | 683.112 | 19.019 | **35.92×** | ok | validation PCM0 (2026-07-21) |
| 512 | 16 | 8192 | 2721.904 | 37.396 | **72.79×** | ok | Jul17 matrix |
| 512 | 32 | 16384 | n/a | 74.043 | — | ok | n/a |
| 1024 | 4 | 4096 | 678.802 | 19.495 | **34.82×** | ok | Jul17 matrix |
| 1024 | 8 | 8192 | 2701.499 | 36.247 | **74.53×** | ok | Jul17 matrix |
| 1024 | 16 | 16384 | 10794.930 | 70.225 | **153.72×** | ok | Jul17 matrix |
| 1024 | 32 | 32768 | n/a | 149.524 | — | ok | n/a |
| 2048 | 4 | 8192 | 2692.243 | 36.276 | **74.22×** | ok | Jul17 matrix |
| 2048 | 8 | 16384 | 10748.293 | 69.117 | **155.51×** | ok | Jul17 matrix |
| 2048 | 16 | 32768 | n/a | 139.783 | — | ok | n/a |
| 2048 | 32 | 65536 | n/a | — | — | **OOM** | n/a |
| 4096 | 4 | 16384 | n/a | 70.458 | — | ok | n/a |
| 4096 | 8 | 32768 | n/a | 140.864 | — | ok | n/a |
| 4096 | 16 | 65536 | n/a | — | — | **OOM** | n/a |
| 4096 | 32 | 131072 | n/a | — | — | **OOM** | n/a |

**New cells:** 17 ok · 3 OOM · 0 timeout/fail

---

## Speedup highlights (both arms present)

| ISL×B | T | Old | New | Speedup |
|------:|--:|----:|----:|--------:|
| 2048×8 | 16384 | 10748.3 | 69.1 | **155.51×** |
| 1024×16 | 16384 | 10794.9 | 70.2 | **153.72×** |
| 1024×8 | 8192 | 2701.5 | 36.2 | **74.53×** |
| 2048×4 | 8192 | 2692.2 | 36.3 | **74.22×** |
| 512×16 | 8192 | 2721.9 | 37.4 | **72.79×** |
| 512×8 | 4096 | 683.1 | 19.0 | **35.92×** |
| 1024×4 | 4096 | 678.8 | 19.5 | **34.82×** |
| 128×32 | 4096 | 692.7 | 20.4 | **34.03×** |
| 512×4 | 2048 | 173.9 | 11.0 | **15.83×** |
| 128×16 | 2048 | 177.0 | 14.8 | **11.94×** |
| 128×4 | 512 | 22.4 | 4.0 | **5.62×** |
| 128×8 | 1024 | 50.2 | 9.0 | **5.58×** |

- Paired cells: **12**
- Median speedup: **~35×**
- Geo-mean speedup: **~33×**
- Largest measured win: **~155×** @ 2048×8 / 1024×16 (T=16k)

---

## Failures (OOM)

All three failures are DRAM OOM at T ≥ 65536 during allocation (KV / activation buffers), not MoE chunking hangs:

| Cell | T | Detail (abbrev) |
|------|--:|-----------------|
| 2048×32 | 65536 | Need ~640 MiB DRAM buffer; banks exhausted |
| 4096×16 | 65536 | Same class (~640 MiB) |
| 4096×32 | 131072 | Need ~1.25 GiB buffer; bank size ~1.02 GiB |

Working envelope for this config: **T ≤ 32768** ok (4096×8, 2048×16, 1024×32 all succeeded).

---

## Old data sources

| Preference | Source | Notes |
|------------|--------|-------|
| Primary (paired) | `validation_summary.csv` | 128×4 & 512×8 PCM0/0 on 2026-07-21 |
| Bulk old matrix | `batched_prefill_smoke/matrix_summary.csv` | Jul17 pre-chunk REAP batched |
| Not used as old batched | `g1_multilink_4_ring_isl_sweep/` | Different harness / closer to serial-ish times |

Missing old cells left as **n/a** — not invented. Jul17 matrix lacked 512×32, 1024×32, 2048×16/32, and all 4096×B cells.

---

## New path scaling note

New prefill tracks ~linear in T (~4–5 ms/token aggregate at mid sizes), vs old ~O(T²) MoE path. Same T cells with different (ISL,B) land near the same prefill_s (e.g. T=8192 → ~36–37s; T=16384 → ~70–74s).

---

## Graphs

- `reap_full_prefill_vs_T.svg` — old vs new prefill vs aggregate tokens
- `reap_full_speedup_heatmap.svg` — speedup grid
- PNG regen (when python_env available): `plot_reap_full_compare.py` → `reap_full_*.png`
