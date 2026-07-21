# Prefill speed-ups: old vs new (REAP + Flash)

**Canonical path:** this file.
**Flash pointer:** `/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/SPEEDUPS.md`
**Written:** 2026-07-21 (from existing measurements; no new marathon device runs).
**Updated:** 2026-07-21 — full REAP ISL×B chunked sweep complete; see **`REAP_FULL_SWEEP_COMPARE.md`**.

Sources: `VALIDATION.md`, `validation_summary.csv`, `sweep_summary.csv`,
`../batched_prefill_smoke/{matrix_summary.csv,RESULTS.md}`,
`reap_full_new_summary.csv` + `old_baseline_inventory.csv` (full grid).

---

## Definitions

| Label | REAP | Flash |
|-------|------|-------|
| **Old** | PCM=0 / CHUNK=0 (legacy full-T), or Jul17 pre-chunk matrix | Jul17 `batched_prefill_smoke` matrix, or sweep PCM=1 where compared to PCM=2 |
| **New** | PCM=1 / CHUNK=4096 (chunked) | Current plateau: PCM≥2 and/or CHUNK≥8192 (~same wall time as PCM1 on re-run) |

Mesh: REAP **8×4** · Flash **4×8**. Prefill times in **seconds**.

---

## 1. Defaults baked (2026-07-21)

| Tree | File | Behavior |
|------|------|----------|
| REAP | `glm4_moe/tt/moe_tt.py` | If PCM **and** CHUNK env unset: **adaptive** — `T > 256` → PCM=1 + CHUNK=4096; else PCM=0 + CHUNK=0. Explicit env always honored. |
| REAP | `glm4_moe/scripts/run_sweep_isl_batch.py` | Sweep child env sets `GLM4_MOE_MOE_SPARSE_PREFILL_PCM=1`, `..._CHUNK_TOKENS=4096`. Attn reduce left at **rs_ag** (not host). |
| Flash | `glm4_moe_lite/tt/moe_tt.py` | Default `GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM` **`"1"` → `"2"`**. CHUNK default stays **4096** (PCM cap dominates; 8192 optional but redundant). |
| Flash | `glm4_moe_lite/scripts/run_sweep_isl_batch.py` | Sets PCM=2, CHUNK=4096 in child env. |

---

## 2. REAP — paired old vs new (validated)

### 2.1 Batched (primary win)

| ISL×B | T | Old (PCM0/0) | New (PCM1/4096) | Speedup | Source |
|------:|--:|-------------:|----------------:|--------:|--------|
| 128×4 | 512 | 22.370 | **4.666** | **4.79×** | validation 2026-07-21 |
| 512×8 | 4096 | 683.112 | **24.317** | **28.1×** | validation 2026-07-21 |

### 2.2 Same-day smoke (rs_ag already on both arms)

| ISL×B | Mode | Old PCM0/0 | New PCM1/4096 | Speedup | Source |
|------:|------|-----------:|--------------:|--------:|--------|
| 128×4 | batched | 12.710 | **6.043** | **2.10×** | sweep_summary 2026-07-20 |
| 128×4 | serial | **5.174** | 6.718 | **0.77×** (PCM1 **hurts**) | sweep_summary |

Validation serial (same pattern):

| ISL×B | Mode | Old PCM0/0 | New PCM1/4096 | Note |
|------:|------|-----------:|--------------:|------|
| 128×4 | serial | **5.018** | 5.569 | **1.11× slower** — do not force chunk on tiny serial |

Adaptive default (`T > 256`) turns chunking **on** for batched 128×4 (T=512) and **off** for each serial user (T=128).

### 2.3 Jul17 pre-chunk matrix (old only) vs new where measured

Older matrix had **no REAP MoE chunking** (and earlier attn path). Label these as **old**.
New times only exist for cells re-measured after the port.

| ISL×B | Old batched (Jul17) | New batched (PCM1/4096) | Speedup | Notes |
|------:|--------------------:|------------------------:|--------:|-------|
| 128×4 | 22.348 | **4.666** | **4.79×** | new = validation |
| 128×8 | 50.198 | — | — | new not remeasured |
| 128×16 | 177.044 | — | — | new not remeasured |
| 128×32 | 692.724 | — | — | new not remeasured |
| 512×4 | 173.928 | — | — | new not remeasured |
| 512×8 | 684.279 | **24.317** | **28.1×** | new = validation |
| 512×16 | 2721.904 | — | — | skip re-run (hours) |
| 1024×4 | 678.802 | — | — | new not remeasured |
| 1024×8 | 2701.499 | — | — | new not remeasured |
| 1024×16 | 10794.930 | — | — | skip (~3h old path) |
| 2048×4 | 2692.243 | — | — | skip |
| 2048×8 | 10748.293 | — | — | skip |

**Serial (Jul17 old, for context):** 128×4 = 5.021s · 512×8 = 89.455s · 1024×8 = 342.396s.

Expectation for unmeasured medium cells: new path should track closer to linear in T (chunked MoE) rather than ~O(T²); do not invent numbers without a re-run.

### 2.4 Full grid (2026-07-21) — canonical compare

**20-cell** batched sweep (`PCM=1`/`CHUNK=4096`, `MAX_TOKENS=131072`): **17 ok · 3 OOM** (T≥65536).

| Metric | Value |
|--------|------:|
| Paired old/new cells | 12 |
| Median speedup | ~35× |
| Peak speedup | **~155×** (2048×8, 1024×16) |
| Working envelope | T ≤ 32768 ok |

Full table + graphs: **`REAP_FULL_SWEEP_COMPARE.md`**, CSV `reap_full_new_summary.csv`.

---

## 3. Flash — old vs new

### 3.1 PCM1 → PCM2 (validation): no material speedup

| ISL×B | Mode | PCM1/4096 | PCM2/4096 | Δ | Source |
|------:|------|----------:|----------:|---|--------|
| 512×8 | batched | 15.409 | 15.404 | **~0%** | validation |
| 1024×16 | batched | 56.904 | 57.267 | noise | validation |

Default bump to **PCM=2** is for **consistency / avoiding flaky PCM1**, not a large TTFT win.

### 3.2 Sweep plateau (2026-07-20) — PCM×CHUNK @ 512×8 batched

| PCM | CHUNK | prefill_s | Note |
|----:|------:|----------:|------|
| 1 | 4096 | **28.317** | **flaky outlier** — did **not** reproduce on validation (15.4s) |
| 1 | 8192 | 15.333 | plateau |
| 2 | 4096 | 15.349 | plateau |
| 2 | 8192 | 15.296 | plateau |
| 4 | 4096 | 15.330 | plateau |
| 4 | 8192 | 15.260 | plateau |

### 3.3 Jul17 matrix vs current plateau (same ISL×B)

Flash already had PCM/chunking in Jul17; gap is other stack changes + measurement noise, **not** a PCM1→PCM2 cliff.

| ISL×B | Jul17 batched (old matrix) | Current plateau (sweep/val) | Approx Δ |
|------:|---------------------------:|----------------------------:|---------:|
| 512×8 | 26.643 | ~15.3–15.4 | ~1.7× |
| 1024×16 | 57.739 | ~57–60 | ~flat |
| 2048×8 | 59.917 | ~59–60 | ~flat |

Serial Flash (sweep, PCM1/4096 unless noted): 512×8 = 16.951 (PCM2: 15.227) · 1024×16 = 55.195 · 2048×8 = 52.622.

---

## 4. Takeaways

1. **REAP batched:** chunking is the big lever — **~5–6×** @ tiny T, **~35×** median across paired cells, **~155×** @ T=16k vs legacy full-T (full grid in `REAP_FULL_SWEEP_COMPARE.md`).
2. **REAP serial tiny:** keep legacy (adaptive `T ≤ 256`); PCM1 costs ~11–30%.
3. **Flash:** already near plateau; **PCM=2 default is safety**, not a speedup story. Treat 28s PCM1/4096 as non-reproducible.
4. **Attn rs_ag** (REAP) stays default — do not force host AR.

---

## 5. Artifact index

| Path | Role |
|------|------|
| `./VALIDATION.md` + `validation_summary.csv` | Paired old/new REAP + Flash PCM1/2 |
| `./sweep_summary.csv` | REAP smoke + Flash full PCM grid |
| `../batched_prefill_smoke/matrix_summary.csv` | Jul17 old matrix (both models) |
| `./REAP_FULL_SWEEP_COMPARE.md` + `reap_full_new_summary.csv` | Full 20-cell new sweep + old/new table |
| Flash `.../glm4_moe_lite/experiments/prefill_opt_sweep/` | Local Flash CSV/logs; SPEEDUPS pointer |
