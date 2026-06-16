# Method 2 ALL-TILE vs Approach 2 vs Baseline — Tracy Comparison

**Tests profiled:**
- `test_conv2d_dram_bottleneck` — Baseline (standard TTNN IR chain)
- `test_conv2d_method2_approach2_dram_bottleneck` — Method 2 Approach 2 (ROW_MAJOR packing)
- `test_conv2d_method2_all_tile` — ALL-TILE (ROW_MAJOR input → tilize → TILE reshape/permute)

**Tracy options:** `--profile-dispatch-cores --device-memory-profiler --op-support-count 1000`
**Reports:** `profiler_output2/all_tile_bn_{1,2}/reports/*/ops_perf_results_*.csv`

---

## Headline Results

| Run | Config 1 (1536×1536) | Config 2 (1280×2304) | vs Baseline |
|-----|---------------------|---------------------|-------------|
| **Baseline** (TILE matmul) | 14.697 ms | 16.764 ms | 1.00× |
| **Method 2 Approach 2** (ROW_MAJOR pack) | **1.793 ms** | **1.950 ms** | **8.2–8.6×** |
| **ALL-TILE** (TILE pack) | 3.866 ms | 4.997 ms | 3.4–3.8× |

**Method 2 Approach 2 is 2.2–2.6× faster than ALL-TILE.**

---

## Config 1 — 1×3×1536×1536

### ALL-TILE op breakdown (6 ops, 3.866 ms)

| Op | Count | Kernel | % | FPU % |
|----|-------|--------|---|-------|
| `PermuteDeviceOperation` | 2 | 1.569 ms | 41% | 13.4% |
| **`ReshapeViewDeviceOperation`** | **2** | **1.546 ms** | **40%** | 0.0% |
| `MatmulDeviceOperation` | 1 | 0.428 ms | 11% | 2.42% |
| `TilizeDeviceOperation` | 1 | 0.323 ms | 8% | 0.0% |

### Method 2 Approach 2 op breakdown (5 ops, 1.793 ms)

| Op | Count | Kernel | % | FPU % |
|----|-------|--------|---|-------|
| `PermuteDeviceOperation` | 2 | 0.992 ms | 55% | 12.1% |
| `MatmulDeviceOperation` | 1 | 0.435 ms | 24% | 2.38% |
| `TilizeDeviceOperation` | 1 | 0.190 ms | 11% | 0.0% |
| `UntilizeDeviceOperation` | 1 | 0.176 ms | 10% | 38.0% |

---

## Config 2 — 1×3×1280×2304

### ALL-TILE op breakdown (6 ops, 4.997 ms)

| Op | Count | Kernel | % | FPU % |
|----|-------|--------|---|-------|
| **`ReshapeViewDeviceOperation`** | **2** | **2.409 ms** | **48%** | 0.0% |
| `PermuteDeviceOperation` | 2 | 1.845 ms | 37% | 15.8% |
| `MatmulDeviceOperation` | 1 | 0.552 ms | 11% | 2.35% |
| `TilizeDeviceOperation` | 1 | 0.191 ms | 4% | 0.0% |

---

## Why ALL-TILE Is Slower Than Approach 2

### The `ReshapeViewDeviceOperation` is the new bottleneck

In the ALL-TILE pipeline, the reshape from TILE `[N,C,H,W]` → TILE `[N,C*K,H/K,W]` is **not a free view** — it dispatches a `ReshapeViewDeviceOperation` kernel because the tile boundaries change (the H dimension goes from 1536 to 48, changing the tile-row structure). This kernel takes **1.55–2.41 ms** (40–48% of total time).

```
[N,3,1536,1536] TILE:  tiles span the 1536×1536 H×W plane (48×48 tile grid)
[N,96,48,1536]  TILE:  tiles span the   48×1536 H×W plane  (2×48 tile grid)
                        → different tile height dimension → data must be reorganised
                        → ReshapeViewDeviceOperation dispatched → ~1.5 ms
```

### Op-by-op comparison (Config 1, 1536×1536)

| Op | Approach 2 (ROW) | ALL-TILE | Difference |
|----|-----------------|---------|------------|
| `ReshapeViewDeviceOperation` | **0 ms** | **1.546 ms** | +1.546 ms ← NEW bottleneck |
| `PermuteDeviceOperation` (×2) | 0.992 ms | 1.569 ms | +0.577 ms |
| `TilizeDeviceOperation` | 0.190 ms | 0.323 ms | +0.133 ms |
| `UntilizeDeviceOperation` | 0.176 ms | **0 ms** | −0.176 ms |
| `MatmulDeviceOperation` | 0.435 ms | 0.428 ms | −0.007 ms |
| **Total** | **1.793 ms** | **3.866 ms** | **+2.073 ms** |

### Why Approach 2 avoids `ReshapeViewDeviceOperation`

In Method 2 Approach 2, the spatial packing is done in **ROW_MAJOR** where `reshape [N,3,H,W]→[N,96,H/K,W]` IS a free view (same contiguous memory, just reinterpreted metadata). No kernel is dispatched — the operation is instant. The tilize then happens on the already-flat `[1,1,packed_sp,96]` tensor.

```
Approach 2: ROW_MAJOR reshape = free view (0 ms) → tilize 17.7 MB (0.19 ms)
ALL-TILE:   TILE reshape requires data copy (1.55 ms) → permute (0.78 ms each)
```

### Why TILE permute is also slower than ROW_MAJOR permute

In Approach 2, the pack permute is `[N,96,48,1536]` ROW_MAJOR → `[N,48,1536,96]` ROW_MAJOR (0.50 ms each).
In ALL-TILE, the pack permute is `[N,96,48,1536]` TILE → `[N,48,1536,96]` TILE (0.78 ms each).

The TILE permute must reorganise data at tile granularity, reading ~18.9 MB of input tiles (with 48/32=2 tile-rows, some padding) and writing 14.2 MB. The ROW_MAJOR permute (`MultiCoreBlockedGeneric`) reads ~17.7 MB in a blocked pattern. The TILE permute ends up doing more work due to the tile padding in the H/K=48 dimension.

---

## Summary

| Approach | Total | vs Baseline | New bottleneck |
|----------|-------|-------------|----------------|
| Baseline | 14.7 ms | 1.00× | MatmulDeviceOperation (96% DRAM BW) |
| **Method 2 Approach 2** | **1.79 ms** | **8.2×** | PermuteDeviceOperation (55%) |
| ALL-TILE | 3.87 ms | 3.8× | ReshapeViewDeviceOperation (40%) |

**Method 2 Approach 2 (ROW_MAJOR packing) remains the fastest approach** because:
1. ROW_MAJOR reshape is a **free view** — zero device cost
2. The subsequent permute operates on **smaller data** (17.7 MB ROW_MAJOR vs 18.9 MB TILE with padding)
3. Tilize cost (0.19 ms) < ReshapeViewDeviceOperation cost (1.55 ms)

The ALL-TILE idea is architecturally clean (no tilize/untilize boundaries) but the TILE reshape `[N,C,H,W]→[N,C*K,H/K,W]` is more expensive than the ROW_MAJOR `reshape+tilize` combination because TILE reshape dispatches a full data-copy kernel while ROW_MAJOR reshape is a zero-cost metadata change.
