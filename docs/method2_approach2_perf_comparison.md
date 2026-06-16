# Method 2 Approach 2 — Performance Comparison vs Baseline

**Test:** `test_conv2d_dram_bottleneck` vs `test_conv2d_method2_approach2_dram_bottleneck`
**Hardware:** Wormhole N150 · 64 Tensix cores · 288 GB/s peak DRAM BW
**Tracy options:** `--profile-dispatch-cores --device-memory-profiler --op-support-count 1000`

**Log files:** `tracy_baseline_bn_1.log`, `tracy_baseline_bn_2.log`, `tracy_method2_bn_1.log`, `tracy_method2_bn_2.log`

**Report paths:**
- Baseline: `profiler_output2/baseline_bn_{1,2}/reports/*/ops_perf_results_*.csv`
- Method 2: `profiler_output2/method2_bn_{1,2}/reports/*/ops_perf_results_*.csv`

---

## Headline Results

| Config | Baseline | Method 2 | Speedup |
|--------|----------|----------|---------|
| 1×3×1536×1536 (Block A) | **14.697 ms** | **1.793 ms** | **8.20×** |
| 1×3×1280×2304 (Block C) | **16.764 ms** | **1.950 ms** | **8.60×** |

---

## Config 1 — 1×3×1536×1536 (Block A)

### Baseline — `test_conv2d_dram_bottleneck` (5 ops, 14.697 ms)

| Op | Count | Kernel | % of total | FPU % | PM Req I BW |
|----|-------|--------|-----------|-------|-------------|
| `PermuteDeviceOperation` | 2 | 7.835 ms | **53%** | 8.4% | 24.3 GB/s |
| `MatmulDeviceOperation` | 1 | 6.429 ms | **44%** | **0.008%** | **277 GB/s** |
| `TilizeDeviceOperation` | 1 | 0.320 ms | 2% | 0.0% | — |
| `CopyDeviceOperation` | 1 | 0.113 ms | 1% | 0.001% | — |

PermuteDeviceOperation + MatmulDeviceOperation together = **97% of total runtime.**
MatmulDeviceOperation FPU=0.008% confirms the DRAM bandwidth bottleneck.

### Method 2 — `test_conv2d_method2_approach2_dram_bottleneck` (5 ops, 1.793 ms)

| Op | Count | Kernel | % of total | FPU % | PM Req I BW |
|----|-------|--------|-----------|-------|-------------|
| `PermuteDeviceOperation` | 2 | 0.992 ms | 55% | 12.1% | 211 GB/s |
| `MatmulDeviceOperation` | 1 | 0.435 ms | 24% | **2.38%** | 277 GB/s |
| `TilizeDeviceOperation` | 1 | 0.190 ms | 11% | 0.001% | — |
| `UntilizeDeviceOperation` | 1 | 0.176 ms | 10% | 38.0% | 211 GB/s |

### Op-level improvements (Config 1)

| Op | Baseline | Method 2 | Speedup |
|----|----------|----------|---------|
| `PermuteDeviceOperation` (total) | 7.835 ms | 0.992 ms | **7.90×** |
| `MatmulDeviceOperation` | 6.429 ms | 0.435 ms | **14.78×** |
| New: `UntilizeDeviceOperation` | — | 0.176 ms | — |
| **Total pipeline** | **14.697 ms** | **1.793 ms** | **8.20×** |

---

## Config 2 — 1×3×1280×2304 (Block C)

### Baseline — `test_conv2d_dram_bottleneck` (5 ops, 16.764 ms)

| Op | Count | Kernel | % of total | FPU % | PM Req I BW |
|----|-------|--------|-----------|-------|-------------|
| `PermuteDeviceOperation` | 2 | 8.513 ms | **51%** | 9.9% | 24.3 GB/s |
| `MatmulDeviceOperation` | 1 | 7.921 ms | **47%** | **0.008%** | **277 GB/s** |
| `TilizeDeviceOperation` | 1 | 0.194 ms | 1% | 0.001% | — |
| `CopyDeviceOperation` | 1 | 0.135 ms | 1% | 0.001% | — |

### Method 2 — `test_conv2d_method2_approach2_dram_bottleneck` (5 ops, 1.950 ms)

| Op | Count | Kernel | % of total | FPU % | PM Req I BW |
|----|-------|--------|-----------|-------|-------------|
| `PermuteDeviceOperation` | 2 | 0.966 ms | 50% | 17.2% | 211 GB/s |
| `MatmulDeviceOperation` | 1 | 0.554 ms | 28% | **2.34%** | 277 GB/s |
| `TilizeDeviceOperation` | 1 | 0.217 ms | 11% | 0.0% | — |
| `UntilizeDeviceOperation` | 1 | 0.213 ms | 11% | 39.2% | 211 GB/s |

### Op-level improvements (Config 2)

| Op | Baseline | Method 2 | Speedup |
|----|----------|----------|---------|
| `PermuteDeviceOperation` (total) | 8.513 ms | 0.966 ms | **8.81×** |
| `MatmulDeviceOperation` | 7.921 ms | 0.554 ms | **14.30×** |
| New: `UntilizeDeviceOperation` | — | 0.213 ms | — |
| **Total pipeline** | **16.764 ms** | **1.950 ms** | **8.60×** |

---

## Why Each Op Improved

### MatmulDeviceOperation: 14.5–14.8× faster

The spatial packing `[N,C,H,W] → [N,C*K,H/K,W]` with K=32 groups K row-groups
into the channel dimension so `C×K = 3×32 = 96` channels fill tile rows completely
(96 = 3 × TILE_WIDTH, **0% padding waste**). Activation DRAM reads drop from
**188.7 MB → 17.7 MB** (10.7× less), causing the proportional kernel speedup.

```
Baseline:  [H×W, 3]  TILE  — each row: [v v v  0 0 … 0]   9.4% useful  → reads 188.7 MB
Method 2:  [H×W/32, 96] TILE — each row: full 3 tiles       100% useful  → reads 17.7 MB
```

FPU utilisation rose from **0.008% → 2.38%** (298× improvement). The matmul is
still DRAM-bound (PM Req I BW unchanged at 277 GB/s) but now reads 10.7× less data.

### PermuteDeviceOperation: 7.9–8.8× faster

The baseline permute `[N,3,H,W] → [N,H,W,3]` in TILE writes **188.7 MB** (C=3 pads to 32).
Method 2's pack permute `[N,96,H/K,W] → [N,H/K,W,96]` writes only **17.7 MB** (C*K=96
is 3 full TILE_WIDTH columns, zero padding waste). PM Req I BW increased from 24.3 GB/s
to 211 GB/s because each core's working set is now 10.7× smaller, enabling better
cache utilisation and higher effective bandwidth.

### New ops: TilizeDeviceOperation + UntilizeDeviceOperation

Tilize (ROW_MAJOR → TILE) and Untilize (TILE → ROW_MAJOR) are required for the
pack/unpack bookkeeping. Together they add 0.37–0.43 ms, a small overhead compared
to the 12.9–14.8 ms savings.

---

## Key Bug Fixed During Investigation

During implementation, `ttnn.reshape` on ROW_MAJOR tensors returns a **view** (shares
the same physical device buffer). A premature `ttnn.deallocate` on the reshape source
while the view was still being consumed by `permute` caused data corruption (PCC≈0.54).

Fix: always defer deallocation of the reshape source until after the next op (which
consumes the view) has completed. This is a general pattern to follow with any
`ttnn.reshape` on ROW_MAJOR data.

---

## Summary

| Metric | Baseline | Method 2 | Change |
|--------|----------|----------|--------|
| Total device kernel — 1536×1536 | 14.697 ms | 1.793 ms | **8.2× faster** |
| Total device kernel — 1280×2304 | 16.764 ms | 1.950 ms | **8.6× faster** |
| MatmulDeviceOperation FPU util | 0.008% | 2.38% | **298× higher** |
| Activation DRAM reads | 188.7 MB | 17.7 MB | **10.7× less** |
| PermuteDeviceOperation write | 188.7 MB | 17.7 MB | **10.7× less** |
| Op count | 5 | 5 | same |
| PCC vs conv2d golden | — | 0.9999–1.0 | numerically correct |
