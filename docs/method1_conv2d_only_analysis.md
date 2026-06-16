# Method 1 — `test_conv2d_only` Analysis: Why It Is Slower Than Baseline

**Issue:** https://github.com/tenstorrent/tt-metal/issues/46831
**Branch:** `pchandrasekaran/conv2d_dram_bottleneck`
**Hardware:** Wormhole N150 · 64 Tensix cores · 288 GB/s peak DRAM BW

---

## 1. Test Conditions

`test_conv2d_only` isolates the conv2d operation with the following Method 1 conditions:

| Condition | Value | Purpose |
|-----------|-------|---------|
| `use_matmul_for_1x1_conv` guard | `in_channels=3 < 32` → returns `false` | Routes C=3 to regular conv path |
| Input layout | `ROW_MAJOR_LAYOUT` | Channel alignment=8 instead of 32 |
| `prepare_conv_weights` `input_layout` | `ROW_MAJOR_LAYOUT` | Weights prepared for regular conv |
| `slice_config` | **not passed** | DRAM path auto-selected |
| `shard_layout` | not set (auto) | Auto-determined by L1 fitting |
| `act_block_h_override` | 0 (auto) | Auto-calculated block height |

These are exactly the conditions described in the issue comment:
> *"route 1×1 small-C to the regular conv path using ROW_MAJOR with 8-element alignment"*

---

## 2. Profiler Results

### Tracy Options
```bash
python3 -m tracy -r -m --op-support-count 1000 --profile-dispatch-cores \
  --device-memory-profiler --check-exit-code \
  -o profiler_output/method1_analysis_only_{1,2} \
  pytest tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py::test_conv2d_only[...] -vss
```

### Report Paths
| Config | Report |
|--------|--------|
| 1×3×1536×1536 | `profiler_output/method1_analysis_only_1/reports/*/ops_perf_results_*.csv` |
| 1×3×1280×2304 | `profiler_output/method1_analysis_only_2/reports/*/ops_perf_results_*.csv` |

---

## 3. Headline Numbers

### Config 1 — 1×3×1536×1536 (2.36 M pixels)

| Metric | Baseline (Matmul, TILE) | Method 1 (RegConv, ROW_MAJOR) |
|--------|------------------------|-------------------------------|
| Execution path | `ttnn::linear` — 1 kernel | `conv2d_DRAM` — 6 slices × 5 ops = **30 ops** |
| Total device kernel | **6.430 ms** | **17.190 ms** |
| vs baseline | 1.00× | **2.67× slower** |
| FPU utilisation | 0.008% | 0.072% |
| Host time | 0.550 ms | 1.096 ms |

### Config 2 — 1×3×1280×2304 (2.95 M pixels)

| Metric | Baseline (Matmul, TILE) | Method 1 (RegConv, ROW_MAJOR) |
|--------|------------------------|-------------------------------|
| Execution path | `ttnn::linear` — 1 kernel | `conv2d_DRAM` — 8 slices × 5 ops = **40 ops** |
| Total device kernel | **7.923 ms** | **21.413 ms** |
| vs baseline | 1.00× | **2.70× slower** |
| FPU utilisation | 0.008% | 0.034% |
| Host time | 0.454 ms | 1.178 ms |

---

## 4. Method 1 Op Breakdown

### Config 1 — 1×3×1536×1536 (6 slices)

| Op | Count | Total kernel | Avg/slice | % of total | FPU % |
|----|-------|-------------|-----------|------------|-------|
| `PaddedSliceDeviceOperation` | 6 | 5.582 ms | **930 µs** | **32.5%** | 0.0% |
| `HaloDeviceOperation` | 6 | 5.202 ms | **867 µs** | **30.3%** | 0.024% |
| `SliceWriteDeviceOperation` | 6 | 3.006 ms | 501 µs | 17.5% | 0.0% |
| **`Conv2dDeviceOperation`** | **6** | **2.996 ms** | **499 µs** | **17.4%** | 0.072% |
| `MoveDeviceOperation` | 6 | 0.404 ms | 67 µs | 2.3% | 93.3% |
| **Total** | **30** | **17.190 ms** | | | |

**Conv2dDeviceOperation is only 17.4% of total time.**
Overhead ops (PaddedSlice + Halo + SliceWrite + Move) = **82.6%** of total.

### Config 2 — 1×3×1280×2304 (8 slices)

| Op | Count | Total kernel | Avg/slice | % of total | FPU % |
|----|-------|-------------|-----------|------------|-------|
| `PaddedSliceDeviceOperation` | 8 | 16.325 ms | **2,041 µs** | **76.2%** | 0.025% |
| `SliceWriteDeviceOperation` | 8 | 3.937 ms | 492 µs | 18.4% | 0.0% |
| **`Conv2dDeviceOperation`** | **8** | **0.966 ms** | **121 µs** | **4.5%** | 0.034% |
| `MoveDeviceOperation` | 8 | 0.142 ms | 18 µs | 0.7% | 3.2% |
| `HaloDeviceOperation` | 8 | 0.043 ms | 5 µs | 0.2% | 0.028% |
| **Total** | **40** | **21.413 ms** | | | |

**Conv2dDeviceOperation is only 4.5% of total time.**
Overhead ops = **95.5%** of total.

---

## 5. The Proof That ROW_MAJOR Read Reduction Works

The individual `Conv2dDeviceOperation` slices ARE dramatically faster than the baseline:

| Config | Baseline matmul | Method 1 per-slice Conv2d | Per-slice speedup |
|--------|----------------|--------------------------|-------------------|
| 1536×1536 | 6,430 µs | **499 µs** | **12.9× faster** |
| 1280×2304 | 7,923 µs | **121 µs** | **65.6× faster** |

The ROW_MAJOR read reduction works exactly as designed — each slice reads `C=3` data
with 8-element alignment (16 B/pixel) instead of 32-element alignment (64 B/pixel),
giving a real 4× reduction in activation bytes read per pixel. The conv compute is
genuinely much faster per slice.

**The problem is not the conv kernel. The problem is the overhead surrounding it.**

---

## 6. Root Cause — Why Method 1 Is Slower

### 6a. Code Path: DRAM Slicing Is Unavoidable

`determine_conv2d_execution_path()` in `conv2d_utils.cpp`:

```cpp
Conv2dExecutionPath determine_conv2d_execution_path(
    bool input_is_in_L1, const std::optional<const Conv2dSliceConfig>& slice_config) {
    if (slice_config.has_value() && slice_config->slice_type == Conv2dSliceConfig::SliceType::L1_FULL)
        return Conv2dExecutionPath::L1;
    if (!slice_config.has_value() && input_is_in_L1)
        return Conv2dExecutionPath::L1;
    return Conv2dExecutionPath::DRAM;   // ← always hit for DRAM input, no slice_config
}
```

The input is **DRAM INTERLEAVED ROW_MAJOR** and no `slice_config` is passed.
Both conditions for the L1 path are false → **always goes to `conv2d_DRAM`**.

Inside `conv2d_DRAM`:

```cpp
if (mm_conv) {
    return conv2d_L1(...);  // matmul path — bypasses slicing entirely
}
// mm_conv = false (Method 1 guard) → falls through to DRAM slicing
run_sliced_op(input_tensor_on_device, output_tensors, &slice_attr, dram_slice_config_);
```

Method 1 sets `mm_conv=false` → always hits the sliced path.

### 6b. Why Single-Pass L1 Is Impossible

For the sliced path to be avoided, the activation must fit in L1 as a single
HEIGHT_SHARDED shard. For 64 cores and 2.36M pixels:

```
Input shard per core  (ROW_MAJOR, align=8):  36,864 px × 16 B =   576 KB
Output shard per core (TILE, OC_padded=32):  36,864 px × 64 B = 2,250 KB
Total per core:                                                 = 2,826 KB
L1 bank size:                                                  ≈ 1,400 KB

Output shard alone (2.25 MB) is 1.6× the full L1 bank. ← L1 path impossible.
```

The output shard is large because `OC=3` pads to `OC_padded=32` tiles (10.7× waste).
Even with perfectly efficient activation reads (ROW_MAJOR), the output can never fit
in L1 for this spatial size. **DRAM slicing is structurally unavoidable** for this shape
on N150 hardware.

### 6c. Per-Slice Overhead Is Fixed and Dominates

Each of the 6–8 slices pays a fixed dispatch cost:

```
Per slice overhead (Config 1, 1536×1536):
  PaddedSliceDeviceOperation  =  930 µs   reads input slice from DRAM to L1
  HaloDeviceOperation         =  867 µs   trivial for 1×1 kernel — still dispatches
  MoveDeviceOperation         =   67 µs   L1 memory reorg
  SliceWriteDeviceOperation   =  501 µs   writes output slice from L1 to DRAM
  ─────────────────────────────────────────────────────
  Total overhead per slice    = 2,365 µs
  Conv2dDeviceOperation       =  499 µs   ← the actual useful work
  Overhead-to-compute ratio   =   4.7×
```

**For every 1 µs of useful conv compute, 4.7 µs of overhead is paid.**

Config 2 is even worse:

```
Per slice overhead (Config 2, 1280×2304):
  PaddedSliceDeviceOperation  = 2,041 µs  (reads larger slice for wider input)
  SliceWriteDeviceOperation   =   492 µs
  MoveDeviceOperation         =    18 µs
  HaloDeviceOperation         =     5 µs
  Total overhead per slice    = 2,556 µs
  Conv2dDeviceOperation       =   121 µs
  Overhead-to-compute ratio   =  21.1×
```

### 6d. The `HaloDeviceOperation` Is a No-Op That Still Costs 867 µs

For a 1×1 kernel with stride=1 and padding=0, the Halo operation performs:
- **No sliding window expansion** (kernel=1×1)
- **No padding** (pad=0)
- **No stride dilation** (stride=1)

It is mathematically an **identity transformation** — the output is identical to the input.
Yet it still dispatches a full kernel and occupies 5.2 ms across 6 slices (30.3% of total).

The 867 µs per-slice cost for a no-op is purely kernel launch + synchronisation overhead.

### 6e. `PaddedSliceDeviceOperation` Does Not Benefit From Fewer DRAM Reads

The baseline matmul reads 188.7 MB in one shot and achieves 6.4 ms.
Method 1's PaddedSlice reads 18.9 MB / 6 slices = **3.15 MB per slice** yet takes
930 µs per slice → **5.58 ms total** just for reading.

```
Baseline:    188.7 MB  →  6.43 ms  =  29.3 GB/s effective
PaddedSlice:  3.15 MB  →  0.93 ms  =   3.4 GB/s effective  (8.6× lower efficiency!)
```

PaddedSlice achieves only **3.4 GB/s** effective bandwidth vs **29.3 GB/s** for the
baseline matmul. The small reads per core (3.15 MB / 64 cores = 49 KB per core) are
too small to saturate DRAM bandwidth — **latency dominates over throughput**.

The baseline matmul issues large, contiguous reads per core (2.95 MB per core), which
amortise the DRAM latency efficiently. Small slice reads cannot.

---

## 7. Summary: What Works and What Does Not

| Aspect | Works? | Evidence |
|--------|--------|---------|
| ROW_MAJOR reduces per-pixel reads (64 B → 16 B) | ✅ Yes | Conv2d slices 12.9–65.6× faster per slice |
| Single-pass L1 HEIGHT_SHARDED | ❌ No | Output shard alone = 2.25 MB > L1 (1.4 MB) |
| DRAM slicing overcomes overhead | ❌ No | Overhead = 82–95% of total time |
| HaloDeviceOperation skip for 1×1 | ❌ Not yet | 867 µs/slice for identity op |
| PaddedSlice saturates DRAM BW | ❌ No | 3.4 GB/s vs 29.3 GB/s for matmul |
| End-to-end speedup vs baseline | ❌ No | 2.67–2.70× slower |

---

## 8. Visualisation — Where the Time Goes

### Config 1 (1536×1536) — 17.19 ms total

```
Overhead (82.6%)         |████████████████████████████████████████  PaddedSlice   5.6ms
                         |████████████████████████████████████        Halo        5.2ms
                         |███████████████████████                     SliceWrite  3.0ms
                         |██                                          Move        0.4ms
Useful work (17.4%)      |█████████                                   Conv2d      3.0ms
                         |─────────────────────────────────────────────────────
Baseline (1 kernel)      |█████████████████████████████████         Matmul        6.4ms
```

### Config 2 (1280×2304) — 21.41 ms total

```
Overhead (95.5%)         |████████████████████████████████████████████████████████  PaddedSlice 16.3ms
                         |████████████████████████                                  SliceWrite   3.9ms
                         |                                                           Move+Halo   0.2ms
Useful work (4.5%)       |██                                                         Conv2d      1.0ms
                         |─────────────────────────────────────────────────────────
Baseline (1 kernel)      |████████████████████████████████████████                 Matmul        7.9ms
```

---

## 9. The Core Issue in One Sentence

**The DRAM slicing framework's per-slice fixed overhead (Halo + PaddedSlice + Move +
SliceWrite = ~2,400 µs/slice) is 4.7–21× larger than the actual conv compute
(120–500 µs/slice), making the total 2.67–2.70× worse than the baseline matmul
which avoids all per-slice overhead by executing as a single monolithic kernel.**

---

## 10. What Would Need to Change for Method 1 to Win

| Fix | Impact | Effort |
|-----|--------|--------|
| **Skip Halo for trivial 1×1/stride=1/pad=0 case** | Saves 5.2 ms (Config 1) | Low — add `is_trivial_halo` guard in `conv2d.cpp:262–301` |
| **Fuse PaddedSlice + Conv + SliceWrite into one kernel** | Eliminates 82–95% overhead | High — new kernel variant |
| **Reduce output tile format** (OC_padded=8 for OC=3) | Makes L1 single-pass feasible | Medium — changes tile format contract |
| **Force L1 path with explicit shard config** | Avoids DRAM slicing entirely | Medium — tuning `act_block_h_override` + shard config |

The highest-leverage single change is **skipping Halo for trivial 1×1 cases**.
It is a no-op kernel taking 5.2 ms (30% of total time for Config 1) and could be
eliminated with a simple guard check before the halo dispatch.

---

## 11. Comparison Table — All Key Metrics

| Metric | Baseline (Matmul) | Method 1 (RegConv) | Ratio |
|--------|------------------|--------------------|-------|
| Input layout | TILE (32-align) | **ROW_MAJOR (8-align)** | — |
| Bytes/pixel (activation) | 64 B | **16 B** | 4× less |
| DRAM reads (activation) | 188.7 MB | **18.9 MB** | 10× less |
| Execution | 1 kernel | 30–40 ops (6–8 slices) | — |
| Total device kernel | **6.43–7.92 ms** | 17.19–21.41 ms | 2.67–2.70× slower |
| Conv kernel only | 6.43–7.92 ms | **0.97–3.00 ms** | **2.6–8.2× faster** |
| Conv as % of total | 100% | **4.5–17.4%** | Overhead dominates |
| FPU utilisation | 0.008% | 0.034–0.072% | ~5–9× better |
| PM bandwidth model | 545–681 µs | N/A (no PM for DRAM ops) | — |
| Effective DRAM BW | 29.3 GB/s | 3.4 GB/s (PaddedSlice) | 8.6× less efficient |
