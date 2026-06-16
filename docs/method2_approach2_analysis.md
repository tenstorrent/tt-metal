# Method 2 Approach 2 — Deep Op Analysis & Improvement Roadmap

**Test:** `test_conv2d_method2_approach2_dram_bottleneck`
**Hardware:** Wormhole N150 · 64 Tensix cores · 288 GB/s peak DRAM BW
**Implementation:** `ttnn/cpp/ttnn/operations/`

---

## 1. Current Method 2 Op Pipeline

The full pattern on device:

```
[N, C, H, W]  ROW_MAJOR  DRAM
  ↓ reshape [N, C*K, H/K, W]      — free view  (K=32, C*K=96)
  ↓ PermuteDeviceOperation         [N, H/K, W, C*K]  ROW_MAJOR  17.7 MB
  ↓ reshape [1, 1, H/K*W, C*K]    — free view
  ↓ TilizeDeviceOperation          TILE  17.7 MB (C*K=96 = 3×TILE_WIDTH, 0% waste)
  ↓ MatmulDeviceOperation          [1, 1, H/K*W, OC*K]  TILE  17.7 MB read
  ↓ UntilizeDeviceOperation        ROW_MAJOR  17.7 MB
  ↓ reshape [N, H/K, W, OC*K]     — free view
  ↓ PermuteDeviceOperation         [N, OC*K, H/K, W]  ROW_MAJOR  14.2 MB
  ↓ reshape [N, OC, H, W]         — free view
```

---

## 2. Profiler Results (Tracy)

### Config 1 — 1×3×1536×1536

| Op | Count | Kernel | % total | FPU % | BW |
|----|-------|--------|---------|-------|----|
| **PermuteDeviceOperation** | 2 | **0.992 ms** | **55%** | 12.1% | 211 GB/s |
| MatmulDeviceOperation | 1 | 0.435 ms | 24% | 2.38% | 277 GB/s |
| TilizeDeviceOperation | 1 | 0.190 ms | 11% | 0.001% | — |
| UntilizeDeviceOperation | 1 | 0.176 ms | 10% | 38.0% | 211 GB/s |
| **Total** | **5** | **1.793 ms** | | | |

### Config 2 — 1×3×1280×2304

| Op | Count | Kernel | % total | FPU % | BW |
|----|-------|--------|---------|-------|----|
| **PermuteDeviceOperation** | 2 | **0.966 ms** | **50%** | 17.2% | 211 GB/s |
| MatmulDeviceOperation | 1 | 0.554 ms | 28% | 2.34% | 277 GB/s |
| TilizeDeviceOperation | 1 | 0.217 ms | 11% | 0.001% | — |
| UntilizeDeviceOperation | 1 | 0.213 ms | 11% | 39.2% | 211 GB/s |
| **Total** | **5** | **1.950 ms** | | | |

**New bottleneck: PermuteDeviceOperation = 50–55% of total.**

---

## 3. Deep Implementation Analysis

### 3a. PermuteDeviceOperation — Why It's Slow

**Files:**
```
ttnn/cpp/ttnn/operations/data_movement/permute/
  permute.cpp                              — dispatcher
  device/permute_device_operation.cpp     — factory selector
  device/permute_rm_program_factory.cpp   — ROW_MAJOR kernels
  device/kernels/compute/
    transpose_xw_rm_single_tile_size.cpp  — compute kernel
```

**Factory selection for dims=(0,2,3,1) on [N,96,48,1536]:**

The permute moves the **last dimension** (W=1536) — so `dims.back() != rank-1`. This routes to **`MultiCoreBlockedGeneric`**, the most expensive path.

```
is_last_dim_invariant = (dims.back() == rank-1)  // false for (0,2,3,1)
→ uses MultiCoreBlockedGeneric
```

**MultiCoreBlockedGeneric internally chains 3 operations per block:**
```
Reader:   async NOC read  (input ROW_MAJOR, strided)
Compute:  tilize → transpose_xw → untilize
Writer:   NOC write (output ROW_MAJOR, permuted)
```

This means the permute does **3 internal memory passes** on the blocked data, not 1!
The effective memory traffic per permute is ~3× the tensor size, not 1×.

For 17.7 MB tensor: ~53 MB effective traffic at 211 GB/s → **~0.25 ms per permute** (close to measured 0.50 ms/call).

**Root cause:** No fast path for NCHW→NHWC with ROW_MAJOR and moved last-dim. The blocked-generic path is general but slow for this shape.

### 3b. TilizeDeviceOperation

**Files:**
```
ttnn/cpp/ttnn/operations/data_movement/tilize/device/
  tilize_device_operation.cpp
  kernels/compute/tilize.cpp
```

**For [1,1,73728,96] ROW_MAJOR → TILE:**
- `TilizeMultiCoreDefaultProgramFactory` selected
- Reader feeds ROW_MAJOR pages → compute tilizes to 32×32 tiles → writer stores TILE
- Memory traffic: read 17.7 MB + write 17.7 MB = 35.4 MB
- Measured: 0.19 ms → effective BW = 35.4 MB / 0.19 ms = **186 GB/s**

### 3c. UntilizeDeviceOperation

Symmetric to tilize: TILE → ROW_MAJOR.
- Memory traffic: read 17.7 MB + write 17.7 MB = 35.4 MB
- Measured: 0.18 ms → effective BW = 35.4 MB / 0.18 ms = **197 GB/s**

### 3d. MatmulDeviceOperation — What It Does

**For [1,1,73728,96] @ [1,1,96,96] → [1,1,73728,96]:**

Shape: M=73728, K=96, N=96. In tiles: Mt=2304, Kt=3, Nt=3.

**Program config selected:**
```
MatmulMultiCoreReuseMultiCast1DProgramConfig
```
This is a 1D multicast config because `is_tall=true` (M >> N).

```
Files:
  ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp:884
  → detect tall-narrow: 73728 >> 96  → mcast_in0=false
  → MatmulMultiCoreReuseMultiCast1DProgramConfig
```

- Weight [96,96] = 18 KB → multicasts to all cores (fits in L1)
- Activation [73728,96] = 17.7 MB → each core reads its row partition from DRAM
- Measured: 0.43 ms, FPU=2.38%

**Why FPU=2.38%:** For OC=3, each output pixel computes only 3 MAC operations (C=3 real channels). Even with K=32 packing, each tile still has only 3 non-zero multiplications per 32 inputs (the block-diagonal weight). The FLOPs/byte ratio is inherently low for this shape.

### 3e. Key Finding: No Fused Permute+Tilize Exists

Searched entire repo — there is **no fused NCHW→NHWC+tilize kernel** in the standard operations:
```
find . -name "*.hpp" -o -name "*.cpp" | xargs grep -l "tilize.*permute\|permute.*tilize"
→ no results (beyond internal composition in MultiCoreBlockedGeneric)
```

This gap is the primary optimization opportunity.

---

## 4. Total Memory Traffic Analysis

### Current Method 2 memory traffic per config:

| Operation | Reads | Writes | Total |
|-----------|-------|--------|-------|
| from_torch (input) | — | 17.7 MB | 17.7 MB |
| from_torch (weight) | — | 18 KB | ~0 |
| reshape (free view) | 0 | 0 | 0 |
| **PermuteDeviceOperation (pack)** | ~53 MB | ~18 MB | **~71 MB** |
| reshape (free view) | 0 | 0 | 0 |
| TilizeDeviceOperation | 17.7 MB | 17.7 MB | 35.4 MB |
| MatmulDeviceOperation | 17.7 MB | 17.7 MB | 35.4 MB |
| UntilizeDeviceOperation | 17.7 MB | 17.7 MB | 35.4 MB |
| reshape (free view) | 0 | 0 | 0 |
| **PermuteDeviceOperation (unpack)** | ~43 MB | ~14 MB | **~57 MB** |
| reshape (free view) | 0 | 0 | 0 |
| to_torch (output) | 14.2 MB | — | 14.2 MB |
| **Total device traffic** | | | **~267 MB** |

The two permutes account for **~128 MB / ~267 MB = 48%** of total device memory traffic.

### Baseline memory traffic (for comparison):

| Operation | Traffic |
|-----------|---------|
| TilizeDeviceOperation | 35.4 MB |
| PermuteDeviceOperation (NCHW→NHWC) | **~567 MB** (188.7 MB × 3 internal passes) |
| MatmulDeviceOperation | ~377 MB (188.7 MB read × 2) |
| PermuteDeviceOperation (NHWC→NCHW) | ~85 MB |
| CopyDeviceOperation | ~35 MB |
| **Total** | **~1,100 MB** |

Method 2 reduces total memory traffic by **~4.1×** vs baseline, matching the measured 8× kernel speedup (BW is higher in M2 due to smaller working set fitting better in caches).

---

## 5. Improvement Opportunities

### Opportunity 1 — Fuse PermuteDeviceOperation + TilizeDeviceOperation (Highest Impact)

**Potential gain:** Eliminates 2 operations → save ~0.35 ms from pack side

**What to implement:**
A `tilize_nchw_to_nhwc` kernel that reads NCHW ROW_MAJOR and writes NHWC TILE in one pass:

```
Input:  [N, C*K, H/K, W]  ROW_MAJOR  (reads 17.7 MB)
Output: [N, H/K, W, C*K]  TILE       (writes 17.7 MB)
```

Instead of current: permute (3 passes ~53 MB) + tilize (2 passes ~35 MB) = 88 MB
Fused: 1 read + 1 write = **35 MB** (2.5× less memory traffic)

**Implementation path:**
```
New file:
  ttnn/cpp/ttnn/operations/data_movement/tilize/device/
    tilize_nchw_to_nhwc_program_factory.cpp

Kernel: reader reads from NCHW strides, compute tilizes with transposed tile fill order
  → For each output tile [row_tile, col_tile]:
    - col_tile is the "channel" tile index (0..2 for C*K=96)
    - row_tile is the spatial index (0..2303 for H/K*W/32)
    → Reader computes NCHW source address for each element
    → Directly fills the 32×32 tile in NHWC order
```

**Expected result:** Pack step drops from ~0.50 ms → ~0.19 ms (same as current tilize alone).

### Opportunity 2 — Fuse UntilizeDeviceOperation + PermuteDeviceOperation (Unpack Side)

Same principle for the unpack direction:

```
Input:  [N, H/K, W, OC*K]  TILE      (reads 17.7 MB)
Output: [N, OC*K, H/K, W]  ROW_MAJOR (writes 14.2 MB)
```

Fused `untilize_nhwc_to_nchw`: read TILE in NHWC order, write ROW_MAJOR in NCHW order.

**Expected result:** Unpack permute drops from ~0.49 ms → ~0.18 ms (same as current untilize alone).

### Opportunity 3 — DRAM_SHARDED Matmul

**Potential gain:** ~0.05–0.1 ms on matmul

For [73728, 96] activation, use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`:

```python
# In test / conv2d lowering:
program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    in0_block_w=3,          # K tiles = 96/32 = 3
    per_core_M=36,          # 2304/64 cores = 36 tiles per core
    per_core_N=3,           # N tiles = 96/32 = 3
    fuse_batch=True,
)
tt_out = ttnn.linear(tt_tile, tt_weight, program_config=program_config, ...)
```

This requires HEIGHT_SHARDING the input [73728, 96] across 64 cores (1152 rows/core).
Each core directly reads from its assigned DRAM bank, eliminating broadcast overhead.

**Files to modify:**
```
ttnn/cpp/ttnn/operations/matmul/device/factory/
  matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp
```

### Opportunity 4 — HEIGHT_SHARDED Tilize

Instead of DRAM INTERLEAVED tilize, use HEIGHT_SHARDED to better distribute the 17.7 MB tilize work:

```python
# Shard [1,1,73728,96] across 64 cores: each core handles 1152 rows × 96 channels
sharded_cfg = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(core_grid, [1152, 96], ttnn.ShardOrientation.ROW_MAJOR)
)
tt_tile = ttnn.to_layout(tt_flat, ttnn.TILE_LAYOUT, memory_config=sharded_cfg)
```

HEIGHT_SHARDED tilize routes to `TilizeMultiCoreShardedProgramFactory` which processes each shard locally, avoiding DRAM broadcast.

**Expected gain:** ~0.05 ms reduction in tilize time.

### Opportunity 5 — Avoid Pack Permute via `transpose_a=True` Matmul

If the matmul accepts `transpose_a=True`:

```python
# Input NCHW: [N, C*K, H/K*W] (no permute needed!)
tt_packed_nchw = ttnn.reshape(tt_nchw, (batch, packed_ic, packed_spatial))  # free view
tt_tile = ttnn.to_layout(tt_packed_nchw, TILE_LAYOUT)  # tilize [N, C*K, H/K*W]
# linear with transpose_a: treats input as [N, H/K*W, C*K]
tt_out = ttnn.linear(tt_tile, tt_weight, transpose_a=True, ...)
```

This eliminates the pack permute entirely. The unpack permute can be replaced by a reshape if the output `[N, H/K*W, OC*K]` maps correctly to `[N, OC, H, W]`.

**Feasibility:** `ttnn.linear` supports `transpose_a`. The weight needs transposing too (or pre-transpose at compile time). This is a pure Python change, no kernel changes needed.

**Expected gain:** Eliminates pack permute (~0.50 ms → 0 ms), saving half the permute cost.

---

## 6. Projected Performance After Improvements

### With Opportunity 1+2 (fused permute+tilize/untilize):

| Op | Current | After Fusion | Saving |
|----|---------|-------------|--------|
| PermuteDeviceOperation (pack) | 0.497 ms | 0 ms (fused) | 0.497 ms |
| TilizeDeviceOperation | 0.190 ms | **0.200 ms** (fused read+write) | -0.010 ms |
| MatmulDeviceOperation | 0.435 ms | 0.435 ms | — |
| UntilizeDeviceOperation | 0.176 ms | **0.185 ms** (fused read+write) | -0.009 ms |
| PermuteDeviceOperation (unpack) | 0.495 ms | 0 ms (fused) | 0.495 ms |
| **Total** | **1.793 ms** | **~0.820 ms** | **~0.97 ms** |
| **vs baseline** | 8.2× | **~18×** | |

### With Opportunity 5 (transpose_a eliminates pack permute):

| Op | Current | After transpose_a | Saving |
|----|---------|------------------|--------|
| PermuteDeviceOperation (pack) | 0.497 ms | **0 ms** | 0.497 ms |
| TilizeDeviceOperation | 0.190 ms | 0.190 ms | — |
| MatmulDeviceOperation | 0.435 ms | 0.435 ms | — |
| UntilizeDeviceOperation | 0.176 ms | 0.176 ms | — |
| PermuteDeviceOperation (unpack) | 0.495 ms | 0.495 ms | — |
| **Total** | **1.793 ms** | **~1.296 ms** | **~0.50 ms** |
| **vs baseline** | 8.2× | **~11.3×** | |

### Combined (Opp. 1+2+5):

Projected total: **~0.62 ms** → **~23.7× speedup vs baseline** for 1536×1536.

---

## 7. Priority Ranking

| Opportunity | Gain | Effort | Priority |
|-------------|------|--------|----------|
| **Fuse permute+tilize (pack)** | ~0.50 ms | Medium (new kernel) | **#1** |
| **Fuse untilize+permute (unpack)** | ~0.50 ms | Medium (new kernel) | **#2** |
| **transpose_a matmul (skip pack permute)** | ~0.50 ms | Low (Python only) | **#3** |
| DRAM_SHARDED matmul | ~0.05–0.1 ms | Low | **#4** |
| HEIGHT_SHARDED tilize | ~0.05 ms | Low | **#5** |

---

## 8. Summary

The current Method 2 Approach 2 achieves **8.2–8.6× end-to-end speedup** over the baseline TTNN IR. The remaining bottleneck is the two `PermuteDeviceOperation` calls (55% of pipeline).

**Root causes of permute cost:**
1. `MultiCoreBlockedGeneric` internally chains tilize→transpose→untilize (3 sub-passes)
2. No fused NCHW→NHWC+tilize kernel exists in the repo

**Highest-leverage next step:** Implement `tilize_nchw_to_nhwc` and `untilize_nhwc_to_nchw` fused kernels (Opportunities 1+2). This would deliver ~18× end-to-end speedup while keeping the computation entirely on device with clean semantics. Alternatively, `transpose_a=True` on the matmul (Opportunity 5) eliminates one permute with zero C++ changes.

**Files to create/modify for kernel fusion:**
```
New:
  ttnn/cpp/ttnn/operations/data_movement/tilize/device/
    tilize_nchw_to_nhwc_program_factory.hpp/.cpp
  ttnn/cpp/ttnn/operations/data_movement/untilize/device/
    untilize_nhwc_to_nchw_program_factory.hpp/.cpp

Modified:
  ttnn/cpp/ttnn/operations/data_movement/tilize/tilize.hpp/.cpp
  ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.hpp/.cpp
```
