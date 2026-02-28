# MM+RS Fused Op Test Results - Default Big Size (xlarge)

## Test File
`tests/nightly/t3000/ccl/default_big_size_test.py`

## Configuration
- **Tensor Size**: M=4096, K=512, N=4096 (biggest xlarge from bklockiewicz's branch)
- **Blocks**: mm_block_m=256, mm_block_k=256, mm_block_n=256
- **Mesh**: Galaxy 8x4 (32 devices)
- **Ring Size**: 4 (cluster_axis=1)
- **RS Output N per device**: 4096/4 = 1024

## Core Grid Tests Summary

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) |
|-----------|-------------|---------|--------|----------|----------|----------|
| 4x8       | 32          | xlarge_4x8 | **PASS** | 1910.78 | 2003.55 | 2068.80 |
| 8x4       | 32          | xlarge_8x4 | **PASS** | 1835.35 | 1879.45 | 1933.20 |
| 7x8       | 56          | xlarge_7x8 | **FAIL** | - | - | - |
| 8x6       | 48          | xlarge_8x6 | **FAIL** | - | - | - |
| 8x7       | 56          | xlarge_8x7 | **FAIL** | - | - | - |
| 8x8       | 64          | xlarge_8x8 | **PASS** | 1821.23 | 1901.88 | 1974.68 |

---

## Test Commands & Results

### Test 1: 4x8 core grid - PASS

**Command:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_4x8" -v -s'
```

**Profiler Report:** `generated/profiler/reports/2026_02_27_23_54_30/ops_perf_results_2026_02_27_23_54_30.csv`

**Performance (32 devices):**
| Metric | Value |
|--------|-------|
| Device Kernel Avg | 2003.55 us |
| Device Kernel Min | 1910.78 us |
| Device Kernel Max | 2068.80 us |
| Host Duration Avg | 82.46 us |

---

### Test 2: 8x4 core grid - PASS

**Command:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x4" -v -s'
```

**Profiler Report:** `generated/profiler/reports/2026_02_27_23_57_21/ops_perf_results_2026_02_27_23_57_21.csv`

**Performance (32 devices):**
| Metric | Value |
|--------|-------|
| Device Kernel Avg | 1879.45 us |
| Device Kernel Min | 1835.35 us |
| Device Kernel Max | 1933.20 us |

---

### Test 3: 7x8 core grid - FAIL

**Command:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_7x8" -v -s'
```

**Error:**
```
TT_FATAL: slice_Wt (32) must be divisible by mm_N_block_wt_val (19)
```

**Explanation:**
- N=4096 → N_tiles = 4096/32 = 128 tiles
- Core grid X = 7 → slice_Wt = 128/7 ≈ 18.3 (not evenly divisible!)
- The kernel internally calculates mm_N_block_wt_val = 19 (rounded), but slice_Wt=32 is not divisible by 19
- **Root cause:** N dimension (4096 tiles=128) is not divisible by core grid X dimension (7)

---

### Test 4: 8x7 core grid - FAIL

**Command:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x7" -v -s'
```

**Error:**
```
TT_FATAL: slice_Ht (128) must be divisible by mm_cores_y_val (7)
```

**Explanation:**
- M=4096 → M_tiles = 4096/32 = 128 tiles
- Core grid Y = 7 → 128/7 ≈ 18.3 (not evenly divisible!)
- **Root cause:** M dimension (128 tiles) is not divisible by core grid Y dimension (7)

---

### Test 5: 8x8 core grid - PASS

**Command:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x8" -v -s'
```

**Profiler Report:** `generated/profiler/reports/2026_02_27_23_59_49/ops_perf_results_2026_02_27_23_59_49.csv`

**Performance (32 devices):**
| Metric | Value |
|--------|-------|
| Device Kernel Avg | 1901.88 us |
| Device Kernel Min | 1821.23 us |
| Device Kernel Max | 1974.68 us |

---
---

# Llama 70B 8K ISL - MM+RS Sizes

## MM+RS Operations in Llama 70B Prefill (per layer)

| Layer | M | K | N | Baseline MM (us) | Baseline RS (us) | Baseline Total (us) |
|-------|---|---|---|------------------|------------------|---------------------|
| FF1 | 8192 | 2048 | 3584 | 744.01 | 687.04 | 1431.05 |
| FF3 | 8192 | 2048 | 3584 | 752.07 | 716.24 | 1468.31 |
| FF2 | 8192 | 3584 | 2048 | 1318.09 | 492.23 | 1810.32 |
| Attn Out | 8192 | 1024 | 2048 | 474.63 | 480.33 | 954.96 |

**Total: 4 MM+RS per layer**
**Baseline source:** `~/teja/Allgather+matmul_fused_perf_results/baseline_8k/8k/prefill.csv`

## Test File
`tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py`

---

## FF1/FF3 Tests (M=8192, K=2048, N=3584)

**All tests use blocks 256/256/256**

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) | Error/PCC |
|-----------|-------------|---------|--------|----------|----------|----------|-----------|
| 4x8       | 32          | ff1_ff3_4x8 | **PASS** | 2974.51 | 3054.10 | 3148.63 | ≥0.99 |
| 8x4       | 32          | ff1_ff3_8x4 | **FAIL** | 3143.09 | 3182.85 | 3240.31 | PCC=0.952 |
| 7x8       | 56          | ff1_ff3_7x8 | **FAIL** | - | - | - | slice_Wt(28)%16≠0 |
| 8x8       | 64          | ff1_ff3_8x8 | **FAIL** | 3256.27 | 3318.25 | 3410.55 | PCC=0.943 |

---

## Llama 70B 128K ISL - FF1/FF3 (M=131072, K=2048, N=3584)

**Test File:** `tests/nightly/t3000/ccl/llama_size_128k_ISL_test.py`

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) | Error/PCC |
|-----------|-------------|---------|--------|----------|----------|----------|-----------|
| 4x8       | 32          | ff1_ff3_4x8 | RAN | 42394.66 | 42578.03 | 42869.05 | PCC check timeout |
| 8x4       | 32          | ff1_ff3_8x4 | **FAIL** | 43571.71 | 43793.65 | 44036.15 | PCC=0.968 |
| 7x8       | 56          | ff1_ff3_7x8 | **FAIL** | - | - | - | slice_Wt(28)%16≠0 |
| 8x8       | 64          | ff1_ff3_8x8 | RAN | 45702.97 | 45869.50 | 46159.79 | PCC check timeout |

---

## Llama 70B 8K ISL - FF2 (M=8192, K=3584, N=2048)

**Test File:** `tests/nightly/t3000/ccl/llama_ff2_8k_ISL_test.py`
**Baseline:** MM=1318.09 us + RS=492.23 us = **1810.32 us**
**N after RS:** 2048/4 = 512 → 16 tiles (better divisibility!)

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) | Error/PCC |
|-----------|-------------|---------|--------|----------|----------|----------|-----------|
| 4x8       | 32          | ff2_4x8 | **PASS** | 2844.69 | 2876.46 | 2919.76 | ≥0.99 |
| 8x4       | 32          | ff2_8x4 | **HUNG/KILLED** | - | - | - | Stuck >16min |
| 7x8       | 56          | ff2_7x8 | **FAIL** | - | - | - | slice_Wt (16) % mm_N_block_wt_val (10) != 0 |
| 8x8       | 64          | ff2_8x8 | **HUNG/KILLED** | - | - | - | Stuck >16min |

### Test Commands

**4x8:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_4x8" -v -s'
```

**8x4:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_8x4" -v -s'
```

**7x8:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_7x8" -v -s'
```

**8x8:**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_8x8" -v -s'
```
