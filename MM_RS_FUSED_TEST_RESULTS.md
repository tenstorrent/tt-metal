# MM+RS Fused Op Test Results

Testing `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async` on Galaxy 8x4 mesh (32 devices).

---

## Default Big Size (xlarge)

**Test File:** `tests/nightly/t3000/ccl/default_big_size_test.py`

**Configuration:**
- Tensor: M=4096, K=512, N=4096
- Blocks: 256/256/256
- Mesh: Galaxy 8x4, ring_size=4

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) |
|-----------|-------------|---------|--------|----------|----------|----------|
| 4x8       | 32          | xlarge_4x8 | **PASS** | 1910.78 | 2003.55 | 2068.80 |
| 8x4       | 32          | xlarge_8x4 | **PASS** | 1835.35 | 1879.45 | 1933.20 |
| 7x8       | 56          | xlarge_7x8 | **FAIL** | - | - | - |
| 8x6       | 48          | xlarge_8x6 | **FAIL** | - | - | - |
| 8x7       | 56          | xlarge_8x7 | **FAIL** | - | - | - |
| 8x8       | 64          | xlarge_8x8 | **PASS** | 1821.23 | 1901.88 | 1974.68 |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x6" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x7" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/default_big_size_test.py -k "xlarge_8x8" -v -s'
```

---

## Llama 70B 8K ISL - Baseline

| Layer | M | K | N | Baseline MM (us) | Baseline RS (us) | Baseline Total (us) |
|-------|---|---|---|------------------|------------------|---------------------|
| FF1 | 8192 | 2048 | 3584 | 744.01 | 687.04 | 1431.05 |
| FF3 | 8192 | 2048 | 3584 | 752.07 | 716.24 | 1468.31 |
| FF2 | 8192 | 3584 | 2048 | 1318.09 | 492.23 | 1810.32 |
| Attn Out | 8192 | 1024 | 2048 | 474.63 | 480.33 | 954.96 |

---

## Llama 70B 8K ISL - FF1/FF3 (M=8192, K=2048, N=3584)

**Test File:** `tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py`
**Blocks:** 256/256/256
**Baseline:** MM=744 us + RS=687 us = 1431 us

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) | Error/PCC |
|-----------|-------------|---------|--------|----------|----------|----------|-----------|
| 4x8       | 32          | ff1_ff3_4x8 | **PASS** | 2974.51 | 3054.10 | 3148.63 | ≥0.99 |
| 8x4       | 32          | ff1_ff3_8x4 | **FAIL** | 3143.09 | 3182.85 | 3240.31 | PCC=0.952 |
| 7x8       | 56          | ff1_ff3_7x8 | **FAIL** | - | - | - | slice_Wt(28)%16≠0 |
| 8x8       | 64          | ff1_ff3_8x8 | **FAIL** | 3256.27 | 3318.25 | 3410.55 | PCC=0.943 |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_8k_ISL_test.py -k "ff1_ff3_8x8" -v -s'
```

---

## Llama 70B 8K ISL - FF2 (M=8192, K=3584, N=2048)

**Test File:** `tests/nightly/t3000/ccl/llama_ff2_8k_ISL_test.py`
**Blocks:** 256/256/256
**Baseline:** MM=1318 us + RS=492 us = 1810 us

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) | Error/PCC |
|-----------|-------------|---------|--------|----------|----------|----------|-----------|
| 4x8       | 32          | ff2_4x8 | **PASS** | 2844.69 | 2876.46 | 2919.76 | ≥0.99 |
| 8x4       | 32          | ff2_8x4 | **HUNG/KILLED** | - | - | - | Stuck >16min |
| 7x8       | 56          | ff2_7x8 | **FAIL** | - | - | - | slice_Wt(16)%10≠0 |
| 8x8       | 64          | ff2_8x8 | **HUNG/KILLED** | - | - | - | Stuck >16min |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_ff2_8k_ISL_test.py -k "ff2_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_ff2_8k_ISL_test.py -k "ff2_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_ff2_8k_ISL_test.py -k "ff2_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_ff2_8k_ISL_test.py -k "ff2_8x8" -v -s'
```

---

## Llama 70B 128K ISL - FF1/FF3 (M=131072, K=2048, N=3584)

**Test File:** `tests/nightly/t3000/ccl/llama_size_128k_ISL_test.py`
**Blocks:** 256/256/256

| Core Grid | Total Cores | Test ID | Status | Min (us) | Avg (us) | Max (us) | Error/PCC |
|-----------|-------------|---------|--------|----------|----------|----------|-----------|
| 4x8       | 32          | ff1_ff3_4x8 | RAN | 42394.66 | 42578.03 | 42869.05 | PCC check timeout |
| 8x4       | 32          | ff1_ff3_8x4 | **FAIL** | 43571.71 | 43793.65 | 44036.15 | PCC=0.968 |
| 7x8       | 56          | ff1_ff3_7x8 | **FAIL** | - | - | - | slice_Wt(28)%16≠0 |
| 8x8       | 64          | ff1_ff3_8x8 | RAN | 45702.97 | 45869.50 | 46159.79 | PCC check timeout |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_128k_ISL_test.py -k "ff1_ff3_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_128k_ISL_test.py -k "ff1_ff3_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_128k_ISL_test.py -k "ff1_ff3_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/llama_size_128k_ISL_test.py -k "ff1_ff3_8x8" -v -s'
```
