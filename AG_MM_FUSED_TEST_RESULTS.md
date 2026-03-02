# AG+MM Fused Op Test Results

Testing `ttnn.experimental.strided_all_gather_minimal_matmul_async` on Galaxy 8x4 mesh (32 devices).
Same structure as MM+RS: default big size, 8K ISL, 128K ISL with notes per config.

---

## Default Big Size (xlarge)

**Test File:** `tests/nightly/t3000/ccl/ag_mm_default_big_size_test.py`

**Configuration:**
- Tensor: M=4096, K=4096, N=4096
- Blocks: 256/256/256 (K_block=8, M_block=8, N_block=8 in tiles)
- Mesh: Galaxy 8x4, ring along cluster_axis=1
- K_tiles = 4096/32 = 128

| K | K_block | Grid | Cores | Test ID | Result | Min (us) | Avg (us) | Max (us) | Notes |
|---|---------|------|-------|---------|--------|----------|----------|----------|-------|
| 4096 | 8 | 4x8 | 32 | xlarge_4x8 | PENDING | - | - | - | K_tiles=128, 128/8=16 ✓ |
| 4096 | 8 | 8x4 | 32 | xlarge_8x4 | PENDING | - | - | - | K_tiles=128, 128/8=16 ✓ |
| 4096 | 8 | 7x8 | 56 | xlarge_7x8 | PENDING | - | - | - | K_tiles=128, 128/7≈18.3 X? |
| 4096 | 8 | 8x8 | 64 | xlarge_8x8 | PENDING | - | - | - | K_tiles=128, 128/8=16 ✓ |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_default_big_size_test.py -k "xlarge_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_default_big_size_test.py -k "xlarge_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_default_big_size_test.py -k "xlarge_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_default_big_size_test.py -k "xlarge_8x8" -v -s'
```

---

## Llama 70B 8K ISL - Attn Out (AG+MM)

**Test File:** `tests/nightly/t3000/ccl/ag_mm_llama_8k_ISL_test.py`

**Configuration:**
- Tensor: M=8192, K=1024, N=2048 (Attn WO after SDPA)
- Blocks: 256/256/256 (K_block=8)
- **Baseline (separate AG + MM):** AG + MM = 474.63 us + 480.33 us ≈ 955 us (from 8K prefill.csv)
- K_tiles = 1024/32 = 32

| K | K_block | Grid | Cores | Test ID | Result | Min (us) | Avg (us) | Max (us) | Notes |
|---|---------|------|-------|---------|--------|----------|----------|----------|-------|
| 1024 | 8 | 4x8 | 32 | attn_out_4x8 | PENDING | - | - | - | K_tiles=32, 32/8=4 ✓ |
| 1024 | 8 | 8x4 | 32 | attn_out_8x4 | PENDING | - | - | - | K_tiles=32, 32/8=4 ✓ |
| 1024 | 8 | 7x8 | 56 | attn_out_7x8 | PENDING | - | - | - | K_tiles=32, 32/7≈4.57 X? |
| 1024 | 8 | 8x8 | 64 | attn_out_8x8 | PENDING | - | - | - | K_tiles=32, 32/8=4 ✓ |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_8k_ISL_test.py -k "attn_out_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_8k_ISL_test.py -k "attn_out_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_8k_ISL_test.py -k "attn_out_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_8k_ISL_test.py -k "attn_out_8x8" -v -s'
```

---

## Llama 70B 128K ISL - Attn Out (AG+MM)

**Test File:** `tests/nightly/t3000/ccl/ag_mm_llama_128k_ISL_test.py`

**Configuration:**
- Tensor: M=131072, K=1024, N=2048
- Blocks: 256/256/256 (K_block=8)
- **Baseline (separate AG + MM):** AG + MM = 6487.70 us + 6531.59 us ≈ 13019 us (from 128K prefill.csv)
- K_tiles = 1024/32 = 32

| K | K_block | Grid | Cores | Test ID | Result | Min (us) | Avg (us) | Max (us) | Notes |
|---|---------|------|-------|---------|--------|----------|----------|----------|-------|
| 1024 | 8 | 4x8 | 32 | attn_out_4x8 | PENDING | - | - | - | K_tiles=32, 32/8=4 ✓ |
| 1024 | 8 | 8x4 | 32 | attn_out_8x4 | PENDING | - | - | - | K_tiles=32, 32/8=4 ✓ |
| 1024 | 8 | 7x8 | 56 | attn_out_7x8 | PENDING | - | - | - | K_tiles=32, 32/7≈4.57 X? |
| 1024 | 8 | 8x8 | 64 | attn_out_8x8 | PENDING | - | - | - | K_tiles=32, 32/8=4 ✓ |

### Test Commands

```bash
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_128k_ISL_test.py -k "attn_out_4x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_128k_ISL_test.py -k "attn_out_8x4" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_128k_ISL_test.py -k "attn_out_7x8" -v -s'
python tools/tracy/profile_this.py -c 'pytest tests/nightly/t3000/ccl/ag_mm_llama_128k_ISL_test.py -k "attn_out_8x8" -v -s'
```

---

## Result legend

- **✔ Works** – PASS, PCC ≥ 0.99 (fill Min/Avg/Max from profiler)
- **X PCC ~0.xx** – Run completed but PCC below threshold (fill timings if available)
- **X PCC bad** – Numerical failure, note in Notes
- **X Hangs** – Test hung / killed (no timings)
- **X FAIL** – Assert / TT_FATAL (e.g. divisibility); put error summary in Notes

Update the tables above as you run each test and fill Result, timings, and Notes (e.g. divisibility like `K_tiles/K_block` or error messages).
