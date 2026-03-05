# Llama 70B Model Prefill – AG+MM FF2 Step (All ISLs)

Single-device kernel time (µs) from profiler `prefill.csv`. 7×8 grid. **Non-fused 4 links** = default. **Non-fused 1 link** = `FF2_AG_1_LINK=1`. **Fused** = `USE_FUSED_AG_MM=1`.

**FF2 step** — Non-fused: show **AG µs + MM µs**. Fused: single op µs.

| ISL | Non-fused 4 links (Current) (AG µs + MM µs) | Non-fused 1 link (AG µs + MM µs) | Fused (µs) |
|-----|---------------------------------------------|-----------------------------------|------------|
| 8k  | 530.09 + 1321.09 = **1851.18**             | 1797.12 + 1321.42 = **3118.54**   | **2103.54** |
| 16k | 1014.37 + 2636.83 = **3651.20**             | 3575.95 + 2622.00 = **6197.95**   | **4087.96**|
| 32k | 2006.65 + 4984.26 = **6990.91**             | 7137.84 + 4992.73 = **12130.57**   | **8077.02** |
| 64k | 4011.54 + 9861.26 = **13872.80**           | 14277.38 + 9854.28 = **24131.66**  | **16142.44** |
| 128k| 8043.93 + 19644.87 = **27688.80**          | ~28150 + ~19645 = **~47600** (projected) | **32256.41** |

Current implementation numbers are already better than fused ( Current Non-fused (column 1) <  Fused (column 3)) .
AllGather bandwidth seems to be limited due to num_links 1 in fused op.
Current implementation already uses 4 num_links for AG ( see times of AG column 1 vs column 2 ),
 MM times remain same as they use 7x8 grid in all cases.

**Prefill single layer total (µs)** — sum of all kernel durations in that run’s `prefill.csv`

| ISL | Non-fused 4 links (Current) | Non-fused 1 link | Fused |
|-----|----------------------------|------------------|-------|
| 8k  | 14105.25                  | 15414.83         | 14360.91 |
| 16k | 26674.02                  | 29193.34         | 27117.82 |
| 32k | 57215.54                  | 62397.62         | 58308.36 |
| 64k | 137454.82                 | 147842.52        | 139795.41 |
| 128k| 373829.37                 | ~404000 (projected) | 378811.66 |

**How to fill (FF2):** Non-fused = sum of FF2 `AllGatherAsyncDeviceOperation` (seq×896→3584) + `MatmulDeviceOperation` (seq×3584×2048). Fused = `AllGatherMinimalMatmulAsyncOp` → `KERNEL_DUR_us`.

---

## Commands (run from repo root)

For each ISL, run the 3 commands. Results: `profiler_sweep_results/<run-name>/<isl>/prefill.csv`.

### 8k

**1. Non-fused 4 links (baseline)**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_baseline_4links_8k
```

**2. Fused**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_fused_8k
```

**3. Non-fused 1 link**
```bash
FF2_AG_1_LINK=1 SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_baseline_1link_8k
```

### 16k

**1. Non-fused 4 links (baseline)**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 16k --run-name ag_mm_baseline_4links_16k
```

**2. Fused**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 16k --run-name ag_mm_fused_16k
```

**3. Non-fused 1 link**
```bash
FF2_AG_1_LINK=1 SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 16k --run-name ag_mm_baseline_1link_16k
```

### 32k

**1. Non-fused 4 links (baseline)**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 32k --run-name ag_mm_baseline_4links_32k
```

**2. Fused**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 32k --run-name ag_mm_fused_32k
```

**3. Non-fused 1 link**
```bash
FF2_AG_1_LINK=1 SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 32k --run-name ag_mm_baseline_1link_32k
```

### 64k

**1. Non-fused 4 links (baseline)**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 64k --run-name ag_mm_baseline_4links_64k
```

**2. Fused**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 64k --run-name ag_mm_fused_64k
```

**3. Non-fused 1 link**
```bash
FF2_AG_1_LINK=1 SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 64k --run-name ag_mm_baseline_1link_64k
```

### 128k

**1. Non-fused 4 links (baseline)**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 128k --run-name ag_mm_baseline_4links_128k
```

**2. Fused**
```bash
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 128k --run-name ag_mm_fused_128k
```

**3. Non-fused 1 link**
```bash
FF2_AG_1_LINK=1 SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 128k --run-name ag_mm_baseline_1link_128k
```
