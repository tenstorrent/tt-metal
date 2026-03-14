# AG+MM Fused Op Testing Guide

## Overview

This guide covers testing the fused AllGather+MatMul operation for Llama 70B FF2 layer.

**Two testing approaches:**
1. **Model-level**: Run Llama 70B prefill with fused/non-fused FF2 paths
2. **Standalone**: Run unit test with configurable grid sizes and link counts

---

## 1. Model-Level Testing (Single Prefill Layer)

### Run Profiler Sweep Script

```bash
cd ~/Metal/tt-metal

# Non-fused path (7x8 grid, 3 links for AG, default)
./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name nonfused_7x8

# Fused path (6x8 grid, 3 links)
USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name fused_6x8
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FUSED_AG_MM` | `0` | Set to `1` to enable fused AG+MM for FF2 |
| `FF2_AG_1_LINK` | `0` | Set to `1` to use 1 link for AG in non-fused path |
| `SKIP_PREFILL_WARMUP` | `0` | Set to `1` to skip warmup (auto-set by script) |

### Grid Configurations

| Path | Grid | Cores | Links |
|------|------|-------|-------|
| Non-fused (default) | 7×8 | 56 | 3 (separate AG) |
| Fused | 6×8 | 48 | 3 (fused) |

---

## 2. Standalone Unit Test (8x8 4-link)

### Run with Profiler

```bash
cd ~/Metal/tt-metal

# Fused 8x8 4 links (Llama 8k FF2: M=8192, K=3584, N=2048)
python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py::test_ag_mm[silicon_arch_name=wormhole_b0-fused-8x8_4links-llama_8k_ff2-galaxy] -v"

# Separate 8x8 4 links
python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py::test_ag_mm[silicon_arch_name=wormhole_b0-separate-8x8_4links-llama_8k_ff2-galaxy] -v"
```

### Available Test Configurations

**Grid/Link options:**
- `8x8_4links` - 64 cores, 4 links
- `8x8_2links` - 64 cores, 2 links
- `6x8_3links` - 48 cores, 3 links
- `6x8_2links` - 48 cores, 2 links
- `4x8_2links` - 32 cores, 2 links
- `7x8_1link` - 56 cores, 1 link
- `8x7_4links` - 56 cores, 4 links (transposed)

**Size options:**
- `wan2_4k4k4k` - M=4096, K=4096, N=4096
- `llama_8k_ff2` - M=8192, K=3584, N=2048
- `llama_128k_ff2` - M=131072, K=3584, N=2048

---

## 3. Analyzing Profiler Results

### Locate Output CSV

Profiler outputs are in:
```
~/Metal/tt-metal/generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

### Extract Clean Single-Layer CSV

```bash
cd ~/Metal/tt-metal

# Find signpost locations
grep -n "signpost" generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv

# Example output:
# 3170:start,signpost,...
# 6339:stop,signpost,...

# Extract between signposts (adjust line numbers based on grep output)
(head -1 generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv && \
 sed -n '3171,6338p' generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv) > clean_single_prefill_layer.csv
```

### Quick Check Key Ops

```bash
# Check AG and MM durations (Device 0)
echo "=== AllGather ===" && \
grep "AllGatherAsyncDeviceOperation" clean_single_prefill_layer.csv | \
awk -F',' '$4==0 {printf "Cores: %s | Duration: %.0f µs\n", $8, $19/1000}'

echo "=== MinimalMatmul ===" && \
grep "MinimalMatmulDeviceOperation" clean_single_prefill_layer.csv | \
awk -F',' '$4==0 {printf "Cores: %s | Duration: %.0f µs\n", $8, $19/1000}'

echo "=== Fused AG+MM ===" && \
grep "AllGatherMinimalMatmulAsyncOp" clean_single_prefill_layer.csv | \
awk -F',' '$4==0 {printf "Cores: %s | Duration: %.0f µs\n", $8, $19/1000}'
```

### Generate tt-perf-report

```bash
cd ~/Metal/tt-metal

# Generate report from clean CSV
tt-perf-report clean_single_prefill_layer.csv --csv tt-perf-output.csv

# Output files:
# - tt-perf-output.csv (detailed op analysis)
# - tt-perf-output_stacked.csv (stacked report)
# - tt-perf-output_stacked.png (visualization)
```

### View Key Ops from tt-perf-report

```bash
# Show AG and MM ops
grep -E "AllGather|MinimalMatmul" tt-perf-output.csv
```

---

## 4. Expected Results (8x8 4-link, Llama 8k FF2)

| Path | Op | Cores | Duration |
|------|-----|-------|----------|
| **FUSED** | AllGatherMinimalMatmulAsyncOp | 72 | ~1,810 µs |
| **SEPARATE** | AllGatherAsync | 32 | ~685 µs |
| **SEPARATE** | MinimalMatmul | 64 | ~1,665 µs |
| **SEPARATE** | **Total** | - | ~2,350 µs |

**Fused wins by ~540 µs (~23% faster)**

---

## 5. File Locations

| File | Description |
|------|-------------|
| `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` | FF2 fused/non-fused path selection |
| `models/demos/llama3_70b_galaxy/tt/llama_ccl.py` | `line_all_gather_matmul()` implementation |
| `models/demos/llama3_70b_galaxy/tt/model_config.py` | Grid configs (6x8 vs 7x8) |
| `scripts/run_profiler_sweep.sh` | Profiler sweep script |
| `tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py` | Standalone test |
