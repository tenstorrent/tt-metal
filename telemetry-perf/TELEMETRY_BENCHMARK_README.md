# Comprehensive Telemetry Performance Benchmark Suite

This directory contains a statistically rigorous benchmark suite to measure tt-telemetry's performance impact on tt-metal workloads.

## Overview

The benchmark suite addresses all critical issues identified in review:
- ✅ Device state reset between tests (thermal/cache contamination)
- ✅ Validated telemetry startup
- ✅ Multiple comparison correction (Bonferroni-Holm)
- ✅ MMIO-only vs full mode comparison
- ✅ ERISC contention validation
- ✅ Memory-bound operations
- ✅ Multiple CCL operations
- ✅ Large tensor sizes (up to 10% DRAM)
- ✅ Extended frequency range (60s to 100us)
- ✅ Sustained workload testing

## Files

### Core Utilities
- `telemetry_benchmark_utils.py` - Shared utilities (device management, stats, telemetry control)

### Test Scripts
- `validate_mmio_only.py` - Core hypothesis test: validates --mmio-only prevents ERISC contention
- `comprehensive_single_device_benchmark.py` - Single-device tests (compute + memory-bound ops)
- `comprehensive_multi_device_benchmark.py` - Multi-device CCL tests (AllGather, ReduceScatter, AllReduce)
- `sustained_workload_test.py` - Long-running test for drift detection

### Orchestration & Analysis
- `run_telemetry_benchmark_suite.py` - Main orchestrator (runs all tests, generates report)
- `analyze_benchmark_results.py` - Post-analysis (impact distribution, recommendations)

## Quick Start

### Phase 1: Reduced Suite (~2-3 hours)

Validates core hypotheses with reduced test matrix:

```bash
cd /tmp
python3 run_telemetry_benchmark_suite.py --phase reduced
```

**Configuration:**
- 3 tensor sizes (1024², 8192², 17408²)
- 6 polling frequencies (60s, 1s, 100ms, 10ms, 1ms, 100us)
- 3 single-device operations (matmul, add, to_memory_config)
- 1 CCL operation (AllGather)
- 100 samples per config
- ~110 total configurations

**Output:**
- `/tmp/mmio_validation_results.json`
- `/tmp/single_device_results_reduced.json`
- `/tmp/multi_device_results_reduced.json`
- `/tmp/sustained_workload_results.json`
- `/tmp/telemetry_final_report_reduced.md`

### Phase 2: Full Suite (~9-12 hours)

Comprehensive analysis with full test matrix:

```bash
cd /tmp
python3 run_telemetry_benchmark_suite.py --phase full
```

**Configuration:**
- 5 tensor sizes (1024² to 17408²)
- 12 polling frequencies (60s to 100us)
- 5 single-device operations (matmul, add, concat, to_memory_config, reshape)
- 3 CCL operations (AllGather, ReduceScatter, AllReduce)
- L1 + DRAM memory configs
- Adaptive warmup
- ~300+ total configurations

## Running Individual Tests

### Core Hypothesis Validation

Tests whether --mmio-only prevents ERISC contention:

```bash
python3 /tmp/validate_mmio_only.py
```

**Expected outcome:**
- MMIO-only: <5% impact vs baseline
- Full mode: >10% impact or failures
- Validates --mmio-only is effective

### Single-Device Benchmark

```bash
# Reduced
python3 /tmp/comprehensive_single_device_benchmark.py reduced

# Full
python3 /tmp/comprehensive_single_device_benchmark.py full
```

### Multi-Device Benchmark

```bash
# Reduced
python3 /tmp/comprehensive_multi_device_benchmark.py reduced

# Full
python3 /tmp/comprehensive_multi_device_benchmark.py full
```

### Sustained Workload Test

```bash
# 1000 iterations
python3 /tmp/sustained_workload_test.py 1000

# 5 minutes duration
python3 /tmp/sustained_workload_test.py 300s
```

## Post-Analysis

After benchmarks complete, run additional analysis:

```bash
python3 /tmp/analyze_benchmark_results.py --phase reduced
```

**Generates:**
- Impact distribution analysis
- Frequency sensitivity analysis
- Production deployment recommendations
- `/tmp/telemetry_analysis_summary_reduced.md`

## Methodology Improvements

### Device State Management

**Problem:** Thermal/cache contamination between tests

**Solution:**
```python
# Clean state wrapper
def run_with_clean_state(test_func):
    cleanup_all_devices()
    thermal_cooldown(30s)
    result = test_func()
    cleanup_all_devices()
    return result
```

### Validated Telemetry Startup

**Problem:** Tests starting before telemetry ready

**Solution:**
```python
# Validation with retry
telemetry_proc = start_telemetry_validated(
    polling_interval="100ms",
    mmio_only=True,
    stabilization_sec=5.0
)
# Waits for health check + metrics fetch + 5s stabilization
```

### Adaptive Warmup

**Problem:** Fixed 10 iterations may be insufficient

**Solution:**
```python
# Continue until CV < 5%
adaptive_warmup(test_func, target_cv=0.05, max_iters=50)
```

### Interleaved Baselines

**Problem:** Baseline drift over time

**Solution:**
```python
baseline_before = run_baseline()
telemetry_result = run_with_telemetry()
baseline_after = run_baseline()

# Use average of bracketing baselines
baseline_mean = (baseline_before + baseline_after) / 2
```

### Multiple Comparison Correction

**Problem:** 54+ hypothesis tests without correction → false positives

**Solution:**
```python
# Bonferroni-Holm correction
p_corrected = apply_multiple_comparison_correction(
    all_p_values,
    method='holm',
    alpha=0.05
)
```

## Statistical Methods

### Significance Testing
- **Mann-Whitney U test**: Non-parametric alternative to t-test
- **Bonferroni-Holm correction**: Controls family-wise error rate
- **Effect size (Cohen's d)**: Quantifies practical significance

### Monotonicity Testing
- **Kendall tau correlation**: Tests for monotonic trend
- **Criteria**: tau > 0.3 AND p < 0.01 (corrected)

### Normality & Outliers
- **Shapiro-Wilk test**: Checks distribution normality
- **IQR method**: Detects outliers (k=1.5 for standard)

## Hardware Requirements

### Single-Device Tests
- 1x Wormhole device (minimum)
- Tests operations on single device

### Multi-Device Tests
- 8x Wormhole devices (T3000 configuration)
- Tests 2-device, 4-device, and 8-device configurations

### Memory Requirements
- Largest tensor: 17408×17408 = 578MB (~5% of 12GB DRAM)
- Tests will not exceed device memory

## Interpreting Results

### Core Hypothesis Test

**Success criteria:**
```
✓ HYPOTHESIS VALIDATED
  - MMIO-only mode shows <5% impact on all tests
  - Full mode shows ≥10% impact OR failures on all tests
  - Conclusion: --mmio-only successfully prevents ERISC contention
```

**Recommendation:** Use `--mmio-only` flag for multi-chip deployments

### Impact Distribution

**Acceptable:**
- Mean impact <2%
- <10% of tests show >5% impact
- No tests show >10% impact

**Concerning:**
- Mean impact >5%
- >50% of tests show >5% impact
- Any tests show >20% impact

### Monotonicity

**Interpretation:**
- Monotonic relationship indicates frequency-dependent impact
- Non-monotonic suggests other factors dominate
- If monotonic: higher frequencies show higher impact

### Drift Analysis

**Acceptable:**
- <5% drift over sustained workload
- Late samples similar to early samples

**Concerning:**
- >5% drift (indicates accumulation)
- Increasing variance over time

## Troubleshooting

### Tests Failing to Start

**Issue:** Devices not available

**Solution:**
```bash
# Reset devices
tt-smi -r 0
tt-smi -r 1
tt-smi -r 2
tt-smi -r 3

# Wait 30 seconds
sleep 30
```

### Telemetry Server Won't Start

**Issue:** Port already in use

**Solution:**
```bash
# Kill existing telemetry
pkill tt-telemetry

# Wait
sleep 5

# Verify port free
lsof -i :7070
```

### Out of Memory Errors

**Issue:** Large tensor tests failing

**Solution:**
```python
# Reduce tensor sizes in test config
SINGLE_DEVICE_SHAPES_REDUCED = [
    (1, 1, 1024, 1024),   # 2MB
    (1, 1, 4096, 4096),   # 32MB
    (1, 1, 8192, 8192),   # 128MB
]
```

### Tests Taking Too Long

**Issue:** Full suite exceeds available time

**Solution:**
```bash
# Run reduced suite first
python3 run_telemetry_benchmark_suite.py --phase reduced

# Or run tests individually
python3 validate_mmio_only.py
```

## Expected Results

### Core Hypothesis

**Expected:** VALIDATED
- MMIO-only prevents ERISC contention
- Full mode shows measurable impact on multi-chip

### Single-Device Impact

**Expected:** MINIMAL (<2% average)
- Telemetry should not significantly impact single-device workloads
- Compute-bound operations less sensitive than memory-bound

### Multi-Device Impact

**Expected:** MODE-DEPENDENT
- MMIO-only: <5% impact
- Full mode: >10% impact or failures

### Monotonicity

**Expected:** PRESENT in some operations
- Higher polling frequencies → higher impact
- Memory-bound operations more sensitive

### Sustained Workload

**Expected:** NO DRIFT
- Performance stable over 1000 iterations
- No accumulation effects

## Production Recommendations

Based on benchmark results, typical recommendation:

```bash
# For multi-chip deployments
./build/tools/tt-telemetry/tt-telemetry \
  --mmio-only \
  --logging-interval 100ms \
  --port 7070
```

**Rationale:**
- `--mmio-only`: Prevents ERISC contention (<5% impact)
- `100ms`: Good balance between granularity and overhead
- `--port 7070`: Standard telemetry port

**Alternative for single-chip:**
```bash
# Single-chip doesn't need --mmio-only
./build/tools/tt-telemetry/tt-telemetry \
  --logging-interval 100ms \
  --port 7070
```

## Data Files

### Result Files (JSON)
- `mmio_validation_results.json` - Core hypothesis test
- `single_device_results_{phase}.json` - Single-device benchmark
- `multi_device_results_{phase}.json` - Multi-device benchmark
- `sustained_workload_results.json` - Sustained test
- `telemetry_analysis_{phase}.json` - Post-analysis data

### Report Files (Markdown)
- `telemetry_final_report_{phase}.md` - Main benchmark report
- `telemetry_analysis_summary_{phase}.md` - Analysis summary

### Intermediate Files
- `*_partial.json` - Saved every 10 tests for crash recovery

## Development

### Adding New Operations

```python
# In SingleDeviceOperations or MultiDeviceOperations class
def run_my_operation(self, shape, memory_config):
    # Setup
    input_tensor = ttnn.from_torch(...)

    # Measure
    start = time.perf_counter()
    output_tensor = ttnn.my_operation(input_tensor)
    ttnn.synchronize_device(self.device)
    end = time.perf_counter()

    # Cleanup
    del output_tensor, input_tensor

    return end - start
```

### Adding New Frequencies

```python
# In test configuration
POLLING_FREQUENCIES_CUSTOM = [
    "60s", "30s", "10s", "5s", "1s",
    "500ms", "100ms", "50ms", "10ms",
    "5ms", "1ms", "500us", "100us"
]
```

### Customizing Sample Size

```python
# In test scripts
N_SAMPLES = 200  # Increase for more statistical power
WARMUP_ITERS = 50  # Increase for better convergence
```

## Citation

If using this benchmark suite in research or publications:

```
Comprehensive Telemetry Performance Benchmark Suite for tt-metal
Addresses ERISC contention, implements statistical rigor (Bonferroni-Holm correction,
Mann-Whitney U tests, Kendall tau monotonicity testing), validates --mmio-only mitigation.
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review error logs in stdout/stderr
3. Check intermediate results files (*_partial.json)
4. Report issues with full error messages

## License

Part of tt-metal project. See main repository for license information.
