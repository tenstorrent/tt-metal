# Quick Start: Experts Matmul Unit Tests

This guide provides a fast path to running the experts matmul unit tests and optimizing performance.

## Prerequisites

- Single TT device (Wormhole or Blackhole)
- Standard test environment with pytest
- GPT-OSS model installed

## 1. Run Tests Quickly

### Using the Helper Script

The easiest way to run tests is using the provided shell script:

```bash
cd /localdev/handrews/tt-metal

# Run stress test (100 iterations to verify stability)
./models/demos/gpt_oss/tests/ops/run_experts_matmul_tests.sh stress

# Run parameter sweep (find optimal configuration)
./models/demos/gpt_oss/tests/ops/run_experts_matmul_tests.sh sweep

# Run both
./models/demos/gpt_oss/tests/ops/run_experts_matmul_tests.sh both
```

### Using pytest Directly

```bash
# Stress test
pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_stress -v -s

# Parameter sweep
pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_param_sweep -v -s
```

## 2. Analyze Results

After running the parameter sweep, analyze the results:

```bash
# Analyze most recent results
python models/demos/gpt_oss/tests/ops/analyze_param_sweep.py

# Analyze specific results file
python models/demos/gpt_oss/tests/ops/analyze_param_sweep.py param_sweep_results/experts_matmul_sweep_batch32_seq1.json

# Export to CSV for external analysis
python models/demos/gpt_oss/tests/ops/analyze_param_sweep.py --csv results.csv
```

The analysis script will show:
- Summary statistics (pass/fail counts, performance ranges)
- Top 10 best configurations
- Parameter impact analysis (which parameters matter most)
- Optimal configuration with detailed metrics

### Profiling with Tracy

To get device-level metrics, wrap the test with Tracy externally:

```bash
# Profile the entire parameter sweep
python -m tracy -p -o experts_matmul_sweep \\
    -a device_kernel_duration \\
    -m "pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_param_sweep -v"

# Results in: .tt_metal_home/profiler_artifacts/experts_matmul_sweep/
```

Or use your existing `gpt_oss_profile` workflow.

## 3. Apply Optimal Configuration

After identifying the best configuration from the parameter sweep:

1. Open `models/demos/gpt_oss/tt/experts_throughput/config.py`

2. Update the `ThroughputProgramConfig` class with optimal values:

```python
@dataclass
class ThroughputProgramConfig:
    # Update with values from analysis
    gate_up_cores: tuple[int, int] = (8, 8)  # From sweep results
    down_cores: tuple[int, int] = (8, 7)     # From sweep results
    in0_block_w: int = 15                    # From sweep results
    out_subblock_h: int = 1                  # From sweep results
    out_subblock_w: int = 1                  # From sweep results
    per_core_M: int = 1                      # From sweep results
```

3. Verify the improvement with end-to-end tests:

```bash
pytest models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_experts_mlp.py -v
```

## 4. Understanding Results

### Stress Test Output

```
==========================================
STRESS TEST RESULTS
==========================================
Iterations: 100
Batch size: 32, Seq len: 1
Performance (us): avg=1234.56, min=1200.00, max=1300.00, std=25.00
PCC: avg=0.998500, min=0.998200
Status: PASSED
==========================================
```

Key metrics:
- **avg**: Average execution time - target for optimization
- **std**: Standard deviation - should be low for stable performance
- **PCC**: Accuracy metric - should stay > 0.998

### Parameter Sweep Output

```
==========================================
TOP 10 CONFIGURATIONS
==========================================
Rank   Time (us)    PCC        Core Grids           in0_blk  subblock   M
--------------------------------------------------------------------------------
1      1150.23      0.998600   (8,8),(8,7)         15       (1,1)      1
2      1175.45      0.998550   (8,8),(8,8)         10       (1,1)      1
3      1189.67      0.998700   (5,9),(5,9)         30       (1,1)      1
...
```

Look for:
- **Lowest time**: Best performance configuration
- **High PCC**: All configs should have PCC > 0.998
- **Consistent results**: Similar configs should have similar performance

## 5. Common Issues

### Memory Errors

**Problem**: Test fails with out of memory errors

**Solution**:
- Reduce batch size in test parametrization
- Check for memory leaks in your changes

### PCC Failures

**Problem**: Some configurations fail PCC validation

**Solution**:
- This is normal - some configs may not be valid
- Focus on passed configurations
- If all fail, check input data types

### No Configs Pass

**Problem**: Parameter sweep shows 0 passed configs

**Solution**:
- Check that device is properly initialized
- Verify model config is loaded correctly
- Check for device compatibility issues

## 6. Next Steps

After optimizing the experts matmul:

1. **Profile end-to-end**: Run full model tests to verify improvement propagates
2. **Test at scale**: Run on full batch sizes and sequence lengths
3. **Document findings**: Note the optimal config and improvement in performance
4. **CI integration**: Consider adding stress test to CI pipeline

## 7. Advanced Usage

### Custom Parameter Ranges

Edit `generate_program_configs()` in `test_experts_matmul.py` to test more configs:

```python
def generate_program_configs():
    # Add more core grid options
    core_grids = [
        (5, 9), (8, 8), (8, 7), (7, 8),
        (6, 8), (4, 9), # Add more options
    ]

    # Test more block widths
    in0_block_w_options = [1, 2, 3, 5, 6, 9, 10, 15, 18, 30]

    # ... rest of function
```

### Different Batch/Seq Configurations

Modify test parametrization in `test_experts_matmul.py`:

```python
@pytest.mark.parametrize("batch_size", [32, 64, 128], ids=["batch32", "batch64", "batch128"])
@pytest.mark.parametrize("seq_len", [1, 128, 256], ids=["seq1", "seq128", "seq256"])
def test_experts_matmul_param_sweep(...):
```

### Automated Optimization

For automated optimization across multiple configurations:

```bash
# Create a sweep script
for batch in 32 64 128; do
    for seq in 1 128; do
        pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_param_sweep \
            -k "batch${batch}" -k "seq${seq}" -v

        # Analyze results
        python models/demos/gpt_oss/tests/ops/analyze_param_sweep.py \
            --csv "results_batch${batch}_seq${seq}.csv"
    done
done
```

## 8. File Overview

```
tests/ops/
├── test_experts_matmul.py           # Main test file
├── README_experts_matmul.md         # Detailed documentation
├── QUICKSTART_experts_matmul.md     # This file
├── run_experts_matmul_tests.sh      # Helper script to run tests
└── analyze_param_sweep.py           # Analysis script for results

param_sweep_results/                 # Created after first sweep
└── experts_matmul_sweep_*.json      # Result files
```

## Questions or Issues?

See the detailed `README_experts_matmul.md` for:
- In-depth explanations of test methodology
- Troubleshooting guide
- Integration with existing tests
- Advanced configuration options
