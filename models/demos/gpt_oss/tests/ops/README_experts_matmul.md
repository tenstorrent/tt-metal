# Experts Matmul Unit Tests

This directory contains focused unit tests for the sparse matmul operations used in the GPT-OSS experts layer.

## Overview

The experts matmul unit tests provide two main testing capabilities:

1. **Stress Test**: Runs the matmul operation many times in a row to verify stability, memory management, and consistent performance
2. **Parameter Sweep**: Systematically tests different program configuration parameters to find optimal settings for best device performance

These tests are designed to run on a **single device** since the sparse matmul operations do not require inter-device communication (CCL ops are only in the dispatch/combine wrappers).

## Test Files

- `test_experts_matmul.py` - Main test file with stress test and parameter sweep implementations

## Running the Tests

### Prerequisites

- Single TT device (N150, N300, or T3000 with single chip)
- Standard test environment with pytest

### Stress Test

The stress test runs the experts matmul operation many times to verify:
- Memory is properly managed (no leaks or corruption)
- Results remain consistent across iterations
- Performance is stable

```bash
# Run stress test with default parameters (100 iterations)
pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_stress -v

# Run with custom iteration count (parametrize in code)
pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_stress -v
```

**Output**: The test will log:
- Progress every 10 iterations
- Average, min, max, and standard deviation of execution time
- Average and minimum PCC (Pearson Correlation Coefficient) for validation
- Overall pass/fail status

### Parameter Sweep Test

The parameter sweep test systematically evaluates different program configuration parameters:

**Parameters tested:**
- `gate_up_cores`: Core grid size for gate/up projections (e.g., (5,9), (8,8))
- `down_cores`: Core grid size for down projection
- `in0_block_w`: Block width for input dimension (K dimension blocking)
- `out_subblock_h`, `out_subblock_w`: Output subblock dimensions
- `per_core_M`: Per-core M dimension parameter

```bash
# Run parameter sweep
pytest models/demos/gpt_oss/tests/ops/test_experts_matmul.py::test_experts_matmul_param_sweep -v
```

**Output**: The test will:
1. Test each configuration combination
2. Measure average execution time over multiple iterations
3. Validate correctness (PCC) for each configuration
4. Save detailed results to JSON file in `param_sweep_results/`
5. Display summary with:
   - Total configs tested
   - Pass/fail counts
   - Best configuration (lowest execution time)
   - Top 5 performing configurations

**Results file**: `param_sweep_results/experts_matmul_sweep_batch{batch}_seq{seq}.json`

The JSON contains detailed metrics for each configuration:
```json
{
  "gate_up_cores": [5, 9],
  "down_cores": [5, 9],
  "in0_block_w": 10,
  "out_subblock_h": 1,
  "out_subblock_w": 1,
  "per_core_M": 1,
  "batch_size": 32,
  "seq_len": 1,
  "avg_time_us": 1234.56,
  "min_time_us": 1200.00,
  "max_time_us": 1300.00,
  "std_time_us": 25.00,
  "passed": true,
  "pcc": 0.9985
}
```

## Analyzing Results

### Analyzing Parameter Sweep Results

After running the parameter sweep, you can analyze the results using Python:

```python
import json
import pandas as pd

# Load results
with open('param_sweep_results/experts_matmul_sweep_batch32_seq1.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)

# Filter to only passed configs
df_passed = df[df['passed'] == True]

# Sort by performance
df_sorted = df_passed.sort_values('avg_time_us')

# Display top 10 configurations
print(df_sorted.head(10))

# Analyze impact of specific parameters
# For example, effect of in0_block_w:
grouped = df_passed.groupby('in0_block_w')['avg_time_us'].agg(['mean', 'min', 'max'])
print(grouped)

# Find optimal core grid
grouped_cores = df_passed.groupby(['gate_up_cores', 'down_cores'])['avg_time_us'].mean()
print(grouped_cores.sort_values())
```

### Key Metrics to Consider

1. **avg_time_us**: Average execution time - primary optimization target
2. **std_time_us**: Standard deviation - look for consistent performance (low std)
3. **pcc**: Accuracy metric - should be > 0.998 for all configs
4. **min_time_us / max_time_us**: Check for outliers or variance

### Applying Optimal Configuration

Once you've identified the optimal configuration from the parameter sweep:

1. Update `ThroughputProgramConfig` in `models/demos/gpt_oss/tt/experts_throughput/config.py`:

```python
@dataclass
class ThroughputProgramConfig:
    # Update with optimal values from sweep
    gate_up_cores: tuple[int, int] = (8, 8)  # Example optimal value
    down_cores: tuple[int, int] = (8, 7)     # Example optimal value
    in0_block_w: int = 15                    # Example optimal value
    out_subblock_h: int = 1
    out_subblock_w: int = 1
    per_core_M: int = 1
```

2. Re-run the full model tests to verify end-to-end improvement

## Test Parameters

### Batch Size and Sequence Length

Default test parameters:
- **Stress Test**: batch_size=32, seq_len=1 (decode mode)
- **Parameter Sweep**: batch_size=32, seq_len=1 (decode mode)

To test different configurations, modify the `@pytest.mark.parametrize` decorators in the test file:

```python
@pytest.mark.parametrize("batch_size", [32, 64], ids=["batch32", "batch64"])
@pytest.mark.parametrize("seq_len", [1, 128], ids=["seq1", "seq128"])
```

### Iteration Counts

Adjust at the top of the test file:
- `STRESS_TEST_ITERS = 100` - Number of stress test iterations
- `STRESS_TEST_WARMUP = 10` - Warmup iterations before stress test
- `PARAM_SWEEP_WARMUP = 5` - Warmup iterations per config
- `PARAM_SWEEP_ITERS = 20` - Measurement iterations per config

### Parameter Sweep Range

To test more or fewer configurations, modify `generate_program_configs()`:

```python
def generate_program_configs():
    # Expand or reduce these lists
    core_grids = [(5, 9), (8, 8), (8, 7), ...]
    in0_block_w_options = [1, 2, 5, 10, 15, 30, ...]
    # ...
```

## Troubleshooting

### Test Failures

**Memory Errors**:
- Reduce batch size or number of iterations
- Check for memory leaks in custom code

**PCC Failures**:
- Verify input data types (should be bfloat16)
- Check that reference implementation matches TTNN implementation
- Some configs may not be valid for the given problem dimensions

**Configuration Errors**:
- Core grid must fit within device compute grid (typically 8x8)
- Block sizes must be factors of the problem dimensions
- Check that in0_block_w divides hidden_size/32 evenly

### Performance Issues

If parameter sweep takes too long:
1. Reduce the number of configurations in `generate_program_configs()`
2. Reduce `PARAM_SWEEP_ITERS` (but keep at least 10 for statistical validity)
3. Run sweep on subset of parameters first

## Integration with Existing Tests

These unit tests complement the existing fused op tests in:
- `tests/fused_op_unit_tests/test_gpt_oss_experts_mlp.py` - Full MLP test with multi-device support
- `tests/fused_op_unit_tests/test_gpt_oss_experts.py` - Full experts forward with CCL ops

The unit tests focus on:
- **Single-device testing** (no CCL overhead)
- **Isolated op testing** (just the matmuls)
- **Parameter optimization** (finding best configs)

Use the unit tests for:
- ✓ Rapid iteration on matmul performance
- ✓ Finding optimal program configurations
- ✓ Debugging matmul-specific issues
- ✓ Stress testing memory management

Use the fused op tests for:
- ✓ End-to-end validation with CCL ops
- ✓ Multi-device behavior
- ✓ Full expert pipeline testing
