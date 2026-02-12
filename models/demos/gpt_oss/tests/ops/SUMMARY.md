# Experts Matmul Unit Tests - Summary

## What Was Created

A comprehensive unit test suite for the GPT-OSS experts sparse matmul operations, designed to run on single devices and optimize performance through systematic testing.

### Files Created

1. **`test_experts_matmul.py`** (804 lines)
   - Main test implementation
   - Stress test (stability and memory management)
   - Parameter sweep test (performance optimization)
   - Reference implementations for validation

2. **`README_experts_matmul.md`**
   - Comprehensive documentation
   - Detailed test descriptions
   - Result analysis guide
   - Integration documentation

3. **`QUICKSTART_experts_matmul.md`**
   - Fast-start guide
   - Common commands
   - Quick troubleshooting
   - Example outputs

4. **`run_experts_matmul_tests.sh`**
   - Convenient test runner script
   - Colored output
   - Multiple test modes (stress, sweep, both)

5. **`analyze_param_sweep.py`**
   - Automated result analysis
   - Performance statistics
   - Parameter impact analysis
   - CSV export capability

## Test Capabilities

### 1. Stress Test

**Purpose**: Verify operation stability and memory management

**What it does**:
- Runs matmul operation 100 times consecutively
- Validates correctness every 10 iterations
- Measures performance consistency (avg, min, max, std)
- Checks for memory leaks

**When to use**:
- After making changes to matmul implementation
- To verify stability before production
- To establish performance baselines

**Command**:
```bash
./models/demos/gpt_oss/tests/ops/run_experts_matmul_tests.sh stress
```

### 2. Parameter Sweep Test

**Purpose**: Find optimal program configuration parameters

**What it does**:
- Systematically tests different configurations
- Tests combinations of:
  - Core grid sizes (gate_up_cores, down_cores)
  - Block widths (in0_block_w)
  - Subblock dimensions
  - Per-core parameters
- Measures average performance over 20 iterations
- Validates correctness (PCC > 0.998)
- Saves results to JSON

**When to use**:
- To optimize performance for new hardware
- When model dimensions change
- To validate configuration choices
- Before major releases

**Command**:
```bash
./models/demos/gpt_oss/tests/ops/run_experts_matmul_tests.sh sweep
```

## Key Features

### Single Device Testing

- ✓ No inter-device communication required
- ✓ Fast iteration cycles
- ✓ Easier debugging
- ✓ Suitable for development and optimization

### Comprehensive Validation

- ✓ Reference implementation for correctness
- ✓ PCC validation (> 0.998)
- ✓ Performance metrics
- ✓ Memory management checks

### Detailed Results

- ✓ JSON output for programmatic analysis
- ✓ Summary statistics
- ✓ Top N configurations
- ✓ Parameter impact analysis
- ✓ CSV export

### Easy to Use

- ✓ Helper scripts for common operations
- ✓ Clear documentation
- ✓ Quick start guide
- ✓ Example outputs

## Parameters Under Test

The parameter sweep tests these configuration parameters:

### Core Grid Sizes
- **gate_up_cores**: Core grid for gate/up projections
  - Default: (5, 9)
  - Options tested: (5,9), (8,8), (8,7), (7,8), (4,8), (5,8)

- **down_cores**: Core grid for down projection
  - Default: (5, 9)
  - Options tested: Same as gate_up_cores

### Block Sizes
- **in0_block_w**: Input block width for K dimension
  - Default: 10
  - Options tested: 1, 2, 5, 10, 15, 30
  - Should be chosen based on hidden_size / 32

### Subblock Dimensions
- **out_subblock_h**: Output subblock height
  - Default: 1
  - Options tested: 1, 2

- **out_subblock_w**: Output subblock width
  - Default: 1
  - Options tested: 1, 2

### Per-Core Parameters
- **per_core_M**: Per-core M dimension
  - Default: 1
  - Options tested: 1, 2

## Expected Results

### Stress Test

Typical output:
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

**Interpretation**:
- Low std deviation = stable performance ✓
- High PCC = accurate results ✓
- No crashes = proper memory management ✓

### Parameter Sweep

Typical output from analysis:
```
Total configurations tested: 216
Passed: 180 (83.3%)
Failed: 36 (16.7%)

Performance Statistics (passed configs):
  Best (min):     1050.23 us
  Worst (max):    2345.67 us
  Mean:           1456.78 us

TOP 10 CONFIGURATIONS
Rank   Time (us)    PCC        Core Grids           in0_blk
1      1050.23      0.998600   (8,8),(8,7)         15
2      1075.45      0.998550   (8,8),(8,8)         10
...

OPTIMAL CONFIGURATION
gate_up_cores:   (8, 8)
down_cores:      (8, 7)
in0_block_w:     15
...
Performance:     1050.23 us (±12.34 us)
```

**Interpretation**:
- 83% pass rate is good (some configs may be invalid)
- 2.2x performance range shows parameter impact
- Optimal config provides clear target for implementation

## Integration with Existing Tests

These unit tests complement existing infrastructure:

### Existing Tests
- `test_gpt_oss_experts_mlp.py`: Full MLP with CCL ops, multi-device
- `test_gpt_oss_experts.py`: Full experts forward pass
- `fused_op_device_perf_utils.py`: Device performance utilities

### New Unit Tests
- **Focus**: Isolated matmul operations
- **Scope**: Single device, no CCL overhead
- **Purpose**: Optimization and debugging

**Use unit tests when**:
- Optimizing matmul performance
- Debugging matmul-specific issues
- Rapid iteration on configurations
- Development on single device

**Use existing tests when**:
- Validating end-to-end behavior
- Testing multi-device scenarios
- Measuring CCL performance
- Production validation

## Performance Optimization Workflow

1. **Baseline**: Run stress test to establish current performance
   ```bash
   ./run_experts_matmul_tests.sh stress
   ```

2. **Sweep**: Run parameter sweep to find optimal config
   ```bash
   ./run_experts_matmul_tests.sh sweep
   ```

3. **Analyze**: Review results to identify best configuration
   ```bash
   python analyze_param_sweep.py
   ```

4. **Apply**: Update ThroughputProgramConfig with optimal values
   ```python
   # In models/demos/gpt_oss/tt/experts_throughput/config.py
   @dataclass
   class ThroughputProgramConfig:
       gate_up_cores: tuple[int, int] = (8, 8)  # Updated
       down_cores: tuple[int, int] = (8, 7)     # Updated
       in0_block_w: int = 15                    # Updated
       # ...
   ```

5. **Verify**: Run end-to-end tests to confirm improvement
   ```bash
   pytest models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_experts_mlp.py -v
   ```

6. **Document**: Record optimal configuration and performance gain

## Future Enhancements

Potential improvements to the test suite:

### Additional Tests
- [ ] Prefill mode testing (seq_len > 1)
- [ ] Different sparsity patterns
- [ ] Bias variations
- [ ] Different activation functions

### Analysis Tools
- [ ] Visualization of parameter relationships
- [ ] Automatic configuration recommendation
- [ ] Performance regression detection
- [ ] Multi-dimensional optimization

### Integration
- [ ] CI/CD integration for nightly runs
- [ ] Performance tracking over time
- [ ] Automatic optimal config updates
- [ ] Cross-device comparison

## Contact and Support

For questions or issues with these tests:

1. Check `README_experts_matmul.md` for detailed documentation
2. Review `QUICKSTART_experts_matmul.md` for common issues
3. Examine test output and error messages
4. Check that device is properly initialized

## Quick Reference

### Running Tests
```bash
# Stress test
./run_experts_matmul_tests.sh stress

# Parameter sweep
./run_experts_matmul_tests.sh sweep

# Both
./run_experts_matmul_tests.sh both
```

### Analyzing Results
```bash
# Latest results
python analyze_param_sweep.py

# Specific file
python analyze_param_sweep.py param_sweep_results/experts_matmul_sweep_batch32_seq1.json

# Export to CSV
python analyze_param_sweep.py --csv results.csv
```

### Key Files
- Tests: `models/demos/gpt_oss/tests/ops/test_experts_matmul.py`
- Config: `models/demos/gpt_oss/tt/experts_throughput/config.py`
- Results: `param_sweep_results/*.json`

### Important Constants
- Stress test iterations: 100
- Stress test warmup: 10
- Param sweep iterations: 20
- Param sweep warmup: 5
- Expected PCC: > 0.998
