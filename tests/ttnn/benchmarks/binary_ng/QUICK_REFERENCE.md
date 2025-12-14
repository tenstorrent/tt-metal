# Quick Reference - Binary NG Benchmark Testing

## 🚀 Quick Start

### Run the benchmark test:
```bash
cd /workspace
TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/benchmarks/binary_ng/example_single_test.py::test_multiple_operations_with_timing -v -s
```

### Check results:
```bash
ls -lah tests/ttnn/benchmarks/binary_ng/results/*.csv
```

### View latest CSV:
```bash
tail -20 tests/ttnn/benchmarks/binary_ng/results/example_multiple_ops_max_ab_*.csv
```

---

## 🔧 Main Fix: Block Sharding

### Problem
Block sharding needs 2D grids compatible with tensor shape.

### Solution
```python
# New function: compute_valid_block_grid(shape, cores)
# Example: (1, 1024) with 8 cores → (2, 4) grid
```

### Location
`tests/ttnn/benchmarks/binary_ng/example_single_test.py:142-184`

---

## 🔀 Multi-Strategy Comparison (NEW) ⭐

### Compare 2+ Strategies
```bash
cd /workspace/tests/ttnn/benchmarks/binary_ng

# 2-way comparison
python compare_multi_strategy.py max_ab max_abc

# 3-way comparison
python compare_multi_strategy.py max_ab max_abc full_grid

# Custom name (no overwriting!)
python compare_multi_strategy.py max_ab max_abc -o my_comparison
```

### Output (Organized!)
All comparisons go to `comparisons/<timestamped_dir>/`:
- `comparison_summary.txt` - Main report
- `merged_data.csv` - All data
- `statistics.json` - Structured stats
- Pairwise scatter plots in subdirectories
- 3-way charts (pie, bar, box, heatmap)

**No overwriting!** Each run creates new timestamped directory.

### Key Insight from 3-Way
```
Winner: full_grid (88.5% of configs, 11.8% faster avg)
  max_ab:    74.31μs avg
  max_abc:   73.70μs avg
  full_grid: 65.53μs avg ⭐
```

### Full Guide
See `MULTI_STRATEGY_COMPARISON_GUIDE.md` for complete documentation.

---

## 📈 2-Way Detailed Visualizations

### Generate Charts (max_ab vs max_abc only)
```bash
cd /workspace/tests/ttnn/benchmarks/binary_ng
python visualize_comparison.py
```

### Output
Creates 5 high-res PNG charts in `visualizations/`:
1. **Scatter plot** - Direct kernel time comparison
2. **Bar chart** - Percentage differences (all 104 configs)
3. **Histogram** - Distribution of time differences
4. **Heatmap** - Avg performance by sharding config
5. **Compute cores analysis** - Performance vs. cores

Plus: `visualization_summary.txt` with statistics

**Note**: Use `compare_multi_strategy.py` for organized multi-way comparisons.

---

## 📊 Key Files

### Modified
1. `example_single_test.py` - Main benchmark test (block sharding fix, compute cores tracking)
2. `test_binary_bcast.py` - Unit test (sharding info printing)

### Created
1. `BLOCK_SHARDING_FIX.md` - Detailed explanation of block sharding fix
2. `SESSION_EXPORT.md` - Full session documentation
3. `QUICK_REFERENCE.md` - This file

### Results
- `results/example_multiple_ops_max_ab_20251113_011040.csv` - Latest clean results (142 configs, no errors)

---

## 🎯 Key Changes

### 1. Block Sharding Grid Computation
**Before**: Hardcoded grids → many errors
**After**: Shape-aware computation → all valid

```python
# Filter invalid configs early
if a_sharding == "block" and a_cores is not None:
    if compute_valid_block_grid(shape_a, a_cores) is None:
        continue  # Skip invalid combination
```

### 2. Compute Cores Tracking
**Before**: From output tensor (0 when interleaved)
**After**: From C++ WORKER_GRID logs (always correct)

```python
# Parse: "WORKER_GRID: strategy=max_ab cores=32"
worker_grid_match = re.search(r'WORKER_GRID:\s*strategy=(\S+)\s+cores=(\d+)', stderr_output)
```

### 3. Tensor Sharding Info
```python
mem_config = tensor.memory_config()  # method call
print(f"Layout: {mem_config.memory_layout}")  # property
if mem_config.shard_spec is not None:
    print(f"Grid: {mem_config.shard_spec.grid}")
```

---

## 📈 Performance Insights

### max_ab vs max_abc
- **Overall**: Nearly identical (~0.5% difference)
- **Height sharding**: max_ab 3% faster
- **Width sharding**: No significant difference
- **Interleaved output**: No significant difference

### Counter-Intuitive Finding
**Fastest**: `a=interleaved, b=sharded, c=interleaved`

**Why?**
- Maximum parallelism (uses b's 32 cores)
- Efficient DRAM reads (distributed across cores)
- No resharding overhead

---

## 🔍 Debug Commands

### Enable C++ logging:
```bash
export TT_METAL_LOGGER_LEVEL=Debug
```

### Enable profiling:
```bash
export TT_METAL_DEVICE_PROFILER=1
```

### View profiler results:
```bash
cat /workspace/generated/profiler/reports/ops_perf_results.csv
```

### Test block grid computation:
```python
from example_single_test import compute_valid_block_grid
compute_valid_block_grid((1, 1024), 8)   # → (2, 4)
compute_valid_block_grid((1024, 1024), 32) # → (4, 8)
```

---

## 📋 Test Status

✅ All 142 test configurations passing
✅ Block sharding with shape-aware grids working
✅ Compute cores correctly tracked
✅ Clean CSV output with no errors

### Sample Results:

| Config | compute_cores | kernel_time_us |
|--------|---------------|----------------|
| 1024×1024 height(8) + 1×1024 width(8) | 8 | 71.4 |
| 1024×1024 block(16) + 1×1024 block(16) | 16 | 63.4 |
| 1×1024 interleaved + 1024×1024 block(32) | 32 | 32.8 |

---

## 🎓 Key Concepts

### Block Sharding Constraints
For shape `(H, W)` with `N` cores, grid `(GH, GW)` must satisfy:
1. `GH × GW = N`
2. `H_padded % GH = 0` (H padded to 32)
3. `W_padded % GW = 0` (W padded to 32)

### Grid Selection Strategies
- **max_abc**: Uses max_ab logic
- **max_ab**: max(A_cores, B_cores)
- **min_ab**: min(A_cores, B_cores)
- **new_grid**: (max(A.x, B.x), max(A.y, B.y)) - element-wise max ⭐ NEW
- **current**: Prefers C → A → B → full grid
- **a_first**: Prefers A → B → C → full grid
- **b_first**: Prefers B → A → C → full grid
- **full_grid**: Always full device grid
- **half_grid**: Half device grid (32 cores)

---

## 🔄 Next Steps

1. Test other operations (POWER, LOGEXP)
2. Test more shape combinations
3. Analyze performance patterns in CSV
4. Document optimal sharding strategies
5. Create automated performance regression tests

---

## 📞 For More Details

- Full documentation: `SESSION_EXPORT.md`
- Block sharding details: `BLOCK_SHARDING_FIX.md`
- Code: `example_single_test.py`
- Results: `results/*.csv`

---

**Last Updated**: November 13, 2025
**Status**: ✅ All tests passing, ready for continuation
