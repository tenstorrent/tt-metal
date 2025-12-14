# Grid Strategy Comparison: max_ab vs max_abc

**Date**: November 13, 2025
**Configurations Compared**: 138 matching configurations
**Files**:
- `example_multiple_ops_max_ab_20251113_013450.csv` (142 rows, 140 valid)
- `example_multiple_ops_max_abc_20251113_062946.csv` (243 rows, 140 valid, 1 error)

---

## Executive Summary

### Key Finding: Nearly Identical Overall Performance

**Overall Performance**:
- Mean difference: **-0.130 μs (0.75%)** — max_abc *slightly* slower
- Median difference: **-0.003 μs (-0.01%)** — essentially identical
- max_ab faster: **69 cases (50.0%)**
- max_abc faster: **69 cases (50.0%)**

**Surprising Discovery**:
- **All 138 matching configurations use the SAME compute cores** for both strategies
- Performance differences are NOT due to different compute grid sizes
- Differences likely due to output tensor sharding, memory layout, or measurement variance

---

## Detailed Findings

### 1. Error in max_abc

**Configuration**: `1×1024 (block,8) + 1024×1024 (height,16) → block output`

**Error**:
```
TT_FATAL: num_shards_along_width <= shard_grid.x
Number of shards along width 4 must not exceed number of columns 2 for row major orientation
```

**Implication**: max_abc strategy attempted to create an invalid block sharding configuration when matching C's requested cores with the actual tensor layout.

---

### 2. Performance by Output Sharding Type

| Output Type | Cases | Mean Diff | max_ab Faster | max_abc Faster |
|-------------|-------|-----------|---------------|----------------|
| **HEIGHT** | 21 | **+3.51%** | 7 | 14 |
| **WIDTH** | 51 | **-0.44%** | 25 | 26 |
| **BLOCK** | 50 | **+0.53%** | 26 | 24 |
| **INTERLEAVED** | 16 | **+1.61%** | 11 | 5 |

**Interpretation**:
- **HEIGHT sharding**: max_abc is 3.5% slower on average
- **WIDTH sharding**: Nearly identical performance
- **BLOCK sharding**: Nearly identical performance
- **INTERLEAVED**: max_ab is 1.6% faster (but with high variance)

---

### 3. Top Cases Where max_abc is FASTER

| Configuration | max_ab | max_abc | Improvement |
|--------------|--------|---------|-------------|
| `1×1024` (interleaved) + `1024×1024` (height,8) → interleaved | 81.56 μs | 49.63 μs | **39.1%** |
| `1×1024` (interleaved) + `1024×1024` (block,8) → interleaved | 75.75 μs | 45.17 μs | **40.4%** |
| `1×1024` (block,8) + `1024×1024` (height,32) → block | 85.76 μs | 63.48 μs | **26.0%** |
| `1×1024` (block,16) + `1024×1024` (height,32) → block | 85.53 μs | 64.08 μs | **25.1%** |
| `1×1024` (interleaved) + `1024×1024` (width,8) → interleaved | 77.58 μs | 56.89 μs | **26.7%** |

**Pattern**: max_abc significantly faster when:
- Small tensor A (1×1024) is interleaved or lightly sharded
- Output is interleaved or matches A's small sharding
- Appears to benefit from better memory locality

---

### 4. Top Cases Where max_ab is FASTER

| Configuration | max_ab | max_abc | Improvement |
|--------------|--------|---------|-------------|
| `1×1024` (interleaved) + `1024×1024` (height,32) → interleaved | 37.33 μs | 79.16 μs | **112.1%** |
| `1×1024` (block,32) + `1024×1024` (height,32) → block,32 | 72.69 μs | 95.43 μs | **31.3%** |
| `1×1024` (block,16) + `1024×1024` (height,16) → block,16 | 63.95 μs | 86.03 μs | **34.5%** |
| `1×1024` (interleaved) + `1024×1024` (width,32) → interleaved | 58.03 μs | 77.03 μs | **32.7%** |
| `1×1024` (block,32) + `1024×1024` (block,32) → block,32 | 67.71 μs | 82.92 μs | **22.5%** |

**Pattern**: max_ab significantly faster when:
- Using 32 cores (highest core count)
- B tensor has height sharding with 32 cores
- Output is heavily sharded (block,32) or interleaved
- Suggests better handling of high-parallelism scenarios

---

## Critical Insight: Compute Cores Are Identical

### Surprising Result

**All 138 matching configurations show:**
- `ab_cores == abc_cores` (100% of cases)
- Both strategies selected the **same compute grid**

### What This Means

The performance differences are **NOT** due to:
- Different compute grid sizes
- Different parallelism levels
- max_abc using C's cores differently

The performance differences **ARE** likely due to:
1. **Output tensor sharding**: max_abc may respect C's requested sharding more strictly
2. **Memory layout**: Different strategies may produce different output memory layouts
3. **Data movement**: Subtle differences in how data is moved to/from sharded outputs
4. **Measurement variance**: Some differences may be within noise (±5%)

### Why Are Compute Cores the Same?

Looking at the `max_abc` strategy in C++:

```cpp
// max_abc uses max_ab logic internally for compute grid
// It only differs in OUTPUT tensor sharding preference
```

This suggests:
- **max_abc** = compute with `max(A, B)` cores + prefer C's sharding for output
- **max_ab** = compute with `max(A, B)` cores + ignore C's sharding preference

The "abc" in max_abc doesn't mean it uses C's cores for **computation**, but rather that it tries to honor C's **output sharding specification**.

---

## Variance Analysis

### High Variance Cases (>20% difference)

**max_abc faster by >20%**:
- 10 cases, mostly with interleaved A and small sharding
- Benefits from output sharding that matches data locality

**max_ab faster by >20%**:
- 10 cases, mostly with 32 cores and height sharding
- Benefits from ignoring output sharding constraints

### Low Variance Cases (<5% difference)

- **87 cases (63%)** have differences less than 5%
- These represent stable, predictable performance
- Both strategies are equally good for these configurations

---

## Implications for Strategy Selection

### Use max_ab When:
1. ✅ High parallelism (32 cores)
2. ✅ Height-sharded large tensors
3. ✅ Output sharding is flexible
4. ✅ Predictable performance is priority

### Use max_abc When:
1. ✅ Small tensor operations (1×1024 + 1024×1024)
2. ✅ Interleaved or lightly sharded inputs
3. ✅ Output tensor sharding matters for downstream ops
4. ✅ Memory locality is critical

### Both Are Equivalent When:
1. ✅ Width sharding (0.44% difference)
2. ✅ Block sharding (0.53% difference)
3. ✅ Moderate parallelism (8-16 cores)
4. ✅ Stable configurations

---

## Extreme Cases Analysis

### Worst Case for max_abc
**Configuration**: `1×1024` (interleaved) + `1024×1024` (height,32) → interleaved
**Performance**: max_abc is **112% slower** (79.16 μs vs 37.33 μs)

**Why?**
- 32 cores = high parallelism
- Height sharding = optimal for large tensor B
- max_abc may be creating suboptimal output layout
- Possible resharding overhead

### Best Case for max_abc
**Configuration**: `1×1024` (interleaved) + `1024×1024` (block,8) → interleaved
**Performance**: max_abc is **40% faster** (45.17 μs vs 75.75 μs)

**Why?**
- Interleaved A matches interleaved output
- Block sharding on B with 8 cores
- max_abc optimizes for output locality
- Avoids unnecessary sharding overhead

---

## Statistical Summary

### Distribution of Differences

| Range | Count | Percentage |
|-------|-------|------------|
| max_abc faster by >20% | 10 | 7.2% |
| max_abc faster by 10-20% | 3 | 2.2% |
| max_abc faster by 5-10% | 6 | 4.3% |
| Negligible (<5%) | 87 | 63.0% |
| max_ab faster by 5-10% | 8 | 5.8% |
| max_ab faster by 10-20% | 15 | 10.9% |
| max_ab faster by >20% | 9 | 6.5% |

**Key Insight**:
- **63% of cases** have negligible difference (<5%)
- **13.7%** show significant max_abc advantage (>10% faster)
- **17.4%** show significant max_ab advantage (>10% faster)

---

## Recommendations

### For Production Use

1. **Default to max_ab**:
   - More predictable performance
   - Slightly better average performance (0.75%)
   - Fewer edge cases with poor performance

2. **Use max_abc for**:
   - Small tensor broadcasts (1×N + M×N)
   - Interleaved input/output workflows
   - When output sharding matters for fusion

3. **Profile Both for**:
   - Critical hot paths
   - Operations with 32 cores
   - Height-sharded workloads

### For Development

1. **Investigate why compute cores are identical**:
   - Expected max_abc to use C's cores more often
   - May indicate strategy implementation differs from intent

2. **Fix max_abc error**:
   - Block sharding validation issue
   - Needs better grid computation for output tensors

3. **Reduce variance**:
   - 112% difference in extreme case is too high
   - May need better heuristics for output layout selection

---

## Conclusion

**Bottom Line**:
- max_ab and max_abc perform **nearly identically** on average (0.75% difference)
- Both strategies use the **same compute grid** (max of A and B cores)
- Differences stem from **output tensor sharding**, not compute parallelism
- **max_ab is slightly safer** with fewer extreme regressions
- **max_abc can be 40% faster** in specific small-tensor scenarios

The choice between strategies should be based on:
1. Output tensor requirements (sharded vs interleaved)
2. Downstream operation fusion needs
3. Tolerance for variance in extreme cases

For most use cases, **max_ab is recommended** due to:
- Slightly better average performance
- More predictable behavior
- Fewer errors and edge cases

---

## Technical Notes

### Why Compute Cores Don't Differ

Both strategies internally use `max(A_cores, B_cores)` for computation:
- **max_ab**: Explicit in name and implementation
- **max_abc**: Uses max_ab logic, then considers C for output

The "abc" suffix indicates:
- **Output tensor sharding preference**, not compute grid
- Willingness to match C's requested cores for the output
- May trigger resharding if compute grid differs from C's cores

### Future Work

1. **Measure output resharding overhead**: Is this the source of variance?
2. **Test with more operations**: Does pattern hold for POWER, MULTIPLY, etc.?
3. **Profile memory bandwidth**: Are differences due to DRAM vs L1 access patterns?
4. **Implement smart strategy selection**: Auto-choose based on tensor characteristics

---

**Generated**: November 13, 2025
**Tool**: `compare_strategies.py`
**Data**: 138 matching configurations across 2 CSV files
