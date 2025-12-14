# Multi-Strategy Comparison Guide

## Overview

This guide explains how to compare multiple grid selection strategies (max_ab, max_abc, full_grid, etc.) in an organized way without overwriting previous results.

---

## Quick Start

### Compare 2 Strategies
```bash
cd /workspace/tests/ttnn/benchmarks/binary_ng
python compare_multi_strategy.py max_ab max_abc
```

### Compare 3 Strategies
```bash
python compare_multi_strategy.py max_ab max_abc full_grid
```

### Compare with Custom Output Directory
```bash
python compare_multi_strategy.py max_ab max_abc -o my_comparison_name
```

### List All Comparisons
```bash
./list_comparisons.sh
```

Shows:
- All comparison directories (most recent first)
- Strategies compared
- Timestamp
- File count and size
- Winner (for 3-way comparisons)
- Quick commands to access results

---

## Organization System

### Directory Structure

All comparisons are stored in timestamped directories under `comparisons/`:

```
tests/ttnn/benchmarks/binary_ng/
├── comparisons/
│   ├── comparison_max_ab_vs_max_abc_20251113_181225/
│   │   ├── comparison_summary.txt           # Main text report
│   │   ├── merged_data.csv                  # All data merged
│   │   ├── statistics.json                  # Structured stats
│   │   ├── max_ab_vs_max_abc/              # Pairwise comparison
│   │   │   └── scatter_max_ab_vs_max_abc.png
│   │   ├── max_ab_vs_full_grid/            # Pairwise comparison
│   │   │   └── scatter_max_ab_vs_full_grid.png
│   │   ├── max_abc_vs_full_grid/           # Pairwise comparison
│   │   │   └── scatter_max_abc_vs_full_grid.png
│   │   ├── three_way_winner_distribution.png   # 3-way pie chart
│   │   ├── three_way_average_performance.png   # 3-way bar chart
│   │   ├── three_way_boxplot.png              # 3-way box plot
│   │   └── three_way_winner_by_sharding.png   # 3-way heatmap
│   │
│   └── comparison_max_ab_vs_max_abc_vs_current_20251113_150000/
│       └── ...  (another comparison, organized separately)
│
├── results/
│   ├── example_multiple_ops_max_ab_20251113_013450.csv
│   ├── example_multiple_ops_max_abc_20251113_062946.csv
│   └── example_multiple_ops_full_grid_20251113_180003.csv
│
└── visualizations/  # For 2-way detailed visualizations (legacy)
```

### Key Benefits

✅ **No Overwriting**: Each comparison gets a unique timestamped directory
✅ **Self-Contained**: All related files in one directory
✅ **Organized**: Clear naming and structure
✅ **Traceable**: Timestamp shows when comparison was run
✅ **Flexible**: Can compare any number of strategies

---

## Output Files Explained

### For All Comparisons

1. **`comparison_summary.txt`** - Main text report
   - Overall statistics
   - Average performance per strategy
   - Winner distribution
   - Pairwise comparisons
   - Best configs for each strategy
   - Performance by output sharding

2. **`merged_data.csv`** - Complete merged dataset
   - All 104 common configurations
   - Kernel times for all strategies
   - Compute cores for all strategies
   - Configuration details (shapes, shardings, cores)

3. **`statistics.json`** - Structured statistics
   - Machine-readable format
   - All numerical results
   - Useful for automated analysis

### For 2-Way Comparisons

Each pair gets a subdirectory with:
- **`scatter_X_vs_Y.png`** - Scatter plot comparing the two strategies

### For 3-Way Comparisons

Four additional charts in main directory:
1. **`three_way_winner_distribution.png`** - Pie chart showing % of wins
2. **`three_way_average_performance.png`** - Bar chart with averages
3. **`three_way_boxplot.png`** - Box plot showing distributions
4. **`three_way_winner_by_sharding.png`** - Heatmap: winner count by output sharding

---

## Real Example: 3-Way Comparison

### Command
```bash
python compare_multi_strategy.py max_ab max_abc full_grid
```

### Output
```
comparisons/comparison_max_ab_vs_max_abc_vs_full_grid_20251113_181225/
├── comparison_summary.txt                         # 3.1 KB
├── merged_data.csv                                # 15 KB, 104 configs
├── statistics.json                                # 2.0 KB
│
├── max_ab_vs_max_abc/
│   └── scatter_max_ab_vs_max_abc.png             # 307 KB
├── max_ab_vs_full_grid/
│   └── scatter_max_ab_vs_full_grid.png           # 314 KB
├── max_abc_vs_full_grid/
│   └── scatter_max_abc_vs_full_grid.png          # 316 KB
│
├── three_way_winner_distribution.png             # 145 KB (pie chart)
├── three_way_average_performance.png             # 121 KB (bar chart)
├── three_way_boxplot.png                         # 131 KB (box plot)
└── three_way_winner_by_sharding.png              # 128 KB (heatmap)
```

### Key Finding from Example
```
🏆 WINNER DISTRIBUTION
  full_grid:  92 configs (88.5%)  ⭐ CLEAR WINNER
  max_abc:     8 configs ( 7.7%)
  max_ab:      4 configs ( 3.8%)

📈 AVERAGE PERFORMANCE
  full_grid: 65.53μs (11.8% faster than max_ab)
  max_abc:   73.70μs
  max_ab:    74.31μs
```

**Conclusion**: `full_grid` is dramatically better for these configurations!

---

## Comparison Workflows

### Workflow 1: Quick 2-Way Check
```bash
# Compare two strategies quickly
python compare_multi_strategy.py max_ab max_abc

# Review summary
cat comparisons/comparison_max_ab_vs_max_abc_*/comparison_summary.txt | head -50
```

### Workflow 2: Comprehensive 3-Way Analysis
```bash
# Compare three strategies
python compare_multi_strategy.py max_ab max_abc full_grid

# Open comparison directory
cd comparisons/comparison_max_ab_vs_max_abc_vs_full_grid_*

# View all charts
ls -lh *.png */

# Read full report
less comparison_summary.txt

# Analyze data programmatically
python
>>> import pandas as pd
>>> df = pd.read_csv('merged_data.csv')
>>> df.describe()
```

### Workflow 3: Historical Tracking
```bash
# Run multiple comparisons over time
python compare_multi_strategy.py max_ab max_abc -o "v1_baseline"
# ... make code changes ...
python compare_multi_strategy.py max_ab max_abc -o "v2_after_optimization"

# Compare directories
diff -r comparisons/v1_baseline/ comparisons/v2_after_optimization/
```

### Workflow 4: Automated Analysis
```bash
# Generate statistics for multiple strategy sets
for strategies in "max_ab max_abc" "max_ab full_grid" "max_abc full_grid"; do
    python compare_multi_strategy.py $strategies
done

# Aggregate results
cat comparisons/*/comparison_summary.txt | grep "AVERAGE PERFORMANCE" -A 5
```

---

## Advanced Usage

### Custom Output Name
```bash
# Use descriptive name instead of timestamp
python compare_multi_strategy.py max_ab max_abc full_grid \
    -o "comparison_before_kernel_optimization"
```

### Specify Results Directory
```bash
# If CSV files are in a different location
python compare_multi_strategy.py max_ab max_abc \
    --results-dir /path/to/other/results/
```

### Compare More Than 3 Strategies
```bash
# Works with 4, 5, or more strategies
python compare_multi_strategy.py max_ab max_abc full_grid a_first b_first

# Will generate:
# - All pairwise comparisons (N choose 2)
# - Multi-way statistics
# - Winner distribution across all N strategies
```

---

## Understanding the Results

### Pairwise Scatter Plots

Points below diagonal = Strategy 2 faster
Points above diagonal = Strategy 1 faster

Colors:
- 🔴 Red = Strategy 1 faster
- 🔵 Teal = Strategy 2 faster
- ⚪ Gray = Tie

### Winner Distribution (Pie Chart)

Shows which strategy wins most often across all configurations.

**Interpretation**:
- Large slice = strategy frequently wins
- Small slice = strategy rarely optimal
- Use to identify dominant strategy

### Average Performance (Bar Chart)

Compares average kernel times with error bars (std deviation).

**Interpretation**:
- Lower bars = better average performance
- Small error bars = consistent performance
- Large error bars = high variance

### Box Plot

Shows full distribution of kernel times.

**Elements**:
- Box = 25th to 75th percentile
- Line in box = median
- Whiskers = min/max (excluding outliers)
- Dots = outliers

### Winner by Sharding (Heatmap)

Shows which strategy wins for each output sharding type.

**Interpretation**:
- Darker cells = more wins for that strategy
- Helps identify strategy strengths by sharding pattern

---

## Tips & Best Practices

### 1. Always Review Summary First
```bash
# Quick overview before diving into charts
head -50 comparisons/latest_dir/comparison_summary.txt
```

### 2. Use Descriptive Output Names for Important Comparisons
```bash
# Instead of timestamp, use meaningful name
python compare_multi_strategy.py max_ab max_abc -o "pre_release_validation"
```

### 3. Keep Historical Comparisons
```bash
# Don't delete old comparison directories
# They're useful for tracking performance trends over time
ls -lt comparisons/  # View chronologically
```

### 4. Compare Apples to Apples
- Ensure all CSV files were generated with same test configurations
- Check that `merged_data.csv` has reasonable row count
- Low merge count may indicate different test sets

### 5. Watch for Outliers
```bash
# Check for extreme values or errors
grep "inf\|nan\|ERROR" comparisons/*/comparison_summary.txt
```

### 6. Automate Regular Comparisons
```bash
#!/bin/bash
# weekly_comparison.sh
DATE=$(date +%Y%m%d)
python compare_multi_strategy.py max_ab max_abc full_grid \
    -o "weekly_report_${DATE}"
```

---

## Interpreting Statistics

### Mean vs Median
- **Mean**: Average, influenced by outliers
- **Median**: Middle value, robust to outliers
- If mean >> median: positive outliers (some very slow configs)
- If mean << median: negative outliers (some very fast configs)

### Standard Deviation
- **Low std**: Consistent performance across configs
- **High std**: Performance varies widely
- Important for reliability assessment

### Winner Distribution
- **Dominant winner** (>80%): Clear best choice
- **Close split** (40-60%): Context-dependent, no clear winner
- **Balanced** (33-33-33%): Each strategy has strengths

### Percentage Differences
- **< 5%**: Negligible practical difference
- **5-15%**: Noticeable but modest improvement
- **15-30%**: Significant performance gain
- **> 30%**: Dramatic difference, investigate why

---

## Troubleshooting

### Problem: No CSV file found for strategy
```
❌ Error: No CSV file found for strategy 'xyz' (pattern: *_xyz_*.csv)
```

**Solution**:
- Check strategy name spelling
- Ensure CSV file exists in results/ directory
- Verify filename follows pattern: `*_strategy_*.csv`

### Problem: Very few merged rows
```
Merged: 5 common configurations
```

**Solution**:
- Different test configurations were run for each strategy
- Rerun benchmarks with identical test configs
- Check for errors that filtered out rows

### Problem: "inf" in statistics
```
Mean diff: +inf%
```

**Solution**:
- One strategy has 0μs for some config (likely error)
- Check for outliers or data quality issues
- Review merged_data.csv for anomalies

### Problem: Charts look wrong
**Solution**:
- Check matplotlib/seaborn versions
- Update with: `pip install matplotlib seaborn --upgrade`
- Regenerate visualizations

---

## Migration from Old System

### Old: Single Visualization Directory
```
visualizations/
├── scatter_kernel_time_comparison.png   # OVERWRITES each time
├── bar_percentage_differences.png       # OVERWRITES each time
└── ...
```

### New: Organized Comparisons Directory
```
comparisons/
├── comparison_max_ab_vs_max_abc_20251113_120000/   # Kept forever
├── comparison_max_ab_vs_max_abc_20251113_150000/   # New run
└── comparison_max_ab_vs_max_abc_vs_full_grid_*/    # 3-way
```

**Migration Steps**:
1. Keep old `visualizations/` for 2-way detailed analysis
2. Use `comparisons/` for all new comparisons
3. Both systems work independently
4. Old 2-way script: `visualize_comparison.py` (still works)
5. New multi-way script: `compare_multi_strategy.py` (recommended)

---

## FAQ

**Q: Can I compare more than 3 strategies?**
A: Yes! The script supports any number of strategies. It will generate all pairwise comparisons and multi-way statistics.

**Q: What if I want to rerun a comparison with the same name?**
A: Use `-o custom_name`. If the directory exists, files will be overwritten (use carefully).

**Q: How do I compare strategies from different dates?**
A: Specify full paths to CSV files (feature to be added) or copy them to results/ with distinguishing names.

**Q: Can I customize the charts?**
A: Yes! Edit `compare_multi_strategy.py`. All matplotlib/seaborn code is clearly marked.

**Q: How much disk space do comparisons use?**
A: ~1-2 MB per comparison (mostly PNG images). Comparisons are cheap to store.

**Q: Should I delete old comparisons?**
A: Generally no. They're small and useful for historical tracking. Archive if needed.

---

## Summary

✅ **Use `compare_multi_strategy.py` for all new comparisons**
✅ **Each comparison gets its own timestamped directory**
✅ **No overwriting - all results are preserved**
✅ **Supports 2, 3, or N-way comparisons**
✅ **Organized structure with clear naming**
✅ **Comprehensive visualizations and statistics**

**Quick Command**:
```bash
python compare_multi_strategy.py max_ab max_abc full_grid
```

**Result**: Complete comparison in `comparisons/comparison_*_<timestamp>/`

---

**Last Updated**: November 13, 2025
**Status**: ✅ Production ready, supports 2+ strategies
