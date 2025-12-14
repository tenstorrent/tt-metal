# Visualization Features - compare_strategies_improved.py

## Overview

The improved comparison tool now generates comprehensive visualizations and structured output directories, similar to the original `compare_multi_strategy.py`.

## What's New

### 🎨 Visualizations

1. **Scatter Plots**
   - Shows performance comparison between two strategies
   - Color-coded: Red (strategy1 faster), Cyan (strategy2 faster), Gray (tie)
   - Diagonal line shows equal performance
   - Annotations show winner counts
   - High-resolution PNG (300 DPI)

### 📁 Organized Output

Automatically creates structured directories:

```
comparisons/comparison_{strategy1}_vs_{strategy2}_{timestamp}/
├── comparison_summary.txt              # Text summary
├── statistics.json                      # JSON statistics
├── comparison_data_{OP}_{BROADCAST}.csv # Data per comparison
└── {strategy1}_vs_{strategy2}/
    └── scatter_{strategy1}_vs_{strategy2}.png
```

### 📊 Multiple Output Formats

1. **Text Summary** (`comparison_summary.txt`)
   - Overall statistics
   - Breakdown by operation and broadcast type
   - Win rates and performance deltas

2. **JSON Statistics** (`statistics.json`)
   - Machine-readable format
   - Detailed metrics per comparison
   - Overall aggregated statistics

3. **CSV Data** (`comparison_data_*.csv`)
   - Raw comparison data for each op_type/broadcast_type
   - All configuration details and timing data
   - Suitable for further analysis

4. **Scatter Plots** (PNG images)
   - Visual representation of performance
   - Easy identification of winners
   - Professional quality for presentations

## Usage

### Generate All Outputs (Default)

```bash
python compare_strategies_improved.py max_ab min_ab --op-type ADD --broadcast-type no_broadcast
```

**Output:**
```
📊 Scatter plot saved: max_ab_vs_min_ab/scatter_max_ab_vs_min_ab.png
💾 Data saved: comparison_data_ADD_no_broadcast.csv
📄 Summary: .../comparison_summary.txt
📊 Statistics: .../statistics.json
✅ All outputs saved to: .../comparison_max_ab_vs_min_ab_20251114_232725
```

### Text-Only Mode (Faster)

```bash
python compare_strategies_improved.py max_ab min_ab --no-viz
```

Skips visualization generation for quick analysis.

### Custom Output Location

```bash
python compare_strategies_improved.py max_ab min_ab --output-dir ./my_comparison
```

## Example Output

### Terminal Output

```
================================================================================
COMPARISON: max_ab vs min_ab
Operation: ADD, Broadcast: no_broadcast
================================================================================
Total configurations compared: 100

Mean time difference: 5.127 μs (14.40%)
Median time difference: 1.038 μs (1.80%)

max_ab faster: 33 cases (33.0%)
min_ab faster: 67 cases (67.0%)

Performance by output sharding:
  height      :  30 cases, avg +27.40%, max_ab: 12 min_ab: 18
  width       :  30 cases, avg  +2.97%, max_ab:  9 min_ab: 21
  block       :  30 cases, avg +17.29%, max_ab:  9 min_ab: 21
  interleaved :  10 cases, avg  +1.05%, max_ab:  3 min_ab:  7

  📊 Scatter plot saved: max_ab_vs_min_ab/scatter_max_ab_vs_min_ab.png
  💾 Data saved: comparison_data_ADD_no_broadcast.csv
```

### Generated Files

#### comparison_summary.txt
```
================================================================================
STRATEGY COMPARISON: max_ab vs min_ab
================================================================================
Generated: 2025-11-14 23:27:26
Output directory: .../comparison_max_ab_vs_min_ab_20251114_232725
================================================================================

OVERALL SUMMARY
--------------------------------------------------------------------------------
Total configurations: 100
max_ab faster: 33 cases (33.0%)
min_ab faster: 67 cases (67.0%)

BREAKDOWN BY OPERATION AND BROADCAST TYPE
--------------------------------------------------------------------------------

ADD / no_broadcast:
  Configurations: 100
  max_ab faster: 33 (33.0%)
  min_ab faster: 67 (67.0%)
  Mean time difference: 5.127 μs (+14.40%)
  Median time difference: 1.038 μs (+1.80%)
```

#### statistics.json
```json
{
  "strategies": ["max_ab", "min_ab"],
  "generated": "2025-11-14T23:27:26.665216",
  "comparisons": {
    "ADD_no_broadcast": {
      "total_configs": 100,
      "faster1": 33,
      "faster2": 67,
      "mean_diff_us": 5.12726,
      "mean_diff_pct": 14.400267740180885
    }
  },
  "overall": {
    "total_configs": 100,
    "max_ab_faster": 33,
    "min_ab_faster": 67,
    "max_ab_win_rate": 33.0,
    "min_ab_win_rate": 67.0
  }
}
```

## Comparison with Original Tool

| Feature | compare_strategies.py | compare_strategies_improved.py |
|---------|----------------------|--------------------------------|
| CSV Discovery | Manual paths | Automatic pattern matching |
| Op Types | Single | Multiple with filtering |
| Broadcast Types | Not supported | Full support with filtering |
| Scatter Plots | ❌ | ✅ |
| Text Summary | Limited | Comprehensive |
| JSON Stats | ❌ | ✅ |
| Directory Structure | ❌ | ✅ Organized |
| Filtering | ❌ | ✅ By op_type/broadcast |

## Tips

1. **For Quick Analysis**: Use `--no-viz` to skip plot generation
2. **For Presentations**: Use default mode to generate high-quality scatter plots
3. **For Further Analysis**: Use generated CSV files for custom visualization
4. **For Automation**: Parse `statistics.json` for programmatic access

## Dependencies

- pandas
- matplotlib
- seaborn
- Python 3.8+

All dependencies are already available in the TT-Metal environment.
