# Strategy Comparison Tool

## Overview

`compare_strategies_improved.py` is a tool to compare binary_ng grid selection strategies across different operation types and broadcast types.

## Features

- **Automatic CSV Discovery**: Finds CSV files matching the pattern `{OP_TYPE}_{BROADCAST_TYPE}_{STRATEGY}_{TIMESTAMP}.csv`
- **Multi-dimensional Comparison**: Groups results by operation type and broadcast type
- **Detailed Statistics**: Shows performance differences, winner distribution, and top cases
- **Flexible Filtering**: Filter by op_type, broadcast_type, or analyze everything
- **Visualization Generation**:
  - Scatter plots showing strategy performance comparisons
  - Color-coded winners (red/cyan) for easy identification
  - Saved to organized comparison directories
- **Multiple Output Formats**:
  - Text summary (`comparison_summary.txt`)
  - JSON statistics (`statistics.json`)
  - CSV data files for each comparison
  - PNG scatter plots for visual analysis

## Usage

The tool supports two comparison modes:

1. **Strategy Mode**: Compare strategies across multiple CSV files (finds latest files automatically)
2. **Direct File Mode**: Compare specific CSV files by their exact filenames

### Strategy Mode - Basic Comparison

Compare two strategies across all available data:

```bash
python compare_strategies_improved.py max_ab min_ab
```

This finds the latest CSV files for each strategy and compares them.

### Filter by Operation Type

Compare only ADD operations:

```bash
python compare_strategies_improved.py max_ab min_ab --op-type ADD
```

Multiple op types:

```bash
python compare_strategies_improved.py max_ab full_grid --op-type ADD POWER LOGADDEXP
```

### Filter by Broadcast Type

Compare only no_broadcast scenarios:

```bash
python compare_strategies_improved.py max_ab min_ab --broadcast-type no_broadcast
```

Multiple broadcast types:

```bash
python compare_strategies_improved.py a_first b_first --broadcast-type no_broadcast row_broadcast
```

### Combined Filters

```bash
python compare_strategies_improved.py max_ab min_ab --op-type ADD --broadcast-type no_broadcast
```

## Direct File Mode

Compare specific CSV files by their exact filenames. This is useful when you want to compare:
- Specific benchmark runs with exact timestamps
- Different versions of the same strategy
- Results from different test sessions

### Compare by Full Filename (with extension)

```bash
python compare_strategies_improved.py ADD_no_broadcast_max_ab_20251115_235255.csv ADD_no_broadcast_half_grid_20251116_002025.csv
```

### Compare by Filename (without extension)

The `.csv` extension is optional:

```bash
python compare_strategies_improved.py ADD_no_broadcast_max_ab_20251115_235255 ADD_no_broadcast_half_grid_20251116_002025
```

### Compare Two Runs of Same Strategy

Useful for checking performance consistency or regression testing:

```bash
python compare_strategies_improved.py ADD_no_broadcast_max_ab_20251115_235255 ADD_no_broadcast_max_ab_20251115_225919
```

### How It Works

The tool automatically detects direct file mode when:
- Input contains a timestamp pattern (`YYYYMMDD_HHMMSS`)
- Input ends with `.csv`
- Input contains 3+ underscores (filename structure pattern)

The strategies are extracted from the filenames for labeling in outputs.

### Custom Results Directory

```bash
python compare_strategies_improved.py max_ab min_ab --results-dir /path/to/results
```

### Custom Output Directory

By default, outputs are saved to `comparisons/comparison_{strategy1}_vs_{strategy2}_{timestamp}/`:

```bash
# Specify custom output location
python compare_strategies_improved.py max_ab min_ab --output-dir /path/to/output
```

### Skip Visualizations (Faster)

For quick text-only comparisons without generating plots:

```bash
python compare_strategies_improved.py max_ab min_ab --no-viz
```

## Output Directory Structure

When visualizations are enabled (default), the tool creates this directory structure:

```
comparisons/comparison_{strategy1}_vs_{strategy2}_{timestamp}/
├── comparison_summary.txt              # Human-readable summary
├── statistics.json                      # Machine-readable stats
├── comparison_data_{OP}_{BROADCAST}.csv # Merged comparison data per op/broadcast
└── {strategy1}_vs_{strategy2}/         # Visualization directory
    └── scatter_{strategy1}_vs_{strategy2}.png  # Scatter plot
```

### Example Output

```
comparisons/comparison_max_ab_vs_min_ab_20251114_232725/
├── comparison_summary.txt
├── statistics.json
├── comparison_data_ADD_no_broadcast.csv
└── max_ab_vs_min_ab/
    └── scatter_max_ab_vs_min_ab.png
```

## Output Sections

### 1. Found CSV Files

Lists all matching CSV files grouped by (op_type, broadcast_type):

```
ADD / no_broadcast:
  max_ab      : ADD_no_broadcast_max_ab_20251114_174838.csv
  min_ab      : ADD_no_broadcast_min_ab_20251114_225616.csv
```

### 2. Per-Configuration Comparison

For each (op_type, broadcast_type) pair:

- **Overall Statistics**: Mean/median time difference, winner count
- **Compute Cores Analysis**: How often strategies use different core counts
- **Output Sharding Breakdown**: Performance by c_sharding type
- **Top Winners**: Best cases for each strategy

### 3. Overall Summary

(When comparing multiple op_types/broadcast_types)

- **By Operation Type**: Aggregate statistics per op_type
- **By Broadcast Type**: Aggregate statistics per broadcast_type
- **Overall Winner**: Total winner across all configurations

## Example Output Interpretation

```
Mean time difference: 5.127 μs (14.40%)
max_ab faster: 33 cases (33.0%)
min_ab faster: 67 cases (67.0%)
```

This means:
- On average, min_ab is 14.40% SLOWER than max_ab (+5.127 μs)
- But min_ab wins in 67% of cases (likely by smaller margins)
- max_ab wins in 33% of cases (likely by larger margins)

## Supported Strategies

- `max_ab` - Maximum of A and B core counts
- `min_ab` - Minimum of A and B core counts
- `max_abc` - Max of A, B, and C core counts
- `a_first` - Prefer A's grid
- `b_first` - Prefer B's grid
- `full_grid` - Always use full device grid
- `current` - Default strategy (prefer C, then A, then B)

## File Naming Convention

CSV files must follow this pattern:

```
{OP_TYPE}_{BROADCAST_TYPE}_{STRATEGY}_{YYYYMMDD}_{HHMMSS}.csv
```

Examples:
- `ADD_no_broadcast_max_ab_20251114_174838.csv`
- `POWER_row_broadcast_min_ab_20251114_180000.csv`
- `LOGADDEXP_col_broadcast_full_grid_20251114_190000.csv`

## Key Insights from max_ab vs min_ab

From the ADD no_broadcast comparison:

1. **max_ab is generally faster** (67% win rate for min_ab but 14.4% slower on average)
   - This suggests max_ab wins by larger margins when it does win
   - min_ab wins more often but with smaller improvements

2. **Core count matters significantly** (54% of cases have different core counts)
   - When cores differ, min_ab is 26.34% slower on average
   - Fewer cores = less parallel memory bandwidth for data-bound operations

3. **Output sharding impacts results**:
   - Height sharding: min_ab +27.40% slower (max parallelism helps)
   - Width sharding: min_ab +2.97% slower (less sensitive)
   - Block sharding: min_ab +17.29% slower
   - Interleaved: min_ab +1.05% slower (both use full grid)

4. **When min_ab wins**:
   - Cases where one tensor uses 8 cores and the other uses 32 cores
   - max_ab's 32-core grid may be overkill, causing overhead
   - min_ab's 8-core grid matches the smaller tensor better

5. **When max_ab wins**:
   - Cases with mismatched sharding (height vs width)
   - More cores = better overall memory bandwidth
   - Outweighs any overhead from using more cores

## Recommendation

For **equal-sized tensors** (no_broadcast) with **data-bound operations**:
- **Use max_ab**: Maximizes memory bandwidth
- **Consider min_ab**: Only when one tensor is much smaller or resources are constrained
