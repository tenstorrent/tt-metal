# Shared Exponent Precision Testing Suite

## Overview

This testing suite provides comprehensive analysis of numerical precision for TT-Metal operations using bfloat8_b (shared exponent) data type compared to bfloat16 reference implementations. The primary focus is investigating how different data patterns and distributions affect precision when using shared exponent representations, particularly for column-based shared exponents.

## Purpose

The suite addresses the critical question: **How do different data patterns and distributions impact numerical precision when using shared exponent formats?**

Key objectives:
- Compare bfloat8_b (shared exponent) precision against bfloat16 reference
- Identify data patterns that cause significant precision degradation
- Analyze the impact of outliers, magnitude differences, and spatial patterns
- Generate comprehensive reports for precision analysis
- Guide optimization strategies for shared exponent implementations

## Architecture

The testing framework consists of several modules:

```
shared_exponent_precision/
├── main.py              # Entry point and orchestration
├── constants.py         # Configuration constants and keys
├── generators.py        # Data pattern and distribution generators
├── runner.py           # Core testing execution engine
├── postprocessing.py   # Results analysis and report generation
└── Readme.md           # This documentation
```

## Data Generation Strategy

### Distribution Types (`generators.py`)

The suite generates various statistical distributions to stress-test precision behavior:

#### Basic Distributions
- **`constant`**: All ones tensor - tests uniform magnitude behavior and shared exponent with identical values
- **`normal_0_1`**: Standard normal N(0,1) - baseline distribution
- **`normal_skewed_mean`**: Normal with shifted mean N(5,1) - tests mean offset effects
- **`normal_high_var_10`**: High variance N(0,10²) - moderate spread testing
- **`normal_high_var_100`**: Very high variance N(0,100²) - extreme spread testing

#### Outlier Distributions
- **`normal_with_outliers`**: N(0,1) with sparse large outliers (0.1% probability, 100x scale)
- **`fa_rand_default`**: Mixed Gaussian with 0.1% outliers from N(0,10²)
- **`fa_rand_aggressive`**: Mixed Gaussian with 1% outliers from N(0,100²)

#### Combined Distributions
- **`skewed_high_var_10`**: N(10,10²) - shifted mean + high variance
- **`skewed_high_var_100`**: N(10,100²) - shifted mean + very high variance

#### Negative Variants
All distributions above are also generated with negated values (`*_negative`) to test sign handling.

### Pattern Types (`generators.py`)

Spatial patterns that create challenging scenarios for shared exponent precision:

#### **Column Gradient Pattern** (`column_magnitude_gradient`)
Creates systematic magnitude differences across columns (10⁻³ to 10³ range).
```python
# Column 0: 10^-3, Column 1: 10^-2.5, ..., Column n: 10^3
```
**Purpose**: Tests column-wise shared exponent behavior with predictable magnitude progression.

#### **Reverse Column Gradient Pattern** (`reverse_column_magnitude_gradient`)
Creates systematic magnitude differences across columns in reverse order (10³ to 10⁻³ range).
```python
# Column 0: 10^3, Column 1: 10^2.5, ..., Column n: 10^-3
```
**Purpose**: Tests column-wise shared exponent behavior with reverse magnitude progression to identify directional sensitivity.

#### **Row Gradient Pattern** (`row_magnitude_gradient`)
Creates systematic magnitude differences across rows (10⁻³ to 10³ range).
```python
# Row 0: 10^-3, Row 1: 10^-2.5, ..., Row n: 10^3
```
**Purpose**: Tests row-wise shared exponent behavior with predictable magnitude progression along the vertical axis.

#### **Reverse Row Gradient Pattern** (`reverse_row_magnitude_gradient`)
Creates systematic magnitude differences across rows in reverse order (10³ to 10⁻³ range).
```python
# Row 0: 10^3, Row 1: 10^2.5, ..., Row n: 10^-3
```
**Purpose**: Tests row-wise shared exponent behavior with reverse magnitude progression to identify vertical directional sensitivity.

#### **Row Uniform Pattern** (`row_uniform_column_varying`)
Uniform values within rows, varying magnitudes across rows in cycles.
```python
# Row patterns cycle through: 10^-2, 10^-1, 10^0, 10^1
```
**Purpose**: Tests row-wise operations with controlled magnitude variations.

#### **Checkerboard Pattern** (`checkerboard_magnitudes`)
Alternating high/low magnitude values in checkerboard arrangement.
```python
if (i + j) % 2 == 0:
    value *= 100  # High magnitude tiles
```
**Purpose**: Tests spatial locality effects and tile boundary precision.

#### **Row Outliers Pattern** (`row_outliers`)
Specific rows (10% by default) contain 1000x magnitude outliers.
```python
# Selected rows scaled by 1000x
```
**Purpose**: Tests impact of concentrated outliers on shared exponent calculation.

#### **Tile Boundaries Pattern** (`tile_boundaries`)
Different magnitudes assigned to each 32x32 tile region.
```python
# Each tile gets magnitude: 10^((tile_i + tile_j) % 4 - 2)
```
**Purpose**: Tests tile-level shared exponent behavior and boundary effects.

#### **Diagonal Gradient Pattern** (`diagonal_gradient`)
Magnitude increases along diagonal direction (10⁻³ to 10³).
```python
dist = (i + j) / (shape[0] + shape[1])
magnitude = 10^(dist * 6 - 3)
```
**Purpose**: Tests gradual spatial magnitude transitions.

#### **Block Pattern** (`block_pattern`)
8x8 blocks with distinct magnitude levels (1e-3, 1e-1, 1, 1e2, 1e4).
```python
# Cycling through 5 different magnitude levels per block
```
**Purpose**: Tests medium-scale spatial magnitude clustering.

#### **FA_Rand Pattern** (`fa_rand_pattern`)
Mixed Gaussian distribution as spatial pattern (0.5% outliers, 100x scale).
```python
fa_rand_custom(shape, base_std=1.0, outlier_std=100.0, outlier_prob=0.005)
```
**Purpose**: Tests random outlier distributions as spatial patterns.

## Operations Tested

The suite tests the following TT-Metal operations:

### Reduction Operations
- **`sum`**: Element summation along specified axis
- **`mean`**: Average calculation along specified axis
- **`max`**: Maximum value extraction along specified axis

### Matrix Operations
- **`matmul`**: Standard matrix multiplication
- **`matmul_tt`**: TT-Metal optimized matrix multiplication with tile configurations
  - Tile widths: 16, 32
  - Transpose options: False, True

### Activation Functions
- **`softmax`**: Softmax activation along specified axis

### Testing Axes
- **Axis 0**: Column-wise operations (reducing rows)
- **Axis 1**: Row-wise operations (reducing columns)

## Test Matrix

The complete test matrix is:
```
Shape Types × Patterns × Distributions × Operations × Axes/Configs
```

### Shape Categories

#### **Single Tile** (32×32)
Tests precision behavior within single tile boundaries.

#### **Multi-tile** (512×512)
Tests precision behavior across multiple tiles (16×16 tiles of 32×32 each).

#### **Rectangular Shapes**
Tests non-square tensor behavior:
- 32×128: Tall rectangles
- 128×32: Wide rectangles
- 64×256: Mixed aspect ratios

## Precision Metrics

For each test case, the following metrics are computed:

### Correlation Metrics
- **PCC (Pearson Correlation Coefficient)**: Measures linear correlation between reference and test results
- **Allclose checks**: Binary pass/fail at 1e-2 and 1e-3 tolerance levels

### Error Metrics
- **Max/Mean Absolute Error**: Direct difference measurements
- **Max/Mean Relative Error**: Percentage error measurements

### ULP (Units in Last Place) Analysis
- **ULP Mean/Max**: Floating-point precision measurements
- **ULP Percentiles**: 50th, 90th, 99th percentile ULP errors

### Input Statistics
- **Min/Max/Mean/Std**: Statistical properties of input data
- **Range**: Dynamic range of input values

## Usage Instructions

### Running the Complete Suite

```bash
cd tests/ttnn/unit_tests/operations/shared_exponent_precision_testing_suite
python3 main.py
```

### Output Files

The suite generates comprehensive analysis in the `bfloat8_experiment_results/` directory:

#### **`raw_results.json`**
Complete numerical results in JSON format for programmatic analysis.

#### **`raw_results.md`**
Comprehensive markdown report with all test results organized hierarchically:
- Shape Type → Pattern → Distribution → Operation → Metrics

#### **`worst_cases_analysis.md`**
Focused analysis of the worst-performing test cases:
- Top 10 worst cases per metric
- Detailed case information (shape, pattern, distribution, operation)
- Contextual metrics for each worst case

#### **`pattern_impact_analysis.md`**
Statistical analysis of how different patterns affect precision:
- Pattern performance rankings by different metrics
- Average, min, max, and standard deviation statistics per pattern
- Operations breakdown showing which operations are tested with each pattern

### Custom Configuration

#### Running Specific Patterns
Modify the `pattern_generators` selection in `_run_shape_experiments()`:

```python
# Test only specific patterns
selected_patterns = ["column_gradient", "reverse_column_magnitude_gradient",
                     "row_gradient", "fa_rand_pattern"]
pattern_generators = {k: v for k, v in generate_test_patterns(shape).items()
                     if k in selected_patterns}
```

#### Adding Custom Distributions
Extend `generate_distributions()` in `generators.py`:

```python
def generate_distributions(shape) -> dict:
    distributions = {}
    # ... existing distributions ...

    # Add custom distribution
    distributions["my_custom_dist"] = torch.custom_function(shape)
    return distributions
```

#### Adding Custom Patterns
Extend `generate_test_patterns()` in `generators.py`:

```python
def my_custom_pattern(shape):
    # Your pattern logic here
    return torch.custom_tensor(shape)

# Add to pattern dictionary
patterns["my_custom"] = lambda: my_custom_pattern(shape)
```

## Analysis Workflow

### 1. Identify Problematic Patterns
Review `pattern_impact_analysis.md` to find patterns causing precision issues:
- Check PCC rankings (lower is worse)
- Review absolute error rankings (higher is worse)
- Examine ULP error rankings (higher is worse)

### 2. Investigate Worst Cases
Review `worst_cases_analysis.md` to understand specific failure modes:
- Analyze the combination of pattern + distribution + operation causing issues
- Look at input statistics to understand data characteristics
- Check related metrics for context

### 3. Deep Dive Analysis
Use `raw_results.json` for programmatic analysis:
- Filter results by specific criteria
- Compute custom statistics
- Generate custom visualizations

### 4. Pattern-Specific Investigation
For patterns of interest, examine:
- How different distributions affect that pattern
- Which operations are most sensitive to that pattern
- Whether the issue is consistent across different shapes
