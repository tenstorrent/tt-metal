# Performance Regression Testing

This directory contains simple tooling for performance regression detection that compares current micro-benchmark results against baseline data to identify performance regressions.

## Components

### Core Files
- `perf_regression.py` - Regression detection tooling
- `gt/` - Ground truth baseline data directory

### Example Implementation
- `test_accessor_perf_regression.py` - Example implementation for accessor benchmarks
- `gt/accessor_benchmarks/` - Ground truth data for accessor benchmarks

## API

### PerformanceData Class

The `PerformanceData` class is the core data container that supports:
- **JSON serialization/deserialization**
- **Statistical analysis methods**
- **Data validation**

```python
from perf_regression import PerformanceData

# Create from dictionary
data = {
    "section1": {
        "subsection1": [100, 101, 99, 102, 98],
        "subsection2": [200, 201, 199, 202, 198]
    },
    "section2": {
        "subsection1": [50, 51, 49, 52, 48]
    }
}
perf_data = PerformanceData.from_dict(data)

# Load from JSON file
perf_data = PerformanceData.from_json_file("path/to/baseline.json")

# Save to JSON file
perf_data.to_json_file("path/to/output.json")

# Access data
samples = perf_data.get_samples("section1", "subsection1")
mean_val = perf_data.mean("section1", "subsection1")
```

### Regression Detection

The main regression detection function:

```python
from perf_regression import check_regression, PerformanceData

# Load baseline and current data
baseline = PerformanceData.from_json_file("gt/my_benchmark.json")
current = PerformanceData.from_dict(my_benchmark_results)

# Check for regressions
results = check_regression(baseline, current, alpha=0.05, min_effect_size_pct=1.0)

# Results structure:
# {
#     "section": {
#         "subsection": {
#             "is_regression": bool,
#             "delta_mean_pct": float,
#             "p_value": float,
#             "message": str,
#             "current_mean": float,
#             "baseline_mean": float
#         }
#     }
# }
```

## Ground Truth Data Format

Ground truth files must be JSON files with the following structure:

```json
{
  "section1": {
    "subsection1": [100, 101, 99, 102, 98, ...],
    "subsection2": [200, 201, 199, 202, 198, ...]
  },
  "section2": {
    "subsection1": [50, 51, 49, 52, 48, ...],
    "subsection2": [75, 76, 74, 77, 73, ...]
  }
}
```

Where:
- **sections** represent major categories (e.g., "rank_2", "rank_3" for different tensor ranks)
- **subsections** represent specific configurations (e.g., Specific TensorAccessor configuration)
- **samples** are arrays of numeric performance measurements

## Regression Detection Logic

The tooling uses a simplified regression detection approach combining statistical and practical significance:

1. **Statistical Significance**: Uses Mann-Whitney U test (p < alpha, default 0.05)
2. **Practical Significance**: Requires minimum effect size (default 1.0% change)
3. **Decision Logic**:

```python
delta_mean_pct = ((current_mean / baseline_mean) - 1.0) * 100.0
p_value = mann_whitney_u(current_samples, baseline_samples)

is_statistically_significant = p_value < alpha
is_practically_significant = abs(delta_mean_pct) >= min_effect_size_pct

if is_statistically_significant and is_practically_significant and delta_mean_pct > 0:
    # Performance regression detected
elif is_statistically_significant and is_practically_significant and delta_mean_pct < 0:
    # Performance improvement detected
elif is_statistically_significant and not is_practically_significant:
    # Statistically significant but trivial change
else:
    # No significant change
```

This approach prevents false alarms from tiny but statistically detectable differences (like 0.001% changes) while still catching real performance regressions.

## Creating New Regression Tests

### 1. Prepare Your Benchmark Data

```python
def run_my_benchmark() -> PerformanceData:
    """Run your benchmark and return results as PerformanceData."""
    # Run your benchmark
    results = run_my_actual_benchmark()

    # Convert to expected format
    formatted_results = {
        "config1": {
            "variant1": [100, 101, 99, 102, 98],  # Your sample data
            "variant2": [200, 201, 199, 202, 198]
        },
        "config2": {
            "variant1": [150, 151, 149, 152, 148]
        }
    }

    return PerformanceData.from_dict(formatted_results)
```

### 2. Create Ground Truth Data

Create a JSON file in `gt/your_benchmark/` with baseline performance data:

```json
{
  "config1": {
    "variant1": [100, 101, 99, 102, 98, ...],
    "variant2": [200, 201, 199, 202, 198, ...]
  },
  "config2": {
    "variant1": [150, 151, 149, 152, 148, ...]
  }
}
```

### 3. Create Pytest Test

```python
import pytest
from perf_regression import PerformanceData, check_regression, summarize_regression_results

def load_my_baseline() -> PerformanceData:
    """Load baseline data for your benchmark."""
    return PerformanceData.from_json_file("gt/your_benchmark/baseline.json")

def test_my_benchmark_regression():
    """Test for regressions in my benchmark."""
    baseline = load_my_baseline()
    current = run_my_benchmark()

    # Run regression check
    results = check_regression(baseline, current)

    # Summarize and check for regressions
    summary = summarize_regression_results(results)

    if summary['regressions'] > 0:
        regression_summary = "\\n".join(summary['regression_messages'])
        assert False, f"Performance regressions detected:\\n{regression_summary}"
```

### 4. Run Tests

```bash
# Run your specific test
python -m pytest test_my_benchmark.py -k test_my_benchmark_regression

# Run with verbose output
python -m pytest test_my_benchmark.py -k test_my_benchmark_regression -v
```

## Example: Accessor Benchmarks

See `test_accessor_perf_regression.py` for a complete example implementation that tests accessor performance across different tensor ranks and configurations.

```bash
# Run accessor regression tests
python -m pytest test_accessor_perf_regression.py -k test_constructor
python -m pytest test_accessor_perf_regression.py -k test_get_noc_addr_page_id
python -m pytest test_accessor_perf_regression.py -k test_baseline_data_integrity
```
