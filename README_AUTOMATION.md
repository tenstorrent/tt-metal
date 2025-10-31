# Activation Function Profiling Automation

Automated profiling system for TTNN activation functions using Tracy profiler.

---

## Files Overview

### 1. `test_all_activations.py` - Test Definitions

Pytest test file that defines activation functions and configurations for profiling.

**Example Configuration:**

```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Shape configurations
@pytest.mark.parametrize(
    "input_shapes, memory_config_type",
    [
        (torch.Size([32, 32]), "L1_interleaved"),           # Small tensors
        (torch.Size([512, 512]), "L1_block_sharded"),       # Block sharding
        (torch.Size([1, 16, 320, 320]), "height_sharded"),  # YOLOv4 shape
    ],
)
# Activation functions to test
@pytest.mark.parametrize(
    "activation_func, func_name, params, has_approx",
    [
        # Simple functions (no parameters)
        (ttnn.relu, "relu", {}, False),
        (ttnn.mish, "mish", {}, False),
        (ttnn.silu, "silu", {}, False),

        # Functions with parameters
        (ttnn.leaky_relu, "leaky_relu", {"alpha": 0.01}, False),
        (ttnn.elu, "elu", {"alpha": 1.0}, False),

        # Approximate mode support
        (ttnn.sigmoid, "sigmoid_accurate", {}, False),  # Accurate
        (ttnn.sigmoid, "sigmoid_approx", {}, True),     # Approximate
    ],
)
def test_activation_functions(input_shapes, memory_config_type, activation_func, func_name, params, has_approx, device):
    """Test activation functions with different shapes and memory configs"""
    # Test implementation...
```

**Parameters:**
- `activation_func`: TTNN function reference (e.g., `ttnn.mish`)
- `func_name`: String identifier (e.g., `"mish"`)
- `params`: Dictionary of function parameters (e.g., `{"alpha": 0.01}`)
- `has_approx`: Boolean flag for approximate mode testing

---

### 2. `automate_activation_profiling.py` - Profiling Automation

Profiles **all combinations** of shapes × activation functions.

**What it does:**
1. Extracts both shape and activation entries
2. Generates unique profile names: `{func_name}_{shape}_{memory_config}`
3. Profiles each combination systematically
4. Restores original file after completion

**Example:**
- 3 shapes × 5 activations = 15 profile runs
- Profile names: `mish_32_32_L1_INTERLEAVED`, `mish_512_512_L1_BLOCK_SHARDED`, etc.

**Usage:**
```bash
python automate_activation_profiling.py
```

**Output:**
```
Found 3 shape configuration entries
Found 5 activation function entries

Total combinations to profile: 15
  Shapes: 3
  Activations: 5

Proceed with profiling 15 combinations? (y/N): y

Processing 1/15
  Shape: [32, 32] Config: L1_interleaved
  Activation: relu
  Profile Name: relu_32_32_L1_INTERLEAVED
✅ SUCCESS

Processing 2/15
  Shape: [32, 32] Config: L1_interleaved
  Activation: mish
  Profile Name: mish_32_32_L1_INTERLEAVED
✅ SUCCESS

...

PROFILING COMPLETE
Successful: 14/15
```

---

### 3. `collect_activation_results.py` - Results Collection and Analysis

Collects Device Kernel Duration (DKD) from profiling results and generates performance report.

**What it does:**
1. Scans `generated/profiler/reports/` directory
2. Finds latest CSV file for each profile
3. Extracts DKD (Device Kernel Duration) in nanoseconds
4. Ranks functions by performance
5. Generates consolidated CSV report

**Usage:**
```bash
# Default pattern
python collect_activation_results.py

# Custom pattern (match specific shape/config)
python collect_activation_results.py 32_32_L1_INTERLEAVED
python collect_activation_results.py 1_16_320_320_HEIGHT_SHARDED
```

**Output:**
```
Collecting DKD results for all 32_32_L1_INTERLEAVED activation functions...

Processing: relu_32_32_L1_INTERLEAVED
  ✅ DKD: 1234 ns | Op: relu | Approx: N | Fidelity: HiFi4

Processing: mish_32_32_L1_INTERLEAVED
  ✅ DKD: 1789 ns | Op: mish | Approx: N | Fidelity: HiFi4

...

================================================================================
ACTIVATION FUNCTION PERFORMANCE SUMMARY
================================================================================
Total Functions Analyzed: 15

PERFORMANCE RANKING (by DKD):
Rank | Func_Name      | DKD (ns) | Op_Type | Math_Fidelity
----------------------------------------------------------
   1 | relu           |     1234 | relu    | HiFi4
   2 | silu           |     1567 | silu    | HiFi4
   3 | mish           |     1789 | mish    | HiFi4
   4 | hardmish       |     1856 | hardmish| HiFi4
...

📊 Detailed results saved to: activation_dkd_results_32_32_L1_INTERLEAVED.csv
```

---

## Complete Workflow Example

### Goal: Profile Mish, Hardmish, and SiLU across multiple shapes

**Step 1: Configure `test_all_activations.py`**

```python
@pytest.mark.parametrize(
    "input_shapes, memory_config_type",
    [
        (torch.Size([32, 32]), "L1_interleaved"),
        (torch.Size([512, 512]), "L1_block_sharded"),
        (torch.Size([1, 16, 320, 320]), "height_sharded"),
    ],
)
@pytest.mark.parametrize(
    "activation_func, func_name, params, has_approx",
    [
        (ttnn.mish, "mish", {}, False),
        (ttnn.hardmish, "hardmish", {}, False),
        (ttnn.silu, "silu", {}, False),
    ],
)
def test_activation_functions(...):
    # Test implementation
```

**Step 2: Run Advanced Profiling**

```bash
cd /home/ubuntu/code/tt-metal
source python_env/bin/activate
python automate_activation_profiling.py
```

This profiles **9 combinations** (3 shapes × 3 functions):
- `mish_32_32_L1_INTERLEAVED`
- `mish_512_512_L1_BLOCK_SHARDED`
- `mish_1_16_320_320_HEIGHT_SHARDED`
- `hardmish_32_32_L1_INTERLEAVED`
- ... and so on

**Step 3: Collect Results for Each Shape**

```bash
# For 32x32 shape
python collect_activation_results.py 32_32_L1_INTERLEAVED

# For 512x512 shape
python collect_activation_results.py 512_512_L1_BLOCK_SHARDED

# For YOLOv4 shape
python collect_activation_results.py 1_16_320_320_HEIGHT_SHARDED
```

**Step 4: Analyze CSV Reports**

Each command generates a CSV file:
- `activation_dkd_results_32_32_L1_INTERLEAVED.csv`
- `activation_dkd_results_512_512_L1_BLOCK_SHARDED.csv`
- `activation_dkd_results_1_16_320_320_HEIGHT_SHARDED.csv`

**Example CSV Output:**
```csv
Rank,Func_Name,DKD_ns,Actual_Op_Type,Math_Fidelity,Timestamp,Directory_Name
1,silu,1567,silu,HiFi4,2024_01_15_10_30_45,silu_32_32_L1_INTERLEAVED
2,mish,1789,mish,HiFi4,2024_01_15_10_31_12,mish_32_32_L1_INTERLEAVED
3,hardmish,1856,hardmish,HiFi4,2024_01_15_10_32_05,hardmish_32_32_L1_INTERLEAVED
```

**Step 5: Compare Performance**

Compare the same function across shapes:
- Mish @ 32×32: 1789 ns
- Mish @ 512×512: 2345 ns
- Mish @ YOLOv4: 3456 ns

---

## Quick Reference

| Script                              | Use Case                                  | Command                                    |
|-------------------------------------|-------------------------------------------|--------------------------------------------|
| `automate_activation_profiling.py`  | Profile all shape × function combinations | `python automate_activation_profiling.py`  |
| `collect_activation_results.py`     | Collect and analyze results               | `python collect_activation_results.py`     |

---

**Environment Setup:**
```bash
cd /home/ubuntu/code/tt-metal
source python_env/bin/activate
rm -rf ~/.cache/tt-metal-cache  # Clear cache before profiling
```
