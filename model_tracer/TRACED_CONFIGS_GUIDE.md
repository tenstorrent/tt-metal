# Model Trace to Test Automation - Complete Guide

## 📋 Table of Contents
- [Quick Reference](#quick-reference)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| **Trace a model** | `python model_tracer/generic_ops_tracer.py <test_path>` |
| **View configurations** | `python model_tracer/analyze_operations.py <operation_name>` |
| **Generate sweep vectors** | `python3 tests/sweep_framework/sweeps_parameter_generator.py --module-name <op_name> --dump-file` |
| **Run sweep test** | `python3 tests/sweep_framework/sweeps_runner.py --module-name <op_name> --suite model_traced` (see [Sweep Framework README](tests/sweep_framework/README.md)) |

### Key Files

- **Tracer**: `model_tracer/generic_ops_tracer.py`
- **Master JSON**: `model_tracer/traced_operations/ttnn_operations_master.json`
- **Analyzer**: `model_tracer/analyze_operations.py`
- **Config Loader**: `tests/sweep_framework/master_config_loader.py`

### Integration Pattern (Copy & Paste)

**Just 3 simple changes to your sweep test:**

```python
# 1. Import the loader and helper function
from tests.sweep_framework.master_config_loader import MasterConfigLoader, unpack_traced_config

# 2. Load and add to parameters (before your parameters dict)
loader = MasterConfigLoader()
# Default: Run exact 30 traced configs
model_traced_params = loader.get_suite_parameters("your_operation_name")
# OR: Run all combinations (30 shapes × dtypes × layouts × memory_configs)
# model_traced_params = loader.get_suite_parameters("your_operation_name", all_cases=True)

parameters = {
    "nightly": {
        # ... your existing nightly tests ...
    },
    "model_traced": model_traced_params,  # Just add this line!
}

# 3. In your run() function, add traced_config_name parameter and ONE line to unpack
def run(
    input_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    traced_config_name=None,  # Add this parameter
    *,
    device,
):
    # Unpack all config values in ONE line
    if traced_config_name:
        input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config = unpack_traced_config(traced_config_name)

    # Rest of your test logic stays the same!
```

**That's it!** Just **1 line** to unpack all values. No module-level dictionaries, no manual parsing, no boilerplate!

**Note:** If you use `all_cases=True`, the `traced_config_name` parameter won't be used - the sweep framework will pass individual parameters (`input_shape`, `input_a_dtype`, etc.) directly via Cartesian product. The unpacking line will simply be skipped.

### Integration Pattern for Binary Operations (2 inputs)

**For binary operations like `add`, `multiply`, etc. that take 2 tensor inputs:**

```python
# 1. Import the loader and BINARY helper function
from tests.sweep_framework.master_config_loader import MasterConfigLoader, unpack_binary_traced_config

# 2. Load and add to parameters (same as unary)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("add")  # Automatically detects binary operation

parameters = {
    "nightly": { ... },
    "model_traced": model_traced_params,  # Same as unary!
}

# 3. In your run() function, use unpack_BINARY_traced_config
def run(
    input_shape=None,  # IMPORTANT: All parameters must have default values!
    input_a_dtype=None,
    input_b_dtype=None,
    input_a_layout=None,
    input_b_layout=None,
    input_a_memory_config=None,
    input_b_memory_config=None,
    traced_config_name=None,
    *,
    device,
):
    # Unpack BINARY config (7 values) in ONE line
    if traced_config_name:
        input_shape, input_a_dtype, input_b_dtype, input_a_layout, input_b_layout, \\
            input_a_memory_config, input_b_memory_config = unpack_binary_traced_config(traced_config_name)

    # Rest of test logic stays the same!
```

**Key Difference**: Use `unpack_binary_traced_config()` for binary ops, `unpack_traced_config()` for unary ops. The system automatically detects operation type!

---

## Overview

### What is This System?

The **Model Trace to Test Automation** system automatically extracts real-world operation configurations from model tests and integrates them into sweep tests. This ensures sweep tests validate operations using the exact same configurations that actual production models use.

### Key Benefits

✅ **Model-Driven Testing**: Test with real-world configurations, not synthetic ones
✅ **Automatic Extraction**: No manual configuration needed
✅ **Comprehensive Coverage**: Captures shapes, dtypes, memory configs (including sharded)
✅ **Deduplication**: Stores only unique configurations across all models
✅ **Easy Integration**: Drop-in compatibility with existing sweep tests

### Real Impact

- **Before**: Sweep tests used manually created configurations that might miss edge cases
- **After**: Sweep tests automatically use configurations from EfficientNet, ResNet, BERT, etc.
- **Result**: Instant validation with real-world model configurations, results shown in Superset dashboard

---

## Quick Start

### 1. Trace a Model Test

```bash
# Trace any model test to extract TTNN operations
python model_tracer/generic_ops_tracer.py /path/to/model/test.py::test_function
```

**Example:**
```bash
python model_tracer/generic_ops_tracer.py /home/ubuntu/tt-metal/models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py::test_efficientnetb0_model
```

**Output:**
- Creates/updates `model_tracer/traced_operations/ttnn_operations_master.json`
- Adds unique configurations from this model
- Shows summary: "Added X new unique configurations"

### 2. View Traced Configurations

```bash
# See all configurations for a specific operation
python model_tracer/analyze_operations.py sigmoid_accurate

# See configurations for any operation
python model_tracer/analyze_operations.py add
python model_tracer/analyze_operations.py matmul
```

### 3. Use in Sweep Tests

Configurations are **automatically loaded**:

```python
# In your sweep test file (e.g., sigmoid_accurate.py)
from tests.sweep_framework.master_config_loader import MasterConfigLoader

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("sigmoid_accurate")

parameters = {
    "model_traced": model_traced_params,  # That's it!
}
```

### 4. Run Sweep Tests

**Note:** See [Sweep Framework README](tests/sweep_framework/README.md) for complete sweep test documentation.

```bash
# Step 1: Generate test vectors
python3 tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --dump-file

# Step 2: Run the sweep test
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --vector-source vectors_export \
  --result-dest results_export \
  --suite model_traced

# Results appear in Superset dashboard
```

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Model Test (e.g., EfficientNet)          │
│                     Uses TTNN operations                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 1. Trace with generic_ops_tracer.py
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              Raw Trace (from TTNN graph capture)             │
│         Contains all operations with full parameters         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 2. Filter & Deduplicate
                      ↓
┌─────────────────────────────────────────────────────────────┐
│         ttnn_operations_master.json (Master Store)           │
│    Grouped by operation, deduplicated configurations         │
│                                                              │
│  {                                                           │
│    "operations": {                                           │
│      "sigmoid_accurate": {                                   │
│        "configurations": [                                   │
│          [{"arg0": {"Tensor": {...}}}],  ← Config 1         │
│          [{"arg0": {"Tensor": {...}}}],  ← Config 2         │
│          ...                                                 │
│        ]                                                     │
│      },                                                      │
│      "add": { ... },                                         │
│      "matmul": { ... }                                       │
│    }                                                         │
│  }                                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 3. Query with analyze_operations.py
                      │    or Load with master_config_loader.py
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                    Sweep Tests                               │
│     Automatically use traced configurations                  │
│                                                              │
│  • sigmoid_accurate: 30 configs from EfficientNet           │
│  • add: 6 configs from models                               │
│  • matmul: configs from attention layers                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **generic_ops_tracer.py** | Traces model tests, extracts operations | `model_tracer/` |
| **ttnn_operations_master.json** | Master storage of all traced configs | `model_tracer/traced_operations/` |
| **analyze_operations.py** | Query tool to view configurations | `model_tracer/` |
| **master_config_loader.py** | Parses master JSON, converts to TTNN objects, provides utilities | `tests/sweep_framework/` |
| **Sweep test files** | Use traced configs via `model_traced` suite | `tests/sweep_framework/sweeps/...` |

---

## Usage Guide

### Tracing Operations from Models

#### Basic Usage

```bash
# Trace a single test
python model_tracer/generic_ops_tracer.py <test_path>

# Examples:
python model_tracer/generic_ops_tracer.py models/demos/wormhole/resnet50/demo/demo.py::test_demo
python model_tracer/generic_ops_tracer.py models/demos/wormhole/distilbert/demo/demo.py::test_demo
```

#### What Gets Captured?

For each TTNN operation, the tracer captures:
- **Operation name** (e.g., `ttnn::sigmoid_accurate`, `ttnn::add`)
- **Tensor shapes** (e.g., `[1, 1, 12544, 32]`)
- **Data types** (e.g., `BFLOAT8_B`, `BFLOAT16`)
- **Memory layout** (e.g., `HEIGHT_SHARDED`, `INTERLEAVED`)
- **Shard specifications** (grid, shard_shape, orientation)
- **All operation arguments**

#### Filtering

The tracer automatically filters:
- ✅ **Includes**: Valid TTNN operations from `Allops.txt`
- ❌ **Excludes**: Infrastructure ops (to_device, from_device, deallocate, etc.)
- ❌ **Excludes**: 36 utility operations (full list in `generic_ops_tracer.py`)

### Analyzing Traced Configurations

#### View Configurations for an Operation

```bash
python model_tracer/analyze_operations.py <operation_name>
```

**Example Output:**
```
📊 Operation: ttnn::sigmoid_accurate
📊 Configurations: 30
================================================================================

📋 Configuration 1:
----------------------------------------
  arg0: Tensor(shape=[1, 1, 12544, 32], dtype=BFLOAT8_B, memory=L1_HEIGHT_SHARDED,
        shard(shard_shape=[224, 32], grid=[(0,0)→(7,6)], orientation=ROW_MAJOR))

📋 Configuration 2:
----------------------------------------
  arg0: Tensor(shape=[1, 1, 1, 8], dtype=BFLOAT8_B, memory=L1_WIDTH_SHARDED,
        shard(shard_shape=[32, 32], grid=[(0,0)→(0,0)], orientation=ROW_MAJOR))

...
```

#### Understanding the Output

- **shape**: Logical tensor shape `[batch, channels, height, width]`
- **dtype**: Data type (BFLOAT8_B, BFLOAT16, etc.)
- **memory**: Memory layout type (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, INTERLEAVED)
- **shard_shape**: Physical shard dimensions `[height, width]`
- **grid**: Core grid ranges `[(start_x, start_y)→(end_x, end_y)]`
- **orientation**: How data is distributed (ROW_MAJOR, COL_MAJOR)

### Integrating into Sweep Tests

Integration is **extremely simple**:

#### Step 1: Import Loader and Helper Function

```python
from tests.sweep_framework.master_config_loader import MasterConfigLoader, unpack_traced_config
```

#### Step 2: Load and Add to Parameters

```python
loader = MasterConfigLoader()

# Option 1 (Default): Run exact traced configs (e.g., 30 tests)
model_traced_params = loader.get_suite_parameters("your_operation_name")

# Option 2: Run all combinations (e.g., 30 shapes × unique dtypes × layouts × memory_configs)
# model_traced_params = loader.get_suite_parameters("your_operation_name", all_cases=True)

parameters = {
    "nightly": {
        # ... your existing nightly tests ...
    },
    "model_traced": model_traced_params,  # Just add this line!
}
```

**Choosing Between Modes:**
- **Default (`all_cases=False`)**: Runs exactly N traced configs as they appeared in real models
  - ✅ Fast - only 30 tests for 30 traced configs
  - ✅ Tests real-world usage patterns
  - ✅ Each test uses the exact config combination from production
- **All Cases (`all_cases=True`)**: Runs all possible combinations (Cartesian product)
  - 📊 Comprehensive - tests all combinations
  - ⏱️ Slower - can generate hundreds or thousands of tests
  - 🔬 Useful for finding edge cases across different parameter combinations

#### Step 3: Update Your Run Function

Add `traced_config_name` parameter and use the helper to unpack in ONE line:

```python
def run(
    input_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    traced_config_name=None,  # Add this parameter
    *,
    device,
) -> list:
    # Unpack all config values in ONE line
    if traced_config_name:
        input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config = unpack_traced_config(traced_config_name)

    # Everything is now ready to use!
    # ... your test implementation ...
```

**Why this approach?**
- ✅ **Ultra-minimal** - just 1 line to unpack all values
- ✅ **No boilerplate** - no module-level dictionaries, no manual parsing
- ✅ **Clean separation** - all complexity handled by the loader
- ✅ **Easy to adopt** - just import, load, and unpack
- ✅ **Generates exactly N test vectors** for N traced configs (no Cartesian product)

#### Complete Example

See `/home/ubuntu/tt-metal/tests/sweep_framework/sweeps/eltwise/unary/sigmoid_accurate/sigmoid_accurate.py` for a full working example.

**That's all you need!** The master config loader handles:
- ✅ Loading from master JSON
- ✅ Parsing tensor configurations (including UnparsedElements)
- ✅ Creating TTNN memory configs with exact shard specs
- ✅ Pairing configs to avoid Cartesian product explosion
- ✅ Formatting for sweep framework
- ✅ Everything else!

---

## How It Works

### 1. Tracing Phase

#### Step 1: Pytest Plugin Injection

```python
# generic_ops_tracer.py creates a pytest plugin dynamically
class OperationsTracingPlugin:
    def pytest_runtest_setup(self, item):
        # Start TTNN graph capture
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    def pytest_runtest_teardown(self, item, nextitem):
        # End capture and serialize
        captured_graph = ttnn.graph.end_graph_capture()
        trace_data = GraphTracerUtils.serialize_graph(captured_graph)
```

#### Step 2: Filtering Valid Operations

```python
def is_valid_operation(self, op_name):
    # 1. Check exclusion list (36 utility operations)
    if op_name in self.excluded_operations:
        return False

    # 2. Check Allops.txt (official list of TTNN operations)
    return op_name in self.valid_operations
```

#### Step 3: Data Cleaning

Handles complex C++ objects that can't be JSON serialized:

```python
def clean_operation_data(self, operation):
    # Fix C++ representations
    # "{32, 32}" → "[32, 32]"
    # [{"x":0,"y":0} - {"x":7,"y":7}] → [{"x":0,"y":0}, {"x":7,"y":7}]

    # Recursively clean all nested structures
    # Convert UnparsedElement entries when possible
```

#### Step 4: Deduplication & Storage

```python
def update_master_file(self, master_file_path, new_operations, test_name):
    # 1. Group by operation name
    # 2. Generate hash for each configuration (arguments only)
    # 3. Add only if hash is unique
    # 4. Update metadata (models, counts, timestamps)
```

### 2. Configuration Loading Phase

#### Step 1: Parse Master JSON

```python
# master_config_loader.py
class MasterConfigLoader:
    def get_operation_configs(self, operation_name):
        # Load master JSON
        # Extract configurations for specific operation
        # Returns list of argument configurations
```

#### Step 2: Extract Tensor Information

```python
# For each configuration:
for config in configs:
    # Extract shape: [1, 1, 12544, 32]
    shape = tensor_spec.get("logical_shape", [])

    # Extract dtype: DataType::BFLOAT8_B
    dtype = tensor_layout.get("dtype", "DataType::BFLOAT16")

    # Extract memory config with EXACT shard specs
    memory_config = parse_memory_config(memory_config_dict, shape)
```

#### Step 3: Parse Sharded Configs

```python
def parse_memory_config(self, memory_config: Dict, tensor_shape: list) -> Any:
    # Extract shard_shape from traced data
    shard_shape = [224, 32]  # EXACT traced values!

    # Extract grid
    grid = [{"x":0,"y":0}, {"x":7,"y":6}]

    # Build CoreRangeSet
    core_range_set = CoreRangeSet({CoreRange(start, end)})

    # Create ShardSpec with EXACT traced shard_shape
    shard_spec = ttnn.ShardSpec(core_range_set, shard_shape, orientation)

    # Create MemoryConfig
    return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)
```

#### Step 4: Format for Sweep Framework

```python
# master_config_loader.py - get_suite_parameters()
def get_suite_parameters(operation_name, all_cases=False):
    if all_cases:
        # Return separate lists for Cartesian product (N×M tests)
        return {
            "input_shape": [shape1, shape2, ...],
            "input_a_memory_config": [mem1, mem2, ...],
            "input_a_dtype": [dtype1, dtype2, ...],
        }
    else:
        # Return config names for exact paired configs (N tests)
        return {
            "traced_config_name": ["op_traced_0", "op_traced_1", ...],
        }
```

### 3. Sweep Test Execution Phase

#### Step 1: Load Config by Name

```python
# Sweep framework passes config name (string)
def run(traced_config_name=None, *, device):
    # Unpack all parameters in ONE line
    if traced_config_name:
        shape, dtype, layout, input_mem_config, output_mem_config = unpack_traced_config(traced_config_name)

    # Now you have all the config values:
    # shape = [1, 1, 12544, 32]
    # input_mem_config = SHARDED with exact shard_shape
    # layout = TILE_LAYOUT
    # dtype = bfloat8_b
```

#### Step 2: Create Tensor

```python
# Use extracted config to create tensor
ttnn_input = ttnn.from_torch(
    torch_input,
    dtype=dtype,              # bfloat8_b
    layout=layout,            # TILE_LAYOUT
    device=device,
    memory_config=mem_config  # HEIGHT_SHARDED with [224,32] shard_shape
)
```

#### Step 3: Run Operation

```python
# Operation runs with exact traced configuration
result = ttnn.sigmoid_accurate(ttnn_input, memory_config=output_mem_config)
```

#### Step 4: Validate

```python
# Check accuracy
pcc = check_with_pcc(torch_reference, ttnn_output, threshold=0.999)
```

---

## File Structure

```
tt-metal/
├── model_tracer/                      # Model tracing tools
│   ├── generic_ops_tracer.py          # Main tracing script
│   ├── analyze_operations.py          # Query tool for master JSON
│   ├── analyze_sweep_results.py       # Sweep results analyzer
│   ├── TRACED_CONFIGS_GUIDE.md        # This guide
│   └── traced_operations/             # Storage directory
│       └── ttnn_operations_master.json # Master configuration store
├── tests/sweep_framework/
│   ├── master_config_loader.py        # Config loader and utilities
│   ├── Allops.txt                     # Official TTNN operations list
│   └── sweeps/
│       └── eltwise/unary/sigmoid_accurate/
│           └── sigmoid_accurate.py    # Example sweep test with traced configs
└── test_sigmoid_traced_configs.py     # Unit test (30 hardcoded configs)
```

### Master JSON Structure

```json
{
  "operations": {
    "sigmoid_accurate": {
      "configurations": [
        [
          {
            "arg0": {
              "Tensor": {
                "tensor_spec": {
                  "logical_shape": [1, 1, 12544, 32],
                  "tensor_layout": {
                    "dtype": "DataType::BFLOAT8_B",
                    "memory_config": {
                      "buffer_type": "BufferType::L1",
                      "memory_layout": "TensorMemoryLayout::HEIGHT_SHARDED",
                      "shard_spec": {
                        "grid": [{"x":0,"y":0}, {"x":7,"y":6}],
                        "shape": [224, 32],
                        "orientation": "ShardOrientation::ROW_MAJOR"
                      }
                    }
                  }
                }
              }
            }
          }
        ]
      ]
    }
  },
  "metadata": {
    "models": ["test_efficientnetb0_model"],
    "unique_operations": 50,
    "total_configurations": 300,
    "last_updated": "2025-10-16 04:22:52"
  }
}
```

---

## Examples

### Example 1: Tracing EfficientNet

```bash
# Trace EfficientNet model
python model_tracer/generic_ops_tracer.py \
  models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py::test_efficientnetb0_model
```

**Output:**
```
🚀 TTNN Operations Tracer
==================================================
📁 test_ttnn_efficientnetb0.py
==================================================
🚀 Running test with operations tracing...
📋 Loaded 200 valid operations from Allops.txt
...
📈 Captured 847 operations
📝 Added 28 new unique configurations to master file

==================================================
📋 RESULTS
==================================================
Test Result: ✅ PASSED
📊 Captured: 847 operations, 52 unique types
💾 File: test_efficientnetb0_model_ops_20251016_042252.json (234,567 bytes)
🔧 Unique Configurations (current test):
   • ttnn::sigmoid_accurate: 30 unique configs (45x executed)
   • ttnn::add: 6 unique configs (89x executed)
   • ttnn::mul: 8 unique configs (67x executed)
   ...
```

### Example 2: Analyzing Sigmoid Configurations

```bash
python model_tracer/analyze_operations.py sigmoid_accurate
```

**Output:**
```
📊 Operation: ttnn::sigmoid_accurate
📊 Configurations: 30
================================================================================

📋 Configuration 1:
----------------------------------------
  arg0: Tensor(shape=[1, 1, 12544, 32], dtype=BFLOAT8_B,
               memory=L1_HEIGHT_SHARDED,
               shard(shard_shape=[224, 32],
                     grid=[(0,0)→(7,6)],
                     orientation=ROW_MAJOR))

📋 Configuration 2:
----------------------------------------
  arg0: Tensor(shape=[1, 1, 1, 8], dtype=BFLOAT8_B,
               memory=L1_WIDTH_SHARDED,
               shard(shard_shape=[32, 32],
                     grid=[(0,0)→(0,0)],
                     orientation=ROW_MAJOR))
...
```

### Example 3: Running Model-Traced Sweep Test

```bash
# Generate vectors for the model_traced suite
python3 tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --dump-file

# Run only the model_traced suite
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --vector-source vectors_export \
  --result-dest results_export \
  --suite model_traced

# Or run all suites (nightly, model_traced, etc.)
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --vector-source vectors_export \
  --result-dest results_export
```

**Output:**
```
✅ Loaded 30 model-traced configurations for sigmoid_accurate
   • Shapes: 30
   • Dtypes: 1 unique
   • Layouts: 1 unique
   • Memory Configs: 30 unique

SWEEPS - Generated 30 test vectors for suite model_traced.
...
test_sweep[model_traced-0] PASSED
test_sweep[model_traced-1] PASSED
...
test_sweep[model_traced-29] PASSED

======================== 30 passed in 33.45s ========================
```

### Example 4: Analyzing Sweep Results

```bash
python analyze_sweep_results.py
```

**Output:**
```
🔍 SWEEP TEST RESULTS ANALYSIS
============================================================
📊 Overall Test Statistics:
   • Total Tests: 30
   • ✅ Passed: 30 (100.0%)
   • ❌ Failed: 0
   • ⏱️  Avg Duration: 0.60s per test

🚀 Performance Metrics:
   • Min PCC: 0.999989
   • Max PCC: 1.000000
   • Avg PCC: 0.999996

🎉 No failures detected!
```

---

## Troubleshooting

### Issue: "No configurations found for operation"

**Cause:** The operation hasn't been traced yet or has a different name.

**Solution:**
```bash
# 1. Check what operations are available
python model_tracer/analyze_operations.py | grep "Operation:"

# 2. Trace a model that uses this operation
python model_tracer/generic_ops_tracer.py <model_test_path>

# 3. Check the operation name format (ttnn::op_name vs ttnn.op_name)
```

### Issue: "Shard Size must be multiple of tile size"

**Cause:** Traced configuration has non-tile-aligned shard shape with TILE_LAYOUT.

**Solution:** This shouldn't happen with the fixed `master_config_loader.py`. The loader now uses exact traced `shard_shape` values that were validated in the original model.

**Verification:**
```python
# Check the shard shape
python model_tracer/analyze_operations.py <operation> | grep "shard_shape"

# Shard dimensions should be multiples of 32 for TILE_LAYOUT
# e.g., [224, 32] ✓, [64, 96] ✓, [33, 17] ✗
```

### Issue: "Memory layout mismatch"

**Cause:** For unary operations, input and output memory layouts must match.

**Solution:**
```python
# In sweep test, ensure output_memory_config matches input
output_mem_config = mem_config  # Same as input for unary ops
```

### Issue: Sweep test fails but unit test passes

**Cause:** Sweep test might be using calculated shard_shape instead of traced values.

**Check:**
```python
# In master_config_loader.py, line ~204
# Should NOT use: ttnn.create_sharded_memory_config_()
# Should manually create: ShardSpec(core_range_set, shard_shape, orientation)
```

**Verify fix:**
```bash
# Unit test should pass
pytest test_sigmoid_traced_configs.py

# Sweep test should also pass
python3 tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate --dump-file
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --vector-source vectors_export --result-dest results_export --suite model_traced
```

### Issue: "UnparsedElement" errors

**Cause:** Complex C++ objects couldn't be JSON serialized during tracing.

**Solution:** The system includes regex-based fixes for common C++ formats. If you still see `UnparsedElement`:

1. The data might still be usable - check `analyze_operations.py` output
2. Some information is recovered via regex parsing
3. For completely unparseable data, it's marked as `PARSE_ERROR`

### Issue: Too many/few test cases

**Control test count with `all_cases` flag:**

```python
# Default: N exact tests (shapes paired with their memory configs)
configs = load_unary_op_configs("sigmoid_accurate", all_cases=False)  # 30 tests

# Cartesian product: N shapes × M memory_configs tests
configs = load_unary_op_configs("sigmoid_accurate", all_cases=True)   # 900 tests
```

---

## Advanced Topics

### Custom Operation Filtering

Edit `generic_ops_tracer.py` to customize which operations are traced:

```python
class OperationsTracingPlugin:
    def __init__(self):
        self.excluded_operations = {
            'ttnn::to_device',
            'ttnn::from_device',
            # Add your custom exclusions here
        }
```

### Adding New Sweep Tests

Here's the complete pattern:

```python
# tests/sweep_framework/sweeps/eltwise/unary/my_op/my_op.py
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Load traced configs (one line!)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("my_op")

# Add to parameters (one line!)
parameters = {
    "nightly": {
        # ... your existing tests ...
    },
    "model_traced": model_traced_params,  # Done!
}

# Your run function stays the same - no special handling needed!
def run(input_shape=None, input_a_dtype=None, ..., *, device):
    # Just use the parameters - everything is set up automatically
    pass
```

That's it! The loader handles everything else.

### Tracing Custom Models

```python
# 1. Ensure your model test uses ttnn operations
def test_my_custom_model(device):
    # ... model code using ttnn.add, ttnn.matmul, etc. ...
    pass

# 2. Trace it
python model_tracer/generic_ops_tracer.py path/to/test.py::test_my_custom_model

# 3. Configurations automatically added to master JSON
```

### Batch Tracing Multiple Models

```bash
#!/bin/bash
# trace_all_models.sh

# Trace all models in a directory
for test in models/demos/wormhole/*/demo/demo.py; do
    echo "Tracing $test"
    python model_tracer/generic_ops_tracer.py "$test::test_demo"
done

# Results accumulated in ttnn_operations_master.json
```

---

## Summary

### What You've Built

✅ **Automatic Configuration Extraction**: From any model test
✅ **Centralized Storage**: Single master JSON with deduplication
✅ **Ultra-Simple Integration**: Easy to add traced configs to any sweep test
✅ **Production Testing**: Validate with real model configurations, instant results
✅ **Analysis Tools**: Query and inspect configurations

### Workflow

```
1. Trace Model → 2. Store in Master JSON → 3. Use in Sweep Tests
     ⬇️                      ⬇️                        ⬇️
   Real-world          Deduplicated           Model-driven
   configs            configurations         model_traced
```

### Simple Integration

```python
from tests.sweep_framework.master_config_loader import MasterConfigLoader
loader = MasterConfigLoader()
parameters = {"model_traced": loader.get_suite_parameters("your_op")}
```

### Next Steps

1. **Trace more models**: Add ResNet, BERT, Whisper, etc.
2. **Expand coverage**: Add `model_traced` suite to more sweep tests
3. **Monitor results**: View in Superset dashboard
4. **Iterate**: As models evolve, re-trace to update configs

---

## Quick Reference

### Common Commands

```bash
# Trace a model
python model_tracer/generic_ops_tracer.py <test_path>

# View configurations
python model_tracer/analyze_operations.py <operation_name>

# Generate and run sweep test with traced configs
python3 tests/sweep_framework/sweeps_parameter_generator.py --module-name <op_name> --dump-file
python3 tests/sweep_framework/sweeps_runner.py --module-name <op_name> --suite model_traced \
  --vector-source vectors_export --result-dest results_export

# Analyze sweep results
python analyze_sweep_results.py
```

### Key Files

- **Tracer**: `model_tracer/generic_ops_tracer.py`
- **Master JSON**: `model_tracer/traced_operations/ttnn_operations_master.json`
- **Analyzer**: `model_tracer/analyze_operations.py`
- **Config Loader**: `tests/sweep_framework/master_config_loader.py`

### Important Concepts

- **Shard Shape**: Physical dimensions of data on each core
- **Grid**: Core range (which cores are used)
- **Memory Layout**: How data is distributed (SHARDED/INTERLEAVED)
- **Deduplication**: Only unique configs stored (via MD5 hash)
- **Config Lookup Pattern**: Avoid serialization issues in sweep framework

---

**Questions or Issues?** Check the troubleshooting section or examine the working examples in the codebase.

**Success!** 🎉 You now have a comprehensive system for model-driven sweep testing with real-world configurations.
