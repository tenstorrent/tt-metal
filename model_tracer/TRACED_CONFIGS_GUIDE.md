# Model Trace to Test Automation Guide

## Overview

Automatically extracts real-world operation configurations from model tests and integrates them into sweep tests. Tests run with the exact configurations that production models use.

**Benefits:**
- ✅ Test with real model configurations (EfficientNet, ResNet, BERT, etc.)
- ✅ Automatic extraction and deduplication
- ✅ Simple 3-line integration into sweep tests
- ✅ Captures shapes, dtypes, layouts, and exact shard specs

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| **Trace a model** | `python model_tracer/generic_ops_tracer.py <test_path>` |
| **View configurations** | `python model_tracer/analyze_operations.py <operation_name>` |
| **Generate sweep vectors** | `python3 tests/sweep_framework/sweeps_parameter_generator.py --module-name <op_name> --dump-file` |
| **Run sweep test** | `python3 tests/sweep_framework/sweeps_runner.py --module-name <op_name> --suite model_traced` |

### Key Files

- **Tracer**: `model_tracer/generic_ops_tracer.py`
- **Master JSON**: `model_tracer/traced_operations/ttnn_operations_master.json`
- **Analyzer**: `model_tracer/analyze_operations.py`
- **Config Loader**: `tests/sweep_framework/master_config_loader.py`

---

## Integration Pattern

### Unary Operations (1 input)

**Just 3 simple changes:**

```python
# 1. Import
from tests.sweep_framework.master_config_loader import MasterConfigLoader, unpack_traced_config

# 2. Load and add to parameters
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("your_operation_name")

parameters = {
    "nightly": { ... },
    "model_traced": model_traced_params,  # Add this line!
}

# 3. Update run() function
def run(
    input_shape,  # Required parameter
    input_a_dtype=ttnn.bfloat16,  # Set actual defaults
    input_a_layout=ttnn.TILE_LAYOUT,
    input_a_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    traced_config_name=None,  # Add this parameter
    *,
    device,
):
    # Unpack in ONE line
    if traced_config_name:
        input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config = unpack_traced_config(traced_config_name)

    # Rest of your test logic stays the same!
```

### Binary Operations (2 inputs)

```python
# 1. Import BINARY helper
from tests.sweep_framework.master_config_loader import MasterConfigLoader, unpack_binary_traced_config

# 2. Load (same as unary)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("add")

parameters = {
    "nightly": { ... },
    "model_traced": model_traced_params,
}

# 3. Unpack BINARY config
def run(
    input_shape,  # Required parameter
    input_a_dtype=ttnn.bfloat16,  # Set actual defaults
    input_b_dtype=ttnn.bfloat16,
    input_a_layout=ttnn.TILE_LAYOUT,
    input_b_layout=ttnn.TILE_LAYOUT,
    input_a_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    input_b_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    traced_config_name=None,
    *,
    device,
):
    if traced_config_name:
        input_shape, input_a_dtype, input_b_dtype, input_a_layout, input_b_layout, \
            input_a_memory_config, input_b_memory_config = unpack_binary_traced_config(traced_config_name)
```

**Test Modes:**
- **Default**: Runs exact N traced configs (fast, real-world patterns)
- **All cases**: `loader.get_suite_parameters("op_name", all_cases=True)` - Cartesian product (comprehensive, slower)

---

## Usage

### 1. Trace a Model

```bash
# Basic usage
python model_tracer/generic_ops_tracer.py /path/to/model/test.py::test_function

# Example
python model_tracer/generic_ops_tracer.py models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py::test_efficientnetb0_model

# Keep trace files (default: auto-deleted after adding to master)
python model_tracer/generic_ops_tracer.py <test_path> --store
```

**Captures:**
- Operation names (e.g., `sigmoid_accurate`, `add`)
- Tensor shapes (e.g., `[1, 1, 12544, 32]`)
- Data types (e.g., `BFLOAT8_B`, `BFLOAT16`)
- Memory layouts (e.g., `HEIGHT_SHARDED`, `INTERLEAVED`)
- Exact shard specifications (grid, shard_shape, orientation)

**Output:**
- Updates `model_tracer/traced_operations/ttnn_operations_master.json`
- Shows summary of unique configurations added

### 2. View Configurations

```bash
# View all configs for an operation
python model_tracer/analyze_operations.py sigmoid_accurate
```

**Example output:**
```
📊 Operation: ttnn::sigmoid_accurate
📊 Configurations: 30

📋 Configuration 1:
  arg0: Tensor(shape=[1, 1, 12544, 32], dtype=BFLOAT8_B,
               memory=L1_HEIGHT_SHARDED,
               shard(shard_shape=[224, 32], grid=[(0,0)→(7,6)], orientation=ROW_MAJOR))
```

### 3. Run Sweep Tests

```bash
# Generate test vectors
python3 tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --dump-file

# Run model_traced suite
python3 tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.sigmoid_accurate.sigmoid_accurate \
  --suite model_traced
```

---

## File Structure

```
tt-metal/
├── model_tracer/
│   ├── generic_ops_tracer.py          # Main tracing script
│   ├── analyze_operations.py          # Query tool
│   └── traced_operations/
│       └── ttnn_operations_master.json # Master config storage
└── tests/sweep_framework/
    ├── master_config_loader.py        # Config loader & utilities
    └── sweeps/
        └── eltwise/unary/sigmoid_accurate/
            └── sigmoid_accurate.py    # Example sweep test
```

---

## Complete Example

See `/home/ubuntu/tt-metal/tests/sweep_framework/sweeps/eltwise/unary/sigmoid_accurate/sigmoid_accurate.py` for a full working example.

---

## Workflow

```
1. Trace Model → 2. Store in Master JSON → 3. Use in Sweep Tests
     ⬇️                      ⬇️                        ⬇️
   Real-world          Deduplicated           Model-driven
   configs            configurations         validation
```

**Simple as:**
```python
from tests.sweep_framework.master_config_loader import MasterConfigLoader
loader = MasterConfigLoader()
parameters = {"model_traced": loader.get_suite_parameters("your_op")}
```

---

For complete documentation, see [Sweep Framework README](tests/sweep_framework/README.md).
