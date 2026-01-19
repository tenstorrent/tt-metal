# Sweep Matrix Computation

This document describes `compute_sweep_matrix.py`, which generates the GitHub Actions matrix for sweep test execution.

## Overview

The script analyzes generated sweep vector files and produces a matrix configuration that maps test modules to appropriate hardware runners. It supports different run types with distinct batching and runner assignment strategies.

## Run Types

### 1. Lead Models (`0 3 * * *` or "ALL SWEEPS (Lead Models)")
- **Purpose**: Execute sweeps traced from lead models (e.g., DeepSeek)
- **Strategy**: Mesh-aware runner assignment based on `mesh_device_shape`
- **Batch Size**: 10 modules per batch
- **Runner Assignment**: Configurable via `get_lead_models_mesh_runner_config()`

### 2. Nightly (Daily schedules except comprehensive)
- **Purpose**: Regular nightly sweep execution
- **Strategy**: All modules on standard wormhole runners
- **Batch Size**: 10 modules per batch
- **Suite**: `nightly`

### 3. Comprehensive (Wed/Sat: `0 4 * * 3,6`)
- **Purpose**: Exhaustive testing with all suite combinations
- **Strategy**: Smaller batches to avoid timeouts
- **Batch Size**: 3 modules per batch
- **Suite**: No override (runs all suites)

### 4. Model Traced (`0 4 * * *` or "ALL SWEEPS (Model Traced)")
- **Purpose**: Execute all model-traced sweeps
- **Strategy**: Standard runner assignment
- **Batch Size**: 10 modules per batch
- **Suite**: `model_traced`

## Usage

### In CI (GitHub Actions)

The script is called automatically by `.github/workflows/ttnn-run-sweeps.yaml`:

```yaml
- id: set-matrix
  env:
    GITHUB_EVENT_SCHEDULE: ${{ github.event.schedule }}
    GITHUB_EVENT_NAME: ${{ github.event_name }}
    SWEEP_NAME: ${{ github.event.inputs.sweep_name }}
    MEASURE_DEVICE_PERF: ${{ github.event.inputs.measure_device_perf }}
    VECTORS_DIR: /tmp/vectors
  run: |
    matrix_json=$(python3 tests/sweep_framework/framework/compute_sweep_matrix.py)
    echo "matrix=$matrix_json" >> "$GITHUB_OUTPUT"
```

### Local Testing

Test with sample environment:

```bash
cd tt-metal

# Test lead models run
export GITHUB_EVENT_SCHEDULE="0 3 * * *"
export GITHUB_EVENT_NAME="schedule"
export SWEEP_NAME=""
export MEASURE_DEVICE_PERF="false"
export VECTORS_DIR="tests/sweep_framework/vectors_export"

python3 tests/sweep_framework/framework/compute_sweep_matrix.py | python3 -m json.tool

# Test nightly run
export GITHUB_EVENT_SCHEDULE="0 0 * * 1"
python3 tests/sweep_framework/framework/compute_sweep_matrix.py | python3 -m json.tool
```

## Configuring Lead Models Runner Mapping

To modify which runners handle which mesh shapes, edit `get_lead_models_mesh_runner_config()` in compute_sweep_matrix.py:

```python
def get_lead_models_mesh_runner_config():
    return [
        {
            # Single-chip operations (1x1 mesh)
            "mesh_shapes": ["1x1"],
            "test_group_name": "lead-models-single-chip",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n150-stable",  # String: single label
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
        {
            # Small multi-chip operations (N300: 2-8 chips)
            "mesh_shapes": ["1x4", "1x8", "2x4"],
            "test_group_name": "lead-models-n300",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
        {
            # Large multi-chip operations (Galaxy TG: 32 chips)
            "mesh_shapes": ["4x8", "8x4"],
            "test_group_name": "lead-models-galaxy",
            "arch": "wormhole_b0",
            "runs_on": [  # Array: multiple labels for GitHub Actions
                "topology-6u",           # 32-chip galaxy topology
                "arch-wormhole_b0",      # Architecture
                "in-service",            # Availability
                "pipeline-functional",   # Pipeline type
            ],
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
    ]
```

### Runner Label Types

The `runs_on` field supports two formats:

1. **String format** (single label):
   ```python
   "runs_on": "tt-ubuntu-2204-n150-stable"
   ```
   Used for dedicated runners with a single identifying label.

2. **Array format** (multiple labels):
   ```python
   "runs_on": ["topology-6u", "arch-wormhole_b0", "in-service", "pipeline-functional"]
   ```
   Used for shared runners that need multiple labels for proper selection. Common for Galaxy and multi-chip configurations.

### Common Runner Labels in CI

| Label | Description | Example Use Case |
|-------|-------------|------------------|
| `topology-6u` | 32-chip Galaxy (TG) in 6U rack | Large model inference, multi-chip operations |
| `config-t3000` | T3000 configuration (8-chip) | Medium-scale multi-chip workloads |
| `arch-wormhole_b0` | Wormhole B0 architecture | Architecture-specific tests |
| `arch-blackhole` | Blackhole architecture | Next-gen hardware tests |
| `in-service` | Runner available for use | Production/stable runners |
| `pipeline-functional` | Functional test pipeline | Standard functional testing |
| `pipeline-model` | Model test pipeline | Model-specific testing |
| `bare-metal` | Bare-metal runner (not containerized) | Performance-sensitive tests |
| `tt-ubuntu-2204-n150-stable` | Single N150 chip (dedicated) | Single-chip operations |
| `tt-ubuntu-2204-n300-llmbox-viommu-stable` | N300 (2-8 chip) configuration | Small multi-chip operations |

**Note**: For Galaxy/multi-chip runners, use array format with multiple labels for proper runner selection. Single-chip runners typically use a single string label.

### Adding New Mesh Configurations

**Example 1: Dedicated runner with single label**

```python
{
    "mesh_shapes": ["4x4"],
    "test_group_name": "lead-models-galaxy-4x4",
    "arch": "wormhole_b0",
    "runs_on": "tt-ubuntu-2204-galaxy-4x4-stable",  # Single string
    "tt_smi_cmd": "tt-smi -r 0-15",
    "suite_name": "model_traced",
},
```

**Example 2: Galaxy runner with multiple labels (recommended for shared hardware)**

```python
{
    "mesh_shapes": ["4x8", "8x4"],
    "test_group_name": "lead-models-galaxy-32chip",
    "arch": "wormhole_b0",
    "runs_on": [  # Array of labels
        "topology-6u",         # Identifies 32-chip galaxy
        "arch-wormhole_b0",    # Architecture requirement
        "in-service",          # Availability status
        "pipeline-functional", # Pipeline assignment
    ],
    "tt_smi_cmd": "tt-smi -r",
    "suite_name": "model_traced",
},
```

**Example 3: Blackhole multi-chip configuration**

```python
{
    "mesh_shapes": ["2x4", "4x2"],
    "test_group_name": "lead-models-blackhole-galaxy",
    "arch": "blackhole",
    "runs_on": [
        "arch-blackhole",
        "topology-6u",
        "in-service",
        "pipeline-functional",
    ],
    "tt_smi_cmd": "tt-smi -r",
    "suite_name": "model_traced",
},
```

### Grouping Multiple Meshes

Multiple mesh shapes can share the same runner:

```python
{
    "mesh_shapes": ["2x2", "2x4", "4x2", "1x4", "1x8"],
    "test_group_name": "lead-models-medium-multi-chip",
    "arch": "wormhole_b0",
    "runs_on": "tt-ubuntu-2204-galaxy-medium-stable",
    "tt_smi_cmd": "tt-smi -r",
    "suite_name": "model_traced",
},
```

## Output Format

The script outputs JSON to stdout with the following structure:

```json
{
  "module": ["list", "of", "all", "modules"],
  "batches": ["comma,separated,modules", "per,batch"],
  "ccl_batches": ["ccl,specific,batches"],
  "include": [
    {
      "test_group_name": "runner-group-name",
      "arch": "wormhole_b0",
      "runs_on": "runner-label",
      "tt_smi_cmd": "tt-smi -r",
      "module_selector": "module1,module2",
      "batch_display": "display-label:module1,module2",
      "suite_name": "suite-name-or-null"
    }
  ]
}
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GITHUB_EVENT_SCHEDULE` | Cron schedule expression | `0 3 * * *` |
| `GITHUB_EVENT_NAME` | GitHub event type | `schedule` |
| `SWEEP_NAME` | Manual run type selection | `ALL SWEEPS (Lead Models)` |
| `MEASURE_DEVICE_PERF` | Enable performance measurement | `true` |
| `VECTORS_DIR` | Path to vector JSON files | `/tmp/vectors` |

## Error Handling

The script validates:
- **Vector directory exists**: Exits with error if `VECTORS_DIR` not found
- **Vector files present**: Exits if no JSON files found
- **Matrix size limits**: GitHub Actions limits matrices to 256 entries
- **Unmapped mesh shapes**: Warns if mesh shapes have no runner config

## Debugging

To see detailed logging:

```bash
python3 tests/sweep_framework/framework/compute_sweep_matrix.py 2>&1 | tee matrix_debug.log
```

Stderr contains:
- Warnings about unmapped mesh shapes
- Summary of modules per mesh shape
- Total matrix entries generated

## Maintenance

### When to Update This Script

1. **Adding new hardware**: Update `get_lead_models_mesh_runner_config()`
2. **Changing batch sizes**: Modify batch size logic in `main()`
3. **New run types**: Add new detection and matrix computation logic
4. **Changing suite names**: Update suite name mapping in `compute_standard_matrix()`

### Testing Changes

Always test locally before committing:

```bash
# Test all run types
for schedule in "0 3 * * *" "0 0 * * 1" "0 4 * * 3,6" "0 4 * * *"; do
    echo "Testing schedule: $schedule"
    GITHUB_EVENT_SCHEDULE="$schedule" \
    GITHUB_EVENT_NAME="schedule" \
    VECTORS_DIR="tests/sweep_framework/vectors_export" \
    python3 tests/sweep_framework/framework/compute_sweep_matrix.py > /dev/null
    echo "✓ Passed"
done
```

## Related Files

- `.github/workflows/ttnn-run-sweeps.yaml`: Workflow that calls this script
- `tests/sweep_framework/sweeps_parameter_generator.py`: Generates vector files
- `tests/sweep_framework/framework/vector_source.py`: Loads vectors for execution
