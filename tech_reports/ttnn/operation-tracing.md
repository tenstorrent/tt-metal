# TTNN Operation Parameter Tracing

Serialize all `ttnn` operation parameters and return values to JSON files for debugging and analysis.

## Quick Start

**Note**: Tracing requires `enable_fast_runtime_mode=false`. Fast mode is enabled by default for performance.

### With pytest

```bash
TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' pytest your_test.py --trace-params
```

### In Python code

```bash
# Set environment variable before running Python
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
```

```python
import ttnn
import ttnn.operation_tracer

ttnn.operation_tracer.enable_tracing(True)

# Your operations are now traced
result = ttnn.add(tensor_a, tensor_b)
```

Trace files are saved to `generated/ttnn/operation_parameters/` by default.

## API Reference

```python
import ttnn.operation_tracer

# Enable/disable tracing
ttnn.operation_tracer.enable_tracing(True)
ttnn.operation_tracer.enable_tracing(False)

# Check if tracing is enabled
ttnn.operation_tracer.is_tracing_enabled()

# Enable tensor value serialization (disabled by default)
ttnn.operation_tracer.enable_tensor_value_serialization(True)
```

## Trace File Format

### File Naming

```
{operation_id}_{operation_name}_{timestamp}.json
```

Example: `3_ttnn_add_20260115_104616_345678.json`

### JSON Structure

See reference examples in [`operation_tracing_examples/`](./operation_tracing_examples/).

Each trace file contains:

| Field | Description |
|-------|-------------|
| `operation_id` | Sequential ID (1, 2, 3, ...) |
| `operation_name` | e.g., `"ttnn.add"` |
| `args` | Positional arguments with `position` and `value` |
| `kwargs` | Keyword arguments |
| `return_value` | Serialized return value |

### Tensor Fields

| Field | Description |
|-------|-------------|
| `type` | `"ttnn.Tensor"` or `"torch.Tensor"` |
| `shape` | Tensor shape as list |
| `dtype` | Data type |
| `layout` | `"TILE"`, `"ROW_MAJOR"`, etc. (ttnn only) |
| `storage_type` | `"DEVICE"`, `"HOST"`, etc. (ttnn only) |
| `values` | Tensor data (only when value serialization enabled) |

## Configuration

### Custom Output Directory

```python
ttnn.CONFIG.root_report_path = "/path/to/traces"
```

### Tensor Value Serialization

By default, only tensor metadata is saved. To include actual values:

```python
ttnn.operation_tracer.enable_tensor_value_serialization(True)
```

**Warning**: This significantly increases file size and overhead. Use only when debugging specific values.

## Performance

| Mode | Overhead |
|------|----------|
| Disabled | Zero (single boolean check) |
| Metadata only (default) | Minimal |
| With values | Significant (CPU transfer + serialization) |

## Limitations

1. **Requires non-fast runtime mode**: Tracing is disabled when `enable_fast_runtime_mode=true` (the default). Set `TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'` to enable tracing.
2. **Large tensors with values**: Creates very large JSON files
3. **Device tensors**: Moved to CPU for value serialization

## Troubleshooting

**No trace files created?**
```python
print(ttnn.operation_tracer.is_tracing_enabled())  # Should be True
print(ttnn.CONFIG.root_report_path)  # Check output directory
```

**Files too large?**
- Disable value serialization (default behavior)
- Use smaller test tensors
