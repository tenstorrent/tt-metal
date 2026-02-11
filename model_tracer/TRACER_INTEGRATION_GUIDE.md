# TTNN Operations Database: Tracer Integration Guide

This guide explains how the model tracer should format its output to be compatible with the `ttnn_ops` database schema.

## Quick Overview

The database stores traced TTNN operations with these key concepts:

1. **Operations** - TTNN operations like `ttnn::add`, `ttnn::matmul`
2. **Configurations** - Unique combinations of arguments + hardware + mesh
3. **Sources** - Which model/test file the configuration came from
4. **Deduplication** - Same (args + hardware + mesh) = same config, even from different sources

## JSON Output Format

The tracer should output JSON in this structure:

```json
{
  "operations": {
    "ttnn::add": {
      "configurations": [
        {
          "arguments": [
            {"arg0": {"Tensor": {...}}},
            {"arg1": {"Tensor": {...}}},
            {"arg2": {"MemoryConfig": {...}}}
          ],
          "source": "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]",
          "machine_info": [
            {
              "board_type": "Wormhole",
              "device_series": "n300",
              "card_count": 1,
              "tensor_placements": [
                {
                  "mesh_device_shape": "[2, 4]",
                  "placement": "[PlacementReplicate]"
                }
              ]
            }
          ]
        }
      ]
    }
  }
}
```

## Field Specifications

### `source` (Required)

Format: `"<file_path> [HF_MODEL:<huggingface_model_id>]"`

```python
# Examples:
"models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B-Instruct]"
"models/demos/whisper/demo.py"  # HF_MODEL tag is optional
```

The loader parses this to extract:
- `source_file` - The file path
- `hf_model_identifier` - The HuggingFace model name (if present)

### `machine_info` (Required for hardware tracking)

```python
{
    "board_type": "Wormhole",     # From tt-smi: board type
    "device_series": "n300",      # From tt-smi: device series
    "card_count": 1,              # Number of cards
    "tensor_placements": [...]    # Optional: mesh configuration
}
```

### `tensor_placements` (Optional, for multi-device)

```python
{
    "mesh_device_shape": "[2, 4]",      # Mesh grid dimensions as string
    "placement": "[PlacementReplicate]", # or "[PlacementShard(0)]" for shard dim 0
    "distribution_shape": "[8]"          # Optional
}
```

**Important**: `tensor_placements` affects config identity. Same arguments with different mesh = different configs.

### `arguments` (Required)

Array of argument dictionaries. Each element is `{argN: value}`:

```python
# Tensor argument
{"arg0": {"Tensor": {
    "storage_type": "StorageType::DEVICE",
    "tensor_spec": {
        "logical_shape": [1, 1, 32, 128],
        "tensor_layout": {
            "dtype": "DataType::BFLOAT16",
            "memory_config": {
                "memory_layout": "TensorMemoryLayout::INTERLEAVED",
                "buffer_type": "BufferType::DRAM"
            }
        }
    }
}}}

# Scalar argument
{"arg1": "-1"}

# MemoryConfig argument
{"arg2": {"MemoryConfig": {
    "memory_layout": "TensorMemoryLayout::INTERLEAVED",
    "buffer_type": "BufferType::DRAM"
}}}

# Null/optional argument
{"arg3": "nullopt"}
```

## Config Identity (Deduplication)

The database deduplicates configurations using a SHA-256 hash of:

```python
config_hash = SHA256({
    "operation": "ttnn::add",
    "arguments": [...],  # The full arguments array
    "hardware": ("Wormhole", "n300", 1),  # (board_type, device_series, card_count)
    "mesh": {
        "mesh_shape": [2, 4],
        "placement_type": "replicate",
        "shard_dim": None
    }
})
```

**What this means:**
- Same args + same hardware + same mesh → merged into one config
- Multiple sources sharing a config → linked via junction table
- Different hardware or mesh → separate configs (even with same args)

## Loading Data

```bash
# Set database connection
export NEON_CONNECTION_STRING="postgresql://..."

# Load JSON to database
python tests/sweep_framework/load_ttnn_ops_data_v2.py load

# Reconstruct full JSON from database
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct output.json

# Reconstruct single operation (faster for testing)
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-op ttnn::add output.json

# Compare original JSON vs reconstructed (verify round-trip)
python tests/sweep_framework/load_ttnn_ops_data_v2.py verify original.json reconstructed.json

# Detect duplicate configurations in JSON (before loading)
python tests/sweep_framework/load_ttnn_ops_data_v2.py duplicates input.json
python tests/sweep_framework/load_ttnn_ops_data_v2.py duplicates input.json ttnn::add  # Filter by operation

# Find line numbers for specific configs (debugging)
python tests/sweep_framework/load_ttnn_ops_data_v2.py find-lines ttnn::add 0,1,5
```

## Common Pitfalls

### 1. Missing `machine_info`

Without hardware info, the loader cannot properly deduplicate configs:

```python
# ❌ Bad - no machine_info
{"arguments": [...], "source": "..."}

# ✅ Good
{"arguments": [...], "source": "...", "machine_info": [{...}]}
```

### 2. Inconsistent Tensor Formats

The loader expects tensors in a specific structure:

```python
# ❌ Bad - missing tensor_spec
{"arg0": {"Tensor": {"dtype": "BFLOAT16", "shape": [1, 32]}}}

# ✅ Good - full structure
{"arg0": {"Tensor": {
    "storage_type": "StorageType::DEVICE",
    "tensor_spec": {
        "logical_shape": [1, 32],
        "tensor_layout": {
            "dtype": "DataType::BFLOAT16",
            "memory_config": {...}
        }
    }
}}}
```

### 3. Source Format Issues

```python
# ❌ Bad - will lose HF model info
"meta-llama/Llama-3.2-1B"

# ✅ Good
"models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]"
```

### 4. Array Sources (Multiple Models)

When the same config appears in multiple models during one trace:

```python
# ✅ Supported - source can be a list
{
    "arguments": [...],
    "source": [
        "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]",
        "models/demos/qwen/demo.py [HF_MODEL:Qwen/Qwen2.5-7B]"
    ],
    "machine_info": [...]
}
```

## Database Schema (Simplified)

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ ttnn_operation  │     │ ttnn_configuration  │     │ ttnn_model      │
├─────────────────┤     ├─────────────────────┤     ├─────────────────┤
│ operation_name  │◄────│ operation_id        │     │ source_file     │
│ base_name       │     │ config_hash (unique)│     │ hf_model_id     │
└─────────────────┘     │ hardware_id ────────┼──┐  │ model_family    │
                        │ mesh_config_id ─────┼──┼──│ is_lead_model   │
                        │ full_config_json    │  │  └────────┬────────┘
                        └──────────┬──────────┘  │           │
                                   │             │           │
                        ┌──────────┴──────────┐  │  ┌────────┴────────┐
                        │ ttnn_config_model   │  │  │ Junction table  │
                        │ (many-to-many)      │◄─┼──│ config ↔ model  │
                        └─────────────────────┘  │  └─────────────────┘
                                                 │
                        ┌────────────────────────┼────────────────────┐
                        │                        │                    │
                        ▼                        ▼                    ▼
               ┌─────────────────┐     ┌─────────────────┐   ┌─────────────────┐
               │ ttnn_hardware   │     │ ttnn_mesh_config│   │ ttnn_argument   │
               ├─────────────────┤     ├─────────────────┤   ├─────────────────┤
               │ board_type      │     │ mesh_shape      │   │ arg_index       │
               │ device_series   │     │ placement_type  │   │ is_tensor       │
               │ card_count      │     │ shard_dim       │   │ tensor_dtype    │
               └─────────────────┘     └─────────────────┘   │ tensor_shape    │
                                                             └─────────────────┘
```

## Querying Configs by Mesh

After loading, you can filter configs by mesh shape:

```sql
-- Find all configs using 2x4 mesh
SELECT o.operation_name, c.config_hash
FROM ttnn_ops.ttnn_configuration c
JOIN ttnn_ops.ttnn_operation o ON c.operation_id = o.ttnn_operation_id
JOIN ttnn_ops.ttnn_mesh_config mc ON c.mesh_config_id = mc.ttnn_mesh_config_id
WHERE mc.mesh_shape = '{2,4}';
```

## Test Result Correlation

The reconstructed JSON includes `config_hash` for each config:

```json
{
  "arguments": [...],
  "config_hash": "abc123...",  // ← SHA-256 hash
  "source": "...",
  "machine_info": [...]
}
```

Sweep test results use this hash as `input_hash`, enabling direct JOINs:

```sql
SELECT * FROM sweep_results sr
JOIN ttnn_ops.ttnn_configuration c
  ON sr.input_hash = c.config_hash;
```

## Files

| File | Purpose |
|------|---------|
| `tests/sweep_framework/load_ttnn_ops_data_v2.py` | Load JSON → DB, reconstruct DB → JSON |
| `model_tracer/generic_ops_tracer.py` | Trace models and output JSON |
| `model_tracer/destructively_create_ttnn_ops_schema_v2.sql` | Database schema DDL |

## Environment Variables

```bash
# For local development
export NEON_CONNECTION_STRING="postgresql://user:pass@host/db"

# For CI (same value, different name)
export TTNN_OPS_DATABASE_URL="postgresql://user:pass@host/db"
```

The loader checks both: `TTNN_OPS_DATABASE_URL` first, then `NEON_CONNECTION_STRING`.
