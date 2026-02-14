# TTNN Operations Database: Tracer Integration Guide (V2)

This guide explains how the model tracer should format its output to be compatible with the `ttnn_ops` database schema **version 2**.

## Version 2 Key Changes

**V2 introduces per-tensor placement** - each tensor argument stores its own placement information:

| Aspect | V1 (Legacy) | V2 (Current) |
|--------|-------------|--------------|
| **Arguments** | Array format `[{arg0: ...}, ...]` | Dictionary format `{arg0: ..., arg1: ...}` |
| **Placement** | Global at `machine_info.tensor_placements` | Per-tensor at `arg0.tensor_placement` |
| **machine_info** | Array with placement info | Single object, hardware only |
| **Mesh config** | Includes placement type | Only mesh shape, no placement |

## Quick Overview

The database stores traced TTNN operations with these key concepts:

1. **Operations** - TTNN operations like `ttnn::add`, `ttnn::matmul`
2. **Configurations** - Unique combinations of arguments + hardware + mesh
3. **Sources** - Which model/test file the configuration came from
4. **Deduplication** - Same (args + per-tensor placements + hardware + mesh) = same config
5. **Per-Tensor Placement** - Each tensor can have different placement strategies (replicate/shard)

## JSON Output Format

The tracer should output JSON in this structure:

```json
{
  "operations": {
    "ttnn::add": {
      "configurations": [
        {
          "config_id": 1,
          "arguments": {
            "arg0": {
              "type": "ttnn.Tensor",
              "original_shape": [1, 1, 32, 128],
              "original_dtype": "DataType.BFLOAT16",
              "layout": "Layout.TILE",
              "storage_type": "StorageType.DEVICE",
              "memory_config": {...},
              "tensor_placement": {
                "placement": "['PlacementShard(2)', 'PlacementShard(3)']",
                "distribution_shape": "[1, 2]",
                "mesh_device_shape": "[1, 2]"
              }
            },
            "arg1": {
              "type": "ttnn.Tensor",
              "original_shape": [1, 1, 32, 128],
              "original_dtype": "DataType.BFLOAT16",
              "layout": "Layout.TILE",
              "storage_type": "StorageType.DEVICE",
              "memory_config": {...}
            }
          },
          "executions": [
            {
              "source": "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]",
              "machine_info": {
                "board_type": "Wormhole",
                "device_series": "n300",
                "card_count": 2,
                "device_ids": [0, 1],
                "device_count": 2,
                "mesh_device_shape": [1, 2]
              },
              "count": 128
            }
          ]
        }
      ]
    }
  }
}
```

## Field Specifications

### `config_id` (Required)

Unique identifier for this configuration within the operation. Sequential integer starting from 1.

### `executions` (Required)

Array of execution instances where this configuration was observed. Each execution records:
- **source**: Which model/test traced this config
- **machine_info**: Hardware where it was executed
- **count**: How many times this config was executed in this source (for tracking frequency)

```python
"executions": [
    {
        "source": "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]",
        "machine_info": {
            "board_type": "Wormhole",
            "device_series": "n300",
            "card_count": 2,
            "device_ids": [0, 1],
            "device_count": 2,
            "mesh_device_shape": [1, 2]
        },
        "count": 128  # This config executed 128 times in this model
    }
]
```

**Multiple executions**: The same config can appear in multiple models/sources:

```python
"executions": [
    {"source": "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]", "machine_info": {...}, "count": 128},
    {"source": "models/demos/qwen/demo.py [HF_MODEL:Qwen/Qwen2.5-7B]", "machine_info": {...}, "count": 256}
]
```

### `source` Format (within executions)

Format: `"<file_path> [HF_MODEL:<huggingface_model_id>]"`

```python
# Examples:
"models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B-Instruct]"
"models/demos/whisper/demo.py"  # HF_MODEL tag is optional
```

The loader parses this to extract:
- `source_file` - The file path
- `hf_model_identifier` - The HuggingFace model name (if present)

### `machine_info` (Required, within each execution)

```python
{
    "board_type": "Wormhole",       # From tt-smi: board type
    "device_series": "n300",        # From tt-smi: device series (N150, n300, etc.)
    "card_count": 2,                # Number of physical cards
    "device_ids": [0, 1],           # Physical device IDs used
    "device_count": 2,              # Number of devices used
    "mesh_device_shape": [1, 2]     # Optional: Mesh grid dimensions (for multi-device)
}
```

**Note**: In v2 format, `machine_info` is nested within each execution and contains hardware info only. Tensor placement is now stored per-tensor in each argument.

### `tensor_placement` (Per-Tensor, Optional for multi-device)

Each tensor argument can have its own placement information:

```python
{
    "placement": "['PlacementShard(2)', 'PlacementShard(3)']",  # List of placement strategies per mesh dimension
    "distribution_shape": "[1, 2]",        # How tensor is distributed across mesh
    "mesh_device_shape": "[1, 2]"          # Mesh grid dimensions for this tensor
}
```

**Placement examples:**
- `"['PlacementReplicate']"` - Tensor replicated on all devices
- `"['PlacementShard(2)']"` - Tensor sharded on dimension 2
- `"['PlacementShard(2)', 'PlacementShard(3)']"` - Sharded on dims 2 and 3 across 2D mesh

**Important**: Each tensor can have different placement strategies. Same operation with different per-tensor placements = different configs.

### `arguments` (Required)

Dictionary (not array) mapping argument names to values. Keys are `arg0`, `arg1`, `arg2`, etc.:

```python
# Tensor argument (single-device)
"arg0": {
    "type": "ttnn.Tensor",
    "original_shape": [1, 1, 32, 128],
    "original_dtype": "DataType.BFLOAT16",
    "layout": "Layout.TILE",
    "storage_type": "StorageType.DEVICE",
    "memory_config": {
        "memory_layout": "TensorMemoryLayout.INTERLEAVED",
        "buffer_type": "BufferType.DRAM",
        "shard_spec": "None",
        "is_sharded": false,
        "interleaved": true
    }
}

# Tensor argument (multi-device with placement)
"arg1": {
    "type": "ttnn.Tensor",
    "original_shape": [1, 4, 1024, 128],
    "original_dtype": "DataType.BFLOAT16",
    "layout": "Layout.TILE",
    "storage_type": "StorageType.DEVICE",
    "memory_config": {...},
    "tensor_placement": {
        "placement": "['PlacementShard(2)', 'PlacementShard(3)']",
        "distribution_shape": "[1, 2]",
        "mesh_device_shape": "[1, 2]"
    }
}

# Scalar argument
"arg2": {
    "type": "int",
    "value": -1
}

# MemoryConfig argument
"arg3": {
    "type": "ttnn.MemoryConfig",
    "memory_layout": "TensorMemoryLayout.INTERLEAVED",
    "buffer_type": "BufferType.DRAM"
}

# Optional/None argument
"arg4": null
```

## Config Identity (Deduplication)

The database deduplicates configurations using a SHA-256 hash of:

```python
config_hash = SHA256({
    "operation": "ttnn::add",
    "arguments": {
        "arg0": {
            "type": "ttnn.Tensor",
            "shape": [1, 1, 32, 128],
            "dtype": "DataType.BFLOAT16",
            "layout": "Layout.TILE",
            "memory_config": {...},
            "tensor_placement": {...}  # Per-tensor placement included in hash
        },
        "arg1": {...}
    },
    "hardware": {
        "board_type": "Wormhole",
        "device_series": "n300",
        "card_count": 2
    },
    "mesh_device_shape": [1, 2]  # Global mesh shape
})
```

**What this means:**
- Same args (including per-tensor placements) + same hardware + same mesh → merged into one config
- Multiple sources sharing identical config → stored as multiple entries in `executions` array
- Each execution tracks: source, machine_info, and count (how many times it ran)
- Different per-tensor placement → separate configs (placement is part of tensor identity)
- Different hardware or mesh → separate configs (even with same args)

**Example - Multiple Executions of Same Config:**
```python
{
    "config_id": 42,
    "arguments": {...},  # Same arguments
    "executions": [
        {"source": "llama/demo.py", "machine_info": {...}, "count": 128},
        {"source": "qwen/demo.py", "machine_info": {...}, "count": 256}
    ]
}
```
Both models use the same config, tracked via 2 execution entries.

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

### 1. Missing `executions` Array

Configs must have executions array with source and machine info:

```python
# ❌ Bad - no executions
{
    "config_id": 1,
    "arguments": {...}
}

# ✅ Good
{
    "config_id": 1,
    "arguments": {...},
    "executions": [
        {
            "source": "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]",
            "machine_info": {...},
            "count": 128
        }
    ]
}
```

### 2. Inconsistent Tensor Formats

The loader expects tensors in a specific structure:

```python
# ❌ Bad - old format or incomplete
{"arg0": {"Tensor": {"dtype": "BFLOAT16", "shape": [1, 32]}}}

# ✅ Good - v2 format with complete structure
"arg0": {
    "type": "ttnn.Tensor",
    "original_shape": [1, 32],
    "original_dtype": "DataType.BFLOAT16",
    "layout": "Layout.TILE",
    "storage_type": "StorageType.DEVICE",
    "memory_config": {
        "memory_layout": "TensorMemoryLayout.INTERLEAVED",
        "buffer_type": "BufferType.DRAM",
        "is_sharded": false
    }
}
```

### 3. Per-Tensor Placement Format

When tracing multi-device tensors, placement must be stored per-tensor:

```python
# ❌ Bad - global placement at machine_info level (old format)
"machine_info": {
    "tensor_placements": [{
        "placement": "[PlacementReplicate]"
    }]
}

# ✅ Good - per-tensor placement (v2 format)
"arg0": {
    "type": "ttnn.Tensor",
    ...,
    "tensor_placement": {
        "placement": "['PlacementShard(2)', 'PlacementShard(3)']",
        "distribution_shape": "[1, 2]",
        "mesh_device_shape": "[1, 2]"
    }
}
```

### 4. Source Format Issues

Source must include file path, optionally with HF_MODEL tag:

```python
# ❌ Bad - only HF model, missing file path
"source": "meta-llama/Llama-3.2-1B"

# ✅ Good - file path with HF_MODEL tag
"source": "models/demos/llama/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]"

# ✅ Also good - file path without HF model
"source": "tests/ttnn/unit_tests/test_add.py"
```

### 5. Arguments Structure

```python
# ❌ Bad - arguments as array (old format)
"arguments": [
    {"arg0": {...}},
    {"arg1": {...}}
]

# ✅ Good - arguments as dictionary (v2 format)
"arguments": {
    "arg0": {...},
    "arg1": {...}
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
               ┌─────────────────┐     ┌─────────────────┐   ┌──────────────────────┐
               │ ttnn_hardware   │     │ ttnn_mesh_config│   │ ttnn_argument        │
               ├─────────────────┤     ├─────────────────┤   ├──────────────────────┤
               │ board_type      │     │ mesh_shape      │   │ arg_index            │
               │ device_series   │     │ device_count    │   │ is_tensor            │
               │ card_count      │     └─────────────────┘   │ tensor_dtype         │
               └─────────────────┘                           │ tensor_shape         │
                                                             │ tensor_placement_json│ ← Per-tensor
                                                             │ tensor_spec_json     │   placement
                                                             └──────────────────────┘
```

**Key Changes in V2:**
- `ttnn_mesh_config`: Stores only mesh shape and device count (no placement)
- `ttnn_argument.tensor_placement_json`: Per-tensor placement stored here (NEW)
- `ttnn_argument.tensor_spec_json`: Full tensor specification including memory config

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
