# Model Tracer Guide

The model tracer extracts real operation configurations from running models and stores them in a PostgreSQL database. From there, configurations can be reconstructed into JSON and used as sweep test vectors — in CI, on a branch, or in nightly runs.

**Schema:** All data lives in `ttnn_ops_v2_5` on Metal Ops PostgreSQL.
**Connection:** Set `TTNN_OPS_DATABASE_URL`

---

## Quick Start

```bash
# 1. Trace a model
python model_tracer/generic_ops_tracer.py models/demos/deepseek_v3/demo/demo.py

# 2. Load into DB (creates trace_run, auto-appends draft to manifest)
python tests/sweep_framework/load_ttnn_ops_data_v2.py load

# 3. Promote the trace in the manifest (edit model_tracer/sweep_manifest.yaml)
#    Change status: draft → active, add to targets if needed

# 4. Reconstruct configs for testing
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest \
    model_tracer/sweep_manifest.yaml \
    model_tracer/traced_operations/ttnn_operations_master.json
```

---

## End-to-End Workflow

### Step 1: Trace a model

Run `generic_ops_tracer.py` against any model demo or test. It runs the model with `--trace-params` enabled, collects per-operation JSON files, deduplicates them, and writes to a master JSON file.

```bash
# Standalone script
python model_tracer/generic_ops_tracer.py models/demos/deepseek_v3/demo/demo.py

# Pytest test
python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text

# With HF model selection
HF_MODEL=meta-llama/Llama-3.2-1B-Instruct \
    python model_tracer/generic_ops_tracer.py models/tt_transformers/demo/simple_text_demo.py::test_demo_text

# Keep raw trace files for later import (e.g., traced on a remote machine)
python model_tracer/generic_ops_tracer.py models/demos/deepseek_v3/demo/demo.py --store

# Import raw traces collected on another machine
python model_tracer/generic_ops_tracer.py --load /path/to/trace_dir
```

Output: `model_tracer/traced_operations/ttnn_operations_master.json`

### Step 2: Load into DB

```bash
python tests/sweep_framework/load_ttnn_ops_data_v2.py load

# Load a specific JSON file
python tests/sweep_framework/load_ttnn_ops_data_v2.py load path/to/master.json

# Load with explicit tt-metal SHA (auto-detected from git if omitted)
python tests/sweep_framework/load_ttnn_ops_data_v2.py load path/to/master.json abc123def456
```

On load the tool:
- Deduplicates configs by `config_hash` (same op + args + hardware + mesh = same config)
- Creates a `trace_run` row capturing model, hardware, and tt-metal SHA
- Links all configs to that trace via `trace_run_config` (new and pre-existing configs both get linked)
- **Auto-appends a `draft` entry** to `model_tracer/sweep_manifest.yaml`

Example output:
```
Loaded 623 configurations (47 new, 576 pre-existing), 891 arguments, 623 model links
Trace runs created: 1, total configs tied to traces: 623
Appended 1 draft entries to manifest registry (model_tracer/sweep_manifest.yaml)
```

### Step 3: Promote in the manifest

The loader appends the trace as `status: draft`. Review and promote it to `active`, then add it to `targets` with the appropriate scope:

```yaml
# model_tracer/sweep_manifest.yaml
targets:
  - model: deepseek_v3
    scope: lead_models    # ← included in both lead_models and model_traced CI runs

registry:
  - trace_id: 42
    status: active        # ← change from draft
    models: [deepseek_v3]
    hardware: {board_type: Wormhole, device_series: tt-galaxy-wh, card_count: 32}
    tt_metal_sha: abc123def456
    config_count: 323
    loaded_at: '2026-03-21'
    notes: 'Post-matmul refactor re-trace'
```

Traces left as `draft` are invisible to model pattern resolution. They can still be used by pinning `trace_id` directly in a target.

### Step 4: Reconstruct configs for testing

```bash
# All targets (model_traced scope — used for branch testing and model_traced CI run)
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest \
    model_tracer/sweep_manifest.yaml \
    model_tracer/traced_operations/ttnn_operations_master_model_traced.json \
    model_traced

# Lead models only (used for lead_models CI run)
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest \
    model_tracer/sweep_manifest.yaml \
    model_tracer/traced_operations/ttnn_operations_master_lead_models.json \
    lead_models

# Dry-run: see which trace IDs will be used without hitting the DB
python tests/sweep_framework/load_ttnn_ops_data_v2.py resolve-manifest \
    model_tracer/sweep_manifest.yaml model_traced

# Reconstruct a single specific trace
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-trace 35 output.json
```

The reconstructed JSON is in the same format as the tracer output and can be fed directly to `sweeps_parameter_generator.py`.

---

## Sweep Manifest

File: `model_tracer/sweep_manifest.yaml`

The manifest is the single place that controls which configs are included when you reconstruct. It has two sections: **targets** (what to reconstruct) and **registry** (log of all known traces).

### Targets

```yaml
targets:
  # Latest active trace per hardware for this model — included in both CI run types
  - model: deepseek_v3
    scope: lead_models

  # Pin an exact trace by ID — included in model_traced CI run only
  - trace_id: 538
    scope: model_traced

  # Latest active trace on specific hardware only
  - model: llama
    hardware: n300
    scope: model_traced

  # Two traces merged (regression comparison)
  - trace_id: 3
    scope: model_traced
  - trace_id: 7
    scope: model_traced
```

| Form | Resolution |
|---|---|
| `model: X` | Latest `active` registry entry per unique `device_series` where model matches X |
| `model: X, hardware: H` | Latest `active` entry where `device_series == H` and model matches X |
| `model: X, trace_id: N` | Trace N exactly. No resolution. Status ignored. |
| `trace_id: N` | Trace N directly. No model resolution needed. |

**Scope values:**

| Scope | Included in `lead_models` run | Included in `model_traced` run |
|---|---|---|
| `lead_models` | ✅ | ✅ |
| `model_traced` | ❌ | ✅ |

- "Latest" = highest `trace_id` among matching active entries.
- Model matching is **case-insensitive substring** against the `models` list in each registry entry.
- Multiple targets resolving to the same `trace_id` are deduplicated.
- Configs appearing in multiple traces are deduplicated by `config_hash`.
- `LEAD_MODELS` in `tests/sweep_framework/framework/constants.py` is derived from manifest targets with `scope: lead_models` at import time.

### Registry

Append-only log. Auto-updated on each `load`. Status is manually curated.

```yaml
registry:
  - trace_id: 538
    status: active              # active | draft | deprecated
    models: [whisper, llama, phi, mistral, qwen, deepseek, stable_diffusion,
             vit, segmentation, falcon7b, sentence_bert, efficientnetb0]
    hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
    tt_metal_sha: null
    config_count: 7030
    loaded_at: '2026-03-14'
    notes: 'Consolidated n300 trace — all models (v2.5 migration)'
```

| Field | Description |
|---|---|
| `trace_id` | Maps to `trace_run_id` in `ttnn_ops_v2_5.trace_run` |
| `status` | `active`: included in model pattern resolution. `draft`: auto-appended on load, not yet reviewed. `deprecated`: excluded from resolution, kept for history. |
| `models` | Model family names (derived from source path or HF identifier). Used for model pattern matching. |
| `hardware` | `board_type`, `device_series`, `card_count` |
| `tt_metal_sha` | Git SHA of tt-metal at trace time (`null` for migrated historical traces) |
| `config_count` | Number of configs linked to this trace |
| `loaded_at` | Date the trace was loaded |
| `notes` | Free text |

### Current traces

| trace_id | hardware | models | configs | status |
|---|---|---|---|---|
| 1 | p150b | whisper | 318 | active |
| 35 | tt-galaxy-wh | deepseek_v3 | 323 | active |
| 538 | n300 | all 12 model families | 7,030 | active |

---

## CI Integration

The sweep manifest drives two scheduled CI run types in `.github/workflows/ttnn-run-sweeps.yaml`:

### How it works

```
generate-master-json-from-db (ubuntu-latest, DB access)
  ├── reconstruct-manifest ... lead_models  → ttnn_operations_master_lead_models.json
  └── reconstruct-manifest ... model_traced → ttnn_operations_master_model_traced.json
       ↓ artifacts
ttnn-generate-sweeps (hardware runner, no DB access)
  ├── lead_models run:   TTNN_MASTER_JSON_PATH=*_lead_models.json
  │     sweeps_parameter_generator.py --model-traced lead --suite-name model_traced
  └── model_traced run:  TTNN_MASTER_JSON_PATH=*_model_traced.json
        sweeps_parameter_generator.py --model-traced all
```

| Run type | Schedule | Scope | Results pushed to DB |
|---|---|---|---|
| `lead_models` | 2 AM UTC daily | `lead_models` targets only | Yes |
| `model_traced` | 3 AM UTC daily | All targets | No |

**To add a model to lead_models:** add a target with `scope: lead_models` and promote the trace to `active` in the registry. The manifest is the single source of truth — `LEAD_MODELS` in `constants.py` is read from it automatically.

---

## Database Schema

All tables live in the `ttnn_ops_v2_5` schema.

```
ttnn_operation          ttnn_model
    │ 1:N                   │ 1:N
    ▼                       ▼
ttnn_configuration ◄── trace_run_config ◄── trace_run
    │  config_hash               (N:M)          │ model_id
    │  full_config_json                         │ board_type
    │  hardware_id ──► ttnn_hardware            │ device_series
    │  mesh_config_id ► ttnn_mesh_config        │ card_count
    │                                           │ tt_metal_sha
    │ 1:N                                       │ traced_at
    ▼
ttnn_argument
    arg_index, tensor_dtype, tensor_shape,
    shard_shape, tensor_placement_json,
    full tensor spec in tensor_spec_json

── legacy (kept for backward compat) ──────────────────
ttnn_configuration_model   (config ↔ model junction, no snapshot)
```

### Key tables

**`trace_run`** — one row per `load` invocation.

| Column | Type | Description |
|---|---|---|
| `trace_run_id` | SERIAL PK | Auto-incrementing, used as `trace_id` in manifest |
| `model_id` | FK → `ttnn_model` | Model entry (may be a combined/multi-model entry) |
| `board_type` | TEXT | e.g., `Wormhole`, `Blackhole` |
| `device_series` | TEXT | e.g., `n300`, `tt-galaxy-wh`, `p150b` |
| `card_count` | INTEGER | Number of cards |
| `tt_metal_sha` | TEXT | Git SHA at trace time |
| `traced_at` | TIMESTAMPTZ | When loaded |
| `config_count` | INTEGER | Number of unique configs in this trace |
| `notes` | TEXT | Free text |

**`trace_run_config`** — links configs to traces (snapshot).

| Column | Type | Description |
|---|---|---|
| `trace_run_id` | FK → `trace_run` | |
| `configuration_id` | FK → `ttnn_configuration` | |
| `execution_count` | INTEGER | How many times this op fired during the trace |

**`ttnn_configuration`** — deduplicated by `config_hash`.

| Column | Description |
|---|---|
| `config_hash` | SHA-256 of (operation + arguments + hardware + mesh). Config identity. |
| `full_config_json` | Complete original config. Source of truth for reconstruction. |
| `primary_*` | Denormalized fields (dtype, shape, layout) for Superset queries. |

### Config identity

Two invocations produce the same `config_hash` when: same operation, same arguments (including per-tensor placement), same hardware, same mesh shape. Pre-existing configs are linked to new traces without duplication.

---

## JSON Format Reference

The tracer and reconstructor both produce this format:

```json
{
  "operations": {
    "ttnn::add": {
      "configurations": [
        {
          "config_hash": "abc123...",
          "arguments": {
            "arg0": {
              "type": "ttnn.Tensor",
              "original_shape": [1, 1, 32, 128],
              "original_dtype": "DataType.BFLOAT16",
              "layout": "Layout.TILE",
              "storage_type": "StorageType.DEVICE",
              "memory_config": {
                "memory_layout": "TensorMemoryLayout.INTERLEAVED",
                "buffer_type": "BufferType.DRAM",
                "shard_spec": null
              },
              "tensor_placement": {
                "placement": "['PlacementShard(2)']",
                "distribution_shape": "[1, 2]",
                "mesh_device_shape": "[1, 2]"
              }
            },
            "arg1": { "type": "int", "value": -1 }
          },
          "executions": [
            {
              "source": "models/demos/deepseek_v3/demo/demo.py",
              "machine_info": {
                "board_type": "Wormhole",
                "device_series": "tt-galaxy-wh",
                "card_count": 32,
                "mesh_device_shape": [4, 8]
              },
              "count": 128
            }
          ]
        }
      ]
    }
  },
  "metadata": {
    "models": ["models/demos/deepseek_v3/demo/demo.py"],
    "unique_operations": 27,
    "total_configurations": 323,
    "trace_run_ids": [35]
  }
}
```

**`tensor_placement`** is per-tensor (V2 format). Different placement = different `config_hash`.
**`full_config_json`** in the DB stores this entire config object for lossless reconstruction.
**`config_hash`** in the JSON equals `input_hash` in sweep test results — enabling JOIN with performance history.

### Common pitfalls

| Problem | Wrong | Right |
|---|---|---|
| Arguments as array | `"arguments": [{"arg0": ...}]` | `"arguments": {"arg0": ...}` |
| Global placement | `"machine_info": {"tensor_placements": [...]}` | `"arg0": {"tensor_placement": {...}}` |
| Source missing path | `"source": "meta-llama/Llama-3.2-1B"` | `"source": "models/.../demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]"` |

---

## CLI Reference

All commands: `python tests/sweep_framework/load_ttnn_ops_data_v2.py <command>`

| Command | Description |
|---|---|
| `load [json] [sha]` | Load JSON into DB. Creates trace_run, appends draft to manifest. |
| `reconstruct-manifest [manifest] [output] [scope]` | Resolve targets → reconstruct → merge. `scope`: `lead_models` or `model_traced` (default: all) |
| `resolve-manifest [manifest] [scope]` | Dry-run: print which trace IDs would be used |
| `reconstruct-trace <id> [output]` | Reconstruct JSON from one specific trace_run |
| `list-traces [filter]` | List all trace_runs in DB |
| `reconstruct [output] [schema] [models]` | Reconstruct from DB filtered by model patterns (legacy) |
| `reconstruct-lead [output] [schema]` | Reconstruct lead models only (legacy) |
| `reconstruct-op <name> [output]` | Reconstruct a single operation |
| `verify [original] [reconstructed]` | Compare two JSON files |
| `duplicates [json] [op]` | Detect duplicate configs in a JSON file |
| `find-lines <op> <i1,i2,...>` | Find line numbers for config indices in a JSON file |

---

## File Locations

| File | Purpose |
|---|---|
| `model_tracer/generic_ops_tracer.py` | Trace models, produce master JSON |
| `model_tracer/sweep_manifest.yaml` | Targets + trace registry (source of truth for CI) |
| `model_tracer/migrate_v2_5_add_trace_runs.sql` | Additive DB migration — adds `trace_run` + `trace_run_config` to `ttnn_ops_v2_5` |
| `model_tracer/cleanup_duplicate_trace_runs.sql` | One-time cleanup: collapses duplicate migrated traces |
| `model_tracer/consolidate_n300_traces.sql` | One-time: merged all n300 traces into a single trace (run once) |
| `model_tracer/destructively_create_ttnn_ops_schema_v2.sql` | Original `ttnn_ops_v2_5` schema DDL (reference) |
| `tests/sweep_framework/load_ttnn_ops_data_v2.py` | Load, reconstruct, manifest resolution |
| `tests/sweep_framework/framework/constants.py` | `LEAD_MODELS` — derived from manifest at import time |
| `tests/sweep_framework/master_config_loader_v2.py` | Loads master JSON for sweep vector generation |
| `tests/sweep_framework/sweeps_parameter_generator.py` | Generates sweep test vectors from reconstructed JSON |
| `tests/sweep_framework/sweeps_runner.py` | Runs sweep tests |
