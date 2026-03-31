# Model Tracer Guide

The model tracer extracts operation configuration data while running models and stores them in a PostgreSQL database. From there, configurations can be reconstructed into JSON and used as sweep test vectors — in CI, on a branch, or in nightly runs.

**Schema:** All data lives in `ttnn_ops_v6` on Metal Ops PostgreSQL (configurable via `--schema`).
**Connection:** Set `TTNN_OPS_DATABASE_URL`

---

## Quick Start

```bash
# 1. Trace a model
python model_tracer/generic_ops_tracer.py models/demos/deepseek_v3/demo/demo.py

# 2. Load into DB (creates trace_run, auto-appends draft to manifest)
python tests/sweep_framework/load_ttnn_ops_data_v2.py load \
    model_tracer/traced_operations/ttnn_operations_master.json

# 3. Promote the trace in the manifest (edit model_tracer/sweep_manifest.yaml)
#    Change status: draft → active, add to the appropriate targets group

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

# Dry run: preview what would be loaded without committing anything
python tests/sweep_framework/load_ttnn_ops_data_v2.py load path/to/master.json --dry-run

# Load into a different schema (default: ttnn_ops_v6)
python tests/sweep_framework/load_ttnn_ops_data_v2.py --schema ttnn_ops_v6 load path/to/master.json
```

On load the tool:
- Deduplicates configs by `config_hash` (same op + args + hardware + mesh = same config)
- Rejects duplicate uploads by `trace_uid` before any data population
- Creates a `trace_run` row capturing `trace_uid`, hardware, and tt-metal SHA
- Stores canonical counts in `trace_run_configuration_model`
- Refreshes derived aggregates in `trace_run_config`, `ttnn_configuration_model`, and `trace_run_model`
- **Auto-appends a `draft` entry** to `model_tracer/sweep_manifest.yaml`

Example output:
```
✅ Loaded 623 configurations (47 new, 576 pre-existing), 623 model links
Trace run created: trace_run_id=42
DB totals after load: 8012 configs, 542 trace_run_config links, 23 models
Appended 1 draft entries to manifest registry (model_tracer/sweep_manifest.yaml)
```

With `--dry-run`, the transaction is rolled back and the summary shows what the DB would look like after commit. No manifest entry is written.

### Step 3: Model names (collisions)

Each model gets a short, lowercase `model_name` derived from its source path or HF identifier when it is first inserted. If the loader reports a `model_name` uniqueness error, disambiguate with:

```bash
# Override a specific model's name
python tests/sweep_framework/load_ttnn_ops_data_v2.py set-model-name \
    --source-file "path/to/demo.py" --model-name custom_name
# or by model ID:
python tests/sweep_framework/load_ttnn_ops_data_v2.py set-model-name \
    --model-id 7 --model-name vit_nightly
```

**Derivation rules:**
- HF model: lowercase last segment after `/` → `meta-llama/Llama-3.2-1B-Instruct` → `llama-3.2-1b-instruct`
- File path: skip generic segments (`models`, `demos`, `experimental`, `wormhole`, `vision`, `demo`, etc.), take first meaningful segment → `models/demos/audio/whisper/demo/demo.py` → `whisper`

`model_name` is UNIQUE in the DB. If two sources derive to the same name, the loader auto-disambiguates by appending `_2`, `_3`, etc. Use `set-model-name` to rename afterwards if the auto-generated suffix isn't ideal.

### Step 4: Promote in the manifest

The loader appends the trace as `status: draft`. Review and promote it to `active`, then add it to the appropriate `targets` group with the exact `model_name` values:

```yaml
# model_tracer/sweep_manifest.yaml
targets:
  lead_models:
    - model: deepseek_v3
      trace: 35

  model_traced:
    - model: [whisper, llama-3.2-1b-instruct, ...]
      trace: [538, 1]

registry:
  - trace_id: 42
    status: active        # ← change from draft
    models: [deepseek_v3]
    hardware: {board_type: Wormhole, device_series: tt-galaxy-wh, card_count: 32}
    trace_uid: 8c4f2c2f-5d9e-49ae-8c25-8e7c2c2f0abc
    tt_metal_sha: abc123def456
    config_count: 323
    loaded_at: '2026-03-21'
    notes: 'Post-matmul refactor re-trace'
```

Traces left as `draft` are invisible to model resolution. They can still be used by pinning `trace: N` directly in a target.

### Step 5: Reconstruct configs for testing

```bash
# Reconstruct all targets into the default path (auto-discovered by MasterConfigLoader)
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest \
    model_tracer/sweep_manifest.yaml \
    model_tracer/traced_operations/ttnn_operations_master.json

# Reconstruct a specific scope (model_traced or lead_models)
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest \
    model_tracer/sweep_manifest.yaml \
    model_tracer/traced_operations/ttnn_operations_master.json \
    model_traced

# Dry-run: see which trace IDs will be used without hitting the DB
python tests/sweep_framework/load_ttnn_ops_data_v2.py resolve-manifest \
    model_tracer/sweep_manifest.yaml model_traced

# Reconstruct a single specific trace (all models in that trace)
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-trace 35 output.json

# Reconstruct from a different schema
python tests/sweep_framework/load_ttnn_ops_data_v2.py --schema ttnn_ops_v6 reconstruct-trace 35 output.json
```

**Path resolution:** `MasterConfigLoader` looks for the master JSON in this order:
1. `TTNN_MASTER_JSON_PATH` env var (if set and file exists)
2. `model_tracer/traced_operations/ttnn_operations_master.json` (default)

In CI, each run type sets `TTNN_MASTER_JSON_PATH` to point at its scoped artifact (e.g., `*_lead_models.json` or `*_model_traced.json`). For local development, writing to the default path means the sweep framework picks it up automatically.

**Important:** `reconstruct-manifest` only includes configs for the models listed in each target entry's `model:` field. If a trace contains 20 models but the target lists only `[whisper]`, only whisper's configs are included in the output.

The reconstructed JSON is in the same format as the tracer output and can be fed directly to `sweeps_parameter_generator.py`.

---

## Sweep Manifest

File: `model_tracer/sweep_manifest.yaml`

The manifest controls which configs are included when reconstructing. It has two sections: **targets** (what to reconstruct) and **registry** (log of all known traces).

### Targets

```yaml
targets:
  lead_models:
    - model: deepseek_v3
      trace: 35

  model_traced:
    - model: [whisper, llama-3.2-1b-instruct, phi-3-mini-128k-instruct]
      trace: [538, 1]
```

Two scope groups: `lead_models` and `model_traced`. Each group is a list of entries. Each entry specifies:

| Field | Description |
|---|---|
| `model: X` | Exact `model_name` from DB. Can be a string or list. Used for both registry resolution and config filtering. |
| `trace: N` or `trace: [N, M]` | Pinned trace ID(s). Skips registry resolution. Configs are still filtered to the listed models. |
| `hardware: H` | Filter registry matches to `device_series == H` (optional, only relevant when `trace` is omitted). |

**Resolution rules:**

| Form | Resolution |
|---|---|
| `model: X` (no trace) | Latest `active` registry entry per unique `device_series` where `X` is in the `models` list |
| `model: X, hardware: H` | Latest `active` entry where `device_series == H` and `X` is in `models` |
| `model: X, trace: N` | Trace N exactly. No registry lookup. Config filtering to model X still applies. |
| `trace: N` (no model) | Trace N directly. All configs in the trace are included. |

- `model_name` matching is **exact** (not substring) against the registry `models` list.
- "Latest" = highest `trace_id` among matching active entries per `device_series`.
- Multiple entries resolving to the same `trace_id` are deduplicated.
- Configs appearing in multiple traces are deduplicated by `config_hash`.

### Registry

Append-only log. Auto-updated on each `load`. Status is manually curated.

```yaml
registry:
  - trace_id: 538
    status: active              # active | draft | deprecated
    models:
      - deepseek-llm-7b-chat
      - llama-3.2-1b-instruct
      - whisper
      # ... one entry per model_name in ttnn_ops_v6.ttnn_model
    hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
    tt_metal_sha: null
    config_count: 7030
    loaded_at: '2026-03-21'
    notes: 'Consolidated n300 trace — all models (v2.5 migration)'
```

| Field | Description |
|---|---|
| `trace_id` | Maps to `trace_run_id` in `ttnn_ops_v6.trace_run` |
| `trace_uid` | Stable unique ID of the uploaded trace artifact. Duplicate uploads are rejected by this field. |
| `status` | `active`: included in model resolution. `draft`: auto-appended on load, not yet reviewed. `deprecated`: excluded, kept for history. |
| `models` | Exact `model_name` values from `ttnn_ops_v6.ttnn_model` for this trace. Used for exact matching against target `model:` entries. |
| `hardware` | `board_type`, `device_series`, `card_count` |
| `tt_metal_sha` | Git SHA of tt-metal at trace time (`null` for migrated historical traces) |
| `config_count` | Number of configs linked to this trace |
| `loaded_at` | Date the trace was loaded |
| `notes` | Free text |

### Current traces

| trace_id | hardware | models | configs | status |
|---|---|---|---|---|
| 1 | p150b (Blackhole) | whisper | 318 | active |
| 35 | tt-galaxy-wh (Wormhole, 32 cards) | deepseek_v3 | 323 | active |
| 538 | n300 (Wormhole) | 19 models (all tt_transformers + vision + audio) | 7,030 | active |

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

**To add a model to a CI run:**
1. Trace the model and load into DB
2. Promote the registry entry to `active`
3. Add the `model_name` to the appropriate `targets` group in the manifest

---

## Database Schema

All tables live in the `ttnn_ops_v6` schema by default. All CLI commands accept `--schema <name>` to target a different schema. The schema must already exist in the database; create it from the DDL template in `model_tracer/destructively_create_ttnn_ops_schema_v6.sql`.

```
ttnn_operation          ttnn_model ◄─── trace_run_model
    │ 1:N                   │ 1:N              │ N:1
    ▼                       ▼                  ▼
ttnn_configuration ◄── ttnn_configuration_model    trace_run
    │  config_hash               (derived)          │ trace_uid
    │  full_config_json                             │ hardware_id ──► ttnn_hardware
    │  hardware_id ──► ttnn_hardware               │ tt_metal_sha
    │  mesh_config_id ► ttnn_mesh_config           │ config_count
    │                                              │ traced_at
    ▼
trace_run_configuration_model (canonical)
    trace_run_id, configuration_id, model_id, execution_count
    ▼
trace_run_config (derived)
    trace_run_id, configuration_id, execution_count
```

### Key tables

**`trace_run`** — one row per uploaded trace artifact.

| Column | Type | Description |
|---|---|---|
| `trace_run_id` | SERIAL PK | Auto-incrementing, used as `trace_id` in manifest |
| `trace_uid` | TEXT UNIQUE | Stable artifact identity used to reject duplicate uploads |
| `hardware_id` | FK → `ttnn_hardware` | Board type, device series, card count |
| `tt_metal_sha` | TEXT | Git SHA at trace time (null for historical traces) |
| `traced_at` | TIMESTAMPTZ | When loaded |
| `config_count` | INTEGER | Number of unique configs in this trace |
| `notes` | TEXT | Free text |

**`trace_run_model`** — which models are in a trace (materialized junction).

| Column | Type | Description |
|---|---|---|
| `trace_run_id` | FK → `trace_run` | |
| `model_id` | FK → `ttnn_model` | Real model, no synthetic aggregates |

**`ttnn_model`** — one row per real model source.

| Column | Type | Description |
|---|---|---|
| `model_name` | TEXT UNIQUE | Short lowercase identifier. Used in manifest. |
| `source_file` | TEXT | Path to the demo/test file |
| `hf_model_identifier` | TEXT | HF model ID if applicable (e.g. `meta-llama/Llama-3.2-1B-Instruct`) |
| `model_family` | TEXT | Inferred family (llama, whisper, etc.) |

**`trace_run_configuration_model`** — canonical per-trace per-config per-model counts.

| Column | Type | Description |
|---|---|---|
| `trace_run_id` | FK → `trace_run` | |
| `configuration_id` | FK → `ttnn_configuration` | |
| `model_id` | FK → `ttnn_model` | |
| `execution_count` | INTEGER | How many times this config fired for that model in that trace |

**`trace_run_config`** — derived per-trace per-config aggregate.

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
| `hardware_id` | FK → `ttnn_hardware` |
| `mesh_config_id` | FK → `ttnn_mesh_config` |

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
                "mesh_device_shape": [4, 8],
                "device_count": 32
              },
              "count": 128,
              "trace_run_ids": [35]
            }
          ]
        }
      ]
    }
  },
  "metadata": {
    "models": ["models/demos/deepseek_v3/demo/demo.py"],
    "unique_operations": 27,
    "total_configurations": 323
  }
}
```

**`tensor_placement`** is per-tensor in arguments. Different placement = different `config_hash`.
**`full_config_json`** in the DB stores this entire config object for lossless reconstruction.
**`config_hash`** in the JSON equals `input_hash` in sweep test results — enabling JOIN with performance history.

### Required fields for loading

The loader validates that every execution has the following fields. Missing any of them raises a `ValueError` naming the operation, config_hash, and missing fields.

| Field | Location | Description |
|---|---|---|
| `config_hash` | `config.config_hash` | SHA-256 identity of the configuration |
| `source` | `execution.source` | File path, optionally with `[HF_MODEL:...]` suffix |
| `trace_uid` | `execution.trace_uid` | Stable trace identity persisted by `generic_ops_tracer.py` |
| `board_type` | `execution.machine_info.board_type` | Hardware board type (e.g. `"Wormhole"`, `"Blackhole"`) |
| `device_series` | `execution.machine_info.device_series` | Device series (e.g. `"n300"`, `"tt-galaxy-wh"`, `"p150b"`) |
| `card_count` | `execution.machine_info.card_count` | Number of cards (e.g. `1`, `32`) |
| `mesh_device_shape` | `execution.machine_info.mesh_device_shape` | Mesh topology as an array (e.g. `[1, 1]`, `[4, 8]`) |
| `device_count` | `execution.machine_info.device_count` | Total device count (e.g. `1`, `32`) |

### Common pitfalls

| Problem | Wrong | Right |
|---|---|---|
| Arguments as array | `"arguments": [{"arg0": ...}]` | `"arguments": {"arg0": ...}` |
| Missing mesh/device info | `"machine_info": {"board_type": "Wormhole"}` | Include `mesh_device_shape` and `device_count` |
| Source missing path | `"source": "meta-llama/Llama-3.2-1B"` | `"source": "models/.../demo.py [HF_MODEL:meta-llama/Llama-3.2-1B]"` |
| Mesh shape in wrong place | `"tensor_placements": [{"mesh_device_shape": ...}]` | `"machine_info": {"mesh_device_shape": [4, 8]}` |

---

## CLI Reference

All commands: `python tests/sweep_framework/load_ttnn_ops_data_v2.py [--schema <name>] <command>`

The global `--schema <name>` flag can appear anywhere on the command line and applies to all commands. Default: `ttnn_ops_v6`.

| Command | Description |
|---|---|
| `load [json] [sha] [--dry-run]` | Load JSON into DB. Creates trace_run, appends draft to manifest. `--dry-run` previews without committing. |
| `set-model-name --source-file P --model-name N` | Override a model's name. Also accepts `--hf-model` or `--model-id`. |
| `delete-trace <id> [--yes]` | Delete a trace_run and any configs tied exclusively to it. Configs shared with other traces are kept. |
| `reconstruct-manifest [manifest] [output] [scope]` | Resolve targets → reconstruct (filtered to target models) → merge. `scope`: `lead_models` or `model_traced` (default: all) |
| `resolve-manifest [manifest] [scope]` | Dry-run: print which trace IDs and model filters would be used |
| `reconstruct-trace <id> [output]` | Reconstruct JSON from one specific trace_run (all models) |
| `list-traces [filter]` | List all trace_runs in DB |
| `reconstruct [output] [models]` | Reconstruct from DB filtered by model patterns (legacy) |
| `reconstruct-op <name> [output]` | Reconstruct a single operation |
| `verify [original] [reconstructed]` | Compare two JSON files |
| `find-lines <op> <i1,i2,...>` | Find line numbers for config indices in a JSON file |

---

## File Locations

| File | Purpose |
|---|---|
| `model_tracer/generic_ops_tracer.py` | Trace models, produce master JSON |
| `model_tracer/sweep_manifest.yaml` | Targets + trace registry (source of truth for CI) |
| `model_tracer/destructively_create_ttnn_ops_schema_v6.sql` | `ttnn_ops_v6` schema DDL |
| `tests/sweep_framework/load_ttnn_ops_data_v2.py` | Load, reconstruct, manifest resolution |
| `tests/sweep_framework/framework/constants.py` | `LEAD_MODELS` — derived from manifest at import time |
| `tests/sweep_framework/master_config_loader_v2.py` | Loads master JSON for sweep vector generation |
| `tests/sweep_framework/sweeps_parameter_generator.py` | Generates sweep test vectors from reconstructed JSON |
| `tests/sweep_framework/sweeps_runner.py` | Runs sweep tests |
