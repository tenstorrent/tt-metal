#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Load ttnn_operations_master.json into Neon.tech PostgreSQL.

Schema design:
- Config identity = operation + arguments + hardware + mesh
- Each (hardware, mesh) combination from machine_info creates a separate config
- All sources are linked to each config via ttnn_configuration_model junction table
- mesh_config_id is a direct FK on ttnn_configuration (not a junction table)
"""

import json
import os
import re
from datetime import date
from pathlib import Path

import psycopg2

try:
    import yaml
except ImportError:
    yaml = None

# Default manifest path (relative to repo root)
_DEFAULT_MANIFEST = "model_tracer/sweep_manifest.yaml"


def _get_manifest_path(manifest_path=None):
    """Resolve manifest path, defaulting to repo-relative location."""
    if manifest_path:
        return manifest_path
    # Try relative to cwd first, then relative to this file's repo root
    if os.path.exists(_DEFAULT_MANIFEST):
        return _DEFAULT_MANIFEST
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # tests/sweep_framework -> repo root
    candidate = repo_root / _DEFAULT_MANIFEST
    if candidate.exists():
        return str(candidate)
    return _DEFAULT_MANIFEST


def _require_yaml():
    if yaml is None:
        raise ImportError("PyYAML is required for manifest operations. " "Install it with: pip install pyyaml")


def _load_manifest(manifest_path=None):
    """Load the manifest YAML. Returns dict with 'targets' and 'registry' keys."""
    _require_yaml()
    path = _get_manifest_path(manifest_path)
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("targets", {})
    data.setdefault("registry", [])
    return data, path


def _append_registry_entries(entries, path):
    """Append new registry entries to the manifest without rewriting the file.

    Preserves existing comments and formatting by appending raw YAML text.
    """
    _require_yaml()
    with open(path, "a") as f:
        f.write("\n")
        for entry in entries:
            f.write(f"  - trace_id: {entry['trace_id']}\n")
            f.write(f"    status: {entry['status']}\n")
            models = entry.get("models", [])
            if len(models) == 1:
                f.write(f"    models: [{models[0]}]\n")
            else:
                f.write("    models:\n")
                for m in models:
                    f.write(f"      - {m}\n")
            hw = entry.get("hardware", {})
            hw_parts = [
                f"board_type: {hw.get('board_type', '?')}",
                f"device_series: {hw.get('device_series', '?')}",
                f"card_count: {hw.get('card_count', 1)}",
            ]
            f.write("    hardware: {" + ", ".join(hw_parts) + "}\n")
            f.write(f"    tt_metal_sha: {entry.get('tt_metal_sha') or 'null'}\n")
            f.write(f"    config_count: {entry.get('config_count', 0)}\n")
            f.write(f"    loaded_at: '{entry.get('loaded_at', '')}'\n")
            f.write(f"    notes: '{entry.get('notes', '')}'\n")
            f.write("\n")


# Connection string from environment (supports both CI and local env var names)
NEON_URL = os.environ.get("TTNN_OPS_DATABASE_URL") or os.environ.get("NEON_CONNECTION_STRING")
if not NEON_URL:
    raise ValueError(
        "Database connection string not found. Please set either "
        "TTNN_OPS_DATABASE_URL or NEON_CONNECTION_STRING environment variable."
    )
JSON_PATH = "model_tracer/traced_operations/ttnn_operations_master.json"


def parse_source(source_str):
    """Parse a single source string into (source_file, hf_model_identifier)."""
    if not source_str:
        return None, None

    # Pattern: "path/to/file.py [HF_MODEL:org/model-name]"
    match = re.match(r"(.+?)\s*\[HF_MODEL:(.+?)\]", source_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return source_str, None


def parse_all_sources(source):
    """Parse source field (string or array) into list of (source_file, hf_model) tuples."""
    if not source:
        return [(None, None)]

    # Normalize to list
    if isinstance(source, str):
        sources = [source]
    elif isinstance(source, list):
        sources = source
    else:
        return [(None, None)]

    result = []
    for s in sources:
        source_file, hf_model = parse_source(s)
        result.append((source_file, hf_model))

    return result if result else [(None, None)]


def extract_model_family(source_file, hf_model):
    """Infer model family from source or HF identifier."""
    combined = f"{source_file or ''} {hf_model or ''}".lower()
    families = ["llama", "qwen", "deepseek", "mistral", "whisper", "efficientnet", "resnet", "bert"]
    for family in families:
        if family in combined:
            return family
    return None


# Path segments that are too generic to identify a model.
_GENERIC_PATH_SEGMENTS = frozenset(
    {
        "models",
        "tests",
        "demos",
        "experimental",
        "wormhole",
        "vision",
        "classification",
        "audio",
        "nightly",
        "single_card",
        "demo",
        "pcc",
        "segmentation_evaluation",
    }
)


def derive_model_name(source_file, hf_model):
    """Derive a short, lowercase model name for use in manifests and the DB.

    Rules:
    - HF model  : lowercase last segment after '/'
      e.g. 'meta-llama/Llama-3.2-1B-Instruct' -> 'llama-3.2-1b-instruct'
    - Non-HF    : strip test-node suffix, walk path segments, skip generic
                  folder/file names, return the first meaningful segment.
      e.g. 'models/demos/audio/whisper/demo/demo.py' -> 'whisper'
    """
    if hf_model:
        return hf_model.split("/")[-1].lower()
    if not source_file:
        return None

    # Strip test-node suffix  e.g. "path/file.py::test_foo[bar]"
    path = re.sub(r"::.*$", "", source_file)
    segments = path.split("/")

    processed = []
    for i, seg in enumerate(segments):
        # Strip file extension from last segment (the filename)
        if i == len(segments) - 1:
            seg = re.sub(r"\.(py|json)$", "", seg)
        # Strip leading 'test_' prefix that test filenames carry
        seg = re.sub(r"^test_", "", seg).lower()
        processed.append(seg)

    # First non-generic, non-empty segment is the model name
    for seg in processed:
        if seg and seg not in _GENERIC_PATH_SEGMENTS:
            return seg

    # Fallback: last non-empty segment
    for seg in reversed(processed):
        if seg:
            return seg

    return None


def parse_array_value(value):
    """Convert string array representation to a proper list for PostgreSQL."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value == "std::nullopt" or value == "nullopt" or value == "null":
            return None
        try:
            parsed = json.loads(value.replace("{", "[").replace("}", "]"))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, fall back to regex-based numeric extraction below.
            pass
        numbers = re.findall(r"-?\d+", value)
        if numbers:
            return [int(n) for n in numbers]
    return None


def parse_placement(placement_str):
    """Parse placement string like '[PlacementShard(3)]' or '[PlacementReplicate]'.

    Returns: (placement_type, shard_dim)
    """
    if not placement_str:
        return "replicate", None

    shard_match = re.search(r"PlacementShard\((\d+)\)", placement_str)
    if shard_match:
        return "shard", int(shard_match.group(1))

    return "replicate", None


def get_or_create_hardware(cur, hardware_cache, board_type, device_series, card_count):
    """Get or create a hardware entry, return (hardware_id, hw_key)."""
    # Normalize device_series (can be string or array)
    if isinstance(device_series, list):
        device_series = device_series[0] if device_series else None

    if not board_type:
        return None, None

    hw_key = (board_type, device_series, card_count)
    if hw_key not in hardware_cache:
        cur.execute(
            """
            INSERT INTO ttnn_ops_v5.ttnn_hardware (board_type, device_series, card_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (board_type, device_series, card_count) DO NOTHING
            RETURNING ttnn_hardware_id
        """,
            hw_key,
        )
        result = cur.fetchone()
        if result:
            hardware_cache[hw_key] = result[0]
        else:
            cur.execute(
                "SELECT ttnn_hardware_id FROM ttnn_ops_v5.ttnn_hardware WHERE board_type=%s AND device_series=%s AND card_count=%s",
                hw_key,
            )
            hardware_cache[hw_key] = cur.fetchone()[0]

    return hardware_cache.get(hw_key), hw_key


def get_or_create_mesh_config(cur, mesh_config_cache, mesh_shape, device_count):
    """Get or create a mesh config entry, return the ID.

    ttnn_mesh_config only stores mesh_shape and device_count.
    Placement info is per-tensor and stored in the argument JSON.
    """
    if not mesh_shape:
        return None

    mesh_key = (tuple(mesh_shape), device_count)

    if mesh_key not in mesh_config_cache:
        cur.execute(
            """
            INSERT INTO ttnn_ops_v5.ttnn_mesh_config (mesh_shape, device_count)
            VALUES (%s, %s)
            ON CONFLICT (mesh_shape, device_count) DO NOTHING
            RETURNING ttnn_mesh_config_id
        """,
            (mesh_shape, device_count),
        )
        result = cur.fetchone()
        if result:
            mesh_config_cache[mesh_key] = result[0]
        else:
            cur.execute(
                """
                SELECT ttnn_mesh_config_id FROM ttnn_ops_v5.ttnn_mesh_config
                WHERE mesh_shape = %s AND device_count = %s
            """,
                (mesh_shape, device_count),
            )
            result = cur.fetchone()
            if result:
                mesh_config_cache[mesh_key] = result[0]

    return mesh_config_cache.get(mesh_key)


def _disambiguate_model_name(cur, base_name):
    """Find the next available model_name by appending _2, _3, etc."""
    cur.execute(
        "SELECT model_name FROM ttnn_ops_v5.ttnn_model WHERE model_name LIKE %s",
        (f"{base_name}%",),
    )
    existing = {row[0] for row in cur.fetchall()}
    suffix = 2
    while f"{base_name}_{suffix}" in existing:
        suffix += 1
    return f"{base_name}_{suffix}"


def get_or_create_model(cur, model_cache, source_file, hf_model):
    """Get or create a model entry, return the ID."""
    if not source_file:
        return None

    model_key = (source_file, hf_model)
    if model_key not in model_cache:
        model_family = extract_model_family(source_file, hf_model)
        model_name = derive_model_name(source_file, hf_model)
        # Look up existing row first. ON CONFLICT doesn't work for NULL hf_model_identifier
        # because NULL != NULL in PostgreSQL UNIQUE constraints.
        cur.execute(
            """
            SELECT ttnn_model_id FROM ttnn_ops_v5.ttnn_model
            WHERE source_file = %s
              AND (hf_model_identifier = %s OR (hf_model_identifier IS NULL AND %s IS NULL))
            """,
            (source_file, hf_model, hf_model),
        )
        row = cur.fetchone()
        if row:
            # Only set model_name when it's NULL to preserve set-model-name overrides.
            cur.execute(
                "UPDATE ttnn_ops_v5.ttnn_model SET model_name = COALESCE(model_name, %s), update_ts = NOW() WHERE ttnn_model_id = %s",
                (model_name, row[0]),
            )
            model_cache[model_key] = row[0]
        else:
            try:
                cur.execute("SAVEPOINT model_insert")
                cur.execute(
                    """
                    INSERT INTO ttnn_ops_v5.ttnn_model
                        (source_file, hf_model_identifier, model_family, model_name)
                    VALUES (%s, %s, %s, %s)
                    RETURNING ttnn_model_id
                    """,
                    (source_file, hf_model, model_family, model_name),
                )
                model_cache[model_key] = cur.fetchone()[0]
                cur.execute("RELEASE SAVEPOINT model_insert")
            except Exception as e:
                err = str(e).lower()
                if "ttnn_model_name_unique" in err or ("unique" in err and "model_name" in err):
                    # Name collision — auto-disambiguate with incremental suffix
                    # Use ROLLBACK TO SAVEPOINT to clear only the failed statement
                    cur.execute("ROLLBACK TO SAVEPOINT model_insert")
                    disambiguated = _disambiguate_model_name(cur, model_name)
                    print(
                        f"  Warning: model_name '{model_name}' already taken, "
                        f"using '{disambiguated}' instead. "
                        f'Rename later with: set-model-name --source-file "{source_file}" --model-name <name>'
                    )
                    cur.execute(
                        """
                        INSERT INTO ttnn_ops_v5.ttnn_model
                            (source_file, hf_model_identifier, model_family, model_name)
                        VALUES (%s, %s, %s, %s)
                        RETURNING ttnn_model_id
                        """,
                        (source_file, hf_model, model_family, disambiguated),
                    )
                    model_cache[model_key] = cur.fetchone()[0]
                else:
                    raise

    return model_cache[model_key]


def parse_mesh_from_machine_info(machine_info, arguments=None):
    """Extract mesh configuration from tensor arguments (V2 format).

    In V2 format, tensor_placement is stored in the tensor arguments themselves,
    not in machine_info. This function extracts mesh config from the first tensor argument.

    Returns: (mesh_shape, device_count, placement_type, shard_dim, distribution_shape)
    """
    placement = None

    # V2 format: Extract tensor_placement from first tensor argument
    if arguments:
        for arg_value in arguments.values():
            if isinstance(arg_value, dict) and arg_value.get("type") == "ttnn.Tensor":
                if "tensor_placement" in arg_value:
                    placement = arg_value["tensor_placement"]
                    break

    if not placement:
        # No tensor_placement = no specific mesh config
        return None, None, None, None, None

    mesh_shape_str = placement.get("mesh_device_shape")
    mesh_shape = parse_array_value(mesh_shape_str)

    if not mesh_shape:
        return None, None, None, None, None

    # Calculate device count
    device_count = machine_info.get("device_count")
    if device_count is None:
        device_count = 1
        for dim in mesh_shape:
            device_count *= dim

    # Parse placement type
    placement_str = placement.get("placement")
    placement_type, shard_dim = parse_placement(placement_str)

    # Parse distribution shape
    distribution_shape = parse_array_value(placement.get("distribution_shape"))

    return mesh_shape, device_count, placement_type, shard_dim, distribution_shape


def get_or_create_trace_run(cur, trace_run_cache, hardware_id, tt_metal_sha=None):
    """Get or create a trace_run for this hardware + SHA combination.

    v5: no model_id on trace_run — models are tracked via trace_run_model.
    """
    tr_key = (hardware_id, tt_metal_sha)
    if tr_key not in trace_run_cache:
        cur.execute(
            """
            INSERT INTO ttnn_ops_v5.trace_run (hardware_id, tt_metal_sha)
            VALUES (%s, %s)
            RETURNING trace_run_id
            """,
            (hardware_id, tt_metal_sha),
        )
        trace_run_cache[tr_key] = cur.fetchone()[0]

    return trace_run_cache[tr_key]


def link_trace_run_config(cur, trace_run_id, config_id, execution_count):
    """Link a configuration to a trace_run via trace_run_config."""
    cur.execute(
        """
        INSERT INTO ttnn_ops_v5.trace_run_config (trace_run_id, configuration_id, execution_count)
        VALUES (%s, %s, %s)
        ON CONFLICT (trace_run_id, configuration_id) DO UPDATE
        SET execution_count = ttnn_ops_v5.trace_run_config.execution_count + EXCLUDED.execution_count
        """,
        (trace_run_id, config_id, execution_count),
    )


def load_data(json_path=None, tt_metal_sha=None, dry_run=False):
    """Main loading function.

    Args:
        json_path: Path to JSON file (defaults to JSON_PATH).
        tt_metal_sha: Git SHA of tt-metal at trace time (optional).
                      If None, attempts to detect from git.
        dry_run: If True, run all logic but roll back every DB write at the end.
                 Prints the same stats so you can verify behaviour without persisting anything.
    """
    json_path = json_path or JSON_PATH
    print(f"Loading JSON from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    operations = data.get("operations", {})
    print(f"Found {len(operations)} operations")

    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    # Caches for normalized tables
    operation_cache = {}
    model_cache = {}
    hardware_cache = {}
    mesh_config_cache = {}
    config_cache = {}  # config_hash -> config_id
    trace_run_cache = {}  # (hardware_id, sha) -> trace_run_id
    print("  Populating trace_run, trace_run_config, trace_run_model + ttnn_configuration_model")

    # Auto-detect tt_metal_sha if not provided
    if tt_metal_sha is None:
        try:
            import subprocess

            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tt_metal_sha = result.stdout.strip()
                print(f"  Detected tt-metal SHA: {tt_metal_sha[:12]}")
        except Exception as e:
            # Best-effort SHA auto-detection: ignore failures but surface for debugging.
            print(f"  Warning: failed to auto-detect tt-metal SHA via git: {e}")

    new_configs = 0  # configs inserted for the first time this load
    total_model_links = 0
    total_trace_run_links = 0  # all configs linked to trace_runs (new + pre-existing)

    for op_name, op_data in operations.items():
        # Insert operation
        if op_name not in operation_cache:
            cur.execute(
                """
                INSERT INTO ttnn_ops_v5.ttnn_operation (operation_name)
                VALUES (%s)
                ON CONFLICT (operation_name) DO NOTHING
            """,
                (op_name,),
            )
            cur.execute(
                "SELECT ttnn_operation_id FROM ttnn_ops_v5.ttnn_operation WHERE operation_name = %s",
                (op_name,),
            )
            operation_cache[op_name] = cur.fetchone()[0]

        op_id = operation_cache[op_name]

        for config_idx, config in enumerate(op_data.get("configurations", [])):
            arguments = config.get("arguments", {})

            # V2 Format: Use executions array (preferred)
            # V1 Format: Use source and machine_info at config level (legacy)
            executions = config.get("executions", [])

            if not executions:
                # Fallback to V1 format for backward compatibility
                source = config.get("source")
                machine_info_list = config.get("machine_info", [{}])

                # Ensure machine_info_list is a list
                if not machine_info_list:
                    machine_info_list = [{}]

                # Convert V1 format to V2 executions format
                source_tuples = parse_all_sources(source)
                executions = []
                for machine_info in machine_info_list:
                    for source_file, hf_model in source_tuples:
                        source_str = source_file
                        if hf_model:
                            source_str = f"{source_file} [HF_MODEL:{hf_model}]"
                        executions.append(
                            {
                                "source": source_str,
                                "machine_info": machine_info,
                                "count": 1,  # Default count for V1 format
                            }
                        )

            config_hash = config.get("config_hash")
            if not config_hash:
                raise ValueError(
                    f"Missing config_hash in configuration for operation '{op_name}' "
                    f"(config index {config_idx}). "
                    f"Re-trace with generic_ops_tracer.py to produce a JSON with config hashes."
                )

            for execution in executions:
                source = execution.get("source")
                machine_info = execution.get("machine_info", {})
                execution_count = execution.get("count", 1)

                # Parse source to get model info
                source_file, hf_model = parse_source(source)
                model_id = get_or_create_model(cur, model_cache, source_file, hf_model)
                # Parse hardware
                board_type = machine_info.get("board_type")
                device_series = machine_info.get("device_series")
                card_count = machine_info.get("card_count", 1)

                hardware_id, _ = get_or_create_hardware(cur, hardware_cache, board_type, device_series, card_count)

                # Parse mesh config
                args_dict = arguments if isinstance(arguments, dict) else {}
                mesh_shape, device_count, _, _, _ = parse_mesh_from_machine_info(machine_info, args_dict)

                mesh_config_id = None
                if mesh_shape:
                    mesh_config_id = get_or_create_mesh_config(cur, mesh_config_cache, mesh_shape, device_count)

                # Check if we've already created this config
                if config_hash in config_cache:
                    config_id = config_cache[config_hash]
                    # Just update last_seen_ts
                    cur.execute(
                        "UPDATE ttnn_ops_v5.ttnn_configuration SET last_seen_ts = NOW() WHERE ttnn_configuration_id = %s",
                        (config_id,),
                    )
                else:
                    # Insert new configuration
                    try:
                        cur.execute(
                            """
                            INSERT INTO ttnn_ops_v5.ttnn_configuration
                            (operation_id, hardware_id, mesh_config_id, config_hash, full_config_json)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (config_hash) DO UPDATE SET last_seen_ts = NOW()
                            RETURNING ttnn_configuration_id
                            """,
                            (op_id, hardware_id, mesh_config_id, config_hash, json.dumps(config)),
                        )
                        config_id = cur.fetchone()[0]
                        config_cache[config_hash] = config_id
                        new_configs += 1

                    except psycopg2.errors.UniqueViolation:
                        conn.rollback()
                        # Config already exists, fetch its ID
                        cur.execute(
                            "SELECT ttnn_configuration_id FROM ttnn_ops_v5.ttnn_configuration WHERE config_hash = %s",
                            (config_hash,),
                        )
                        result = cur.fetchone()
                        if result:
                            config_id = result[0]
                            config_cache[config_hash] = config_id

                # Link this execution's model to the config via junction table (even if config existed)
                if model_id is not None:
                    cur.execute(
                        """
                        INSERT INTO ttnn_ops_v5.ttnn_configuration_model (configuration_id, model_id, execution_count)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (configuration_id, model_id) DO UPDATE
                        SET last_seen_ts = NOW(),
                            execution_count = ttnn_ops_v5.ttnn_configuration_model.execution_count + EXCLUDED.execution_count
                    """,
                        (config_id, model_id, execution_count),
                    )
                    total_model_links += 1

                    # Link via trace_run (keyed by hardware_id + sha, no model_id)
                    if hardware_id:
                        trace_run_id = get_or_create_trace_run(
                            cur,
                            trace_run_cache,
                            hardware_id,
                            tt_metal_sha,
                        )
                        link_trace_run_config(cur, trace_run_id, config_id, execution_count)
                        total_trace_run_links += 1

        if not dry_run:
            conn.commit()
        print(f"  Loaded {op_name}: {len(op_data.get('configurations', []))} JSON configs")

    # Update config_count and populate trace_run_model for each new trace_run
    if trace_run_cache:
        for tr_id in trace_run_cache.values():
            cur.execute(
                """
                UPDATE ttnn_ops_v5.trace_run
                SET config_count = (
                    SELECT COUNT(*) FROM ttnn_ops_v5.trace_run_config
                    WHERE trace_run_id = %s
                )
                WHERE trace_run_id = %s
                """,
                (tr_id, tr_id),
            )
            # Derive model links for this trace from ttnn_configuration_model
            cur.execute(
                """
                INSERT INTO ttnn_ops_v5.trace_run_model (trace_run_id, model_id)
                SELECT DISTINCT %s, cm.model_id
                FROM ttnn_ops_v5.trace_run_config trc
                JOIN ttnn_ops_v5.ttnn_configuration_model cm
                    ON cm.configuration_id = trc.configuration_id
                WHERE trc.trace_run_id = %s
                ON CONFLICT DO NOTHING
                """,
                (tr_id, tr_id),
            )

    def _fetch_db_totals(cur):
        counts = {}
        for table, label in [
            ("ttnn_operation", "Operations"),
            ("ttnn_model", "Models"),
            ("ttnn_hardware", "Hardware configs"),
            ("ttnn_mesh_config", "Mesh configs"),
            ("ttnn_configuration", "Configurations"),
            ("ttnn_configuration_model", "Config-Model links"),
            ("trace_run", "Trace runs"),
            ("trace_run_config", "Trace-Config links"),
            ("trace_run_model", "Trace-Model links"),
        ]:
            cur.execute(f"SELECT COUNT(*) FROM ttnn_ops_v5.{table}")
            counts[label] = cur.fetchone()[0]
        return counts

    if dry_run:
        post_load_counts = _fetch_db_totals(cur)
        conn.rollback()
        print("\n🔍 DRY RUN — all DB writes rolled back.")
    else:
        conn.commit()

    print(
        f"\n{'Would load' if dry_run else '✅ Loaded'} {total_trace_run_links} configurations "
        f"({new_configs} new, {total_trace_run_links - new_configs} pre-existing), {total_model_links} model links"
    )
    print(
        f"   Trace runs {'that would be ' if dry_run else ''}created: {len(trace_run_cache)}, "
        f"total configs tied to traces: {total_trace_run_links}"
    )

    if dry_run:
        counts = post_load_counts
        print("\nDB totals if this load were committed:")
    else:
        counts = _fetch_db_totals(cur)
        print("\nDB totals after load:")
    for label, n in counts.items():
        print(f"   {label}: {n}")

    conn.close()

    if dry_run:
        return

    # Auto-append draft entries to manifest registry
    if trace_run_cache and yaml is not None:
        _append_manifest_drafts(trace_run_cache)


def _append_manifest_drafts(trace_run_cache):
    """Append draft registry entries to the manifest for newly created trace_runs."""
    try:
        data, path = _load_manifest()
    except Exception as e:
        print(f"  Warning: could not update manifest registry: {e}")
        return

    # Fetch hardware details and per-trace model names from DB
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        hw_map = {}
        trace_models_map = {}  # trace_run_id -> sorted list of model_names
        for (hardware_id, _sha), trace_run_id in trace_run_cache.items():
            if hardware_id and hardware_id not in hw_map:
                cur.execute(
                    "SELECT board_type, device_series, card_count FROM ttnn_ops_v5.ttnn_hardware WHERE ttnn_hardware_id = %s",
                    (hardware_id,),
                )
                row = cur.fetchone()
                if row:
                    hw_map[hardware_id] = row
            cur.execute(
                """
                SELECT m.model_name FROM ttnn_ops_v5.trace_run_model trm
                JOIN ttnn_ops_v5.ttnn_model m ON m.ttnn_model_id = trm.model_id
                WHERE trm.trace_run_id = %s AND m.model_name IS NOT NULL
                ORDER BY m.model_name
                """,
                (trace_run_id,),
            )
            trace_models_map[trace_run_id] = [r[0] for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        print(f"  Warning: could not fetch hardware/model details for manifest: {e}")
        hw_map = {}
        trace_models_map = {}

    existing_ids = {entry.get("trace_id") for entry in data["registry"]}
    added = 0

    for (hardware_id, sha), trace_run_id in trace_run_cache.items():
        if trace_run_id in existing_ids:
            continue

        hw = hw_map.get(hardware_id, ("unknown", "unknown", 1))
        board_type, device_series, card_count = hw
        models = trace_models_map.get(trace_run_id) or ["unknown"]
        entry = {
            "trace_id": trace_run_id,
            "status": "draft",
            "models": models,
            "hardware": {
                "board_type": board_type,
                "device_series": device_series,
                "card_count": card_count,
            },
            "tt_metal_sha": sha,
            "config_count": None,  # updated after commit
            "loaded_at": str(date.today()),
            "notes": "",
        }
        data["registry"].append(entry)
        existing_ids.add(trace_run_id)
        added += 1

    if added:
        new_entries = data["registry"][-added:]
        # Fetch config counts for new entries
        try:
            conn = psycopg2.connect(NEON_URL)
            cur = conn.cursor()
            for entry in new_entries:
                if entry.get("config_count") is None:
                    cur.execute(
                        "SELECT COUNT(*) FROM ttnn_ops_v5.trace_run_config WHERE trace_run_id = %s",
                        (entry["trace_id"],),
                    )
                    entry["config_count"] = cur.fetchone()[0]
            conn.close()
        except Exception as e:
            # Non-fatal: config_count will remain None in the manifest entry.
            print(f"  Warning: could not populate config_count for new manifest entries: {e}")

        _append_registry_entries(new_entries, path)
        print(f"  Appended {added} draft entries to manifest registry ({path})")


# =============================================================================
# RECONSTRUCTION FUNCTIONS
# =============================================================================


def format_source(source_file, hf_model):
    """Format source_file and hf_model back to original source string format."""
    if not source_file:
        return None
    if hf_model:
        return f"{source_file} [HF_MODEL:{hf_model}]"
    return source_file


def _validate_schema(schema):
    """Validate schema name to prevent SQL injection (used in f-string queries)."""
    if not re.match(r"^[a-zA-Z_]\w*$", schema):
        raise ValueError(f"Invalid schema name: {schema!r}")


def reconstruct_from_db(output_path=None, schema="ttnn_ops_v5", model_filter=None):
    """Reconstruct ttnn_operations_master.json from the database.

    Args:
        output_path: Path to write the reconstructed JSON
        schema: Database schema to use (default: "ttnn_ops_v5")
        model_filter: List of model patterns to filter by (e.g., ["deepseek_v3"]).
                      Only configurations linked to models whose source_file contains
                      one of these patterns (case-insensitive) will be included.
                      If None, all configurations are included.

    This recreates the original JSON structure:
    {
        "operations": {
            "<op_name>": {
                "configurations": [
                    {
                        "arguments": [...],
                        "source": <string | array>,
                        "machine_info": [...]
                    },
                    ...
                ]
            },
            ...
        }
    }
    """
    filter_desc = f", model_filter={model_filter}" if model_filter else ""
    _validate_schema(schema)
    print(f"Reconstructing JSON from database (schema: {schema}{filter_desc})...")
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    if model_filter:
        # Only fetch operations that have at least one config linked to a matching model
        like_clauses = " OR ".join(["m.source_file ILIKE %s"] * len(model_filter))
        like_params = [f"%{pattern}%" for pattern in model_filter]
        cur.execute(
            f"""
            SELECT DISTINCT o.ttnn_operation_id, o.operation_name
            FROM {schema}.ttnn_operation o
            JOIN {schema}.ttnn_configuration c ON c.operation_id = o.ttnn_operation_id
            JOIN {schema}.ttnn_configuration_model cm ON cm.configuration_id = c.ttnn_configuration_id
            JOIN {schema}.ttnn_model m ON m.ttnn_model_id = cm.model_id
            WHERE {like_clauses}
            ORDER BY o.operation_name
        """,
            like_params,
        )
    else:
        cur.execute(
            f"""
            SELECT ttnn_operation_id, operation_name
            FROM {schema}.ttnn_operation
            ORDER BY operation_name
        """
        )
    operations = cur.fetchall()
    print(f"Found {len(operations)} operations")

    result = {"operations": {}}

    # Build model filter subquery once (reused per operation)
    model_filter_clause = ""
    model_filter_params = []
    if model_filter:
        like_clauses = " OR ".join(["m2.source_file ILIKE %s"] * len(model_filter))
        model_filter_clause = f"""
            AND EXISTS (
                SELECT 1 FROM {schema}.ttnn_configuration_model cm2
                JOIN {schema}.ttnn_model m2 ON m2.ttnn_model_id = cm2.model_id
                WHERE cm2.configuration_id = c.ttnn_configuration_id
                AND ({like_clauses})
            )"""
        model_filter_params = [f"%{pattern}%" for pattern in model_filter]

    for op_id, op_name in operations:
        cur.execute(
            f"""
            SELECT
                c.ttnn_configuration_id,
                c.config_hash,
                c.full_config_json,
                c.hardware_id,
                c.mesh_config_id,
                h.board_type,
                h.device_series,
                h.card_count,
                mc.mesh_shape,
                mc.device_count
            FROM {schema}.ttnn_configuration c
            LEFT JOIN {schema}.ttnn_hardware h ON h.ttnn_hardware_id = c.hardware_id
            LEFT JOIN {schema}.ttnn_mesh_config mc ON mc.ttnn_mesh_config_id = c.mesh_config_id
            WHERE c.operation_id = %s
            {model_filter_clause}
            ORDER BY c.ttnn_configuration_id
        """,
            [op_id] + model_filter_params,
        )
        configs = cur.fetchall()

        if not configs:
            continue

        configurations = []

        for config_row in configs:
            (
                config_id,
                config_hash,
                full_config_json,
                _hardware_id,
                _mesh_config_id,
                board_type,
                device_series,
                card_count,
                mesh_shape,
                device_count,
            ) = config_row

            # Get sources linked to this config (filtered by model_filter if set)
            source_filter_clause = ""
            source_filter_params = []
            if model_filter:
                like_clauses = " OR ".join(["m.source_file ILIKE %s"] * len(model_filter))
                source_filter_clause = f" AND ({like_clauses})"
                source_filter_params = [f"%{pattern}%" for pattern in model_filter]

            cur.execute(
                f"""
                SELECT m.source_file, m.hf_model_identifier, cm.execution_count
                FROM {schema}.ttnn_configuration_model cm
                JOIN {schema}.ttnn_model m ON m.ttnn_model_id = cm.model_id
                WHERE cm.configuration_id = %s
                {source_filter_clause}
                ORDER BY m.source_file, m.hf_model_identifier
            """,
                [config_id] + source_filter_params,
            )
            source_rows = cur.fetchall()

            # Use full_config_json for arguments (preserves exact original structure)
            if full_config_json:
                arguments = full_config_json.get("arguments", {})
            else:
                arguments = []

            config_dict = {"arguments": arguments, "config_hash": config_hash}

            executions = []
            for source_file, hf_model, exec_count in source_rows:
                source_str = format_source(source_file, hf_model)
                if not source_str:
                    continue

                execution = {"source": source_str, "machine_info": {}, "count": exec_count}

                if board_type:
                    execution["machine_info"]["board_type"] = board_type
                    execution["machine_info"]["device_series"] = device_series
                    execution["machine_info"]["card_count"] = card_count

                if mesh_shape:
                    execution["machine_info"]["mesh_device_shape"] = mesh_shape
                    if device_count:
                        execution["machine_info"]["device_count"] = device_count

                executions.append(execution)

            if executions:
                config_dict["executions"] = executions

            configurations.append(config_dict)

        if configurations:
            result["operations"][op_name] = {"configurations": configurations}

    conn.close()

    print(f"Reconstructed {len(result['operations'])} operations")

    # Count total configs
    total_configs = sum(len(op["configurations"]) for op in result["operations"].values())
    print(f"Total configurations: {total_configs}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    return result


def reconstruct_from_trace_run(trace_run_id, output_path=None, schema="ttnn_ops_v5", model_names=None):
    """Reconstruct JSON for a specific trace_run, matching the tracer's output format.

    This produces output identical to what generic_ops_tracer.py would generate
    for a fresh trace of the same model on the same hardware.

    Args:
        trace_run_id: ID of the trace_run to reconstruct.
        output_path: Path to write the reconstructed JSON.
        schema: Database schema to read from (default: "ttnn_ops_v5").
        model_names: Optional set/list of model_name values to include. When provided,
            only configs belonging to those models are reconstructed. None = all models.

    Returns:
        Dict in the tracer's master JSON format.
    """
    _validate_schema(schema)
    filter_desc = f", models={sorted(model_names)}" if model_names is not None else ""
    print(f"Reconstructing JSON from trace_run {trace_run_id} (schema: {schema}{filter_desc})...")
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    # Fetch trace_run metadata (v5: hardware via FK, no model_id)
    cur.execute(
        f"""
        SELECT
            tr.trace_run_id,
            h.board_type, h.device_series, h.card_count,
            tr.tt_metal_sha, tr.traced_at, tr.config_count, tr.notes
        FROM {schema}.trace_run tr
        JOIN {schema}.ttnn_hardware h ON h.ttnn_hardware_id = tr.hardware_id
        WHERE tr.trace_run_id = %s
        """,
        (trace_run_id,),
    )
    tr_row = cur.fetchone()
    if not tr_row:
        print(f"Trace run {trace_run_id} not found")
        conn.close()
        return None

    (_, board_type, device_series, card_count, tt_metal_sha, traced_at, _, _) = tr_row

    # Fetch all models for this trace; then apply model_names filter if specified.
    cur.execute(
        f"""
        SELECT m.ttnn_model_id, m.source_file, m.hf_model_identifier, m.model_name
        FROM {schema}.trace_run_model trm
        JOIN {schema}.ttnn_model m ON m.ttnn_model_id = trm.model_id
        WHERE trm.trace_run_id = %s
        ORDER BY m.ttnn_model_id
        """,
        (trace_run_id,),
    )
    trace_models = cur.fetchall()  # [(model_id, source_file, hf_model_identifier, model_name), ...]

    if model_names is not None:
        model_names_set = set(model_names)
        trace_models = [row for row in trace_models if row[3] in model_names_set]
        if not trace_models:
            print(f"  No models matching {sorted(model_names_set)} found in trace {trace_run_id}, skipping.")
            conn.close()
            return None

    trace_model_ids = {row[0] for row in trace_models}
    all_sources = [format_source(sf, hf) for _, sf, hf, _ in trace_models if format_source(sf, hf)]

    # Build machine_info (same format as tracer)
    machine_info = {
        "board_type": board_type,
        "device_series": device_series,
        "card_count": card_count,
    }

    # Fetch all configurations with their per-config model sources.
    # Join ttnn_configuration_model filtered to models in this trace so each
    # config gets the correct execution sources rather than a single trace-level source.
    cur.execute(
        f"""
        SELECT
            o.operation_name,
            c.ttnn_configuration_id,
            c.config_hash,
            c.full_config_json,
            mc.mesh_shape,
            mc.device_count,
            cm.execution_count,
            m.source_file,
            m.hf_model_identifier
        FROM {schema}.trace_run_config trc
        JOIN {schema}.ttnn_configuration c
            ON c.ttnn_configuration_id = trc.configuration_id
        JOIN {schema}.ttnn_operation o
            ON o.ttnn_operation_id = c.operation_id
        LEFT JOIN {schema}.ttnn_mesh_config mc
            ON mc.ttnn_mesh_config_id = c.mesh_config_id
        JOIN {schema}.ttnn_configuration_model cm
            ON cm.configuration_id = trc.configuration_id
        JOIN {schema}.ttnn_model m
            ON m.ttnn_model_id = cm.model_id
           AND m.ttnn_model_id = ANY(%s)
        WHERE trc.trace_run_id = %s
        ORDER BY o.operation_name, c.ttnn_configuration_id, m.ttnn_model_id
        """,
        (list(trace_model_ids), trace_run_id),
    )
    rows = cur.fetchall()
    conn.close()

    # Group by (op_name, config_id): collect one execution entry per model source
    result = {"operations": {}, "metadata": {}}
    ops = {}  # op_name -> {config_id -> config_dict}

    for (
        op_name,
        config_id,
        config_hash,
        full_config_json,
        mesh_shape,
        device_count,
        execution_count,
        source_file,
        hf_model_identifier,
    ) in rows:
        arguments = full_config_json.get("arguments", {}) if full_config_json else {}

        exec_machine_info = machine_info.copy()
        if mesh_shape:
            exec_machine_info["mesh_device_shape"] = mesh_shape
            if device_count:
                exec_machine_info["device_count"] = device_count

        source = format_source(source_file, hf_model_identifier)

        if op_name not in ops:
            ops[op_name] = {}

        if config_id not in ops[op_name]:
            ops[op_name][config_id] = {
                "config_hash": config_hash,
                "arguments": arguments,
                "executions": [],
            }

        ops[op_name][config_id]["executions"].append(
            {
                "source": source,
                "machine_info": exec_machine_info,
                "count": execution_count,
            }
        )

    # Build final structure
    for op_name in sorted(ops.keys()):
        result["operations"][op_name] = {"configurations": list(ops[op_name].values())}

    # Add metadata
    total_configs = sum(len(op["configurations"]) for op in result["operations"].values())
    result["metadata"] = {
        "models": all_sources,
        "unique_operations": len(result["operations"]),
        "total_configurations": total_configs,
        "trace_run_id": trace_run_id,
        "board_type": board_type,
        "device_series": device_series,
        "card_count": card_count,
        "tt_metal_sha": tt_metal_sha,
        "traced_at": str(traced_at) if traced_at else None,
    }

    print(f"Reconstructed {len(result['operations'])} operations, {total_configs} configurations")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    return result


def list_trace_runs(model_filter=None):
    """List all trace runs, optionally filtered by model.

    Args:
        model_filter: List of model patterns to filter by (e.g., ["deepseek_v3"]).
    """
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    # v5: no model_id on trace_run; hardware via FK; models via trace_run_model
    query = """
        SELECT
            tr.trace_run_id,
            h.device_series,
            h.card_count,
            tr.tt_metal_sha,
            tr.traced_at,
            tr.config_count,
            tr.notes,
            STRING_AGG(COALESCE(m.model_name, m.source_file), ', '
                       ORDER BY COALESCE(m.model_name, m.source_file)) AS models
        FROM ttnn_ops_v5.trace_run tr
        JOIN ttnn_ops_v5.ttnn_hardware h ON h.ttnn_hardware_id = tr.hardware_id
        LEFT JOIN ttnn_ops_v5.trace_run_model trm ON trm.trace_run_id = tr.trace_run_id
        LEFT JOIN ttnn_ops_v5.ttnn_model m ON m.ttnn_model_id = trm.model_id
    """
    params = []

    if model_filter:
        like_clauses = " OR ".join(["m.source_file ILIKE %s"] * len(model_filter))
        query += f" WHERE ({like_clauses})"
        params = [f"%{p}%" for p in model_filter]

    query += " GROUP BY tr.trace_run_id, h.device_series, h.card_count ORDER BY tr.traced_at DESC"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No trace runs found")
        return []

    print(f"\n{'ID':>4}  {'Hardware':20}  {'SHA':12}  {'Configs':>7}  {'Traced At':20}  Models")
    print("-" * 120)

    for tr_id, device_series, card_count, sha, traced_at, cfg_count, _, models in rows:
        hw = f"{device_series} ({card_count}x)"
        sha_short = sha[:12] if sha else "-"
        traced = str(traced_at)[:19] if traced_at else "-"
        print(f"{tr_id:>4}  {hw:20}  {sha_short:12}  {cfg_count or 0:>7}  {traced:20}  {models or ''}")

    return rows


def resolve_manifest(manifest_path=None, scope=None):
    """Resolve manifest targets to a list of trace_run_ids.

    The manifest targets section is a mapping with two scope groups:
      targets:
        lead_models:
          - model: X [, trace: N] [, hardware: H]
        model_traced:
          - model: X [, trace: N] [, hardware: H]

    Resolution rules per entry:
      - trace: N provided  -> use that trace_id directly (pinned)
      - trace omitted      -> latest active trace per unique device_series where model
                             exactly matches a model_name in the registry models list

    Args:
        manifest_path: Path to sweep_manifest.yaml (optional).
        scope: Which group to resolve.
            - 'lead_models'  : only the lead_models group
            - 'model_traced' : only the model_traced group
            - None           : both groups combined

    Returns:
        List of unique trace_run_id integers.
    """
    data, _ = _load_manifest(manifest_path)
    targets_map = data.get("targets", {})
    registry = data.get("registry", [])

    if not targets_map:
        print("No targets in manifest")
        return []

    # Select which scope groups to process
    if scope == "lead_models":
        groups = {"lead_models": targets_map.get("lead_models", [])}
    elif scope == "model_traced":
        groups = {"model_traced": targets_map.get("model_traced", [])}
    else:
        groups = {k: v for k, v in targets_map.items()}

    resolved_ids = []
    total_entries = sum(len(v) for v in groups.values())
    print(f"Scope: {scope or 'all'} ({total_entries} entries across {len(groups)} group(s))")

    for group_name, entries in groups.items():
        if not entries:
            continue
        for entry in entries:
            model_val = entry.get("model")
            pinned_trace = entry.get("trace")
            hw_filter = entry.get("hardware")

            if pinned_trace is not None:
                # Pinned trace: single int or list of ints, use directly
                if isinstance(pinned_trace, list):
                    resolved_ids.extend(int(t) for t in pinned_trace)
                else:
                    resolved_ids.append(int(pinned_trace))
                continue

            if not model_val:
                print(f"  Warning: entry in {group_name} has no 'model' or 'trace', skipping: {entry}")
                continue

            # model may be a string or a list; each model is resolved independently
            # so each gets the latest active trace available for it
            patterns = [model_val] if isinstance(model_val, str) else list(model_val)

            for pattern in patterns:
                # Filter registry to active entries matching this model
                candidates = []
                for reg_entry in registry:
                    if reg_entry.get("status") != "active":
                        continue
                    reg_models = reg_entry.get("models", [])
                    if pattern not in reg_models:
                        continue
                    if hw_filter:
                        hw = reg_entry.get("hardware", {})
                        if hw.get("device_series") != hw_filter:
                            continue
                    candidates.append(reg_entry)

                if not candidates:
                    print(f"  Warning: no active traces match model '{pattern}' in {group_name}")
                    continue

                # Pick latest (highest trace_id) per device_series
                by_hw = {}
                for reg_entry in candidates:
                    ds = reg_entry.get("hardware", {}).get("device_series", "unknown")
                    existing = by_hw.get(ds)
                    if existing is None or reg_entry["trace_id"] > existing["trace_id"]:
                        by_hw[ds] = reg_entry

                for reg_entry in by_hw.values():
                    resolved_ids.append(reg_entry["trace_id"])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for tid in resolved_ids:
        if tid not in seen:
            seen.add(tid)
            unique.append(tid)

    print(f"Resolved to {len(unique)} trace_run_id(s): {unique}")
    return unique


def _resolve_manifest_with_models(manifest_path=None, scope=None):
    """Like resolve_manifest but returns {trace_id: set_of_model_names | None}.

    None means no filter (all models in the trace).
    A set means only reconstruct configs belonging to those model_names.
    """
    data, _ = _load_manifest(manifest_path)
    targets_map = data.get("targets", {})
    registry = data.get("registry", [])

    if scope == "lead_models":
        groups = {"lead_models": targets_map.get("lead_models") or []}
    elif scope == "model_traced":
        groups = {"model_traced": targets_map.get("model_traced") or []}
    else:
        groups = {k: v for k, v in targets_map.items() if v}

    # trace_id -> set of model_names to include (None = all models)
    trace_model_map = {}

    for entries in groups.values():
        for entry in entries or []:
            model_val = entry.get("model")
            pinned_trace = entry.get("trace")
            hw_filter = entry.get("hardware")

            models = [model_val] if isinstance(model_val, str) else (list(model_val) if model_val else [])

            def _add(tid, model_list):
                if tid not in trace_model_map:
                    trace_model_map[tid] = set() if model_list else None
                if trace_model_map[tid] is not None and model_list:
                    trace_model_map[tid].update(model_list)
                elif not model_list:
                    trace_model_map[tid] = None  # no model filter = all

            if pinned_trace is not None:
                tids = [int(pinned_trace)] if not isinstance(pinned_trace, list) else [int(t) for t in pinned_trace]
                for tid in tids:
                    _add(tid, models)
                continue

            # Registry-resolved: each model resolves to latest active trace per device_series
            for model_name in models:
                candidates = [
                    r
                    for r in registry
                    if r.get("status") == "active"
                    and model_name in r.get("models", [])
                    and (not hw_filter or r.get("hardware", {}).get("device_series") == hw_filter)
                ]
                by_hw = {}
                for r in candidates:
                    ds = r.get("hardware", {}).get("device_series", "unknown")
                    if ds not in by_hw or r["trace_id"] > by_hw[ds]["trace_id"]:
                        by_hw[ds] = r
                for r in by_hw.values():
                    _add(r["trace_id"], [model_name])

    return trace_model_map


def reconstruct_from_manifest(manifest_path=None, output_path=None, scope=None, schema="ttnn_ops_v5"):
    """Reconstruct merged JSON from manifest targets.

    Resolves targets to (trace_run_id, model_names) pairs, reconstructs each
    filtered to the specified models, then merges configs deduplicated by config_hash.

    Args:
        manifest_path: Path to sweep_manifest.yaml (optional).
        output_path: Path to write the merged JSON (optional).
        scope: 'lead_models' or 'model_traced' (None = all targets).
        schema: Database schema to read from (default: "ttnn_ops_v5").

    Returns:
        Merged dict in the tracer's master JSON format.
    """
    trace_model_map = _resolve_manifest_with_models(manifest_path, scope=scope)
    if not trace_model_map:
        print("Nothing to reconstruct")
        return {"operations": {}, "metadata": {}}

    print(f"Resolved {len(trace_model_map)} trace(s):")
    for tid, mnames in trace_model_map.items():
        label = sorted(mnames) if mnames is not None else "all"
        print(f"  trace {tid}: {label}")

    # Reconstruct each trace filtered to its model set
    all_results = []
    for tid, mnames in trace_model_map.items():
        result = reconstruct_from_trace_run(tid, schema=schema, model_names=mnames)
        if result:
            all_results.append(result)

    if not all_results:
        print("No traces could be reconstructed")
        return {"operations": {}, "metadata": {}}

    # Merge: union operations, deduplicate configs by config_hash
    merged = {"operations": {}, "metadata": {"trace_run_ids": sorted(trace_model_map.keys()), "models": []}}

    seen_hashes = {}  # config_hash -> True (global dedup across all traces)
    total_configs = 0
    total_deduped = 0

    for result in all_results:
        # Merge metadata
        for model in result.get("metadata", {}).get("models", []):
            if model not in merged["metadata"]["models"]:
                merged["metadata"]["models"].append(model)

        # Merge operations
        for op_name, op_data in result.get("operations", {}).items():
            if op_name not in merged["operations"]:
                merged["operations"][op_name] = {"configurations": []}

            for config in op_data.get("configurations", []):
                ch = config.get("config_hash")
                total_configs += 1
                if ch and ch in seen_hashes:
                    total_deduped += 1
                    continue
                if ch:
                    seen_hashes[ch] = True
                merged["operations"][op_name]["configurations"].append(config)

    # Update metadata
    final_configs = sum(len(op["configurations"]) for op in merged["operations"].values())
    merged["metadata"]["unique_operations"] = len(merged["operations"])
    merged["metadata"]["total_configurations"] = final_configs

    print(f"Merged {len(all_results)} traces: {final_configs} configs ({total_deduped} duplicates removed)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Saved to {output_path}")

    return merged


def reconstruct_single_operation(operation_name, output_path=None):
    """Reconstruct JSON for a single operation (faster for testing)."""
    print(f"Reconstructing {operation_name} from database...")
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    # Get operation ID
    cur.execute(
        "SELECT ttnn_operation_id FROM ttnn_ops_v5.ttnn_operation WHERE operation_name = %s",
        (operation_name,),
    )
    row = cur.fetchone()
    if not row:
        print(f"Operation {operation_name} not found")
        conn.close()
        return None

    op_id = row[0]

    # Get all configurations
    cur.execute(
        """
        SELECT
            c.ttnn_configuration_id,
            c.config_hash,
            c.full_config_json,
            h.board_type,
            h.device_series,
            h.card_count,
            mc.mesh_shape,
            mc.device_count
        FROM ttnn_ops_v5.ttnn_configuration c
        LEFT JOIN ttnn_ops_v5.ttnn_hardware h ON h.ttnn_hardware_id = c.hardware_id
        LEFT JOIN ttnn_ops_v5.ttnn_mesh_config mc ON mc.ttnn_mesh_config_id = c.mesh_config_id
        WHERE c.operation_id = %s
        ORDER BY c.ttnn_configuration_id
    """,
        (op_id,),
    )
    configs = cur.fetchall()

    configurations = []

    for config_row in configs:
        (
            config_id,
            config_hash,
            full_config_json,
            board_type,
            device_series,
            card_count,
            mesh_shape,
            device_count,
        ) = config_row

        # Get sources
        cur.execute(
            """
            SELECT m.source_file, m.hf_model_identifier, cm.execution_count
            FROM ttnn_ops_v5.ttnn_configuration_model cm
            JOIN ttnn_ops_v5.ttnn_model m ON m.ttnn_model_id = cm.model_id
            WHERE cm.configuration_id = %s
            ORDER BY m.source_file, m.hf_model_identifier
        """,
            (config_id,),
        )
        source_rows = cur.fetchall()

        arguments = full_config_json.get("arguments", {}) if full_config_json else []
        config_dict = {"config_hash": config_hash, "arguments": arguments}

        executions = []
        for source_file, hf_model, exec_count in source_rows:
            source_str = format_source(source_file, hf_model)
            if not source_str:
                continue
            execution = {"source": source_str, "machine_info": {}, "count": exec_count}
            if board_type:
                execution["machine_info"]["board_type"] = board_type
                execution["machine_info"]["device_series"] = device_series
                execution["machine_info"]["card_count"] = card_count
            if mesh_shape:
                execution["machine_info"]["mesh_device_shape"] = mesh_shape
                if device_count:
                    execution["machine_info"]["device_count"] = device_count
            executions.append(execution)

        if executions:
            config_dict["executions"] = executions

        configurations.append(config_dict)

    conn.close()

    result = {"operations": {operation_name: {"configurations": configurations}}}

    print(f"Reconstructed {len(configurations)} configurations")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    return result


def verify_reconstruction(original_path, reconstructed_path=None):
    """Compare original JSON with reconstructed version.

    Note: The reconstruction may have MORE configs than original because:
    - Original: 1 config with machine_info array of 2 items
    - Reconstructed: 2 configs (one per hardware+mesh combination)

    This function compares at the operation level and reports differences.
    """
    print(f"Loading original from {original_path}...")
    with open(original_path) as f:
        original = json.load(f)

    if reconstructed_path:
        print(f"Loading reconstructed from {reconstructed_path}...")
        with open(reconstructed_path) as f:
            reconstructed = json.load(f)
    else:
        print("Reconstructing from database...")
        reconstructed = reconstruct_from_db()

    orig_ops = set(original.get("operations", {}).keys())
    recon_ops = set(reconstructed.get("operations", {}).keys())

    print(f"\n=== Comparison ===")
    print(f"Original operations: {len(orig_ops)}")
    print(f"Reconstructed operations: {len(recon_ops)}")

    missing_ops = orig_ops - recon_ops
    extra_ops = recon_ops - orig_ops

    if missing_ops:
        print(f"Missing operations: {missing_ops}")
    if extra_ops:
        print(f"Extra operations: {extra_ops}")

    # Compare config counts per operation
    common_ops = orig_ops & recon_ops
    print(f"\nComparing {len(common_ops)} common operations...")

    diffs = []
    for op_name in sorted(common_ops):
        orig_configs = len(original["operations"][op_name].get("configurations", []))
        recon_configs = len(reconstructed["operations"][op_name].get("configurations", []))

        if orig_configs != recon_configs:
            diffs.append((op_name, orig_configs, recon_configs))

    if diffs:
        print(f"\nConfig count differences (expected due to machine_info expansion):")
        for op_name, orig, recon in diffs[:10]:
            print(f"  {op_name}: {orig} -> {recon}")
        if len(diffs) > 10:
            print(f"  ... and {len(diffs) - 10} more")
    else:
        print("All operation config counts match!")

    return {
        "original_ops": len(orig_ops),
        "reconstructed_ops": len(recon_ops),
        "missing_ops": list(missing_ops),
        "extra_ops": list(extra_ops),
        "config_diffs": diffs,
    }


def find_config_line_numbers(json_path, operation_name, config_indices):
    """Find the line numbers for specific config indices in the JSON file.

    Args:
        json_path: Path to JSON file
        operation_name: Name of the operation (e.g., 'ttnn::add')
        config_indices: List of config indices to find

    Returns:
        Dict mapping config index to line number
    """

    print(f"Finding line numbers for {operation_name} configs {config_indices}...")

    with open(json_path) as f:
        content = f.read()

    # Find the start of the operation section
    op_start = content.find(f'"{operation_name}"')
    if op_start == -1:
        print(f"Operation {operation_name} not found")
        return {}

    # Find the end of this operation (start of next operation or end of operations object)
    next_op_pattern = re.compile(r'"ttnn::[^"]+": \{')
    matches = list(next_op_pattern.finditer(content, op_start + len(operation_name) + 10))

    if matches:
        op_end = matches[0].start()
    else:
        op_end = len(content)

    # Find all config starts within this operation section
    config_pattern = re.compile(r'\{\s*"arguments"')
    config_matches = list(config_pattern.finditer(content, op_start))

    # Filter to only configs within this operation
    config_positions = []
    for m in config_matches:
        if m.start() < op_end:
            config_positions.append(m.start())
        else:
            break

    # Calculate line numbers for requested indices
    result = {}
    for idx in config_indices:
        if idx < len(config_positions):
            pos = config_positions[idx]
            line_num = content[:pos].count("\n") + 1
            result[idx] = line_num
        else:
            result[idx] = None

    for idx, line in result.items():
        if line:
            print(f"  Config {idx}: Line {line}")
        else:
            print(f"  Config {idx}: Not found")

    return result


def set_model_name(source_file=None, hf_model=None, model_id=None, new_name=None):
    """Override model_name for a specific ttnn_model row.

    Identify the row by source_file (+ optional hf_model), hf_model alone, or
    model_id. The new name is lowercased before being stored.
    """
    if not new_name:
        print("Error: --model-name is required")
        return
    if not (source_file or hf_model or model_id):
        print("Error: provide --source-file, --hf-model, or --model-id to identify the row")
        return

    new_name = new_name.lower()
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    if model_id is not None:
        cur.execute(
            "UPDATE ttnn_ops_v5.ttnn_model SET model_name = %s, update_ts = NOW()"
            " WHERE ttnn_model_id = %s"
            " RETURNING ttnn_model_id, source_file, hf_model_identifier",
            (new_name, int(model_id)),
        )
    elif source_file and hf_model:
        cur.execute(
            "UPDATE ttnn_ops_v5.ttnn_model SET model_name = %s, update_ts = NOW()"
            " WHERE source_file = %s AND hf_model_identifier = %s"
            " RETURNING ttnn_model_id, source_file, hf_model_identifier",
            (new_name, source_file, hf_model),
        )
    elif source_file:
        cur.execute(
            "UPDATE ttnn_ops_v5.ttnn_model SET model_name = %s, update_ts = NOW()"
            " WHERE source_file = %s"
            " RETURNING ttnn_model_id, source_file, hf_model_identifier",
            (new_name, source_file),
        )
    else:  # hf_model only
        cur.execute(
            "UPDATE ttnn_ops_v5.ttnn_model SET model_name = %s, update_ts = NOW()"
            " WHERE hf_model_identifier = %s"
            " RETURNING ttnn_model_id, source_file, hf_model_identifier",
            (new_name, hf_model),
        )

    updated_rows = cur.fetchall()
    if not updated_rows:
        print("No matching model found — nothing updated.")
        conn.close()
        return

    conn.commit()
    conn.close()
    for row in updated_rows:
        print(f"Updated ttnn_model_id={row[0]}: model_name set to '{new_name}'")
        print(f"  source_file={row[1]!r}  hf_model={row[2]!r}")


def delete_trace_run(trace_run_id, yes=False):
    """Delete a trace_run and any configs that belong exclusively to it.

    A config is deleted only if no other trace_run links to it.
    Configs shared with other traces are unlinked but not deleted.

    Prints a summary and asks for confirmation unless --yes is passed.
    """
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    cur.execute(
        "SELECT config_count, notes FROM ttnn_ops_v5.trace_run WHERE trace_run_id = %s",
        (trace_run_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"Trace run {trace_run_id} not found.")
        conn.close()
        return

    config_count, notes = row

    cur.execute(
        """
        SELECT COUNT(*) FROM ttnn_ops_v5.trace_run_config trc
        WHERE trc.trace_run_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM ttnn_ops_v5.trace_run_config other
              WHERE other.configuration_id = trc.configuration_id
                AND other.trace_run_id != %s
          )
        """,
        (trace_run_id, trace_run_id),
    )
    exclusive_count = cur.fetchone()[0]
    shared_count = (config_count or 0) - exclusive_count

    print(f"Trace run {trace_run_id}: {config_count} configs, notes={notes!r}")
    print(f"  {exclusive_count} configs will be DELETED (exclusive to this trace)")
    print(f"  {shared_count} configs will be UNLINKED only (shared with other traces)")

    if not yes:
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            conn.close()
            return

    # Collect exclusive config IDs before touching any rows
    cur.execute(
        """
        SELECT trc.configuration_id FROM ttnn_ops_v5.trace_run_config trc
        WHERE trc.trace_run_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM ttnn_ops_v5.trace_run_config other
              WHERE other.configuration_id = trc.configuration_id
                AND other.trace_run_id != %s
          )
        """,
        (trace_run_id, trace_run_id),
    )
    exclusive_ids = [r[0] for r in cur.fetchall()]

    # 1. ttnn_configuration_model rows for exclusive configs
    if exclusive_ids:
        cur.execute(
            "DELETE FROM ttnn_ops_v5.ttnn_configuration_model WHERE configuration_id = ANY(%s)",
            (exclusive_ids,),
        )

    # 2. All trace_run_config rows for this trace
    cur.execute("DELETE FROM ttnn_ops_v5.trace_run_config WHERE trace_run_id = %s", (trace_run_id,))

    # 3. trace_run_model rows for this trace
    cur.execute("DELETE FROM ttnn_ops_v5.trace_run_model WHERE trace_run_id = %s", (trace_run_id,))

    # 4. Exclusive configs themselves
    if exclusive_ids:
        cur.execute(
            "DELETE FROM ttnn_ops_v5.ttnn_configuration WHERE ttnn_configuration_id = ANY(%s)",
            (exclusive_ids,),
        )
    deleted_configs = len(exclusive_ids)

    # 5. The trace_run itself
    cur.execute("DELETE FROM ttnn_ops_v5.trace_run WHERE trace_run_id = %s", (trace_run_id,))

    conn.commit()
    conn.close()
    print(f"Done: trace run {trace_run_id} deleted, {deleted_configs} configs removed, {shared_count} unlinked.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "load":
            args = sys.argv[2:]
            dry_run = "--dry-run" in args
            args = [a for a in args if a != "--dry-run"]
            json_path = args[0] if len(args) > 0 else None
            sha = args[1] if len(args) > 1 else None
            load_data(json_path=json_path, tt_metal_sha=sha, dry_run=dry_run)
        elif cmd == "reconstruct":
            output = sys.argv[2] if len(sys.argv) > 2 else "ttnn_operations_reconstructed.json"
            schema = sys.argv[3] if len(sys.argv) > 3 else "ttnn_ops_v5"
            model_filter = sys.argv[4].split(",") if len(sys.argv) > 4 else None
            reconstruct_from_db(output, schema, model_filter)
        elif cmd == "reconstruct-trace":
            if len(sys.argv) < 3:
                print("Usage: python load_ttnn_ops_data_v2.py reconstruct-trace <trace_run_id> [output.json]")
                sys.exit(1)
            tr_id = int(sys.argv[2])
            output = sys.argv[3] if len(sys.argv) > 3 else None
            reconstruct_from_trace_run(tr_id, output)
        elif cmd == "list-traces":
            model_filter = sys.argv[2].split(",") if len(sys.argv) > 2 else None
            list_trace_runs(model_filter)
        elif cmd == "reconstruct-manifest":
            _args = sys.argv[2:]
            # Allow omitting the manifest path: if the first arg ends in .json it's the output
            if _args and _args[0].endswith(".json"):
                manifest, output = None, _args[0]
                _args = _args[1:]
            else:
                manifest = _args[0] if _args else None
                output = _args[1] if len(_args) > 1 else None
                _args = _args[2:]
            scope = _args[0] if _args else None
            schema = _args[1] if len(_args) > 1 else "ttnn_ops_v5"
            reconstruct_from_manifest(manifest, output, scope, schema)
        elif cmd == "resolve-manifest":
            manifest = sys.argv[2] if len(sys.argv) > 2 else None
            scope = sys.argv[3] if len(sys.argv) > 3 else None
            resolve_manifest(manifest, scope)
        elif cmd == "reconstruct-op":
            if len(sys.argv) < 3:
                print("Usage: python load_ttnn_ops_data_v2.py reconstruct-op <operation_name> [output.json]")
                sys.exit(1)
            op_name = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else None
            reconstruct_single_operation(op_name, output)
        elif cmd == "verify":
            original = sys.argv[2] if len(sys.argv) > 2 else JSON_PATH
            reconstructed = sys.argv[3] if len(sys.argv) > 3 else None
            verify_reconstruction(original, reconstructed)
        elif cmd == "find-lines":
            if len(sys.argv) < 4:
                print("Usage: python load_ttnn_ops_data_v2.py find-lines <operation> <index1,index2,...>")
                sys.exit(1)
            op_name = sys.argv[2]
            indices = [int(i) for i in sys.argv[3].split(",")]
            json_file = sys.argv[4] if len(sys.argv) > 4 else JSON_PATH
            find_config_line_numbers(json_file, op_name, indices)
        elif cmd == "delete-trace":
            if len(sys.argv) < 3:
                print("Usage: python load_ttnn_ops_data_v2.py delete-trace <trace_run_id> [--yes]")
                sys.exit(1)
            _tr_id = int(sys.argv[2])
            _yes = "--yes" in sys.argv
            delete_trace_run(_tr_id, yes=_yes)
        elif cmd == "set-model-name":
            # Parse --key value flags
            _args = sys.argv[2:]
            _kv = {}
            i = 0
            while i < len(_args):
                if _args[i].startswith("--") and i + 1 < len(_args):
                    _kv[_args[i][2:]] = _args[i + 1]
                    i += 2
                else:
                    i += 1
            set_model_name(
                source_file=_kv.get("source-file"),
                hf_model=_kv.get("hf-model"),
                model_id=_kv.get("model-id"),
                new_name=_kv.get("model-name"),
            )
        else:
            print(f"Unknown command: {cmd}")
            print("Usage:")
            print(
                "  python load_ttnn_ops_data_v2.py load [json_path] [sha] [--dry-run]           # Load JSON to DB (--dry-run rolls back)"
            )
            print(
                "  python load_ttnn_ops_data_v2.py reconstruct [output] [schema] [models]      # Reconstruct JSON from DB"
            )
            print(
                "  python load_ttnn_ops_data_v2.py reconstruct-trace <id> [output.json]         # Reconstruct from trace_run"
            )
            print(
                "  python load_ttnn_ops_data_v2.py reconstruct-manifest [manifest] [output] [scope]  # Reconstruct from manifest"
            )
            print(
                "  python load_ttnn_ops_data_v2.py resolve-manifest [manifest] [scope]               # Show resolved trace IDs"
            )
            print("  python load_ttnn_ops_data_v2.py list-traces [model_filter]                   # List trace runs")
            print(
                "  python load_ttnn_ops_data_v2.py reconstruct-op <name>                       # Reconstruct single op"
            )
            print("  python load_ttnn_ops_data_v2.py verify [original] [reconstructed]            # Compare files")
            print(
                "  python load_ttnn_ops_data_v2.py find-lines <op> <i1,i2>                     # Find config line numbers"
            )
            print(
                "  python load_ttnn_ops_data_v2.py delete-trace <id> [--yes]                     # Delete trace and its exclusive configs"
            )
            print(
                "  python load_ttnn_ops_data_v2.py set-model-name --source-file P --model-name N  # Override a model's name"
            )
    else:
        load_data()
