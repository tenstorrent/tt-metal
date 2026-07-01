#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Load ttnn_operations_master.json into Snowflake (SELF_SERVE.TTNN_OPS_V6).

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

try:
    import yaml
except ImportError:
    yaml = None

from model_tracer.mesh_metadata import normalize_machine_info

# Default manifest path (relative to repo root)
_DEFAULT_MANIFEST = "model_tracer/trace_selection_registry.yaml"


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
        # Create an on-disk skeleton so subsequent append operations produce valid YAML
        manifest_dir = os.path.dirname(path)
        if manifest_dir and not os.path.exists(manifest_dir):
            os.makedirs(manifest_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write("targets: {}\nregistry:\n")
    if not isinstance(data, dict):
        data = {}
    if data.get("targets") is None or not isinstance(data.get("targets"), dict):
        data["targets"] = {}
    if data.get("registry") is None or not isinstance(data.get("registry"), list):
        data["registry"] = []
    return data, path


def _append_registry_entries(entries, path):
    """Append new registry entries to the manifest without rewriting the file.

    Preserves existing comments and formatting by appending raw YAML text.
    """
    _require_yaml()
    with open(path, "a") as f:
        for entry in entries:
            # Blank line separator before each appended entry (keeps a single
            # trailing newline at end-of-file so end-of-file-fixer stays happy).
            f.write("\n")
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
            if entry.get("trace_uid"):
                f.write(f"    trace_uid: {entry['trace_uid']}\n")
            f.write(f"    tt_metal_sha: {entry.get('tt_metal_sha') or 'null'}\n")
            f.write(f"    config_count: {entry.get('config_count', 0)}\n")
            loaded_at = str(entry.get("loaded_at", ""))
            notes = str(entry.get("notes", ""))
            # Escape single quotes for YAML single-quoted scalars
            loaded_at_escaped = loaded_at.replace("'", "''")
            notes_escaped = notes.replace("'", "''")
            f.write(f"    loaded_at: '{loaded_at_escaped}'\n")
            f.write(f"    notes: '{notes_escaped}'\n")
            pytest_args = entry.get("pytest_args")
            if pytest_args:
                pytest_args_escaped = str(pytest_args).replace("'", "''")
                f.write(f"    pytest_args: '{pytest_args_escaped}'\n")
            else:
                f.write("    pytest_args: null\n")
            tt_kmd = entry.get("tt_kmd")
            tt_smi = entry.get("tt_smi")
            tt_firmware = entry.get("tt_firmware")
            f.write(f"    tt_kmd: {json.dumps(tt_kmd) if tt_kmd is not None else 'null'}\n")
            f.write(f"    tt_smi: {json.dumps(tt_smi) if tt_smi is not None else 'null'}\n")
            f.write(f"    tt_firmware: {json.dumps(tt_firmware) if tt_firmware is not None else 'null'}\n")


DEFAULT_SCHEMA = "ttnn_ops_v6"

JSON_PATH = "model_tracer/traced_operations/ttnn_operations_master.json"


# ---------------------------------------------------------------------------
# Snowflake backend (reads AND writes).
#
# Data lives in <SNOWFLAKE_DATABASE>.<SCHEMA> (e.g. SELF_SERVE.TTNN_OPS_V6).
# Auth prefers a service-account RSA keypair (for CI); falls back to SSO.
#   SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER            (required)
#   SNOWFLAKE_PRIVATE_KEY or SNOWFLAKE_PRIVATE_KEY_PATH  (+ optional _PASSPHRASE)
#   SNOWFLAKE_ROLE (default SELF_SERVE), SNOWFLAKE_WAREHOUSE (default PUBLIC),
#   SNOWFLAKE_DATABASE (default SELF_SERVE)
#
# Snowflake SQL dialect notes handled on the write path:
#   - no ON CONFLICT / RETURNING / SAVEPOINT, no enforced UNIQUE/PK
#   - id columns are plain NUMBER; ids come from per-column SEQUENCEs
#     (see _ensure_sequences / _next_id), allocated then INSERTed explicitly
#   - membership tests use `IN (<placeholders>)`
#   - timestamps use CURRENT_TIMESTAMP()
# ---------------------------------------------------------------------------
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "SELF_SERVE")


def _load_private_key():
    """Load an RSA private key (PEM in env or file) into DER/PKCS8 bytes, or None."""
    key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY")
    if not key_path and not key_pem:
        return None
    from cryptography.hazmat.primitives import serialization

    raw = open(key_path, "rb").read() if key_path else key_pem.encode()
    passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    return serialization.load_pem_private_key(raw, password=passphrase.encode() if passphrase else None).private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _get_read_connection():
    """Connection for read/reconstruct queries (Snowflake)."""
    import snowflake.connector

    params = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        role=os.environ.get("SNOWFLAKE_ROLE", "SELF_SERVE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "PUBLIC"),
        database=SNOWFLAKE_DATABASE,
    )
    pkey = _load_private_key()
    if pkey is not None:
        params["private_key"] = pkey
    else:
        params["authenticator"] = "externalbrowser"
        params["client_store_temporary_credential"] = True
    return snowflake.connector.connect(**params)


def _get_write_connection():
    """Connection for write/load queries (Snowflake).

    Mirrors _get_read_connection. Autocommit is disabled so load_data's
    explicit commit()/rollback() (and the dry_run rollback) behave predictably.
    """
    import snowflake.connector

    params = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        role=os.environ.get("SNOWFLAKE_ROLE", "SELF_SERVE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "PUBLIC"),
        database=SNOWFLAKE_DATABASE,
        autocommit=False,
    )
    pkey = _load_private_key()
    if pkey is not None:
        params["private_key"] = pkey
    else:
        params["authenticator"] = "externalbrowser"
        params["client_store_temporary_credential"] = True
    return snowflake.connector.connect(**params)


def _qualified_schema(schema):
    return f"{SNOWFLAKE_DATABASE}.{schema.upper()}"


def _schema_prefix(schema):
    """Qualified table-name prefix."""
    return _qualified_schema(schema)


# Map id column -> table for Snowflake SEQUENCE emulation of auto-increment.
# Snowflake does not enforce PK/UNIQUE and these id columns are plain NUMBER,
# so we allocate ids from a per-column SEQUENCE seeded at MAX(id)+1.
_ID_COLUMN_TABLE = {
    "ttnn_operation_id": "ttnn_operation",
    "ttnn_model_id": "ttnn_model",
    "ttnn_hardware_id": "ttnn_hardware",
    "ttnn_mesh_config_id": "ttnn_mesh_config",
    "ttnn_configuration_id": "ttnn_configuration",
    "trace_run_id": "trace_run",
}


def _seq_name(schema, table):
    """Snowflake sequence name for a table's id column."""
    return f"{_qualified_schema(schema)}.SEQ_{table.upper()}"


def _ensure_sequences(cur, schema):
    """Create (once per load) a Snowflake SEQUENCE per id column, seeded MAX(id)+1.

    Snowflake has no autoincrement on these NUMBER id columns, so we back each
    id column with a sequence. START is computed from the current MAX so we
    never collide with pre-existing rows. IF NOT EXISTS makes this idempotent.
    """
    S = _qualified_schema(schema)
    for id_col, table in _ID_COLUMN_TABLE.items():
        cur.execute(f"SELECT COALESCE(MAX({id_col}), 0) + 1 FROM {S}.{table}")
        start = int(cur.fetchone()[0])
        cur.execute(f"CREATE SEQUENCE IF NOT EXISTS {_seq_name(schema, table)} START = {start}")


def _next_id(cur, schema, table):
    """Allocate the next id from the Snowflake sequence backing `table`."""
    cur.execute(f"SELECT {_seq_name(schema, table)}.NEXTVAL")
    return int(cur.fetchone()[0])


def _agg_ordered(col):
    """DISTINCT array-aggregate of `col`, ordered by `col`."""
    return f"ARRAY_AGG(DISTINCT {col}) WITHIN GROUP (ORDER BY {col})"


def _agg_distinct_nonnull(col):
    """DISTINCT array-aggregate of `col` with NULLs removed.

    Snowflake has no ARRAY_REMOVE, so NULLs are dropped in Python (see
    _clean_str_list) after normalizing the returned array.
    """
    return f"ARRAY_AGG(DISTINCT {col}) WITHIN GROUP (ORDER BY {col})"


def _string_agg(col, sep):
    """Comma-style aggregate of `col` ordered by `col`."""
    return f"LISTAGG({col}, '{sep}') WITHIN GROUP (ORDER BY {col})"


def _norm_array(value):
    """Normalize an ARRAY/VARIANT column to a Python object.

    Snowflake returns VARIANT/ARRAY as a JSON string. None stays None.
    """
    if value is None or not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _clean_str_list(value):
    """Normalize an aggregated array and drop NULL/None entries."""
    return [v for v in (_norm_array(value) or []) if v is not None]


def _as_int(value):
    """Coerce a numeric aggregate (Snowflake may return Decimal) to int for JSON."""
    return int(value) if value is not None else value


def _fetch_sources_by_op(cur, S, op_id, extra_where="", extra_params=None):
    """Bulk-fetch execution sources for ALL configs of an operation in ONE query.

    Avoids the per-config N+1 pattern, which is pathologically slow against
    Snowflake (per-query warehouse/network latency).

    Returns {configuration_id: [(source_file, hf_model, exec_count,
                                 trace_run_ids, pytest_args_seen), ...]}.
    """
    cur.execute(
        f"""
        SELECT
            trcm.configuration_id,
            m.source_file,
            m.hf_model_identifier,
            SUM(trcm.execution_count) AS execution_count,
            {_agg_ordered('trcm.trace_run_id')} AS trace_run_ids,
            {_agg_distinct_nonnull('tr.pytest_args')} AS pytest_args_seen
        FROM {S}.trace_run_configuration_model trcm
        JOIN {S}.ttnn_configuration c ON c.ttnn_configuration_id = trcm.configuration_id
        JOIN {S}.ttnn_model m ON m.ttnn_model_id = trcm.model_id
        JOIN {S}.trace_run tr ON tr.trace_run_id = trcm.trace_run_id
        WHERE c.operation_id = %s
        {extra_where}
        GROUP BY trcm.configuration_id, m.source_file, m.hf_model_identifier
        ORDER BY trcm.configuration_id, m.source_file, m.hf_model_identifier
        """,
        [op_id] + (extra_params or []),
    )
    by_config = {}
    for cid, source_file, hf_model, exec_count, trace_run_ids, pytest_args_seen in cur.fetchall():
        by_config.setdefault(cid, []).append((source_file, hf_model, exec_count, trace_run_ids, pytest_args_seen))
    return by_config


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
    """Convert string array representation to a proper list."""
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


_BASE_OP_PREFIX_RE = re.compile(r"^(ttnn::experimental::|ttnn::transformer::|experimental::|ttnn::)")


def _base_operation_name(operation_name):
    """Compute base_operation_name.

    Postgres stored this as a GENERATED ALWAYS column; Snowflake has no generated
    columns, so the loader computes and inserts it (same regex as the schema).
    """
    if operation_name is None:
        return None
    return _BASE_OP_PREFIX_RE.sub("", operation_name)


def get_or_create_hardware(cur, hardware_cache, board_type, device_series, card_count, schema=DEFAULT_SCHEMA):
    """Get or create a hardware entry, return (hardware_id, hw_key)."""
    _validate_schema(schema)
    # Normalize device_series (can be string or array)
    if isinstance(device_series, list):
        device_series = device_series[0] if device_series else None

    if not board_type:
        return None, None

    hw_key = (board_type, device_series, card_count)
    if hw_key not in hardware_cache:
        S = _schema_prefix(schema)
        # No ON CONFLICT/RETURNING: SELECT-then-insert (single-threaded loader).
        cur.execute(
            f"SELECT ttnn_hardware_id FROM {S}.ttnn_hardware WHERE board_type=%s AND device_series=%s AND card_count=%s",
            hw_key,
        )
        row = cur.fetchone()
        if row:
            hardware_cache[hw_key] = row[0]
        else:
            new_id = _next_id(cur, schema, "ttnn_hardware")
            cur.execute(
                f"INSERT INTO {S}.ttnn_hardware (ttnn_hardware_id, board_type, device_series, card_count, create_ts) "
                f"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP())",
                (new_id, board_type, device_series, card_count),
            )
            hardware_cache[hw_key] = new_id

    return hardware_cache.get(hw_key), hw_key


def get_or_create_mesh_config(cur, mesh_config_cache, mesh_shape, device_count, schema=DEFAULT_SCHEMA):
    """Get or create a mesh config entry, return the ID.

    ttnn_mesh_config only stores mesh_shape and device_count.
    Placement info is per-tensor and stored in the argument JSON.
    """
    _validate_schema(schema)
    if not mesh_shape:
        return None

    mesh_key = (tuple(mesh_shape), device_count)

    if mesh_key not in mesh_config_cache:
        S = _schema_prefix(schema)
        # mesh_shape is an ARRAY column: match/insert via PARSE_JSON of the
        # JSON-encoded list. ARRAY equality works in Snowflake.
        mesh_json = json.dumps(mesh_shape)
        cur.execute(
            f"SELECT ttnn_mesh_config_id FROM {S}.ttnn_mesh_config WHERE mesh_shape = PARSE_JSON(%s) AND device_count = %s",
            (mesh_json, device_count),
        )
        row = cur.fetchone()
        if row:
            mesh_config_cache[mesh_key] = row[0]
        else:
            new_id = _next_id(cur, schema, "ttnn_mesh_config")
            cur.execute(
                f"INSERT INTO {S}.ttnn_mesh_config (ttnn_mesh_config_id, mesh_shape, device_count, create_ts) "
                f"SELECT %s, PARSE_JSON(%s), %s, CURRENT_TIMESTAMP()",
                (new_id, mesh_json, device_count),
            )
            mesh_config_cache[mesh_key] = new_id

    return mesh_config_cache.get(mesh_key)


def _disambiguate_model_name(cur, base_name, schema=DEFAULT_SCHEMA):
    """Find the next available model_name by appending _2, _3, etc."""
    _validate_schema(schema)
    S = _schema_prefix(schema)
    cur.execute(
        f"SELECT model_name FROM {S}.ttnn_model WHERE model_name LIKE %s",
        (f"{base_name}%",),
    )
    existing = {row[0] for row in cur.fetchall()}
    suffix = 2
    while f"{base_name}_{suffix}" in existing:
        suffix += 1
    return f"{base_name}_{suffix}"


def get_or_create_model(cur, model_cache, source_file, hf_model, schema=DEFAULT_SCHEMA):
    """Get or create a model entry, return the ID."""
    _validate_schema(schema)
    if not source_file:
        return None

    model_key = (source_file, hf_model)
    if model_key not in model_cache:
        model_family = extract_model_family(source_file, hf_model)
        model_name = derive_model_name(source_file, hf_model)
        S = _schema_prefix(schema)
        # Look up existing row first. Snowflake has no enforced UNIQUE, and a
        # NULL hf_model_identifier must be matched explicitly (NULL != NULL).
        cur.execute(
            f"""
            SELECT ttnn_model_id FROM {S}.ttnn_model
            WHERE source_file = %s
              AND (hf_model_identifier = %s OR (hf_model_identifier IS NULL AND %s IS NULL))
            """,
            (source_file, hf_model, hf_model),
        )
        row = cur.fetchone()
        if row:
            # Only set model_name when it's NULL to preserve set-model-name overrides.
            cur.execute(
                f"UPDATE {S}.ttnn_model SET model_name = COALESCE(model_name, %s), update_ts = CURRENT_TIMESTAMP() WHERE ttnn_model_id = %s",
                (model_name, row[0]),
            )
            model_cache[model_key] = row[0]
        else:
            # Snowflake won't raise on a duplicate model_name (no enforced UNIQUE),
            # so proactively pick a free name BEFORE inserting.
            insert_name = model_name
            if model_name is not None:
                cur.execute(
                    f"SELECT 1 FROM {S}.ttnn_model WHERE model_name = %s",
                    (model_name,),
                )
                if cur.fetchone():
                    insert_name = _disambiguate_model_name(cur, model_name, schema=schema)
                    print(
                        f"  Warning: model_name '{model_name}' already taken, "
                        f"using '{insert_name}' instead. "
                        f'Rename later with: set-model-name --source-file "{source_file}" --model-name <name>'
                    )
            new_id = _next_id(cur, schema, "ttnn_model")
            cur.execute(
                f"""
                INSERT INTO {S}.ttnn_model
                    (ttnn_model_id, source_file, hf_model_identifier, model_family, model_name, create_ts, update_ts)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
                """,
                (new_id, source_file, hf_model, model_family, insert_name),
            )
            model_cache[model_key] = new_id

    return model_cache[model_key]


def parse_mesh_from_machine_info(machine_info, arguments=None):
    """Extract mesh configuration from machine_info.

    Reads mesh_device_shape and device_count directly from the execution's
    machine_info block. Callers such as load_data() may require that
    machine_info.device_count be explicitly present and consistent with the
    mesh shape.

    Returns: (mesh_shape, device_count, placement_type, shard_dim, distribution_shape)
    """
    normalized_machine_info = normalize_machine_info(machine_info, arguments=arguments)
    mesh_shape = parse_array_value(normalized_machine_info.get("mesh_device_shape"))

    if not mesh_shape:
        return None, None, None, None, None

    device_count = normalized_machine_info.get("device_count")
    if device_count is None:
        device_count = 1
        for dim in mesh_shape:
            device_count *= dim

    # Per-tensor placement info (still read from arguments for placement_type)
    placement_type, shard_dim, distribution_shape = None, None, None
    if arguments:
        for arg_value in arguments.values():
            if isinstance(arg_value, dict) and arg_value.get("type") == "ttnn.Tensor":
                placement = arg_value.get("tensor_placement")
                if placement:
                    placement_type, shard_dim = parse_placement(placement.get("placement"))
                    distribution_shape = parse_array_value(placement.get("distribution_shape"))
                    break

    return mesh_shape, device_count, placement_type, shard_dim, distribution_shape


def _normalize_trace_sw_value(value):
    """Normalize trace software metadata to a storable scalar string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _parse_trace_sw_value(value):
    """Best-effort parse for software metadata read back from DB."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except (TypeError, ValueError):
            return value
    return value


def get_or_create_trace_run(
    cur,
    trace_run_cache,
    trace_uid,
    hardware_id,
    tt_metal_sha=None,
    pytest_args=None,
    tt_kmd=None,
    tt_smi=None,
    tt_firmware=None,
    schema=DEFAULT_SCHEMA,
):
    """Create and cache a trace_run for a unique trace_uid.

    pytest_args is the verbatim CLI args (everything after `--`) used to
    invoke generic_ops_tracer.py for this trace. Stored on trace_run so we
    can answer "have I traced model X with these args on hardware Y?"
    purely via SQL.
    """
    _validate_schema(schema)
    if trace_uid not in trace_run_cache:
        S = _schema_prefix(schema)
        new_id = _next_id(cur, schema, "trace_run")
        cur.execute(
            f"""
            INSERT INTO {S}.trace_run
                (trace_run_id, trace_uid, hardware_id, tt_metal_sha, pytest_args, tt_kmd, tt_smi, tt_firmware, traced_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """,
            (new_id, trace_uid, hardware_id, tt_metal_sha, pytest_args, tt_kmd, tt_smi, tt_firmware),
        )
        trace_run_cache[trace_uid] = new_id

    return trace_run_cache[trace_uid]


def link_trace_run_configuration_model(cur, trace_run_id, config_id, model_id, execution_count, schema=DEFAULT_SCHEMA):
    """Insert canonical per-trace per-config per-model execution counts."""
    _validate_schema(schema)
    S = _schema_prefix(schema)
    # No ON CONFLICT: SELECT existing count, then additive UPDATE or INSERT.
    cur.execute(
        f"""
        SELECT execution_count FROM {S}.trace_run_configuration_model
        WHERE trace_run_id = %s AND configuration_id = %s AND model_id = %s
        """,
        (trace_run_id, config_id, model_id),
    )
    row = cur.fetchone()
    if row is not None:
        cur.execute(
            f"""
            UPDATE {S}.trace_run_configuration_model
            SET last_seen_ts = CURRENT_TIMESTAMP(),
                execution_count = execution_count + %s
            WHERE trace_run_id = %s AND configuration_id = %s AND model_id = %s
            """,
            (execution_count, trace_run_id, config_id, model_id),
        )
    else:
        cur.execute(
            f"""
            INSERT INTO {S}.trace_run_configuration_model
                (trace_run_id, configuration_id, model_id, execution_count, first_seen_ts, last_seen_ts)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """,
            (trace_run_id, config_id, model_id, execution_count),
        )


def _alloc_ids_bulk(cur, schema, table, count):
    """Allocate `count` ids from the Snowflake sequence backing `table` in ONE round-trip.

    Uses TABLE(GENERATOR(ROWCOUNT => count)) to pull `count` NEXTVALs at once,
    avoiding the per-id round-trip of _next_id. Returns a list of ints (may be
    non-contiguous, but always unique and monotonic). Returns [] for count <= 0.
    """
    if count <= 0:
        return []
    seq = _seq_name(schema, table)
    # NEXTVAL inside a GENERATOR row set yields `count` fresh sequence values.
    cur.execute(f"SELECT {seq}.NEXTVAL FROM TABLE(GENERATOR(ROWCOUNT => {int(count)}))")
    return [int(r[0]) for r in cur.fetchall()]


# Row batch size for multi-row INSERT/MERGE. full_config_json can be large text,
# so keep this modest to stay well under Snowflake's bind-parameter limits.
_CONFIG_INSERT_BATCH = 500
_LINK_MERGE_BATCH = 1000


def _bulk_insert_configurations(cur, schema, new_config_rows):
    """Bulk multi-row INSERT of brand-new ttnn_configuration rows.

    new_config_rows: list of
        (config_id, operation_id, hardware_id, mesh_config_id, config_hash,
         full_config_json, status)
    tuples. Timestamps default to CURRENT_TIMESTAMP(). Inserts in batches of
    _CONFIG_INSERT_BATCH rows per statement.
    """
    if not new_config_rows:
        return
    S = _schema_prefix(schema)
    cols = (
        "ttnn_configuration_id, operation_id, hardware_id, mesh_config_id, "
        "config_hash, full_config_json, status, first_seen_ts, last_seen_ts"
    )
    # 7 bound values per row + literal CURRENT_TIMESTAMP() for the two ts cols.
    row_tmpl = "(%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())"
    for start in range(0, len(new_config_rows), _CONFIG_INSERT_BATCH):
        chunk = new_config_rows[start : start + _CONFIG_INSERT_BATCH]
        values_clause = ", ".join([row_tmpl] * len(chunk))
        params = []
        for cfg_id, op_id, hw_id, mesh_id, cfg_hash, full_json, status in chunk:
            params.extend([cfg_id, op_id, hw_id, mesh_id, cfg_hash, full_json, status])
        cur.execute(
            f"INSERT INTO {S}.ttnn_configuration ({cols}) VALUES {values_clause}",
            params,
        )


def _bulk_merge_trace_links(cur, schema, link_rows):
    """Bulk MERGE trace_run_configuration_model rows (additive execution_count).

    link_rows: list of (trace_run_id, configuration_id, model_id, execution_count).
    Duplicates on (trace_run_id, configuration_id, model_id) must already be
    pre-aggregated by the caller (summed execution_count) so each source key is
    unique within a MERGE batch (Snowflake errors if a MERGE target row matches
    multiple source rows). MERGE handles both fresh (INSERT) and incremental
    (additive UPDATE) loads correctly. Batched by _LINK_MERGE_BATCH.
    """
    if not link_rows:
        return
    S = _schema_prefix(schema)
    for start in range(0, len(link_rows), _LINK_MERGE_BATCH):
        chunk = link_rows[start : start + _LINK_MERGE_BATCH]
        values_clause = ", ".join(["(%s, %s, %s, %s)"] * len(chunk))
        params = []
        for tr_id, cfg_id, model_id, exec_count in chunk:
            params.extend([tr_id, cfg_id, model_id, exec_count])
        cur.execute(
            f"""
            MERGE INTO {S}.trace_run_configuration_model t
            USING (
                SELECT column1 AS trace_run_id,
                       column2 AS configuration_id,
                       column3 AS model_id,
                       column4 AS execution_count
                FROM VALUES {values_clause}
            ) s
            ON (t.trace_run_id = s.trace_run_id
                AND t.configuration_id = s.configuration_id
                AND t.model_id = s.model_id)
            WHEN MATCHED THEN UPDATE SET
                execution_count = t.execution_count + s.execution_count,
                last_seen_ts = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT
                (trace_run_id, configuration_id, model_id, execution_count, first_seen_ts, last_seen_ts)
                VALUES (s.trace_run_id, s.configuration_id, s.model_id, s.execution_count,
                        CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """,
            params,
        )


def refresh_trace_aggregates(cur, trace_run_ids, configuration_ids, schema=DEFAULT_SCHEMA):
    """Refresh derived aggregate tables from canonical trace_run_configuration_model rows."""
    _validate_schema(schema)
    trace_run_ids = sorted(set(trace_run_ids or []))
    configuration_ids = sorted(set(configuration_ids or []))
    S = _schema_prefix(schema)

    def _in(ids):
        """(clause, params) for `col IN (%s, ..)`."""
        return "IN (" + ", ".join(["%s"] * len(ids)) + ")", list(ids)

    if trace_run_ids:
        clause, params = _in(trace_run_ids)
        cur.execute(f"DELETE FROM {S}.trace_run_config WHERE trace_run_id {clause}", params)
        cur.execute(
            f"""
            INSERT INTO {S}.trace_run_config (trace_run_id, configuration_id, execution_count)
            SELECT trace_run_id, configuration_id, SUM(execution_count)
            FROM {S}.trace_run_configuration_model
            WHERE trace_run_id {clause}
            GROUP BY trace_run_id, configuration_id
            """,
            params,
        )

        cur.execute(f"DELETE FROM {S}.trace_run_model WHERE trace_run_id {clause}", params)
        cur.execute(
            f"""
            INSERT INTO {S}.trace_run_model (trace_run_id, model_id)
            SELECT DISTINCT trace_run_id, model_id
            FROM {S}.trace_run_configuration_model
            WHERE trace_run_id {clause}
            """,
            params,
        )

        cur.execute(
            f"""
            UPDATE {S}.trace_run tr
            SET config_count = agg.config_count
            FROM (
                SELECT trace_run_id, COUNT(DISTINCT configuration_id) AS config_count
                FROM {S}.trace_run_configuration_model
                WHERE trace_run_id {clause}
                GROUP BY trace_run_id
            ) agg
            WHERE tr.trace_run_id = agg.trace_run_id
            """,
            params,
        )

    if configuration_ids:
        clause, params = _in(configuration_ids)
        cur.execute(
            f"DELETE FROM {S}.ttnn_configuration_model WHERE configuration_id {clause}",
            params,
        )
        cur.execute(
            f"""
            INSERT INTO {S}.ttnn_configuration_model
                (configuration_id, model_id, execution_count, first_seen_ts, last_seen_ts)
            SELECT
                configuration_id,
                model_id,
                SUM(execution_count),
                MIN(first_seen_ts),
                MAX(last_seen_ts)
            FROM {S}.trace_run_configuration_model
            WHERE configuration_id {clause}
            GROUP BY configuration_id, model_id
            """,
            params,
        )


def load_data(json_path=None, tt_metal_sha=None, dry_run=False, schema=DEFAULT_SCHEMA, emit_id_file=None):
    """Main loading function.

    Args:
        json_path: Path to JSON file (defaults to JSON_PATH).
        tt_metal_sha: Git SHA of tt-metal at trace time (optional).
                      If None, attempts to detect from git.
        dry_run: If True, run all logic but roll back every DB write at the end.
                 Prints the same stats so you can verify behaviour without persisting anything.
        schema: Database schema to load into (default: DEFAULT_SCHEMA).
        emit_id_file: Optional path to write the newly-created trace_run_id(s) to,
                      one integer per line. Only written on a successful (non-dry-run)
                      load that actually created trace_runs. CI pipelines consume this
                      to pin downstream validation to the traces just uploaded.
    """
    _validate_schema(schema)
    json_path = json_path or JSON_PATH
    print(f"Loading JSON from {json_path} into schema '{schema}'...")
    with open(json_path) as f:
        data = json.load(f)

    operations = data.get("operations", {})
    print(f"Found {len(operations)} operations")

    default_trace_uid = data.get("metadata", {}).get("trace_uid")
    file_trace_uids = set()
    for op_data in operations.values():
        for config in op_data.get("configurations", []):
            executions = config.get("executions", [])
            if not executions:
                continue
            for execution in executions:
                trace_uid = execution.get("trace_uid", default_trace_uid)
                if not trace_uid:
                    raise ValueError(
                        "Missing execution.trace_uid in input JSON. "
                        "Re-trace with generic_ops_tracer.py to produce v6-compatible trace metadata."
                    )
                file_trace_uids.add(trace_uid)

    conn = _get_write_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)
    # Back the NUMBER id columns with sequences (Snowflake has no autoincrement).
    _ensure_sequences(cur, schema)

    # Caches for normalized tables
    operation_cache = {}
    model_cache = {}
    hardware_cache = {}
    mesh_config_cache = {}
    trace_run_cache = {}  # trace_uid -> trace_run_id
    print("  Populating trace_run_configuration_model + derived aggregate tables")

    # Auto-detect tt_metal_sha if not provided
    if tt_metal_sha is None:
        try:
            import subprocess

            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tt_metal_sha = result.stdout.strip()
                print(f"  Detected tt-metal SHA: {tt_metal_sha[:12]}")
        except Exception as e:
            print(f"  Warning: failed to auto-detect tt-metal SHA via git: {e}")

    if file_trace_uids:
        uid_list = list(file_trace_uids)
        placeholders = ", ".join(["%s"] * len(uid_list))
        cur.execute(
            f"SELECT trace_uid FROM {S}.trace_run WHERE trace_uid IN ({placeholders})",
            uid_list,
        )
        existing_trace_uids = sorted(row[0] for row in cur.fetchall())
        if existing_trace_uids:
            conn.close()
            joined = ", ".join(existing_trace_uids)
            print(f"Trace upload aborted: the following trace_uid values already exist in {schema}.trace_run: {joined}")
            print("This trace was already uploaded, so no data population was performed.")
            return

    new_configs = 0  # configs inserted for the first time this load
    total_model_links = 0
    trace_run_ids_touched = set()
    configuration_ids_touched = set()
    trace_config_pairs = set()

    # --- Batching state (collected in Python, flushed with bulk SQL below) ---
    #
    # We keep the cheap/bounded dimension lookups (operation, model, hardware,
    # mesh, trace_run) row-by-row (Python-cached), but defer the two high-volume
    # writes -- ttnn_configuration inserts and trace_run_configuration_model
    # links -- into batched statements.
    #
    # existing_config_ids: config_hash -> ttnn_configuration_id already in the DB.
    #   Fetched once up front; on a fresh/empty schema this is empty.
    cur.execute(f"SELECT config_hash, ttnn_configuration_id FROM {S}.ttnn_configuration")
    existing_config_ids = {row[0]: int(row[1]) for row in cur.fetchall()}

    # pending_new_configs: config_hash -> (operation_id, hardware_id, mesh_config_id,
    #   full_config_json_text, status). First occurrence of a hash wins (matches
    #   the old "SELECT-then-insert once, UPDATE last_seen on repeats" semantics).
    pending_new_configs = {}
    pending_new_order = []  # config_hashes in first-seen order (stable id assignment)
    # link_agg: (trace_uid_marker) not used; links keyed by (trace_run_id,
    #   config_hash, model_id) -> summed execution_count. config_hash is resolved
    #   to configuration_id after ids are assigned. Pre-aggregating here makes the
    #   MERGE source unique per key.
    link_agg = {}

    for op_name, op_data in operations.items():
        # Insert operation
        if op_name not in operation_cache:
            cur.execute(
                f"SELECT ttnn_operation_id FROM {S}.ttnn_operation WHERE operation_name = %s",
                (op_name,),
            )
            row = cur.fetchone()
            if row:
                operation_cache[op_name] = row[0]
            else:
                new_op_id = _next_id(cur, schema, "ttnn_operation")
                cur.execute(
                    f"INSERT INTO {S}.ttnn_operation "
                    f"(ttnn_operation_id, operation_name, base_operation_name, create_ts) "
                    f"VALUES (%s, %s, %s, CURRENT_TIMESTAMP())",
                    (new_op_id, op_name, _base_operation_name(op_name)),
                )
                operation_cache[op_name] = new_op_id

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

            for exec_idx, execution in enumerate(executions):
                source = execution.get("source")
                machine_info = normalize_machine_info(execution.get("machine_info", {}), arguments=arguments)
                execution_count = execution.get("count", 1)
                trace_uid = execution.get("trace_uid", default_trace_uid)
                # pytest_args is per-invocation (1:1 with trace_uid). Older
                # JSONs won't have this field; treat absence as None so
                # trace_run.pytest_args stays NULL for legacy data.
                pytest_args = execution.get("pytest_args")
                tt_kmd = _normalize_trace_sw_value(machine_info.get("tt_kmd"))
                tt_smi = _normalize_trace_sw_value(machine_info.get("tt_smi"))
                tt_firmware = _normalize_trace_sw_value(machine_info.get("tt_firmware"))

                # Validate required fields
                missing = []
                if not source:
                    missing.append("execution.source")
                if not trace_uid:
                    missing.append("execution.trace_uid")
                if not machine_info.get("board_type"):
                    missing.append("machine_info.board_type")
                if not machine_info.get("device_series"):
                    missing.append("machine_info.device_series")
                if machine_info.get("card_count") is None:
                    missing.append("machine_info.card_count")
                if not machine_info.get("mesh_device_shape"):
                    missing.append("machine_info.mesh_device_shape")
                if machine_info.get("device_count") is None:
                    missing.append("machine_info.device_count")
                if missing:
                    raise ValueError(
                        f"Operation '{op_name}', config_hash '{config_hash}', "
                        f"execution index {exec_idx}: missing required fields: {', '.join(missing)}. "
                        f"Re-trace with generic_ops_tracer.py to produce complete execution data."
                        f" Ensure that generic_ops_tracer.py is generating configuration according to model_tracer/README.md"
                    )

                # Parse source to get model info
                source_file, hf_model = parse_source(source)
                model_id = get_or_create_model(cur, model_cache, source_file, hf_model, schema=schema)
                # Parse hardware
                board_type = machine_info.get("board_type")
                device_series = machine_info.get("device_series")
                card_count = machine_info.get("card_count", 1)

                hardware_id, _ = get_or_create_hardware(
                    cur, hardware_cache, board_type, device_series, card_count, schema=schema
                )

                # Parse mesh config
                args_dict = arguments if isinstance(arguments, dict) else {}
                mesh_shape, device_count, _, _, _ = parse_mesh_from_machine_info(machine_info, args_dict)

                mesh_config_id = None
                if mesh_shape:
                    mesh_config_id = get_or_create_mesh_config(
                        cur, mesh_config_cache, mesh_shape, device_count, schema=schema
                    )

                # Resolve/queue the configuration. Dedup by config_hash: reuse a
                # DB-existing id, else queue a brand-new one (first occurrence
                # captures operation/hardware/mesh/full_config_json). The old code
                # bumped last_seen_ts on repeats; on a fresh insert last_seen_ts is
                # already CURRENT_TIMESTAMP(), so the repeat-UPDATE is a semantic
                # no-op we drop.
                config_id = existing_config_ids.get(config_hash)
                if config_id is None and config_hash not in pending_new_configs:
                    pending_new_configs[config_hash] = (
                        op_id,
                        hardware_id,
                        mesh_config_id,
                        json.dumps(config),
                        "observed",
                    )
                    pending_new_order.append(config_hash)
                    new_configs += 1

                # Link this execution canonically via trace + config + model
                if model_id is not None:
                    if hardware_id:
                        trace_run_kwargs = {"pytest_args": pytest_args, "schema": schema}
                        if tt_kmd is not None or tt_smi is not None or tt_firmware is not None:
                            trace_run_kwargs.update(
                                {
                                    "tt_kmd": tt_kmd,
                                    "tt_smi": tt_smi,
                                    "tt_firmware": tt_firmware,
                                }
                            )
                        trace_run_id = get_or_create_trace_run(
                            cur,
                            trace_run_cache,
                            trace_uid,
                            hardware_id,
                            tt_metal_sha,
                            **trace_run_kwargs,
                        )
                        trace_run_ids_touched.add(trace_run_id)
                        # Aggregate additively per (trace_run, config_hash, model);
                        # config_hash is resolved to configuration_id after ids are
                        # assigned below.
                        link_key = (trace_run_id, config_hash, model_id)
                        link_agg[link_key] = link_agg.get(link_key, 0) + execution_count
                        total_model_links += 1

        print(f"  Loaded {op_name}: {len(op_data.get('configurations', []))} JSON configs")

    # --- Bulk flush: allocate ids for new configs, insert, then MERGE links ---
    # 1) Allocate ids for all new configs in ONE round-trip and bulk-insert them.
    new_ids = _alloc_ids_bulk(cur, schema, "ttnn_configuration", len(pending_new_order))
    config_id_by_hash = dict(existing_config_ids)
    new_config_rows = []
    for cfg_hash, cfg_id in zip(pending_new_order, new_ids):
        op_id_v, hw_id_v, mesh_id_v, full_json_v, status_v = pending_new_configs[cfg_hash]
        config_id_by_hash[cfg_hash] = cfg_id
        new_config_rows.append((cfg_id, op_id_v, hw_id_v, mesh_id_v, cfg_hash, full_json_v, status_v))
    _bulk_insert_configurations(cur, schema, new_config_rows)

    # 2) Resolve link config_hashes -> configuration_ids and MERGE in batches.
    link_rows = []
    for (trace_run_id, cfg_hash, model_id), exec_count in link_agg.items():
        cfg_id = config_id_by_hash[cfg_hash]
        configuration_ids_touched.add(cfg_id)
        trace_config_pairs.add((trace_run_id, cfg_id))
        link_rows.append((trace_run_id, cfg_id, model_id, exec_count))
    _bulk_merge_trace_links(cur, schema, link_rows)

    if trace_run_ids_touched or configuration_ids_touched:
        refresh_trace_aggregates(cur, trace_run_ids_touched, configuration_ids_touched, schema=schema)

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
            ("trace_run_configuration_model", "Trace-Config-Model links"),
            ("trace_run_config", "Trace-Config links"),
            ("trace_run_model", "Trace-Model links"),
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {S}.{table}")
            counts[label] = cur.fetchone()[0]
        return counts

    if dry_run:
        post_load_counts = _fetch_db_totals(cur)
        conn.rollback()
        print("\n🔍 DRY RUN — all DB writes rolled back.")
    else:
        conn.commit()

    print(
        f"\n{'Would load' if dry_run else '✅ Loaded'} {len(trace_config_pairs)} trace/config links "
        f"({new_configs} new configs), {total_model_links} canonical trace/config/model links"
    )
    print(
        f"   Trace runs {'that would be ' if dry_run else ''}created: {len(trace_run_cache)}, "
        f"total configs tied to traces: {len(trace_config_pairs)}"
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

    if emit_id_file and trace_run_cache:
        new_ids = sorted(set(trace_run_cache.values()))
        emit_path = Path(emit_id_file).resolve()
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        emit_path.write_text("\n".join(str(i) for i in new_ids) + "\n")
        print(f"  Wrote {len(new_ids)} trace_run_id(s) to {emit_path}")

    # Auto-append draft entries to manifest registry
    if trace_run_cache and yaml is not None:
        _append_manifest_drafts(trace_run_cache, schema=schema)


def _append_manifest_drafts(trace_run_cache, schema=DEFAULT_SCHEMA):
    """Append draft registry entries to the manifest for newly created trace_runs."""
    _validate_schema(schema)
    try:
        data, path = _load_manifest()
    except Exception as e:
        print(f"  Warning: could not update manifest registry: {e}")
        return

    # Fetch hardware details and per-trace model names from DB
    S = _schema_prefix(schema)
    try:
        conn = _get_write_connection()
        cur = conn.cursor()
        hw_map = {}
        trace_models_map = {}  # trace_run_id -> sorted list of model_names
        trace_details_map = {}  # trace_run_id -> (hardware_id, sha, trace_uid)
        for trace_run_id in trace_run_cache.values():
            cur.execute(
                f"SELECT hardware_id, tt_metal_sha, trace_uid, pytest_args, tt_kmd, tt_smi, tt_firmware"
                f" FROM {S}.trace_run WHERE trace_run_id = %s",
                (trace_run_id,),
            )
            tr_row = cur.fetchone()
            if not tr_row:
                continue
            hardware_id, sha, trace_uid, pytest_args, tt_kmd, tt_smi, tt_firmware = tr_row
            hardware_id = _as_int(hardware_id)
            trace_details_map[trace_run_id] = (hardware_id, sha, trace_uid, pytest_args, tt_kmd, tt_smi, tt_firmware)
            if hardware_id and hardware_id not in hw_map:
                cur.execute(
                    f"SELECT board_type, device_series, card_count FROM {S}.ttnn_hardware WHERE ttnn_hardware_id = %s",
                    (hardware_id,),
                )
                row = cur.fetchone()
                if row:
                    hw_map[hardware_id] = (row[0], row[1], _as_int(row[2]))
            cur.execute(
                f"""
                SELECT m.model_name FROM {S}.trace_run_model trm
                JOIN {S}.ttnn_model m ON m.ttnn_model_id = trm.model_id
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
        trace_details_map = {}

    existing_ids = {entry.get("trace_id") for entry in data["registry"]}
    existing_trace_uids = {entry.get("trace_uid") for entry in data["registry"] if entry.get("trace_uid")}
    added = 0

    for trace_run_id in trace_run_cache.values():
        tr_row = trace_details_map.get(trace_run_id)
        if not tr_row:
            continue
        hardware_id, sha, trace_uid, pytest_args, tt_kmd, tt_smi, tt_firmware = tr_row
        if trace_uid and trace_uid in existing_trace_uids:
            continue
        if not trace_uid and trace_run_id in existing_ids:
            continue
        if trace_uid and trace_run_id in existing_ids:
            print(
                f"  Note: trace_run_id {trace_run_id} already exists in manifest registry, "
                f"but trace_uid {trace_uid} is new; appending a new draft entry."
            )

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
            "trace_uid": trace_uid,
            "tt_metal_sha": sha,
            "config_count": None,  # updated after commit
            "loaded_at": str(date.today()),
            "notes": "",
            "pytest_args": pytest_args,
            "tt_kmd": tt_kmd,
            "tt_smi": tt_smi,
            "tt_firmware": _parse_trace_sw_value(tt_firmware),
        }
        data["registry"].append(entry)
        existing_ids.add(trace_run_id)
        if trace_uid:
            existing_trace_uids.add(trace_uid)
        added += 1

    if added:
        new_entries = data["registry"][-added:]
        try:
            conn = _get_write_connection()
            cur = conn.cursor()
            for entry in new_entries:
                if entry.get("config_count") is None:
                    cur.execute(
                        f"SELECT COUNT(*) FROM {S}.trace_run_config WHERE trace_run_id = %s",
                        (entry["trace_id"],),
                    )
                    entry["config_count"] = _as_int(cur.fetchone()[0])
            conn.close()
        except Exception as e:
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


def _merge_trace_run_ids(existing_ids, incoming_ids):
    """Merge two trace_run_ids collections deterministically."""
    return sorted(set(existing_ids or []) | set(incoming_ids or []))


def _machine_info_key(machine_info):
    """Stable key for merging execution buckets with identical machine info."""
    return json.dumps(machine_info or {}, sort_keys=True, default=str)


def _merge_execution_lists(existing_executions, incoming_executions):
    """Merge execution buckets by (source, machine_info), summing counts and provenance.

    pytest_args is per-trace, so when multiple traces collapse into one bucket
    here we accumulate them into pytest_args_seen (a sorted, deduplicated list).
    Single-string pytest_args from individual reconstructions also fold into
    that list so callers always see a uniform shape across merged outputs.
    """
    merged = {}
    for execution in list(existing_executions or []) + list(incoming_executions or []):
        source = execution.get("source")
        machine_info = execution.get("machine_info", {})
        key = (source, _machine_info_key(machine_info))
        if key not in merged:
            merged[key] = {
                "source": source,
                "machine_info": machine_info,
                "count": 0,
                "trace_run_ids": [],
                "pytest_args_seen": [],
            }
        merged[key]["count"] += execution.get("count", 0)
        merged[key]["trace_run_ids"] = _merge_trace_run_ids(
            merged[key]["trace_run_ids"], execution.get("trace_run_ids")
        )

        incoming_pytest_args = []
        if execution.get("pytest_args"):
            incoming_pytest_args.append(execution["pytest_args"])
        if execution.get("pytest_args_seen"):
            incoming_pytest_args.extend(execution["pytest_args_seen"])
        if incoming_pytest_args:
            merged[key]["pytest_args_seen"] = sorted(set(merged[key]["pytest_args_seen"]) | set(incoming_pytest_args))
    return list(merged.values())


def _validate_schema(schema):
    """Validate schema name to prevent SQL injection (used in f-string queries)."""
    if not re.match(r"^[a-zA-Z_]\w*$", schema):
        raise ValueError(f"Invalid schema name: {schema!r}")


def reconstruct_from_db(output_path=None, schema=DEFAULT_SCHEMA, model_filter=None):
    """Reconstruct ttnn_operations_master.json from the database.

    Args:
        output_path: Path to write the reconstructed JSON
        schema: Database schema to use (default: "ttnn_ops_v6")
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
    conn = _get_read_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)

    if model_filter:
        # Only fetch operations that have at least one config linked to a matching model
        like_clauses = " OR ".join(["m.source_file ILIKE %s"] * len(model_filter))
        like_params = [f"%{pattern}%" for pattern in model_filter]
        cur.execute(
            f"""
            SELECT DISTINCT o.ttnn_operation_id, o.operation_name
            FROM {S}.ttnn_operation o
            JOIN {S}.ttnn_configuration c ON c.operation_id = o.ttnn_operation_id
            JOIN {S}.ttnn_configuration_model cm ON cm.configuration_id = c.ttnn_configuration_id
            JOIN {S}.ttnn_model m ON m.ttnn_model_id = cm.model_id
            WHERE {like_clauses}
            ORDER BY o.operation_name
        """,
            like_params,
        )
    else:
        cur.execute(
            f"""
            SELECT ttnn_operation_id, operation_name
            FROM {S}.ttnn_operation
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
                SELECT 1 FROM {S}.ttnn_configuration_model cm2
                JOIN {S}.ttnn_model m2 ON m2.ttnn_model_id = cm2.model_id
                WHERE cm2.configuration_id = c.ttnn_configuration_id
                AND ({like_clauses})
            )"""
        model_filter_params = [f"%{pattern}%" for pattern in model_filter]

    # Source-side model filter (applied when bulk-fetching per-op execution sources).
    source_filter_clause = ""
    source_filter_params = []
    if model_filter:
        like_clauses = " OR ".join(["m.source_file ILIKE %s"] * len(model_filter))
        source_filter_clause = f" AND ({like_clauses})"
        source_filter_params = [f"%{pattern}%" for pattern in model_filter]

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
            FROM {S}.ttnn_configuration c
            LEFT JOIN {S}.ttnn_hardware h ON h.ttnn_hardware_id = c.hardware_id
            LEFT JOIN {S}.ttnn_mesh_config mc ON mc.ttnn_mesh_config_id = c.mesh_config_id
            WHERE c.operation_id = %s
            {model_filter_clause}
            ORDER BY c.config_hash
        """,
            [op_id] + model_filter_params,
        )
        configs = cur.fetchall()

        if not configs:
            continue

        # Bulk-fetch sources for all configs of this op in one query (avoids N+1).
        sources_by_cfg = _fetch_sources_by_op(cur, S, op_id, source_filter_clause, source_filter_params)

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

            mesh_shape = _norm_array(mesh_shape)

            source_rows = sources_by_cfg.get(config_id, [])

            # Use full_config_json as the base config (preserves all original fields)
            # Snowflake returns VARIANT/ARRAY as a JSON string
            if full_config_json:
                if isinstance(full_config_json, str):
                    try:
                        full_config_json = json.loads(full_config_json)
                    except (TypeError, ValueError):
                        full_config_json = {}
            else:
                full_config_json = {}

            config_dict = dict(full_config_json)
            config_dict.pop("executions", None)
            config_dict.pop("source", None)
            config_dict.pop("machine_info", None)
            config_dict["config_hash"] = config_hash

            executions = []
            for source_file, hf_model, exec_count, trace_run_ids, pytest_args_seen in source_rows:
                source_str = format_source(source_file, hf_model)
                if not source_str:
                    continue

                execution = {
                    "source": source_str,
                    "machine_info": {},
                    "count": _as_int(exec_count),
                    "trace_run_ids": list(_norm_array(trace_run_ids) or []),
                    # Aggregated view: a (config, model) pair may have been
                    # produced under several pytest_args variants. Surface
                    # them all for inspection. For a lossless single-trace
                    # round-trip use reconstruct_from_trace_run instead.
                    "pytest_args_seen": _clean_str_list(pytest_args_seen),
                }

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
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Saved to {output_path}")

    return result


def reconstruct_from_trace_run(trace_run_id, output_path=None, schema=DEFAULT_SCHEMA, model_names=None):
    """Reconstruct JSON for a specific trace_run, matching the tracer's output format.

    This produces output identical to what generic_ops_tracer.py would generate
    for a fresh trace of the same model on the same hardware.

    Args:
        trace_run_id: ID of the trace_run to reconstruct.
        output_path: Path to write the reconstructed JSON.
        schema: Database schema to read from (default: "ttnn_ops_v6").
        model_names: Optional set/list of model_name values to include. When provided,
            only configs belonging to those models are reconstructed. None = all models.

    Returns:
        Dict in the tracer's master JSON format.
    """
    _validate_schema(schema)
    filter_desc = f", models={sorted(model_names)}" if model_names is not None else ""
    print(f"Reconstructing JSON from trace_run {trace_run_id} (schema: {schema}{filter_desc})...")
    conn = _get_read_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)

    # Fetch trace_run metadata (v5: hardware via FK, no model_id)
    cur.execute(
        f"""
        SELECT
            tr.trace_run_id,
            h.board_type, h.device_series, h.card_count,
            tr.tt_metal_sha, tr.traced_at, tr.config_count, tr.notes,
            tr.pytest_args, tr.tt_kmd, tr.tt_smi, tr.tt_firmware
        FROM {S}.trace_run tr
        JOIN {S}.ttnn_hardware h ON h.ttnn_hardware_id = tr.hardware_id
        WHERE tr.trace_run_id = %s
        """,
        (trace_run_id,),
    )
    tr_row = cur.fetchone()
    if not tr_row:
        print(f"Trace run {trace_run_id} not found")
        conn.close()
        return None

    (
        _,
        board_type,
        device_series,
        card_count,
        tt_metal_sha,
        traced_at,
        _,
        _,
        trace_pytest_args,
        trace_tt_kmd,
        trace_tt_smi,
        trace_tt_firmware,
    ) = tr_row

    # Fetch all models for this trace; then apply model_names filter if specified.
    cur.execute(
        f"""
        SELECT m.ttnn_model_id, m.source_file, m.hf_model_identifier, m.model_name
        FROM {S}.trace_run_model trm
        JOIN {S}.ttnn_model m ON m.ttnn_model_id = trm.model_id
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

    # Fetch all configurations with their per-config hardware and model sources.
    # Hardware comes from each config's own hardware_id FK (not the trace_run's),
    # so configs with different hardware within the same trace are reconstructed correctly.
    # (IN (...) membership test for Snowflake.)
    _model_ids = list(trace_model_ids)
    _model_in = ", ".join(["%s"] * len(_model_ids)) or "NULL"
    cur.execute(
        f"""
        SELECT
            o.operation_name,
            c.ttnn_configuration_id,
            c.config_hash,
            c.full_config_json,
            h.board_type,
            h.device_series,
            h.card_count,
            mc.mesh_shape,
            mc.device_count,
            trcm.execution_count,
            m.source_file,
            m.hf_model_identifier
        FROM {S}.trace_run_configuration_model trcm
        JOIN {S}.ttnn_configuration c
            ON c.ttnn_configuration_id = trcm.configuration_id
        JOIN {S}.ttnn_operation o
            ON o.ttnn_operation_id = c.operation_id
        LEFT JOIN {S}.ttnn_hardware h
            ON h.ttnn_hardware_id = c.hardware_id
        LEFT JOIN {S}.ttnn_mesh_config mc
            ON mc.ttnn_mesh_config_id = c.mesh_config_id
        JOIN {S}.ttnn_model m
            ON m.ttnn_model_id = trcm.model_id
           AND m.ttnn_model_id IN ({_model_in})
        WHERE trcm.trace_run_id = %s
        ORDER BY o.operation_name, c.config_hash, m.source_file, m.hf_model_identifier
        """,
        _model_ids + [trace_run_id],
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
        cfg_board_type,
        cfg_device_series,
        cfg_card_count,
        mesh_shape,
        device_count,
        execution_count,
        source_file,
        hf_model_identifier,
    ) in rows:
        # Snowflake returns VARIANT/ARRAY as a JSON string
        if isinstance(full_config_json, str):
            try:
                full_config_json = json.loads(full_config_json)
            except (TypeError, ValueError):
                full_config_json = {}
        if not full_config_json:
            full_config_json = {}

        mesh_shape = _norm_array(mesh_shape)

        exec_machine_info = {}
        if cfg_board_type:
            exec_machine_info["board_type"] = cfg_board_type
            exec_machine_info["device_series"] = cfg_device_series
            exec_machine_info["card_count"] = cfg_card_count
        if mesh_shape:
            exec_machine_info["mesh_device_shape"] = mesh_shape
            if device_count:
                exec_machine_info["device_count"] = device_count
        if trace_tt_kmd:
            exec_machine_info["tt_kmd"] = trace_tt_kmd
        if trace_tt_smi:
            exec_machine_info["tt_smi"] = trace_tt_smi
        if trace_tt_firmware:
            exec_machine_info["tt_firmware"] = _parse_trace_sw_value(trace_tt_firmware)

        source = format_source(source_file, hf_model_identifier)

        if op_name not in ops:
            ops[op_name] = {}

        if config_id not in ops[op_name]:
            config_dict = dict(full_config_json)
            config_dict.pop("executions", None)
            config_dict.pop("source", None)
            config_dict.pop("machine_info", None)
            config_dict["config_hash"] = config_hash
            config_dict["executions"] = []
            ops[op_name][config_id] = config_dict

        ops[op_name][config_id]["executions"].append(
            {
                "source": source,
                "machine_info": exec_machine_info,
                "count": _as_int(execution_count),
                "trace_run_ids": [trace_run_id],
                "pytest_args": trace_pytest_args,
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
        "pytest_args": trace_pytest_args,
        "tt_kmd": trace_tt_kmd,
        "tt_smi": trace_tt_smi,
        "tt_firmware": _parse_trace_sw_value(trace_tt_firmware),
    }

    print(f"Reconstructed {len(result['operations'])} operations, {total_configs} configurations")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Saved to {output_path}")

    return result


def list_variants(model_pattern, schema=DEFAULT_SCHEMA):
    """List every (model, pytest_args, hardware) combination traced for a model.

    Answers "did I trace model X with these pytest args on this hardware?".
    Each row is one trace_run, so repeated runs of the same variant appear
    as separate rows (with different traced_at timestamps).

    Args:
        model_pattern: ILIKE pattern matched against ttnn_model.source_file,
            e.g. "%flux1%". The leading/trailing wildcards are added if absent.
        schema: Database schema to query (default: DEFAULT_SCHEMA).
    """
    _validate_schema(schema)
    if not model_pattern:
        print("Error: provide a source_file pattern (e.g. 'flux1' or '%flux1%')")
        return []

    if "%" not in model_pattern:
        model_pattern = f"%{model_pattern}%"

    conn = _get_read_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)
    cur.execute(
        f"""
        SELECT
            m.source_file,
            COALESCE(m.hf_model_identifier, '') AS hf_model,
            COALESCE(tr.pytest_args, '') AS pytest_args,
            h.board_type,
            h.device_series,
            h.card_count,
            tr.tt_kmd,
            tr.tt_smi,
            tr.tt_firmware,
            tr.trace_uid,
            tr.trace_run_id,
            tr.traced_at,
            tr.config_count
        FROM {S}.trace_run tr
        JOIN {S}.trace_run_model trm ON trm.trace_run_id = tr.trace_run_id
        JOIN {S}.ttnn_model m ON m.ttnn_model_id = trm.model_id
        JOIN {S}.ttnn_hardware h ON h.ttnn_hardware_id = tr.hardware_id
        WHERE m.source_file ILIKE %s
        ORDER BY m.source_file, tr.pytest_args NULLS FIRST, h.device_series, tr.traced_at DESC
        """,
        (model_pattern,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"No traces match source_file ILIKE {model_pattern!r}")
        return []

    print(f"\nVariants matching {model_pattern!r}:")
    print(f"{'trace_id':>8}  {'hardware':22}  {'configs':>7}  {'traced_at':20}  source_file / pytest_args")
    print("-" * 140)
    for sf, hf, pa, board, ds, cc, tt_kmd, tt_smi, tt_firmware, _uid, tr_id, traced_at, cfg_count in rows:
        hw = f"{ds} ({cc}x)"
        traced = str(traced_at)[:19] if traced_at else "-"
        label = sf if not hf else f"{sf} [HF_MODEL:{hf}]"
        pa_disp = pa if pa else "(no pytest_args recorded)"
        print(f"{tr_id:>8}  {hw:22}  {cfg_count or 0:>7}  {traced:20}  {label}")
        print(f"{'':>8}  {'':22}  {'':>7}  {'':20}    args: {pa_disp}")
        if tt_kmd or tt_smi or tt_firmware:
            print(
                f"{'':>8}  {'':22}  {'':>7}  {'':20}    "
                f"sw: kmd={tt_kmd or '-'} smi={tt_smi or '-'} fw={tt_firmware or '-'}"
            )

    return rows


def list_trace_runs(model_filter=None, schema=DEFAULT_SCHEMA):
    """List all trace runs, optionally filtered by model.

    Args:
        model_filter: List of model patterns to filter by (e.g., ["deepseek_v3"]).
        schema: Database schema to query (default: DEFAULT_SCHEMA).
    """
    _validate_schema(schema)
    conn = _get_read_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)

    query = f"""
        SELECT
            tr.trace_run_id,
            h.device_series,
            h.card_count,
            tr.tt_metal_sha,
            tr.traced_at,
            tr.config_count,
            tr.notes,
            tr.tt_kmd,
            tr.tt_smi,
            tr.tt_firmware,
            {_string_agg("COALESCE(m.model_name, m.source_file)", ", ")} AS models
        FROM {S}.trace_run tr
        JOIN {S}.ttnn_hardware h ON h.ttnn_hardware_id = tr.hardware_id
        LEFT JOIN {S}.trace_run_model trm ON trm.trace_run_id = tr.trace_run_id
        LEFT JOIN {S}.ttnn_model m ON m.ttnn_model_id = trm.model_id
    """
    params = []

    if model_filter:
        like_clauses = " OR ".join(["m.source_file ILIKE %s"] * len(model_filter))
        query += f" WHERE ({like_clauses})"
        params = [f"%{p}%" for p in model_filter]

    query += (
        " GROUP BY tr.trace_run_id, h.device_series, h.card_count, tr.tt_metal_sha,"
        " tr.traced_at, tr.config_count, tr.notes, tr.tt_kmd, tr.tt_smi, tr.tt_firmware"
        " ORDER BY tr.traced_at DESC"
    )
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No trace runs found")
        return []

    print(f"\n{'ID':>4}  {'Hardware':20}  {'SHA':12}  {'Configs':>7}  {'Traced At':20}  {'SW':46}  Models")
    print("-" * 200)

    for tr_id, device_series, card_count, sha, traced_at, cfg_count, _, tt_kmd, tt_smi, tt_firmware, models in rows:
        hw = f"{device_series} ({card_count}x)"
        sha_short = sha[:12] if sha else "-"
        traced = str(traced_at)[:19] if traced_at else "-"
        sw_summary = f"kmd={tt_kmd or '-'} smi={tt_smi or '-'} fw={tt_firmware or '-'}"
        print(f"{tr_id:>4}  {hw:20}  {sha_short:12}  {cfg_count or 0:>7}  {traced:20}  {sw_summary:46}  {models or ''}")

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
        manifest_path: Path to trace_selection_registry.yaml (optional).
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


def _resolve_manifest_with_models(manifest_path=None, scope=None, model_filter=None):
    """Like resolve_manifest but returns {trace_id: set_of_model_names | None}.

    None means no filter (all models in the trace).
    A set means only reconstruct configs belonging to those model_names.

    Args:
        model_filter: Optional list of model names to restrict to.  When set,
            only targets whose model list intersects with *model_filter* are
            resolved, and only the intersecting model names are passed through.
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

    # Build a lookup of which models each trace contains (from registry metadata).
    # Used to skip pinned traces that don't contain filtered models.
    registry_models_by_tid = {}
    for r in registry:
        registry_models_by_tid[r.get("trace_id")] = set(r.get("models", []))

    for entries in groups.values():
        for entry in entries or []:
            model_val = entry.get("model")
            pinned_trace = entry.get("trace")
            hw_filter = entry.get("hardware")

            models = [model_val] if isinstance(model_val, str) else (list(model_val) if model_val else [])

            # Apply CLI model filter: only keep models that match the filter
            if model_filter:
                models = [m for m in models if m in model_filter]
                if not models:
                    continue

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
                    if model_filter:
                        # Only add this trace if the registry confirms it contains at least
                        # one of the filtered models.  This avoids hitting the DB for traces
                        # that clearly don't have the requested model.
                        registry_models_for_tid = registry_models_by_tid.get(tid, set())
                        trace_models = [m for m in models if m in registry_models_for_tid]
                        if trace_models:
                            _add(tid, trace_models)
                    else:
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


def reconstruct_from_manifest(
    manifest_path=None, output_path=None, scope=None, schema=DEFAULT_SCHEMA, model_filter=None
):
    """Reconstruct merged JSON from manifest targets.

    Resolves targets to (trace_run_id, model_names) pairs, reconstructs each
    filtered to the specified models, then merges configs deduplicated by config_hash.

    Args:
        manifest_path: Path to trace_selection_registry.yaml (optional).
        output_path: Path to write the merged JSON (optional).
        scope: 'lead_models' or 'model_traced' (None = all targets).
        schema: Database schema to read from (default: "ttnn_ops_v6").
        model_filter: Optional list of model names to restrict reconstruction to.

    Returns:
        Merged dict in the tracer's master JSON format.
    """
    trace_model_map = _resolve_manifest_with_models(manifest_path, scope=scope, model_filter=model_filter)
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

    seen_hashes = {}  # config_hash -> merged config dict (global dedup across all traces)
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
                    seen_hashes[ch]["executions"] = _merge_execution_lists(
                        seen_hashes[ch].get("executions"),
                        config.get("executions"),
                    )
                    total_deduped += 1
                    continue
                if ch:
                    config["executions"] = _merge_execution_lists([], config.get("executions"))
                    seen_hashes[ch] = config
                merged["operations"][op_name]["configurations"].append(config)

    # Sort configs within each operation by config_hash for deterministic output
    for op_data in merged["operations"].values():
        op_data["configurations"].sort(key=lambda c: c.get("config_hash", ""))

    # Update metadata
    final_configs = sum(len(op["configurations"]) for op in merged["operations"].values())
    merged["metadata"]["unique_operations"] = len(merged["operations"])
    merged["metadata"]["total_configurations"] = final_configs

    print(f"Merged {len(all_results)} traces: {final_configs} configs ({total_deduped} duplicates removed)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        print(f"Saved to {output_path}")

    return merged


def reconstruct_single_operation(operation_name, output_path=None, schema=DEFAULT_SCHEMA):
    """Reconstruct JSON for a single operation (faster for testing)."""
    _validate_schema(schema)
    print(f"Reconstructing {operation_name} from database (schema: {schema})...")
    conn = _get_read_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)

    # Get operation ID
    cur.execute(
        f"SELECT ttnn_operation_id FROM {S}.ttnn_operation WHERE operation_name = %s",
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
        f"""
        SELECT
            c.ttnn_configuration_id,
            c.config_hash,
            c.full_config_json,
            h.board_type,
            h.device_series,
            h.card_count,
            mc.mesh_shape,
            mc.device_count
        FROM {S}.ttnn_configuration c
        LEFT JOIN {S}.ttnn_hardware h ON h.ttnn_hardware_id = c.hardware_id
        LEFT JOIN {S}.ttnn_mesh_config mc ON mc.ttnn_mesh_config_id = c.mesh_config_id
        WHERE c.operation_id = %s
        ORDER BY c.config_hash
    """,
        (op_id,),
    )
    configs = cur.fetchall()

    # Bulk-fetch sources for all configs of this op in one query (avoids N+1).
    sources_by_cfg = _fetch_sources_by_op(cur, S, op_id)

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

        mesh_shape = _norm_array(mesh_shape)

        source_rows = sources_by_cfg.get(config_id, [])

        # Snowflake returns VARIANT/ARRAY as a JSON string
        if isinstance(full_config_json, str):
            try:
                full_config_json = json.loads(full_config_json)
            except (TypeError, ValueError):
                full_config_json = {}
        if not full_config_json:
            full_config_json = {}

        config_dict = dict(full_config_json)
        config_dict.pop("executions", None)
        config_dict.pop("source", None)
        config_dict.pop("machine_info", None)
        config_dict["config_hash"] = config_hash

        executions = []
        for source_file, hf_model, exec_count, trace_run_ids, pytest_args_seen in source_rows:
            source_str = format_source(source_file, hf_model)
            if not source_str:
                continue
            execution = {
                "source": source_str,
                "machine_info": {},
                "count": _as_int(exec_count),
                "trace_run_ids": list(_norm_array(trace_run_ids) or []),
                "pytest_args_seen": _clean_str_list(pytest_args_seen),
            }
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
            json.dump(result, f, indent=2, sort_keys=True)
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


def set_model_name(source_file=None, hf_model=None, model_id=None, new_name=None, schema=DEFAULT_SCHEMA):
    """Override model_name for a specific ttnn_model row.

    Identify the row by source_file (+ optional hf_model), hf_model alone, or
    model_id. The new name is lowercased before being stored.
    """
    _validate_schema(schema)
    if not new_name:
        print("Error: --model-name is required")
        return
    if not (source_file or hf_model or model_id):
        print("Error: provide --source-file, --hf-model, or --model-id to identify the row")
        return

    new_name = new_name.lower()
    conn = _get_write_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)

    # Build the WHERE clause + params once.
    if model_id is not None:
        where, params = "ttnn_model_id = %s", (int(model_id),)
    elif source_file and hf_model:
        where, params = "source_file = %s AND hf_model_identifier = %s", (source_file, hf_model)
    elif source_file:
        where, params = "source_file = %s", (source_file,)
    else:  # hf_model only
        where, params = "hf_model_identifier = %s", (hf_model,)

    # No RETURNING: SELECT the affected rows first, then UPDATE.
    cur.execute(
        f"SELECT ttnn_model_id, source_file, hf_model_identifier FROM {S}.ttnn_model WHERE {where}",
        params,
    )
    updated_rows = cur.fetchall()
    if updated_rows:
        cur.execute(
            f"UPDATE {S}.ttnn_model SET model_name = %s, update_ts = CURRENT_TIMESTAMP() WHERE {where}",
            (new_name,) + tuple(params),
        )

    if not updated_rows:
        print("No matching model found — nothing updated.")
        conn.close()
        return

    conn.commit()
    conn.close()
    for row in updated_rows:
        print(f"Updated ttnn_model_id={row[0]}: model_name set to '{new_name}'")
        print(f"  source_file={row[1]!r}  hf_model={row[2]!r}")


def _find_manifest_trace_references(trace_run_id, manifest_path=None):
    """Return references to a trace in the sweep manifest targets and registry."""
    try:
        data, path = _load_manifest(manifest_path)
    except Exception as e:
        return None, f"could not load manifest: {e}"

    targets = []
    for scope_name, entries in (data.get("targets") or {}).items():
        for index, entry in enumerate(entries or []):
            pinned_trace = entry.get("trace")
            if pinned_trace is None:
                continue
            if isinstance(pinned_trace, list):
                pinned_ids = [int(t) for t in pinned_trace]
            else:
                pinned_ids = [int(pinned_trace)]
            if trace_run_id in pinned_ids:
                targets.append(
                    {
                        "scope": scope_name,
                        "index": index,
                        "models": entry.get("model"),
                    }
                )

    registry_entries = []
    for index, entry in enumerate(data.get("registry") or []):
        entry_trace_id = entry.get("trace_id")
        try:
            entry_trace_id = int(entry_trace_id) if entry_trace_id is not None else None
        except (TypeError, ValueError):
            entry_trace_id = None
        if entry_trace_id == trace_run_id:
            registry_entries.append(
                {
                    "index": index,
                    "status": entry.get("status"),
                    "trace_uid": entry.get("trace_uid"),
                    "models": entry.get("models"),
                }
            )

    return {"path": path, "targets": targets, "registry_entries": registry_entries}, None


def delete_trace_run(trace_run_id, yes=False, schema=DEFAULT_SCHEMA):
    """Delete a trace_run and any configs that belong exclusively to it.

    A config is deleted only if no other trace_run links to it.
    Configs shared with other traces are unlinked but not deleted.

    Prints a summary and asks for confirmation unless --yes is passed.
    """
    _validate_schema(schema)
    conn = _get_write_connection()
    cur = conn.cursor()
    S = _schema_prefix(schema)

    cur.execute(
        f"SELECT config_count, notes FROM {S}.trace_run WHERE trace_run_id = %s",
        (trace_run_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"Trace run {trace_run_id} not found.")
        conn.close()
        return

    config_count, notes = row
    config_count = _as_int(config_count)

    manifest_refs, manifest_error = _find_manifest_trace_references(trace_run_id)

    cur.execute(
        f"""
        SELECT COUNT(*) FROM {S}.trace_run_config trc
        WHERE trc.trace_run_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM {S}.trace_run_configuration_model other
              WHERE other.configuration_id = trc.configuration_id
                AND other.trace_run_id != %s
          )
        """,
        (trace_run_id, trace_run_id),
    )
    exclusive_count = _as_int(cur.fetchone()[0])
    shared_count = (config_count or 0) - exclusive_count

    print(f"Trace run {trace_run_id}: {config_count} configs, notes={notes!r}")
    print(f"  {exclusive_count} configs will be DELETED (exclusive to this trace)")
    print(f"  {shared_count} configs will be UNLINKED only (shared with other traces)")
    if manifest_refs:
        target_refs = manifest_refs.get("targets", [])
        registry_refs = manifest_refs.get("registry_entries", [])
        if target_refs or registry_refs:
            print(f"  Warning: trace {trace_run_id} is still referenced in {manifest_refs['path']}")
            for ref in target_refs:
                print(f"    Target ref: scope={ref['scope']} entry_index={ref['index']} models={ref['models']!r}")
            for ref in registry_refs:
                print(
                    f"    Registry ref: entry_index={ref['index']} status={ref['status']!r} "
                    f"trace_uid={ref['trace_uid']!r}"
                )
    elif manifest_error:
        print(f"  Warning: {manifest_error}")

    if not yes:
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            conn.close()
            return

    cur.execute(
        f"""
        SELECT DISTINCT trcm.configuration_id
        FROM {S}.trace_run_configuration_model trcm
        WHERE trcm.trace_run_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM {S}.trace_run_configuration_model other
              WHERE other.configuration_id = trcm.configuration_id
                AND other.trace_run_id != %s
          )
        """,
        (trace_run_id, trace_run_id),
    )
    exclusive_ids = [_as_int(r[0]) for r in cur.fetchall()]
    cur.execute(
        f"""
        SELECT DISTINCT configuration_id
        FROM {S}.trace_run_configuration_model
        WHERE trace_run_id = %s
        """,
        (trace_run_id,),
    )
    affected_config_ids = [_as_int(r[0]) for r in cur.fetchall()]

    cur.execute(f"DELETE FROM {S}.trace_run_configuration_model WHERE trace_run_id = %s", (trace_run_id,))

    cur.execute(f"DELETE FROM {S}.trace_run_config WHERE trace_run_id = %s", (trace_run_id,))

    cur.execute(f"DELETE FROM {S}.trace_run_model WHERE trace_run_id = %s", (trace_run_id,))

    remaining_config_ids = [config_id for config_id in affected_config_ids if config_id not in exclusive_ids]
    if remaining_config_ids:
        refresh_trace_aggregates(cur, [], remaining_config_ids, schema=schema)

    if exclusive_ids:
        placeholders = ", ".join(["%s"] * len(exclusive_ids))
        cur.execute(
            f"DELETE FROM {S}.ttnn_configuration_model WHERE configuration_id IN ({placeholders})",
            exclusive_ids,
        )
        cur.execute(
            f"DELETE FROM {S}.ttnn_configuration WHERE ttnn_configuration_id IN ({placeholders})",
            exclusive_ids,
        )
    deleted_configs = len(exclusive_ids)

    cur.execute(f"DELETE FROM {S}.trace_run WHERE trace_run_id = %s", (trace_run_id,))

    conn.commit()
    conn.close()
    print(f"Done: trace run {trace_run_id} deleted, {deleted_configs} configs removed, {shared_count} unlinked.")


if __name__ == "__main__":
    import sys

    def _extract_schema(argv):
        """Extract --schema VALUE from argv, return (schema, remaining_argv)."""
        schema = DEFAULT_SCHEMA
        remaining = []
        i = 0
        while i < len(argv):
            if argv[i] == "--schema" and i + 1 < len(argv):
                schema = argv[i + 1]
                i += 2
            else:
                remaining.append(argv[i])
                i += 1
        return schema, remaining

    _schema, sys.argv = _extract_schema(sys.argv)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "load":
            args = sys.argv[2:]
            dry_run = "--dry-run" in args
            args = [a for a in args if a != "--dry-run"]
            emit_id_file = None
            filtered = []
            i = 0
            while i < len(args):
                if args[i] == "--emit-id-file" and i + 1 < len(args):
                    emit_id_file = args[i + 1]
                    i += 2
                elif args[i].startswith("--emit-id-file="):
                    emit_id_file = args[i].split("=", 1)[1]
                    i += 1
                else:
                    filtered.append(args[i])
                    i += 1
            args = filtered
            json_path = args[0] if len(args) > 0 else None
            sha = args[1] if len(args) > 1 else None
            load_data(
                json_path=json_path,
                tt_metal_sha=sha,
                dry_run=dry_run,
                schema=_schema,
                emit_id_file=emit_id_file,
            )
        elif cmd == "reconstruct":
            output = sys.argv[2] if len(sys.argv) > 2 else "ttnn_operations_reconstructed.json"
            model_filter = sys.argv[3].split(",") if len(sys.argv) > 3 else None
            reconstruct_from_db(output, _schema, model_filter)
        elif cmd == "reconstruct-trace":
            if len(sys.argv) < 3:
                print(
                    "Usage: python load_ttnn_ops_data_v2.py reconstruct-trace <trace_run_id> [output.json] [--schema S]"
                )
                sys.exit(1)
            tr_id = int(sys.argv[2])
            output = sys.argv[3] if len(sys.argv) > 3 else None
            reconstruct_from_trace_run(tr_id, output, schema=_schema)
        elif cmd == "list-traces":
            model_filter = sys.argv[2].split(",") if len(sys.argv) > 2 else None
            list_trace_runs(model_filter, schema=_schema)
        elif cmd == "list-variants":
            if len(sys.argv) < 3:
                print(
                    "Usage: python load_ttnn_ops_data_v2.py list-variants <source_file_pattern> [--schema S]\n"
                    "  Lists every (model, pytest_args, hardware) combination ever traced\n"
                    "  for source_files matching the ILIKE pattern (e.g. 'flux1' or '%%flux1%%')."
                )
                sys.exit(1)
            list_variants(sys.argv[2], schema=_schema)
        elif cmd == "reconstruct-manifest":
            _args = sys.argv[2:]
            # Extract --models-filter if present.
            #
            # Tolerate shell word-splitting of the value: an unquoted shell variable
            # holding "gpt_oss, tt_dit" (with a space) expands to two argv tokens
            # ("gpt_oss," and "tt_dit"). Greedily absorb consecutive non-flag tokens
            # ONLY across comma boundaries (prev ends with "," or next starts with
            # ","), so a legitimate scope positional that happens to follow without a
            # comma is left untouched.
            _model_filter = None
            if "--models-filter" in _args:
                idx = _args.index("--models-filter")
                consumed = []
                i = idx + 1
                while i < len(_args) and not _args[i].startswith("--"):
                    consumed.append(_args[i])
                    if i + 1 >= len(_args):
                        i += 1
                        break
                    # Continue absorbing only across an explicit comma boundary.
                    if _args[i].endswith(",") or _args[i + 1].startswith(","):
                        i += 1
                        continue
                    i += 1
                    break
                if consumed:
                    # "".join glues "gpt_oss," + "tt_dit" -> "gpt_oss,tt_dit",
                    # and also handles a lone "," token between the names.
                    joined = "".join(consumed)
                    _model_filter = [m.strip() for m in joined.split(",") if m.strip()]
                    _args = _args[:idx] + _args[i:]
                else:
                    _args = _args[:idx]
            if _args and _args[0].endswith(".json"):
                manifest, output = None, _args[0]
                _args = _args[1:]
            else:
                manifest = _args[0] if _args else None
                output = _args[1] if len(_args) > 1 else None
                _args = _args[2:]
            scope = _args[0] if _args else None
            reconstruct_from_manifest(manifest, output, scope, _schema, model_filter=_model_filter)
        elif cmd == "resolve-manifest":
            manifest = sys.argv[2] if len(sys.argv) > 2 else None
            scope = sys.argv[3] if len(sys.argv) > 3 else None
            resolve_manifest(manifest, scope)
        elif cmd == "reconstruct-op":
            if len(sys.argv) < 3:
                print(
                    "Usage: python load_ttnn_ops_data_v2.py reconstruct-op <operation_name> [output.json] [--schema S]"
                )
                sys.exit(1)
            op_name = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else None
            reconstruct_single_operation(op_name, output, schema=_schema)
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
                print("Usage: python load_ttnn_ops_data_v2.py delete-trace <trace_run_id> [--yes] [--schema S]")
                sys.exit(1)
            _tr_id = int(sys.argv[2])
            _yes = "--yes" in sys.argv
            delete_trace_run(_tr_id, yes=_yes, schema=_schema)
        elif cmd == "set-model-name":
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
                schema=_schema,
            )
        else:
            print(f"Unknown command: {cmd}")
            print("Usage (all commands accept --schema <name>, default: ttnn_ops_v6):")
            print(
                "  python load_ttnn_ops_data_v2.py load [json_path] [sha] [--dry-run] [--emit-id-file PATH]  # Load JSON to DB"
            )
            print(
                "  python load_ttnn_ops_data_v2.py reconstruct [output] [models]                # Reconstruct JSON from DB"
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
                "  python load_ttnn_ops_data_v2.py list-variants <pattern>                     # List traced (args, hw) variants per model"
            )
            print(
                "  python load_ttnn_ops_data_v2.py reconstruct-op <name> [output.json]          # Reconstruct single op"
            )
            print("  python load_ttnn_ops_data_v2.py verify [original] [reconstructed]            # Compare files")
            print(
                "  python load_ttnn_ops_data_v2.py find-lines <op> <i1,i2>                     # Find config line numbers"
            )
            print(
                "  python load_ttnn_ops_data_v2.py delete-trace <id> [--yes]                   # Delete trace + exclusive configs"
            )
            print(
                "  python load_ttnn_ops_data_v2.py set-model-name --source-file P --model-name N  # Override a model's name"
            )
    else:
        load_data(schema=_schema)
