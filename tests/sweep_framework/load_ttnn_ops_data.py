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
import hashlib
import os
import re
import psycopg2

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


def compute_config_hash(operation_name, arguments, hardware, mesh_config):
    """Compute SHA-256 hash for configuration deduplication.

    Config identity = operation + arguments + hardware + mesh
    """
    normalized = {
        "operation": operation_name,
        "arguments": arguments,
        "hardware": hardware,
        "mesh": mesh_config,
    }
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


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


def extract_tensor_info(tensor_data):
    """Extract flattened tensor info from Tensor argument."""
    if not isinstance(tensor_data, dict) or "Tensor" not in tensor_data:
        return None

    tensor = tensor_data.get("Tensor", {})
    if tensor is None:
        return {"is_tensor": True, "tensor_dtype": None}

    spec = tensor.get("tensor_spec", {})
    layout = spec.get("tensor_layout", {})
    mem_config = layout.get("memory_config", {})

    result = {
        "is_tensor": True,
        "tensor_shape": parse_array_value(spec.get("logical_shape")),
        "tensor_dtype": layout.get("dtype"),
        "tensor_layout": tensor.get("layout"),
        "tensor_storage_type": tensor.get("storage_type"),
        "tensor_memory_layout": mem_config.get("memory_layout"),
        "tensor_buffer_type": mem_config.get("buffer_type"),
    }

    shard_spec = mem_config.get("shard_spec") or mem_config.get("nd_shard_spec")
    if shard_spec and isinstance(shard_spec, dict):
        shard_shape_raw = shard_spec.get("shard_shape") or shard_spec.get("shape")
        result["shard_shape"] = parse_array_value(shard_shape_raw)
        result["shard_orientation"] = shard_spec.get("orientation")

        grid = shard_spec.get("grid", [])
        if grid and len(grid) >= 2:
            if isinstance(grid[0], dict):
                result["core_grid_x"] = grid[1].get("x", 0) - grid[0].get("x", 0) + 1
                result["core_grid_y"] = grid[1].get("y", 0) - grid[0].get("y", 0) + 1

    return result


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
            INSERT INTO ttnn_ops.ttnn_hardware (board_type, device_series, card_count)
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
                "SELECT ttnn_hardware_id FROM ttnn_ops.ttnn_hardware WHERE board_type=%s AND device_series=%s AND card_count=%s",
                hw_key,
            )
            hardware_cache[hw_key] = cur.fetchone()[0]

    return hardware_cache.get(hw_key), hw_key


def get_or_create_mesh_config(
    cur, mesh_config_cache, mesh_shape, device_count, placement_type, shard_dim, distribution_shape
):
    """Get or create a mesh config entry, return the ID."""
    if not mesh_shape:
        return None, None

    dist_tuple = tuple(distribution_shape) if distribution_shape else None
    mesh_key = (tuple(mesh_shape), device_count, placement_type, shard_dim, dist_tuple)

    if mesh_key not in mesh_config_cache:
        cur.execute(
            """
            INSERT INTO ttnn_ops.ttnn_mesh_config (mesh_shape, device_count, placement_type, shard_dim, distribution_shape)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (mesh_shape, device_count, placement_type, shard_dim, distribution_shape) DO NOTHING
            RETURNING ttnn_mesh_config_id
        """,
            (mesh_shape, device_count, placement_type, shard_dim, distribution_shape),
        )
        result = cur.fetchone()
        if result:
            mesh_config_cache[mesh_key] = result[0]
        else:
            cur.execute(
                """
                SELECT ttnn_mesh_config_id FROM ttnn_ops.ttnn_mesh_config
                WHERE mesh_shape = %s AND device_count = %s AND placement_type = %s
                  AND shard_dim IS NOT DISTINCT FROM %s
                  AND distribution_shape IS NOT DISTINCT FROM %s
            """,
                (mesh_shape, device_count, placement_type, shard_dim, distribution_shape),
            )
            result = cur.fetchone()
            if result:
                mesh_config_cache[mesh_key] = result[0]

    mesh_info = {"mesh_shape": mesh_shape, "placement_type": placement_type, "shard_dim": shard_dim}
    return mesh_config_cache.get(mesh_key), mesh_info


def get_or_create_model(cur, model_cache, source_file, hf_model):
    """Get or create a model entry, return the ID."""
    if not source_file:
        return None

    model_key = (source_file, hf_model)
    if model_key not in model_cache:
        model_family = extract_model_family(source_file, hf_model)
        cur.execute(
            """
            INSERT INTO ttnn_ops.ttnn_model (source_file, hf_model_identifier, model_family)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_file, hf_model_identifier) DO UPDATE SET update_ts = NOW()
            RETURNING ttnn_model_id
        """,
            (source_file, hf_model, model_family),
        )
        model_cache[model_key] = cur.fetchone()[0]

    return model_cache[model_key]


def extract_primary_tensor_info(arguments):
    """Extract primary tensor info from the first tensor argument."""
    if not arguments:
        return {}

    for arg in arguments:
        if isinstance(arg, dict):
            for k, v in arg.items():
                if isinstance(v, dict) and "Tensor" in v:
                    info = extract_tensor_info(v)
                    if info:
                        return info
    return {}


def parse_mesh_from_machine_info(machine_info):
    """Extract mesh configuration from machine_info.

    Returns: (mesh_shape, device_count, placement_type, shard_dim, distribution_shape)
    """
    tensor_placements = machine_info.get("tensor_placements", [])

    if not tensor_placements:
        # No tensor_placements = no specific mesh config
        return None, None, None, None, None

    # Take first placement
    placement = tensor_placements[0]
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


def load_data():
    """Main loading function."""
    print(f"Loading JSON from {JSON_PATH}...")
    with open(JSON_PATH) as f:
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

    total_configs = 0
    total_args = 0
    total_model_links = 0

    for op_name, op_data in operations.items():
        # Insert operation
        if op_name not in operation_cache:
            cur.execute(
                """
                INSERT INTO ttnn_ops.ttnn_operation (operation_name)
                VALUES (%s)
                ON CONFLICT (operation_name) DO UPDATE SET update_ts = NOW()
                RETURNING ttnn_operation_id
            """,
                (op_name,),
            )
            operation_cache[op_name] = cur.fetchone()[0]

        op_id = operation_cache[op_name]

        for config in op_data.get("configurations", []):
            arguments = config.get("arguments", [])
            source = config.get("source")
            machine_info_list = config.get("machine_info", [{}])

            # Ensure machine_info_list is a list
            if not machine_info_list:
                machine_info_list = [{}]

            # Parse all sources (string or array -> list of tuples)
            source_tuples = parse_all_sources(source)

            # Get/create all model IDs for this config's sources
            model_ids = []
            for source_file, hf_model in source_tuples:
                model_id = get_or_create_model(cur, model_cache, source_file, hf_model)
                model_ids.append(model_id)

            # Extract primary tensor info (same for all hardware/mesh variants)
            primary_info = extract_primary_tensor_info(arguments)

            # Process each machine_info entry - each creates a potentially unique config
            # because hardware + mesh are part of config identity
            for machine_info in machine_info_list:
                # Parse hardware
                board_type = machine_info.get("board_type")
                device_series = machine_info.get("device_series")
                card_count = machine_info.get("card_count", 1)

                hardware_id, hw_key = get_or_create_hardware(cur, hardware_cache, board_type, device_series, card_count)

                # Parse mesh config
                mesh_shape, device_count, placement_type, shard_dim, distribution_shape = parse_mesh_from_machine_info(
                    machine_info
                )

                mesh_config_id = None
                mesh_info = None
                if mesh_shape:
                    mesh_config_id, mesh_info = get_or_create_mesh_config(
                        cur, mesh_config_cache, mesh_shape, device_count, placement_type, shard_dim, distribution_shape
                    )

                # Compute config hash: op + args + hardware + mesh
                config_hash = compute_config_hash(op_name, arguments, hw_key if hardware_id else None, mesh_info)

                # Check if we've already created this config
                if config_hash in config_cache:
                    config_id = config_cache[config_hash]
                    # Just update last_seen_ts
                    cur.execute(
                        "UPDATE ttnn_ops.ttnn_configuration SET last_seen_ts = NOW() WHERE ttnn_configuration_id = %s",
                        (config_id,),
                    )
                else:
                    # Insert new configuration
                    try:
                        cur.execute(
                            """
                            INSERT INTO ttnn_ops.ttnn_configuration
                            (operation_id, hardware_id, mesh_config_id, primary_dtype, primary_storage_type, primary_layout,
                             primary_memory_layout, primary_buffer_type, primary_shape, config_hash, full_config_json)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (config_hash) DO UPDATE SET last_seen_ts = NOW()
                            RETURNING ttnn_configuration_id
                        """,
                            (
                                op_id,
                                hardware_id,
                                mesh_config_id,
                                primary_info.get("tensor_dtype"),
                                primary_info.get("tensor_storage_type"),
                                primary_info.get("tensor_layout"),
                                primary_info.get("tensor_memory_layout"),
                                primary_info.get("tensor_buffer_type"),
                                primary_info.get("tensor_shape"),
                                config_hash,
                                json.dumps(config),
                            ),
                        )
                        config_id = cur.fetchone()[0]
                        config_cache[config_hash] = config_id
                        total_configs += 1

                        # Insert arguments (only for new configs)
                        for i, arg in enumerate(arguments):
                            if isinstance(arg, dict):
                                arg_name = list(arg.keys())[0]
                                arg_value = arg[arg_name]

                                unsupported_type_string = None
                                if isinstance(arg_value, str) and "unsupported type" in arg_value:
                                    unsupported_type_string = arg_value

                                tensor_info = extract_tensor_info(
                                    {"Tensor": arg_value}
                                    if isinstance(arg_value, dict) and "Tensor" not in arg_value
                                    else arg_value
                                    if isinstance(arg_value, dict)
                                    else {}
                                )
                                is_tensor = tensor_info is not None and tensor_info.get("is_tensor", False)
                                is_tensor_list = (
                                    isinstance(arg_value, list)
                                    and len(arg_value) > 0
                                    and isinstance(arg_value[0], dict)
                                )

                                cur.execute(
                                    """
                                    INSERT INTO ttnn_ops.ttnn_argument
                                    (configuration_id, arg_index, arg_name, is_tensor, is_tensor_list, tensor_count,
                                     tensor_dtype, tensor_storage_type, tensor_layout, tensor_memory_layout,
                                     tensor_buffer_type, tensor_shape, shard_shape, shard_orientation,
                                     core_grid_x, core_grid_y, scalar_value_json, unsupported_type_string)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (configuration_id, arg_index) DO NOTHING
                                """,
                                    (
                                        config_id,
                                        i,
                                        arg_name,
                                        is_tensor,
                                        is_tensor_list,
                                        len(arg_value) if is_tensor_list else None,
                                        tensor_info.get("tensor_dtype") if tensor_info else None,
                                        tensor_info.get("tensor_storage_type") if tensor_info else None,
                                        tensor_info.get("tensor_layout") if tensor_info else None,
                                        tensor_info.get("tensor_memory_layout") if tensor_info else None,
                                        tensor_info.get("tensor_buffer_type") if tensor_info else None,
                                        tensor_info.get("tensor_shape") if tensor_info else None,
                                        tensor_info.get("shard_shape") if tensor_info else None,
                                        tensor_info.get("shard_orientation") if tensor_info else None,
                                        tensor_info.get("core_grid_x") if tensor_info else None,
                                        tensor_info.get("core_grid_y") if tensor_info else None,
                                        json.dumps(arg_value)
                                        if not is_tensor and not is_tensor_list and not unsupported_type_string
                                        else None,
                                        unsupported_type_string,
                                    ),
                                )
                                total_args += 1

                    except psycopg2.errors.UniqueViolation:
                        conn.rollback()
                        # Config already exists, fetch its ID
                        cur.execute(
                            "SELECT ttnn_configuration_id FROM ttnn_ops.ttnn_configuration WHERE config_hash = %s",
                            (config_hash,),
                        )
                        result = cur.fetchone()
                        if result:
                            config_id = result[0]
                            config_cache[config_hash] = config_id
                        continue

                # Link ALL sources to this config via junction table
                for model_id in model_ids:
                    if model_id is None:
                        continue
                    cur.execute(
                        """
                        INSERT INTO ttnn_ops.ttnn_configuration_model (configuration_id, model_id)
                        VALUES (%s, %s)
                        ON CONFLICT (configuration_id, model_id) DO UPDATE SET last_seen_ts = NOW()
                    """,
                        (config_id, model_id),
                    )
                    total_model_links += 1

        conn.commit()
        print(f"  Loaded {op_name}: {len(op_data.get('configurations', []))} JSON configs")

    conn.commit()
    print(f"\n✅ Loaded {total_configs} unique configurations, {total_args} arguments, {total_model_links} model links")

    # Print stats
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_operation")
    print(f"   Operations: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_model")
    print(f"   Models: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_hardware")
    print(f"   Hardware configs: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_mesh_config")
    print(f"   Mesh configs: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_configuration")
    print(f"   Configurations: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_argument")
    print(f"   Arguments: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_configuration_model")
    print(f"   Config-Model links: {cur.fetchone()[0]}")

    conn.close()


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


def format_mesh_placement(mesh_shape, placement_type, shard_dim):
    """Format mesh config back to tensor_placements format."""
    if not mesh_shape:
        return None

    placement_dict = {"mesh_device_shape": json.dumps(mesh_shape)}

    # Add placement string if not default replicate
    if placement_type == "shard" and shard_dim is not None:
        placement_dict["placement"] = f"[PlacementShard({shard_dim})]"
    elif placement_type == "replicate":
        placement_dict["placement"] = "[PlacementReplicate]"

    return placement_dict


def reconstruct_from_db(output_path=None):
    """Reconstruct ttnn_operations_master.json from the database.

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
    print("Reconstructing JSON from database...")
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    # Get all operations
    cur.execute(
        """
        SELECT ttnn_operation_id, operation_name
        FROM ttnn_ops.ttnn_operation
        ORDER BY operation_name
    """
    )
    operations = cur.fetchall()
    print(f"Found {len(operations)} operations")

    result = {"operations": {}}

    for op_id, op_name in operations:
        # Get all configurations for this operation
        cur.execute(
            """
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
                mc.device_count,
                mc.placement_type,
                mc.shard_dim,
                mc.distribution_shape
            FROM ttnn_ops.ttnn_configuration c
            LEFT JOIN ttnn_ops.ttnn_hardware h ON h.ttnn_hardware_id = c.hardware_id
            LEFT JOIN ttnn_ops.ttnn_mesh_config mc ON mc.ttnn_mesh_config_id = c.mesh_config_id
            WHERE c.operation_id = %s
            ORDER BY c.ttnn_configuration_id
        """,
            (op_id,),
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
                hardware_id,
                mesh_config_id,
                board_type,
                device_series,
                card_count,
                mesh_shape,
                device_count,
                placement_type,
                shard_dim,
                distribution_shape,
            ) = config_row

            # Get all sources linked to this config
            cur.execute(
                """
                SELECT m.source_file, m.hf_model_identifier
                FROM ttnn_ops.ttnn_configuration_model cm
                JOIN ttnn_ops.ttnn_model m ON m.ttnn_model_id = cm.model_id
                WHERE cm.configuration_id = %s
                ORDER BY m.source_file, m.hf_model_identifier
            """,
                (config_id,),
            )
            source_rows = cur.fetchall()

            # Format sources
            sources = [format_source(sf, hf) for sf, hf in source_rows]
            sources = [s for s in sources if s]  # Remove None values

            if len(sources) == 0:
                source = None
            elif len(sources) == 1:
                source = sources[0]
            else:
                source = sources

            # Build machine_info
            machine_info = []
            if board_type:
                mi = {
                    "board_type": board_type,
                    "device_series": device_series,
                    "card_count": card_count,
                }

                # Add tensor_placements if mesh config exists
                if mesh_shape:
                    placement = format_mesh_placement(mesh_shape, placement_type, shard_dim)
                    if placement:
                        mi["tensor_placements"] = [placement]

                machine_info.append(mi)

            # Use full_config_json for arguments (preserves exact original structure)
            if full_config_json:
                arguments = full_config_json.get("arguments", [])
            else:
                arguments = []

            config_dict = {"arguments": arguments}

            # Include config_hash for direct correlation with database
            if config_hash:
                config_dict["config_hash"] = config_hash

            if source:
                config_dict["source"] = source

            if machine_info:
                config_dict["machine_info"] = machine_info

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


def reconstruct_single_operation(operation_name, output_path=None):
    """Reconstruct JSON for a single operation (faster for testing)."""
    print(f"Reconstructing {operation_name} from database...")
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()

    # Get operation ID
    cur.execute(
        "SELECT ttnn_operation_id FROM ttnn_ops.ttnn_operation WHERE operation_name = %s",
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
            c.hardware_id,
            c.mesh_config_id,
            h.board_type,
            h.device_series,
            h.card_count,
            mc.mesh_shape,
            mc.device_count,
            mc.placement_type,
            mc.shard_dim,
            mc.distribution_shape
        FROM ttnn_ops.ttnn_configuration c
        LEFT JOIN ttnn_ops.ttnn_hardware h ON h.ttnn_hardware_id = c.hardware_id
        LEFT JOIN ttnn_ops.ttnn_mesh_config mc ON mc.ttnn_mesh_config_id = c.mesh_config_id
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
            hardware_id,
            mesh_config_id,
            board_type,
            device_series,
            card_count,
            mesh_shape,
            device_count,
            placement_type,
            shard_dim,
            distribution_shape,
        ) = config_row

        # Get sources
        cur.execute(
            """
            SELECT m.source_file, m.hf_model_identifier
            FROM ttnn_ops.ttnn_configuration_model cm
            JOIN ttnn_ops.ttnn_model m ON m.ttnn_model_id = cm.model_id
            WHERE cm.configuration_id = %s
            ORDER BY m.source_file, m.hf_model_identifier
        """,
            (config_id,),
        )
        source_rows = cur.fetchall()

        sources = [format_source(sf, hf) for sf, hf in source_rows]
        sources = [s for s in sources if s]

        if len(sources) == 0:
            source = None
        elif len(sources) == 1:
            source = sources[0]
        else:
            source = sources

        # Build machine_info
        machine_info = []
        if board_type:
            mi = {
                "board_type": board_type,
                "device_series": device_series,
                "card_count": card_count,
            }
            if mesh_shape:
                placement = format_mesh_placement(mesh_shape, placement_type, shard_dim)
                if placement:
                    mi["tensor_placements"] = [placement]
            machine_info.append(mi)

        if full_config_json:
            arguments = full_config_json.get("arguments", [])
        else:
            arguments = []

        config_dict = {"arguments": arguments}

        # Include config_hash for direct correlation with database
        if config_hash:
            config_dict["config_hash"] = config_hash

        if source:
            config_dict["source"] = source
        if machine_info:
            config_dict["machine_info"] = machine_info

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


def detect_duplicates(json_path=None, operation_filter=None, show_examples=3):
    """Detect duplicate configurations in the original JSON file.

    A duplicate is when the same (operation + arguments + hardware + mesh) appears
    multiple times. These get deduplicated when loaded into the database.

    Args:
        json_path: Path to JSON file (defaults to JSON_PATH)
        operation_filter: If provided, only analyze this operation (e.g., 'ttnn::add')
        show_examples: Number of duplicate groups to show in detail

    Returns:
        Dict with duplicate statistics and details
    """
    json_path = json_path or JSON_PATH
    print(f"Detecting duplicates in {json_path}...")

    with open(json_path) as f:
        data = json.load(f)

    operations = data.get("operations", {})

    if operation_filter:
        if operation_filter not in operations:
            print(f"Operation {operation_filter} not found")
            return None
        operations = {operation_filter: operations[operation_filter]}

    total_configs = 0
    total_unique = 0
    total_duplicates = 0

    results = {
        "by_operation": {},
        "duplicate_examples": [],
    }

    for op_name, op_data in operations.items():
        configs = op_data.get("configurations", [])
        hash_to_indices = {}  # hash -> list of config indices

        for idx, config in enumerate(configs):
            args = config.get("arguments", [])
            mi_list = config.get("machine_info", [{}])

            for mi in mi_list:
                # Extract hardware
                board = mi.get("board_type")
                series = mi.get("device_series")
                if isinstance(series, list):
                    series = series[0] if series else None
                card = mi.get("card_count", 1)
                hw = (board, series, card) if board else None

                # Extract mesh
                tp = mi.get("tensor_placements", [])
                mesh = None
                if tp:
                    mesh_shape = tp[0].get("mesh_device_shape")
                    placement = tp[0].get("placement")
                    mesh = (mesh_shape, placement)

                # Compute hash
                config_hash = compute_config_hash(op_name, args, hw, mesh)

                if config_hash not in hash_to_indices:
                    hash_to_indices[config_hash] = []
                hash_to_indices[config_hash].append(
                    {
                        "index": idx,
                        "source": config.get("source"),
                        "hw": hw,
                        "mesh": mesh,
                    }
                )

        # Count duplicates for this operation
        unique_count = len(hash_to_indices)
        config_count = len(configs)
        dup_count = config_count - unique_count

        total_configs += config_count
        total_unique += unique_count
        total_duplicates += dup_count

        if dup_count > 0:
            results["by_operation"][op_name] = {
                "total_configs": config_count,
                "unique_configs": unique_count,
                "duplicates": dup_count,
            }

            # Collect duplicate examples
            if len(results["duplicate_examples"]) < show_examples:
                for h, entries in hash_to_indices.items():
                    if len(entries) > 1:
                        results["duplicate_examples"].append(
                            {
                                "operation": op_name,
                                "hash": h[:16] + "...",
                                "occurrences": len(entries),
                                "indices": [e["index"] for e in entries],
                                "sources": list(set(str(e["source"])[:60] for e in entries)),
                                "hw": entries[0]["hw"],
                                "mesh": entries[0]["mesh"],
                            }
                        )
                        if len(results["duplicate_examples"]) >= show_examples:
                            break

    # Print summary
    print(f"\n=== Duplicate Detection Summary ===")
    print(f"Total configs in JSON: {total_configs}")
    print(f"Unique configs (by hash): {total_unique}")
    print(f"Duplicates: {total_duplicates} ({100*total_duplicates/total_configs:.1f}%)")

    if results["by_operation"]:
        print(f"\nOperations with duplicates ({len(results['by_operation'])}):")
        sorted_ops = sorted(results["by_operation"].items(), key=lambda x: x[1]["duplicates"], reverse=True)
        for op_name, stats in sorted_ops[:15]:
            print(f"  {op_name}: {stats['total_configs']} -> {stats['unique_configs']} ({stats['duplicates']} dups)")
        if len(sorted_ops) > 15:
            print(f"  ... and {len(sorted_ops) - 15} more operations")

    if results["duplicate_examples"]:
        print(f"\nExample duplicate groups:")
        for ex in results["duplicate_examples"]:
            print(f"\n  {ex['operation']} (hash: {ex['hash']})")
            print(
                f"    Occurrences: {ex['occurrences']} at indices {ex['indices'][:5]}{'...' if len(ex['indices']) > 5 else ''}"
            )
            print(f"    Hardware: {ex['hw']}")
            print(f"    Mesh: {ex['mesh']}")
            print(f"    Sources: {ex['sources'][:2]}")

    results["summary"] = {
        "total_configs": total_configs,
        "unique_configs": total_unique,
        "duplicates": total_duplicates,
        "duplicate_percentage": 100 * total_duplicates / total_configs if total_configs > 0 else 0,
    }

    return results


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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "load":
            load_data()
        elif cmd == "reconstruct":
            output = sys.argv[2] if len(sys.argv) > 2 else "ttnn_operations_reconstructed.json"
            reconstruct_from_db(output)
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
        elif cmd == "duplicates":
            json_file = sys.argv[2] if len(sys.argv) > 2 else JSON_PATH
            op_filter = sys.argv[3] if len(sys.argv) > 3 else None
            detect_duplicates(json_file, op_filter)
        elif cmd == "find-lines":
            if len(sys.argv) < 4:
                print("Usage: python load_ttnn_ops_data_v2.py find-lines <operation> <index1,index2,...>")
                sys.exit(1)
            op_name = sys.argv[2]
            indices = [int(i) for i in sys.argv[3].split(",")]
            json_file = sys.argv[4] if len(sys.argv) > 4 else JSON_PATH
            find_config_line_numbers(json_file, op_name, indices)
        else:
            print(f"Unknown command: {cmd}")
            print("Usage:")
            print("  python load_ttnn_ops_data_v2.py load                    # Load JSON to DB")
            print("  python load_ttnn_ops_data_v2.py reconstruct [output]    # Reconstruct JSON from DB")
            print("  python load_ttnn_ops_data_v2.py reconstruct-op <name>   # Reconstruct single op")
            print("  python load_ttnn_ops_data_v2.py verify [original] [reconstructed]  # Compare files")
            print("  python load_ttnn_ops_data_v2.py duplicates [json] [op]  # Detect duplicates")
            print("  python load_ttnn_ops_data_v2.py find-lines <op> <i1,i2> # Find config line numbers")
    else:
        load_data()
