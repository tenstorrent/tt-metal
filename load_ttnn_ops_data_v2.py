#!/usr/bin/env python3
"""Load ttnn_operations_master.json into Neon.tech PostgreSQL for testing.

Updated for junction table schema where configurations link to multiple mesh configs.
"""

import json
import hashlib
import os
import re
import psycopg2
from psycopg2.extras import execute_values

# Connection string from environment or hardcode for testing
NEON_URL = os.environ.get("NEON_CONNECTION_STRING", "postgresql://...")
JSON_PATH = "model_tracer/traced_operations/ttnn_operations_master_test.json"

# Default 1x1 mesh - every config links to this
DEFAULT_MESH = {
    "mesh_shape": [1, 1],
    "device_count": 1,
    "placement_type": "replicate",
    "shard_dim": None,
    "distribution_shape": None,
}


def parse_source(source_str):
    """Parse source string into source_file and hf_model_identifier."""
    if not source_str:
        return source_str, None

    # source can be a list in new format - take first element
    if isinstance(source_str, list):
        source_str = source_str[0] if source_str else None
        if not source_str:
            return None, None

    # Pattern: "path/to/file.py [HF_MODEL:org/model-name]"
    match = re.match(r"(.+?)\s*\[HF_MODEL:(.+?)\]", source_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return source_str, None


def extract_model_family(source_file, hf_model):
    """Infer model family from source or HF identifier."""
    combined = f"{source_file or ''} {hf_model or ''}".lower()
    families = ["llama", "qwen", "deepseek", "mistral", "whisper", "efficientnet", "resnet", "bert"]
    for family in families:
        if family in combined:
            return family
    return None


def compute_config_hash(operation_name, arguments, hardware):
    """Compute SHA-256 hash for configuration deduplication.

    NOTE: Mesh is NOT included in the hash. Same args can have multiple mesh variants.
    """
    normalized = {
        "operation": operation_name,
        "arguments": arguments,
        "hardware": hardware
        # mesh is excluded - linked via junction table
    }
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def parse_array_value(value):
    """Convert string array representation to a proper list for PostgreSQL."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value == "std::nullopt" or value == "nullopt":
            return None
        try:
            parsed = json.loads(value.replace("{", "[").replace("}", "]"))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
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

    # Check for shard
    shard_match = re.search(r"PlacementShard\((\d+)\)", placement_str)
    if shard_match:
        return "shard", int(shard_match.group(1))

    # Default to replicate
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


def get_or_create_mesh_config(
    cur, mesh_config_cache, mesh_shape, device_count, placement_type, shard_dim, distribution_shape
):
    """Get or create a mesh config entry, return the ID."""
    # Convert distribution_shape to tuple for hashability
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
            # Already exists, fetch it
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

    return mesh_config_cache.get(mesh_key)


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

    # Ensure default 1x1 mesh exists and cache it
    default_mesh_id = get_or_create_mesh_config(
        cur,
        mesh_config_cache,
        DEFAULT_MESH["mesh_shape"],
        DEFAULT_MESH["device_count"],
        DEFAULT_MESH["placement_type"],
        DEFAULT_MESH["shard_dim"],
        DEFAULT_MESH["distribution_shape"],
    )
    conn.commit()
    print(f"Default 1x1 mesh config ID: {default_mesh_id}")

    total_configs = 0
    total_args = 0
    total_mesh_links = 0

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
            source = config.get("source", "")
            machine_info = config.get("machine_info", [{}])[0] if config.get("machine_info") else {}

            # Parse model
            source_file, hf_model = parse_source(source)
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
            model_id = model_cache[model_key]

            # Parse hardware
            board_type = machine_info.get("board_type")
            device_series = machine_info.get("device_series")
            if isinstance(device_series, list):
                device_series = device_series[0] if device_series else None
            card_count = machine_info.get("card_count", 1)
            hw_key = (board_type, device_series, card_count)
            if hw_key[0] and hw_key not in hardware_cache:
                cur.execute(
                    """
                    INSERT INTO ttnn_ops.ttnn_hardware (board_type, device_series, card_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
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
            hardware_id = hardware_cache.get(hw_key)

            # Compute config hash (WITHOUT mesh - mesh is linked via junction table)
            config_hash = compute_config_hash(op_name, arguments, hw_key if hardware_id else None)

            # Extract primary tensor info from arg0
            primary_info = {}
            if arguments:
                for arg in arguments:
                    if isinstance(arg, dict):
                        for k, v in arg.items():
                            if isinstance(v, dict) and "Tensor" in v:
                                info = extract_tensor_info(v)
                                if info:
                                    primary_info = info
                                    break
                        if primary_info:
                            break

            # Insert configuration (without mesh_config_id)
            try:
                cur.execute(
                    """
                    INSERT INTO ttnn_ops.ttnn_configuration
                    (operation_id, hardware_id, primary_dtype, primary_storage_type, primary_layout,
                     primary_memory_layout, primary_buffer_type, primary_shape, config_hash, full_config_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (config_hash) DO UPDATE SET last_seen_ts = NOW()
                    RETURNING ttnn_configuration_id
                """,
                    (
                        op_id,
                        hardware_id,
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
                total_configs += 1

                # Link config to model
                cur.execute(
                    """
                    INSERT INTO ttnn_ops.ttnn_configuration_model (configuration_id, model_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """,
                    (config_id, model_id),
                )

                # Link config to default 1x1 mesh (always)
                cur.execute(
                    """
                    INSERT INTO ttnn_ops.ttnn_configuration_mesh (configuration_id, mesh_config_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """,
                    (config_id, default_mesh_id),
                )
                total_mesh_links += 1

                # Parse and link additional mesh configs from tensor_placements
                tensor_placements = machine_info.get("tensor_placements", [])
                for placement in tensor_placements:
                    mesh_shape_str = placement.get("mesh_device_shape")
                    mesh_shape = parse_array_value(mesh_shape_str)

                    if mesh_shape and mesh_shape != [1, 1]:
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

                        # Get or create this mesh config
                        mesh_config_id = get_or_create_mesh_config(
                            cur,
                            mesh_config_cache,
                            mesh_shape,
                            device_count,
                            placement_type,
                            shard_dim,
                            distribution_shape,
                        )

                        if mesh_config_id:
                            # Link config to this mesh
                            cur.execute(
                                """
                                INSERT INTO ttnn_ops.ttnn_configuration_mesh (configuration_id, mesh_config_id)
                                VALUES (%s, %s)
                                ON CONFLICT DO NOTHING
                            """,
                                (config_id, mesh_config_id),
                            )
                            total_mesh_links += 1

                # Insert arguments
                for i, arg in enumerate(arguments):
                    if isinstance(arg, dict):
                        arg_name = list(arg.keys())[0]
                        arg_value = arg[arg_name]

                        # Check for unsupported type strings
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
                            isinstance(arg_value, list) and len(arg_value) > 0 and isinstance(arg_value[0], dict)
                        )

                        cur.execute(
                            """
                            INSERT INTO ttnn_ops.ttnn_argument
                            (configuration_id, arg_index, arg_name, is_tensor, is_tensor_list, tensor_count,
                             tensor_dtype, tensor_storage_type, tensor_layout, tensor_memory_layout,
                             tensor_buffer_type, tensor_shape, shard_shape, shard_orientation,
                             core_grid_x, core_grid_y, scalar_value_json, unsupported_type_string)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
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
                continue

        conn.commit()
        print(f"  Loaded {op_name}: {len(op_data.get('configurations', []))} configs")

    conn.commit()
    print(f"\n✅ Loaded {total_configs} configurations, {total_args} arguments, {total_mesh_links} mesh links")

    # Print stats
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_operation")
    print(f"   Operations: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_model")
    print(f"   Models: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_configuration")
    print(f"   Configurations: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_argument")
    print(f"   Arguments: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_mesh_config")
    print(f"   Mesh Configs: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ttnn_ops.ttnn_configuration_mesh")
    print(f"   Config-Mesh Links: {cur.fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    load_data()
