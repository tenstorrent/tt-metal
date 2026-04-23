#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare master JSON for validation by aligning it with what the sweep tracer produces.

The master JSON (from DB reconstruction) may contain fields that the sweep tracer
does not capture, causing spurious diffs in the validator. This script:

1. Fixes shard_spec format: "None" (string) → null
2. Strips output metadata keys from arguments that the tracer's _normalize_for_hash()
   strips but the validator's normalize() doesn't
3. Aligns argument keys to match what each sweep test actually passes to the TTNN op:
   - Keeps memory_config for ops where sweep tests explicitly inject it
   - Strips keys that sweep tests never pass (layout, compute_kernel_config, etc.)
   - Renames named kwargs to positional arg format where sweep tests use positional args
   - Derives memory_config from tensor args when sweep test injects it but master lacks it
4. Populates tensor_placements in machine_info from arguments
5. Recomputes config_hash using the same _compute_config_hash logic as the sweep tracer
6. Deduplicates configs by hash to avoid hash collision mismatches

This must run BEFORE sweep vector generation, since vectors carry config_hash as
sweep_source_hash for the join in validate_sweep_trace.py.

Only uses stdlib + model_tracer modules (no ttnn/torch) — safe to run on ubuntu-latest.
"""

import json
import os
import sys

# Ensure repo root is on sys.path for model_tracer imports
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from model_tracer.generic_ops_tracer import _compute_config_hash  # noqa: E402


# ---------------------------------------------------------------------------
# Master JSON preparation
# ---------------------------------------------------------------------------

# Keys to strip from top-level arguments — these are output metadata that
# _normalize_for_hash() strips for hashing but the validator's normalize()
# doesn't strip, causing extra_key diffs.
_STRIP_FROM_ARGS = frozenset({
    "output_tensor",
    "indices_tensor",
    "attention_sink",
    "original_dtype",
    "original_shape",
    "storage_type",
})

# Keys to strip from ALL ops — these are never present in sweep traces.
# They are either infrastructure or handled separately by sweep tests.
_ALWAYS_STRIP = frozenset({
    "layout",  # tensor creation parameter, not an op kwarg
})

# Per-op keys to strip — present in master but not in sweep traces because
# the sweep test handles them separately (as explicit params, not via op_kwargs).
_OP_STRIP_KEYS = {
    "ttnn.matmul": {"compute_kernel_config", "program_config"},
    "ttnn.linear": {"compute_kernel_config", "program_config"},
    "ttnn.rms_norm": {"compute_kernel_config"},
}

# Ops where the sweep test explicitly injects memory_config to the op.
# For these ops, memory_config should be KEPT in the master (not stripped).
# If the master doesn't have memory_config, derive it from the first tensor arg.
_OPS_WITH_MEMORY_CONFIG = frozenset({
    "ttnn.matmul",
    "ttnn.linear",
    "ttnn.rms_norm",
    "ttnn.transpose",
    "ttnn.experimental.rotary_embedding_llama",
})

# Ops where named kwargs need to be renamed to positional arg format
# to match what the sweep test passes as positional arguments.
_KWARG_RENAMES = {
    "ttnn.scatter": {"dim": "arg1", "index": "arg2", "src": "arg3"},
    "ttnn.topk": {"k": "arg1"},
    "ttnn.slice": {"starts": "arg1", "ends": "arg2", "steps": "arg3"},
}

# Per-op keys to strip that are specific to certain ops
_OP_EXTRA_STRIP = {
    "ttnn.add": {"dtype"},  # add sweep test consumes dtype param but doesn't pass to op
    "ttnn.reshape": {"arg2"},  # reshape sweep test only uses arg0, arg1
}

# Ops where the sweep test always passes dtype and memory_config as explicit kwargs
# (even when None), so the master should preserve them.
_OPS_WITH_DTYPE_MEMCFG = frozenset({
    "ttnn.embedding",
})


def _fix_shard_spec(obj):
    """Recursively convert "shard_spec": "None" (string) to null."""
    if isinstance(obj, dict):
        if "shard_spec" in obj and obj["shard_spec"] == "None":
            obj["shard_spec"] = None
        for v in obj.values():
            _fix_shard_spec(v)
    elif isinstance(obj, list):
        for item in obj:
            _fix_shard_spec(item)


def _get_tensor_memory_config(arguments):
    """Extract memory_config from the first tensor argument."""
    for key in sorted(arguments.keys()):
        val = arguments.get(key)
        if isinstance(val, dict) and "memory_config" in val:
            mc = val["memory_config"]
            if isinstance(mc, dict) and ("buffer_type" in mc or "memory_layout" in mc):
                return mc
    return None


def _strip_and_align(op_name, arguments):
    """Strip output metadata and align argument keys to match sweep trace format.

    Modifies arguments dict in-place. Returns True if any keys were changed.
    """
    changed = False

    # 1. Strip output metadata (always, all ops)
    for key in _STRIP_FROM_ARGS:
        if key in arguments:
            del arguments[key]
            changed = True

    # 2. Strip universally-removed keys
    for key in _ALWAYS_STRIP:
        if key in arguments:
            del arguments[key]
            changed = True

    # 3. Strip op-specific keys
    for key in _OP_STRIP_KEYS.get(op_name, set()):
        if key in arguments:
            del arguments[key]
            changed = True

    for key in _OP_EXTRA_STRIP.get(op_name, set()):
        if key in arguments:
            del arguments[key]
            changed = True

    # 4. Handle memory_config alignment
    if op_name in _OPS_WITH_MEMORY_CONFIG:
        mc = arguments.get("memory_config")
        if mc is None and "memory_config" not in arguments:
            # Master doesn't have memory_config at all — derive from first tensor
            derived = _get_tensor_memory_config(arguments)
            if derived is not None:
                arguments["memory_config"] = derived
                changed = True
        elif mc is None:
            # memory_config key exists but value is None — derive from first tensor
            derived = _get_tensor_memory_config(arguments)
            if derived is not None:
                arguments["memory_config"] = derived
                changed = True
            else:
                del arguments["memory_config"]
                changed = True
        # else: mc is a dict — keep it as-is
    elif op_name in _OPS_WITH_DTYPE_MEMCFG:
        # For embedding: sweep always passes dtype and memory_config (even None).
        # Ensure they exist in the master; don't strip None values for these keys.
        pass  # handled below in None-stripping (skip these keys)
    else:
        # For other ops: strip memory_config (build_op_kwargs filters it)
        mc = arguments.get("memory_config")
        if isinstance(mc, dict) and ("buffer_type" in mc or "memory_layout" in mc):
            del arguments["memory_config"]
            changed = True

    # 5. Rename named kwargs to positional format
    renames = _KWARG_RENAMES.get(op_name, {})
    for old_name, new_name in renames.items():
        if old_name in arguments and new_name not in arguments:
            arguments[new_name] = arguments.pop(old_name)
            changed = True

    # 6. Handle typecast: rename 'dtype' to 'arg1' when dtype is a dict
    #    (represents the target dtype, passed as positional arg by sweep test)
    if op_name == "ttnn.typecast" and "dtype" in arguments and "arg1" not in arguments:
        arguments["arg1"] = arguments.pop("dtype")
        changed = True

    # 6b. Slice: add default steps (arg3) if missing.
    #     The sweep test always passes steps=[1]*len(shape) to ttnn.slice,
    #     so the tracer captures it.  If the master doesn't have it, add it.
    if op_name == "ttnn.slice" and "arg3" not in arguments:
        # Derive step size from arg1 (starts) length
        starts = arguments.get("arg1")
        if isinstance(starts, (list, tuple)):
            arguments["arg3"] = [1] * len(starts)
            changed = True

    # 6d. Normalize nested storage_type to match sweep test behavior.
    #     The sweep test defaults to StorageType::DEVICE when the top-level
    #     storage_type is None (stripped above).  If the master has a nested
    #     arg0.storage_type=HOST but no top-level storage_type to tell the
    #     sweep test to use HOST, the sweep will create DEVICE tensors.
    #     Update the master's nested metadata to match.
    if "storage_type" not in arguments:  # top-level was stripped or absent
        for key in arguments:
            val = arguments[key]
            if isinstance(val, dict) and val.get("storage_type") == "StorageType.HOST":
                val["storage_type"] = "StorageType.DEVICE"
                changed = True

    # 7. Strip None-valued top-level kwargs.
    #    build_op_kwargs() skips None values, so sweep tracer never sees them.
    #    Skip positional args (arg0, arg1, ...) and special keys for ops that
    #    always pass them (embedding passes dtype=None, memory_config=None).
    preserve_none_keys = set()
    if op_name in _OPS_WITH_DTYPE_MEMCFG:
        preserve_none_keys = {"dtype", "memory_config"}

    for key in list(arguments.keys()):
        if arguments[key] is None:
            if key.startswith("arg") and key[3:].isdigit():
                continue  # keep positional args
            if key in preserve_none_keys:
                continue  # keep special keys for this op
            del arguments[key]
            changed = True

    return changed


def _populate_tensor_placements(machine_info, arguments):
    """Populate tensor_placements in machine_info from arguments.

    This mirrors the logic in generic_ops_tracer.convert_json_to_master_format
    that builds global tensor_placements from per-tensor placement data.
    Without this, the DB-reconstructed machine_info lacks tensor_placements,
    causing mesh_config=None in hash computation.
    """
    if machine_info is None:
        return
    _tensor_placements = []
    for arg_val in arguments.values():
        tp = arg_val.get("tensor_placement") if isinstance(arg_val, dict) else None
        if tp and isinstance(tp, dict):
            _tensor_placements.append(tp)
            break  # only need the first tensor's placement for the global key
    if _tensor_placements:
        machine_info["tensor_placements"] = _tensor_placements


# Known mesh topologies: maps total device count → canonical 2D shape.
# Used to convert flattened 1D shapes (e.g., [1, 32]) to the actual topology.
_KNOWN_MESH_SHAPES = {
    32: [4, 8],   # Galaxy
    8: [1, 8],    # T3000 / 8-device
    2: [1, 2],    # n300
}


def _canonicalize_mesh_shape(mesh_shape):
    """Convert flattened 1D mesh shapes to canonical 2D shapes.

    E.g., [1, 32] → [4, 8] for Galaxy, [32] → [4, 8].
    Returns the canonical shape or the original if no mapping exists.
    """
    total = 1
    for d in mesh_shape:
        total *= d
    # Only canonicalize if it's a 1D-flattened shape (has a 1 dimension or is 1D)
    is_flat = len(mesh_shape) == 1 or (len(mesh_shape) == 2 and mesh_shape[0] == 1)
    if is_flat and total in _KNOWN_MESH_SHAPES:
        return _KNOWN_MESH_SHAPES[total]
    return mesh_shape


def _normalize_mesh_device_shapes(arguments, actual_mesh_str):
    """Normalize per-tensor mesh_device_shape to match the actual device mesh.

    The DB may store a flattened or 1D mesh shape (e.g., "[1, 32]") while the
    sweep tracer records the actual mesh shape (e.g., "[4, 8]").  When both
    represent the same total number of devices, replace the stored value with
    the actual mesh shape string so the validator comparison matches.
    """
    try:
        actual_mesh = json.loads(actual_mesh_str) if isinstance(actual_mesh_str, str) else actual_mesh_str
        actual_total = 1
        for d in actual_mesh:
            actual_total *= d
    except Exception:
        return

    for val in arguments.values():
        if not isinstance(val, dict):
            continue
        tp = val.get("tensor_placement")
        if not isinstance(tp, dict):
            continue
        mds = tp.get("mesh_device_shape")
        if mds is None:
            continue
        try:
            stored = json.loads(mds) if isinstance(mds, str) else mds
            stored_total = 1
            for d in stored:
                stored_total *= d
        except Exception:
            continue
        # If same total devices but different shape, use actual
        if stored_total == actual_total and stored != actual_mesh:
            tp["mesh_device_shape"] = json.dumps(actual_mesh) if isinstance(mds, str) else actual_mesh


def _dedup_configs(configs):
    """Keep only the first configuration per config_hash.

    Hash collisions occur because _normalize_for_hash() strips original_shape,
    original_dtype, and storage_type before hashing, so configs with different
    shapes can get the same hash. The vector generator processes configs in order
    and picks the first one per hash. We must dedup the same way so the
    validator's master_index matches the vector that was actually executed.
    """
    seen = set()
    result = []
    for config in configs:
        ch = config.get("config_hash")
        if ch is None or ch not in seen:
            if ch is not None:
                seen.add(ch)
            result.append(config)
    return result


def prepare(data):
    """Prepare master JSON data for validation.

    Modifies data in-place:
    1. Fixes shard_spec format
    2. Aligns argument keys per-op to match sweep trace format
    3. Populates tensor_placements and recomputes config_hash
    4. Deduplicates configurations by hash
    """
    _fix_shard_spec(data)

    stats = {"args_stripped": 0, "hashes_changed": 0, "configs_deduped": 0}

    for op_name, op_info in data.get("operations", {}).items():
        for config in op_info.get("configurations", []):
            arguments = config.get("arguments", {})

            # Align argument keys to match sweep trace format
            if _strip_and_align(op_name, arguments):
                stats["args_stripped"] += 1

            # Get machine_info from executions
            machine_info = None
            executions = config.get("executions", [])
            if executions and isinstance(executions[0], dict):
                machine_info = executions[0].get("machine_info")

            # Populate tensor_placements from arguments (for hash computation)
            if machine_info is not None:
                _populate_tensor_placements(machine_info, arguments)

            # Normalize per-tensor mesh_device_shape to match actual mesh.
            # The DB may store a flattened shape (e.g., "[1, 32]") while the
            # sweep tracer records the actual mesh shape (e.g., "[4, 8]").
            # Also fix machine_info mesh shape: [1, N] → known 2D shape.
            if machine_info is not None:
                actual_mesh = machine_info.get("mesh_device_shape")
                if isinstance(actual_mesh, list):
                    canonical = _canonicalize_mesh_shape(actual_mesh)
                    if canonical != actual_mesh:
                        machine_info["mesh_device_shape"] = canonical
                    actual_str = json.dumps(canonical)
                    _normalize_mesh_device_shapes(arguments, actual_str)

            # Recompute config_hash using the same function as the sweep tracer
            old_hash = config.get("config_hash")
            new_hash = _compute_config_hash(op_name, arguments, machine_info)
            if new_hash != old_hash:
                config["config_hash"] = new_hash
                stats["hashes_changed"] += 1

        # Dedup after hash recomputation — keep first per hash
        configs = op_info.get("configurations", [])
        deduped = _dedup_configs(configs)
        removed = len(configs) - len(deduped)
        if removed:
            op_info["configurations"] = deduped
            stats["configs_deduped"] += removed

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_master_for_validation.py <json_file>", file=sys.stderr)
        return 1

    json_file = sys.argv[1]
    with open(json_file) as f:
        data = json.load(f)

    stats = prepare(data)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"Prepared {json_file} for validation:")
    print(f"  Arguments stripped: {stats['args_stripped']}")
    print(f"  Hashes recomputed: {stats['hashes_changed']}")
    print(f"  Configs deduped: {stats['configs_deduped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
