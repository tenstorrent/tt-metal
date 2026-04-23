#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize master/sweep JSON arguments and recompute config hashes.

Standalone script that can run without ttnn or tqdm installed.  Used by
the validation workflow to canonicalize argument format in both master and
sweep JSONs before the validator compares them.

Usage:
    python model_tracer/normalize_for_validation.py <json_file> [<json_file2> ...]
"""

import copy
import hashlib
import json
import re
import sys


# ---------------------------------------------------------------------------
# Placement canonicalization
# ---------------------------------------------------------------------------

def _canonicalize_placement_str(placement_str):
    """Canonicalize a placement string to a stable representation.

    Handles old ``"PlacementShard(0)"``, new ``"PlacementShard(dim=0)"``,
    and list formats ``"['PlacementShard(0)', ...]"``.
    """
    entries = re.findall(
        r"Placement(?:Shard\((?:dim=)?-?\d+\)|Replicate(?:\(\))?)",
        str(placement_str),
    )
    canonical = []
    for e in entries:
        e = re.sub(r"PlacementShard\(dim=(-?\d+)\)", r"PlacementShard(\1)", e)
        e = e.replace("PlacementReplicate()", "PlacementReplicate")
        canonical.append(e)
    return canonical or [str(placement_str)]


# ---------------------------------------------------------------------------
# Argument normalization (mirrors _normalize_for_hash in generic_ops_tracer)
# ---------------------------------------------------------------------------

_OBJECT_ADDR_RE = re.compile(r"\bat\s+0x[0-9a-fA-F]+\b")


def _normalize_for_hash(obj):
    """Normalize arguments in-place for stable comparison and hashing."""
    if isinstance(obj, dict):
        # memory_config.hash is a device-specific pointer
        if "hash" in obj and isinstance(obj["hash"], int):
            del obj["hash"]

        # shard_spec: canonicalize None → string "None"
        if "shard_spec" in obj and obj["shard_spec"] is None:
            obj["shard_spec"] = "None"

        # Canonicalize tensor_placement dicts
        if "tensor_placement" in obj and isinstance(obj["tensor_placement"], dict):
            tp = obj["tensor_placement"]
            if "placement" in tp:
                tp["placement"] = _canonicalize_placement_str(tp["placement"])
            for shape_key in ("distribution_shape", "mesh_device_shape"):
                val = tp.get(shape_key)
                if isinstance(val, str):
                    try:
                        tp[shape_key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass

        # Strip metadata keys (non-functional)
        for meta_key in ("original_dtype", "original_shape", "storage_type"):
            obj.pop(meta_key, None)

        # Strip output/return-value tensors
        for output_key in ("output_tensor", "indices_tensor", "attention_sink"):
            obj.pop(output_key, None)

        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, str):
                obj[k] = _OBJECT_ADDR_RE.sub("", v)
            else:
                _normalize_for_hash(v)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = _OBJECT_ADDR_RE.sub("", item)
            else:
                _normalize_for_hash(item)


# ---------------------------------------------------------------------------
# Hash computation (mirrors _compute_config_hash in generic_ops_tracer)
# ---------------------------------------------------------------------------

def _extract_hardware_and_mesh(machine_info):
    """Extract hash-relevant hardware and mesh fields from machine_info."""
    hardware = None
    if machine_info:
        board_type = machine_info.get("board_type")
        if board_type:
            device_series = machine_info.get("device_series")
            if isinstance(device_series, list):
                device_series = device_series[0] if device_series else None
            hardware = (board_type, device_series, machine_info.get("card_count", 1))

    mesh_config = None
    if machine_info and "tensor_placements" in machine_info:
        placements = machine_info.get("tensor_placements", [])
        if placements:
            placement = placements[0]
            mesh_shape_value = placement.get("mesh_device_shape")
            if mesh_shape_value:
                try:
                    mesh_shape = json.loads(mesh_shape_value) if isinstance(mesh_shape_value, str) else mesh_shape_value
                    if mesh_shape:
                        placement_str = placement.get("placement", "")
                        shard_dim = None
                        if "PlacementShard" in placement_str:
                            match = re.search(r"PlacementShard\((?:dim=)?(\d+)\)", placement_str)
                            if match:
                                shard_dim = int(match.group(1))
                        mesh_config = {
                            "mesh_shape": mesh_shape,
                            "placement_type": "shard" if shard_dim is not None else "replicate",
                            "shard_dim": shard_dim,
                        }
                except Exception:
                    pass

    return hardware, mesh_config


def _compute_config_hash(op_name, op_args, machine_info):
    """Compute stable config hash."""
    hardware, mesh_config = _extract_hardware_and_mesh(machine_info)
    hash_args = copy.deepcopy(op_args)
    _normalize_for_hash(hash_args)
    normalized = {
        "operation": op_name,
        "arguments": hash_args,
        "hardware": hardware,
        "mesh": mesh_config,
    }
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_and_recompute(json_file):
    """Normalize arguments and recompute config_hash in a JSON file."""
    print(f"🔧 Normalizing {json_file}...")

    with open(json_file, "r") as f:
        data = json.load(f)

    hash_updated = 0
    for op_name, op_data in data.get("operations", {}).items():
        for config in op_data.get("configurations", []):
            # Normalize raw arguments in-place
            args = config.get("arguments", {})
            _normalize_for_hash(args)

            # Recompute config_hash
            old_hash = config.get("config_hash")
            machine_info = None
            executions = config.get("executions", [])
            if executions and isinstance(executions[0], dict):
                machine_info = executions[0].get("machine_info")
            new_hash = _compute_config_hash(op_name, args, machine_info)
            if new_hash != old_hash:
                config["config_hash"] = new_hash
                hash_updated += 1

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"✅ Done: {hash_updated} hashes updated")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <json_file> [<json_file2> ...]", file=sys.stderr)
        return 1

    for path in sys.argv[1:]:
        normalize_and_recompute(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
