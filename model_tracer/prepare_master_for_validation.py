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
   strips but the validator's normalize() doesn't: output_tensor, indices_tensor,
   attention_sink, original_dtype, original_shape, storage_type
3. Strips `memory_config` from top-level arguments — this is a kwarg that models pass
   to ops but the sweep framework's build_op_kwargs() intentionally filters out
   (most ops don't accept it as a kwarg)
4. Populates tensor_placements in machine_info from arguments (same as the tracer's
   convert_json_to_master_format does) so that hash recomputation produces correct
   mesh_config
5. Recomputes config_hash using the same _compute_config_hash logic as the sweep tracer

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

# Top-level kwargs that models pass but build_op_kwargs() filters out.
# The sweep tracer won't see these, so they cause extra_key diffs.
_STRIP_KWARGS = frozenset({
    "memory_config",
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


def _strip_output_metadata(arguments):
    """Strip output metadata and filtered kwargs from arguments dict.

    Aligns master arguments with what the sweep framework produces:
    - Strips output metadata keys (_normalize_for_hash strips these for hashing
      but the validator doesn't, causing extra_key diffs)
    - Strips memory_config kwarg (build_op_kwargs filters it)
    - Strips None-valued kwargs (build_op_kwargs skips None values)
    """
    for key in _STRIP_FROM_ARGS:
        arguments.pop(key, None)

    # Strip memory_config only when it's a top-level kwarg (dict with buffer_type etc).
    # This is a kwarg that models pass to ops but build_op_kwargs intentionally
    # filters out because most ops don't accept it as a keyword argument.
    for key in _STRIP_KWARGS:
        if key in arguments:
            val = arguments[key]
            if isinstance(val, dict) and ("buffer_type" in val or "memory_layout" in val):
                del arguments[key]

    # Strip None-valued top-level kwargs.  build_op_kwargs() skips None values
    # (line 258: "if value is None: continue"), so the sweep tracer never sees
    # them.  Examples: clamp's min=None, optional kwargs not specified by model.
    # Only strip named kwargs (not positional args like arg0, arg1, etc.)
    for key in list(arguments.keys()):
        if arguments[key] is None and not (key.startswith("arg") and key[3:].isdigit()):
            del arguments[key]


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


def prepare(data):
    """Prepare master JSON data for validation.

    Modifies data in-place:
    1. Fixes shard_spec format
    2. Strips output metadata from arguments
    3. Populates tensor_placements and recomputes config_hash
    """
    _fix_shard_spec(data)

    stats = {"args_stripped": 0, "hashes_changed": 0}

    for op_name, op_info in data.get("operations", {}).items():
        for config in op_info.get("configurations", []):
            arguments = config.get("arguments", {})

            # Strip output metadata and filtered kwargs
            keys_before = set(arguments.keys())
            _strip_output_metadata(arguments)
            if set(arguments.keys()) != keys_before:
                stats["args_stripped"] += 1

            # Get machine_info from executions
            machine_info = None
            executions = config.get("executions", [])
            if executions and isinstance(executions[0], dict):
                machine_info = executions[0].get("machine_info")

            # Populate tensor_placements from arguments (for hash computation)
            if machine_info is not None:
                _populate_tensor_placements(machine_info, arguments)

            # Recompute config_hash using the same function as the sweep tracer
            old_hash = config.get("config_hash")
            new_hash = _compute_config_hash(op_name, arguments, machine_info)
            if new_hash != old_hash:
                config["config_hash"] = new_hash
                stats["hashes_changed"] += 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
