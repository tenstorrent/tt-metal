# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recompute precision-specific context records from measured full-model capacity."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected")
    args = parser.parse_args()
    model = Path(__file__).resolve().parents[1]
    root = model / "doc/datatype_sweep"
    target = model / "doc/context_contract.json"
    contract = json.loads(target.read_text())
    records = {}
    for path in sorted(root.glob("capacity_*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("status") != "pass" or not data.get("all_layers"):
            continue
        policy = data["runtime_policy"]["requested"]
        assert data["context"] == contract["hf_advertised_context"] == 262144
        assert data["full_context_prefill_length"] == 262144
        assert data["terminal_decode_position"] == 262143
        prepared = data["memory"]["prepared_live_traces"]
        allocated = prepared["total_bytes_allocated_per_bank"] * prepared["num_banks"]
        free = prepared["total_bytes_free_per_bank"] * prepared["num_banks"]
        spec_bytes = sum(
            # One K + one V page: 8 tiles each per attention layer per chip.
            16 * {"bfloat8_b": 1088, "bfloat16": 2048}[spec["key_dtype"]]
            for spec in data["cache_spec"].values()
        )
        records[policy["id"]] = {
            "status": "capacity_passed",
            "supported_context": 262144,
            "capability_reduction": None,
            "kv_dtype": policy["kv_cache_dtype"],
            "precision_policy": policy,
            "kv_bytes_per_physical_block_per_device": spec_bytes,
            "full_layer_kv_bytes_per_user_per_device": spec_bytes * 8192,
            "measured_all_persistent_allocator_bytes_per_device": allocated,
            "measured_free_bytes_per_device_after_preparation": free,
            "memory_view_semantics": "MeshDevice symmetric allocator; bank fields multiplied by8 per chip, not summed across TP ranks",
            "largest_contiguous_free_bytes_per_bank": prepared["largest_contiguous_bytes_free_per_bank"],
            "largest_tested_prefill": 262144,
            "largest_tested_decode_position": 262143,
            "tested_non_aligned_long_lengths": [262143],
            "capacity_reserve_bytes_per_device": data["budget_reserve_bytes_per_device"],
            "validated_total_token_capacity_batch1": data["validated_total_token_capacity"],
            "max_tested_batch_note": "262144 per request; does not claim32 simultaneous maximum-context requests fit",
            "trace_allocation_tracking": data["trace_allocation_tracking"],
            "evidence": str(path.relative_to(model / "doc")),
        }
    contract["datatype_sweep_candidates"] = records
    if args.selected:
        if args.selected not in records:
            raise ValueError("Selected policy needs its own full-context capacity record")
        contract["datatype_sweep"] = records[args.selected]
    target.write_text(json.dumps(contract, indent=2) + "\n")
    print("CONTEXT_RECOMPUTED", list(records), "selected", args.selected)


if __name__ == "__main__":
    main()
