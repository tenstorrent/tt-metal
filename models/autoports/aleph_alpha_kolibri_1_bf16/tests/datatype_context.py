# SPDX-License-Identifier: Apache-2.0
"""Recompute selected precision's context evidence from actual candidate allocations."""

import hashlib
import json
from pathlib import Path


def main():
    model = Path(__file__).resolve().parents[1]
    root = model / "doc/datatype_sweep"
    selected = json.loads((root / "selected_precision_config.json").read_text())
    final = json.loads((root / "selected_default/result.json").read_text())
    command = json.loads((root / "selected_default/run.command.json").read_text())
    assert command["exit_code"] == 0
    assert final["precision_config"] == selected
    assert final["status"] == "pass"
    assert final["runtime_summary"]["capacity"] == 1048576
    assert final["capability"]["capacity"] == 1048576
    assert final["long_prefix"]["initialized_full_prefix"]
    assert final["long_prefix"]["native_next_positions"] == {"kv": [1048576], "rope": [1048576]}
    assert final["batch32"]["layer_count"] == 50
    dtype = selected["runtime"]["kv_cache_dtype"]
    tile_bytes = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048}[dtype]
    cache_tiles = 2 * 128 * (10 * 1048576 + 40 * 8704) // 1024
    candidates = {}
    for path in (root / "candidates").glob("*/result.json"):
        result = json.loads(path.read_text())
        if "allocator" not in result:
            continue
        candidate_dtype = result["precision_config"]["runtime"]["kv_cache_dtype"]
        candidates[result["config_id"]] = dict(
            kv_cache_dtype=candidate_dtype,
            context=1048576,
            cache_bytes_per_device=cache_tiles
            * {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048}[candidate_dtype],
            allocator=result["allocator"],
            status=result["status"],
            evidence=str(path.relative_to(model / "doc")),
        )
    context_path = model / "doc/context_contract.json"
    contract = json.loads(context_path.read_text())
    contract["active_runtime_stage"] = "datatype_sweep"
    contract["active_precision_config"] = "datatype_sweep/selected_precision_config.json"
    contract["datatype_sweep"] = dict(
        status="validated",
        selected_config_id=selected["config_id"],
        selected_precision_config="datatype_sweep/selected_precision_config.json",
        selected_precision_sha256=hashlib.sha256((root / "selected_precision_config.json").read_bytes()).hexdigest(),
        supported_context=1048576,
        model_max_context=1048576,
        capability_reduction=None,
        target_mesh=[1, 4],
        batch_for_max_context=1,
        kv_cache_dtype=dtype,
        prefill_fill_dtype=dtype,
        decode_update_dtype="bfloat16",
        full_attention_layers=10,
        sliding_attention_layers=40,
        full_attention_cache_tokens=1048576,
        sliding_ring_tokens=8704,
        cache_page_tokens=32,
        cache_bytes_per_device=cache_tiles * tile_bytes,
        allocator=final["allocator"],
        candidate_capacity_evidence=candidates,
        runtime_default_context=selected["runtime"]["max_context"],
        validation=final["capability"],
        full_prefix_validation=final.get("long_prefix"),
        batch32_validation=final.get("batch32"),
        evidence=["datatype_sweep/selected_default/result.json"],
        inherited_full_prefix_evidence="full_model/full_validation_final.json",
        qualification="Fresh full-capacity allocation, non-aligned eager/traced prompts through8209 and final-address probe; the address probe does not initialize a1M prefix. Chunk/page geometry remains unchanged. Historical Stage6 full-prefix execution is retained as geometry evidence, not new-dtype long-context accuracy.",
    )
    if "long_prefix" in final:
        assert final["long_prefix"]["last_consumed_position"] == 1048575
        contract["datatype_sweep"][
            "qualification"
        ] = "Fresh selected-config all50-layer execution of every row of a1,048,543-token non-aligned prefix followed by34 generated tokens through the last valid address. Capacity test, not long-context language-quality scoring."
    context_path.write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
