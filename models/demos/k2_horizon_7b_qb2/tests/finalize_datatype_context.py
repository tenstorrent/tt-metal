"""Recompute stage8 capacity from actual selected-policy storage exports."""

import argparse
import hashlib
import json
from pathlib import Path

from .full_model_capacity import ledger, packed_tensor_bytes, validate_capacity_evidence

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/datatype_sweep")


def candidate_ledger(path):
    runtime = json.loads(path.read_text())
    config = runtime["dtype_policy"]
    result = ledger(path, optimized=True)
    rows = result["storage_bytes_per_device"]
    for key in ["vocab_sharded_bfp8_head_prefill", "vocab_sharded_bfp8_head_decode"]:
        del rows[key]
    head_dtype = config["head"]["weight_dtype"]
    rows["vocab_sharded_head_prefill"] = packed_tensor_bytes((4096, 65536), head_dtype)
    rows["vocab_sharded_head_decode"] = 8 * packed_tensor_bytes((4096, 8192), head_dtype)
    result["head_policy"] = config["head"]
    result["total_reserved_bytes_per_device"] = sum(rows.values())
    result["exposed_dram_bytes_per_device"] = 33978715136
    result["headroom_bytes_per_device"] = result["exposed_dram_bytes_per_device"] - sum(rows.values())
    result["config_id"] = config["config_id"]
    assert result["headroom_bytes_per_device"] > 0
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selected-run", type=Path)
    p.add_argument("--capacity-artifact", type=Path)
    args = p.parse_args()
    summaries = {}
    paths = [*(DOC / "runs").glob("*.json"), *(DOC / "qualifications").glob("*_qualification.json")]
    for path in paths:
        artifact = json.loads(path.read_text())
        if artifact.get("layers") == 36 and len(artifact.get("runtime_policies", [])) == 36:
            summaries[artifact["config_id"]] = candidate_ledger(path)
    (DOC / "candidate_context_ledgers.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if args.selected_run:
        runtime = json.loads(args.selected_run.read_text())
        result = candidate_ledger(args.selected_run)
        evidence = json.loads(args.capacity_artifact.read_text())
        validate_capacity_evidence(evidence, policy=runtime)
        assert evidence["optimized_buffers"]["max_history_and_full_cache_coexist"]
        selected = DOC / "selected_precision_config.json"
        assert json.loads(selected.read_text()) == evidence["precision_config"] == runtime["dtype_policy"]
        dram = evidence["capacity"]["dram"]
        exposed = dram["num_banks"] * dram["total_bytes_per_bank"]
        result.update(
            exposed_dram_bytes_per_device=exposed,
            headroom_bytes_per_device=exposed - result["total_reserved_bytes_per_device"],
            status="full-stack allocation and late-context execution validated",
            capacity_probe=evidence["capacity"],
            late_context_probe=evidence["late_context"],
            optimized_buffers_probe=evidence["optimized_buffers"],
            evidence=[str(args.capacity_artifact)],
            selected_precision_artifact=str(selected),
            selected_precision_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
            execution_scope="36 layers; max-context cache plus token history/prepared prefill. Initialized-zero prefix for structural late-context execution; not full-context HF quality.",
        )
        assert result["headroom_bytes_per_device"] > 0
        contract = DOC.parent / "context_contract.json"
        obj = json.loads(contract.read_text())
        obj["datatype_sweep"] = result
        contract.write_text(json.dumps(obj, indent=2) + "\n")
        print("SELECTED_CONTEXT", result["supported_context"], "headroom", result["headroom_bytes_per_device"])


if __name__ == "__main__":
    main()
