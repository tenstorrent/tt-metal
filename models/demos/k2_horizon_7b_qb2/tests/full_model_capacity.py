"""Exact persistent tensor storage arithmetic for the TP4 full model."""

import argparse
import hashlib
import json
from pathlib import Path

TILE_BYTES = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048}


def packed_tensor_bytes(shape, dtype):
    """One rank's tile-padded K/N storage, including block-float exponents."""
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError(f"Expected positive integer K/N dimensions, got {shape!r}")
    if dtype not in TILE_BYTES:
        raise ValueError(f"Unsupported packed dtype: {dtype!r}")
    return ((shape[0] + 31) // 32) * ((shape[1] + 31) // 32) * TILE_BYTES[dtype]


def layer_storage(policy, weight_allocations, *, context=524288):
    """Count an exported layer without importing the model or opening devices."""
    if not weight_allocations:
        raise ValueError("A layer must export its actual weight allocations")
    weights = 0
    for role, mode, shape, dtype in weight_allocations:
        if role not in {"qkv", "o", "mlp", "down", "swiglu"} or mode not in {"prefill", "decode"}:
            raise ValueError(f"Unexpected weight allocation role/mode: {role!r}/{mode!r}")
        # Repeated MLP entries are separate gate/up allocations; never deduplicate.
        weights += packed_tensor_bytes(shape, dtype)
    kv_dtype = policy["kv"]
    kv = 2 * ((context + 31) // 32) * 2 * packed_tensor_bytes((32, 128), kv_dtype)
    gather_dtypes = [policy["prefill_qkv_activation"], policy["prefill_mlp_activation"]]
    # A layer retains one shape/dtype at a time, so use max(QKV, MLP), not their sum.
    gather = max(packed_tensor_bytes((4096, 4096), dtype) for dtype in gather_dtypes)
    return {
        "decoder_weights": weights,
        "kv_batch1_max_context": kv,
        "max_prefill_gather_storage": gather,
        "kv_dtype": kv_dtype,
        "prefill_gather_dtypes": gather_dtypes,
    }


def policy_storage(artifact, *, expected_layers=36, context=524288):
    policies = artifact.get("runtime_policies", [])
    allocations = artifact.get("weight_allocations", [])
    if (
        artifact.get("layers") != expected_layers
        or len(policies) != expected_layers
        or len(allocations) != expected_layers
    ):
        raise ValueError(f"Policy artifact must export all {expected_layers} layers and their weight allocations")
    return [layer_storage(policy, weights, context=context) for policy, weights in zip(policies, allocations)]


def ledger(policy_artifact=None, *, optimized=False):
    context, layers = 524288, 36
    rows = {
        "decoder_weights": 68419584 * layers,
        "hidden_sharded_bf16_embedding": 250624 * 1024 * 2,
        "vocab_sharded_bfp8_head_prefill": (4096 // 32) * (65536 // 32) * 1088,
        "vocab_sharded_bfp8_head_decode": 4 * (4096 // 32) * (16384 // 32) * 1088,
        "final_norm_tile_padded_bf16": 32 * 1024 * 2,
        "absolute_rope_cos_sin_bf16": 2 * context * 128 * 2,
        "kv_bfp8_batch1_max_context": layers * 2 * (context // 32) * 2 * 4 * 1088,
        "page_table_int32_batch1": context // 32 * 4,
        "max_prefill_gather_storage": layers * (4096 // 32) * (4096 // 32) * 1088,
        "trace_region": 200000000,
        "shared_decode_pool_one_bucket_dram": 327680,
        # Conservative allowance beyond individually counted persistent tensors:
        # masks/parameter buffers, temporary logits/activation/CCL buffers and
        # collective pool storage (obsolete batch buckets are now retired). No weights or KV hidden here.
        "additional_workspace_reserve": 6 * 1024**3,
    }
    per_layer = None
    if optimized:
        rows.update(
            uint32_token_history_max_context=context * 32 * 4,
            uint32_history_cursor=32 * 4,
            prepared_prefill_tokens_and_positions=2 * 4096 * 4,
            prepared_prefill_page_table=context // 32 * 4,
            prepared_prefill_logits_bf16=32 * 65536 * 2,
        )
    policy_metadata = {}
    kv_dtype = "BFP8_B"
    if policy_artifact is not None:
        policy_path = Path(policy_artifact)
        raw = policy_path.read_bytes()
        artifact = json.loads(raw)
        per_layer = policy_storage(artifact, expected_layers=layers, context=context)
        rows["decoder_weights"] = sum(layer["decoder_weights"] for layer in per_layer)
        rows["max_prefill_gather_storage"] = sum(layer["max_prefill_gather_storage"] for layer in per_layer)
        kv_dtypes = {layer["kv_dtype"] for layer in per_layer}
        kv_dtype = next(iter(kv_dtypes)).upper() if len(kv_dtypes) == 1 else "mixed"
        if kv_dtypes == {"bfloat8_b"}:
            kv_dtype = "BFP8_B"
            rows["kv_bfp8_batch1_max_context"] = sum(layer["kv_batch1_max_context"] for layer in per_layer)
        else:
            del rows["kv_bfp8_batch1_max_context"]
            rows["kv_mixed_batch1_max_context"] = sum(layer["kv_batch1_max_context"] for layer in per_layer)
        policy_metadata = {
            "policy_artifact": str(policy_path),
            "policy_artifact_sha256": hashlib.sha256(raw).hexdigest(),
            "per_layer_storage_bytes_per_device": per_layer,
            "prefill_gather_buffer_policy": "One shape/dtype at a time per layer; sum of each layer's larger QKV/MLP 4096x4096 packed gather.",
        }
    return {
        "supported_context": context,
        "hf_advertised_context": context,
        "target_mesh": [1, 4],
        "num_layers": layers,
        "kv_dtype": kv_dtype,
        "page_size": 32,
        "local_kv_heads": 2,
        "head_policy": (
            "BFP8/HiFi2/FP32 accumulator, TP4 vocabulary, 8x8192 local splits, K4/N8, 1 reader/bank"
            if optimized
            else "BFP8/HiFi2/FP32 accumulator, TP4 vocabulary, 4x16384 local splits, K2/N16, 1 reader/bank"
        ),
        "storage_bytes_per_device": rows,
        "total_reserved_bytes_per_device": sum(rows.values()),
        "exposed_dram_bytes_per_device": 34225520640,
        "headroom_bytes_per_device": 34225520640 - sum(rows.values()),
        "capacity_reduction": False,
        "joint_batch_context_contract": "1..32 fixed slots, disjoint owned physical pages; joint cache allocation must fit physical DRAM",
        "status": "calculated; full-stack allocation probe pending",
        **policy_metadata,
    }


def validate_capacity_evidence(evidence, *, policy=None, policy_sha256=None):
    """Reject historical or unrelated capacity results for a selected policy."""
    capacity = evidence.get("capacity", {})
    if (
        evidence.get("pass") is not True
        or capacity.get("layers") != 36
        or capacity.get("context") != 524288
        or capacity.get("cache_pairs") != 36
        or evidence.get("late_context", {}).get("finite_logits") is not True
    ):
        raise ValueError("Capacity evidence must pass full36 max-context allocation and finite late-context execution")
    if policy is not None:
        matching_exports = (
            evidence.get("runtime_policies") == policy["runtime_policies"]
            and evidence.get("weight_allocations") == policy["weight_allocations"]
        )
        matching_artifact = policy_sha256 is not None and evidence.get("policy_artifact_sha256") == policy_sha256
        if not (matching_exports or matching_artifact):
            raise ValueError(
                "Capacity evidence must identify the selected policy by matching runtime exports or artifact SHA256"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    doc = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        default=doc / "selected_policy.json",
        help="Full36 runtime_policies and actual weight_allocations export",
    )
    parser.add_argument(
        "--capacity-artifact",
        type=Path,
        default=doc / "context_selected.json",
        help="Exact max-context probe for the selected policy",
    )
    parser.add_argument("--optimized", action="store_true")
    args = parser.parse_args()
    path = Path("models/demos/k2_horizon_7b_qb2/doc/context_contract.json")
    obj = json.loads(path.read_text())
    section = "optimized_full_model" if args.optimized else "full_model"
    obj[section] = ledger(args.policy_artifact, optimized=args.optimized)
    # The CLI always uses the selected policy. ledger() without an artifact is
    # retained only for explicit historical baseline arithmetic in CPU analyses.
    probe = args.capacity_artifact
    if probe is not None:
        evidence = json.loads(probe.read_text())
        if args.optimized and not evidence.get("optimized_buffers", {}).get("max_history_and_full_cache_coexist"):
            raise ValueError("Optimized capacity evidence must include max-size history and prepared prefill buffers")
        policy = json.loads(args.policy_artifact.read_text()) if args.policy_artifact is not None else None
        validate_capacity_evidence(evidence, policy=policy, policy_sha256=obj[section].get("policy_artifact_sha256"))
        dram = evidence["capacity"]["dram"]
        arena_bytes = dram["num_banks"] * dram["total_bytes_per_bank"]
        obj[section].update(
            exposed_dram_bytes_per_device=arena_bytes,
            headroom_bytes_per_device=arena_bytes - obj[section]["total_reserved_bytes_per_device"],
            dram_accounting_note="Observed allocator arena from this exact mesh/trace configuration. Explicit trace-region bytes remain in the reserve even where already carved from this arena, making headroom conservative.",
            status="full-stack allocation and late-context execution validated",
            capacity_probe=evidence["capacity"],
            capacity_probe_policy_note="Explicit capacity evidence for this ledger; changed decoder policies require matching runtime exports or exact policy artifact SHA256. The 6GiB workspace reserve is retained.",
            late_context_probe=evidence["late_context"],
            evidence=[str(probe)],
            execution_scope="36 layers with all weights and max-context KV; initialized-zero prefix for late-context structural test. This does not claim an all-layer 524288-token HF quality comparison.",
        )
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps(obj[section], indent=2))


if __name__ == "__main__":
    main()
