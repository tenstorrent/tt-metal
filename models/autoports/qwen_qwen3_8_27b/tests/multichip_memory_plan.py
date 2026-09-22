# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only TP4 stored-tile capacity arithmetic; no device imports."""

import argparse
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["multichip_decoder", "optimized_multichip_decoder"], default="multichip_decoder"
    )
    parser.add_argument("--policy-file", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy_file.read_text()) if args.policy_file else {}
    stage = args.stage

    # BFP4 tile is 512 mantissa bytes plus 64 exponent bytes.
    def projection(k, n, role):
        prefill = ((k + 31) // 32) * ((n + 31) // 32) * 576
        readers = policy.get(role + "_readers", 1)
        bank_width = ((n + 256 * readers - 1) // (256 * readers)) * 32 * readers
        decode = ((k + 31) // 32) * (bank_width // 32) * 8 * 576
        return {
            "local_shape": [k, n],
            "bank_shard": [k, bank_width],
            "prefill_bytes": prefill,
            "decode_bytes": decode,
            "total_bytes": prefill + decode,
        }

    common = {
        n: projection(*shape, {"mlp_gate_up": "gate", "mlp_down": "down", "output": "output"}[n])
        for n, shape in {
            "mlp_gate_up": (5120, 8704),
            "mlp_down": (4352, 5120),
            "output": (1536, 5120),
        }.items()
    }
    kinds = {}
    for kind, n in [("linear_attention", 4160), ("full_attention", 3584)]:
        rows = {**common, "attention": projection(5120, n, "attention")}
        kinds[kind] = {"projections": rows, "weight_bytes": sum(v["total_bytes"] for v in rows.values())}
    context = 262144
    kv = 16 * 2 * (context // 32) * 1 * (256 // 32) * 1088
    recurrence = 48 * 12 * 128 * 128 * 4
    conv = 48 * 3 * 2560 * 2
    weights = 48 * kinds["linear_attention"]["weight_bytes"] + 16 * kinds["full_attention"]["weight_bytes"]
    # Reserve untied BF16 embedding + LM head, tensor-parallel, one copy each.
    terminal = 2 * 248320 * 5120 * 2 // 4
    # TILE-padded replicated norms (~40 MiB) and four conv taps (~30 MiB),
    # plus Q/K norms, FP32 delta constants, A and dt_bias. Reserve128 MiB.
    constants = 128 * 1024 * 1024
    # Unaligned concat may retain input, TILE chunks, RM chunks, RM output,
    # and retiled output; allow an additional retained original stack input.
    # Six streams (~15 GiB) plus2GiB trace/CCL and1GiB scratch.
    reserve = 18 * 1024**3
    total = weights + kv + recurrence + conv + terminal + constants + reserve
    result = {
        "status": "planned_not_capacity_validated",
        "mesh": [1, 4],
        "context": context,
        "projection_policy": "BFP4, prefill interleaved + decode bank-sharded copies",
        "layers": kinds,
        "all_decoder_projection_bytes_per_device": weights,
        "kv_bytes_per_device": kv,
        "kv_bytes_per_token_per_full_layer": 544,
        "recurrence_bytes_per_device_per_user": recurrence,
        "conv_bytes_per_device_per_user": conv,
        "embedding_lm_head_reserve_bytes_per_device": terminal,
        "norm_and_constant_reserve_bytes_per_device": constants,
        "trace_activation_reserve_bytes_per_device": reserve,
        "total_bytes_per_device": total,
        "capacity_probe_reserved_bytes_per_device": total - reserve + 2 * 1024**3,
        "capacity_probe_reservation_scope": "All planned persistent weights/KV/recurrent/terminal/constants plus2GiB trace/CCL. Real full-length activations are live in the probe, not double-counted as empty buffers.",
        "device_dram_bytes": 8 * 4267336064,
        "headroom_bytes_per_device": 8 * 4267336064 - total,
        "shared_collective_l1_workspace": {
            "ownership": "One TT_CCL context per ordered CQ0 layer stack; no concurrent sharing",
            "shape": [1, 1, 32, 20480],
            "shard": [32, 20480 // policy.get("allreduce_cores", 80)],
            "cores": policy.get("allreduce_cores", 80),
            "bytes_per_core": 1310720 // policy.get("allreduce_cores", 80),
            "bytes_per_device": 1310720,
            "count_per_stack": 1,
        },
        "note": "B1 advertised context; larger batch capacity must be checked independently. No capability reduction.",
    }
    source_hash = hashlib.sha256((DOC.parent / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    evidence = []
    for name in (
        "capacity_l0_s262143",
        "capacity_l0_s262144",
        "capacity_l3_s262143",
        "capacity_l3_s262144",
        "capacity_stack_s262143",
    ):
        path = DOC / stage / (name + ".json")
        if not path.exists():
            break
        run = json.loads(path.read_text())
        if stage == "optimized_multichip_decoder":
            exit_marker = path.with_suffix(".exit_status")
            if (
                not exit_marker.exists()
                or exit_marker.read_text().strip() != "0"
                or run.get("baseline") is not False
                or run.get("mesh_shape") != [1, 4]
                or run.get("policy") != {}
                or run.get("cache_pcc_diagnostic_only", False)
            ):
                break
        if (
            run.get("source_sha256") != source_hash
            or min(run.get("pcc", {"missing": 0}).values()) < 0.995
            or run.get("capacity_reservation_bytes_per_device", 0) < result["capacity_probe_reserved_bytes_per_device"]
        ):
            break
        if "s262143" in name and not (run.get("trace_bitwise_equal") and run.get("post_decode_state_bitwise_equal")):
            break
        evidence.append(stage + "/" + path.name)
    if len(evidence) == 5:
        result.update(status="validated", capacity_evidence=evidence, source_sha256=source_hash)
    (DOC / stage / "memory_capacity_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    contract = json.loads((DOC / "context_contract.json").read_text())
    contract[stage] = {
        **{k: v for k, v in result.items() if k not in ("layers",)},
        "supported_context": 262144,
        "capability_reduction": None,
        "public_sequence_alignment_requirement": None,
        "largest_tested_prefill": 262144 if len(evidence) == 5 else None,
        "largest_tested_decode_context": 262144 if len(evidence) == 5 else None,
        "batch_coverage": {
            "largest_tested_batch": 32,
            "prefill_length_at_batch_32": 257,
            "decode_context_at_batch_32": 258,
            "full_context_batch": 1,
        },
        "kv_layout": "BFP8 TILE [physical_pages,1,32,256] per rank, identical replicated page tables",
        "evidence": evidence
        + [
            "multichip_decoder/mesh_plan.md",
            stage + "/memory_capacity_plan.json",
            "multichip_decoder/topology_initial.log",
        ],
    }
    (DOC / "context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
