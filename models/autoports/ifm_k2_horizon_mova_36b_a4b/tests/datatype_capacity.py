# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hard lower bounds reject policies that cannot preserve the advertised context."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    plan = json.loads((ROOT / "doc/optimized_multichip_decoder/memory_capacity_plan.json").read_text())
    physical = contract["full_model"]["physical_allocator_bytes"]
    # Required tensors alone, omitting all scratch/traces/constants/reservations.
    weights = sum(l["count"] * l["weight_bytes"] for l in plan["layers"].values())
    embedding = contract["full_model"]["per_device_bytes"]["replicated_embedding_bf16"]
    bf16_kv = 48 * 2 * 524288 * 2 * 128 * 2
    rows = [
        dict(
            config_id="kv_bf16",
            required_weights=weights,
            required_kv=bf16_kv,
            required_embedding=embedding,
            hard_lower_bound_bytes=weights + bf16_kv + embedding,
        )
    ]
    w = plan["layers"]["3"]["weight_rows"]
    expert_names = ("value_experts", "experts.gate_up", "experts.down_proj", "experts.ep_gate_up", "experts.ep_down")
    expert_bytes = sum(w[k]["prefill_bytes"] for k in expert_names)
    # Scaling BFP4 tile576 -> BFP8 tile1088. This deliberately omits the EP value bank.
    delta = 45 * expert_bytes * (1088 - 576) // 576
    bfp8_kv = contract["full_model"]["per_device_bytes"]["stack_kv_bfp8"]
    for fidelity in ("LoFi", "HiFi2"):
        rows.append(
            dict(
                config_id=f"experts_bfp8_{fidelity}",
                required_weights=weights + delta,
                required_kv=bfp8_kv,
                required_embedding=embedding,
                hard_lower_bound_bytes=weights + delta + bfp8_kv + embedding,
            )
        )
    # The all-BFP8 canonical policy also reduces its head from BF16 to BFP8.
    # Count that saving explicitly; required projection growth still exceeds it.
    # Only existing prefill banks are counted, omitting all extra decode banks.
    projection_bytes = sum(
        layer["count"] * row["prefill_bytes"]
        for layer in plan["layers"].values()
        for name, row in layer["weight_rows"].items()
        if row["dtype"] == "DataType.BFLOAT4_B" and name.startswith(("attention", "self_attn.o_proj", "mlp."))
    )
    projection_delta = projection_bytes * (1088 - 576) // 576
    for fidelity in ("LoFi", "HiFi2"):
        rows.append(
            dict(
                config_id=f"canonical_bfp8_kv8_{fidelity}",
                required_weights=weights + delta + projection_delta,
                required_kv=bfp8_kv,
                required_embedding=embedding,
                required_head_banks=2 * 335544320 * 1088 // 2048,
                required_projection_delta_bytes=projection_delta,
                hard_lower_bound_bytes=weights + delta + projection_delta + bfp8_kv + embedding,
            )
        )
    for row in rows:
        row.setdefault("required_head_banks", 2 * 335544320)
        row["hard_lower_bound_bytes"] += row["required_head_banks"]
        row.update(
            context=524288,
            physical_allocator_bytes=physical,
            status="capacity_fail",
            capability_reduction=None,
            source="doc/optimized_multichip_decoder/memory_capacity_plan.json",
            reason="Required persistent tensors alone exceed physical DRAM; no context reduction attempted",
            full_model_evaluated=False,
        )
        assert row["hard_lower_bound_bytes"] > physical, row
    (DOC / "capacity_rejections.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
