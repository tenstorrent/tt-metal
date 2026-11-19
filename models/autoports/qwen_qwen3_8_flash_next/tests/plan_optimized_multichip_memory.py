# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only stage5 weight accounting from the production arithmetic functions.

AST extraction deliberately excludes TTNN and model imports; this tool never
opens a device. Selection and hardware validation remain parent-owned gates.
"""

import ast
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "tt/dram_projection.py"
    tree = ast.parse(source.read_text())
    names = {"DramProjectionConfig", "phase_configs", "weight_memory_plan"}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(nodes) == 3
    namespace = {"math": math, "dataclass": dataclass, "__name__": __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    plan = namespace["weight_memory_plan"]
    cfg = json.loads((root / "doc/functional_decoder/target_text_config.json").read_text())
    layers = Counter(cfg["layer_types"])
    total, gdn, qsa = cfg["num_hidden_layers"], layers["linear_attention"], layers["full_attention"]
    assert (total, gdn, qsa) == (48, 36, 12)
    baseline = json.loads((root / "doc/multichip_decoder/memory_capacity_plan.json").read_text())
    roles = {
        "gr": dict(k=10240, widths=[1], dtype="bfloat16", native_packed=False, copies=2 * total, packed=False),
        "gdn-compact": dict(k=2560, widths=[1560], dtype="bfloat16", native_packed=True, copies=gdn, packed=True),
        "gdn-three-projection": dict(
            k=2560, widths=[1536, 12, 12], dtype="bfloat16", native_packed=True, copies=gdn, packed=True
        ),
        "shared": dict(k=2560, widths=[160, 160], dtype="bfloat8_b", native_packed=False, copies=total, packed=False),
        "qsa": dict(k=2560, widths=[3072, 256, 256], dtype="bfloat8_b", native_packed=True, copies=qsa, packed=True),
        "wo": dict(k=1536, widths=[2560], dtype="bfloat8_b", native_packed=False, copies=qsa, packed=False),
    }
    entries = {}
    for name, role in roles.items():
        policy = dict(dtype=role["dtype"], packed=role["packed"])
        configs = {
            "both_phases_same_storage": dict(decode=policy, rows=policy),
            "decode_only_native_rows_retained": dict(decode=policy, rows=None),
        }
        if name == "shared":
            configs["decode_split_rows_packed"] = dict(decode=policy, rows=dict(policy, packed=True))
            configs["both_phases_packed"] = dict(decode=dict(policy, packed=True), rows=dict(policy, packed=True))
        variants = {}
        for variant, config in configs.items():
            row = plan(
                role["k"], role["widths"], config, native_dtype=role["dtype"], native_packed=role["native_packed"]
            )
            row["phase_config_storage_fields"] = config
            row["all_layer_native_bytes_per_rank"] = row["native_bytes"] * role["copies"]
            row["all_layer_steady_bytes_per_rank"] = row["steady_weight_bytes"] * role["copies"]
            row["all_layer_delta_bytes_per_rank"] = row["steady_delta_vs_native_bytes"] * role["copies"]
            variants[variant] = row
        entries[name] = dict(
            input_width=role["k"],
            logical_output_widths=role["widths"],
            copies_per_rank=role["copies"],
            variants=variants,
        )
    primary_roles = ["gr", "gdn-compact", "shared", "qsa", "wo"]
    scenarios = {}
    for name, shared_variant in [
        ("all_roles_single_storage", "both_phases_same_storage"),
        ("all_roles_shared_split_decode_packed_rows", "decode_split_rows_packed"),
    ]:
        chosen = {role: shared_variant if role == "shared" else "both_phases_same_storage" for role in primary_roles}
        delta = sum(
            entries[role]["variants"][variant]["all_layer_delta_bytes_per_rank"] for role, variant in chosen.items()
        )
        target = baseline["summary"]["existing_resident_reservation_target_bytes"] + delta
        physical = baseline["target"]["observed_allocatable_bytes_per_device"]
        scenarios[name] = dict(
            selection_pending=True,
            role_variants=chosen,
            all_layer_weight_delta_bytes_per_rank=delta,
            all_layer_weight_delta_bytes_four_ranks=delta * 4,
            conservative_resident_reservation_target_bytes_per_rank=target,
            charged_against_static_allowance_bytes=0,
            headroom_before_runtime_bytes=physical - target,
            headroom_after_1GiB_runtime_bytes=physical - target - 2**30,
            headroom_after_2GiB_runtime_bytes=physical - target - 2**31,
            headroom_after_1GiB_runtime_and_32769_resident_io_bytes=physical - target - 2**30 - 335872000,
            headroom_after_2GiB_runtime_and_32769_resident_io_bytes=physical - target - 2**31 - 335872000,
            full_32769_io_plus_separate_2GiB_envelope_fits=physical - target - 2**31 - 335872000 >= 0,
            maximum_two_slab_rows_after_separate_2GiB_envelope=((physical - target - 2**31) // (32 * 2560 * 2 * 2))
            * 32,
        )
    report = dict(
        schema=1,
        status="pending policy selection and final-default reservation/context evidence",
        cpu_only=True,
        ttnn_imported=False,
        context_contract_mutated=False,
        model_layers=dict(total=total, gdn=gdn, qsa=qsa, gr_injections=total * 2),
        target=baseline["target"],
        baseline_resident_reservation_bytes_per_rank=baseline["summary"]["existing_resident_reservation_target_bytes"],
        inherited_runtime_reserve=dict(
            default_bytes=1073741824,
            stress_envelope_bytes=2147483648,
            envelope_reduction_proposed=False,
            full_stack_trace_measured=False,
            baseline_after_2GiB_headroom_bytes=827803840,
            source="doc/context_contract.json::multichip_decoder.runtime_reserve and sharing_recommendations",
            interpretation="Both existing unmeasured envelopes remain. Use the separate2GiB margin for conservative carry-forward;1GiB is an additional inherited diagnostic, not a replacement.",
        ),
        long_harness_reservation=dict(
            source="tests/probe_multichip_decoder.py::--reserve-bytes",
            allocation_order="construct representative decoder and request state; filler brings allocated DRAM to target; only then allocate input/output and execute model",
            target_scope="Full-stack resident weight/static/state budget before runtime and IO; not target plus a physically allocated1GiB or2GiB runtime block",
            passed_run_proves="Representative decoder executes within the remaining DRAM alongside filler; no measured full-stack2GiB runtime guarantee",
            inherited_full_stack_contract="Retain2GiB planning envelope with bounded stack IO. A full32769 resident two-slab smoke cannot independently demonstrate2GiB extra headroom under either all-role candidate.",
        ),
        exact_scope="Steady DRAM weight payload including equal eight-bank shard padding; excludes measured allocator high-water and full-stack construction claims.",
        source_contracts={
            "weight_math": "tt/dram_projection.py::weight_memory_plan (AST-executed unchanged)",
            "phase_weight_deduplication": "_PreparedGroup weight cache key is (dtype, packed); geometry/fidelity variants share weights within one PhaseDramProjectionGroup",
            "native_release": "Both phases selected clears native callback/old wrapper, and shared policy empties self.shared; one selected phase retains native weights",
            "gdn_compaction": "apply_dram_candidate concatenates Z1536/A12/B12 into one1560-wide logical projection before padding; old grouped form pads each branch before concatenating",
            "baseline": "doc/multichip_decoder/memory_capacity_plan.json",
        },
        roles=entries,
        scenarios=scenarios,
        unchanged=dict(
            kv_dtype="bfloat16",
            kv_layout="[capacity/32,1,32,256] local tiled cache per rank",
            supported_context=262144,
            kv_bytes_per_rank_all12_layers=3221225472,
            state_eight_bank_reservation_at_max=3502316032,
            expert_storage="bfloat4_b accepted prior stage",
            shared_down_and_scalar_gate=True,
        ),
        pending_context_contract_section=dict(
            status="pending selection and final evidence; do not insert as accepted",
            supported_context=262144,
            capability_reduction=False,
            kv_cache_dtype_layout_changed=False,
            memory_plan="optimized_multichip_decoder/final_context_memory_plan.json",
            selected_scenario=None,
            persistent_weight_delta_bytes_per_rank=None,
            resident_reservation_target_bytes_per_rank=None,
            runtime_reserve_default_bytes=1073741824,
            runtime_reserve_stress_envelope_bytes=2147483648,
            runtime_reserve_reduction=False,
            bounded_io_required_to_preserve_2GiB_envelope=True,
            context_smoke_artifacts=[],
            context_validation_policy="32K–64K final-default smoke with updated all-layer reservation; no max-context sweep unless KV dtype/layout is selected",
        ),
        limitations=[
            "BFP8 KV remains a separate non-default experiment and contributes no saving to these scenarios.",
            "No capacity reduction is proposed. Full resident max-context input/output still does not fit; bounded stack IO is retained.",
            "No full model is constructed; totals multiply decoder role counts for the existing future-stack arithmetic/reservation gate.",
            "Weight construction overlap is per projection group, not a bound on all temporary upload buffers or whole-model peak.",
            "Parallel indexer adds invocation-local FP32 partials of409600 bytes at Q1 or245760 bytes at Q2..32, plus81920-byte FP32 result; these belong to runtime reserve, not persistent weights.",
        ],
    )
    output = root / "doc/optimized_multichip_decoder/final_context_memory_plan.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
