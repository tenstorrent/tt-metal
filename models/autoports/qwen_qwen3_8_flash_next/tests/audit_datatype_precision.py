# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only audit of recorded constructed precision; never imports TTNN.

Accepts candidate, final-validation and resident token-out reports. A successful
audit establishes agreement of recorded operands/configs with the resolved
policy, not model accuracy, performance, or direct introspection of every fixed
numerical contract. Existing output evidence is never overwritten.
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DRIVERS = (
    "run_optimized_full_model_validation",
    "benchmark_full_model_resident",
    "run_full_model_acceptance",
)
EVIDENCE_SCOPE = {
    "tensor_introspection": [
        "Projection decode/rows weight_dtypes and shared_down.dtype",
        "Expert gate/up/down sets of actual resident tensor dtypes",
    ],
    "constructed_config_introspection": [
        "Projection, expert and shared-down compute fidelity/accumulation flags",
        "Projection input activation_dtype (execution config, not tensor readback)",
        "Top-level/final-mixer/layer CCL dtype selectors",
        "Layer KV dtype selector (not direct cache tensor introspection)",
    ],
    "declared_or_source_only": [
        "Projection output_dtype is a literal BF16 summary declaration",
        "Expert workload fields are literals: 10 active, 128 local, routed fused execution",
        "Residual/logits/sampling dtypes are validated fixed contracts; no separate actual tensor dtype in this summary",
        "fixed_numerics, max_context and kv_layout are validated fixed assumptions, not runtime tensor introspection",
        "Policy validation and fixed_contract_validation do not themselves prove execution",
    ],
    "source_references_relative_to_model": [
        "tt/precision.py:load_policy and layer_policy",
        "tt/model.py:precision_summary, construction and logits checks",
        "tt/dram_projection.py:_PreparedGroup construction and execution",
        "tt/multichip_decoder.py:projection/expert construction and KV allocation",
        "tt/generator.py:build_generator and cache/sample/logit allocation",
    ],
    "limits": [
        "Audits saved telemetry; does not rerun devices or prove telemetry source identity at measurement time",
        "Selected comparison checks recorded requested policy on recognized default paths; retain measurement source provenance too",
        "Accuracy, capacity, trace performance, P1 and final selection gates remain separate",
    ],
}


def load_precision_module():
    # Import the standalone stdlib-only policy module without importing tt packages.
    spec = importlib.util.spec_from_file_location("datatype_precision_cpu", MODEL_DIR / "tt/precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_summaries(value, path="$"):
    """Keep the JSON path when a report nests precision under performance."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "runtime_precision":
                yield child_path, child
            else:
                yield from find_summaries(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from find_summaries(child, f"{path}[{index}]")


def audit_summary(summary, precision, expected_policy=None):
    """Return compact counts and exact mismatches; safe for in-memory controls."""
    counts = Counter()
    failures = []

    def check(path, actual, expected, kind):
        counts[kind] += 1
        if actual != expected:
            failures.append({"path": path, "actual": actual, "expected": expected, "evidence_kind": kind})

    def compute(path, value, fidelity, fp32, packer):
        expected = {
            "math_fidelity": f"MathFidelity.{fidelity}",
            "fp32_dest_acc_en": fp32,
            "packer_l1_acc": packer,
            "math_approx_mode": False,
        }
        for key, target in expected.items():
            check(f"{path}.{key}", value.get(key), target, "constructed_config_introspection")

    def weights(path, actual, dtype):
        check(f"{path}.nonempty_list", isinstance(actual, list) and bool(actual), True, "structure")
        if isinstance(actual, list):
            check(path, actual, [f"DataType.{dtype.upper()}"] * len(actual), "tensor_introspection")

    def projection(path, value, policy, role, packer):
        for phase in ("decode", "rows"):
            item = value.get(phase, {})
            prefix = f"{path}.{phase}"
            weights(f"{prefix}.weight_dtypes", item.get("weight_dtypes"), policy["weight_groups"][role])
            compute(f"{prefix}.compute", item.get("compute", {}), policy["compute_fidelities"][role], True, packer)
            check(
                f"{prefix}.activation_dtype",
                item.get("activation_dtype"),
                policy["activation_dtype"],
                "constructed_config_introspection",
            )
            check(f"{prefix}.output_dtype", item.get("output_dtype"), "bfloat16", "declared_or_source_only")

    try:
        policy = precision.load_policy(MODEL_DIR, summary["requested"])
        check("requested.validated", True, True, "policy_validation")
        if expected_policy is not None:
            check("requested.matches_expected_policy", policy, expected_policy, "policy_identity")
        layers = summary["layers"]
        check("layers.ordered_ids", [row.get("layer") for row in layers], list(range(48)), "structure")
        for row in layers:
            index = row["layer"]
            resolved = precision.layer_policy(policy, index)
            prefix = f"layers[{index}]"
            groups = row.get("groups", {})
            expected_groups = {"gr": 2, "shared": 1}
            expected_groups.update({"qsa_qkv": 1, "qsa_out": 1} if index % 4 == 3 else {"gdn_zab": 1})
            check(f"{prefix}.groups.roles", sorted(groups), sorted(expected_groups), "structure")
            for role, count in expected_groups.items():
                entries = groups.get(role, [])
                check(f"{prefix}.groups.{role}.count", len(entries), count, "structure")
                for entry_index, group in enumerate(entries):
                    projection(f"{prefix}.groups.{role}[{entry_index}]", group, resolved, role, True)
            experts = row.get("experts", {})
            check(
                f"{prefix}.experts.weights.roles",
                sorted(experts.get("weights", {})),
                ["down", "gate", "up"],
                "structure",
            )
            for role in ("gate", "up", "down"):
                weights(
                    f"{prefix}.experts.weights.{role}",
                    experts.get("weights", {}).get(role),
                    resolved["weight_groups"][f"expert_{role}"],
                )
            compute(
                f"{prefix}.experts.compute",
                experts.get("compute", {}),
                resolved["compute_fidelities"]["experts"],
                False,
                False,
            )
            for key, target in (
                ("active_experts_per_token", 10),
                ("local_experts", 128),
                ("execution", "routed moe_fused_swiglu"),
            ):
                check(f"{prefix}.experts.{key}", experts.get(key), target, "declared_or_source_only")
            shared = row.get("shared_down", {})
            check(
                f"{prefix}.shared_down.dtype",
                shared.get("dtype"),
                f"DataType.{resolved['weight_groups']['shared_down'].upper()}",
                "tensor_introspection",
            )
            compute(
                f"{prefix}.shared_down.compute",
                shared.get("compute", {}),
                resolved["compute_fidelities"]["shared_down"],
                True,
                False,
            )
            for field, policy_field in (("ccl_dtype", "ccl_dtype"), ("kv_dtype", "kv_cache_dtype")):
                check(
                    f"{prefix}.{field}",
                    row.get(field),
                    f"DataType.{resolved[policy_field].upper()}",
                    "constructed_config_introspection",
                )
        projection("lm_head", summary.get("lm_head", {}), policy, "lm_head", False)
        for field in ("top_level_ccl_dtype", "final_mixer_ccl_dtype"):
            check(
                field, summary.get(field), f"DataType.{policy['ccl_dtype'].upper()}", "constructed_config_introspection"
            )
        check(
            "fixed_contract_validation",
            summary.get("fixed_contract_validation"),
            "load_policy rejects unsupported residual/logits/sampling/fixed-numerics changes",
            "declared_or_source_only",
        )
        config_id = policy["config_id"]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        failures.append({"path": "summary_schema", "error": f"{type(exc).__name__}: {exc}"})
        config_id = None
    return {
        "config_id": config_id,
        "passed": not failures,
        "checked_equalities": sum(counts.values()),
        "checks_by_evidence_kind": dict(counts),
        "mismatches": failures,
    }


def audit_document(document, precision, selected=None):
    build = document.get("build_kwargs", {})
    override = build.get("precision_config")
    command = document.get("command", [])
    if not isinstance(command, list):
        raise ValueError("Recorded command must be an argv list")
    default_driver = any(Path(str(arg)).name.removesuffix(".py").split(".")[-1] in DEFAULT_DRIVERS for arg in command)
    if override is not None:
        expected = precision.load_policy(MODEL_DIR, override)
        construction = "explicit_policy_artifact"
    elif default_driver or "build_kwargs" in document:
        expected = selected
        construction = "default_policy_path"
    else:
        expected = None
        construction = "unproven_construction_path"
    records = []
    for path, summary in find_summaries(document):
        record = audit_summary(summary, precision, expected)
        record.update(
            summary_path=path,
            construction=construction,
            expected_policy_source=str(override) if override is not None else ("--selected" if selected else None),
            selected_default_checked=construction == "default_policy_path" and selected is not None,
        )
        if selected is not None and construction == "unproven_construction_path":
            record["passed"] = False
            record["mismatches"].append(
                {"path": "construction", "error": "Cannot establish default versus override path"}
            )
        records.append(record)
    if not records:
        raise ValueError("No runtime_precision summary; requested JSON alone is insufficient evidence")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected", type=Path, help="Require recognized default runs to match this selected policy")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is immutable")
    precision = load_precision_module()
    selected = precision.load_policy(MODEL_DIR, args.selected) if args.selected else None
    records = []
    for path in args.inputs:
        try:
            for row in audit_document(json.loads(path.read_text()), precision, selected):
                records.append({"input": str(path), **row})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            records.append({"input": str(path), "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cpu_only": True,
        "scope": "Recorded precision propagation; no stage acceptance or performance claim",
        "selected_artifact": str(args.selected) if args.selected else None,
        "passed": all(row["passed"] for row in records),
        "checked_equalities": sum(row.get("checked_equalities", 0) for row in records),
        "evidence_scope": EVIDENCE_SCOPE,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another task creating this path.
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "records": len(records), "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
