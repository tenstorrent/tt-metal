# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paired public raw-output scoring using the single canonical BF16 reducer.

Accuracy only, with explicit unit-gradient evidence for two-input operations.
No timing, package readiness, or independent mathematical oracle lives here.
"""
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path


def _unit_gradient_attested(directory: Path, sidecar: Path) -> bool:
    """Require both completed input records; a backward name is not evidence."""
    from ttpoly.spec.program import ProgramIOContract

    io_contract = ProgramIOContract.from_value(json.loads(sidecar.read_text())["io_contract"])
    if not io_contract.fuse_grad:
        return False
    invocation = json.loads((directory / "invocation.json").read_text())
    if tuple(invocation.get("tensor_roles", ())) != io_contract.tensor_input_roles:
        raise ValueError("public invocation tensor roles disagree with compiler I/O")
    expected = {
        "io_contract": io_contract.to_dict(),
        "auxiliary_inputs": [
            {
                "tensor_index": io_contract.gradient_input_index,
                "role": "incoming_gradient",
                "encoding": "constant_bfloat16",
                "raw_bits": "0x3f80",
                "value": 1.0,
                "count": 65536,
            }
        ],
    }
    for arm in ("candidate", "stock"):
        observed = json.loads((directory / f"{arm}.inputs.json").read_text())
        auxiliary = observed.get("auxiliary_inputs") if isinstance(observed, dict) else None
        unit = auxiliary[0] if isinstance(auxiliary, list) and len(auxiliary) == 1 else {}
        if (
            observed != expected
            or not isinstance(unit, dict)
            or type(unit.get("count")) is not int
            or type(unit.get("tensor_index")) is not int
            or isinstance(unit.get("value"), bool)
        ):
            raise ValueError(f"{arm}: missing exact full-space unit-gradient input attestation")
    return True


def score_operation(directory: Path, operation: str, *, reducer=None) -> dict:
    if reducer is None:
        from ttpoly.bf16_scoring import _run_bf16_exhaustive_raw

        reducer = _run_bf16_exhaustive_raw
    sidecar = directory / "semantic.json"
    if not sidecar.is_file():
        raise ValueError("missing compiler semantic.json")
    gradient_attested = _unit_gradient_attested(directory, sidecar)
    for arm in ("candidate", "stock"):
        if (directory / f"{arm}.bf16").stat().st_size != 65536 * 2:
            raise ValueError(f"{arm}: expected exactly 65536 raw BF16 outputs")
    result = {"operation": operation, "status": "accuracy_scored", "pr_ready": False}
    comparison_populations = []
    for arm, label, use_stock_reference in (
        ("candidate", "candidate", True),
        ("stock", "stock", True),
        ("stock", "stock_declared_semantics_diagnostic", False),
    ):
        record = io.StringIO()
        with redirect_stdout(record):
            coverage = reducer(
                operation,
                directory / f"{arm}.bf16",
                semantic_sidecar=sidecar,
                class_reference_bf16=directory / "stock.bf16" if use_stock_reference else None,
                attest_fused_gradient_one=gradient_attested,
            )
        lines = record.getvalue().strip().splitlines()
        fields = lines[0].split(",") if len(lines) == 1 else []
        if len(fields) != 11:
            raise ValueError(f"{arm}: canonical accuracy record must have exactly 11 fields")
        values = [float(field) for field in fields]
        if math.isnan(values[8]) or values[8] < 0:
            raise ValueError(f"{arm}: missing or invalid max pure ULP")
        if (
            coverage.get("complete") is not True
            or coverage.get("count") != 65536
            or coverage.get("start_bit") != 0
            or coverage.get("end_bit_exclusive") != 65536
            or coverage.get("coverage_kind") != "all_encoding_traversal_with_declared_class_checks"
        ):
            raise ValueError(f"{arm}: reducer did not report complete classified BF16 traversal")
        mismatches = coverage.get("total_class_policy_mismatch_count")
        invalid = coverage.get("metric_counts", {}).get("invalid_output_for_finite_reference")
        if type(mismatches) is not int or mismatches < 0 or type(invalid) is not int or invalid < 0:
            raise ValueError(f"{arm}: missing class or finite-reference failure counts")
        if use_stock_reference:
            population = {
                name: coverage[name] for name in ("contract_points", "population_counts", "population_input_classes")
            }
            comparison_populations.append(population)
        result[label] = {
            "count": coverage["count"],
            "max_pure_ulp": values[8] if math.isfinite(values[8]) else "inf",
            "class_mismatches": mismatches,
            "class_reference": coverage["class_reference_kind"],
            "conformance_status": coverage["device_conformance_status"],
            "invalid_finite_reference_outputs": invalid,
            "numeric_contract_count": coverage["contract_points"],
        }
    if comparison_populations[0] != comparison_populations[1]:
        raise ValueError("candidate and stock numeric reference populations disagree")
    candidate, stock = (float(result[arm]["max_pure_ulp"]) for arm in ("candidate", "stock"))
    result["ulp_comparison"] = "better" if candidate < stock else "equal" if candidate == stock else "worse"
    # This is deliberately separate from typed conformance: finite terminal
    # actions can correctly return a different class than inaccurate stock.
    import numpy as np
    from ttpoly.bf16_scoring import _bf16_policy_classes

    raw_classes = [
        _bf16_policy_classes(np.fromfile(directory / f"{arm}.bf16", dtype="<u2")) for arm in ("candidate", "stock")
    ]
    result["literal_output_class_differences"] = int(np.count_nonzero(raw_classes[0] != raw_classes[1]))
    if gradient_attested:
        result["gradient_scope"] = "all_65536_activation_encodings_with_incoming_gradient_fixed_to_one"
    return result
