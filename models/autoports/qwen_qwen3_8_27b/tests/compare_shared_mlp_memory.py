# SPDX-License-Identifier: Apache-2.0
"""Compare independently closed native fixture arms; no TTNN imports."""
import argparse
import json
from pathlib import Path


def compare(control, candidate):
    assert control["status"] == candidate["status"] == "pass"
    assert (control["shared"], candidate["shared"]) == ("0", "1")
    for name in ("batch", "layers", "native_context", "physical_pool_tokens_per_row", "runtime", "source_sha256"):
        assert control[name] == candidate[name], ("Unmatched arm identity", name)
    assert len(control["records"]) == len(candidate["records"]) > 0
    for left, right in zip(control["records"], candidate["records"]):
        for field in ("length", "logical_lengths", "tokens", "logits", "states"):
            assert left[field] == right[field], ("Native exactness failure", left["length"], field)
    layers = 64 if control["layers"] == "all64" else len(control["layers"])
    expected = layers * (5120 // 32) * (8704 // 32) * 576
    deltas = {}
    for phase in ("after_construction", "after_preparation"):
        left, right = control[phase], candidate[phase]
        assert left["exact_allocator_reconciliation"] and right["exact_allocator_reconciliation"]
        delta = left["allocator_bytes_per_device"] - right["allocator_bytes_per_device"]
        removed_owner_bytes = sum(
            row["physical_bytes_per_device"]
            for row in left["allocations"]
            if any(alias["path"].endswith(".gateup") for alias in row["aliases"])
        )
        remaining_owner_bytes = sum(
            row["physical_bytes_per_device"]
            for row in right["allocations"]
            if any(alias["path"].endswith(".gateup") for alias in row["aliases"])
        )
        assert removed_owner_bytes == expected and remaining_owner_bytes == 0, (
            "Packed physical weight owner census mismatch",
            phase,
            removed_owner_bytes,
            expected,
            remaining_owner_bytes,
        )
        assert delta > 0, ("No net resident allocation reduction", phase, delta)
        if phase == "after_construction":
            assert delta == expected, ("Unexpected construction allocation delta", delta, expected)
        deltas[phase] = dict(
            net_reduction=delta,
            removed_packed_weight_bytes=removed_owner_bytes,
            other_allocation_delta=delta - removed_owner_bytes,
        )
    return dict(
        status="pass",
        batch=control["batch"],
        layers=control["layers"],
        all_devices_tokens_logits_and_complete_logical_states_exact=True,
        expected_removed_weight_payload_bytes_per_device=expected,
        measured_allocator_reduction_bytes_per_device=deltas,
        limitations=[
            "Diagnostic tracked timing is not clean headline performance.",
            "Bounded external request pool does not establish serving B32 retention/capacity.",
            "Raw allocator deltas include any changed persistent runtime/program allocations.",
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("control", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = compare(json.loads(args.control.read_text()), json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
