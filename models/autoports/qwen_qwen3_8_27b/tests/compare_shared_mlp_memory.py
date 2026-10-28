# SPDX-License-Identifier: Apache-2.0
"""Compare independently closed native fixture arms; no TTNN imports."""
import argparse
import json
from pathlib import Path


def compare(control, candidate, expected_lengths=None, require_final_metadata=False):
    assert control["status"] == candidate["status"] == "pass"
    assert (control["shared"], candidate["shared"]) == ("0", "1")
    for name in ("batch", "layers", "native_context", "physical_pool_tokens_per_row", "runtime", "source_sha256"):
        assert control[name] == candidate[name], ("Unmatched arm identity", name)
    for name in ("effective_precision", "environment", "layer_types", "requested_outputs"):
        if name in control or name in candidate:
            assert control.get(name) == candidate.get(name), ("Unmatched effective metadata", name)
    if require_final_metadata:
        for arm in (control, candidate):
            assert arm.get("close_mesh_device_returned") is True
            assert arm.get("throw_exception_on_fallback") is True
            assert all(
                name in arm for name in ("effective_precision", "environment", "layer_types", "prepared_linear_calls")
            )
            assert arm["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
            assert arm["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
            assert arm["prepared_linear_calls"], "No actual prepared kernel argument evidence"
            roles = (".gate", ".up") if arm["shared"] == "1" else (".gateup",)
            shared_roles = [
                row
                for row in arm["prepared_linear_calls"]
                if row["owner"].endswith(roles) and row["input_shape"] == [1, 1, 32, 5120]
            ]
            assert shared_roles, "No actual decode MLP projection evidence"
            for row in shared_roles:
                expected_readers = arm.get("shared_mlp_readers", 1) if arm["shared"] == "1" else 2
                assert row["program"]["in0_block_w"] == 2
                assert row["program"]["num_workers_per_dram_bank"] == expected_readers
                assert row["weight_dtype"] == "DataType.BFLOAT4_B"
                assert row["math_fidelity"] == "MathFidelity.LoFi"
            assert arm["effective_precision"]["weight_groups"]["mlp"] == "bfloat4_b"
            assert arm["effective_precision"]["compute_fidelities"]["projection"] == "LoFi"
            assert all(
                arm["environment"][flag] == "1"
                for flag in ("QWEN38_GROUPED_RECURRENCE", "QWEN38_MASKED_RECURRENCE", "QWEN38_FLAT_PREFILL")
            )
            assert arm["trace_release"]["actual_released_dram_bytes_per_device"] > 0
    assert len(control["records"]) == len(candidate["records"]) > 0
    if expected_lengths is not None:
        assert [
            row["length"] for row in control["records"]
        ] == expected_lengths, "Missing/extra requested boundary cases"
    expected_layers = set(range(64)) if control["layers"] == "all64" else set(control["layers"])
    for arm in (control, candidate):
        for row in arm["records"]:
            assert len(row["tokens"]) == arm["batch"] and len(row["logits"]) == 4
            outputs = arm.get("requested_outputs", len(row["tokens"][0]))
            assert outputs > 0 and all(len(tokens) == outputs for tokens in row["tokens"])
            assert {int(name.split(".")[0]) for name in row["states"]} == expected_layers
            for layer in expected_layers:
                fields = {name.split(".")[1] for name in row["states"] if name.startswith(f"{layer}.")}
                assert fields in ({"conv", "recurrent"}, {"key", "value"}), ("Incomplete layer state", layer, fields)
            assert all(len(shards) == 4 for shards in row["states"].values())
            if require_final_metadata:
                assert row["allocation"]["exact_allocator_reconciliation"]
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

        def separate_bytes(allocations):
            return sum(
                row["physical_bytes_per_device"]
                for row in allocations
                if any(alias["path"].endswith((".gate", ".up")) for alias in row["aliases"])
            )

        extra_separate = separate_bytes(right["allocations"]) - separate_bytes(left["allocations"])
        expected_extra = (
            layers * 2 * (5120 // 32) * (256 // 32) * 576 if candidate.get("shared_mlp_readers", 1) == 2 else 0
        )
        assert extra_separate == expected_extra, ("Unexpected shared-owner padding", extra_separate, expected_extra)
        if phase == "after_construction":
            assert delta == expected - extra_separate, (
                "Unexpected construction allocation delta",
                delta,
                expected,
                extra_separate,
            )
        deltas[phase] = dict(
            net_reduction=delta,
            removed_packed_weight_bytes=removed_owner_bytes,
            added_separate_owner_padding_bytes=extra_separate,
            other_allocation_delta=delta - (removed_owner_bytes - extra_separate),
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
    p.add_argument("--expected-lengths", type=int, nargs="+")
    p.add_argument("--require-final-metadata", action="store_true")
    args = p.parse_args()
    result = compare(
        json.loads(args.control.read_text()),
        json.loads(args.candidate.read_text()),
        args.expected_lengths,
        args.require_final_metadata,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
