# SPDX-License-Identifier: Apache-2.0
"""Select the next bounded native capacity trial from closed real receipts.

This tool never opens a device or invents an observed maximum. A selected native
pool still requires the real0.30 serving prefix/lifecycle qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path

PER_ID = 17285120
GUARD = 512 * 1024**2
FRAGMENT_GUARD = 8 * 1024**2
LENGTHS = [513, 4096, 16384]


def pool_bytes(ids):
    return ids * PER_ID + ((-3 * ids) % 8) * 5120 * 16


def validate_pass(row):
    assert row["status"] == "pass" and row["close_mesh_device_returned"] is True
    assert row["layers"] == "all64" and row["batch"] == 32 and row["native_context"] == 262144
    assert row["requested_outputs"] == 128 and row["configured_trace_region_bytes"] == 128 * 1024**2
    assert row["throw_exception_on_fallback"] is True
    assert row["shared_mlp_readers"] == 1 and row["shared_mlp_cores"] == 80
    assert row["capacity_guards"]["min_normal_free_bytes"] == GUARD
    assert row["capacity_guards"]["fragment_guard_bytes_per_bank"] == FRAGMENT_GUARD
    assert row["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert row["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    assert all(
        row["environment"][flag] == "1"
        for flag in [
            "QWEN38_GROUPED_RECURRENCE",
            "QWEN38_MASKED_RECURRENCE",
            "QWEN38_FLAT_PREFILL",
        ]
    )
    ids = row["capacity_pool_ids"]
    assert row["physical_cache_pool"]["actual_bytes_per_device"] == pool_bytes(ids)
    assert row["physical_cache_pool"]["global_ids"] == ids
    assert row["physical_cache_pool"]["checkpoint_tokens"] == 512
    assert row["physical_cache_pool"]["unique_snapshot_pools"] == 16
    assert [r["length"] for r in row["records"]] == LENGTHS * row["stress_cycles"]
    assert all(len(r["tokens"]) == 32 and all(len(tokens) == 128 for tokens in r["tokens"]) for r in row["records"])
    assert all(len(r["logits"]) == 4 and r["allocation"]["exact_allocator_reconciliation"] for r in row["records"])
    for phase in ["after_construction", "after_preparation"]:
        assert row[phase]["exact_allocator_reconciliation"]
        assert row[phase]["independently_reconciled_allocator_domains"] == [
            "DRAM",
            "TRACE",
        ]
    obs = row["capacity_observation"]
    assert obs["minimum_free_from_allocation_events"] >= GUARD
    assert obs["minimum_trace_free_from_allocation_events"] >= 8 * 1024**2
    assert obs["largest_contiguous_free_per_bank"] >= obs["maximum_request_allocation_per_bank"] + FRAGMENT_GUARD


def receipt_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def campaign_identity(row):
    keys = (
        "source_sha256",
        "runtime",
        "capacity_image_digest",
        "vllm_distribution_version",
        "checkpoint_file_sha256",
        "effective_precision",
        "capacity_study_id",
        "capacity_guards",
        "environment",
        "shared_mlp_readers",
        "shared_mlp_cores",
        "layers",
        "batch",
        "native_context",
        "requested_outputs",
        "configured_trace_region_bytes",
        "throw_exception_on_fallback",
        "layer_types",
    )
    return {key: row.get(key) for key in keys}


def validate_boundary_contract(row):
    expected = dict(
        layers="all64",
        batch=32,
        native_context=262144,
        requested_outputs=128,
        configured_trace_region_bytes=128 * 1024**2,
        shared_mlp_readers=1,
        shared_mlp_cores=80,
        throw_exception_on_fallback=True,
    )
    if any(row.get(k) != value for k, value in expected.items()):
        raise ValueError("Unsafe boundary changed the registered workload/geometry")
    if row.get("capacity_guards") != dict(
        min_normal_free_bytes=512 * 1024**2,
        fragment_guard_bytes_per_bank=8 * 1024**2,
        min_trace_free_bytes=8 * 1024**2,
    ):
        raise ValueError("Receipt changed registered capacity guards")
    if row.get("stress_cycles") not in (1, 3):
        raise ValueError("Unexpected stress cycle contract")
    pool = row.get("physical_cache_pool", {})
    n = row["capacity_pool_ids"]
    actual = n * 17285120 + ((-3 * n) % 8) * 5120 * 16
    if (
        pool.get("actual_bytes_per_device") != actual
        or pool.get("global_ids") != n
        or pool.get("checkpoint_tokens") != 512
        or pool.get("unique_snapshot_pools") != 16
    ):
        raise ValueError("Unsafe boundary lacks exact physical pool ownership")
    for phase in ("after_construction", "after_preparation"):
        allocation = row.get(phase, {})
        if not allocation.get("exact_allocator_reconciliation") or allocation.get(
            "independently_reconciled_allocator_domains"
        ) != ["DRAM", "TRACE"]:
            raise ValueError("Unsafe boundary lacks reconciled allocator domains")
    records = row.get("records", [])
    expected_lengths = [513, 4096, 16384] * row["stress_cycles"]
    if [r["length"] for r in records] != expected_lengths[: len(records)] or len(records) > len(expected_lengths):
        raise ValueError("Unsafe boundary changed completed workload prefix")
    if row.get("status") == "pass" and len(records) != len(expected_lengths):
        raise ValueError("Passing receipt must complete every registered stress case")
    if row.get("status") != "pass" and not records:
        raise ValueError("Unsafe boundary must include a completed stress-case witness")
    for r in records:
        if (
            len(r.get("tokens", [])) != 32
            or any(len(t) != 128 for t in r["tokens"])
            or len(r.get("logits", [])) != 4
            or not r.get("allocation", {}).get("exact_allocator_reconciliation")
        ):
            raise ValueError("Unsafe boundary has incomplete completed-request evidence")


def known_guard_failure(row):
    return (
        row.get("close_mesh_device_returned") is True
        and not row.get("cleanup_error")
        and not row.get("close_error")
        and row.get("error_type") == "AssertionError"
    ) and any(
        text in row.get("error", "")
        for text in (
            "Transient headroom failed",
            "TRACE headroom failed",
            "Fragmentation guard failed",
        )
    )


def next_trials(receipts, max_attempts=8):
    assert all(r["shared"] in ("0", "1") for r in receipts)
    identities = [campaign_identity(r) for r in receipts]
    assert not identities or all(value == identities[0] for value in identities), "Unmatched campaign identity"
    # Inspect every receipt before reducing repeated trials at an ID. A later
    # successful retry cannot erase an unexplained crash or unknown failure.
    unknown = [
        dict(
            ids=r["capacity_pool_ids"],
            shared=r["shared"],
            error=r.get("error", ""),
            receipt_sha256=receipt_digest(r),
        )
        for r in receipts
        if r["status"] != "pass" and not known_guard_failure(r)
    ]
    if unknown:
        return {
            "blocked": dict(
                status="needs_owner_review",
                issues=unknown,
                no_device_trials_may_continue=True,
            )
        }
    witness = {}
    for row in receipts:
        if row["status"] == "pass":
            validate_pass(row)
        # Completed requests remain correctness witnesses even when a later
        # headroom check fails in the same receipt.
        for record in row.get("records", []):
            if not all(key in record for key in ("length", "input_sha256", "tokens", "logits")):
                continue
            key = record["length"]
            value = (record["input_sha256"], record["tokens"], record["logits"])
            assert (
                key not in witness or witness[key] == value
            ), "Native fixed-work output/logit mismatch; stop capacity search"
            witness[key] = value
    result = {}
    for shared in ["0", "1"]:
        arm = [r for r in receipts if r["shared"] == shared]
        effective = {}
        for row in sorted(arm, key=lambda r: r["stress_cycles"]):
            effective[row["capacity_pool_ids"]] = row
        failures = []
        for n, row in effective.items():
            if row["status"] == "pass":
                continue
            error = row.get("error", "")
            if not row.get("close_mesh_device_returned") or not any(
                text in error
                for text in [
                    "Transient headroom failed",
                    "TRACE headroom failed",
                    "Fragmentation guard failed",
                ]
            ):
                result[shared] = dict(status="needs_owner_review", ids=n, error=error)
                break
            failures.append(n)
        if shared in result:
            continue
        passing = [n for n, r in effective.items() if r["status"] == "pass"]
        if not arm:
            n, cycles = 1368, 1
        elif not passing:
            n, cycles = max(1249, min(failures) - 32), 1
            if min(failures) <= 1249:
                result[shared] = dict(status="fixed_B32_stress_infeasible_without_structural_change")
                continue
        else:
            low = max(passing)
            high = min((n for n in failures if n > low), default=None)
            if high == low + 1:
                if effective[low]["stress_cycles"] == 3:
                    validate_boundary_contract(effective[high])
                    result[shared] = dict(
                        status="native_max_safe_verified_at_one_ID_boundary",
                        ids=low,
                        actual_physical_cache_bytes_per_device=pool_bytes(low),
                        next_ID_failed_same_guards=high,
                        selected_receipt_sha256=receipt_digest(effective[low]),
                        unsafe_receipt_sha256=receipt_digest(effective[high]),
                        campaign_identity=campaign_identity(effective[low]),
                        requires_actual_serving_qualification=True,
                    )
                    continue
                n, cycles = low, 3
            elif high is not None:
                n, cycles = (low + high) // 2, 1
            else:
                # Conservative proposal only; all points require real stress.
                margin = effective[low]["capacity_observation"]["minimum_free_from_allocation_events"] - GUARD
                step = max(1, min(128, margin // (PER_ID + 4 * 139264 + 65536)))
                n, cycles = low + step, 1
        if len(arm) >= max_attempts:
            result[shared] = dict(
                status="bounded_search_incomplete",
                largest_tested_pass=max(passing, default=None),
            )
            continue
        result[shared] = dict(
            status="next_native_trial",
            ids=n,
            cycles=cycles,
            predicted_physical_cache_bytes_per_device=pool_bytes(n),
            serving_budget_candidate_bytes_per_device=pool_bytes(n),
            argv=[
                "--full-stack",
                "--batch",
                "32",
                "--capacity-pool-ids",
                str(n),
                "--lengths",
                "513",
                "4096",
                "16384",
                "--outputs",
                "128",
                "--cycles",
                str(cycles),
                "--repeats",
                "0",
            ],
        )
    if any(row["status"] == "needs_owner_review" for row in result.values()):
        return {
            "blocked": dict(
                status="needs_owner_review",
                issues=[row for row in result.values() if row["status"] == "needs_owner_review"],
                no_device_trials_may_continue=True,
            )
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.receipts]
    result = dict(
        status="prepared_or_observed_native_search_not_serving_acceptance",
        max_attempts_per_arm=8,
        normal_free_guard_bytes=GUARD,
        fragment_guard_bytes_per_bank=FRAGMENT_GUARD,
        checkpoint_tokens=512,
        arms=next_trials(rows),
        limits=[
            "Proposed IDs are not measured. Only closed real receipts can select a native maximum.",
            "B32/A32 same-private-transcript real0.30 serving stress and prefix comparison must follow.",
            "Unknown failures, timeouts, OOMs and numerical mismatches stop for owner review.",
            "Current1368 reference anchors both weight modes; max-control minus reference isolates unused headroom.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
