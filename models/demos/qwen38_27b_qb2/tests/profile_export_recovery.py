# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strict receipt and launch checks for CPU-only full-profile export recovery."""

from models.demos.qwen38_27b_qb2.tests.full_trace_profile import SCOPE


def terminal(properties, invocation):
    if properties.get("InvocationID") not in ("", invocation):
        raise ValueError("Observed a different invocation")
    # A missing unit is not sufficient proof that its process tree is gone.
    return (
        properties.get("LoadState") == "loaded"
        and properties.get("InvocationID") == invocation
        and properties.get("MainPID") == "0"
        and properties.get("ActiveState") in ("inactive", "failed")
    )


def validate_pair(profile, baseline):
    for row in (profile, baseline):
        if (
            row.get("state") != "completed"
            or row.get("passed") is not True
            or row.get("cleanup_completed") is not True
            or row.get("scope") != SCOPE
            or row.get("layer_indices") != list(range(64))
            or row.get("prefill_calls") != 0
            or len(row.get("replays", [])) != 3
            or len(row.get("device_ids", [])) != 4
            or len(set(row["device_ids"])) != 4
        ):
            raise ValueError("Recovery requires both complete clean hardware tests")
        for key in ("output_hashes", "token_hashes"):
            if len(row.get(key, [])) != 5 or len(set(row[key])) != 1:
                raise ValueError("Hardware outputs are incomplete or not repeatable")
    for key in (
        "input_tokens",
        "batch",
        "device_ids",
        "precision",
        "operand_hashes",
        "output_hashes",
        "token_hashes",
        "source_sha256",
    ):
        if key not in profile or profile[key] != baseline.get(key):
            raise ValueError("Profile/unprofiled mismatch: " + key)
    if not profile["operand_hashes"] or not profile["source_sha256"]:
        raise ValueError("Missing input/source provenance")
    # Earlier captures recorded this only inside the precision policy. Accept
    # that schema only when the effective policy still identifies both arms.
    recurrence = profile["precision"].get("decode_recurrence")
    if not recurrence or any(row.get("expected_recurrence", recurrence) != recurrence for row in (profile, baseline)):
        raise ValueError("Profile/unprofiled recurrence mismatch")


def require_unstarted_failure(properties, receipt, invocation):
    if not terminal(properties, invocation):
        raise ValueError("Original follower is still live or cannot be verified")
    if (
        properties.get("Result") == "success"
        or receipt.get("state") != "failed"
        or receipt.get("hardware_started") is not False
        or receipt.get("detail") != "Predecessor exited unsuccessfully"
    ):
        raise ValueError("Only an unstarted dependency failure may be replaced")


def replacement_command(
    command,
    *,
    old_unit,
    new_unit,
    old_output,
    new_output,
    old_log,
    new_log,
    after_unit,
    after_invocation,
    after_receipt,
):
    if command[:2] != ["systemd-run", "--user"]:
        raise ValueError("Expected a user service launch")
    result = list(command)
    substitutions = {
        "--unit=" + old_unit.removesuffix(".service"): "--unit=" + new_unit.removesuffix(".service"),
        "--property=StandardOutput=append:" + old_log: "--property=StandardOutput=append:" + new_log,
        "--property=StandardError=append:" + old_log: "--property=StandardError=append:" + new_log,
    }
    for old, new in substitutions.items():
        if result.count(old) != 1:
            raise ValueError("Missing or duplicate service launch field")
        result[result.index(old)] = new
    for key, value in (
        ("--output", new_output),
        ("--after-unit", after_unit),
        ("--after-invocation", after_invocation),
        ("--after-receipt", after_receipt),
    ):
        if result.count(key) != 1:
            raise ValueError("Missing or duplicate dependency argument")
        index = result.index(key) + 1
        if key == "--output" and result[index] != old_output:
            raise ValueError("Original output directory changed")
        result[index] = value
    if "--property=KillMode=control-group" not in result:
        raise ValueError("Recovery requires bounded control-group ownership")
    return result
