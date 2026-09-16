# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Classify raw router lifecycle and shared-heartbeat observations."""

from __future__ import annotations

from typing import Any

BASE_FW_HEARTBEAT_MAGIC = 0xABCD0000
BASE_FW_HEARTBEAT_MASK = 0xFFFF0000


def _enum(raw: int | None, table: dict[str, int]) -> dict[str, Any]:
    return {
        "raw": raw,
        "name": next((name for name, value in table.items() if value == raw), None),
    }


def classify_liveness(
    sample: dict[str, Any],
    heartbeat: dict[str, int],
) -> dict[str, Any]:
    """Classify the bounded shared-heartbeat sequence without assuming monotonicity."""

    magic = heartbeat["magic"]
    mask = heartbeat["magic_mask"]
    classified = []
    fabric_values = []
    base_count = 0
    other_count = 0
    for observation in sample.get("liveness", []):
        raw = observation.get("heartbeat")
        if isinstance(raw, int) and raw & mask == magic:
            sample_format = "fabric"
            fabric_values.append(raw)
        elif isinstance(raw, int) and raw & BASE_FW_HEARTBEAT_MASK == BASE_FW_HEARTBEAT_MAGIC:
            sample_format = "base_fw"
            base_count += 1
        else:
            sample_format = "other"
            other_count += 1
        classified.append({"t": observation.get("t"), "raw": raw, "format": sample_format})

    reset = sample.get("health", {}).get("reset_bits", {}).get("erisc0")
    status = sample.get("status")
    if reset or status == "reset":
        classification = "reset"
    elif status == "unreadable":
        classification = "unknown"
    elif len(fabric_values) < 2:
        classification = "insufficient"
    elif len(set(fabric_values)) == 1:
        classification = "static"
    else:
        classification = "advancing"
    return {
        "classification": classification,
        "fabric_samples": len(fabric_values),
        "base_fw_samples": base_count,
        "other_samples": other_count,
        "samples": classified,
    }


def decode_lifecycle(sample: dict[str, Any], enums: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Name lifecycle words and conservatively classify the exit state."""

    lifecycle = sample.get("lifecycle", {})
    edm = _enum(lifecycle.get("edm_status"), enums.get("EDMStatus", {}))
    termination = _enum(lifecycle.get("termination_signal"), enums.get("TerminationSignal", {}))
    go = _enum(lifecycle.get("go_signal"), enums.get("RunMsg", {}))

    if edm["raw"] == 0 or edm["name"] is None:
        exit_state = "wiped_or_never_ran"
    elif edm["name"] == "TERMINATED" and termination["raw"] not in (None, 0):
        exit_state = "orderly_exit"
    elif edm["name"] == "READY_FOR_TRAFFIC" and termination["raw"] not in (None, 0):
        exit_state = "teardown_stuck"
    elif edm["name"] == "READY_FOR_TRAFFIC" and termination["raw"] == 0:
        exit_state = "running_or_host_gone"
    elif edm["name"] in {
        "STARTED",
        "REMOTE_HANDSHAKE_COMPLETE",
        "LOCAL_HANDSHAKE_COMPLETE",
        "INITIALIZATION_STARTED",
        "TXQ_INITIALIZED",
        "STREAM_REG_INITIALIZED",
        "DOWNSTREAM_EDM_SETUP_STARTED",
        "EDM_VCS_SETUP_COMPLETE",
        "WORKER_INTERFACES_INITIALIZED",
        "ETHERNET_HANDSHAKE_COMPLETE",
        "VCS_OPENED",
        "ROUTING_TABLE_INITIALIZED",
        "INITIALIZATION_COMPLETE",
    }:
        exit_state = "initializing"
    else:
        exit_state = "unknown"
    return {
        "edm_status": edm,
        "termination_signal": termination,
        "go_signal": go,
        "exit_state": exit_state,
    }
