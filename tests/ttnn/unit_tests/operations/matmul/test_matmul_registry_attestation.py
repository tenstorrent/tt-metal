# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
import ttnn


pytestmark = pytest.mark.use_module_device

REGISTRY = ttnn._ttnn.operations.matmul
ZERO_SHA256 = "0" * 64
EXPECTED_KEYS = {
    "artifact_kind",
    "schema_version",
    "device_attestation_status",
    "codegen_recipe_abi",
    "board_capability_class",
    "actual_semantic_source_sha256",
    "actual_build_identity_sha256",
    "actual_topology_sha256",
    "actual_runtime_capability_sha256",
}
KNOWN_STATUSES = {
    "success",
    "query_failed",
    "device_uninitialized",
    "remote_device",
    "not_one_chip",
    "active_sub_device_manager",
    "unsupported_architecture",
    "unsupported_board",
    "unsupported_cluster",
    "board_cluster_mismatch",
    "firmware_unavailable",
    "invalid_capability",
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def test_matmul_registry_compatibility_attestation_binding_contract(device) -> None:
    assert callable(REGISTRY.matmul_registry_compatibility_attestation)

    report = dict(REGISTRY.matmul_registry_compatibility_attestation(device))

    assert set(report) == EXPECTED_KEYS
    assert report["artifact_kind"] == "ttnn_matmul_registry_runtime_attestation"
    assert type(report["schema_version"]) is int and report["schema_version"] == 1
    assert type(report["codegen_recipe_abi"]) is int and report["codegen_recipe_abi"] > 0
    assert type(report["board_capability_class"]) is int
    assert isinstance(report["device_attestation_status"], str)
    assert report["device_attestation_status"] in KNOWN_STATUSES

    digest_names = {
        "actual_semantic_source_sha256",
        "actual_build_identity_sha256",
        "actual_topology_sha256",
        "actual_runtime_capability_sha256",
    }
    assert all(_is_sha256(report[name]) for name in digest_names)
    assert report["actual_semantic_source_sha256"] != ZERO_SHA256
    assert report["actual_build_identity_sha256"] != ZERO_SHA256

    if report["device_attestation_status"] == "success":
        assert report["actual_topology_sha256"] != ZERO_SHA256
        assert report["actual_runtime_capability_sha256"] != ZERO_SHA256
    else:
        # The public binding must never expose partial device evidence after a
        # failed query. Consumers reject the status and receive zero digests.
        assert report["actual_topology_sha256"] == ZERO_SHA256
        assert report["actual_runtime_capability_sha256"] == ZERO_SHA256
