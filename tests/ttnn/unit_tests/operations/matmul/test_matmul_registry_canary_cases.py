# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tests.ttnn.unit_tests.operations.matmul.matmul_registry_canary_cases import (
    CanaryLockError,
    CanarySemanticCase,
    represented_semantic_cases,
)


def _entry(domain: str, *, alpha_bits=None, beta_bits=None) -> dict:
    return {
        "domain": domain,
        "key": {"alpha_f32_bits": alpha_bits, "beta_f32_bits": beta_bits},
    }


def test_empty_lock_has_no_cases_without_inventing_required_domains() -> None:
    assert represented_semantic_cases({"entries": []}) == ()
    with pytest.raises(CanaryLockError, match="no valid dense entry"):
        represented_semantic_cases({"entries": []}, require_populated=True)


def test_single_dense_matmul_entry_is_a_complete_canary_case_set() -> None:
    lock = {"entries": [_entry("dense.matmul")]}

    assert represented_semantic_cases(lock) == (CanarySemanticCase("dense.matmul"),)


def test_cases_are_unique_and_follow_only_represented_operation_semantics() -> None:
    lock = {
        "entries": [
            _entry("dense.addmm", alpha_bits=0x3F800000, beta_bits=0x80000000),
            _entry("dense.linear"),
            _entry("dense.addmm", alpha_bits=0x3F800000, beta_bits=0),
            _entry("dense.linear"),
        ]
    }

    assert represented_semantic_cases(lock) == (
        CanarySemanticCase("dense.linear"),
        CanarySemanticCase("dense.addmm", 0),
        CanarySemanticCase("dense.addmm", 0x80000000),
    )
    assert represented_semantic_cases(lock)[-1].beta == -0.0


@pytest.mark.parametrize(
    "entry",
    [
        _entry("dense.matmul", alpha_bits=0x3F800000),
        _entry("dense.addmm", alpha_bits=0x3F000000, beta_bits=0),
        _entry("dense.addmm", alpha_bits=0x3F800000, beta_bits=0x3F800000),
        _entry("unsupported.domain"),
        {"domain": "dense.matmul", "key": None},
    ],
)
def test_malformed_or_unsupported_semantics_fail_closed(entry: dict) -> None:
    with pytest.raises(CanaryLockError):
        represented_semantic_cases({"entries": [entry]})
