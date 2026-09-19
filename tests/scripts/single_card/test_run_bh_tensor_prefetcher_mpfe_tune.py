# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("run_bh_tensor_prefetcher_mpfe_tune.py")
SPEC = importlib.util.spec_from_file_location("run_bh_tensor_prefetcher_mpfe_tune", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generate_candidates_uses_ordered_weight_tuples():
    candidates = MODULE.generate_candidates((0, 1))

    assert len(candidates) == 8
    assert {candidate.active for candidate in candidates} == {(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)}
    dynamic = next(candidate for candidate in candidates if candidate.dynamic and candidate.active == (0, 0, 1))
    assert dynamic.idle == (1, 1, 1)


def test_parse_and_verify_policy_marker():
    candidate = MODULE.Candidate(dynamic=True, low=0, medium=1, high=5)
    output = "info TENSOR_PREFETCHER_MPFE_POLICY idle=5/5/5 active=0/1/5 dynamic=1\n"

    policies = MODULE.parse_policy_markers(output)

    MODULE.verify_policy(candidate, policies)
    assert policies == [{"idle": (5, 5, 5), "active": (0, 1, 5), "dynamic": True}]


def test_parse_policy_rejects_invalid_dynamic_value():
    output = "TENSOR_PREFETCHER_MPFE_POLICY idle=0/1/5 active=0/1/5 dynamic=false\n"

    with pytest.raises(ValueError, match="invalid dynamic"):
        MODULE.parse_policy_markers(output)


def test_parse_timer_sums_multiple_lifetimes():
    output = "\n".join(
        [
            "TENSOR_PREFETCHER_MPFE_ACTIVE_LIFETIME -- elapsed: 1200us",
            "timer TENSOR_PREFETCHER_MPFE_ACTIVE_LIFETIME -- elapsed: 1.5ms",
            "timer TENSOR_PREFETCHER_MPFE_ACTIVE_LIFETIME -- elapsed: 300000ns",
        ]
    )

    elapsed_us, lifetime_count = MODULE.parse_timer_total_us(output)

    assert elapsed_us == pytest.approx(3000)
    assert lifetime_count == 3


def test_rank_candidates_by_median():
    static = MODULE.Candidate(False, 0, 1, 5)
    dynamic = MODULE.Candidate(True, 0, 1, 5)
    records = []
    for candidate, samples in ((static, (11, 10, 12)), (dynamic, (9, 8, 20))):
        for run_index, elapsed_us in enumerate(samples):
            records.append(
                {
                    "phase": "search",
                    "run_index": run_index,
                    "candidate": MODULE._candidate_dict(candidate),
                    "status": "passed",
                    "elapsed_us": elapsed_us,
                }
            )

    ranking = MODULE.rank_candidates(records, "search")

    assert [row["label"] for row in ranking] == [dynamic.label, static.label]
    assert [row["median_us"] for row in ranking] == [9, 11]


def test_resume_rejects_manifest_mismatch(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"schema_version": 1, "command": ["model.py"]}
    MODULE.validate_or_write_manifest(path, manifest, resume=False)

    with pytest.raises(RuntimeError, match="does not match"):
        MODULE.validate_or_write_manifest(
            path, {"schema_version": 1, "command": ["different.py"]}, resume=True
        )


def test_resume_retries_failed_run():
    candidate = MODULE._candidate_dict(MODULE.Candidate(False, 0, 1, 5))
    records = [
        {"phase": "search", "candidate": candidate, "run_index": 0, "status": "failed"},
        {"phase": "search", "candidate": candidate, "run_index": 1, "status": "passed"},
    ]

    assert MODULE._completed_keys(records) == {("search", "static-015", 1)}
