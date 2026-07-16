#!/usr/bin/env python3
"""Tests for prepare_test_matrix event routing and sku allowlist."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parent / "prepare_test_matrix.py"
SKU_CONFIG = REPO_ROOT / ".github" / "sku_config.yaml"


@pytest.fixture
def tests_yaml(tmp_path: Path) -> Path:
    """Logical-only smoke-like fixture (no prio twin keys)."""
    path = tmp_path / "tests.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: sample test
              cmd: echo ok
              skus:
                wh_n150_civ2:
                  timeout: 15
                bh_p150b_civ2_viommu:
                  timeout: 15
                wh_llmbox_civ2_viommu:
                  timeout: 15
              team: runtime
              owner_id: U000
            """
        )
    )
    return path


def run_matrix(tests_yaml: Path, enabled: str, *extra: str) -> list:
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(tests_yaml),
        enabled,
        str(SKU_CONFIG),
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    # Last line with matrix= for local runs
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("matrix="):
            return json.loads(line[len("matrix=") :])
    raise AssertionError(f"No matrix= in output:\n{result.stdout}")


def test_default_no_event_keeps_logical_skus(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS")
    skus = sorted(e["sku"] for e in matrix)
    assert skus == [
        "bh_p150b_civ2_viommu",
        "wh_llmbox_civ2_viommu",
        "wh_n150_civ2",
    ]
    assert all("logical_sku" not in e for e in matrix)


def test_merge_group_rewrites_aliased_skus(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--event", "merge_group")
    by_logical = {e.get("logical_sku", e["sku"]): e for e in matrix}

    assert by_logical["wh_n150_civ2"]["sku"] == "wh_n150_civ2"
    assert "logical_sku" not in by_logical["wh_n150_civ2"]

    assert by_logical["bh_p150b_civ2_viommu"]["sku"] == "bh_p150b_civ2_viommu_prio"
    assert by_logical["bh_p150b_civ2_viommu"]["runs_on"] == ["tt-ubuntu-2204-P150b-viommu-prio-stable"]

    assert by_logical["wh_llmbox_civ2_viommu"]["sku"] == "wh_llmbox_civ2_prio"
    assert by_logical["wh_llmbox_civ2_viommu"]["runs_on"] == ["tt-ubuntu-2204-N300-llmbox-viommu-prio-stable"]


def test_pull_request_does_not_rewrite(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--event", "pull_request")
    skus = sorted(e["sku"] for e in matrix)
    assert skus == [
        "bh_p150b_civ2_viommu",
        "wh_llmbox_civ2_viommu",
        "wh_n150_civ2",
    ]


def test_empty_allowlist_skips_all(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--sku-allowlist", "")
    assert matrix == []


def test_allowlist_intersects(tests_yaml: Path):
    matrix = run_matrix(
        tests_yaml,
        "ALL_SKUS_IN_TESTS",
        "--sku-allowlist",
        "wh_n150_civ2",
    )
    assert len(matrix) == 1
    assert matrix[0]["sku"] == "wh_n150_civ2"


def test_allowlist_then_merge_group_route(tests_yaml: Path):
    matrix = run_matrix(
        tests_yaml,
        "ALL_SKUS_IN_TESTS",
        "--sku-allowlist",
        "bh_p150b_civ2_viommu",
        "--event",
        "merge_group",
    )
    assert len(matrix) == 1
    assert matrix[0]["sku"] == "bh_p150b_civ2_viommu_prio"
    assert matrix[0]["logical_sku"] == "bh_p150b_civ2_viommu"


def test_sku_config_aliases_point_at_grouped_prio_skus():
    with open(SKU_CONFIG) as f:
        cfg = yaml.safe_load(f)["skus"]

    assert cfg["bh_p150b_civ2_viommu"]["merge_queue_sku"] == "bh_p150b_civ2_viommu_prio"
    assert cfg["wh_llmbox_civ2_viommu"]["merge_queue_sku"] == "wh_llmbox_civ2_prio"
    assert cfg["cpu_medium"]["merge_queue_sku"] == "cpu_medium_prio"

    # Many-to-one is allowed; prio targets must exist as concrete entries
    for logical in ("bh_p150b_civ2_viommu", "wh_llmbox_civ2_viommu", "cpu_medium"):
        alias = cfg[logical]["merge_queue_sku"]
        assert alias in cfg
        assert "runs_on" in cfg[alias]
