#!/usr/bin/env python3
"""Tests for prepare_test_matrix event routing and sku allowlist."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parent / "prepare_test_matrix.py"
SKU_CONFIG = REPO_ROOT / ".github" / "sku_config.yaml"
PIPELINE = REPO_ROOT / "tests" / "pipeline_reorg"


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


def run_matrix(
    tests_yaml: Path,
    enabled: str,
    *extra: str,
    env: dict | None = None,
) -> list:
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(tests_yaml),
        enabled,
        str(SKU_CONFIG),
        *extra,
    ]
    run_env = os.environ.copy()
    run_env.pop("GITHUB_OUTPUT", None)
    if env:
        run_env.update(env)
    else:
        run_env.pop("MATRIX_EVENT_NAME", None)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=run_env)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("matrix="):
            return json.loads(line[len("matrix=") :])
    raise AssertionError(f"No matrix= in output:\n{result.stdout}")


def concrete_skus(matrix: list) -> set[str]:
    return {e["sku"] for e in matrix}


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


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "schedule"])
def test_non_merge_group_events_do_not_rewrite(tests_yaml: Path, event: str):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--event", event)
    assert "bh_p150b_civ2_viommu_prio" not in concrete_skus(matrix)
    assert "bh_p150b_civ2_viommu" in concrete_skus(matrix)


def test_empty_allowlist_skips_all(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--sku-allowlist", "")
    assert matrix == []


def test_whitespace_allowlist_skips_all(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--sku-allowlist", "  ,  ")
    assert matrix == []


def test_allowlist_star_is_not_all_at_script_level(tests_yaml: Path):
    """Workflows treat '*' as omit-flag; the script itself has no '*' sentinel."""
    matrix = run_matrix(tests_yaml, "ALL_SKUS_IN_TESTS", "--sku-allowlist", "*")
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


def test_allowlist_unknown_sku_yields_empty(tests_yaml: Path):
    matrix = run_matrix(
        tests_yaml,
        "ALL_SKUS_IN_TESTS",
        "--sku-allowlist",
        "does_not_exist",
    )
    assert matrix == []


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


def test_rewrite_preserves_logical_timeout_and_names_concrete_sku(tmp_path: Path):
    path = tmp_path / "tests.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: timeout check
              cmd: echo ok
              skus:
                bh_p150b_civ2_viommu:
                  timeout: 11
              team: triage
            """
        )
    )
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS", "--event", "merge_group")
    assert len(matrix) == 1
    assert matrix[0]["timeout"] == 11
    assert matrix[0]["sku"] == "bh_p150b_civ2_viommu_prio"
    assert matrix[0]["name"] == "timeout check [bh_p150b_civ2_viommu_prio]"


def test_cpu_medium_routes_on_merge_group(tmp_path: Path):
    path = tmp_path / "tests.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: fabric cpu
              cmd: echo ok
              skus:
                cpu_medium:
                  timeout: 10
              team: scaleout
            """
        )
    )
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS", "--event", "merge_group")
    assert matrix[0]["sku"] == "cpu_medium_prio"
    assert matrix[0]["logical_sku"] == "cpu_medium"
    assert matrix[0]["runs_on"] == ["tt-ubuntu-2204-medium-prio-stable"]


def test_matrix_event_name_env_triggers_rewrite(tests_yaml: Path):
    matrix = run_matrix(
        tests_yaml,
        "ALL_SKUS_IN_TESTS",
        env={"MATRIX_EVENT_NAME": "merge_group"},
    )
    assert "bh_p150b_civ2_viommu_prio" in concrete_skus(matrix)


def test_explicit_event_overrides_env(tests_yaml: Path):
    matrix = run_matrix(
        tests_yaml,
        "ALL_SKUS_IN_TESTS",
        "--event",
        "pull_request",
        env={"MATRIX_EVENT_NAME": "merge_group"},
    )
    assert "bh_p150b_civ2_viommu_prio" not in concrete_skus(matrix)


def test_backcompat_no_flags_with_dual_sku_list(tmp_path: Path):
    """Synthetic: listing prio twins still expands both if someone misconfigures a YAML.

    Production pipeline_reorg lists must not contain *_prio keys (see
    test_pipeline_reorg_yamls_have_no_prio_sku_keys).
    """
    path = tmp_path / "legacy.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: legacy
              cmd: echo ok
              skus:
                bh_p150b_civ2_viommu:
                  timeout: 10
                bh_p150b_civ2_viommu_prio:
                  timeout: 10
              team: runtime
            """
        )
    )
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS")
    assert concrete_skus(matrix) == {"bh_p150b_civ2_viommu", "bh_p150b_civ2_viommu_prio"}
    assert len(matrix) == 2


def test_dual_sku_list_plus_merge_group_duplicates_prio(tmp_path: Path):
    """Hazard if a test list still has prio twins AND --event merge_group is set.

    Gate/non-gate pipeline_reorg YAMLs must not list *_prio; this guards the script
    contract so a regression is obvious.
    """
    path = tmp_path / "legacy.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: legacy
              cmd: echo ok
              skus:
                bh_p150b_civ2_viommu:
                  timeout: 10
                bh_p150b_civ2_viommu_prio:
                  timeout: 10
              team: runtime
            """
        )
    )
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS", "--event", "merge_group")
    assert [e["sku"] for e in matrix] == [
        "bh_p150b_civ2_viommu_prio",
        "bh_p150b_civ2_viommu_prio",
    ]


def test_backcompat_explicit_csv_enabled_skus_filters(tests_yaml: Path):
    matrix = run_matrix(tests_yaml, "wh_n150_civ2,bh_p150b_civ2_viommu")
    assert concrete_skus(matrix) == {"wh_n150_civ2", "bh_p150b_civ2_viommu"}


def test_all_skus_on_empty_list_yaml_yields_empty_matrix(tmp_path: Path):
    """Placeholder gate list ('[]') + ALL_SKUS_IN_TESTS skips, does not fail."""
    path = tmp_path / "empty.yaml"
    path.write_text("[]\n")
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS", "--event", "merge_group")
    assert matrix == []


def test_all_skus_on_comment_only_yaml_yields_empty_matrix(tmp_path: Path):
    """Commented-out entries (yaml parses to None) also skip rather than fail."""
    path = tmp_path / "comments.yaml"
    path.write_text("# just a header\n# - name: not yet\n")
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS")
    assert matrix == []


def test_models_merge_gate_placeholder_skips(tmp_path: Path):
    """The real (currently empty) models gate list must not fail on merge_group."""
    matrix = run_matrix(
        PIPELINE / "models_merge_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "merge_group",
    )
    assert matrix == []


def test_broken_merge_queue_alias_fails(tmp_path: Path):
    tests = tmp_path / "tests.yaml"
    tests.write_text(
        textwrap.dedent(
            """\
            - name: bad alias
              cmd: echo ok
              skus:
                wh_n150_civ2:
                  timeout: 5
              team: runtime
            """
        )
    )
    broken_cfg = tmp_path / "sku_config.yaml"
    broken_cfg.write_text(
        textwrap.dedent(
            """\
            skus:
              wh_n150_civ2:
                runs_on: [runner-a]
                merge_queue_sku: missing_prio
            """
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tests),
            "ALL_SKUS_IN_TESTS",
            str(broken_cfg),
            "--event",
            "merge_group",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k not in ("GITHUB_OUTPUT", "MATRIX_EVENT_NAME")},
    )
    assert result.returncode != 0
    assert "missing_prio" in result.stdout + result.stderr


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


# ---------------------------------------------------------------------------
# Integration: real gate test lists (expected post-migration behavior)
# ---------------------------------------------------------------------------


def test_runtime_smoke_merge_gate_excludes_n150():
    """Merge-gate smoke keeps prior coverage (no wh_n150_civ2); PR-gate list owns n150."""
    matrix = run_matrix(
        PIPELINE / "runtime_validation_merge_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "push",
    )
    assert concrete_skus(matrix) == {
        "wh_n300_civ2",
        "wh_llmbox_civ2_viommu",
        "bh_p150b_civ2_viommu",
    }
    assert "wh_n150_civ2" not in concrete_skus(matrix)


def test_runtime_smoke_merge_gate_routes_prio_on_merge_group():
    matrix = run_matrix(
        PIPELINE / "runtime_validation_merge_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "merge_group",
    )
    assert concrete_skus(matrix) == {
        "wh_n300_civ2",
        "wh_llmbox_civ2_prio",
        "bh_p150b_civ2_viommu_prio",
    }


def test_runtime_basic_merge_gate_matches_prior_coverage():
    push = run_matrix(
        PIPELINE / "runtime_validation_basic_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "push",
    )
    mq = run_matrix(
        PIPELINE / "runtime_validation_basic_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "merge_group",
    )
    assert concrete_skus(push) == {"wh_n150_civ2", "bh_p150b_civ2_viommu"}
    assert concrete_skus(mq) == {"wh_n150_civ2", "bh_p150b_civ2_viommu_prio"}


def test_llk_merge_gate_allowlist_wh_only():
    matrix = run_matrix(
        PIPELINE / "llk_merge_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "push",
        "--sku-allowlist",
        "wh_n150_civ2",
    )
    assert concrete_skus(matrix) == {"wh_n150_civ2"}
    assert len(matrix) == 5  # 4 FD shards + 1 SD


def test_llk_merge_gate_allowlist_bh_routes_prio():
    matrix = run_matrix(
        PIPELINE / "llk_merge_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "merge_group",
        "--sku-allowlist",
        "bh_p150b_civ2_viommu",
    )
    assert concrete_skus(matrix) == {"bh_p150b_civ2_viommu_prio"}
    assert len(matrix) == 3  # 2 FD shards + 1 SD
    assert all(e["logical_sku"] == "bh_p150b_civ2_viommu" for e in matrix)


def test_llk_pr_gate_uses_viommu_not_bh_p150b_civ2():
    """Functional change: PR LLK BH moved from bh_p150b_civ2 to bh_p150b_civ2_viommu."""
    pr = run_matrix(
        PIPELINE / "llk_pr_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "pull_request",
    )
    mq = run_matrix(
        PIPELINE / "llk_pr_gate_tests.yaml",
        "ALL_SKUS_IN_TESTS",
        "--event",
        "merge_group",
    )
    assert concrete_skus(pr) == {"wh_n150_civ2", "bh_p150b_civ2_viommu"}
    assert "bh_p150b_civ2" not in concrete_skus(pr)
    assert concrete_skus(mq) == {"wh_n150_civ2", "bh_p150b_civ2_viommu_prio"}


def test_pipeline_reorg_yamls_have_no_prio_sku_keys():
    """Prio SKUs belong only in sku_config (merge_queue_sku targets), never in test lists."""
    leftovers = []
    sku_prio = re.compile(r"^(\s*)([A-Za-z0-9_]*_prio)\s*:")
    for path in sorted(PIPELINE.rglob("*.yaml")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = sku_prio.match(line)
            if m:
                leftovers.append(f"{path.name}:{i}:{m.group(2)}")
    assert leftovers == []
