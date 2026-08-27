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
    sku_config: Path | None = None,
) -> list:
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(tests_yaml),
        enabled,
        str(sku_config or SKU_CONFIG),
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


def test_cmd_strips_indented_comments(tmp_path: Path):
    """Whole-line '#' comments inside a cmd block are dropped from the matrix cmd.

    Regression for galaxy_stress_tests.yaml: a comment carrying a lone apostrophe
    and parentheses (e.g. "overrides setup-job's 5s default") otherwise lands in
    the matrix JSON's cmd and breaks downstream shell single-quoting of the cmd.
    """
    path = tmp_path / "comment_cmd.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: commented test
              cmd: |
                # 0 disables the timeout (overrides setup-job's 5s default), which
                # else false-fails slow subtests (size_4096 unicast ~6.2s).
                run_the_thing --filter 'name.*_short'
              skus:
                wh_n150_civ2:
                  timeout: 15
              team: runtime
              owner_id: U000
            """
        )
    )
    matrix = run_matrix(path, "wh_n150_civ2")
    assert len(matrix) == 1
    cmd = matrix[0]["cmd"]
    # Comment lines (and their apostrophes/parentheses) are gone; the real
    # command line — including its own single-quoted arg — survives intact.
    assert "#" not in cmd
    assert "setup-job's" not in cmd
    assert "run_the_thing --filter 'name.*_short'" in cmd


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
# Multihost routing (single-host vs exabox workflow split)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sku_name,sku_yaml,expected_multihost",
    [
        (
            "legacy_alloc",
            """\
              legacy_alloc:
                runs_on: [exabox-multihost-with-nfs]
                allocation:
                  type: Count
                  count: 4
            """,
            True,
        ),
        (
            "exabox_label",
            """\
              exabox_label:
                runs_on: [exabox-multihost-ci-sc4]
            """,
            True,
        ),
        (
            "ordinary",
            """\
              ordinary:
                runs_on: [arch-blackhole, in-service]
            """,
            False,
        ),
    ],
    ids=["legacy_allocation", "exabox_multihost_label", "ordinary_runner"],
)
def test_multihost_flag_from_sku_config(tmp_path: Path, sku_name: str, sku_yaml: str, expected_multihost: bool):
    """Legs with allocation or an exabox-multihost* runs_on label are multihost."""
    tests = tmp_path / "tests.yaml"
    tests.write_text(
        textwrap.dedent(
            f"""\
            - name: routing probe
              cmd: echo ok
              skus:
                {sku_name}:
                  timeout: 10
              team: models
            """
        )
    )
    cfg = tmp_path / "sku_config.yaml"
    cfg.write_text(textwrap.dedent(f"skus:\n{sku_yaml}"))
    matrix = run_matrix(tests, "ALL_SKUS_IN_TESTS", sku_config=cfg)
    assert len(matrix) == 1
    assert matrix[0]["sku"] == sku_name
    assert matrix[0]["multihost"] is expected_multihost


def test_real_exabox_skus_are_marked_multihost(tmp_path: Path):
    """Production bh_sc* SKUs route via exabox-multihost-ci-* labels in sku_config."""
    tests = tmp_path / "tests.yaml"
    tests.write_text(
        textwrap.dedent(
            """\
            - name: sc1 probe
              cmd: echo ok
              skus:
                bh_sc1:
                  timeout: 10
              team: models
            - name: sc4 probe
              cmd: echo ok
              skus:
                bh_sc4:
                  timeout: 10
              team: models
            - name: single-host probe
              cmd: echo ok
              skus:
                bh_p150:
                  timeout: 10
              team: models
            """
        )
    )
    by_sku = {e["sku"]: e for e in run_matrix(tests, "bh_sc1,bh_sc4,bh_p150")}
    assert by_sku["bh_sc1"]["multihost"] is True
    assert by_sku["bh_sc4"]["multihost"] is True
    assert by_sku["bh_p150"]["multihost"] is False


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


def _run_matrix_raw(tests_yaml: Path, *extra: str) -> subprocess.CompletedProcess:
    """Invoke the script without asserting success, for failure-path assertions."""
    run_env = os.environ.copy()
    run_env.pop("GITHUB_OUTPUT", None)
    run_env.pop("MATRIX_EVENT_NAME", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tests_yaml), "ALL_SKUS_IN_TESTS", str(SKU_CONFIG), *extra],
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
    )


_COMMANDLESS_YAML = """\
- name: commandless test
  model: google/gemma-4-E2B-it
  skus:
    wh_n150_civ2:
      timeout: 15
  team: models
  owner_id: U000
"""


def test_absent_cmd_is_allowed_with_flag(tmp_path: Path):
    """--allow-missing-cmd lets vLLM-style entries omit `cmd` (impl builds it)."""
    path = tmp_path / "tests.yaml"
    path.write_text(_COMMANDLESS_YAML)
    matrix = run_matrix(path, "ALL_SKUS_IN_TESTS", "--allow-missing-cmd")
    assert len(matrix) == 1
    assert "cmd" not in matrix[0]


def test_absent_cmd_rejected_by_default(tmp_path: Path):
    """Without the flag the strict contract holds: a missing `cmd` is an error."""
    path = tmp_path / "tests.yaml"
    path.write_text(_COMMANDLESS_YAML)
    result = _run_matrix_raw(path)  # no --allow-missing-cmd
    assert result.returncode != 0, result.stdout
    assert "cmd is missing" in result.stdout + result.stderr


@pytest.mark.parametrize("empty_cmd", ['""', '"   "'])
@pytest.mark.parametrize("extra", [(), ("--allow-missing-cmd",)])
def test_present_but_empty_cmd_is_always_rejected(tmp_path: Path, empty_cmd: str, extra: tuple):
    """An empty `cmd` runs nothing but reports success — rejected even with the flag."""
    path = tmp_path / "tests.yaml"
    path.write_text(
        textwrap.dedent(
            f"""\
            - name: empty cmd test
              cmd: {empty_cmd}
              skus:
                wh_n150_civ2:
                  timeout: 15
              team: runtime
              owner_id: U000
            """
        )
    )
    result = _run_matrix_raw(path, *extra)
    assert result.returncode != 0, result.stdout
    assert "cmd is present but empty" in result.stdout + result.stderr


def _run_with_skus(tests_yaml: Path, enabled: str, *extra: str) -> subprocess.CompletedProcess:
    """Invoke the script with an explicit enabled-SKU list, without asserting success."""
    run_env = os.environ.copy()
    run_env.pop("GITHUB_OUTPUT", None)
    run_env.pop("MATRIX_EVENT_NAME", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tests_yaml), enabled, str(SKU_CONFIG), *extra],
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
    )


def sim_libs_line(result) -> str:
    """The sim-libs value the script printed (stdout form, no GITHUB_OUTPUT)."""
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("sim-libs="):
            return line[len("sim-libs=") :]
    raise AssertionError(f"No sim-libs= in output:\n{result.stdout}")


def test_sim_libs_lists_only_selected_sim_skus(tmp_path: Path):
    """sim-libs carries each selected sim SKU's ttsim_lib, deduped, and nothing for HW SKUs."""
    path = tmp_path / "tests.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: sim and hw test
              cmd: echo ok
              skus:
                wh_n300_civ2:
                  timeout: 15
                sim_wh_n300:
                  timeout: 15
                sim_bh_p150:
                  timeout: 15
              team: runtime
              owner_id: U000
            - name: second test on the same sim sku
              cmd: echo ok
              skus:
                sim_wh_n300:
                  timeout: 15
              team: runtime
              owner_id: U000
            """
        )
    )
    result = _run_with_skus(path, "wh_n300_civ2,sim_wh_n300,sim_bh_p150")
    assert result.returncode == 0, result.stdout + result.stderr
    # sorted + unique, despite sim_wh_n300 appearing in two entries
    assert json.loads(sim_libs_line(result)) == ["libttsim_bh.so", "libttsim_wh_x2.so"]

    # Every sim leg also carries its own lib, which is what setup-ttsim installs.
    matrix = json.loads(re.search(r"^matrix=(.*)$", result.stdout, re.M).group(1))
    libs = {e["sku"]: e.get("ttsim_lib") for e in matrix}
    assert libs["sim_wh_n300"] == "libttsim_wh_x2.so"
    assert libs["sim_bh_p150"] == "libttsim_bh.so"
    assert libs["wh_n300_civ2"] is None


def test_sim_libs_empty_for_hardware_only_matrix(tmp_path: Path, tests_yaml: Path):
    """No sim SKUs selected -> empty sim-libs, which is how impls skip fetch-ttsim."""
    result = _run_with_skus(tests_yaml, "wh_n150_civ2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sim_libs_line(result) == "[]"


def test_tests_present_but_no_enabled_sku_is_fatal(tests_yaml: Path):
    """Tests exist and the SKU set cannot run any of them: a miswired pipeline, so fail."""
    result = _run_with_skus(tests_yaml, "bh_galaxy")
    assert result.returncode != 0, result.stdout
    assert "No tests selected for enabled SKUs" in result.stdout


@pytest.mark.parametrize("body", ["", "# placeholder, no tests yet\n"])
@pytest.mark.parametrize("enabled", ["wh_n150_civ2", "ALL_SKUS_IN_TESTS"])
def test_no_tests_in_yaml_warns_and_passes(tmp_path: Path, body: str, enabled: str):
    """An empty / placeholder tests YAML is vacuously correct: warn, emit matrix=[], exit 0.

    Covers both SKU forms because the explicit-list form reaches build_test_matrix while
    ALL_SKUS_IN_TESTS short-circuits in main; e.g. l2-nightly drives the placeholder
    ttsim_unit_tests.yaml with an explicit list.
    """
    path = tmp_path / "tests.yaml"
    path.write_text(body)
    result = _run_with_skus(path, enabled)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert re.search(r"^matrix=\[\]$", result.stdout, re.M)
    assert sim_libs_line(result) == "[]"
