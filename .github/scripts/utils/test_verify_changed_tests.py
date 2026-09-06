#!/usr/bin/env python3
"""Tests for verify_changed_tests.py -- entry diffing, leg scoping, filtering and review gating."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "verify_changed_tests.py"

BASE_TESTS_YAML = """\
- name: unit alpha
  cmd: ./build/test/alpha
  skus:
    wh_n150_civ2:
      timeout: 10
    wh_n300_civ2:
      timeout: 10
  team: llk
  owner_id: U001
  arch: wormhole_b0
  dispatch_mode: fd

- name: unit beta
  cmd: ./build/test/beta
  skus:
    bh_p150:
      timeout: 20
  team: runtime
  owner_id: U002
  arch: blackhole
  dispatch_mode: fd
"""

BASE_GALAXY_YAML = """\
- name: galaxy alpha
  cmd: pytest tests/galaxy/test_alpha.py
  skus:
    wh_galaxy:
      timeout: 30
  team: models
  owner_id: U003
"""

SKU_CONFIG = """\
skus:
  wh_n150_civ2: {}
  wh_n300_civ2: {}
  bh_p150: {}
  wh_galaxy: {}
"""

DEFAULT_REVIEW_SKUS = "wh_galaxy"


class Repo:
    """A throwaway git repo the script can be pointed at."""

    def __init__(self, root: Path):
        self.root = root
        self.base = ""

    def write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def commit_base(self) -> None:
        self.git("add", "-A")
        self.git("commit", "-m", "base")
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, check=True, capture_output=True, text=True)
        self.base = result.stdout.strip()

    def scope(self, review_skus: str = DEFAULT_REVIEW_SKUS, files: list[str] | None = None):
        """
        Run the scope-only path.

        --event merge_group is exactly that path: it resolves the touched entries
        without building a matrix or calling the reviews API.
        """
        argv = [
            sys.executable,
            str(SCRIPT),
            "--event",
            "merge_group",
            "--base",
            self.base,
            "--sku-config",
            ".github/sku_config.yaml",
            "--review-skus",
            review_skus,
            "--output",
            "result.json",
        ]
        if files is not None:
            argv += ["--files", *files]
        result = subprocess.run(argv, cwd=self.root, capture_output=True, text=True)
        payload = None
        if result.returncode == 0:
            payload = json.loads((self.root / "result.json").read_text())
        return result.returncode, payload, result.stderr


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.git("init", "-q")
    r.git("config", "user.email", "gate@test")
    r.git("config", "user.name", "gate")
    r.write(".github/sku_config.yaml", SKU_CONFIG)
    r.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML)
    r.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML)
    r.commit_base()
    return r


def legs_for(payload, name):
    return [leg for leg in payload["run_legs"] if leg["name"] == name]


# --- nothing to do -----------------------------------------------------------


def test_no_changes_is_no_op(repo: Repo):
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"
    assert payload["run_legs"] == []
    assert payload["expected_leg_count"] == 0


def test_owner_id_change_needs_no_hardware(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("U001", "U999"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"
    assert payload["metadata_only"] == ["tests/pipeline_reorg/sample_unit_tests.yaml: unit alpha | arch=wormhole_b0"]


def test_team_change_needs_no_hardware(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("team: llk", "team: runtime"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"
    assert len(payload["metadata_only"]) == 1


def test_timeout_change_needs_no_hardware(repo: Repo):
    """The ceiling is already enforced statically by verify_time_budget.py."""
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace("      timeout: 10\n    wh_n300_civ2", "      timeout: 12\n    wh_n300_civ2"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"
    assert len(payload["metadata_only"]) == 1


def test_removed_entry_needs_no_hardware(repo: Repo):
    """Nothing is left to prove green."""
    kept = BASE_TESTS_YAML.split("\n- name: unit beta")[0] + "\n"
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", kept)
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"


def test_non_matrix_yaml_is_skipped(repo: Repo):
    """ttsim-skip-list.yaml is a per-arch mapping, not a list of test entries."""
    repo.write("tests/pipeline_reorg/ttsim-skip-list.yaml", "wormhole_b0:\n  - some::test\n")
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"
    assert payload["skipped_files"] == ["tests/pipeline_reorg/ttsim-skip-list.yaml"]


# --- behaviour-affecting edits ----------------------------------------------


def test_cmd_change_runs_every_sku_leg_of_that_entry(repo: Repo):
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace("./build/test/alpha", "./build/test/alpha --gtest_filter=X"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "run"
    assert {leg["sku"] for leg in payload["run_legs"]} == {"wh_n150_civ2", "wh_n300_civ2"}
    assert payload["expected_leg_count"] == 2
    # The untouched entry stays out of scope.
    assert legs_for(payload, "unit beta") == []
    assert payload["profiles"] == ["cpp"]


def test_added_entry_runs(repo: Repo):
    added = BASE_TESTS_YAML + textwrap.dedent(
        """
        - name: unit gamma
          cmd: ./build/test/gamma
          skus:
            bh_p150:
              timeout: 5
          team: llk
          owner_id: U004
          arch: blackhole
        """
    )
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", added)
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "run"
    assert [leg["reason"] for leg in payload["run_legs"]] == ["added"]
    assert legs_for(payload, "unit gamma")[0]["sku"] == "bh_p150"


def test_rename_is_add_plus_delete(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("unit alpha", "unit alpha v2"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "run"
    assert {leg["name"] for leg in payload["run_legs"]} == {"unit alpha v2"}
    assert all(leg["reason"] == "added" for leg in payload["run_legs"])


def test_adding_a_sku_runs_the_entry(repo: Repo):
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace(
            "    bh_p150:\n      timeout: 20\n",
            "    bh_p150:\n      timeout: 20\n    wh_n150_civ2:\n      timeout: 20\n",
        ),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "run"
    assert {leg["sku"] for leg in legs_for(payload, "unit beta")} == {"bh_p150", "wh_n150_civ2"}


def test_per_sku_tier_change_runs(repo: Repo):
    """tier selects which pipeline an entry runs in, so it is behaviour-affecting."""
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace(
            "    bh_p150:\n      timeout: 20\n", "    bh_p150:\n      timeout: 20\n      tier: 2\n"
        ),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "run"


def test_two_files_at_once_collect_both_profiles(repo: Repo):
    """One PR touching a gtest pipeline and a pytest pipeline needs two builds."""
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    repo.write(
        "tests/pipeline_reorg/sample_python_tests.yaml",
        BASE_GALAXY_YAML.replace("wh_galaxy", "bh_p150"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["profiles"] == ["cpp", "python"]
    assert payload["expected_leg_count"] == 3


def test_review_only_legs_do_not_pull_in_a_build(repo: Repo):
    """A blocked galaxy leg is never dispatched, so its build flavour is not needed."""
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "blocked"
    assert payload["profiles"] == []


# --- build profile derivation ------------------------------------------------


def test_pytest_command_needs_the_wheel_build(repo: Repo):
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace("./build/test/beta", "pytest tests/unit/test_beta.py"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["profiles"] == ["python"]


def test_sim_leg_takes_the_build_its_command_needs(repo: Repo):
    """Running under ttsim is a runtime concern, not a build flavour."""
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n")
    # pytest cmd -> wheel, exactly as the same entry's hardware legs would get.
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        BASE_GALAXY_YAML.replace("wh_galaxy", "sim_wh_n150"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["profiles"] == ["python"]

    # gtest cmd on the same sim SKU -> plain cpp build, no wheel.
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        BASE_GALAXY_YAML.replace("wh_galaxy", "sim_wh_n150").replace(
            "pytest tests/galaxy/test_alpha.py", "./build/test/sim_alpha"
        ),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["profiles"] == ["cpp"]


def test_listed_yaml_selects_the_tracy_profile(repo: Repo):
    """Nothing in an entry says "profiler build", so that one case is told to the gate."""
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    argv = [
        sys.executable,
        str(SCRIPT),
        "--event",
        "merge_group",
        "--base",
        repo.base,
        "--sku-config",
        ".github/sku_config.yaml",
        "--review-skus",
        DEFAULT_REVIEW_SKUS,
        "--tracy-files",
        "sample_unit_tests.yaml",
    ]
    result = subprocess.run(argv + ["--output", "result.json"], cwd=repo.root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((repo.root / "result.json").read_text())["profiles"] == ["profiler"]


# --- review gating -----------------------------------------------------------


def test_galaxy_leg_blocks_instead_of_running(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "blocked"
    assert payload["run_legs"] == []
    assert len(payload["review_legs"]) == 1
    assert payload["review_legs"][0]["sku"] == "wh_galaxy"
    assert payload["review_legs"][0]["file"] == "tests/pipeline_reorg/sample_galaxy_tests.yaml"


def test_removing_a_sku_from_the_review_list_makes_it_runnable(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, payload, _ = repo.scope(review_skus="")
    assert code == 0
    assert payload["status"] == "run"
    assert payload["review_legs"] == []
    assert len(payload["run_legs"]) == 1


# --- fail-closed conditions --------------------------------------------------


def test_brand_new_yaml_is_scoped_generically(repo: Repo):
    """A new pipeline needs no registration: every entry in it is an addition."""
    repo.write("tests/pipeline_reorg/brand_new_tests.yaml", BASE_TESTS_YAML)
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "run"
    assert payload["expected_leg_count"] == 3
    assert all(leg["reason"] == "added" for leg in payload["run_legs"])


def test_duplicate_composite_key_fails_closed(repo: Repo):
    """name alone is not unique; an ambiguous key must not be guessed at."""
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML + BASE_TESTS_YAML)
    code, _, stderr = repo.scope()
    assert code == 1
    assert "share the key" in stderr


def test_entry_without_skus_fails_closed(repo: Repo):
    no_skus = BASE_TESTS_YAML + textwrap.dedent(
        """
        - name: unit orphan
          cmd: ./build/test/orphan
          team: llk
          owner_id: U005
        """
    )
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", no_skus)
    code, _, stderr = repo.scope()
    assert code == 1
    assert "no skus mapping" in stderr


def test_unknown_review_sku_name_fails_closed(repo: Repo):
    code, _, stderr = repo.scope(review_skus="wh_galaxyy")
    assert code == 1
    assert "not present in" in stderr


def test_shard_and_arch_disambiguate_same_name(repo: Repo):
    """Two entries sharing a name are distinct legs, and only the edited one runs."""
    sharded = textwrap.dedent(
        """\
        - name: shared name
          cmd: ./build/test/s --shard=0
          skus:
            bh_p150:
              timeout: 5
          team: llk
          owner_id: U006
          arch: blackhole
          gtest_shard_index: 0

        - name: shared name
          cmd: ./build/test/s --shard=1
          skus:
            bh_p150:
              timeout: 5
          team: llk
          owner_id: U006
          arch: blackhole
          gtest_shard_index: 1
        """
    )
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", sharded)
    repo.commit_base()
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", sharded.replace("--shard=1", "--shard=1 --extra"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["expected_leg_count"] == 1
    assert payload["run_legs"][0]["gtest_shard_index"] == 1


# --- filter ------------------------------------------------------------------


def run_with_matrices(repo: Repo, matrices: dict[str, list], review_skus: str = DEFAULT_REVIEW_SKUS):
    """Invoke the gate with prepare_test_matrix output stubbed out."""
    matrix_dir = repo.root / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    for stem, rows in matrices.items():
        (matrix_dir / f"{stem}.json").write_text(json.dumps(rows))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            repo.base,
            "--sku-config",
            ".github/sku_config.yaml",
            "--review-skus",
            review_skus,
            "--matrix-dir",
            str(matrix_dir),
            "--output",
            "result.json",
        ],
        cwd=repo.root,
        capture_output=True,
        text=True,
    )
    payload = json.loads((repo.root / "result.json").read_text()) if result.returncode == 0 else None
    return result.returncode, payload, result.stderr


def matrix_row(name, sku, **extra):
    row = {"name": name, "sku": sku, "cmd": f"./run {name}", "timeout": 10, "runs_on": "runner-label"}
    row.update(extra)
    return row


def test_filter_keeps_only_the_touched_legs(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    rows = [
        matrix_row("unit alpha2 [wh_n150_civ2]", "wh_n150_civ2", arch="wormhole_b0"),
        matrix_row("unit alpha2 [wh_n300_civ2]", "wh_n300_civ2", arch="wormhole_b0"),
        matrix_row("unit beta [bh_p150]", "bh_p150", arch="blackhole"),
    ]
    code, payload, stderr = run_with_matrices(repo, {"sample_unit_tests": rows})
    assert code == 0, stderr
    assert len(payload["legs"]) == 2
    assert {r["sku"] for r in payload["legs"]} == {"wh_n150_civ2", "wh_n300_civ2"}


def test_filter_fails_when_a_leg_has_no_matrix_row(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    rows = [matrix_row("unit alpha2 [wh_n150_civ2]", "wh_n150_civ2", arch="wormhole_b0")]
    code, _, stderr = run_with_matrices(repo, {"sample_unit_tests": rows})
    assert code == 1
    assert "did not resolve to a matrix row" in stderr


def test_filter_rejects_multihost_legs(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    rows = [
        matrix_row("unit alpha2 [wh_n150_civ2]", "wh_n150_civ2", arch="wormhole_b0", multihost=True),
        matrix_row("unit alpha2 [wh_n300_civ2]", "wh_n300_civ2", arch="wormhole_b0"),
    ]
    code, _, stderr = run_with_matrices(repo, {"sample_unit_tests": rows})
    assert code == 1
    assert "multi-host runners" in stderr


def test_filter_splits_simulator_legs(repo: Repo):
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n  sim_bh_p150: {}\n")
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        BASE_GALAXY_YAML.replace(
            "    wh_galaxy:\n      timeout: 30\n",
            "    sim_wh_n150:\n      timeout: 5\n    sim_bh_p150:\n      timeout: 5\n",
        ),
    )
    rows = [
        matrix_row("galaxy alpha [sim_wh_n150]", "sim_wh_n150", ttsim_lib="wh"),
        matrix_row("galaxy alpha [sim_bh_p150]", "sim_bh_p150", ttsim_lib="bh"),
    ]
    code, payload, stderr = run_with_matrices(repo, {"sample_sim_tests": rows})
    assert code == 0, stderr
    assert len(payload["legs"]) == 2
    assert all(r["sku"].startswith("sim_") for r in payload["legs"])
    assert payload["sim_libs"] == ["bh", "wh"]


def test_sim_leg_without_a_ttsim_lib_fails_closed(repo: Repo):
    """Its binary could not be fetched, so the leg could never run."""
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n")
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        BASE_GALAXY_YAML.replace("wh_galaxy", "sim_wh_n150"),
    )
    rows = [matrix_row("galaxy alpha [sim_wh_n150]", "sim_wh_n150")]
    code, _, stderr = run_with_matrices(repo, {"sample_sim_tests": rows})
    assert code == 1
    assert "name no ttsim_lib" in stderr


def test_hardware_and_sim_legs_share_one_matrix(repo: Repo):
    """One run job dispatches every leg; each picks its path off the sim_ prefix."""
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n")
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        BASE_GALAXY_YAML.replace("wh_galaxy", "sim_wh_n150"),
    )
    rows_hw = [
        matrix_row("unit alpha2 [wh_n150_civ2]", "wh_n150_civ2", arch="wormhole_b0"),
        matrix_row("unit alpha2 [wh_n300_civ2]", "wh_n300_civ2", arch="wormhole_b0"),
    ]
    rows_sim = [matrix_row("galaxy alpha [sim_wh_n150]", "sim_wh_n150", ttsim_lib="libttsim_wh.so")]
    code, payload, stderr = run_with_matrices(repo, {"sample_unit_tests": rows_hw, "sample_sim_tests": rows_sim})
    assert code == 0, stderr
    skus = {r["sku"] for r in payload["legs"]}
    assert skus == {"wh_n150_civ2", "wh_n300_civ2", "sim_wh_n150"}
    # Nothing is dropped on the way into the matrix.
    assert len(payload["legs"]) == payload["expected_leg_count"]
    assert payload["sim_libs"] == ["libttsim_wh.so"]


# --- digest ------------------------------------------------------------------


def test_digest_is_stable_and_scope_sensitive(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    _, first, _ = repo.scope()
    _, again, _ = repo.scope()
    assert first["leg_digest"] == again["leg_digest"]

    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("beta", "beta2"))
    _, different, _ = repo.scope()
    assert different["leg_digest"] != first["leg_digest"]


# --- packages legs -----------------------------------------------------------


def test_deb_path_cmd_takes_the_packages_install(repo: Repo):
    """/usr/share/tt-metalium comes from the debs, which no build artifact provides."""
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace("./build/test/alpha", "cmake -S /usr/share/tt-metalium/examples/eltwise_binary"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    touched = [leg for leg in payload["run_legs"] if leg["name"] == "unit alpha"]
    assert touched and all(leg["packages"] for leg in touched)
    # It still takes a normal build flavour; packages is orthogonal to the build.
    assert {leg["profile"] for leg in touched} == {"cpp"}


def test_build_tree_cmd_does_not_take_the_packages_install(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert not any(leg["packages"] for leg in payload["run_legs"])


def test_packages_flag_reaches_the_dispatched_row(repo: Repo):
    repo.write(
        "tests/pipeline_reorg/sample_unit_tests.yaml",
        BASE_TESTS_YAML.replace("./build/test/alpha", "cmake -S /usr/share/tt-metalium/examples/eltwise_binary"),
    )
    rows = [
        matrix_row("unit alpha [wh_n150_civ2]", "wh_n150_civ2", arch="wormhole_b0"),
        matrix_row("unit alpha [wh_n300_civ2]", "wh_n300_civ2", arch="wormhole_b0"),
    ]
    code, payload, stderr = run_with_matrices(repo, {"sample_unit_tests": rows})
    assert code == 0, stderr
    assert all(row["gate_packages"] for row in payload["legs"])
