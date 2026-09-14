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
DEFAULT_NON_MATRIX = "ttsim-skip-list.yaml"


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

    def scope(
        self,
        review_skus: str = DEFAULT_REVIEW_SKUS,
        files: list[str] | None = None,
        non_matrix_files: str = DEFAULT_NON_MATRIX,
    ):
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
            "--non-matrix-files",
            non_matrix_files,
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


def test_declared_non_matrix_yaml_is_skipped(repo: Repo):
    """ttsim-skip-list.yaml is a per-arch mapping, not a list of test entries."""
    repo.write("tests/pipeline_reorg/ttsim-skip-list.yaml", "wormhole_b0:\n  - some::test\n")
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "no_op"
    assert payload["skipped_files"] == ["tests/pipeline_reorg/ttsim-skip-list.yaml"]


def test_undeclared_non_list_yaml_fails_closed(repo: Repo):
    """A matrix reshaped into a mapping must not read as "no tests"."""
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", "wormhole_b0:\n  - some::test\n")
    code, _, stderr = repo.scope()
    assert code == 1
    assert "is not a list of test entries" in stderr
    assert "NON_MATRIX_YAMLS" in stderr


def test_reshaping_a_matrix_is_caught_even_if_the_old_shape_was_valid(repo: Repo):
    """The base revision parsed fine; only the new shape is broken."""
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", "just-a-scalar\n")
    code, _, stderr = repo.scope()
    assert code == 1
    assert "is not a list of test entries" in stderr


def test_declaring_a_file_non_matrix_does_not_leak_to_others(repo: Repo):
    """The exemption is per-file, not a blanket relaxation."""
    repo.write("tests/pipeline_reorg/ttsim-skip-list.yaml", "wormhole_b0:\n  - some::test\n")
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", "blackhole:\n  - other::test\n")
    code, _, stderr = repo.scope()
    assert code == 1
    assert "sample_unit_tests.yaml" in stderr


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
    assert payload["profiles"] == ["default"]


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


def test_review_only_legs_do_not_pull_in_a_build(repo: Repo):
    """A blocked galaxy leg is never dispatched, so its build flavour is not needed."""
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["status"] == "blocked"
    assert payload["profiles"] == []


# --- build profile derivation ------------------------------------------------


def test_every_build_carries_the_wheel(repo: Repo):
    """
    Command text cannot decide whether a wheel is needed.

    ops_integration_tests.yaml runs `python3 scripts/...` and its pipeline passes a
    wheel; runtime_unit_tests.yaml runs `python3 tests/scripts/...` and its pipeline
    does not; shell wrappers hide python entirely. A spare wheel costs build time, a
    missing one fails the leg, so one flavour carries it for everything.
    """
    for cmd in (
        "./build/test/alpha --gtest_filter=X",
        "pytest tests/unit/test_alpha.py",
        "python tt-train/scripts/run_models.py --model_config x.yaml",
        "python3 scripts/detect_undocumented_ttnn_ops.py",
        "./tests/scripts/run_ttnn_examples.sh",
    ):
        repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("./build/test/alpha", cmd))
        code, payload, _ = repo.scope()
        assert code == 0, cmd
        assert payload["profiles"] == ["default"], cmd


def test_two_files_needing_the_same_flavour_build_once(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_unit_tests.yaml", BASE_TESTS_YAML.replace("alpha", "alpha2"))
    repo.write(
        "tests/pipeline_reorg/sample_python_tests.yaml",
        BASE_GALAXY_YAML.replace("wh_galaxy", "bh_p150"),
    )
    code, payload, _ = repo.scope()
    assert code == 0
    assert payload["profiles"] == ["default"]


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
    assert {leg["profile"] for leg in touched} == {"default"}


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


# --- mirrored definitions ----------------------------------------------------

MIRRORED_YAML = """\
- name: t3k fabric tests
  cmd: ./build/test/fabric
  skus:
    bh_p150:
      timeout: 10
  team: scaleout
  owner_id: U010
"""


def test_mirrored_entry_in_two_yamls_is_not_a_collision(repo: Repo):
    """
    Real pipelines mirror entries: "t3k fabric tests" lives in both
    fabric_sanity_tests.yaml and fabric_tests.yaml. Touching both must dispatch both.
    """
    repo.write("tests/pipeline_reorg/fabric_sanity_tests.yaml", MIRRORED_YAML)
    repo.write("tests/pipeline_reorg/fabric_tests.yaml", MIRRORED_YAML)
    row = [matrix_row("t3k fabric tests [bh_p150]", "bh_p150")]
    code, payload, stderr = run_with_matrices(repo, {"fabric_sanity_tests": list(row), "fabric_tests": list(row)})
    assert code == 0, stderr
    assert len(payload["legs"]) == 2
    assert {r["source_yaml"] for r in payload["legs"]} == {"fabric_sanity_tests", "fabric_tests"}
    assert len(payload["legs"]) == payload["expected_leg_count"]


def test_mirrored_entry_touched_in_one_yaml_runs_only_that_one(repo: Repo):
    repo.write("tests/pipeline_reorg/fabric_sanity_tests.yaml", MIRRORED_YAML)
    repo.write("tests/pipeline_reorg/fabric_tests.yaml", MIRRORED_YAML)
    repo.commit_base()
    repo.write(
        "tests/pipeline_reorg/fabric_tests.yaml",
        MIRRORED_YAML.replace("./build/test/fabric", "./build/test/fabric --extra"),
    )
    row_sanity = [matrix_row("t3k fabric tests [bh_p150]", "bh_p150")]
    row_tests = [matrix_row("t3k fabric tests [bh_p150]", "bh_p150", cmd="./build/test/fabric --extra")]
    code, payload, stderr = run_with_matrices(repo, {"fabric_sanity_tests": row_sanity, "fabric_tests": row_tests})
    assert code == 0, stderr
    assert len(payload["legs"]) == 1
    assert payload["legs"][0]["source_yaml"] == "fabric_tests"


def test_same_entry_twice_in_one_yaml_still_fails_closed(repo: Repo):
    """Mirroring across yamls is fine; ambiguity inside one yaml is not."""
    repo.write("tests/pipeline_reorg/fabric_tests.yaml", MIRRORED_YAML)
    row = matrix_row("t3k fabric tests [bh_p150]", "bh_p150")
    code, _, stderr = run_with_matrices(repo, {"fabric_tests": [row, dict(row)]})
    assert code == 1
    assert "same entry twice" in stderr


# --- CODEOWNERS resolution ---------------------------------------------------

CODEOWNERS = """\
# last match wins
/* @tenstorrent/metalium-developers-infra
tests/pipeline_reorg/ @roseli-TT @tdowdallTT
tests/pipeline_reorg/sample_galaxy_tests.yaml @mtairum @uaydonat
tests/pipeline_reorg/sample_team_tests.yaml @tenstorrent/metalium-developers-infra
"""


def owners_of(repo: Repo, path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rules = mod.load_codeowners(str(repo.root / ".github/CODEOWNERS"))
    return mod.owners_for(path, rules)


def test_codeowners_last_match_wins(repo: Repo):
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    ind, teams = owners_of(repo, "tests/pipeline_reorg/sample_galaxy_tests.yaml")
    assert ind == {"mtairum", "uaydonat"}
    assert teams == set()


def test_codeowners_directory_rule_is_the_fallback(repo: Repo):
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    ind, _ = owners_of(repo, "tests/pipeline_reorg/sample_unit_tests.yaml")
    assert ind == {"roseli-TT", "tdowdallTT"}


def test_codeowners_team_only_path_yields_no_individuals(repo: Repo):
    """A team cannot be expanded with the workflow token, so this must fall back."""
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    ind, teams = owners_of(repo, "tests/pipeline_reorg/sample_team_tests.yaml")
    assert ind == set()
    assert teams == {"tenstorrent/metalium-developers-infra"}


# --- owner review gating -----------------------------------------------------


def run_reviews(repo: Repo, reviews: list, head_sha: str = "headsha"):
    """Invoke the gate with the reviews API stubbed by a local http server."""
    import http.server, json as _json, threading

    payload = _json.dumps(reviews).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    shim = repo.root / "shim.py"
    shim.write_text(
        "import runpy, sys, urllib.request\n"
        "_orig = urllib.request.Request\n"
        "def _patched(url, *a, **k):\n"
        f"    return _orig('http://127.0.0.1:{port}/reviews', *a, **k)\n"
        "urllib.request.Request = _patched\n"
        f"sys.argv = ['gate', '--base', {repo.base!r}, '--sku-config', '.github/sku_config.yaml',\n"
        f"            '--review-skus', {DEFAULT_REVIEW_SKUS!r}, '--non-matrix-files', {DEFAULT_NON_MATRIX!r},\n"
        f"            '--codeowners', '.github/CODEOWNERS', '--unsupported-files', 'sample_vllm_tests.yaml',\n"
        f"            '--matrix-dir', 'empty', '--repo', 'o/r', '--pr', '1', '--head-sha', {head_sha!r},\n"
        "            '--output', 'result.json']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    (repo.root / "empty").mkdir(exist_ok=True)
    (repo.root / "empty/placeholder.json").write_text("[]")
    result = subprocess.run([sys.executable, str(shim)], cwd=repo.root, capture_output=True, text=True)
    server.shutdown()
    return result.returncode, result.stdout, result.stderr


def test_approval_from_the_files_own_code_owner_satisfies_the_gate(repo: Repo):
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, stdout, stderr = run_reviews(
        repo, [{"state": "APPROVED", "commit_id": "headsha", "user": {"login": "mtairum"}}]
    )
    assert code == 0, stderr
    assert "approved by: @mtairum" in stdout


def test_approval_from_an_unrelated_owner_does_not_satisfy_the_gate(repo: Repo):
    """roseli-TT owns the directory, but this file has its own more specific owners."""
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, _, stderr = run_reviews(repo, [{"state": "APPROVED", "commit_id": "headsha", "user": {"login": "roseli-TT"}}])
    assert code == 1
    assert "@mtairum" in stderr


def test_approval_on_a_stale_commit_does_not_count(repo: Repo):
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    repo.write("tests/pipeline_reorg/sample_galaxy_tests.yaml", BASE_GALAXY_YAML.replace("test_alpha", "test_beta"))
    code, _, stderr = run_reviews(repo, [{"state": "APPROVED", "commit_id": "oldsha", "user": {"login": "mtairum"}}])
    assert code == 1
    assert "approving review on headsha" in stderr


def test_team_owned_path_falls_back_to_any_approval(repo: Repo):
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    repo.write("tests/pipeline_reorg/sample_team_tests.yaml", BASE_GALAXY_YAML)
    code, stdout, stderr = run_reviews(
        repo, [{"state": "APPROVED", "commit_id": "headsha", "user": {"login": "anyone"}}]
    )
    assert code == 0, stderr
    assert "falling back to the normal CODEOWNERS review requirement" in stdout


def test_team_owned_path_with_no_approval_still_blocks(repo: Repo):
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    repo.write("tests/pipeline_reorg/sample_team_tests.yaml", BASE_GALAXY_YAML)
    code, _, stderr = run_reviews(repo, [])
    assert code == 1
    assert "a code owner" in stderr


# --- unsupported yamls -------------------------------------------------------


def test_unsupported_yaml_is_blocked_not_skipped(repo: Repo):
    """vllm entries carry no cmd, so they route to review rather than passing."""
    repo.write(".github/CODEOWNERS", CODEOWNERS)
    repo.write("tests/pipeline_reorg/sample_vllm_tests.yaml", BASE_GALAXY_YAML.replace("wh_galaxy", "bh_p150"))
    code, _, stderr = run_reviews(repo, [])
    assert code == 1
    assert "sample_vllm_tests.yaml" in stderr


def test_unsupported_yaml_never_produces_a_run_leg(repo: Repo):
    repo.write("tests/pipeline_reorg/sample_vllm_tests.yaml", BASE_GALAXY_YAML.replace("wh_galaxy", "bh_p150"))
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
        "--unsupported-files",
        "sample_vllm_tests.yaml",
        "--output",
        "result.json",
    ]
    result = subprocess.run(argv, cwd=repo.root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads((repo.root / "result.json").read_text())
    assert payload["status"] == "blocked"
    assert payload["run_legs"] == []
    assert {leg["blocked_by"] for leg in payload["review_legs"]} == {"unsupported_yaml"}


# --- simulator timeout stripping ---------------------------------------------

TIMEOUT_YAML = """\
- name: ttnn group
  cmd: pytest --timeout 60 -xv tests/ttnn/unit_tests/test_a.py
  skus:
    wh_n150_civ2:
      timeout: 10
    sim_wh_n150:
      timeout: 30
  team: ttnn
  owner_id: U020
"""


def test_sim_legs_lose_the_pytest_per_test_timeout(repo: Repo):
    """ttsim is 10-50x slower than silicon, so a hardware-sized limit kills slow-but-correct tests."""
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n")
    repo.write("tests/pipeline_reorg/sample_timeout_tests.yaml", TIMEOUT_YAML)
    rows = [
        matrix_row("ttnn group [wh_n150_civ2]", "wh_n150_civ2", cmd="pytest --timeout 60 -xv tests/x.py"),
        matrix_row(
            "ttnn group [sim_wh_n150]",
            "sim_wh_n150",
            cmd="pytest --timeout 60 -xv tests/x.py",
            ttsim_lib="libttsim_wh.so",
        ),
    ]
    code, payload, stderr = run_with_matrices(repo, {"sample_timeout_tests": rows})
    assert code == 0, stderr

    by_sku = {r["sku"]: r for r in payload["legs"]}
    # Hardware keeps its limit untouched.
    assert "--timeout 60" in by_sku["wh_n150_civ2"]["cmd"]
    assert "gate_stripped_timeout" not in by_sku["wh_n150_civ2"]
    # The simulator leg loses it, and nothing else about the cmd changes.
    assert "--timeout" not in by_sku["sim_wh_n150"]["cmd"]
    assert by_sku["sim_wh_n150"]["cmd"] == "pytest -xv tests/x.py"
    assert by_sku["sim_wh_n150"]["gate_stripped_timeout"] is True


def test_sim_leg_without_a_timeout_is_untouched(repo: Repo):
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n")
    repo.write(
        "tests/pipeline_reorg/sample_timeout_tests.yaml",
        TIMEOUT_YAML.replace("pytest --timeout 60 -xv", "./build/test/alpha"),
    )
    rows = [
        matrix_row("ttnn group [wh_n150_civ2]", "wh_n150_civ2", cmd="./build/test/alpha"),
        matrix_row("ttnn group [sim_wh_n150]", "sim_wh_n150", cmd="./build/test/alpha", ttsim_lib="libttsim_wh.so"),
    ]
    code, payload, stderr = run_with_matrices(repo, {"sample_timeout_tests": rows})
    assert code == 0, stderr
    for row in payload["legs"]:
        assert row["cmd"] == "./build/test/alpha"
        assert "gate_stripped_timeout" not in row


def test_every_pytest_timeout_occurrence_is_stripped(repo: Repo):
    """The real cmds chain several pytest invocations, each with its own limit."""
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n")
    chained = "pytest --timeout 60 a.py && pytest --timeout 300 b.py"
    sim_only = TIMEOUT_YAML.replace("pytest --timeout 60 -xv tests/ttnn/unit_tests/test_a.py", chained).replace(
        "    wh_n150_civ2:\n      timeout: 10\n", ""
    )
    repo.write("tests/pipeline_reorg/sample_timeout_tests.yaml", sim_only)
    rows = [matrix_row("ttnn group [sim_wh_n150]", "sim_wh_n150", cmd=chained, ttsim_lib="libttsim_wh.so")]
    code, payload, stderr = run_with_matrices(repo, {"sample_timeout_tests": rows})
    assert code == 0, stderr
    assert payload["legs"][0]["cmd"] == "pytest a.py && pytest b.py"


# --- ttsim skip list ---------------------------------------------------------

SKIP_LIST = """\
wormhole_b0:
  - tests/ttnn/unit_tests/test_a.py::test_wh_only
  - tests/ttnn/unit_tests/test_b.py::test_wh_two[param_x]
blackhole:
  - tests/ttnn/unit_tests/test_c.py::test_bh_only
"""


def run_with_skips(repo: Repo, matrices: dict[str, list]):
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
            "",
            "--non-matrix-files",
            DEFAULT_NON_MATRIX,
            "--ttsim-skip-list",
            "tests/pipeline_reorg/ttsim-skip-list.yaml",
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


def sim_repo(repo: Repo):
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_wh_n150: {}\n  sim_bh_p150: {}\n")
    repo.write("tests/pipeline_reorg/ttsim-skip-list.yaml", SKIP_LIST)
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        "- name: g\n  cmd: pytest tests/x.py\n  skus:\n"
        "    sim_wh_n150:\n      timeout: 5\n    sim_bh_p150:\n      timeout: 5\n"
        "    wh_n150_civ2:\n      timeout: 5\n  team: ttnn\n  owner_id: U030\n",
    )
    return [
        matrix_row("g [sim_wh_n150]", "sim_wh_n150", cmd="pytest tests/x.py", ttsim_lib="libttsim_wh.so"),
        matrix_row("g [sim_bh_p150]", "sim_bh_p150", cmd="pytest tests/x.py", ttsim_lib="libttsim_bh.so"),
        matrix_row("g [wh_n150_civ2]", "wh_n150_civ2", cmd="pytest tests/x.py"),
    ]


def test_sim_legs_carry_their_archs_deselect_args(repo: Repo):
    rows = sim_repo(repo)
    code, payload, stderr = run_with_skips(repo, {"sample_sim_tests": rows})
    assert code == 0, stderr
    by_sku = {r["sku"]: r for r in payload["legs"]}

    wh = by_sku["sim_wh_n150"]["gate_pytest_deselect"]
    assert "--deselect=tests/ttnn/unit_tests/test_a.py::test_wh_only" in wh
    assert "--deselect=tests/ttnn/unit_tests/test_b.py::test_wh_two[param_x]" in wh
    # The wormhole leg must not pick up the blackhole list.
    assert "test_c.py" not in wh

    bh = by_sku["sim_bh_p150"]["gate_pytest_deselect"]
    assert bh == "--deselect=tests/ttnn/unit_tests/test_c.py::test_bh_only"

    # Hardware legs get no deselects at all.
    assert "gate_pytest_deselect" not in by_sku["wh_n150_civ2"]


def test_sim_sku_with_no_skip_list_arch_gets_no_deselects(repo: Repo):
    repo.write(".github/sku_config.yaml", SKU_CONFIG + "  sim_quasar_xl: {}\n")
    repo.write("tests/pipeline_reorg/ttsim-skip-list.yaml", SKIP_LIST)
    repo.write(
        "tests/pipeline_reorg/sample_sim_tests.yaml",
        "- name: q\n  cmd: pytest tests/x.py\n  skus:\n    sim_quasar_xl:\n      timeout: 5\n"
        "  team: llk\n  owner_id: U031\n",
    )
    rows = [matrix_row("q [sim_quasar_xl]", "sim_quasar_xl", cmd="pytest tests/x.py", ttsim_lib="libttsim_qsr.so")]
    code, payload, stderr = run_with_skips(repo, {"sample_sim_tests": rows})
    assert code == 0, stderr
    assert payload["legs"][0]["gate_pytest_deselect"] == ""


def test_missing_skip_list_is_not_fatal(repo: Repo):
    rows = sim_repo(repo)
    (repo.root / "tests/pipeline_reorg/ttsim-skip-list.yaml").unlink()
    code, payload, stderr = run_with_skips(repo, {"sample_sim_tests": rows})
    assert code == 0, stderr
    assert all(r.get("gate_pytest_deselect", "") == "" for r in payload["legs"] if r["sku"].startswith("sim_"))
