#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test for multihost rank-scoped inspector log resolution.
#
# Unlike test_parse_inspector_logs_paths.py (which tests get_log_directory()
# path math in isolation), these tests build a realistic fake directory tree
# that mimics a 3-host/3-rank tt-run job, then drive get_log_directory() +
# get_kernels() together to verify:
#   1. The rank resolution selects the correct host/rank subdirectory
#   2. The data read from that directory actually belongs to the correct rank
#      (not just the right path)
#   3. All supported rank environment variables (OMPI / PMI / SLURM / PMIX /
#      TT_MESH_HOST_RANK) select the right rank
#   4. Fallback and explicit-path-override behaviours work end-to-end

import pytest
import sys
import textwrap
from pathlib import Path

metal_home = Path(__file__).resolve().parent.parent.parent.parent
triage_home = metal_home / "tools" / "triage"
sys.path.insert(0, str(triage_home))

from parse_inspector_logs import get_kernels, get_log_directory

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_ALL_RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "PMIX_RANK", "TT_MESH_HOST_RANK")

# Three simulated hosts matching the <hostname>_rank_<N> naming convention.
_HOSTS = ["node-0", "node-1", "node-2"]


def _make_kernels_yaml(rank: int) -> str:
    """Minimal kernels.yaml with a name that encodes the rank for assertions."""
    return textwrap.dedent(
        f"""\
        - kernel:
            watcher_kernel_id: {rank}
            name: rank_{rank}_matmul_kernel
            path: /fake/path/rank_{rank}/kernel.cpp
            source: void kernel_main() {{}}
            program_id: {rank}
    """
    )


def _make_inspector_dir(logs_root: Path, hostname: str, rank: int) -> Path:
    """Create <logs_root>/<hostname>_rank_<N>/generated/inspector/ with a kernels.yaml."""
    inspector_dir = logs_root / f"{hostname}_rank_{rank}" / "generated" / "inspector"
    inspector_dir.mkdir(parents=True)
    (inspector_dir / "kernels.yaml").write_text(_make_kernels_yaml(rank))
    return inspector_dir


@pytest.fixture()
def multihost_tree(tmp_path: Path) -> dict:
    """
    Create a fake 3-host/3-rank inspector log tree:

        <tmp>/logs/
            node-0_rank_0/generated/inspector/kernels.yaml   (kernel name: rank_0_matmul_kernel)
            node-1_rank_1/generated/inspector/kernels.yaml   (kernel name: rank_1_matmul_kernel)
            node-2_rank_2/generated/inspector/kernels.yaml   (kernel name: rank_2_matmul_kernel)

    Returns a dict with:
        logs_root  – Path to the logs root
        dirs       – {rank: Path} mapping to each inspector dir
    """
    logs_root = tmp_path / "logs"
    dirs = {}
    for rank, host in enumerate(_HOSTS):
        dirs[rank] = _make_inspector_dir(logs_root, host, rank)
    return {"logs_root": logs_root, "dirs": dirs}


def _clear_rank_vars(monkeypatch) -> None:
    for var in _ALL_RANK_VARS:
        monkeypatch.delenv(var, raising=False)


def _assert_rank_data(log_dir: str, expected_rank: int) -> None:
    """Drive get_kernels() and verify the kernel name encodes the expected rank."""
    kernels = get_kernels(log_dir)
    assert kernels, f"No kernels found in {log_dir}"
    names = [k.name for k in kernels.values()]
    assert any(
        f"rank_{expected_rank}_" in name for name in names
    ), f"Expected kernel name containing 'rank_{expected_rank}_', got: {names}"
    # Verify no other rank's kernels bled in
    for other_rank in range(3):
        if other_rank != expected_rank:
            assert not any(
                f"rank_{other_rank}_" in name for name in names
            ), f"rank_{other_rank} kernel unexpectedly present in rank_{expected_rank} dir: {names}"


# ---------------------------------------------------------------------------
# Core rank selection: each supported env var resolves the right rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank_var,rank",
    [
        ("OMPI_COMM_WORLD_RANK", 0),
        ("OMPI_COMM_WORLD_RANK", 1),
        ("OMPI_COMM_WORLD_RANK", 2),
        ("PMI_RANK", 1),
        ("SLURM_PROCID", 2),
        ("PMIX_RANK", 0),
        ("TT_MESH_HOST_RANK", 1),
    ],
)
def test_rank_var_selects_correct_data(monkeypatch, multihost_tree, rank_var: str, rank: int):
    """Each supported rank env var resolves get_log_directory() to the right rank's data."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv(rank_var, str(rank))

    log_dir = get_log_directory()

    assert log_dir == str(
        multihost_tree["dirs"][rank]
    ), f"{rank_var}={rank}: expected dir for rank {rank}, got {log_dir}"
    _assert_rank_data(log_dir, rank)


# ---------------------------------------------------------------------------
# Precedence: higher-priority var wins when multiple are set
# ---------------------------------------------------------------------------


def test_ompi_beats_pmi_end_to_end(monkeypatch, multihost_tree):
    """OMPI_COMM_WORLD_RANK takes precedence over PMI_RANK; data confirms correct rank selected."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("PMI_RANK", "2")  # should be ignored

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][0])
    _assert_rank_data(log_dir, expected_rank=0)


def test_pmi_beats_slurm_end_to_end(monkeypatch, multihost_tree):
    """PMI_RANK takes precedence over SLURM_PROCID; data confirms correct rank selected."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("PMI_RANK", "1")
    monkeypatch.setenv("SLURM_PROCID", "2")  # should be ignored

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][1])
    _assert_rank_data(log_dir, expected_rank=1)


def test_slurm_beats_pmix_end_to_end(monkeypatch, multihost_tree):
    """SLURM_PROCID takes precedence over PMIX_RANK."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("PMIX_RANK", "0")  # should be ignored

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][2])
    _assert_rank_data(log_dir, expected_rank=2)


# ---------------------------------------------------------------------------
# Fallback: no rank env → rank 0
# ---------------------------------------------------------------------------


def test_no_rank_env_falls_back_to_rank_zero_data(monkeypatch, multihost_tree):
    """Without any rank env var, triage falls back to rank 0 and reads rank 0 data."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][0])
    _assert_rank_data(log_dir, expected_rank=0)


# ---------------------------------------------------------------------------
# Explicit path override: bypasses all rank resolution
# ---------------------------------------------------------------------------


def test_explicit_path_override_bypasses_rank_resolution(monkeypatch, multihost_tree):
    """An explicit log_directory argument overrides ALL rank env var logic."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")  # would select rank 1 otherwise

    explicit = str(multihost_tree["dirs"][2])
    log_dir = get_log_directory(log_directory=explicit)

    assert log_dir == explicit
    _assert_rank_data(log_dir, expected_rank=2)


# ---------------------------------------------------------------------------
# Invalid rank value: skipped, falls back to next var / rank 0
# ---------------------------------------------------------------------------


def test_invalid_rank_value_falls_through_to_next_var(monkeypatch, multihost_tree):
    """A non-integer rank value in OMPI var is skipped; PMI_RANK is used instead."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "not_a_number")
    monkeypatch.setenv("PMI_RANK", "2")

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][2])
    _assert_rank_data(log_dir, expected_rank=2)


def test_negative_rank_value_falls_through_to_next_var(monkeypatch, multihost_tree):
    """A negative rank value is skipped; the next var is consulted."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "-1")
    monkeypatch.setenv("PMI_RANK", "1")

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][1])
    _assert_rank_data(log_dir, expected_rank=1)


# ---------------------------------------------------------------------------
# Rank env points to non-existent rank: falls back to rank 0
# ---------------------------------------------------------------------------


def test_rank_env_for_missing_rank_falls_back_to_rank_zero(monkeypatch, multihost_tree):
    """If the rank env points to a rank with no inspector dir, fall back to rank 0."""
    _clear_rank_vars(monkeypatch)
    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(multihost_tree["logs_root"]))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "99")  # rank 99 dir doesn't exist

    log_dir = get_log_directory()
    assert log_dir == str(multihost_tree["dirs"][0])
    _assert_rank_data(log_dir, expected_rank=0)
