#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
from pathlib import Path


metal_home = Path(__file__).resolve().parent.parent.parent.parent
triage_home = metal_home / "tools" / "triage"
sys.path.insert(0, str(triage_home))

from parse_inspector_logs import get_log_directory

# All rank environment variables supported by _get_current_rank() in order of precedence.
_ALL_RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "PMIX_RANK", "TT_MESH_HOST_RANK")


def test_get_log_directory_prefers_rank_scoped_path_over_legacy_default(monkeypatch, tmp_path: Path):
    logs_root = tmp_path / "logs"
    default_dir = logs_root / "generated" / "inspector"
    default_dir.mkdir(parents=True)
    rank_dir = logs_root / "host-a_rank_3" / "generated" / "inspector"
    rank_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")

    assert get_log_directory() == str(rank_dir)


def test_get_log_directory_falls_back_to_legacy_default_when_no_rank_scoped_path(monkeypatch, tmp_path: Path):
    logs_root = tmp_path / "logs"
    default_dir = logs_root / "generated" / "inspector"
    default_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))

    assert get_log_directory() == str(default_dir)


def test_get_log_directory_falls_back_to_rank_scoped_path(monkeypatch, tmp_path: Path):
    logs_root = tmp_path / "logs"
    rank_dir = logs_root / "host-a_rank_3" / "generated" / "inspector"
    rank_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))

    assert get_log_directory() == str(rank_dir)


def test_get_log_directory_prefers_current_rank_when_multiple_rank_scoped_paths(monkeypatch, tmp_path: Path):
    logs_root = tmp_path / "logs"
    rank_one_dir = logs_root / "host-b_rank_1" / "generated" / "inspector"
    rank_zero_dir = logs_root / "host-a_rank_0" / "generated" / "inspector"
    rank_one_dir.mkdir(parents=True)
    rank_zero_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")

    assert get_log_directory() == str(rank_one_dir)


def test_get_log_directory_falls_back_to_rank_zero_without_rank_env(monkeypatch, tmp_path: Path):
    logs_root = tmp_path / "logs"
    rank_one_dir = logs_root / "host-b_rank_1" / "generated" / "inspector"
    rank_zero_dir = logs_root / "host-a_rank_0" / "generated" / "inspector"
    rank_one_dir.mkdir(parents=True)
    rank_zero_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.delenv("PMI_RANK", raising=False)
    monkeypatch.delenv("TT_MESH_HOST_RANK", raising=False)

    assert get_log_directory() == str(rank_zero_dir)


def test_get_log_directory_falls_back_to_tt_metal_home_when_logs_path_unset(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("TT_METAL_LOGS_PATH", raising=False)
    monkeypatch.setenv("TT_METAL_HOME", str(tmp_path))

    assert get_log_directory() == str(tmp_path / "generated" / "inspector")


@pytest.mark.parametrize("rank_var", ["PMI_RANK", "SLURM_PROCID", "PMIX_RANK", "TT_MESH_HOST_RANK"])
def test_get_log_directory_uses_rank_var_as_primary_source(monkeypatch, tmp_path: Path, rank_var: str):
    """PMI_RANK, SLURM_PROCID, PMIX_RANK, and TT_MESH_HOST_RANK all work as primary rank sources."""
    logs_root = tmp_path / "logs"
    rank_dir = logs_root / "node-a_rank_5" / "generated" / "inspector"
    rank_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))
    # Clear all higher-precedence vars so the target var is primary
    for var in _ALL_RANK_VARS:
        if var != rank_var:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(rank_var, "5")

    assert get_log_directory() == str(rank_dir)


def test_get_log_directory_ompi_rank_takes_precedence_over_pmi(monkeypatch, tmp_path: Path):
    """When both OMPI_COMM_WORLD_RANK and PMI_RANK are set, OMPI takes precedence."""
    logs_root = tmp_path / "logs"
    rank_ompi_dir = logs_root / "node-a_rank_1" / "generated" / "inspector"
    rank_pmi_dir = logs_root / "node-b_rank_7" / "generated" / "inspector"
    rank_ompi_dir.mkdir(parents=True)
    rank_pmi_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("PMI_RANK", "7")
    for var in ("SLURM_PROCID", "PMIX_RANK", "TT_MESH_HOST_RANK"):
        monkeypatch.delenv(var, raising=False)

    assert get_log_directory() == str(rank_ompi_dir)


def test_get_log_directory_pmi_rank_takes_precedence_over_slurm(monkeypatch, tmp_path: Path):
    """When both PMI_RANK and SLURM_PROCID are set (OMPI absent), PMI takes precedence."""
    logs_root = tmp_path / "logs"
    rank_pmi_dir = logs_root / "node-a_rank_2" / "generated" / "inspector"
    rank_slurm_dir = logs_root / "node-b_rank_9" / "generated" / "inspector"
    rank_pmi_dir.mkdir(parents=True)
    rank_slurm_dir.mkdir(parents=True)

    monkeypatch.setenv("TT_METAL_LOGS_PATH", str(logs_root))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.setenv("PMI_RANK", "2")
    monkeypatch.setenv("SLURM_PROCID", "9")
    for var in ("PMIX_RANK", "TT_MESH_HOST_RANK"):
        monkeypatch.delenv(var, raising=False)

    assert get_log_directory() == str(rank_pmi_dir)
