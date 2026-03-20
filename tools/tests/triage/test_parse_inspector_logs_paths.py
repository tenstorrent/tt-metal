#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


metal_home = Path(__file__).resolve().parent.parent.parent.parent
triage_home = metal_home / "tools" / "triage"
sys.path.insert(0, str(triage_home))

from parse_inspector_logs import get_log_directory


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
