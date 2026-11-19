# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for rankfile host inference in tt-run.

The rankfile grammar tt-run writes itself uses ``slot=`` (scripts/scaleout, generate_rank_bindings,
and the docs under models/demos/deepseek_v3_b1), while ``parse_rankfile`` accepts both spellings.
``extract_rankfile_hosts`` feeds ``normalize_rankfile_mpi_args``, which injects ``--host`` when a
rankfile is present, so a spelling it cannot read silently drops the host list.

Loaded by path so these tests run without a built ttnn extension.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TTRUN_PATH = Path(__file__).resolve().parents[3] / "ttnn" / "ttnn" / "distributed" / "ttrun.py"


def _load_ttrun():
    spec = importlib.util.spec_from_file_location("ttrun_rankfile_host_tests", TTRUN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ttrun = _load_ttrun()

RANKFILE_SLOT = "rank 0=host-a slot=0\nrank 1=host-b slot=0\nrank 2=host-a slot=1\n"
RANKFILE_SLOTS = "rank 0=host-a slots=0\nrank 1=host-b slots=0\nrank 2=host-a slots=1\n"


def test_extract_rankfile_hosts_reads_slots_spelling(tmp_path):
    rankfile = tmp_path / "rankfile"
    rankfile.write_text(RANKFILE_SLOTS)

    assert ttrun.extract_rankfile_hosts(str(rankfile)) == ["host-a", "host-b"]


def test_extract_rankfile_hosts_reads_slot_spelling(tmp_path):
    rankfile = tmp_path / "rankfile"
    rankfile.write_text(RANKFILE_SLOT)

    assert ttrun.extract_rankfile_hosts(str(rankfile)) == ["host-a", "host-b"]


def test_build_host_slots_and_extract_agree_on_slots_spelling(tmp_path):
    rankfile = tmp_path / "rankfile"
    rankfile.write_text(RANKFILE_SLOTS)

    host_str, total_ranks = ttrun.build_host_slots_from_rankfile(rankfile)

    assert host_str == "host-a:2,host-b:1"
    assert total_ranks == 3
    assert ttrun.extract_rankfile_hosts(str(rankfile)) == ["host-a", "host-b"]


def test_normalize_rankfile_mpi_args_infers_host_for_slots_spelling(tmp_path):
    rankfile = tmp_path / "rankfile"
    rankfile.write_text(RANKFILE_SLOTS)

    normalized = ttrun.normalize_rankfile_mpi_args(
        ["--rankfile", str(rankfile)], ttrun.RankfileSyntax.RANKFILE, cwd=tmp_path
    )

    assert "--host" in normalized
    assert normalized[normalized.index("--host") + 1] == "host-a,host-b"
