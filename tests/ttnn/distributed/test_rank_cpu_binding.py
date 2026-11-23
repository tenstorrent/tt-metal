# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""How tt-run's per-rank CPU binding splits a host's CPUs between the ranks that share it."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "ttnn" / "ttnn" / "distributed" / "rank_cpu_binding.py"
_spec = importlib.util.spec_from_file_location("rank_cpu_binding", _SCRIPT)
rcb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcb)


def _smt(num_cores: int):
    """Siblings on a host whose CPU c and c + num_cores are the two threads of core c."""

    def siblings(cpu: int) -> set[int]:
        core = cpu % num_cores
        return {core, core + num_cores}

    return siblings


def _no_smt(cpu: int) -> set[int]:
    return {cpu}


@pytest.mark.parametrize(
    "text, cpus",
    [("1,2", {1, 2}), ("0-3", {0, 1, 2, 3}), ("0-1,8", {0, 1, 8}), (" 4 , 6-7 ", {4, 6, 7}), ("", set())],
)
def test_parse_cpu_list(text, cpus):
    assert rcb.parse_cpu_list(text) == cpus


@pytest.mark.parametrize("text", ["a", "1-", "3-1", "1,,x"])
def test_parse_cpu_list_rejects_garbage(text):
    with pytest.raises(ValueError):
        rcb.parse_cpu_list(text)


def test_format_cpu_list_round_trips():
    assert rcb.format_cpu_list({0, 1, 2, 3, 8, 10, 11}) == "0-3,8,10-11"
    assert rcb.parse_cpu_list(rcb.format_cpu_list({0, 1, 2, 3, 8, 10, 11})) == {0, 1, 2, 3, 8, 10, 11}


def test_physical_cores_groups_hyperthreads():
    assert rcb.physical_cores(range(8), _smt(4)) == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_four_ranks_on_a_galaxy_host_get_disjoint_whole_cores():
    """32 cores with hyperthreads, 4 ranks: 8 whole cores each, no CPU in two ranks."""
    allowed = range(64)
    shares = [rcb.cpus_for_local_rank(allowed, r, 4, siblings=_smt(32)) for r in range(4)]
    assert shares[0] == set(range(0, 8)) | set(range(32, 40))
    assert shares[3] == set(range(24, 32)) | set(range(56, 64))
    assert set().union(*shares) == set(allowed)
    assert sum(len(s) for s in shares) == 64


def test_excluded_cpus_leave_their_whole_core_out_of_every_share():
    allowed = range(64)
    shares = [rcb.cpus_for_local_rank(allowed, r, 4, exclude={1, 2}, siblings=_smt(32)) for r in range(4)]
    used = set().union(*shares)
    assert not used & {1, 2, 33, 34}
    assert sum(len(s) for s in shares) == len(used)  # disjoint
    # 30 cores over 4 ranks: the first two ranks take 8, the last two 7.
    assert [len(s) // 2 for s in shares] == [8, 8, 7, 7]


def test_respects_the_process_cpu_mask():
    """Only CPUs the process may already run on are split."""
    allowed = set(range(8, 24))
    shares = [rcb.cpus_for_local_rank(allowed, r, 2, siblings=_no_smt) for r in range(2)]
    assert shares == [set(range(8, 16)), set(range(16, 24))]


@pytest.mark.parametrize(
    "local_rank, local_size", [(0, 1), (0, 0), (2, 2), (-1, 2)], ids=["one_rank", "no_ranks", "rank_past_end", "neg"]
)
def test_no_split_without_two_valid_local_ranks(local_rank, local_size):
    assert rcb.cpus_for_local_rank(range(16), local_rank, local_size, siblings=_no_smt) is None


def test_no_split_when_ranks_would_get_fewer_than_two_cores():
    """A mock or CPU-only run with more ranks than cores stays unpinned."""
    assert rcb.cpus_for_local_rank(range(8), 0, 5, siblings=_no_smt) is None
    assert rcb.cpus_for_local_rank(range(8), 0, 4, siblings=_no_smt) == {0, 1}
