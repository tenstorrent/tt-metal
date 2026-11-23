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


def _one_node(cpu: int) -> int:
    return 0


def _two_nodes(cpus_per_node: int):
    return lambda cpu: cpu // cpus_per_node


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
    shares = [rcb.cpus_for_local_rank(allowed, r, 4, siblings=_smt(32), node_of=_one_node) for r in range(4)]
    assert shares[0] == set(range(0, 8)) | set(range(32, 40))
    assert shares[3] == set(range(24, 32)) | set(range(56, 64))
    assert set().union(*shares) == set(allowed)
    assert sum(len(s) for s in shares) == 64


def test_excluded_cpus_leave_their_whole_core_out_of_every_share():
    allowed = range(64)
    shares = [rcb.cpus_for_local_rank(allowed, r, 4, exclude={1, 2}, siblings=_smt(32), node_of=_one_node) for r in range(4)]
    used = set().union(*shares)
    assert not used & {1, 2, 33, 34}
    assert sum(len(s) for s in shares) == len(used)  # disjoint
    # 30 cores over 4 ranks: the first two ranks take 8, the last two 7.
    assert [len(s) // 2 for s in shares] == [8, 8, 7, 7]


def test_respects_the_process_cpu_mask():
    """Only CPUs the process may already run on are split."""
    allowed = set(range(8, 24))
    shares = [rcb.cpus_for_local_rank(allowed, r, 2, siblings=_no_smt, node_of=_one_node) for r in range(2)]
    assert shares == [set(range(8, 16)), set(range(16, 24))]


@pytest.mark.parametrize(
    "local_rank, local_size", [(0, 1), (0, 0), (2, 2), (-1, 2)], ids=["one_rank", "no_ranks", "rank_past_end", "neg"]
)
def test_no_split_without_two_valid_local_ranks(local_rank, local_size):
    assert rcb.cpus_for_local_rank(range(16), local_rank, local_size, siblings=_no_smt, node_of=_one_node) is None


def test_no_split_when_ranks_would_get_fewer_than_two_cores():
    """A mock or CPU-only run with more ranks than cores stays unpinned."""
    assert rcb.cpus_for_local_rank(range(8), 0, 5, siblings=_no_smt, node_of=_one_node) is None
    assert rcb.cpus_for_local_rank(range(8), 0, 4, siblings=_no_smt, node_of=_one_node) == {0, 1}


def test_every_rank_gets_cores_on_every_numa_node():
    """Two sockets of 16 cores, 4 ranks: 4 cores per rank on each socket, so devices on either stay local."""
    shares = [rcb.cpus_for_local_rank(range(32), r, 4, siblings=_no_smt, node_of=_two_nodes(16)) for r in range(4)]
    assert shares[0] == set(range(0, 4)) | set(range(16, 20))
    assert shares[3] == set(range(12, 16)) | set(range(28, 32))
    assert set().union(*shares) == set(range(32))


def test_leftover_cores_rotate_across_numa_nodes():
    """Two nodes of 5 cores over 4 ranks: each node has one leftover, given to different ranks."""
    shares = [rcb.cpus_for_local_rank(range(10), r, 4, siblings=_no_smt, node_of=_two_nodes(5)) for r in range(4)]
    assert [len(s) for s in shares] == [3, 3, 2, 2]
    assert sum(len(s) for s in shares) == len(set().union(*shares)) == 10


def test_usable_cores_drop_excluded_cpus_and_their_siblings():
    assert rcb.usable_cores(range(8), {1}, _smt(4)) == [[0, 4], [2, 6], [3, 7]]


@pytest.fixture
def run_main(monkeypatch):
    """Run main() with a fake CPU mask and exec, returning (pinned CPUs or None, exec'd argv)."""

    def run(argv, allowed, local_rank=None, local_size=None):
        for name, value in ((rcb.LOCAL_RANK_ENV, local_rank), (rcb.LOCAL_SIZE_ENV, local_size)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, str(value))
        pinned, execd = [], []
        monkeypatch.setattr(rcb.os, "sched_getaffinity", lambda pid: set(allowed))
        monkeypatch.setattr(rcb.os, "sched_setaffinity", lambda pid, cpus: pinned.append(set(cpus)))
        monkeypatch.setattr(rcb.os, "execvp", lambda file, args: execd.append(list(args)))
        monkeypatch.setattr(rcb, "thread_siblings", _no_smt)
        monkeypatch.setattr(rcb, "numa_node", _one_node)
        rcb.main(argv)
        return (pinned[-1] if pinned else None), execd[-1]

    return run


def test_main_pins_its_share_and_execs_the_program(run_main):
    pinned, execd = run_main(["--", "prog", "-x"], range(16), local_rank=1, local_size=2)
    assert pinned == set(range(8, 16))
    assert execd == ["prog", "-x"]


def test_main_applies_exclude_without_a_split(run_main):
    """One rank on the host still stays off excluded CPUs."""
    pinned, _ = run_main(["--exclude", "1,2", "--", "prog"], range(8), local_rank=0, local_size=1)
    assert pinned == {0, 3, 4, 5, 6, 7}


def test_main_merges_repeated_excludes(run_main):
    pinned, _ = run_main(["--exclude", "1", "--exclude", "2", "--", "prog"], range(8))
    assert pinned == {0, 3, 4, 5, 6, 7}


def test_main_leaves_the_mask_alone_without_mpi_or_exclude(run_main):
    pinned, execd = run_main(["--", "prog"], range(8))
    assert pinned is None and execd == ["prog"]


def test_main_exits_127_when_the_program_is_missing(monkeypatch):
    monkeypatch.delenv(rcb.LOCAL_SIZE_ENV, raising=False)
    with pytest.raises(SystemExit) as e:
        rcb.main(["--", "/nonexistent/tt-run-test-program"])
    assert e.value.code == rcb.EXEC_FAILED_EXIT_CODE
