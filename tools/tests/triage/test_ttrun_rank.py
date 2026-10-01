# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# tt-run must export TT_RUN_RANK per rank, and triage must derive its per-rank identity from
# it (falling back to OMPI_COMM_WORLD_RANK under raw mpirun). These need no hardware - if rank
# resolution breaks, every rank silently collapses onto rank 0's inspector data and output path.

import importlib.util
import os
import sys

import pytest

metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_home = os.path.join(metal_home, "tools", "triage")

sys.path.insert(0, triage_home)

from utils import resolve_mpi_rank


def _load_ttrun():
    # Load by file path: importing ttnn.distributed.ttrun would drag in the ttnn C extension.
    path = os.path.join(metal_home, "ttnn", "ttnn", "distributed", "ttrun.py")
    spec = importlib.util.spec_from_file_location("ttrun_for_triage_rank_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ttrun = _load_ttrun()


@pytest.fixture(autouse=True)
def clean_rank_env(monkeypatch):
    """A rank var inherited from the caller must not decide the outcome of any test here."""
    for var in ("TT_RUN_RANK", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)


def _config(tmp_path, num_ranks):
    mesh_graph = tmp_path / "mesh_graph.yaml"
    mesh_graph.touch()
    return ttrun.TTRunConfig(
        rank_bindings=[ttrun.RankBinding(rank=rank, mesh_id=0) for rank in range(num_ranks)],
        mesh_graph_desc_path=mesh_graph,
    )


def test_ttrun_exports_rank_per_rank(tmp_path):
    """Every rank gets TT_RUN_RANK equal to its own global MPI rank, not the launcher's."""
    config = _config(tmp_path, 4)

    for binding in config.rank_bindings:
        env = ttrun.get_rank_environment(binding, config)
        assert env["TT_RUN_RANK"] == str(binding.rank)


def test_ttrun_rank_reaches_mpirun_args(tmp_path):
    """TT_RUN_RANK survives into the -x list mpirun actually receives."""
    config = _config(tmp_path, 4)

    for binding in config.rank_bindings:
        args = ttrun.build_rank_environment_args(binding, config)
        assert f"TT_RUN_RANK={binding.rank}" in args


def test_stale_parent_rank_does_not_leak(tmp_path, monkeypatch):
    """A TT_RUN_RANK left over in the launching shell must not override the per-rank value."""
    monkeypatch.setenv("TT_RUN_RANK", "99")
    config = _config(tmp_path, 2)

    for binding in config.rank_bindings:
        env = ttrun.get_rank_environment(binding, config)
        assert env["TT_RUN_RANK"] == str(binding.rank)


def test_tt_run_rank_wins_over_launcher_rank(monkeypatch):
    """TT_RUN_RANK is the override: when both are set, it decides."""
    monkeypatch.setenv("TT_RUN_RANK", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "7")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    assert resolve_mpi_rank() == 2


def test_falls_back_to_launcher_rank(monkeypatch):
    """Raw mpirun sets only the OMPI vars - triage must still find its rank."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "7")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    assert resolve_mpi_rank() == 7


def test_single_rank_world_has_no_rank(monkeypatch):
    """Metal only rank-suffixes Inspector output when the world holds more than one rank."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    assert resolve_mpi_rank() is None


def test_no_rank_env_at_all():
    """A plain non-MPI run stays rank-less."""
    assert resolve_mpi_rank() is None


def test_unparseable_rank_is_ignored(monkeypatch):
    """A malformed rank must degrade to rank-less, not raise."""
    monkeypatch.setenv("TT_RUN_RANK", "not-a-number")
    assert resolve_mpi_rank() is None
