#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Multiprocess MPI end-to-end test for rank-scoped inspector log resolution.
#
# Launches N MPI ranks (via mpirun), each of which writes a rank-tagged marker
# file under <logs_root>/<hostname>_rank_<N>/generated/inspector/.  After a
# barrier, rank 0 sets each rank's env vars and calls get_log_directory() to
# verify the correct directory is selected for every rank.
#
# Usage:
#   mpirun -np 4 python3 -m pytest --import-mode=importlib -v \
#       tools/tests/triage/test_multihost_rank_resolution_mpi.py
#
# Requirements:
#   - mpi4py (available in the CI venv)
#   - No Tenstorrent hardware needed (pure Python log-directory logic)

import json
import os
import socket
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# MPI bootstrap — must happen before any test collection
# ---------------------------------------------------------------------------

try:
    from mpi4py import MPI
except ImportError:
    pytest.skip("mpi4py not available — skipping MPI tests", allow_module_level=True)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
HOSTNAME = socket.gethostname()

# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------

_metal_home = Path(__file__).resolve().parent.parent.parent.parent
_triage_home = _metal_home / "tools" / "triage"
sys.path.insert(0, str(_triage_home))

from parse_inspector_logs import get_kernels, get_log_directory  # noqa: E402

# ---------------------------------------------------------------------------
# Shared temp directory — rank 0 creates it, broadcasts the path
# ---------------------------------------------------------------------------

_shared_tmpdir: str | None = None


def _get_shared_tmpdir() -> str:
    """Create a shared temp directory on rank 0 and broadcast to all ranks."""
    global _shared_tmpdir
    if _shared_tmpdir is not None:
        return _shared_tmpdir

    if RANK == 0:
        _shared_tmpdir = tempfile.mkdtemp(prefix="mpi_rank_resolution_")
    else:
        _shared_tmpdir = None
    _shared_tmpdir = COMM.bcast(_shared_tmpdir, root=0)
    return _shared_tmpdir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_RANK_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "SLURM_PROCID",
    "PMIX_RANK",
    "TT_MESH_HOST_RANK",
)


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


def _make_marker_file(logs_root: str, rank: int) -> Path:
    """Create <logs_root>/<hostname>_rank_<N>/generated/inspector/ with a
    kernels.yaml marker that encodes the rank."""
    inspector_dir = Path(logs_root) / f"{HOSTNAME}_rank_{rank}" / "generated" / "inspector"
    inspector_dir.mkdir(parents=True, exist_ok=True)
    kernels_path = inspector_dir / "kernels.yaml"
    kernels_path.write_text(_make_kernels_yaml(rank))
    return inspector_dir


def _clear_rank_env() -> dict[str, str | None]:
    """Remove all rank env vars and return their original values for restore."""
    saved = {}
    for var in _ALL_RANK_VARS:
        saved[var] = os.environ.pop(var, None)
    return saved


def _restore_rank_env(saved: dict[str, str | None]) -> None:
    """Restore rank env vars from a saved dict."""
    for var, val in saved.items():
        if val is not None:
            os.environ[var] = val
        else:
            os.environ.pop(var, None)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _cleanup_shared_tmpdir() -> None:
    """Remove the shared temp directory (rank 0 only, after barrier)."""
    COMM.Barrier()
    if RANK == 0:
        import shutil

        tmpdir = _get_shared_tmpdir()
        if tmpdir and Path(tmpdir).exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultihostRankResolutionMPI:
    """MPI multiprocess tests for rank-scoped inspector log directory resolution.

    Each MPI rank creates its own marker file, then rank 0 verifies that
    get_log_directory() resolves correctly for each rank's env vars.
    """

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self):
        """Setup shared tmpdir, create markers, barrier, then cleanup."""
        self.logs_root = _get_shared_tmpdir()
        # Each rank writes its own marker
        self.my_inspector_dir = _make_marker_file(self.logs_root, RANK)

        # Gather all ranks' inspector dirs to rank 0 for verification
        all_dirs = COMM.gather(str(self.my_inspector_dir), root=0)
        self.all_inspector_dirs = all_dirs  # Only non-None on rank 0

        # Barrier: all markers written before any rank reads
        COMM.Barrier()

        yield

        # Cleanup after all tests in this class
        # (actual tmpdir removal happens in session-scoped finalizer)

    def test_each_rank_wrote_marker(self):
        """Every rank's marker file exists and contains the correct rank tag."""
        assert self.my_inspector_dir.exists(), f"Rank {RANK}: inspector dir missing"
        kernels_yaml = self.my_inspector_dir / "kernels.yaml"
        assert kernels_yaml.exists(), f"Rank {RANK}: kernels.yaml missing"
        content = kernels_yaml.read_text()
        assert f"rank_{RANK}_matmul_kernel" in content, f"Rank {RANK}: marker file does not contain expected rank tag"

    def test_rank_resolution_via_ompi_var(self):
        """Rank 0 sets OMPI_COMM_WORLD_RANK for each rank and verifies resolution."""
        if RANK != 0:
            # Non-zero ranks just participate in the barrier
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            for target_rank in range(SIZE):
                os.environ["OMPI_COMM_WORLD_RANK"] = str(target_rank)
                log_dir = get_log_directory()
                expected = self.all_inspector_dirs[target_rank]
                assert log_dir == expected, f"OMPI rank {target_rank}: expected {expected}, got {log_dir}"
                # Also verify the data belongs to the correct rank
                kernels = get_kernels(log_dir)
                assert kernels, f"No kernels found in {log_dir}"
                names = [k.name for k in kernels.values()]
                assert any(
                    f"rank_{target_rank}_" in n for n in names
                ), f"Rank {target_rank}: kernel name mismatch in {names}"
        finally:
            _restore_rank_env(saved)
            # Remove TT_METAL_LOGS_PATH so it doesn't leak
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()

    def test_rank_resolution_via_pmi_var(self):
        """Rank 0 sets PMI_RANK for each rank and verifies resolution."""
        if RANK != 0:
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            for target_rank in range(SIZE):
                os.environ["PMI_RANK"] = str(target_rank)
                log_dir = get_log_directory()
                expected = self.all_inspector_dirs[target_rank]
                assert log_dir == expected, f"PMI rank {target_rank}: expected {expected}, got {log_dir}"
        finally:
            _restore_rank_env(saved)
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()

    def test_rank_resolution_via_tt_mesh_host_rank(self):
        """Rank 0 sets TT_MESH_HOST_RANK for each rank and verifies resolution."""
        if RANK != 0:
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            for target_rank in range(SIZE):
                os.environ["TT_MESH_HOST_RANK"] = str(target_rank)
                log_dir = get_log_directory()
                expected = self.all_inspector_dirs[target_rank]
                assert log_dir == expected, f"TT_MESH_HOST_RANK rank {target_rank}: expected {expected}, got {log_dir}"
        finally:
            _restore_rank_env(saved)
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()

    def test_no_rank_env_falls_back_to_rank_zero(self):
        """Without any rank env var, get_log_directory() falls back to rank 0."""
        if RANK != 0:
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            log_dir = get_log_directory()
            expected = self.all_inspector_dirs[0]
            assert log_dir == expected, f"No rank env: expected rank 0 dir {expected}, got {log_dir}"
        finally:
            _restore_rank_env(saved)
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()

    def test_ompi_precedence_over_pmi(self):
        """OMPI_COMM_WORLD_RANK takes precedence over PMI_RANK."""
        if RANK != 0:
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            # Set OMPI to rank 0, PMI to rank 1 — OMPI should win
            target_ompi = 0
            target_pmi = min(1, SIZE - 1)
            if target_ompi == target_pmi:
                # Only 1 rank — skip this test meaningfully
                COMM.Barrier()
                return
            os.environ["OMPI_COMM_WORLD_RANK"] = str(target_ompi)
            os.environ["PMI_RANK"] = str(target_pmi)
            log_dir = get_log_directory()
            expected = self.all_inspector_dirs[target_ompi]
            assert log_dir == expected, f"Precedence: expected OMPI rank {target_ompi} dir, got {log_dir}"
        finally:
            _restore_rank_env(saved)
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()

    def test_explicit_path_overrides_rank_env(self):
        """An explicit log_directory argument bypasses all rank resolution."""
        if RANK != 0:
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            os.environ["OMPI_COMM_WORLD_RANK"] = "0"
            # Explicitly request rank (SIZE-1)'s directory
            target = SIZE - 1
            explicit_dir = self.all_inspector_dirs[target]
            log_dir = get_log_directory(log_directory=explicit_dir)
            assert log_dir == explicit_dir, f"Explicit override: expected {explicit_dir}, got {log_dir}"
            # Verify data belongs to the right rank
            kernels = get_kernels(log_dir)
            names = [k.name for k in kernels.values()]
            assert any(
                f"rank_{target}_" in n for n in names
            ), f"Explicit override: kernel name mismatch for rank {target}"
        finally:
            _restore_rank_env(saved)
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()

    def test_invalid_rank_falls_back_to_rank_zero(self):
        """A non-existent rank value falls back to rank 0 directory."""
        if RANK != 0:
            COMM.Barrier()
            return

        saved = _clear_rank_env()
        try:
            os.environ["TT_METAL_LOGS_PATH"] = self.logs_root
            os.environ["OMPI_COMM_WORLD_RANK"] = "9999"
            log_dir = get_log_directory()
            expected = self.all_inspector_dirs[0]
            assert log_dir == expected, f"Invalid rank fallback: expected rank 0 dir {expected}, got {log_dir}"
        finally:
            _restore_rank_env(saved)
            os.environ.pop("TT_METAL_LOGS_PATH", None)

        COMM.Barrier()


# ---------------------------------------------------------------------------
# Session-scoped cleanup
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Clean up the shared tmpdir after all tests complete."""
    try:
        _cleanup_shared_tmpdir()
    except Exception:
        pass  # Best-effort cleanup
