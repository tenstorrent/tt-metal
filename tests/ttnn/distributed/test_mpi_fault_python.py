# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for mpi_fault.py — Python ULFM wrapper.

Tests install_ulfm_handler(), ulfm_guard() context manager,
MPIRankFailureError exception, and graceful degradation without mpi4py.

These run without a real MPI runtime by mocking mpi4py where needed.
"""

import importlib.util
import pathlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# mpi_fault.py lives at ttnn/ttnn/distributed/mpi_fault.py in the source tree
# but is NOT installed into the ttnn site-package.  Load it by absolute path
# so the tests work regardless of which ttnn package is active.
_mpi_fault_path = pathlib.Path(__file__).resolve().parents[3] / "ttnn" / "ttnn" / "distributed" / "mpi_fault.py"


def _load_mpi_fault_from_source() -> object:
    """Load mpi_fault.py directly from the source tree.

    This bypasses the installed ttnn package (which does not include
    mpi_fault.py) and executes the source file in a fresh module object
    registered under 'ttnn.distributed.mpi_fault'.  Any import statements
    inside mpi_fault.py still go through builtins.__import__, so patches
    (e.g. mocking mpi4py) applied at call-time are honoured.
    """
    spec = importlib.util.spec_from_file_location("ttnn.distributed.mpi_fault", _mpi_fault_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ttnn.distributed.mpi_fault"] = mod
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# Utility: fresh import of mpi_fault with mocked mpi4py
# =====================================================================


@contextmanager
def _fresh_mpi_fault_import(mpi_available=True, ulfm_available=True):
    """Import mpi_fault.py with a mocked mpi4py environment.

    Args:
        mpi_available: If True, mock mpi4py.MPI is importable.
        ulfm_available: If True, mock MPI has ULFM error constants.

    Yields:
        The freshly imported mpi_fault module.
    """
    # Remove cached module to get a fresh import
    mod_name = "ttnn.distributed.mpi_fault"
    saved = sys.modules.pop(mod_name, None)
    saved_mpi = sys.modules.pop("mpi4py", None)
    saved_mpi_MPI = sys.modules.pop("mpi4py.MPI", None)

    try:
        if mpi_available:
            mock_MPI = MagicMock()
            mock_MPI.ERRORS_RETURN = "ERRORS_RETURN_SENTINEL"

            if ulfm_available:
                mock_MPI.ERR_PROC_FAILED = 54
                mock_MPI.ERR_PROC_FAILED_PENDING = 55
                mock_MPI.ERR_REVOKED = 78
            else:
                # Remove ULFM attributes to simulate non-ULFM build
                for attr in ("ERR_PROC_FAILED", "ERR_PROC_FAILED_PENDING", "ERR_REVOKED"):
                    if hasattr(mock_MPI, attr):
                        delattr(mock_MPI, attr)

            mock_MPI.Exception = type(
                "MPIException",
                (Exception,),
                {
                    "__init__": lambda self, msg="", error_code=0: (
                        setattr(self, "_error_code", error_code) or Exception.__init__(self, msg)
                    ),
                    "Get_error_code": lambda self: self._error_code,
                },
            )

            mock_mpi4py = MagicMock()
            mock_mpi4py.MPI = mock_MPI
            sys.modules["mpi4py"] = mock_mpi4py
            sys.modules["mpi4py.MPI"] = mock_MPI
        else:
            # Simulate mpi4py not installed
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "mpi4py" or name.startswith("mpi4py."):
                    raise ImportError("No module named 'mpi4py'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                mpi_fault_mod = _load_mpi_fault_from_source()

                yield mpi_fault_mod
                return

        mpi_fault_mod = _load_mpi_fault_from_source()

        yield mpi_fault_mod
    finally:
        # Restore original module state
        sys.modules.pop(mod_name, None)
        if saved is not None:
            sys.modules[mod_name] = saved
        # Restore mpi4py
        sys.modules.pop("mpi4py", None)
        sys.modules.pop("mpi4py.MPI", None)
        if saved_mpi is not None:
            sys.modules["mpi4py"] = saved_mpi
        if saved_mpi_MPI is not None:
            sys.modules["mpi4py.MPI"] = saved_mpi_MPI


# =====================================================================
# MPIRankFailureError exception
# =====================================================================


class TestMPIRankFailureError:
    """Test the MPIRankFailureError exception class."""

    def test_basic_construction(self):
        with _fresh_mpi_fault_import() as mf:
            err = mf.MPIRankFailureError(rank=3, error_code=54, operation="Allreduce")
            assert err.rank == 3
            assert err.error_code == 54
            assert err.operation == "Allreduce"
            assert err.failed_ranks == []

    def test_with_failed_ranks(self):
        with _fresh_mpi_fault_import() as mf:
            err = mf.MPIRankFailureError(rank=0, error_code=78, operation="Barrier", failed_ranks=[1, 3, 5])
            assert err.failed_ranks == [1, 3, 5]

    def test_str_includes_rank_and_operation(self):
        with _fresh_mpi_fault_import() as mf:
            err = mf.MPIRankFailureError(rank=2, error_code=54, operation="Bcast")
            msg = str(err)
            assert "2" in msg  # rank
            assert "Bcast" in msg  # operation
            assert "54" in msg  # error code

    def test_is_runtime_error(self):
        with _fresh_mpi_fault_import() as mf:
            err = mf.MPIRankFailureError(rank=0, error_code=54, operation="test")
            assert isinstance(err, RuntimeError)

    def test_failed_ranks_default_empty(self):
        with _fresh_mpi_fault_import() as mf:
            err = mf.MPIRankFailureError(rank=0, error_code=54, operation="test", failed_ranks=None)
            assert err.failed_ranks == []


# =====================================================================
# install_ulfm_handler
# =====================================================================


class TestInstallUlfmHandler:
    """Test install_ulfm_handler() behavior."""

    def test_installs_errors_return_on_comm(self):
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            mf.install_ulfm_handler(comm)
            comm.Set_errhandler.assert_called_once()

    def test_default_comm_world(self):
        """When no comm is passed, should use MPI.COMM_WORLD."""
        with _fresh_mpi_fault_import() as mf:
            # With mocked MPI available, default to COMM_WORLD
            mf.install_ulfm_handler()
            # COMM_WORLD is a MagicMock attribute, verify Set_errhandler was called
            mf.MPI.COMM_WORLD.Set_errhandler.assert_called_once()

    def test_no_mpi_available_is_noop(self):
        """Without mpi4py, install_ulfm_handler should be a silent no-op."""
        with _fresh_mpi_fault_import(mpi_available=False) as mf:
            assert mf._MPI_AVAILABLE is False
            # Should not raise
            mf.install_ulfm_handler()


# =====================================================================
# ulfm_guard — fast_fail mode
# =====================================================================


class TestUlfmGuardFastFail:
    """Test ulfm_guard() in fast_fail mode (default)."""

    def test_success_path_passes_through(self):
        """Normal operation: body executes, no exception."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            executed = False
            with mf.ulfm_guard(comm, "test_op"):
                executed = True
            assert executed

    def test_ulfm_error_calls_sys_exit_70(self):
        """ULFM error in fast_fail mode should call sys.exit(70)."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0
            comm.Get_size.return_value = 4
            # Don't have Revoke or Get_failed
            comm.Revoke = MagicMock()

            # Create an MPI.Exception with a ULFM error code
            mpi_exc = mf.MPI.Exception("proc failed", error_code=54)

            with pytest.raises(SystemExit) as exc_info:
                with mf.ulfm_guard(comm, "Allreduce"):
                    raise mpi_exc

            assert exc_info.value.code == 70

    def test_non_ulfm_error_propagates(self):
        """Non-ULFM MPI errors should propagate normally."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            # Error code 999 is not a ULFM code
            mpi_exc = mf.MPI.Exception("generic error", error_code=999)

            with pytest.raises(mf.MPI.Exception):
                with mf.ulfm_guard(comm, "test"):
                    raise mpi_exc

    def test_fast_fail_diagnostic_to_stderr(self, capsys):
        """fast_fail should print structured diagnostic to stderr."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 2
            comm.Get_size.return_value = 8
            comm.Revoke = MagicMock()
            # No Get_failed method
            del comm.Get_failed

            mpi_exc = mf.MPI.Exception("proc failed", error_code=54)

            with pytest.raises(SystemExit):
                with mf.ulfm_guard(comm, "Barrier"):
                    raise mpi_exc

            captured = capsys.readouterr()
            # Diagnostic should mention rank, operation, and error info
            assert "Rank 2" in captured.err or "rank 2" in captured.err.lower()
            assert "Barrier" in captured.err


# =====================================================================
# ulfm_guard — fault_tolerant mode
# =====================================================================


class TestUlfmGuardFaultTolerant:
    """Test ulfm_guard() in fault_tolerant mode."""

    def test_raises_mpi_rank_failure_error(self):
        """fault_tolerant mode should raise MPIRankFailureError."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0
            # No Get_failed
            del comm.Get_failed

            mpi_exc = mf.MPI.Exception("proc failed", error_code=54)

            with pytest.raises(mf.MPIRankFailureError) as exc_info:
                with mf.ulfm_guard(comm, "Scatter", policy="fault_tolerant"):
                    raise mpi_exc

            err = exc_info.value
            assert err.rank == 0
            assert err.error_code == 54
            assert err.operation == "Scatter"

    def test_fault_tolerant_with_failed_ranks(self):
        """If Get_failed is available, exception includes failed rank list."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0

            # Mock Get_failed returning a group with 2 failed ranks
            failed_group = MagicMock()
            failed_group.Get_size.return_value = 2
            comm.Get_failed.return_value = failed_group

            mpi_exc = mf.MPI.Exception("proc failed", error_code=54)

            with pytest.raises(mf.MPIRankFailureError) as exc_info:
                with mf.ulfm_guard(comm, "Allreduce", policy="fault_tolerant"):
                    raise mpi_exc

            # failed_ranks should be [0, 1] (range of failed_group size)
            assert exc_info.value.failed_ranks == [0, 1]

    def test_success_path_in_fault_tolerant(self):
        """No exception means no error — guard passes through."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            result = None
            with mf.ulfm_guard(comm, "test", policy="fault_tolerant"):
                result = 42
            assert result == 42


# =====================================================================
# ulfm_guard — no mpi4py installed
# =====================================================================


class TestUlfmGuardNoMPI:
    """Test ulfm_guard behavior when mpi4py is not available."""

    def test_guard_is_passthrough_without_mpi(self):
        """Without mpi4py, ulfm_guard should be a transparent passthrough."""
        with _fresh_mpi_fault_import(mpi_available=False) as mf:
            assert mf._MPI_AVAILABLE is False
            executed = False
            with mf.ulfm_guard(None, "test"):
                executed = True
            assert executed

    def test_exceptions_propagate_without_mpi(self):
        """Without mpi4py, non-MPI exceptions should propagate normally."""
        with _fresh_mpi_fault_import(mpi_available=False) as mf:
            with pytest.raises(ValueError):
                with mf.ulfm_guard(None, "test"):
                    raise ValueError("app error")


# =====================================================================
# ulfm_guard — no ULFM support (mpi4py present but no ULFM constants)
# =====================================================================


class TestUlfmGuardNoULFM:
    """Test ulfm_guard when mpi4py is present but without ULFM extensions."""

    def test_non_ulfm_mpi_error_propagates(self):
        """MPI exceptions should propagate when ULFM error codes are unknown."""
        with _fresh_mpi_fault_import(ulfm_available=False) as mf:
            comm = MagicMock()
            # _ULFM_ERROR_CODES should be empty
            assert len(mf._ULFM_ERROR_CODES) == 0

            mpi_exc = mf.MPI.Exception("some error", error_code=42)
            with pytest.raises(mf.MPI.Exception):
                with mf.ulfm_guard(comm, "test"):
                    raise mpi_exc


# =====================================================================
# _try_get_failed_ranks edge cases
# =====================================================================


class TestTryGetFailedRanks:
    """Test the _try_get_failed_ranks internal helper."""

    def test_no_get_failed_method(self):
        """Comm without Get_failed should return empty list."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock(spec=[])  # No methods at all
            result = mf._try_get_failed_ranks(comm)
            assert result == []

    def test_get_failed_raises(self):
        """If Get_failed raises, should return empty list (best-effort)."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_failed.side_effect = RuntimeError("comm broken")
            result = mf._try_get_failed_ranks(comm)
            assert result == []

    def test_get_failed_returns_group(self):
        """Normal case: returns list of rank indices from the failed group."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            failed_group = MagicMock()
            failed_group.Get_size.return_value = 3
            comm.Get_failed.return_value = failed_group
            result = mf._try_get_failed_ranks(comm)
            assert result == [0, 1, 2]


# =====================================================================
# _ulfm_fast_fail edge cases
# =====================================================================


class TestUlfmFastFail:
    """Test the _ulfm_fast_fail internal helper."""

    def test_revoke_failure_is_tolerated(self):
        """If comm.Revoke() fails, should still exit 70."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0
            comm.Revoke.side_effect = RuntimeError("already revoked")
            comm.Get_size.return_value = 4
            del comm.Get_failed

            with pytest.raises(SystemExit) as exc_info:
                mf._ulfm_fast_fail(comm, 0, 54, "test")

            assert exc_info.value.code == 70

    def test_comm_size_query_failure_tolerated(self):
        """If comm.Get_size() fails, should still exit 70 with sentinel -1."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Revoke = MagicMock()
            comm.Get_size.side_effect = RuntimeError("comm broken")
            del comm.Get_failed

            with pytest.raises(SystemExit) as exc_info:
                mf._ulfm_fast_fail(comm, 0, 54, "test")

            assert exc_info.value.code == 70


# =====================================================================
# ULFM error code detection
# =====================================================================


class TestULFMErrorCodeDetection:
    """Test that ULFM error codes are correctly identified."""

    def test_ulfm_error_codes_populated(self):
        """When ULFM constants exist, _ULFM_ERROR_CODES should be non-empty."""
        with _fresh_mpi_fault_import(ulfm_available=True) as mf:
            assert len(mf._ULFM_ERROR_CODES) == 3
            assert 54 in mf._ULFM_ERROR_CODES  # ERR_PROC_FAILED
            assert 55 in mf._ULFM_ERROR_CODES  # ERR_PROC_FAILED_PENDING
            assert 78 in mf._ULFM_ERROR_CODES  # ERR_REVOKED

    def test_no_ulfm_error_codes_without_ulfm(self):
        """Without ULFM constants, _ULFM_ERROR_CODES should be empty."""
        with _fresh_mpi_fault_import(ulfm_available=False) as mf:
            assert len(mf._ULFM_ERROR_CODES) == 0
