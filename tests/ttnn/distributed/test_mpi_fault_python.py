# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for mpi_fault.py — Python ULFM wrapper.

Tests install_ulfm_handler(), ulfm_guard() context manager,
MPIRankFailureError exception, and graceful degradation without mpi4py.

These run without a real MPI runtime by mocking mpi4py where needed.
"""

import importlib
import importlib.util
import pathlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

_MOD_NAME = "ttnn.distributed.mpi_fault"
# Source-tree path used as fallback when the package is not installed.
_SOURCE_PATH = pathlib.Path(__file__).resolve().parents[3] / "ttnn" / "ttnn" / "distributed" / "mpi_fault.py"


def _import_mpi_fault() -> object:
    """Import ttnn.distributed.mpi_fault, falling back to the source tree.

    mpi_fault.py is part of the ttnn.distributed package and is importable
    via ``import ttnn.distributed.mpi_fault`` in CI (where the wheel is
    installed) and in editable installs.  When the package is not installed
    (e.g. local dev without a wheel build) we load the source file directly
    so the tests still work without jumping through extra setup hoops.
    """
    try:
        return importlib.import_module(_MOD_NAME)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(_MOD_NAME, _SOURCE_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MOD_NAME] = mod
        spec.loader.exec_module(mod)
        return mod


# =====================================================================
# Utility: fresh import of mpi_fault with mocked mpi4py
# =====================================================================


@contextmanager
def _fresh_mpi_fault_import(mpi_available=True, ulfm_available=True):
    """Import ttnn.distributed.mpi_fault with a mocked mpi4py environment.

    Each call evicts the cached module from sys.modules so that the
    module-level ``try: from mpi4py import MPI`` runs again against
    whatever mock we install for that test.

    Args:
        mpi_available: If True, inject a mock mpi4py into sys.modules.
        ulfm_available: If True, mock MPI has ULFM error constants.

    Yields:
        The freshly imported mpi_fault module.
    """
    saved = sys.modules.pop(_MOD_NAME, None)
    saved_mpi = sys.modules.pop("mpi4py", None)
    saved_mpi_MPI = sys.modules.pop("mpi4py.MPI", None)

    try:
        if mpi_available:
            mock_MPI = MagicMock()
            mock_MPI.ERRORS_RETURN = "ERRORS_RETURN_SENTINEL"
            # Standard MPI_UNDEFINED sentinel used by MPI_Group_translate_ranks
            mock_MPI.UNDEFINED = -32766

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

            yield _import_mpi_fault()
        else:
            # Simulate mpi4py not installed by intercepting any import of it.
            # builtins.__import__ is called for every ``import``/``from``
            # statement executed during module load, so patching it is the
            # right hook here.
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "mpi4py" or name.startswith("mpi4py."):
                    raise ImportError("No module named 'mpi4py'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                yield _import_mpi_fault()
    finally:
        # Restore original module state
        sys.modules.pop(_MOD_NAME, None)
        if saved is not None:
            sys.modules[_MOD_NAME] = saved
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

    def test_ulfm_error_calls_os_exit_70(self):
        """ULFM error in fast_fail mode should call os._exit(70).

        We changed from sys.exit(70) to os._exit(70) so that Python atexit
        handlers (including MPI_Finalize) are bypassed on a revoked
        communicator.  os._exit() does NOT raise SystemExit — it terminates
        the process directly — so we mock it instead of using pytest.raises.
        """
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0
            comm.Get_size.return_value = 4
            # Don't have Revoke or Get_failed
            comm.Revoke = MagicMock()

            # Create an MPI.Exception with a ULFM error code
            mpi_exc = mf.MPI.Exception("proc failed", error_code=54)

            with patch("os._exit") as mock_os_exit:
                with mf.ulfm_guard(comm, "Allreduce"):
                    raise mpi_exc

            mock_os_exit.assert_called_once_with(70)

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

            # os._exit() must be mocked: it terminates the process directly
            # without raising SystemExit, so pytest.raises(SystemExit) won't
            # catch it and the test process would be killed instead.
            with patch("os._exit"):
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
        """If Get_failed is available, exception includes world rank list."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0

            # Mock Get_failed returning a group with 2 failed ranks
            failed_group = MagicMock()
            failed_group.Get_size.return_value = 2
            comm.Get_failed.return_value = failed_group
            comm.Get_group.return_value = MagicMock()  # world group

            # Translate_ranks maps failed-group local indices → world ranks
            mf.MPI.Group.Translate_ranks.return_value = [0, 1]
            mf.MPI.UNDEFINED = -32766  # standard MPI_UNDEFINED sentinel

            mpi_exc = mf.MPI.Exception("proc failed", error_code=54)

            with pytest.raises(mf.MPIRankFailureError) as exc_info:
                with mf.ulfm_guard(comm, "Allreduce", policy="fault_tolerant"):
                    raise mpi_exc

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
        """Normal case: returns world rank integers translated from the failed group."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            failed_group = MagicMock()
            failed_group.Get_size.return_value = 3
            comm.Get_failed.return_value = failed_group
            comm.Get_group.return_value = MagicMock()  # world group

            # Translate_ranks maps failed-group local indices → world ranks
            mf.MPI.Group.Translate_ranks.return_value = [0, 1, 2]
            mf.MPI.UNDEFINED = -32766  # standard MPI_UNDEFINED sentinel

            result = mf._try_get_failed_ranks(comm)
            assert result == [0, 1, 2]


# =====================================================================
# _ulfm_fast_fail edge cases
# =====================================================================


class TestUlfmFastFail:
    """Test the _ulfm_fast_fail internal helper."""

    def test_revoke_failure_is_tolerated(self):
        """If comm.Revoke() fails, should still call os._exit(70)."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Get_rank.return_value = 0
            comm.Revoke.side_effect = RuntimeError("already revoked")
            comm.Get_size.return_value = 4
            del comm.Get_failed

            # os._exit() bypasses atexit/MPI_Finalize; mock it so the test
            # process survives and we can assert the call happened.
            with patch("os._exit") as mock_os_exit:
                mf._ulfm_fast_fail(comm, 0, 54, "test")

            mock_os_exit.assert_called_once_with(70)

    def test_comm_size_query_failure_tolerated(self):
        """If comm.Get_size() fails, should still call os._exit(70)."""
        with _fresh_mpi_fault_import() as mf:
            comm = MagicMock()
            comm.Revoke = MagicMock()
            comm.Get_size.side_effect = RuntimeError("comm broken")
            del comm.Get_failed

            # os._exit() bypasses atexit/MPI_Finalize; mock it so the test
            # process survives and we can assert the call happened.
            with patch("os._exit") as mock_os_exit:
                mf._ulfm_fast_fail(comm, 0, 54, "test")

            mock_os_exit.assert_called_once_with(70)


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
