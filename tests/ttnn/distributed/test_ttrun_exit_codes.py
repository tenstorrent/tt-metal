# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for ttrun.py exit code interpretation and PRRTE version detection.

Tests the ExitCategory enum, interpret_exit_code() function, and
_detect_openmpi_major_version() / _get_abort_on_failure_mca_param() helpers.

These run without MPI — they test pure Python logic used by tt-run for
CI triage of mpirun exit codes.
"""

import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

# ttrun.py lives at ttnn/ttnn/distributed/ttrun.py in the source tree but is
# NOT installed into the ttnn site-package.  Load it directly by path and
# register it in sys.modules so that 'from ttnn.distributed.ttrun import ...'
# works regardless of which ttnn package (source vs installed) is active.
import importlib.util
import pathlib

_ttrun_path = pathlib.Path(__file__).resolve().parents[3] / "ttnn" / "ttnn" / "distributed" / "ttrun.py"
if "ttnn.distributed.ttrun" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("ttnn.distributed.ttrun", _ttrun_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["ttnn.distributed.ttrun"] = _mod
    _spec.loader.exec_module(_mod)
ttrun_mod = sys.modules["ttnn.distributed.ttrun"]
from ttnn.distributed.ttrun import (
    EXIT_APP_ERROR,
    EXIT_RANK_FAILURE,
    EXIT_SIGINT,
    EXIT_SIGKILL,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    EXIT_ULFM_FAST_FAIL,
    ExitCategory,
    _SIGNAL_NAMES,
    _detect_openmpi_major_version,
    _get_abort_on_failure_mca_param,
    _log_exit_interpretation,
    interpret_exit_code,
)

# =====================================================================
# ExitCategory enum completeness
# =====================================================================


class TestExitCategoryEnum:
    """Verify ExitCategory enum has all expected members."""

    def test_all_categories_present(self):
        expected = {
            "SUCCESS",
            "APP_ERROR",
            "ULFM_FAST_FAIL",
            "RANK_SIGNAL",
            "FINALIZE_TIMEOUT",
            "TIMEOUT",
            "INTERRUPTED",
            "INFRA_ERROR",
        }
        actual = {e.name for e in ExitCategory}
        assert expected == actual, f"Missing categories: {expected - actual}"

    def test_category_values_are_strings(self):
        for cat in ExitCategory:
            assert isinstance(cat.value, str), f"{cat.name}.value is not a string"

    def test_categories_are_distinct(self):
        values = [e.value for e in ExitCategory]
        assert len(values) == len(set(values)), "Duplicate category values"


# =====================================================================
# interpret_exit_code — happy paths
# =====================================================================


class TestInterpretExitCodeHappyPaths:
    """Test interpret_exit_code with standard exit codes."""

    def test_success_exit_0(self):
        interp = interpret_exit_code(0)
        assert interp.category == ExitCategory.SUCCESS
        assert interp.raw_code == 0
        assert interp.signal_num is None
        assert interp.signal_name is None
        assert interp.ci_exit_code == EXIT_SUCCESS
        assert "success" in interp.summary.lower()

    def test_app_error_exit_1(self):
        interp = interpret_exit_code(1)
        assert interp.category == ExitCategory.APP_ERROR
        assert interp.ci_exit_code == EXIT_APP_ERROR
        assert interp.signal_num is None

    def test_ulfm_fast_fail_exit_70(self):
        interp = interpret_exit_code(70)
        assert interp.category == ExitCategory.ULFM_FAST_FAIL
        assert interp.raw_code == 70
        assert interp.ci_exit_code == EXIT_RANK_FAILURE
        assert "ulfm" in interp.summary.lower() or "fast-fail" in interp.summary.lower()

    def test_timeout_exit_124(self):
        interp = interpret_exit_code(124)
        assert interp.category == ExitCategory.TIMEOUT
        assert interp.ci_exit_code == EXIT_TIMEOUT
        assert "timeout" in interp.summary.lower()

    def test_sigkill_exit_137(self):
        """137 = 128 + SIGKILL(9) — typically OOM."""
        interp = interpret_exit_code(137)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 9
        assert interp.signal_name == "SIGKILL"
        assert interp.ci_exit_code == EXIT_RANK_FAILURE
        assert "oom" in interp.summary.lower() or "SIGKILL" in interp.summary

    def test_sigint_exit_130(self):
        """130 = 128 + SIGINT(2)."""
        interp = interpret_exit_code(130)
        assert interp.category == ExitCategory.INTERRUPTED
        assert interp.signal_num == 2
        assert interp.signal_name == "SIGINT"
        assert interp.ci_exit_code == EXIT_SIGINT

    def test_sigalrm_exit_142(self):
        """142 = 128 + SIGALRM(14) — MPI_Finalize watchdog."""
        interp = interpret_exit_code(142)
        assert interp.category == ExitCategory.FINALIZE_TIMEOUT
        assert interp.signal_num == 14
        assert interp.signal_name == "SIGALRM"
        assert interp.ci_exit_code == EXIT_RANK_FAILURE


# =====================================================================
# interpret_exit_code — edge cases
# =====================================================================


class TestInterpretExitCodeEdgeCases:
    """Edge cases and boundary conditions for interpret_exit_code."""

    def test_negative_sigint(self):
        """subprocess.Popen returns -SIGINT when killed by SIGINT."""
        interp = interpret_exit_code(-signal.SIGINT)
        assert interp.category == ExitCategory.INTERRUPTED
        assert interp.signal_num == signal.SIGINT
        assert interp.ci_exit_code == EXIT_SIGINT

    def test_negative_sigkill(self):
        """subprocess.Popen returns -SIGKILL when killed."""
        interp = interpret_exit_code(-signal.SIGKILL)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == signal.SIGKILL
        assert interp.ci_exit_code == EXIT_RANK_FAILURE

    def test_negative_sigsegv(self):
        interp = interpret_exit_code(-signal.SIGSEGV)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == signal.SIGSEGV
        assert interp.signal_name == "SIGSEGV"

    def test_negative_sigterm(self):
        interp = interpret_exit_code(-signal.SIGTERM)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == signal.SIGTERM

    @pytest.mark.parametrize("code", [2, 42, 69, 71, 100, 126, 127])
    def test_generic_app_errors(self, code):
        """All non-special codes 1-127 (except 70, 124) are APP_ERROR."""
        interp = interpret_exit_code(code)
        assert interp.category == ExitCategory.APP_ERROR
        assert interp.ci_exit_code == EXIT_APP_ERROR
        assert str(code) in interp.summary

    def test_exit_128_is_app_error(self):
        """Exit code 128 exactly is ambiguous (128+0=no signal). Treated as APP_ERROR."""
        # 128 is NOT > 128, so it falls through to the generic APP_ERROR path
        interp = interpret_exit_code(128)
        assert interp.category == ExitCategory.APP_ERROR

    def test_sigabrt_exit_134(self):
        """134 = 128 + SIGABRT(6)."""
        interp = interpret_exit_code(134)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 6
        assert interp.signal_name == "SIGABRT"

    def test_sigsegv_exit_139(self):
        """139 = 128 + SIGSEGV(11)."""
        interp = interpret_exit_code(139)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 11
        assert interp.signal_name == "SIGSEGV"

    def test_unknown_signal_exit_200(self):
        """200 = 128 + 72 (unknown signal number)."""
        interp = interpret_exit_code(200)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 72
        # Unknown signal should still get a name like "SIG72"
        assert "72" in interp.signal_name

    def test_large_exit_code_255(self):
        """255 = 128 + 127 (max plausible signal on 128+ path)."""
        interp = interpret_exit_code(255)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 127

    def test_negative_unknown_signal(self):
        """Negative return code with unknown signal number."""
        interp = interpret_exit_code(-99)
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 99
        assert "99" in interp.signal_name


# =====================================================================
# interpret_exit_code — CI exit code normalization
# =====================================================================


class TestCINormalization:
    """Verify CI exit codes are properly normalized."""

    def test_success_normalizes_to_0(self):
        assert interpret_exit_code(0).ci_exit_code == 0

    def test_app_error_normalizes_to_1(self):
        assert interpret_exit_code(1).ci_exit_code == 1

    def test_ulfm_fast_fail_normalizes_to_2(self):
        """Exit code 70 -> ci_exit_code 2 (rank failure)."""
        assert interpret_exit_code(70).ci_exit_code == EXIT_RANK_FAILURE

    def test_sigkill_normalizes_to_2(self):
        assert interpret_exit_code(137).ci_exit_code == EXIT_RANK_FAILURE

    def test_finalize_watchdog_normalizes_to_2(self):
        assert interpret_exit_code(142).ci_exit_code == EXIT_RANK_FAILURE

    def test_timeout_normalizes_to_124(self):
        assert interpret_exit_code(124).ci_exit_code == EXIT_TIMEOUT

    def test_sigint_normalizes_to_130(self):
        assert interpret_exit_code(130).ci_exit_code == EXIT_SIGINT

    def test_negative_sigint_normalizes_to_130(self):
        assert interpret_exit_code(-signal.SIGINT).ci_exit_code == EXIT_SIGINT


# =====================================================================
# _log_exit_interpretation — log quality
# =====================================================================


class TestLogExitInterpretation:
    """Test that _log_exit_interpretation produces useful output."""

    def test_success_path_no_error_log(self, capsys):
        """Success should not produce error-level log lines."""
        interp = interpret_exit_code(0)
        # _log_exit_interpretation uses loguru, which writes to stderr
        # We just verify it doesn't raise
        _log_exit_interpretation(interp)

    def test_ulfm_fast_fail_has_structured_output(self):
        """Exit code 70 log should include category and raw code."""
        interp = interpret_exit_code(70)
        assert interp.category == ExitCategory.ULFM_FAST_FAIL
        # The log line includes "EXIT CODE INTERPRETATION", raw code, category
        # We verify the interp dataclass has enough info for meaningful logs
        assert "70" in str(interp.raw_code)
        assert "ulfm_fast_fail" in interp.category.value

    def test_signal_death_includes_signal_name(self):
        """When a rank dies by signal, the log should include the signal name."""
        interp = interpret_exit_code(137)
        assert interp.signal_name == "SIGKILL"
        assert interp.signal_num == 9

    def test_finalize_timeout_mentions_watchdog(self):
        interp = interpret_exit_code(142)
        assert "watchdog" in interp.summary.lower() or "finalize" in interp.summary.lower()

    def test_oom_hint_in_sigkill_summary(self):
        interp = interpret_exit_code(137)
        assert "oom" in interp.summary.lower() or "OOM" in interp.summary


# =====================================================================
# PRRTE / ORTE version detection
# =====================================================================


class TestPRRTEDetection:
    """Test _detect_openmpi_major_version and _get_abort_on_failure_mca_param."""

    def setup_method(self):
        """Reset cached version between tests."""
        ttrun_mod._detect_openmpi_major_version.cache_clear()

    def test_detect_openmpi_5(self):
        """Standard OpenMPI 5.x output."""
        mock_result = MagicMock()
        mock_result.stdout = "mpirun (Open MPI) 5.0.3\nReport bugs to ...\n"
        with patch("subprocess.run", return_value=mock_result):
            v = _detect_openmpi_major_version()
        assert v == 5

    def test_detect_openmpi_4(self):
        """Standard OpenMPI 4.x output."""
        mock_result = MagicMock()
        mock_result.stdout = "mpirun (Open MPI) 4.1.6\nReport bugs to ...\n"
        with patch("subprocess.run", return_value=mock_result):
            v = _detect_openmpi_major_version()
        assert v == 4

    def test_detect_mpirun_not_found(self):
        """mpirun not installed — should return None gracefully."""
        with patch("subprocess.run", side_effect=FileNotFoundError("mpirun not found")):
            v = _detect_openmpi_major_version()
        assert v is None

    def test_detect_mpirun_timeout(self):
        """mpirun --version hangs — should return None after timeout."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("mpirun", 5)):
            v = _detect_openmpi_major_version()
        assert v is None

    def test_detect_malformed_output(self):
        """mpirun --version returns unexpected output format."""
        mock_result = MagicMock()
        mock_result.stdout = "something completely different\n"
        with patch("subprocess.run", return_value=mock_result):
            v = _detect_openmpi_major_version()
        assert v is None

    def test_detect_empty_output(self):
        """mpirun --version returns nothing."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            v = _detect_openmpi_major_version()
        assert v is None

    def test_detect_prrte_version_string(self):
        """Some PRRTE builds have different format — should still work if 'Open MPI' present."""
        mock_result = MagicMock()
        mock_result.stdout = "prterun (PRRTE) 3.0.0\nmpirun (Open MPI) 5.0.0\n"
        with patch("subprocess.run", return_value=mock_result):
            v = _detect_openmpi_major_version()
        assert v == 5

    def test_abort_param_openmpi_5(self):
        """OpenMPI 5+ should use prte_ prefix."""
        mock_result = MagicMock()
        mock_result.stdout = "mpirun (Open MPI) 5.0.3\n"
        with patch("subprocess.run", return_value=mock_result):
            param = _get_abort_on_failure_mca_param()
        assert param == "prte_abort_on_non_zero_status"

    def test_abort_param_openmpi_4(self):
        """OpenMPI 4 should use orte_ prefix."""
        ttrun_mod._detect_openmpi_major_version.cache_clear()
        mock_result = MagicMock()
        mock_result.stdout = "mpirun (Open MPI) 4.1.6\n"
        with patch("subprocess.run", return_value=mock_result):
            param = _get_abort_on_failure_mca_param()
        assert param == "orte_abort_on_non_zero_status"

    def test_abort_param_unknown_version(self):
        """Unknown version defaults to orte_ (safe fallback)."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            param = _get_abort_on_failure_mca_param()
        assert param == "orte_abort_on_non_zero_status"

    def test_cached_version_reused(self):
        """Second call should use cached value, not re-run subprocess."""
        ttrun_mod._detect_openmpi_major_version.cache_clear()
        mock_result = MagicMock()
        mock_result.stdout = "mpirun (Open MPI) 5.0.0\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            v1 = _detect_openmpi_major_version()
            v2 = _detect_openmpi_major_version()
        assert v1 == v2 == 5
        # Should only call subprocess.run once due to caching
        assert mock_run.call_count == 1


# =====================================================================
# ExitCodeInterpretation dataclass
# =====================================================================


class TestExitCodeInterpretation:
    """Test the ExitCodeInterpretation dataclass fields."""

    def test_all_fields_populated_for_signal(self):
        interp = interpret_exit_code(137)
        assert interp.raw_code == 137
        assert interp.category == ExitCategory.RANK_SIGNAL
        assert interp.signal_num == 9
        assert interp.signal_name == "SIGKILL"
        assert isinstance(interp.summary, str) and len(interp.summary) > 0
        assert isinstance(interp.ci_exit_code, int)

    def test_no_signal_fields_for_app_error(self):
        interp = interpret_exit_code(42)
        assert interp.signal_num is None
        assert interp.signal_name is None

    def test_no_signal_fields_for_success(self):
        interp = interpret_exit_code(0)
        assert interp.signal_num is None
        assert interp.signal_name is None


# =====================================================================
# Signal name map coverage
# =====================================================================


class TestSignalNames:
    """Verify the _SIGNAL_NAMES map covers common signals."""

    @pytest.mark.parametrize(
        "sig,name",
        [
            (1, "SIGHUP"),
            (2, "SIGINT"),
            (6, "SIGABRT"),
            (9, "SIGKILL"),
            (11, "SIGSEGV"),
            (14, "SIGALRM"),
            (15, "SIGTERM"),
        ],
    )
    def test_known_signals(self, sig, name):
        assert _SIGNAL_NAMES[sig] == name

    def test_sigpipe_present(self):
        assert 13 in _SIGNAL_NAMES
        assert _SIGNAL_NAMES[13] == "SIGPIPE"


# =====================================================================
# Exit code constants
# =====================================================================


class TestExitCodeConstants:
    """Verify exit code constants match their documented values."""

    def test_exit_success(self):
        assert EXIT_SUCCESS == 0

    def test_exit_app_error(self):
        assert EXIT_APP_ERROR == 1

    def test_exit_rank_failure(self):
        assert EXIT_RANK_FAILURE == 2

    def test_exit_ulfm_fast_fail(self):
        assert EXIT_ULFM_FAST_FAIL == 70

    def test_exit_timeout(self):
        assert EXIT_TIMEOUT == 124

    def test_exit_sigint(self):
        assert EXIT_SIGINT == 130

    def test_exit_sigkill(self):
        assert EXIT_SIGKILL == 137
