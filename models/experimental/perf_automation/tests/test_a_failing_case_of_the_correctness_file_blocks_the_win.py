"""A win needs every case of the correctness file to pass, not only its PCC line.

The gate used to judge the scraped PCC alone, so a case that prints no PCC (rendered-speech
WER/MOS, the stop rule, batch independence) could fail while the edit was banked. The rule is
relative to the unedited model: a case already failing there is tolerated, any other failing case
blocks the win and is named."""

from pathlib import Path

from agent import pcc_runner
from agent.pcc_runner import UNNAMED_FAILURE, _verdict_from_output, parse_failed_tests

_PASS = "stage PCC prefill = 0.9995\ne2e PCC=0.9998\n===== 11 passed, 5 warnings in 900.1s (0:15:00) ====="
_QUALITY_FAILS = (
    "stage PCC prefill = 0.9995\n"
    "corpus WER: TT=0.0900 HF=0.0233 (margin 0.05)\n"
    "e2e PCC=0.9998\n"
    "=========================== short test summary info ============================\n"
    "FAILED models/demos/m/tests/e2e/test_e2e.py::test_signal_quality_wer_and_mos - AssertionError: TT wer 0.0900\n"
    "===== 10 passed, 1 failed, 5 warnings in 900.1s (0:15:00) ====="
)
_QUALITY = "test_signal_quality_wer_and_mos"


def test_the_failed_cases_are_read_from_the_short_summary():
    assert parse_failed_tests(_QUALITY_FAILS) == [_QUALITY]
    assert parse_failed_tests(_PASS) == []


def test_an_errored_case_counts_as_failed_and_a_name_is_not_repeated():
    out = (
        "ERROR t.py::test_setup - fixture blew up\n"
        "FAILED t.py::test_a - first assertion\n"
        "FAILED t.py::test_a - same case, second line\n"
        "===== 1 failed, 1 error in 1.0s ====="
    )
    assert parse_failed_tests(out) == ["test_setup", "test_a"]


def test_a_failure_that_is_counted_but_not_named_is_kept_not_dropped():
    out = "e2e PCC=0.9998\n===== 10 passed, 2 failed in 9.0s ====="
    assert parse_failed_tests(out) == [UNNAMED_FAILURE, UNNAMED_FAILURE]


def test_a_case_that_passes_on_the_unedited_model_and_fails_now_blocks_the_win():
    v = _verdict_from_output(_QUALITY_FAILS, threshold=0.99)
    assert v["status"] == "tests_failed"
    assert v["pcc"] == 0.9995 and v["pcc_verified"] is True  # the PCC itself was fine
    assert v["failed_tests"] == [_QUALITY] and v["new_failed_tests"] == [_QUALITY]
    assert _QUALITY in v["error"]


def test_a_case_that_already_fails_on_the_unedited_model_is_tolerated():
    v = _verdict_from_output(_QUALITY_FAILS, threshold=0.99, baseline_failed=[_QUALITY])
    assert v["status"] == "ok" and v["failed_tests"] == [_QUALITY]


def test_the_tolerance_is_per_case_not_a_blanket_pass():
    other = "test_run_ended_on_the_models_stop_rule"
    v = _verdict_from_output(_QUALITY_FAILS.replace(_QUALITY, other), threshold=0.99, baseline_failed=[_QUALITY])
    assert v["status"] == "tests_failed" and v["new_failed_tests"] == [other]


def test_a_low_pcc_is_still_pcc_low_when_every_case_passed():
    v = _verdict_from_output("e2e PCC=0.90\n===== 11 passed in 9.0s =====", threshold=0.99)
    assert v["status"] == "pcc_low" and v["failed_tests"] == []


def test_every_case_passing_is_ok_and_reports_no_failed_case():
    v = _verdict_from_output(_PASS, threshold=0.99)
    assert v["status"] == "ok" and v["failed_tests"] == []


def test_run_pcc_hands_the_contexts_baseline_to_the_verdict(monkeypatch, tmp_path):
    from agent import gitio, probes

    monkeypatch.setattr(gitio, "repo_root", lambda p: tmp_path)
    monkeypatch.setattr(probes, "wait_for_memory_headroom_before_device_work", lambda *a, **k: None)

    def _fake_execute(cmd, cwd, env, timeout_s, log_path, **kw):
        Path(log_path).write_text(_QUALITY_FAILS)
        return 1

    monkeypatch.setattr(probes, "_execute", _fake_execute)

    class _Ctx:
        manifest = {"pathmap": {"pcc": {"end_to_end": {"path": "t.py", "threshold": 0.99}}}, "config": {}}

        def __init__(self, tolerated):
            self._tolerated = tolerated

        def model_root(self):
            return tmp_path

        def baseline_failed_tests(self):
            return self._tolerated

    assert pcc_runner.run_pcc(_Ctx(None))["status"] == "tests_failed"
    assert pcc_runner.run_pcc(_Ctx([_QUALITY]))["status"] == "ok"
