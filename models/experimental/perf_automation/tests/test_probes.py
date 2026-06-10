"""Tests for the real boundaries (execution faked — no hardware, no creds)."""

import subprocess
from pathlib import Path

import pytest

from agent.config import ConfigError
from agent.probes import (
    PreflightError,
    TracyRunError,
    board_to_arch,
    build_tracy_command,
    _extract_json_object,
    make_run_profiled,
    preflight_collect,
    sdk_model_files_runner,
)


def test_board_to_arch():
    assert board_to_arch("n300 L") == "wormhole"
    assert board_to_arch("N150") == "wormhole"
    assert board_to_arch("p150b") == "blackhole"
    assert board_to_arch("unknown-board") is None


def test_build_tracy_command_is_profile_this_form():
    cmd = build_tracy_command("models/x/test_perf.py", "S128", "/tmp/out")
    # raw profile_this command: -v -r -p (NO --no-runtime-analysis) + -o
    assert cmd[:6] == ["python", "-m", "tracy", "-v", "-r", "-p"]
    assert "--no-runtime-analysis" not in cmd
    assert cmd[cmd.index("-o") + 1] == "/tmp/out"
    m_idx = cmd.index("-m", cmd.index("-o"))  # the SECOND -m (tracy's), not python's
    assert cmd[m_idx + 1] == "pytest"
    assert cmd[cmd.index("-k") + 1] == "S128"
    assert cmd[-1] == "-sv"


def test_extract_json_object_tolerates_prose():
    raw = 'Here:\n{"pcc": {"end_to_end": "a.py"}, "model_files": ["a.py"]}\nDone.'
    assert _extract_json_object(raw).startswith('{"pcc"')


def test_runner_fails_fast_without_env_agent(tmp_path):
    with pytest.raises(ConfigError):
        sdk_model_files_runner(env_agent_path=tmp_path / ".env.agent")


def _fake_execute(returncode=0, log_text="", csv_in_outdir=False, csv_in_generated=False):
    def fake(cmd, cwd, env, timeout_s, log_path):
        assert env["TT_METAL_DEVICE_PROFILER"] == "1"
        Path(log_path).write_text(log_text)
        out_dir = Path(cmd[cmd.index("-o") + 1])
        if csv_in_outdir:
            d = out_dir / "reports" / "ts1"
            d.mkdir(parents=True, exist_ok=True)
            (d / "ops_perf_results_ts1.csv").write_text("OP CODE,X\nMatmul,1\n")
        if csv_in_generated:
            d = Path(cwd) / "generated" / "profiler" / "m" / "reports" / "ts2"
            d.mkdir(parents=True, exist_ok=True)
            (d / "ops_perf_results_ts2.csv").write_text("OP CODE,X\nMatmul,1\n")
        return returncode

    return fake


def test_run_profiled_prefers_directed_output_dir(tmp_path):
    rp = make_run_profiled(tmp_path, "t.py", "S128", execute=_fake_execute(csv_in_outdir=True, csv_in_generated=True))
    csv_path, wall_ms = rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert csv_path == tmp_path / "profiles" / "run0_raw.csv"
    assert csv_path.read_text().startswith("OP CODE")
    assert (tmp_path / "profiles" / "run0_tracy.log").is_file()


def test_run_profiled_falls_back_to_watermark_glob(tmp_path):
    rp = make_run_profiled(tmp_path, "t.py", "S128", execute=_fake_execute(csv_in_generated=True))
    csv_path, _ = rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert csv_path.read_text().startswith("OP CODE")


def test_run_profiled_crash_on_nonzero_exit(tmp_path):
    rp = make_run_profiled(tmp_path, "t.py", execute=_fake_execute(returncode=3))
    with pytest.raises(TracyRunError, match="exit 3"):
        rp("e2e", 1, 128, tmp_path / "profiles", 0)


def test_run_profiled_crash_when_no_csv_found(tmp_path):
    rp = make_run_profiled(tmp_path, "t.py", execute=_fake_execute())
    with pytest.raises(TracyRunError, match="no ops_perf_results"):
        rp("e2e", 1, 128, tmp_path / "profiles", 0)


def _fake_collect(stdout, returncode=0):
    def fake(cmd, cwd, env=None, capture_output=None, text=None, timeout=None):
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    return fake


def test_preflight_passes_when_tests_selected(tmp_path):
    n = preflight_collect(tmp_path, "t.py", "S128", runner=_fake_collect("1 test collected in 0.5s"))
    assert n == 1


def test_preflight_raises_on_zero_selection(tmp_path):
    with pytest.raises(PreflightError, match="selects no cases"):
        preflight_collect(
            tmp_path, "t.py", "S512", runner=_fake_collect("5 deselected, 0 tests collected", returncode=5)
        )


def test_collect_cases_and_first_param(tmp_path):
    from agent.probes import collect_cases, first_case_param

    out = "models/x/test_m.py::test_full[S128]\n" "models/x/test_m.py::test_full[S1024]\n" "5 tests collected in 0.5s\n"
    ids, tail = collect_cases(tmp_path, "models/x/test_m.py", runner=_fake_collect(out))
    assert len(ids) == 2
    assert "collected" in tail
    assert first_case_param(ids[0]) == "S128"
    assert first_case_param("path::test_no_params") is None


def test_run_profiled_injects_visible_devices(tmp_path):
    seen = {}

    def spy_execute(cmd, cwd, env, timeout_s, log_path):
        seen.update(env)
        return _fake_execute(csv_in_outdir=True)(cmd, cwd, env, timeout_s, log_path)

    rp = make_run_profiled(tmp_path, "t.py", "S128", execute=spy_execute, extra_env={"TT_METAL_VISIBLE_DEVICES": "0"})
    rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert seen["TT_METAL_VISIBLE_DEVICES"] == "0"


def test_match_input_seq_len():
    from agent.probes import InputMatchError, match_input_to_case

    params = ["S128", "S1024", "S2048", "S4096", "S8192"]
    assert match_input_to_case("128", params) == "S128"
    assert match_input_to_case("4096", params) == "S4096"
    with pytest.raises(InputMatchError, match="NO test case"):
        match_input_to_case("512", params)  # the S512 lesson, now permanent


def test_match_input_image_size():
    from agent.probes import InputMatchError, match_input_to_case

    params = ["128x128", "256x256", "512x512"]
    assert match_input_to_case("128x128", params) == "128x128"
    with pytest.raises(InputMatchError):
        match_input_to_case("64x64", params)


def test_match_input_ambiguous_stops():
    from agent.probes import InputMatchError, match_input_to_case

    with pytest.raises(InputMatchError, match="ambiguous"):
        match_input_to_case("8", params=["224-b8", "448-b8"])
