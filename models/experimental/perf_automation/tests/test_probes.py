"""Tests for the real boundaries (execution faked — no hardware, no creds)."""

import subprocess
import sys
from pathlib import Path

import pytest

from agent.probes import (
    PreflightError,
    TracyRunError,
    board_to_arch,
    build_tracy_command,
    detect_perf_crash,
    _extract_json_object,
    make_run_profiled,
    preflight_collect,
    sdk_model_files_runner,
)


def test_detect_perf_crash_tt_fatal():
    log = "TT_FATAL: Data type must be BFLOAT16 and is FLOAT32\n=== 1 failed in 28s ==="
    assert detect_perf_crash(log) and "TT_FATAL" in detect_perf_crash(log)


def test_detect_perf_crash_error_class():
    # pytest collection/fixture failure prints "N errors", never "failed".
    log = "E   RuntimeError: device hang\n=== 1 error in 5s ==="
    assert detect_perf_crash(log) is not None


def test_detect_perf_crash_abort_segfault():
    log = "terminate called after throwing an instance of 'std::runtime_error'\n=== 1 failed ==="
    assert detect_perf_crash(log) is not None


def test_detect_perf_crash_benign_perf_assert_is_not_a_crash():
    # The model ran fully, only the perf-threshold assert failed -> valid measurement, NOT a crash.
    log = "E   assert 88.1 < 85.0  # perf regression\n=== 1 failed in 30s ==="
    assert detect_perf_crash(log) is None


def test_detect_perf_crash_clean_pass_is_none():
    assert detect_perf_crash("e2e PCC=0.99\n=== 1 passed in 30s ===") is None


def test_board_to_arch():
    assert board_to_arch("n300 L") == "wormhole"
    assert board_to_arch("N150") == "wormhole"
    assert board_to_arch("p150b") == "blackhole"
    assert board_to_arch("unknown-board") is None


def test_build_tracy_command_is_profile_this_form():
    cmd = build_tracy_command("models/x/test_perf.py", "S128", "/tmp/out")
    # raw profile_this command: -v -r -p (NO --no-runtime-analysis) + -o
    # uses sys.executable (not a bare "python") so it always resolves to the SAME interpreter
    # running this process, not whatever "python" happens to be first on PATH.
    assert cmd[:6] == [sys.executable, "-m", "tracy", "-v", "-r", "-p"]
    assert "--no-runtime-analysis" not in cmd
    assert cmd[cmd.index("-o") + 1] == "/tmp/out"
    m_idx = cmd.index("-m", cmd.index("-o"))  # the SECOND -m (tracy's), not python's
    assert cmd[m_idx + 1] == "pytest"
    assert cmd[cmd.index("-k") + 1] == "S128"
    assert cmd[-1] == "-sv"


def test_extract_json_object_tolerates_prose():
    raw = 'Here:\n{"pcc": {"end_to_end": "a.py"}, "model_files": ["a.py"]}\nDone.'
    assert _extract_json_object(raw).startswith('{"pcc"')


def test_runner_builds_without_env_agent(tmp_path):
    runner = sdk_model_files_runner(env_agent_path=tmp_path / ".env.agent")
    assert callable(runner)


def _fake_collect(stdout, returncode=0):
    def fake(cmd, cwd, env=None, capture_output=None, text=None, timeout=None):
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    return fake


_COLLECT_ONE = _fake_collect("t.py::test_full[S128]\n1 test collected in 0.1s\n")


@pytest.fixture(autouse=True)
def _clear_node_cache():
    from agent.probes import _NODE_ID_CACHE

    _NODE_ID_CACHE.clear()
    yield
    _NODE_ID_CACHE.clear()


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
    rp = make_run_profiled(
        tmp_path,
        "t.py",
        "S128",
        execute=_fake_execute(csv_in_outdir=True, csv_in_generated=True),
        collect_runner=_COLLECT_ONE,
    )
    csv_path, wall_ms = rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert csv_path == tmp_path / "profiles" / "run0_raw.csv"
    assert csv_path.read_text().startswith("OP CODE")
    assert (tmp_path / "profiles" / "run0_tracy.log").is_file()


def test_run_profiled_falls_back_to_watermark_glob(tmp_path):
    rp = make_run_profiled(
        tmp_path, "t.py", "S128", execute=_fake_execute(csv_in_generated=True), collect_runner=_COLLECT_ONE
    )
    csv_path, _ = rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert csv_path.read_text().startswith("OP CODE")


def test_run_profiled_crash_on_nonzero_exit(tmp_path):
    rp = make_run_profiled(tmp_path, "t.py", execute=_fake_execute(returncode=3), collect_runner=_COLLECT_ONE)
    with pytest.raises(TracyRunError, match="exit 3"):  # allow-pytest.raises: no expect_error fixture
        rp("e2e", 1, 128, tmp_path / "profiles", 0)


def test_run_profiled_crash_when_no_csv_found(tmp_path):
    rp = make_run_profiled(tmp_path, "t.py", execute=_fake_execute(), collect_runner=_COLLECT_ONE)
    with pytest.raises(TracyRunError, match="no ops_perf_results"):  # allow-pytest.raises: no expect_error fixture
        rp("e2e", 1, 128, tmp_path / "profiles", 0)


def test_preflight_passes_when_tests_selected(tmp_path):
    n = preflight_collect(tmp_path, "t.py", "S128", runner=_fake_collect("1 test collected in 0.5s"))
    assert n == 1


def test_preflight_raises_on_zero_selection(tmp_path):
    with pytest.raises(PreflightError, match="selects no cases"):  # allow-pytest.raises: no expect_error fixture
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


_TWO_NODES = "models/x/test_m.py::test_full[in0-device_params0]\nmodels/x/test_m.py::test_full[in1-device_params0]\n2 tests collected\n"


def test_resolve_node_id_exact_param_match(tmp_path):
    from agent.probes import resolve_node_id

    nid = resolve_node_id(tmp_path, "models/x/test_m.py", "in0-device_params0", runner=_fake_collect(_TWO_NODES))
    assert nid == "models/x/test_m.py::test_full[in0-device_params0]"


def test_resolve_node_id_unique_substring(tmp_path):
    from agent.probes import resolve_node_id

    nid = resolve_node_id(tmp_path, "models/x/test_m.py", "in1", runner=_fake_collect(_TWO_NODES))
    assert nid == "models/x/test_m.py::test_full[in1-device_params0]"


def test_resolve_node_id_self_heals_on_bad_hint(tmp_path):
    from agent.probes import resolve_node_id

    one = "models/x/test_m.py::test_text_generation_perf[device_params0]\n1 test collected\n"
    nid = resolve_node_id(tmp_path, "models/x/test_m.py", "in0", runner=_fake_collect(one))
    assert nid == "models/x/test_m.py::test_text_generation_perf[device_params0]"


def test_resolve_node_id_empty_raises_clear_error(tmp_path):
    from agent.probes import PreflightError, resolve_node_id

    with pytest.raises(PreflightError, match="collects no tests"):  # allow-pytest.raises: no expect_error fixture
        resolve_node_id(tmp_path, "models/x/test_m.py", "in0", runner=_fake_collect("0 tests collected\n"))


def test_resolve_node_id_caches_by_mtime(tmp_path):
    from agent.probes import resolve_node_id

    test_file = tmp_path / "t.py"
    test_file.write_text("# test\n")
    calls = [0]

    def counting(cmd, cwd, env=None, capture_output=None, text=None, timeout=None):
        calls[0] += 1
        return subprocess.CompletedProcess(cmd, 0, "t.py::test_full[in0]\n1 test collected\n", "")

    a = resolve_node_id(tmp_path, "t.py", "in0", runner=counting)
    b = resolve_node_id(tmp_path, "t.py", "in0", runner=counting)
    assert a == b == "t.py::test_full[in0]"
    assert calls[0] == 1  # second call served from cache


def test_run_profiled_runs_exact_node_id_no_dash_k(tmp_path):
    seen = {}

    def spy_execute(cmd, cwd, env, timeout_s, log_path):
        seen["cmd"] = list(cmd)
        return _fake_execute(csv_in_outdir=True)(cmd, cwd, env, timeout_s, log_path)

    rp = make_run_profiled(
        tmp_path, "models/x/test_m.py", "in0", execute=spy_execute, collect_runner=_fake_collect(_TWO_NODES)
    )
    rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert "-k" not in seen["cmd"]
    assert "models/x/test_m.py::test_full[in0-device_params0]" in seen["cmd"]


def test_run_profiled_injects_visible_devices(tmp_path):
    seen = {}

    def spy_execute(cmd, cwd, env, timeout_s, log_path):
        seen.update(env)
        return _fake_execute(csv_in_outdir=True)(cmd, cwd, env, timeout_s, log_path)

    rp = make_run_profiled(
        tmp_path,
        "t.py",
        "S128",
        execute=spy_execute,
        extra_env={"TT_METAL_VISIBLE_DEVICES": "0"},
        collect_runner=_COLLECT_ONE,
    )
    rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert seen["TT_METAL_VISIBLE_DEVICES"] == "0"


def test_match_input_seq_len():
    from agent.probes import InputMatchError, match_input_to_case

    params = ["S128", "S1024", "S2048", "S4096", "S8192"]
    assert match_input_to_case("128", params) == "S128"
    assert match_input_to_case("4096", params) == "S4096"
    with pytest.raises(InputMatchError, match="NO test case"):  # allow-pytest.raises: no expect_error fixture
        match_input_to_case("512", params)  # the S512 lesson, now permanent


def test_match_input_image_size():
    from agent.probes import InputMatchError, match_input_to_case

    params = ["128x128", "256x256", "512x512"]
    assert match_input_to_case("128x128", params) == "128x128"
    with pytest.raises(InputMatchError):  # allow-pytest.raises: no expect_error fixture
        match_input_to_case("64x64", params)


def test_match_input_ambiguous_stops():
    from agent.probes import InputMatchError, match_input_to_case

    with pytest.raises(InputMatchError, match="ambiguous"):  # allow-pytest.raises: no expect_error fixture
        match_input_to_case("8", params=["224-b8", "448-b8"])
