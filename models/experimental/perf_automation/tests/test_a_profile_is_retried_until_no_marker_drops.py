"""A profile that drops markers is re-run -- on a reset board, with the knob the drops point at --
until one keeps every marker; only when the attempts run out is the cleanest finished one kept.

WH Galaxy, 2026-09-28: the per-op baseline "1 passed" with 11,520 drop warnings (every site of 32
chips x 72 cores x 5 RISCs, each once, in the single read after a trace capture) and then the tracy
capture tool segfaulted at shutdown, leaving no CSV. The tool raised on the exit code, the retry
that followed ran on a board nobody had reset, and optimize never got a baseline. These pin: a
retry after any unclean end resets first, the remedy follows the evidence, and a finished attempt
is never thrown away for a later one that crashed.
"""

import subprocess
from pathlib import Path

import pytest

from agent import device_recovery as dr
from agent import probes

_SITE = "Profiler DRAM buffers were full, markers were dropped! device {d}, worker core {x}, {y}, Risc {r},  bufferEndIndex = 12000. Please either decrease"


def _burst(n_devices=2):
    return "\n".join(
        _SITE.format(d=d, x=x, y=18, r=r) for d in range(n_devices) for x in (18, 19) for r in ("BRISC", "NCRISC")
    )


def _collect_one(cmd, cwd, env=None, capture_output=None, text=None, timeout=None):
    return subprocess.CompletedProcess(cmd, 0, "t.py::test_full[S128]\n1 test collected in 0.1s\n", "")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    probes._NODE_ID_CACHE.clear()
    monkeypatch.setattr(probes, "_await_cool", lambda *a, **k: None)
    monkeypatch.setenv(dr.DEVICES_ENV, "all")
    yield
    probes._NODE_ID_CACHE.clear()


def _scripted(steps, seen_env):
    """Each step is (exit code, log text, write a CSV?); the CSV names its attempt."""
    n = [0]

    def execute(cmd, cwd, env, timeout_s, log_path):
        code, log, csv = steps[min(n[0], len(steps) - 1)]
        n[0] += 1
        seen_env.append(dict(env))
        Path(log_path).write_text(log + "\nattempt=%d\n" % n[0])
        if csv:
            d = Path(cmd[cmd.index("-o") + 1]) / "reports" / ("ts%d" % n[0])
            d.mkdir(parents=True, exist_ok=True)
            (d / "ops_perf_results_x.csv").write_text("OP CODE,ATTEMPT\nMatmul,%d\n" % n[0])
        return code

    return execute


def _run(tmp_path, steps):
    envs, resets = [], []
    rp = probes.make_run_profiled(
        tmp_path,
        "t.py",
        "S128",
        execute=_scripted(steps, envs),
        collect_runner=_collect_one,
        device_reset=lambda **kw: resets.append(kw) or True,
        extra_env={probes.PERF_FLUSH_EVERY_ENV: "4"},
    )
    csv, _ = rp("e2e", 1, 128, tmp_path / "profiles", 0)
    return csv, envs, resets


# -- the evidence ---------------------------------------------------------------------------------


def test_a_single_overflowed_read_is_not_repeated():
    ev = probes.marker_drop_evidence(_burst())
    assert ev["drops"] == 8 and ev["sites"] == 8 and ev["repeated"] is False


def test_a_site_dropping_on_several_reads_is_repeated():
    ev = probes.marker_drop_evidence(_burst(1) + "\n" + _burst(1))
    assert ev["sites"] == 4 and ev["repeated"] is True


def test_an_imbalance_alone_still_counts_as_a_drop():
    ev = probes.marker_drop_evidence("some end markers were dropped due to DRAM-buffer overflow")
    assert ev["drops"] == 1 and ev["sites"] == 0


def test_no_drop_is_no_evidence():
    assert probes.marker_drop_evidence("1 passed in 3s") is None


# -- the remedy -----------------------------------------------------------------------------------


def test_one_overflowed_read_grows_the_buffer():
    assert probes.choose_marker_drop_remedy({"repeated": False}, 0, "4") == {probes._SUPPORT_COUNT_ENV: "8000"}


def test_repeated_drops_drain_more_often():
    # by the heal factor, so a buffer-sized interval reaches a per-few-ops one within the budget
    assert probes.choose_marker_drop_remedy({"repeated": True}, 0, "250") == {probes.PERF_FLUSH_EVERY_ENV: "31"}
    assert probes.choose_marker_drop_remedy({"repeated": True}, 0, "31") == {probes.PERF_FLUSH_EVERY_ENV: "3"}
    assert probes.choose_marker_drop_remedy({"repeated": True}, 0, "4") == {probes.PERF_FLUSH_EVERY_ENV: "1"}


def test_repeated_drops_at_the_fastest_drain_grow_the_buffer():
    assert probes.choose_marker_drop_remedy({"repeated": True}, 8000, "1") == {probes._SUPPORT_COUNT_ENV: "64000"}


def test_an_unknown_drain_is_not_guessed():
    assert probes.choose_marker_drop_remedy({"repeated": True}, 0, None) == {probes._SUPPORT_COUNT_ENV: "8000"}


def test_a_full_buffer_falls_back_to_draining_then_to_nothing():
    top = probes._MAX_PROFILER_SUPPORT_COUNT
    assert probes.choose_marker_drop_remedy({"repeated": False}, top, "4") == {probes.PERF_FLUSH_EVERY_ENV: "1"}
    assert probes.choose_marker_drop_remedy({"repeated": False}, top, "1") is None


# -- the run --------------------------------------------------------------------------------------


def test_drops_are_retried_on_a_reset_board_until_none_remain(tmp_path):
    csv, envs, resets = _run(tmp_path, [(0, _burst(), True), (0, "1 passed", True)])
    assert len(envs) == 2
    assert envs[1][probes._SUPPORT_COUNT_ENV] == "8000", "the burst grew the buffer"
    assert len(resets) == 1 and resets[0].get("fault_is_certain") is True
    assert "markers were dropped" in resets[0]["error_text"]
    assert csv.read_text().endswith("Matmul,2\n"), "the complete attempt is the result"
    assert not (tmp_path / "profiles" / "run0.partial").exists()
    assert "attempt=1" in (tmp_path / "profiles" / "run0_tracy.log.attempt1").read_text()
    assert "attempt=2" in (tmp_path / "profiles" / "run0_tracy.log").read_text()
    assert not (tmp_path / "profiles" / "run0_raw.csv.best").exists()


def test_when_every_attempt_drops_the_cleanest_one_is_kept(tmp_path):
    fewest = _burst(1)  # 4 drops
    steps = [(0, _burst(3), True), (0, fewest, True)] + [(0, _burst(2), True)] * 10
    csv, envs, resets = _run(tmp_path, steps)
    assert len(envs) == probes._MAX_HEAL_ATTEMPTS + 1
    assert len(resets) == probes._MAX_HEAL_ATTEMPTS, "every retry started on a reset board"
    assert csv.read_text().endswith("Matmul,2\n"), "the attempt with the fewest drops"
    log = (tmp_path / "profiles" / "run0_tracy.log").read_text()
    assert "attempt=2" in log and "kept the cleanest finished attempt (4 marker drop(s))" in log
    assert (tmp_path / "profiles" / "run0.partial").is_file()


def test_a_crash_after_a_pass_is_reset_and_run_again(tmp_path):
    crashed = "1 passed in 4975s\nSegmentation fault (core dumped)"
    csv, envs, resets = _run(tmp_path, [(139, crashed, False), (0, "1 passed", True)])
    assert len(envs) == 2 and len(resets) == 1
    assert "Segmentation fault" in resets[0]["error_text"]
    assert csv.read_text().endswith("Matmul,2\n")


def test_a_finished_attempt_survives_later_crashes(tmp_path):
    crashed = "1 passed\nSegmentation fault (core dumped)"
    csv, envs, resets = _run(tmp_path, [(0, _burst(), True)] + [(139, crashed, False)] * 10)
    assert csv.read_text().endswith("Matmul,1\n")
    assert "attempt=1" in (tmp_path / "profiles" / "run0_tracy.log").read_text()
    assert (tmp_path / "profiles" / "run0.partial").is_file()


def test_a_plain_nonzero_exit_is_still_raised_without_a_retry(tmp_path):
    with pytest.raises(probes.TracyRunError, match="exit 3"):  # allow-pytest.raises: no expect_error fixture
        _run(tmp_path, [(3, "usage error", False), (0, "1 passed", True)])


def test_a_crash_with_nothing_finished_still_raises_once_attempts_run_out(tmp_path):
    crashed = "Segmentation fault (core dumped)"
    with pytest.raises(probes.TracyRunError, match="exit 139"):  # allow-pytest.raises: no expect_error fixture
        _run(tmp_path, [(139, crashed, False)] * 10)


# -- the forward's drain interval and pytest's own timeout ----------------------------------------


def test_a_capacity_bridged_forward_drains_at_the_buffers_size(monkeypatch):
    from agent import profiler_drain as pd
    from agent.measure import _capacity_scaled_osl

    class _Run:
        @staticmethod
        def coverage_cache_get_ops_per_step(repo_root, node, case, **_kw):
            return 38_604

    monkeypatch.setattr(probes, "_cc_optimize", lambda name: _Run())
    monkeypatch.delenv(probes._SUPPORT_COUNT_ENV, raising=False)
    osl, flush = _capacity_scaled_osl(None, "r", "n", "c", 128)
    assert int(flush) == pd.capacity_cadence(), "the same interval the stage pass reads at"


def test_every_tool_launched_pytest_turns_pytests_timeout_off():
    """A @pytest.mark.timeout on a test beats "-o timeout=0"; only disabling the plugin ends it."""
    import pathlib
    import re

    root = pathlib.Path(probes.__file__).resolve().parents[1]
    offenders = [
        f"{p.relative_to(root)}:{i}"
        for p in root.rglob("*.py")
        if "tests" not in p.parts
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1)
        if re.search(r'"timeout=0"', line)
    ]
    assert offenders == [], offenders
    assert probes.PYTEST_NO_TIMEOUT == ("-p", "no:timeout")
    cmd = probes.build_tracy_command("t.py", None, "/tmp/out")
    i = cmd.index("pytest")
    assert cmd[i + 1 : i + 3] == list(probes.PYTEST_NO_TIMEOUT)
