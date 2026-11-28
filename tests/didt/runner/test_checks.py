# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tests.didt.runner.checks import (
    GddrEdc,
    Result,
    Status,
    child_failure,
    classify_pytest,
    collected_node_ids,
    gddr_edc,
    gddr_result,
    loop_seconds,
    missing_node_ids,
    read_tail,
    sweep_skipped_windows,
    umd_metal_ids,
)

TESTDATA = Path(__file__).parent / "testdata"
LOGURU = "2026-10-31 03:12:44.123 | {} | {} - {}\n"


def test_pytest_skip_is_fail():
    r = classify_pytest("SKIPPED [1] x.py:1: golden trace not found\n=== 1 skipped in 3s ===", 1)
    assert r.status == Status.FAIL
    assert "skipped" in r.reason


def test_pytest_expected_count():
    assert classify_pytest("=== 2 passed in 9s ===", 2).status == Status.PASS
    assert classify_pytest("=== 1 passed in 9s ===", 2).status == Status.FAIL


def test_pytest_failed_reason():
    text = "FAILED tests/didt/x.py::t[p] - AssertionError: ND on device 3\n=== 1 failed in 9s ===\n"
    assert classify_pytest(text, 1) == Result(Status.FAIL, "AssertionError: ND on device 3")


def test_pytest_real_summary_with_duration():
    assert classify_pytest("===== 1 passed in 68.99s (0:01:08) =====\n", 1) == Result(Status.PASS, "1 passed")


def test_status_values_are_spec_names():
    assert [s.value for s in Status] == ["PENDING", "RUN", "PASS", "WARN", "FAIL", "HANG", "SKIP"]


def test_collected_node_ids_flat():
    flat = "tests/didt/a.py::t[p1]\ntests/didt/b.py::u[q]\n\n2 tests collected in 1.2s\n"
    assert collected_node_ids(flat) == ["tests/didt/a.py::t[p1]", "tests/didt/b.py::u[q]"]


def test_collected_node_ids_tree_gives_full_ids():
    # `pytest --collect-only -q` prints the tree, not flat ids: pytest.ini's addopts -vv outweighs one -q.
    tree = (
        "collecting ... collected 3 items\n\n"
        "<Dir tt-metal>\n"
        "  <Package tests>\n"
        "    <Dir didt>\n"
        "      <Module test_ff1_matmul.py>\n"
        "        <Function test_ff1_matmul[blackhole-galaxy-without_gelu]>\n"
        "      <Package runner>\n"
        "        <Module test_checks.py>\n"
        "          <Class TestX>\n"
        "            <Function test_a>\n"
        "          <Function test_umd_metal_ids>\n\n"
        "========================== 3 tests collected in 0.08s ==========================\n"
    )
    assert collected_node_ids(tree) == [
        "tests/didt/test_ff1_matmul.py::test_ff1_matmul[blackhole-galaxy-without_gelu]",
        "tests/didt/runner/test_checks.py::TestX::test_a",
        "tests/didt/runner/test_checks.py::test_umd_metal_ids",
    ]


def test_missing_node_ids_compares_full_id():
    assert missing_node_ids(["a/x.py::t[p]"], ["b/y.py::t[p]"]) == ["a/x.py::t[p]"]
    assert missing_node_ids(["a/x.py::t[p]", "b/y.py::u"], ["a/x.py::t[p]"]) == ["b/y.py::u"]


def test_sweep_skipped_windows():
    where = "tests.didt.scratch_core_sweep:test_core_sweep:108"
    log = LOGURU.format("WARNING ", where, "  window 11x10 at (0,0): device 21 NOT deterministic, 96 elements")
    log += LOGURU.format("INFO    ", where, "skipped windows: 0 -> []")
    assert sweep_skipped_windows(log) == 0
    assert sweep_skipped_windows(LOGURU.format("INFO    ", where, "skipped windows: 1 -> [((0, 0), 'TT_THROW')]")) == 1
    assert sweep_skipped_windows("=== 1 failed in 9s ===") is None


def test_child_failure_names_the_throw():
    text = "E   RuntimeError: TT_THROW: trisc0 build failed. Log: /tmp/x.s: Assembler messages: (see)\n"
    text += "=== 1 failed in 9.00s ===\n"
    assert child_failure(text, 1) == "TT_THROW: trisc0 build failed. Log: /tmp/x.s: Assembler messages:"


def test_child_failure_device_timeout_before_throw():
    text = "Device 0: Timeout waiting for dispatch.\nTT_THROW: device timeout in fetch queue wait\n"
    assert child_failure(text, 1) == "Device 0: Timeout waiting for dispatch."


def test_child_failure_falls_back_to_pytest_summary():
    text = "FAILED tests/x.py::t[p] - AssertionError: non-deterministic outputs\n=== 1 failed in 9.00s ===\n"
    assert child_failure(text, 1) == "AssertionError: non-deterministic outputs"


def test_child_failure_killed_without_output():
    assert child_failure("", -9) == "killed by signal 9, no diagnostic line; check dmesg"
    assert child_failure("partial output\n", 135) == "exit code 135, no diagnostic line; check dmesg"


def test_read_tail_reads_only_the_end(tmp_path):
    log = tmp_path / "log"
    log.write_text("=== 1 passed in 9s ===\n" + "x" * 64 + "END\n")
    assert read_tail(log, 16).endswith("END\n") and "passed" not in read_tail(log, 16)


def _smi_chip(bus, **telem):
    base = {"ENABLED_GDDR": "0xff", "GDDR_UNCORR_ERRS": "0x0"}
    base.update({f"GDDR_{k}_{k + 1}_CORR_ERRS": "0x0" for k in (0, 2, 4, 6)})
    return {"board_info": {"bus_id": bus}, "smbus_telem": {**base, **telem}}


def test_gddr_edc_decodes_instance_bank_and_direction():
    snap = {
        "device_info": [
            _smi_chip("0000:81:00.0"),
            # instance 5 corrected writes 26 (bits 24-31), instance 4 corrected reads 3 (bits 0-7)
            _smi_chip("0000:c7:00.0", GDDR_4_5_CORR_ERRS="0x1a000003"),
            # instance 0 disabled, so instance 3 is bank 2; uncorrected write on instance 3 is bit 7
            _smi_chip("0000:87:00.0", ENABLED_GDDR="0xfe", GDDR_UNCORR_ERRS="0x80"),
        ]
    }
    assert gddr_edc(snap) == [
        GddrEdc("0000:c7:00.0", 4, 4, 3, 0, 0, 0),
        GddrEdc("0000:c7:00.0", 5, 5, 0, 26, 0, 0),
        GddrEdc("0000:87:00.0", 3, 2, 0, 0, 0, 1),
    ]


def test_umd_metal_ids():
    line = "UMD | Opening local chip ids/PCIe ids: {0, 1, 2, 3}/[0, 1, 3, 2] and remote chip ids {}"
    assert umd_metal_ids(line) == {0: 0, 1: 1, 3: 2, 2: 3}
    assert umd_metal_ids("nothing") == {}


def test_gddr_result_warns_on_corrected_and_fails_on_uncorrected():
    corr = ("tt-smi.json", GddrEdc("0000:c7:00.0", 5, 5, 0, 26, 0, 0))
    r = gddr_result(Result(Status.PASS, "1 passed"), [corr], {"0000:c7:00.0": 30})
    assert r.status == Status.WARN and r.reason == "1 passed; corrected GDDR EDC on 1 instance(s)"
    assert r.details == ["dev 30 (c7:00.0) GDDR 5 / bank 5: corrected rd/wr 0/26, uncorrected rd/wr 0/0 [tt-smi.json]"]
    uncorr = ("tt-smi_02.json", GddrEdc("0000:87:00.0", 3, 3, 0, 0, 1, 0))
    assert gddr_result(Result(Status.PASS), [corr, uncorr], {}).status == Status.FAIL
    assert gddr_result(Result(Status.HANG, "triage x"), [corr], {}).status == Status.HANG
    assert gddr_result(Result(Status.PASS, "ok"), [], {}) == Result(Status.PASS, "ok")


def test_loop_seconds_real_log():
    # testdata/iteration_loop.log: see testdata/SOURCES.txt; loop 12:04:27.134 -> 12:07:11.506
    seconds, iterations = loop_seconds(TESTDATA / "iteration_loop.log")
    assert round(seconds, 3) == 164.372 and iterations == 1000


def test_loop_seconds_real_nd_logs():
    # testdata/SOURCES.txt; duty 12:21:41.623 -> 12:23:20.973, child 12:44:09.915 -> 12:45:42.370,
    # pass 15:20:33.478 -> 15:22:09.995
    for name, expected in (
        ("nd_duty_dev16.log", (99.35, 10)),
        ("nd_write_ack_chip16.log", (92.455, 10)),
        ("nd_pass_duty.log", (96.517, 10)),
    ):
        seconds, iterations = loop_seconds(TESTDATA / name)
        assert (round(seconds, 3), iterations) == expected, name


def test_loop_seconds_without_iteration_lines(tmp_path):
    (tmp_path / "log").write_text("Device 27: Timeout waiting for dispatch\n")
    assert loop_seconds(tmp_path / "log") is None
    assert loop_seconds(tmp_path / "missing") is None
