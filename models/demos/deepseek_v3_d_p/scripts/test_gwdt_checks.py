# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

from gwdt_checks import (
    Status,
    check_smi,
    classify_loop,
    classify_pytest,
    collected_node_ids,
    didt_bad_devices,
    fabric_results,
    log_root,
    progress,
    run_name,
)


def _chip(fw="19.16.0.0", tdp="0x4b", key="TDP_LIMIT_MAX"):
    return {"firmwares": {"fw_bundle_version": fw}, "smbus_telem": {key: tdp}}


def test_smi_uniform_low_tdp_warns():
    r = check_smi({"device_info": [_chip()] * 32})
    assert r.status == Status.WARN
    assert "fw_bundle_version 19.16.0.0 (32/32)" in r.details
    assert "TDP_LIMIT_MAX 75 W (32/32)" in r.details
    assert "TDP 75 W < 130 W -> DS_PERF_IGNORE_POWER=1" in r.reason


def test_smi_uniform_high_tdp_passes():
    assert check_smi({"device_info": [_chip(tdp="0x96")] * 32}).status == Status.PASS


def test_smi_wrong_chip_count_fails():
    r = check_smi({"device_info": [_chip(tdp="0x96")] * 28})
    assert r.status == Status.FAIL
    assert r.reason == "found 28 chips, need 32"


def test_smi_mixed_fw_warns_with_chip_ids():
    chips = [_chip(tdp="0x96")] * 32
    chips[16] = _chip(fw="19.15.0.0", tdp="0x96")
    r = check_smi({"device_info": chips})
    assert r.status == Status.WARN
    assert "fw_bundle_version 19.16.0.0 chips 0-15,17-31" in r.details
    assert "fw_bundle_version 19.15.0.0 chips 16" in r.details


def test_smi_tdp_fallback_key():
    r = check_smi({"device_info": [_chip(tdp="0x96", key="SMBUS_TX_TDP_LIMIT_MAX")] * 32})
    assert "TDP_LIMIT_MAX 150 W (32/32)" in r.details


def test_pytest_skip_is_fail():
    r = classify_pytest("SKIPPED [1] x.py:1: golden trace not found\n=== 1 skipped in 3s ===", 1)
    assert r.status == Status.FAIL
    assert "skipped" in r.reason


def test_pytest_expected_count():
    assert classify_pytest("=== 2 passed in 9s ===", 2).status == Status.PASS
    assert classify_pytest("=== 1 passed in 9s ===", 2).status == Status.FAIL


def test_loop_triage_is_hang(tmp_path):
    (tmp_path / "log_01").write_text("forward_layer_3_start\nTEST_DONE_EXIT=137\n")
    (tmp_path / "crash_triage_01.csv").write_text("dev,core\n")
    r = classify_loop(tmp_path / "log_01", tmp_path / "crash_triage_01.csv")
    assert r.status == Status.HANG
    assert str(tmp_path / "crash_triage_01.csv") in r.reason


def test_loop_pass(tmp_path):
    (tmp_path / "log_01").write_text("=== 1 passed in 9s ===\nTEST_DONE_EXIT=0\n")
    assert classify_loop(tmp_path / "log_01", tmp_path / "none.csv").status == Status.PASS


def test_loop_signal_is_fail(tmp_path):
    (tmp_path / "log_01").write_text("TEST_DONE_EXIT=135\n")
    r = classify_loop(tmp_path / "log_01", tmp_path / "none.csv")
    assert r.status == Status.FAIL
    assert r.reason.startswith("SIGBUS")


def test_loop_determinism_reason(tmp_path):
    (tmp_path / "log_01").write_text(
        "[determinism] iter 1 KV cache DIFFERS from iter 0: 112 element(s), max abs delta 3.1e-02\n"
        "=== 1 failed in 9s ===\nTEST_DONE_EXIT=1\n"
    )
    r = classify_loop(tmp_path / "log_01", tmp_path / "none.csv")
    assert r.status == Status.FAIL
    assert "iter 1 KV cache DIFFERS" in r.reason


def test_didt_bad_devices():
    log = (
        "Number of non-deterministic outputs on device 3 is 0\n"
        "Number of non-deterministic outputs on device 17 is 2\n"
    )
    assert didt_bad_devices(log) == {17: 2}


def test_fabric_results():
    log = "PASSED x.py::test_fabric_opens[FABRIC_2D]\n" "FAILED x.py::test_fabric_opens[FABRIC_2D_TORUS_XY] - E\n"
    assert fabric_results(log) == {"FABRIC_2D": True, "FABRIC_2D_TORUS_XY": False}


def test_log_root_falls_back_when_not_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "access", lambda p, m: False)
    assert log_root("nobody", tmp_path) == tmp_path / "galaxy-workload-deployment-test"


def test_run_name():
    assert (
        run_name(datetime(2026, 10, 29, 3, 0), "bh-glx-110-c10u20", "70d7d6c")
        == "LOG_2026_10_29_03_00_bh-glx-110-c10u20_70d7d6c"
    )


def test_progress_forward():
    log = (
        "2026-10-29 03:00:00.000 | INFO | Building TtPrefillTransformer\n"
        "2026-10-29 03:48:10.000 | INFO | forward_layer_0_start\n"
        "iter 0 done (20 chunks) in 1.0 seconds\n"
        "forward_layer_33_end\n"
    )
    assert progress(log, 600, now=1000.0, mtime=996.0) == "iter 1/600  forward_layer_33_end  load 48m10s  idle 4s"


def test_progress_loading():
    log = "2026-10-29 03:00:00.000 | INFO | Building TtPrefillTransformer\nBuilding layer 12/61\n"
    assert progress(log, 600, now=10.0, mtime=10.0) == "iter 0/600  loading layer 12/61  idle 0s"


def test_collected_node_ids_tree_format():
    log = (
        "collected 4 items / 3 deselected / 1 selected\n"
        "<Dir tt-metal-3>\n"
        "            <Function test_fabric_opens[FABRIC_2D_TORUS_XY]>\n"
        "================= 1/4 tests collected (3 deselected) in 1.50s ==================\n"
    )
    assert collected_node_ids(log) == ["test_fabric_opens[FABRIC_2D_TORUS_XY]"]


def test_collected_node_ids_flat_format():
    assert collected_node_ids("a.py::test_x[p1]\na.py::test_x[p2]\n\n2 tests collected\n") == [
        "a.py::test_x[p1]",
        "a.py::test_x[p2]",
    ]


def test_collected_node_ids_ignores_not_found_error():
    assert collected_node_ids("collecting ... ERROR: not found: /x/a.py::test_y\n") == []
