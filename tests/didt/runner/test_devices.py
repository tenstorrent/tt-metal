# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tests.didt.runner.chips import Chip, by_dev
from tests.didt.runner.devices import (
    core_sweep_devices,
    nd_report_devices,
    run_to_run_devices,
    triage_hang_devices,
    write_ack_devices,
)

TESTDATA = Path(__file__).parent / "testdata"
TRAY_BUS_HIGH = {1: 0x0, 2: 0x4, 3: 0xC, 4: 0x8}
LOGURU = "2026-11-26 12:05:01.100 | {} | {} - {}\n"


def _galaxy(skip=()) -> dict[int, Chip]:
    rows = sorted(
        (f"0000:{(TRAY_BUS_HIGH[t] << 4) | a:02x}:00.0", t, a)
        for t in range(1, 5)
        for a in range(1, 9)
        if (t, a) not in skip
    )
    return by_dev([Chip(i, bdf, t, a, f"0x{t:02x}{a:014x}", (t - 1) * 8 + a - 1) for i, (bdf, t, a) in enumerate(rows)])


chips32 = _galaxy()
IDENTITY = ("dev", "bdf", "tray", "asic", "asic_id", "kmd")


def read(name: str) -> str:
    return (TESTDATA / name).read_text()


def test_triage_hang_devices_real_shape():
    d, warnings = triage_hang_devices(read("crash_triage_hang.csv"), chips32)
    assert [x["dev"] for x in d] == [27] and d[0]["tray"] == 3 and d[0]["asic"] == 4
    assert d[0]["bdf"] == "0000:c4:00.0" and d[0]["asic_id"] == chips32[27].asic_id and d[0]["kmd"] == 19
    assert d[0]["hung"] is True and d[0]["op"] == "MinimalMatmulDeviceOperation" and d[0]["core_count"] == 120
    # triage lists 5 cores, then "..."
    assert d[0]["cores"][0] == {"x": 0, "y": 8, "translated": [1, 10]}
    assert len(d[0]["cores"]) == 5
    assert warnings == []


def test_triage_tray_asic_mismatch_warns():
    text = read("crash_triage_hang.csv").replace("T3:N4 (Device 27)", "T3:N5 (Device 27)")
    d, warnings = triage_hang_devices(text, chips32)
    assert d[0]["tray"] == 3 and d[0]["asic"] == 4
    assert warnings == ["device 27: triage says T3:N5, chip snapshot says T3:N4 (0000:c4:00.0)"]


def test_triage_empty_or_other_sections_is_empty():
    assert triage_hang_devices("", chips32) == ([], [])
    assert triage_hang_devices("check_arc.py:\nDev,Status\n0,ok\n", chips32) == ([], [])


def test_triage_device_not_in_snapshot_keeps_dev_only():
    d, _ = triage_hang_devices(read("crash_triage_hang.csv"), _galaxy(skip={(t, a) for t in (3,) for a in range(1, 9)}))
    assert d[0]["dev"] == 27 and "bdf" not in d[0] and d[0]["hung"] is True


def test_nd_report_top_cores_and_pattern():
    devices, warnings = nd_report_devices(read("nd_report.log"), chips32)
    assert warnings == []
    assert [x["dev"] for x in devices] == [19, 26]
    d19, d26 = devices
    assert {k: d19[k] for k in IDENTITY} == chips32[19].as_dict()
    assert d19["failed_checks"] == 3 and d19["checks"] == 20 and d19["hung"] is False
    assert d19["pattern"] == "single core (6,8)"
    assert d19["cores"] == [
        {
            "x": 6,
            "y": 8,
            "translated": [7, 10],
            "failed_checks": 3,
            "signature": "face column lane 3 in 3/3 checks; 2-96 elements per check; bits 7:100%",
        }
    ]
    assert d26["pattern"].startswith("scattered (10 cores")
    assert len(d26["cores"]) == 8 and d26["cores"][0]["x"] == 0 and d26["cores"][0]["failed_checks"] == 2


def test_nd_report_bdf_mismatch_warns():
    text = read("nd_report.log").replace(
        "Chip map device 19: UBB4 ASIC4 /dev/tenstorrent/27 0000:84:00.0",
        "Chip map device 19: UBB4 ASIC5 /dev/tenstorrent/28 0000:85:00.0",
    )
    devices, warnings = nd_report_devices(text, chips32)
    d19 = devices[0]
    assert d19["dev"] == 19 and d19["bdf"] == "0000:85:00.0"
    # identity follows the log's bdf: that chip's tray/asic/asic_id/kmd from the snapshot
    assert (d19["tray"], d19["asic"], d19["asic_id"]) == (4, 5, chips32[20].asic_id)
    assert warnings == ["device 19: log says bdf 0000:85:00.0, chip snapshot says 0000:84:00.0; kept the log's"]


def test_nd_report_interval_zero_is_empty():
    assert nd_report_devices("=== 1 passed in 9s ===\n", chips32) == ([], [])


def test_nd_report_missing_chip_joins_on_snapshot_order():
    # tray 4 asic 2 (0000:82:00.0) absent: 0000:84:00.0 is dev 18, not 19
    chips31 = _galaxy(skip={(4, 2)})
    text = (
        read("nd_report.log")
        .replace("device 19", "device 18")
        .replace("device 18: UBB4 ASIC3", "device 99: UBB4 ASIC3")
    )
    devices, warnings = nd_report_devices(text, chips31)
    assert devices[0]["dev"] == 18 and devices[0]["bdf"] == "0000:84:00.0" and devices[0]["kmd"] == 27
    assert not [w for w in warnings if w.startswith("device 18")]


def test_write_ack_nd_chip_uses_parent_chip_id():
    # A child run with TT_VISIBLE_DEVICES=<chip> sees one chip, as device 0.
    child = "".join(
        line
        for line in read("nd_report.log")
        .replace("device 19", "device 0")
        .replace("0000:84:00.0", "0000:83:00.0")
        .splitlines(True)
        if "device 0" in line or "failed in" in line
    )
    devices, warnings = write_ack_devices({18: "ND chip 18 (runtime id 0): ..."}, {18: child}, chips32)
    assert [d["dev"] for d in devices] == [18]
    assert devices[0]["hung"] is False and devices[0]["bdf"] == "0000:83:00.0" and devices[0]["failed_checks"] == 3
    assert warnings == []


def test_write_ack_hung_chip():
    devices, _ = write_ack_devices({27: "HUNG chip 27 (runtime id 0): BusDeviceFunction=0000:c4:00.0"}, {}, chips32)
    assert devices == [{**chips32[27].as_dict(), "hung": True}]


def test_write_ack_other_failure_names_chip_only():
    devices, _ = write_ack_devices(
        {5: "no diagnostic line -- process killed (Bus error / OOM); check dmesg"}, {}, chips32
    )
    assert devices == [chips32[5].as_dict()]


def test_run_to_run_devices():
    where = "tests.didt.test_matmul_run_to_run_determinism:test_matmul_run_to_run_determinism:169"
    line = "  cores=110 (11x10 at 0,0): shard {s} (device id {d}), logical core ({x}, {y}) -> physical ({px},{py}): run-to-run NOT deterministic, {n} elements"
    text = (
        LOGURU.format("WARNING ", where, line.format(s=3, d=21, x=6, y=8, px=7, py=10, n=40))
        + LOGURU.format("WARNING ", where, line.format(s=3, d=21, x=6, y=8, px=7, py=10, n=2))
        + LOGURU.format("WARNING ", where, line.format(s=3, d=21, x=1, y=0, px=2, py=2, n=100))
        + LOGURU.format("WARNING ", where, line.format(s=5, d=5, x=0, y=0, px=1, py=2, n=1))
    )
    devices = run_to_run_devices(text, chips32)
    assert [d["dev"] for d in devices] == [5, 21]
    assert devices[1]["cores"] == [
        {"x": 1, "y": 0, "translated": [2, 2], "elements": 100},
        {"x": 6, "y": 8, "translated": [7, 10], "elements": 42},
    ]
    assert devices[1]["bdf"] == chips32[21].bdf and devices[1]["hung"] is False


def test_core_sweep_devices():
    where = "tests.didt.scratch_core_sweep:test_core_sweep:108"
    text = (
        LOGURU.format("WARNING ", where, "  window 11x10 at (0,0): device 21 NOT deterministic, 96 elements")
        + LOGURU.format("WARNING ", where, "  window 11x10 at (0,0): device 3 NOT deterministic, 1 elements")
        + LOGURU.format("WARNING ", where, "  window 1x1 at (6,8): device 21 NOT deterministic, 4 elements")
        + LOGURU.format("INFO    ", where, "skipped windows: 0 -> []")
    )
    devices = core_sweep_devices(text, chips32)
    assert [d["dev"] for d in devices] == [3, 21]
    assert devices[1]["elements"] == 100 and devices[1]["cores"] == [{"x": 6, "y": 8, "elements": 4}]
    assert "cores" not in devices[0]
