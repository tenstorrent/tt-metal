# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
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


# chips[] of the run the real nd_*.log excerpts come from (testdata/SOURCES.txt)
c10u08 = by_dev([Chip(**c) for c in json.loads(read("chips_c10u08.json"))])


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


def test_nd_report_real_minimal():
    devices, warnings = nd_report_devices(read("nd_minimal_dev16.log"), c10u08)
    assert warnings == []
    (d,) = devices
    assert {k: d[k] for k in IDENTITY} == {
        "dev": 16,
        "bdf": "0000:81:00.0",
        "tray": 4,
        "asic": 1,
        "asic_id": "0x38ce2fee992a3907",
        "kmd": 24,
    }
    assert (d["failed_checks"], d["checks"], d["hung"]) == (2, 20, False)
    assert d["pattern"] == (
        "scattered (24 cores in 2 rows, 12 columns): "
        "(0,1) (1,1) (2,1) (3,1) (4,1) (5,1) (6,1) (7,1) (8,1) (9,1) (10,1) (11,1) +12 more"
    )
    # 24 cores failed; the log details 8 of them
    assert [(c["x"], c["y"], c["translated"], c["failed_checks"]) for c in d["cores"]] == [
        (0, 1, [1, 3], 1),
        (1, 1, [2, 3], 1),
        (2, 1, [3, 3], 1),
        (3, 1, [4, 3], 1),
        (4, 1, [5, 3], 1),
        (5, 1, [6, 3], 1),
        (6, 1, [7, 3], 1),
        (7, 1, [10, 3], 1),
    ]
    assert d["cores"][0]["signature"] == (
        "whole tiles in 1/1 checks; most common face column 4 (1/1 checks); 7601-7601 elements per check; "
        "bits 3:16% 2:16% 4:14%"
    )
    assert d["cores"][2]["signature"] == "no structure"


def test_nd_report_real_duty():
    # This log also holds log_nd_cores' per-check "ND device 16 [...] pattern:" line, which is not a summary.
    devices, warnings = nd_report_devices(read("nd_duty_dev16.log"), c10u08)
    assert warnings == []
    (d,) = devices
    assert {k: d[k] for k in IDENTITY} == c10u08[16].as_dict()
    assert (d["failed_checks"], d["checks"], d["pattern"]) == (3, 10, "rectangle x=0..11 y=0..8 (108 cores)")
    assert [(c["x"], c["y"], c["failed_checks"]) for c in d["cores"]] == [(x, 1, 3) for x in range(8)]
    assert d["cores"][3]["signature"] == (
        "row strips in 1/1 checks; most common face column 5 (1/1 checks); 255669-255669 elements per check; "
        "bits 13:10% 12:10% 11:8%"
    )


def test_nd_report_interval_zero_is_empty():
    assert nd_report_devices("=== 1 passed in 9s ===\n", c10u08) == ([], [])


def test_nd_report_real_pass_is_empty():
    # 32 Chip map lines and "ND report: 0 of 32", no summary
    assert nd_report_devices(read("nd_pass_duty.log"), c10u08) == ([], [])


def test_nd_report_bdf_mismatch_warns():
    text = read("nd_minimal_dev16.log").replace(
        "Chip map device 16: UBB4 ASIC1 /dev/tenstorrent/24 0000:81:00.0",
        "Chip map device 16: UBB4 ASIC2 /dev/tenstorrent/25 0000:82:00.0",
    )
    devices, warnings = nd_report_devices(text, c10u08)
    (d,) = devices
    # identity follows the log's bdf: that chip's tray/asic/asic_id/kmd from the snapshot
    assert d == {**d, **c10u08[17].as_dict(), "dev": 16}
    assert warnings == ["device 16: log says bdf 0000:82:00.0, chip snapshot says 0000:81:00.0; kept the log's"]


def test_nd_report_missing_chip_joins_on_snapshot_order():
    # 0000:48:00.0 absent: every later chip moves down one, so 0000:81:00.0 is dev 15
    kept = [c for c in c10u08.values() if c.bdf != "0000:48:00.0"]
    chips31 = by_dev([Chip(i, c.bdf, c.tray, c.asic, c.asic_id, c.kmd) for i, c in enumerate(kept)])
    text = "".join(
        line.replace("device 16", "device 15")
        for line in read("nd_minimal_dev16.log").splitlines(True)
        if "Chip map device 15:" not in line
    )
    devices, warnings = nd_report_devices(text, chips31)
    assert devices[0]["dev"] == 15 and devices[0]["bdf"] == "0000:81:00.0" and devices[0]["kmd"] == 24
    assert devices[0]["asic_id"] == c10u08[16].asic_id
    assert warnings == []


def test_write_ack_nd_chip_uses_parent_chip_id():
    # The child ran with TT_VISIBLE_DEVICES=16, saw one chip and logged it as device 0.
    why = "ND chip 16 (runtime id 0): BusDeviceFunction=0000:81:00.0 /dev/tenstorrent/24"
    devices, warnings = write_ack_devices({16: why}, {16: read("nd_write_ack_chip16.log")}, c10u08)
    (d,) = devices
    assert {k: d[k] for k in IDENTITY} == c10u08[16].as_dict()
    assert (d["failed_checks"], d["checks"], d["pattern"], d["hung"]) == (3, 10, "whole grid (120 cores)", False)
    assert d["cores"][0] == {"x": 0, "y": 0, "translated": [1, 2], "failed_checks": 3, "signature": "no structure"}
    assert d["cores"][7]["translated"] == [10, 2]
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
