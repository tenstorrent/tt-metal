# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The run report (diag/report.py) from recorded finding lines, no device: the tray lines and their causes, the final
verdict with its fault families, and the repair line with the fleet's count for its first action."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

SUITE_DIR = Path(__file__).resolve().parent.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.append(str(SUITE_DIR / "galaxy"))

from diag import events, report  # noqa: E402
from diag.model import FAIL  # noqa: E402

POWER = [
    "POWER DROP tray 4 (8 chips lost: 81:00.0 82:00.0 83:00.0 84:00.0 85:00.0 86:00.0 87:00.0 88:00.0) in the bus-bar "
    "stage quiet-burst 1-2/2, cycle 2/2 (2 s quiet then 2 s full load), BMC after: tray 1 ERR W, tray 4 ERR W",
    "PCIE FAIL 81:00.0 (tray 4) endpoint, root port 0000:80:01.1, host host-a: off PCIe (config space reads ffff); "
    "root port link 2.5GT/s x1, slot majority 16GT/s x1",
    "HSC FAIL tray 4 (81:00.0-88:00.0) HSC 0x40: VIN_UV_WARN+OFF latched during stage 4 bus-bar stage "
    "(STATUS_WORD 0x0000 -> 0x0048)",
]
LINK = [
    "PINPOINT LINK HANG 02:00.0 (tray 1) e6,7 <-> 06:00.0 (tray 1) e0,1: its 2-chip all-gather hung after the reset",
    "ETH WARN 01:00.0 (tray 1) pre: ETH ports [10, 11] enabled but not up, still down after 3 s",
]
NUMERIC = [
    "NUMERIC FAIL 88:00.0 (tray 4) chip 31 [heavy]: 3 mismatching chunks of 4, first in chunk 2, non-finite elements 0 "
    "(stage 5 numeric verify heavy chunk)",
]
HISTORY = "# family\tfixed\ttried\npower/feed\t3\t11\nlink\t3\t3\nmath\t6\t8\n"


def run(tmp_path: Path, findings: list[str], stages: dict | None = None, **kw) -> list[str]:
    (tmp_path / "findings.txt").write_text("\n".join(findings) + "\n")
    (tmp_path / "finding_stage.tsv").write_text("".join(f"{s}\t{l}\n" for l, s in (stages or {}).items()))
    (tmp_path / "history.tsv").write_text(HISTORY)
    args = argparse.Namespace(repair_history=tmp_path / "history.tsv", rerun="run_diag.sh", **kw)
    return report.lines(tmp_path, args)


def test_whole_tray_power_loss(tmp_path):
    out = run(tmp_path, POWER, {POWER[0]: "busbar"}, rc=8, verdict="FAIL (exit 8)")
    assert out[:4] == [
        "PASS tray 1",
        "PASS tray 2",
        "PASS tray 3",
        "FAIL tray 4 (3 findings)",
    ]  # BMC after: not tray 1's
    why = [l for l in out if l.startswith("    whole tray 4 |")][0]
    assert why.endswith(
        "cause: whole-tray power loss -> next: 1. Replace the bus bar set -> 2. if the tray still drops, replace tray 4"
    )
    assert any(l.startswith("  Family power/feed: ") and "tray 4" in l for l in out)  # the PCIe loss is the drop's
    assert not any(l.startswith("  Family PCIe") for l in out)
    assert out[-1] == (
        "Repair recommendation: replace the bus bar set (fixed 3 of 11 in fleet records); bus bar set / tray 12 V feed A of "
        "tray 4 (HSC 0x40 VIN_UV_WARN+OFF newly set during this run), then tray 4 board if it persists"
    )


def test_pcie_bandwidth_line_gives_a_pcie_cause(tmp_path):  # pcie_lines' own line, through the report
    from quiet_clock import PCIE_FLOOR_GBS, PCIE_WARN_FRAC, pcie_lines

    def row(b, h):
        return dict(chip=f"{b} (tray 2, ASIC 1, device id 0)", mb=32, h2d_gbs=h, d2h_gbs=20, bad="")

    rs = [row("41:00.0", 0.1), *(row(f"4{i}:00.0", 20) for i in (2, 3, 4))]
    line = pcie_lines(rs, PCIE_FLOOR_GBS, PCIE_WARN_FRAC, "h", sysfs="/nonexistent")[0]
    out = run(tmp_path, [line], rc=8, verdict="FAIL (exit 8)")
    assert out[1] == "FAIL tray 2 (1 finding)" and "cause: PCIe path (PCIE/LINK)" in out[2], out
    assert "Final verdict: FAIL: 1 fault: low PCIe bandwidth or wrong read back" in out, out
    assert any(
        x.startswith("  Fault 1: low PCIe bandwidth") and "where: tray 2 ASIC 1 / 41:00.0" in x for x in out
    ), out


def test_inbox_link_hang(tmp_path):
    out = run(tmp_path, LINK, rc=8, verdict="FAIL (exit 8)")
    assert out[0] == "FAIL tray 1 (1 finding, 1 warn)"
    assert out[1].startswith(
        "    tray 1 ASIC 2 / 02:00.0 [ETH ch 6,7] <-> tray 1 ASIC 6 / 06:00.0 [ETH ch 0,1] | stage 8 per-link sweep |"
    ) and out[1].endswith(
        "cause: link pair hung: cable, connector or an end chip -> next: reseat or replace the cable; triage "
        + out[1].rsplit("triage ", 1)[1]
    )
    assert "Final verdict: FAIL: 1 fault: PINPOINT LINK hang" in out
    assert any(l.startswith("  Family link: 1 fault on 2 chips, tray 1: data between chips over Ethernet") for l in out)
    assert out[-1] == (
        "Repair recommendation: reseat the cable or tray-to-tray connection (fixed 3 of 3 in fleet records); the cable or "
        "port of link tray 1 ASIC 2 / 02:00.0 [ETH ch 6,7] <-> tray 1 ASIC 6 / 06:00.0 [ETH ch 0,1]"
    )


def test_numeric_fault(tmp_path):
    out = run(tmp_path, NUMERIC, rc=8, verdict="FAIL (exit 8)", commit="abc123")
    assert out[3].startswith("FAIL tray 4 (1 finding)")
    assert "| stage 5 numeric verify heavy chunk |" in out[4] and "cause: chip compute fault: chip/board" in out[4]
    assert (
        "  Fault 1: wrong compute results (numeric check, heavy load): 3 chunks wrong of 4 | where: tray 4 ASIC 8 / "
        "88:00.0 (tt-smi chip 31) (stage 5 numeric verify heavy chunk) | do: hardware only if the same check on the same "
        "commit stays clean on clean hosts; then the tray 4 board carrying 88:00.0" in out
    )
    assert out[-1].startswith(
        "Repair recommendation: stored Vmin 760 mV (fixed 6 of 8 in fleet records); the tray 4 board "
    )
    assert "commit abc123" in out[-1]
    assert report.lines(tmp_path, argparse.Namespace(only="result")) == [
        "numeric fault 88:00.0 tray 4 (stage 5 numeric verify heavy chunk)"
    ]


def test_clean_run(tmp_path):
    out = run(tmp_path, LINK[1:], rc=0, verdict="PASS")
    assert out[0] == "PASS tray 1 (1 warn)" and out[2:5] == ["PASS tray 2", "PASS tray 3", "PASS tray 4"]
    assert out[-2:] == [
        "Final verdict: PASS: no fault found during this run",
        "Repair recommendation: none (all 4 trays pass)",
    ]


SEL = (
    "SEL-HISTORY WARN: tray 4: 3 BMC SEL entries of the last 30 days name its power path (info for the repair: "
    "not from this run)"
)
HSC_UNREAD = [
    f"HSC INCOMPLETE tray {t} ({b}) HSC {a}: STATUS_INPUT not read at the run start, comm fault (CML) set; its VIN_UV "
    "latch state is unknown, so this tray's 12 V feed was not judged"
    for t, b in ((1, "01:00.0-08:00.0"), (4, "81:00.0-88:00.0"))
    for a in ("0x40", "0x42")
]


def test_warn_colon_line_is_a_warn_not_a_failed_check(tmp_path):  # b02u08 leg B: 'RESULT: 1 failed check' with PASS
    out = run(tmp_path, [SEL], rc=0, verdict="PASS")
    assert out[3] == "PASS tray 4 (1 warn)"
    assert "RESULT: 1 failed check" not in "\n".join(out)


def test_unread_hsc_is_incomplete_never_pass(tmp_path):  # b02u08 leg B: STATUS_INPUT not read on tray 1 and 4
    out = run(tmp_path, HSC_UNREAD, rc=9, verdict="INCOMPLETE (exit 9)")
    assert out[0].startswith("INCOMPLETE tray 1 (power controller not read: 2 HSC finding(s)")
    i4 = next(i for i, l in enumerate(out) if l.startswith("INCOMPLETE tray 4 (power controller not read"))
    assert "PASS tray 2" in out and "PASS tray 3" in out and not any(l.startswith("FAIL") for l in out[: i4 + 1])
    assert any("12 V power controller (HSC) could not be read" in l for l in out)
    assert out[-1].startswith("Repair recommendation: none determined (INCOMPLETE (exit 9))")


def test_unread_hsc_with_exit_0_is_not_a_pass(tmp_path):  # the bash verdict said PASS (rc 0): the report must not
    out = run(tmp_path, HSC_UNREAD + [SEL], rc=0, verdict="PASS")
    assert out[-2].startswith("Final verdict: INCOMPLETE (not a fault): tray power controllers (HSC) not read on tray 1")
    assert "Final verdict: PASS" not in "\n".join(out)
    assert out[-1].startswith("Repair recommendation: none determined (INCOMPLETE")


def test_known_dropped_tray_hsc_unread_is_not_incomplete(tmp_path):
    out = run(tmp_path, HSC_UNREAD[2:], rc=0, verdict="PASS", known_tray="4")
    assert "KNOWN-DROP tray 4 (known dropped, not tested)" in out
    assert not any("INCOMPLETE" in l for l in out)


def test_host_wide_hsc_incomplete_marks_every_tray(tmp_path):
    host = "HSC INCOMPLETE host: no HSC read: ipmitool rc 124; tray 12 V under-voltage latches and min VIN not judged"
    out = run(tmp_path, [host], rc=0, verdict="PASS")
    assert [l for l in out if l.startswith(("INCOMPLETE tray", "PASS tray"))][:4] == [
        f"INCOMPLETE tray {t} (power controller not read: 1 HSC finding(s): 12 V feed not judged)" for t in range(1, 5)
    ]
    assert "Final verdict: PASS" not in "\n".join(out)
    assert "not read on tray 1" not in "\n".join(out)  # the host line says "tray 12 V", not tray 1


def test_incomplete_and_known_tray(tmp_path):
    out = run(
        tmp_path, POWER, rc=9, verdict="INCOMPLETE (exit 9)", known_tray="4", stopped="no load ran: discovery failed"
    )
    assert out[3] == "KNOWN-DROP tray 4 (known dropped, not tested)"
    assert out[0] == "INCOMPLETE tray 1 (no load ran: discovery failed)"
    assert out[-1].startswith("Repair recommendation: none determined (INCOMPLETE (exit 9)); rerun run_diag.sh")


def test_verdict_from_the_health_check_report(tmp_path):
    (tmp_path / "diag_report.json").write_text(json.dumps({"overall_status": "FAIL"}))
    out = report.lines(tmp_path, argparse.Namespace(repair_history=None))
    assert out == [
        "Final verdict: FAIL: no finding line names a part; see the lines above",
        "Repair recommendation: none determined (FAIL); fix the cause above and rerun the health check",
    ]


def test_repair_history_needs_no_file():
    fam = {}
    r = report.recommend(
        8,
        "FAIL",
        "FAIL tray 2 (1 finding)\n    whole tray 2 | x | y | cause: whole-tray power loss -> next: z",
        fam=fam,
    )
    assert report.lead(r, fam, report.history(None)).startswith("replace the bus bar set (no fleet records); bus bar")


CASES = [  # (finding lines, rc, verdict, texts the report holds)
    (
        [
            "ETH FAIL 44:00.0 (tray 2) c2: local ETH ports [4, 5] not up (to 84:00.0 (tray 4) port 4, 84:00.0 (tray 4) "
            "port 5); still down after the c2 reset; CONFIG: retimer/ETH speed x400G, expected x200G: tray 2 ASIC 4 / "
            "44:00.0 [ETH ch 4,5]"
        ],
        8,
        "FAIL (exit 8)",
        [
            "  Fault 1: CONFIG: retimer/ETH speed x400G, expected x200G: Ethernet link down, still down after the start (C2) "
            "reset | where: in-box link between tray 2 and tray 4 (no QSFP-DD port): tray 2 ASIC 4 / 44:00.0 [ETH ch 4,5] "
            "<-> tray 4 ASIC 4 / 84:00.0 [ETH ch 4,5] | do: set the retimers to x200G and rerun, before any cabling or "
            "board step",
            "Repair recommendation: set the retimers of link ",
        ],
    ),
    (
        [
            "ETH FAIL c2:00.0 (tray 3) c9: 1 QSFP cable missing or bad: c2:00.0<->82:00.0 (tray 3<->4) (layout subtorus)",
            "ETH FAIL 82:00.0 (tray 4) c2: local ETH ports [0, 1] not up (to c2:00.0 (tray 3) port 0, c2:00.0 (tray 3) "
            "port 1); still down after the c2 reset",
        ],
        8,
        "FAIL (exit 8)",
        [
            "Final verdict: FAIL: 1 fault: QSFP-DD cable missing or bad\n",
            "  Fault 1: QSFP-DD cable missing or bad: link down, still down after the start (C2) reset (cabling judged "
            "against layout subtorus) | where: tray 3 QSFP-DD port 4 / ASIC 2 / c2:00.0 <-> tray 4 QSFP-DD port 4",
        ],
    ),
    (
        [
            "PREFLIGHT FAIL 81:00.0 (tray 4) chip 24: tt_fw_bundle_ver 19.7.0.0, 24 of 32 chips 19.17.1.0",
            "PREFLIGHT FAIL 82:00.0 (tray 4) chip 25: tt_fw_bundle_ver 19.7.0.0, 24 of 32 chips 19.17.1.0",
        ],
        143,
        "INCOMPLETE (exit 143)",
        [
            "Final verdict: INCOMPLETE (not a fault): tray 4 runs firmware 19.7.0.0 while the other chips run 19.17.1.0; "
            "tt-umd discovery fails on mixed firmware; next: reflash tray 4 to 19.17.1.0, then retest",
            "    82:00.0 (tray 4) | preflight | PREFLIGHT FAIL: tt_fw_bundle_ver 19.7.0.0, 24 of 32 chips 19.17.1.0 | "
            "cause: firmware version differs from the other chips -> next: reflash tray 4 to the box's firmware bundle",
            "Repair recommendation: no hardware repair (firmware: reflash tray 4)",
        ],
    ),
    (
        [
            "THERMAL FAIL host: 1 SEL thermal trip event(s) during the run (8ba | 11/05/2026 | 08:45:49 | Temperature "
            "TEMP_PWRBRK1 | Upper Critical going high | Asserted | Reading 91 > Threshold 9)",
            "AER FAIL 41:00.0 (tray 2): 3 uncorrectable during the run",
        ],
        8,
        "FAIL (exit 8)",
        [
            "  Fault 1: power brick 1 overheated during the run (BMC sensor TEMP_PWRBRK1 read 91 C, above its upper critical"
            " limit) | where: power brick 1, one of the PDB's 54-to-12 V DC-DC modules",
            "do: comes with the power brick 1 overheat: fix brick 1 first and rerun; only if it repeats then, the PCIe path",
            "  Family thermal: 1 fault, the box: a part ran over its temperature limit",
        ],
    ),
    (
        [
            "CHIP DROP tray 4 (1 chips lost: 84:00.0) in the bus-bar stage 4",
            "CHIP DROP tray 4 (1 chips lost: 84:00.0) in the bus-bar stage step",
        ],
        8,
        "FAIL (exit 8)",
        ["the chips came back and dropped again: tray 4 in the bus-bar stage step (dropped 2 times)"],
    ),
    (
        [
            "POWER DROP tray 3 (8 chips lost: c1:00.0 c2:00.0 c3:00.0 c4:00.0 c5:00.0 c6:00.0 c7:00.0 c8:00.0) in the "
            "bus-bar stage 4",
            "RESET FAIL 0000:41:00.0 (tray 2) unopenable (C9 end reset): on PCIe with its tt-kmd node, but tt-umd cannot "
            "open it; in-band IPMI not answering",
        ],
        8,
        "FAIL (exit 8)",
        [
            "POWER-DOWNSTREAM of the power drop of tray 3: 1 chip of tray 2 stay on PCIe but do not open after the end (C9)"
            " reset",
            "cause: POWER-DOWNSTREAM: of the power drop of tray 3",
            "Repair recommendation: replace the bus bar set (no fleet records); bus bar set / tray 12 V feed of tray 3, then the"
            " tray 3 board if it still drops",
        ],
    ),
    (
        [
            "NUMERIC FAIL 04:00.0 (tray 1) chip 3 [dram]: 2/64 checked outputs mismatched (DRAM bank histogram [0, 2, 0, "
            "0, 0, 0, 0, 0])"
        ],
        8,
        "FAIL (exit 8)",
        [
            "  Family DRAM: 1 fault on 1 chip, tray 1: a chip's GDDR memory returned wrong data",
            "Repair recommendation: flash the current board firmware (no fleet records); the tray 1 board carrying 04:00.0",
        ],
    ),
    (
        [
            "HSC WARN tray 3 (c1:00.0-c8:00.0) HSC 0x40: marginal 12 V feed during stage 4 bus-bar stage: VIN min 11.06 V; "
            "WARN only, not a finding"
        ],
        0,
        "PASS",
        [
            "cause: marginal 12 V feed of tray 3, no trip or latch this run",
            "Repair recommendation: none (all 4 trays pass)",
        ],
    ),
]


CASES.append(  # the captures' over-current bit name
    (
        [
            "HSC FAIL tray 2 (41:00.0-48:00.0) HSC 0x42: OFF+IOUT_OC_FAULT set during busbar_mix (STATUS_WORD 0x0000 -> 0x0050)"
        ],
        8,
        "FAIL (exit 8)",
        [
            "cause: POWER: tray 2 12 V feed B / bus bar: HSC 0x42 OFF+IOUT_OC_FAULT newly set during this run",
            "Repair recommendation: replace the bus bar set (no fleet records); bus bar set / tray 12 V feed B of tray 2 (HSC "
            "0x42 OFF+IOUT_OC_FAULT newly set during this run), then tray 2 board if it persists",
        ],
    )
)


def test_cases(tmp_path):
    for i, (findings, rc, vt, want) in enumerate(CASES):
        d = tmp_path / str(i)
        d.mkdir()
        (d / "findings.txt").write_text("\n".join(findings) + "\n")
        out = "\n".join(report.lines(d, argparse.Namespace(rc=rc, verdict=vt, repair_history=None)))
        for w in want:
            assert w in out, (i, w, out)


def faults(tmp_path, name, findings):  # (Fault lines, all lines) of one run in its own dir
    d = tmp_path / name
    d.mkdir()
    out = run(d, findings, rc=1, verdict="FAIL")
    return [x for x in out if x.startswith("  Fault ")], out


def test_do_texts_unknown_bdf_name_the_board_not_tray_none(tmp_path):  # a BDF outside the tray map
    t = [
        "AER FAIL f1:00.0 (tray 1) root port 0000:f0:01.1, host X: uncorrectable nonfatal +1 fatal +0 during the run",
        "PCIE FAIL f2:00.0 (tray 1): off pcie",
        "PCIE FAIL f3:00.0 (tray 1): root port link 2.5GT/s x4 weak",
        "LINK FAIL one chip f4:00.0 (tray 1) tt-smi 3: all failed",
    ]
    o, out = faults(tmp_path, "a", t)
    assert len(o) == 4 and not any("tray None" in x for x in out), out
    assert o[0].endswith("then the board") and o[1].endswith("then the board or its PCIe slot"), o
    o, _ = faults(tmp_path, "b", [x.replace("f2:", "c2:") for x in t])
    assert o[1].endswith("then the tray 3 board or its PCIe slot"), o
    o, out = faults(
        tmp_path,
        "c",
        [
            "LINK FAIL f1:00.0 (tray 1) e0 <-> 41:00.0 (tray 2) e1: down",
            "LINK FAIL f1:00.0 (tray 1) e8 <-> f5:00.0 (tray 1) e9: down",
        ],
    )
    assert not any("None" in x or "tray ?" in x for x in out), out
    assert "reseat tray 2 and the tray holding f1:00.0;" in o[0] and "<-> f1:00.0 [ETH ch 0] |" in o[0], o
    assert (
        "between the tray holding f1:00.0 and the tray holding f5:00.0 (no QSFP-DD port): f1:00.0 [ETH ch 8] <->"
        in o[1]
    ), o
    eth = "ETH FAIL f1:00.0 (tray 1) c9: enabled ETH ports [0, 1] down before the run and still down after the c9 reset"
    o, _ = faults(tmp_path, "d", [eth])
    assert o[0].endswith("| where: f1:00.0 [ETH ch 0,1] | do: glx_reset and retest; if they stay down, the board"), o


def test_family_chain_is_pcie(tmp_path):  # ARC timeout -> AER on the root port -> MCE is one chip's PCIe path
    _, out = faults(
        tmp_path,
        "a",
        ["CHAIN FAIL 41:00.0 (tray 2): ARC timeout x2 -> AER on its root port x1 -> MCE x1 during the run"],
    )
    assert [x.split(":")[0] for x in out if x.startswith("  Family ")] == ["  Family PCIe"], out


def test_first_action_thermal_and_psu_get_no_prefix(tmp_path):  # no fleet records for these: the item names its step
    _, out = faults(tmp_path, "a", CASES[3][0][:1])
    assert out[-1].startswith("Repair recommendation: cooling: TEMP_PWRBRK1") and "fleet records" not in out[-1], out
    i = "the PSU named in the PSU FINDING line; the tray 3 board carrying 85:00.0"
    fam = {"the PSU named in the PSU FINDING line": None, "the tray 3 board carrying 85:00.0": "math"}
    assert report.lead(i, fam, report.history(None)) == i


def test_ethernet_bandwidth_link_finding(tmp_path):
    """A links_bandwidth low pair: one LINK FAIL line, its cause, verdict wording and repair line."""
    # the 0000:-prefixed BDFs are what events._check_lines emits from UMD discovery
    line = (
        "LINK FAIL 0000:c1:00.0 (tray 3) e16 <-> 0000:c2:00.0 (tray 3) e0: Ethernet bandwidth 118 Gbps per link "
        "each way, below 140 Gbps"
    )
    warn = "links WARN 02:00.0 (tray 1) e8: 1 pair not judged (under 2 warm launches)"
    out = run(tmp_path, [line, warn], {line: "C5", warn: "C5"}, rc=8, verdict="FAIL (exit 8)")
    assert any("Ethernet bandwidth 118 Gbps per link each way, below 140 Gbps" in l for l in out), out
    assert "Final verdict: FAIL: 1 fault: Ethernet link below bandwidth" in out
    repair = next(l for l in out if l.startswith("Repair recommendation:"))
    assert "reseat or replace the slow Ethernet link" in repair
    low = [l for l in out if "Ethernet link up and passing data, but below the Deployment Test's bandwidth" in l]
    assert low and "reseat the trays at both ends" in low[0]


def test_ethernet_bandwidth_events_lines(tmp_path):
    """A links_bandwidth check's data['failed'] renders as one LINK FAIL per low pair; ends and why carry through."""
    ends = [
        {
            "a": {"bdf": "0000:c1:00.0", "tray": 3, "channel": 16},
            "b": {"bdf": "0000:c2:00.0", "tray": 3, "channel": 0},
        }
    ]
    failed = [
        {"why": "Ethernet bandwidth 118 Gbps per link each way, below 140 Gbps", "ends": ends},
    ]
    report = {
        "phases": {
            "tests": {
                "checks": [{"name": "links_bandwidth", "status": FAIL, "details": "x", "data": {"failed": failed}}]
            }
        }
    }
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.object(events, "args", argparse.Namespace(dry_run=False, output=Path(d) / "r.json")):
            events.check_findings(report)
            got = (Path(d) / "finding_stage.tsv").read_text().splitlines()
    assert got == [
        "C5\tLINK FAIL 0000:c1:00.0 (tray 3) e16 <-> 0000:c2:00.0 (tray 3) e0: "
        "Ethernet bandwidth 118 Gbps per link each way, below 140 Gbps"
    ]


def test_link_failures_get_their_own_verdicts(tmp_path):
    """A hang in the warm rerun is still a LINK HANG; only a data mismatch is 'wrong data'."""
    end = lambda b, t, c: {"bdf": b, "tray": t, "channel": c}  # noqa: E731
    ends = [{"a": end("0000:c1:00.0", 3, 16), "b": end("0000:c2:00.0", 3, 0)}]
    whys = ["warm rerun hung", "rc 1", "no tt-kmd node on c1:00.0: not opened", "data mismatch"]
    lines = []
    for w in whys:
        data = {"failed": [{"why": w, "ends": ends}]}
        lines += events._check_lines({"name": "links_pair", "status": FAIL, "details": "x", "data": data})
    assert [l.split()[1] for l in lines] == ["HANG", "FAIL", "FAIL", "FAIL"], lines
    got = []
    for ln in lines:
        _, out = faults(tmp_path, f"l{len(got)}", [ln])
        got.append("\n".join(out))
    for g, what in zip(got, ["Ethernet link hung", "test error (rc 1)", "not opened", "wrong data"]):
        assert what in g, g
    assert "wrong data" not in "".join(got[:3]), got


def test_not_opened_link_names_the_missing_chip_not_the_cable(tmp_path):
    """A link whose chip has no tt-kmd node was never tested: the repair is that chip's PCIe path, not a cable."""
    cases = [
        (
            "no tt-kmd node on c1:00.0: not opened",
            "tray 3 ASIC 1 / c1:00.0",
            "the PCIe path (slot, retimer) of c1:00.0, then the tray 3 board",
        ),
        (
            "warm rerun no tt-kmd node on c1:00.0: not opened",
            "tray 3 ASIC 1 / c1:00.0",
            "the PCIe path (slot, retimer) of c1:00.0, then the tray 3 board",
        ),
        (
            "no tt-kmd node on c1:00.0 and c2:00.0: not opened",
            "tray 3 ASIC 1 / c1:00.0 and tray 3 ASIC 2 / c2:00.0",
            "the PCIe path (slot, retimer) of c1:00.0 and c2:00.0, then the tray 3 board",
        ),
        (
            "no tt-kmd node on 41:00.0 and c1:00.0: not opened",
            "tray 2 ASIC 1 / 41:00.0 and tray 3 ASIC 1 / c1:00.0",
            "the PCIe path (slot, retimer) of 41:00.0 and c1:00.0, then the tray 2 board and the tray 3 board",
        ),
        (
            "no tt-kmd node on aa:00.0: not opened",
            "aa:00.0",
            "the PCIe path (slot, retimer) of aa:00.0, then the board",
        ),
    ]
    for i, (why, named, repair) in enumerate(cases):
        peer = "0000:aa:00.0" if "aa:00.0" in why else "0000:c1:00.0"
        _, out = faults(tmp_path, f"n{i}", [f"LINK FAIL {peer} (tray 3) e16 <-> 0000:c2:00.0 (tray 3) e0: {why}"])
        text = "\n".join(out)
        assert any(f"Ethernet link not opened (no tt-kmd node on {named})" in x for x in out if "Fault" in x), text
        assert "Family PCIe" in text, text
        assert f"Repair recommendation: {repair}" in text, text
        assert not re.search(
            r"reseat|cable", "\n".join(l for l in out if l.startswith(("Repair", "  Fault", "    tray")))
        ), text
