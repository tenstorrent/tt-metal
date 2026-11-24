#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for diag/captures.py and the BMC reads under it, on a fake ipmitool and a fake sysfs: no BMC, no chip."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag import aer, captures, hsc  # noqa: E402

# The fake answers `raw 0x06 0x52 <bus> <addr> <n> <code> [data]` from regs.json ("<bus> <addr> <code>" -> bytes, null
# = no answer; unknown reads answer zeros), `raw 0x04 0x2d <n>` with 10 (x scale) and `sel elist`. Fake.raw answers the
# same in-process.
FAKE = r"""#!/usr/bin/env python3
import json, os, sys
d = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
open(os.path.join(d, "calls.log"), "a").write(" ".join(a) + "\n")
regs = json.load(open(os.path.join(d, "regs.json")))
if a[:1] == ["sel"]:
    print("1 | SEL entry")
elif a[:3] == ["raw", "0x06", "0x52"]:
    bus, addr, n = a[3], a[4], int(a[5], 16)
    if n:
        v = regs.get(f"{bus} {addr} {a[6] if len(a) > 6 else a[5]}", " ".join(["00"] * n))
        if v is None:
            sys.exit(1)
        print(v)
elif a[:3] == ["raw", "0x04", "0x2d"]:
    print("0a 00")
"""


def _args(**kw) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    captures.add_args(ap)
    return ap.parse_args([]) if not kw else argparse.Namespace(**{**vars(ap.parse_args([])), **kw})


class Fake:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.ipmi = self.d / "ipmitool"
        self.ipmi.write_text(FAKE)
        self.ipmi.chmod(0o755)
        self.regs: dict = {}
        self.save()
        self.pci = self.d / "pci"
        self.pci.mkdir()
        self.log: list[str] = []

    def raw(self, _bmc, *a, lock=True):
        """Bmc.raw in-process, as the fake binary answers (a process per read would make each test take seconds)."""
        self.log.append("raw " + " ".join(f"{x:#04x}" for x in a))
        if a[:2] == (0x06, 0x52):
            if not a[4]:
                return []
            v = self.regs.get(f"{a[2]:#04x} {a[3]:#04x} {a[5 if len(a) > 5 else 4]:#04x}", " ".join(["00"] * a[4]))
            return None if v is None else [int(x, 16) for x in v.split()]
        return [10, 0]

    def bmc(self):
        return hsc.Bmc(str(self.ipmi), str(self.d / "lock"))

    def save(self):
        (self.d / "regs.json").write_text(json.dumps(self.regs))

    def set(self, tray, addr, code, value):
        self.regs[f"{hsc.HSC_BUS[tray]:#04x} {addr:#04x} {code:#04x}"] = value
        self.save()

    def calls(self) -> list[str]:
        return self.log

    def chip(self, bdf, nonfatal=0):
        c = self.pci / bdf
        c.mkdir(exist_ok=True)
        (c / "vendor").write_text("0x1e52\n")
        (c / "config").write_bytes(b"\x52\x1e\x00\x00")
        for f, n in (("aer_dev_correctable", 0), ("aer_dev_nonfatal", nonfatal), ("aer_dev_fatal", 0)):
            (c / f).write_text(f"TOTAL_ERR_X {n}\n")

    def run(self, event, step="busbar", status="-", **kw):
        a = _args(ipmitool=str(self.ipmi), i2c_lock=str(self.d / "lock"), **kw)
        with mock.patch.object(captures, "PCI", self.pci), mock.patch.object(
            captures, "_poll_start"
        ), mock.patch.object(hsc.Bmc, "raw", lambda b, *x, lock=True: self.raw(b, *x, lock=lock)):
            return captures.run(event, step, status, self.d / "out", a)


class TestHscJudge(unittest.TestCase):
    def setUp(self):
        self.f = Fake()
        self.addCleanup(self.f.tmp.cleanup)
        for t in hsc.HSC_BUS:
            for a in hsc.HSC_ADDRS:
                self.f.set(t, a, 0x88, "80 01")  # 12.0 V
                self.f.set(t, a, 0x4A, "db 00")  # warn limit 219 A
                self.f.set(t, a, 0x46, "e4 00")  # trip 228 A

    def test_new_uv_latch_fails_and_names_the_feed(self):
        self.f.run("stage-start")
        self.f.set(4, 0x40, 0x7C, "20")
        out = self.f.run("stage-end")
        self.assertEqual(len(out), 1, out)
        self.assertTrue(
            out[0].startswith("HSC FAIL tray 4 (81:00.0-88:00.0) HSC 0x40: VIN_UV_WARN latched during busbar"), out
        )

    def test_latch_is_named_once_and_c9_judges_against_the_run_start(self):
        self.f.run("stage-start")
        self.f.set(2, 0x42, 0x79, "40 00")
        self.assertIn("OFF set during busbar", self.f.run("stage-end")[0])
        self.assertEqual(self.f.run("stage-end", "C9"), [])  # already named
        self.f.set(1, 0x40, 0x7C, "10")
        out = self.f.run("stage-end", "C9")  # no stage-start: the run-start read is the before
        self.assertEqual(
            out,
            [
                "HSC FAIL tray 1 (01:00.0-08:00.0) HSC 0x40: VIN_UV_FAULT latched during C9 "
                "(clear before; STATUS_INPUT 0x00 -> 0x10, STATUS_WORD 0x0000 -> 0x0000)"
            ],
        )

    def test_pre_existing_latch_is_one_host_warn(self):
        self.f.set(3, 0x40, 0x7C, "20")
        out = self.f.run("stage-start")
        self.assertEqual(len(out), 1)
        self.assertRegex(out[0], r"^HSC WARN host: tray 3 .* HSC 0x40: VIN_UV_WARN .*pre-existing, not from this run$")

    def test_peak_at_warn_limit_warns(self):
        self.f.run("stage-start")
        self.f.set(1, 0x42, 0xA6, "c0 0d")  # 3520 x 1/16 = 220 A
        out = self.f.run("stage-end")
        self.assertTrue(
            any(
                "HSC WARN tray 1 (01:00.0-08:00.0) HSC 0x42: IOUT peak 220.0 A" in ln and "margin 8.0 A" in ln
                for ln in out
            ),
            out,
        )

    def test_clear_faults_opt_in_once_and_never_with_an_hsc_off(self):
        self.f.run("stage-start")
        self.assertFalse([c for c in self.f.calls() if c.endswith(" 0x00 0x03")])  # off by default
        f2 = Fake()
        self.addCleanup(f2.tmp.cleanup)
        f2.run("reset", "C2", hsc_clear=True)
        f2.run("stage-start", hsc_clear=True)
        clears = [c for c in f2.calls() if c.endswith(" 0x00 0x03")]
        self.assertEqual(len(clears), 8)
        f3 = Fake()
        self.addCleanup(f3.tmp.cleanup)
        f3.set(2, 0x40, 0x79, "40 00")
        f3.run("reset", "C2", hsc_clear=True)
        self.assertFalse([c for c in f3.calls() if c.endswith(" 0x00 0x03")])

    def test_no_route_reads_nothing(self):
        a = _args(ipmitool="/nonexistent/ipmitool")
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(captures.run("stage-start", "busbar", "-", t, a), [])
            self.assertEqual(captures.run("stage-end", "busbar", "-", t, a), [])


class TestBricks(unittest.TestCase):
    def test_vout_by_each_bricks_own_mode_and_the_mux_restored(self):
        f = Fake()
        self.addCleanup(f.tmp.cleanup)
        f.regs["0x09 0xec 0x01"] = "05"  # the mux's control byte before
        f.regs["0x09 0xc0 0x20"] = "16"  # VOUT_MODE 2^-10
        f.regs["0x09 0xc2 0x20"] = "14"  # VOUT_MODE 2^-12
        f.regs["0x09 0xc0 0x8b"] = "00 30"  # 12288
        f.regs["0x09 0xc2 0x8b"] = "00 c0"  # 49152
        f.save()
        with mock.patch.object(hsc.Bmc, "raw", lambda b, *x, lock=True: f.raw(b, *x, lock=lock)):
            rows, notes = hsc.brick_pass(f.bmc())
        v = {r[0]: r[1] for r in rows}
        self.assertEqual((v["U1"], v["U2"]), (12.0, 12.0))
        self.assertIn("fan rail current not read: SHUNT_CAL unset", notes)
        mux = [c for c in f.calls() if c.startswith("raw 0x06 0x52 0x09 0xec 0x00")]
        self.assertEqual([c.split()[-1] for c in mux], ["0x01", "0x05", "0x02", "0x05", "0x04", "0x05", "0x08", "0x05"])

    def test_no_mux_write_without_the_flag_and_a_site_row(self):
        f = Fake()
        self.addCleanup(f.tmp.cleanup)
        f.run("stage-start")
        site = f.d / "site.tsv"
        site.write_text("# rows\nbrick_mux other-host why\n")
        f.run("stage-start", brick_mux=True, site_data=str(site))
        self.assertFalse([c for c in f.calls() if " 0x09 0xec " in c])
        site.write_text("brick_mux * every box\n")
        f.run("stage-start", brick_mux=True, site_data=str(site))
        self.assertTrue([c for c in f.calls() if " 0x09 0xec " in c])


class TestEvidence(unittest.TestCase):
    def test_stage_aer_drop_dump_and_cpld_not_installed(self):
        f = Fake()
        self.addCleanup(f.tmp.cleanup)
        f.chip("0000:81:00.0")
        f.chip("0000:02:00.0")
        f.run("stage-start")
        f.chip("0000:81:00.0", nonfatal=1)
        import shutil

        shutil.rmtree(f.pci / "0000:02:00.0")
        with mock.patch.object(captures, "log") as log:
            f.run("after", "mix", "fail")
        said = " ".join(c.args[0] for c in log.call_args_list)
        self.assertIn("chips lost during mix: 02:00.0 (tray 1)", said)
        self.assertIn("CPLD dump: not installed", said)
        self.assertIn("SEL saved:", said)
        self.assertTrue(list((f.d / "out" / "captures").glob("drop_*_tray1.txt")))
        out = f.f_end = f.run("stage-end", "busbar")
        self.assertIn("AER FAIL 81:00.0 (tray 4): nonfatal +1 during busbar", out)

    def test_vr_pg_fall_two_polls_in_a_row(self):
        d = Path(tempfile.mkdtemp())
        rows = [
            "100 400 400 400 400 50 ff ff ff ff",
            "102 400 400 400 400 50 ff 7f ff ff",
            "104 400 400 400 400 50 ff ERR ff ff",
            "106 400 400 400 400 50 ff 7f ff ff",
            "108 400 400 400 400 50 3f ff ff ff",
            "110 400 400 400 400 50 ff ff ff ff",
        ]
        (d / "power.tsv").write_text("\n".join(rows) + "\n")
        out = captures._power_lines(d, {"t0": 0}, "busbar", {})
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("12V FAIL tray 2 (41:00.0-48:00.0): vr_pg 7f from"), out)
        self.assertEqual(captures._power_lines(d, {"t0": 0}, "busbar", {"resets": [105]}), [])  # near a reset


class TestAerRate(unittest.TestCase):
    def test_kernel_events_native_and_firmware_first(self):
        lines = [
            "pcieport 0000:00:01.1: AER: Corrected error message received from 0000:41:00.0",
            "pcieport 0000:00:01.1: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID) 0000:41:00.0",
            "{3}[Hardware Error]: event severity: corrected",
            "{3}[Hardware Error]:   device_id: 0000:81:00.0",
            "{4}[Hardware Error]: event severity: fatal",
            "{4}[Hardware Error]:   device_id: 0000:81:00.0",
        ]
        self.assertEqual(aer.kernel_events(lines), {"0000:00:01.1": [1, 0], "0000:81:00.0": [1, 1]})

    def test_a_root_port_over_the_rate_warns(self):
        d = Path(tempfile.mkdtemp())
        rows = [
            "100\t0000:40:01.1\t0000:41:00.0\t3\t3",
            "160\t0000:40:01.1\t0000:41:00.0\t900\t900",
            "160\t0000:80:01.1\t0000:81:00.0\t2\t2",
        ]
        (d / "aer_min.tsv").write_text("\n".join(rows) + "\n")
        self.assertEqual(
            captures._aer_rate(d, 0, "busbar"),
            [
                "AER WARN weak PCIe link 41:00.0 (tray 2): root port 0000:40:01.1 corrected AER peak 900/min "
                "(over 500/min in 1 of 2 min, +903 in all) during busbar"
            ],
        )


class TestCli(unittest.TestCase):
    def test_sample_and_a_poller_that_stops(self):
        f = Fake()
        self.addCleanup(f.tmp.cleanup)
        env = {**os.environ, "PYTHONPATH": str(SUITE_DIR)}
        base = [sys.executable, "-m", "diag.captures"]
        flags = ["--ipmitool", str(f.ipmi), "--i2c-lock", str(f.d / "lock")]
        out = subprocess.run(
            [*base, "sample", *flags], capture_output=True, text=True, env=env, cwd=SUITE_DIR
        ).stdout.splitlines()
        self.assertEqual(out[0], f"#route ok {f.ipmi}")
        self.assertEqual(out[1].split("\t")[1:], ["1", "0x40", "0x88", "0"])
        o = f.d / "out"
        subprocess.run([*base, "stage-start", "busbar", "-", str(o), *flags], env=env, cwd=SUITE_DIR, check=True)
        pid = json.loads((o / "captures" / "state.json").read_text())["poll_pid"]
        time.sleep(1.5)
        subprocess.run([*base, "stage-end", "busbar", "-", str(o), *flags], env=env, cwd=SUITE_DIR, check=True)
        self.assertFalse(captures._alive(pid))
        vin = (o / "captures" / "vin.tsv").read_text().splitlines()
        self.assertTrue(vin and vin[0].split("\t")[1:4] == ["1", "0x40", "0"], vin[:1])
        self.assertEqual(len((o / "captures" / "power.tsv").read_text().splitlines()[0].split()), 17)


if __name__ == "__main__":
    unittest.main()
