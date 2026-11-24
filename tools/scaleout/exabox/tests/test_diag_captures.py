#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""diag/captures.py and the BMC reads under it, on a fake ipmitool and a fake sysfs: no BMC, no chip."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SUITE_DIR = Path(__file__).resolve().parent.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag import aer, captures, hsc  # noqa: E402

# Answers `raw 0x06 0x52 <bus> <addr> <n> <code>` with zeros, `raw 0x04 0x2d <n>` with 10 and `sel elist` with a row.
FAKE = r"""#!/usr/bin/env python3
import sys
a = sys.argv[1:]
if a[:1] == ["sel"]:
    print("1 | SEL entry")
elif a[:3] == ["raw", "0x06", "0x52"] and int(a[5], 16):
    print(" ".join(["00"] * int(a[5], 16)))
elif a[:3] == ["raw", "0x04", "0x2d"]:
    print("0a 00")
"""


def _args(**kw) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    captures.add_args(ap)
    return argparse.Namespace(**{**vars(ap.parse_args([])), **kw})


class Fake:
    """Bmc.raw in-process from regs ("<bus> <addr> <code>" -> bytes, None = no answer; zeros otherwise)."""

    def __init__(self, case: unittest.TestCase):
        tmp = tempfile.TemporaryDirectory()
        case.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)
        self.ipmi = self.d / "ipmitool"
        self.ipmi.write_text(FAKE)
        self.ipmi.chmod(0o755)
        self.regs, self.calls = {}, []
        self.pci = self.d / "pci"
        self.pci.mkdir()

    def raw(self, _bmc, *a, lock=True):
        self.calls.append("raw " + " ".join(f"{x:#04x}" for x in a))
        if a[:2] != (0x06, 0x52):
            return [10, 0]
        v = self.regs.get(f"{a[2]:#04x} {a[3]:#04x} {a[5 if len(a) > 5 else 4]:#04x}", " ".join(["00"] * a[4]))
        return None if v is None else [int(x, 16) for x in v.split()]

    def set(self, tray, addr, code, value):
        self.regs[f"{hsc.HSC_BUS[tray]:#04x} {addr:#04x} {code:#04x}"] = value

    def chip(self, bdf, nonfatal=0):
        c = self.pci / bdf
        c.mkdir(exist_ok=True)
        (c / "vendor").write_text("0x1e52\n")
        (c / "config").write_bytes(b"\x52\x1e\x00\x00")
        for f, n in (("aer_dev_correctable", 0), ("aer_dev_nonfatal", nonfatal), ("aer_dev_fatal", 0)):
            (c / f).write_text(f"TOTAL_ERR_X {n}\n")

    def raw_patch(self):
        return mock.patch.object(hsc.Bmc, "raw", lambda b, *x, lock=True: self.raw(b, *x, lock=lock))

    def run(self, event, step="busbar", status="-", **kw):
        a = _args(ipmitool=str(self.ipmi), i2c_lock=str(self.d / "lock"), **kw)
        with mock.patch.object(captures, "PCI", self.pci), mock.patch.object(captures, "_poll_start"), self.raw_patch():
            return captures.run(event, step, status, self.d / "out", a)

    def clears(self):
        return [c for c in self.calls if c.endswith(" 0x00 0x03")]


class TestHscJudge(unittest.TestCase):
    def setUp(self):
        self.f = Fake(self)
        for t in hsc.HSC_BUS:
            for a in hsc.HSC_ADDRS:
                self.f.set(t, a, 0x88, "80 01")  # 12.0 V
                self.f.set(t, a, 0x4A, "db 00")  # warn limit 219 A
                self.f.set(t, a, 0x46, "e4 00")  # trip 228 A

    def test_weakest_feed_warns_and_the_sag_line_lists_every_feed(self):
        self.f.run("stage-start")
        now, rows = time.time() + 1, []
        for t in hsc.HSC_BUS:
            for a in hsc.HSC_ADDRS:
                loaded = 382 if (t, a) == (4, 0x40) else 390  # 11.94 V against 12.19 V
                rows += [f"{now:.0f}\t{t}\t{a:#04x}\t390\t160"] * 3 + [f"{now:.0f}\t{t}\t{a:#04x}\t{loaded}\t720"] * 3
        (self.f.d / "out" / "captures" / "vin.tsv").write_text("\n".join(rows) + "\n")
        with mock.patch.object(captures, "log") as log:
            out = self.f.run("stage-end")
        self.assertEqual(
            out,
            [
                "HSC WARN tray 4 (81:00.0-88:00.0) HSC 0x40: weakest feed during busbar: VIN median 11.94 V at IOUT >= 40 A, "
                "0.25 V under 12.19 V, the median of the other 7 feed(s) (WARN at >= 0.25 V); WARN only, not a finding"
            ],
        )
        said = [c.args[0] for c in log.call_args_list]
        self.assertIn(
            "HSC sag after busbar: R max mOhm at IOUT >= 40 A t1/0x40 0.0 t1/0x42 0.0 t2/0x40 0.0 t2/0x42 0.0 "
            "t3/0x40 0.0 t3/0x42 0.0 t4/0x40 5.6 t4/0x42 0.0 (WARN over 13.0, information over 11.0)",
            said,
        )

    def test_new_latch_fails_once_naming_the_feed_and_a_stage_end_alone_judges_against_the_run_start(self):
        self.f.run("stage-start")
        self.f.set(2, 0x42, 0x79, "40 00")
        self.f.set(4, 0x40, 0x7C, "20")
        out = self.f.run("stage-end")
        self.assertEqual(len(out), 2, out)
        self.assertIn("HSC FAIL tray 2 (41:00.0-48:00.0) HSC 0x42: OFF set during busbar", out[0])
        self.assertTrue(
            out[1].startswith("HSC FAIL tray 4 (81:00.0-88:00.0) HSC 0x40: VIN_UV_WARN latched during busbar")
        )
        self.assertEqual(self.f.run("stage-end", "C9"), [])  # already named
        self.f.set(1, 0x40, 0x7C, "10")
        self.assertEqual(
            self.f.run("stage-end", "C9"),
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
        out = " | ".join(self.f.run("stage-end"))
        self.assertIn("HSC WARN tray 1 (01:00.0-08:00.0) HSC 0x42: IOUT peak 220.0 A", out)
        self.assertIn("margin 8.0 A", out)

    def test_clear_faults_opt_in_once_and_never_with_an_hsc_off(self):
        self.f.run("stage-start")
        self.assertFalse(self.f.clears())  # off by default
        f2 = Fake(self)
        f2.run("reset", "C2", hsc_clear=True)
        f2.run("stage-start", hsc_clear=True)
        self.assertEqual(len(f2.clears()), 8)
        f3 = Fake(self)
        f3.set(2, 0x40, 0x79, "40 00")
        f3.run("reset", "C2", hsc_clear=True)
        self.assertFalse(f3.clears())

    def test_no_route_reads_nothing(self):
        a = _args(ipmitool="/nonexistent/ipmitool")
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(captures.run("stage-start", "busbar", "-", t, a), [])
            self.assertEqual(captures.run("stage-end", "busbar", "-", t, a), [])


class TestBricks(unittest.TestCase):
    def test_vout_by_each_bricks_own_mode_and_the_mux_restored(self):
        f = Fake(self)
        f.regs.update(
            {"0x09 0xec 0x01": "05", "0x09 0xc0 0x20": "16", "0x09 0xc2 0x20": "14"}
        )  # mux byte, 2^-10, 2^-12
        f.regs.update({"0x09 0xc0 0x8b": "00 30", "0x09 0xc2 0x8b": "00 c0"})  # 12288, 49152
        with f.raw_patch():
            rows, notes = hsc.brick_pass(hsc.Bmc(str(f.ipmi), str(f.d / "lock")))
        v = {r[0]: r[1] for r in rows}
        self.assertEqual((v["U1"], v["U2"]), (12.0, 12.0))
        self.assertIn("fan rail current not read: SHUNT_CAL unset", notes)
        mux = [c.split()[-1] for c in f.calls if c.startswith("raw 0x06 0x52 0x09 0xec 0x00")]
        self.assertEqual(mux, ["0x01", "0x05", "0x02", "0x05", "0x04", "0x05", "0x08", "0x05"])

    def test_no_mux_write_without_the_flag_and_a_site_row(self):
        f = Fake(self)
        f.run("stage-start")
        site = f.d / "site.tsv"
        site.write_text("# rows\nbrick_mux other-host why\n")
        f.run("stage-start", brick_mux=True, site_data=str(site))
        self.assertFalse([c for c in f.calls if " 0x09 0xec " in c])
        site.write_text("brick_mux * every box\n")
        f.run("stage-start", brick_mux=True, site_data=str(site))
        self.assertTrue([c for c in f.calls if " 0x09 0xec " in c])


class TestEvidence(unittest.TestCase):
    def test_stage_aer_drop_dump_and_cpld_not_installed(self):
        f = Fake(self)
        f.chip("0000:81:00.0")
        f.chip("0000:02:00.0")
        f.run("stage-start")
        f.chip("0000:81:00.0", nonfatal=1)
        shutil.rmtree(f.pci / "0000:02:00.0")
        with mock.patch.object(captures, "log") as log:
            f.run("after", "mix", "fail")
        said = " ".join(c.args[0] for c in log.call_args_list)
        for s in ("chips lost during mix: 02:00.0 (tray 1)", "CPLD dump: not installed", "SEL saved:"):
            self.assertIn(s, said)
        self.assertTrue(list((f.d / "out" / "captures").glob("drop_*_tray1.txt")))
        self.assertIn("AER FAIL 81:00.0 (tray 4): nonfatal +1 during busbar", f.run("stage-end", "busbar"))

    def test_vr_pg_fall_two_polls_in_a_row(self):
        d = Fake(self).d
        rows = ["100 400 400 400 400 50 ff ff ff ff", "102 400 400 400 400 50 ff 7f ff ff"]
        rows += ["104 400 400 400 400 50 ff ERR ff ff", "106 400 400 400 400 50 ff 7f ff ff"]
        rows += ["108 400 400 400 400 50 3f ff ff ff", "110 400 400 400 400 50 ff ff ff ff"]
        (d / "power.tsv").write_text("\n".join(rows) + "\n")
        out = captures._power_lines(d, {"t0": 0}, "busbar", {})
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("12V FAIL tray 2 (41:00.0-48:00.0): vr_pg 7f from"), out)
        self.assertEqual(captures._power_lines(d, {"t0": 0}, "busbar", {"resets": [105]}), [])  # near a reset
        self.assertEqual(captures.vr_pg_fell(d, 0), {2: ("102", "7f")})  # what stops a running part
        (d / "state.json").write_text(json.dumps({"resets": [105]}))
        self.assertEqual(captures.vr_pg_fell(d, 0), {})

    def test_load_line_from_the_power_poll(self):
        d = Path(tempfile.mkdtemp())
        psu_air = "900 900 900 900 25 41"
        rows = [
            f"100 400 410 420 430 1600 ff ff ff ff {psu_air}",
            f"102 600 610 620 630 2400 ff ff ff ff {psu_air[:-2]}47",
        ]
        (d / "power.tsv").write_text("\n".join(rows) + "\n")
        (d / "chips.tsv").write_text("101\t0000:41:00.0\t180.5\t200.0\n103\t0000:01:00.0\t90.0\t95.0\n")
        self.assertEqual(
            captures.load_line(d, 0, "mix"),
            "LOAD mix: tray sum mean 2.06 kW, peak 2.46 kW; tray 1/2/3/4 mean 500/510/520/530 W; outlet air peak 47 C; "
            "peak chip 180 W (0000:41:00.0, tray 2)",
        )
        self.assertEqual((d / "load.txt").read_text().count("LOAD mix"), 1)
        self.assertEqual(captures.load_line(d, 200, "mix"), "")


class TestVcore(unittest.TestCase):
    def test_sampler_reads_each_chip_until_its_stop(self):
        f = Fake(self)
        f.chip("0000:01:00.0")
        h = f.pci / "0000:01:00.0" / "hwmon" / "hwmon3"
        h.mkdir(parents=True)
        (h / "in0_input").write_text("790\n")
        c = f.pci / "0000:01:00.0" / "tenstorrent" / "tenstorrent!0"
        c.mkdir(parents=True)
        (c / "tt_aiclk").write_text("1350\n")
        threading.Timer(0.35, (f.d / "vcore.stop").touch).start()
        with mock.patch.object(captures, "PCI", f.pci):
            captures.vcore(f.d, os.getpgrp(), _args(vcore_s=0.01))  # under the 0.1 s floor: 0.1 s
        rows = (f.d / "vcore.tsv").read_text().splitlines()
        self.assertTrue(2 <= len(rows) <= 5, rows)
        self.assertEqual(rows[0].split("\t")[1:], ["0000:01:00.0", "790", "1350"])
        self.assertEqual(rows[1].split("\t")[3], "")  # AICLK every 10th sample

    def test_report_lines(self):
        d = Path(tempfile.mkdtemp()) / "captures"
        d.mkdir()
        rows = [
            f"1.0\t0000:{b}:00.0\t{v}\t1350"
            for b, vs in (("01", (700, 800)), ("02", (780, 790)), ("81", (820, 840)))
            for v in vs
        ]
        (d / "vcore.tsv").write_text("\n".join(rows) + "\n")
        (d.parent / "findings.txt").write_text("NUMERIC FAIL 02:00.0 (tray 1): bad\n")
        self.assertEqual(
            captures.vcore_lines(d),
            [
                "NUMERIC VCORE 02:00.0 (tray 1): min 780 mV, mean 785 mV over 2 samples, numeric hit (numeric verify: heavy chunk)",
                "NUMERIC VCORE 2 chips with no numeric hit: min 700 mV, mean 790 mV (numeric verify: heavy chunk)",
                "VCORE tray 1 under load (heavy chunk; mean [min-max] mV @ AICLK MHz): 01:00.0 750 [700-800] @1350, "
                "02:00.0 785 [780-790] @1350",
                "VCORE tray 4 under load (heavy chunk; mean [min-max] mV @ AICLK MHz): 81:00.0 830 [820-840] @1350",
                "VCORE box under load: 3 chips, median of chip means 785 mV (range 750-830), median AICLK 1350 MHz",
                "VCORE HIGH 81:00.0 (tray 4): mean 830 mV under load, 45 mV over the box median 785 mV (flag at +30 mV), "
                "AICLK 1350; report only",
            ],
        )


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
        d = Fake(self).d
        rows = ["100\t0000:40:01.1\t0000:41:00.0\t3\t3", "160\t0000:40:01.1\t0000:41:00.0\t900\t900"]
        (d / "aer_min.tsv").write_text("\n".join([*rows, "160\t0000:80:01.1\t0000:81:00.0\t2\t2"]) + "\n")
        self.assertEqual(
            captures._aer_rate(d, 0, "busbar"),
            [
                "AER WARN weak PCIe link 41:00.0 (tray 2): root port 0000:40:01.1 corrected AER peak 900/min "
                "(over 500/min in 1 of 2 min, +903 in all) during busbar"
            ],
        )


class TestCli(unittest.TestCase):
    def test_sample_and_a_poller_that_stops(self):
        f = Fake(self)
        env = {**os.environ, "PYTHONPATH": str(SUITE_DIR)}
        flags = ["--ipmitool", str(f.ipmi), "--i2c-lock", str(f.d / "lock")]
        cli = lambda *a, **k: subprocess.run(
            [sys.executable, "-m", "diag.captures", *a, *flags], env=env, cwd=SUITE_DIR, **k
        )  # noqa: E731
        out = cli("sample", capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(out[0], f"#route ok {f.ipmi}")
        self.assertEqual(out[1].split("\t")[1:], ["1", "0x40", "0x88", "0"])
        o = f.d / "out"
        cli("stage-start", "busbar", "-", str(o), check=True)
        pid = json.loads((o / "captures" / "state.json").read_text())["poll_pid"]
        time.sleep(1.5)
        cli("stage-end", "busbar", "-", str(o), check=True)
        self.assertFalse(captures._alive(pid))
        vin = (o / "captures" / "vin.tsv").read_text().splitlines()
        self.assertTrue(vin and vin[0].split("\t")[1:4] == ["1", "0x40", "0"], vin[:1])
        self.assertEqual(len((o / "captures" / "power.tsv").read_text().splitlines()[0].split()), 17)


if __name__ == "__main__":
    unittest.main()
