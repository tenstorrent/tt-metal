#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""galaxy/: the Blackhole galaxy device programs, without a device.

Each program is started as a script from galaxy/ or imported by its flat name with
galaxy/ on sys.path, so every import it makes must be a galaxy/ module, the standard
library, or one of the device stack's own packages. The modules that need no device
package at import time must import cleanly here.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
GALAXY_DIR = TESTS_DIR.parent / "health_check_test_suite" / "galaxy"
sys.path.insert(0, str(GALAXY_DIR))

DEVICE_PACKAGES = {"numpy", "torch", "ttexalens", "ttnn", "tt_umd", "yaml"}
FILES = sorted(GALAXY_DIR.glob("*.py"))
LOCAL = {f.stem for f in FILES}


def imports(path):
    """(top-level imports, all imports) of a file, by root package."""
    top, every = set(), set()
    for n in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and not n.level:
            names = [n.module]
        else:
            continue
        roots = {m.split(".")[0] for m in names}
        every |= roots
        if n.col_offset == 0:
            top |= roots
    return top, every


class GalaxyPrograms(unittest.TestCase):
    def test_imports_stay_inside(self):
        self.assertGreater(len(FILES), 20)
        for f in FILES:
            out = imports(f)[1] - LOCAL - DEVICE_PACKAGES - set(sys.stdlib_module_names)
            self.assertEqual(out, set(), f"{f.name} imports outside galaxy/")

    def test_no_path_tricks(self):
        sibling = "sys.path.append(os.path.dirname(os.path.realpath(__file__)))"
        for f in FILES:
            lines = [l.strip() for l in f.read_text().splitlines() if "sys.path" in l]
            for l in lines:
                self.assertTrue(l.startswith(sibling), f"{f.name}: {l}")

    def test_import_without_device(self):
        for f in FILES:
            if imports(f)[0] & DEVICE_PACKAGES:
                continue
            with self.subTest(module=f.stem):
                importlib.import_module(f.stem)

    def test_tray_of_bdf(self):
        import chunk_sched as cs

        self.assertEqual(
            [cs.tray_of(b) for b in ("0000:01:00.0", "0000:41:00.0", "0000:c1:00.0", "0000:88:00.0", "zz")],
            [1, 2, 3, 4, None],
        )

    def test_pcie_lines_fail_under_the_floor_warn_under_the_median_share(self):
        from quiet_clock import pcie_lines

        row = lambda b, h, d, bad="": dict(
            chip=f"{b} (tray 1, ASIC 2, device id 0)", mb=32, h2d_gbs=h, d2h_gbs=d, bad=bad
        )  # noqa: E731
        rs = [
            row("01:00.0", 0.1, 20),
            row("02:00.0", 20, 5),
            row("03:00.0", 20, 20),
            row("04:00.0", None, None, "RuntimeError: x"),
        ]
        out = pcie_lines(rs, 0.25, 0.5, "h", sysfs="/nonexistent")
        self.assertTrue(
            out[0].startswith("PCIE FAIL 01:00.0 (tray 1) ASIC 2, host h: PCIe bandwidth host to chip 0.10 GB/s")
        )
        self.assertIn("host to chip under the 0.25 GB/s floor", out[0])
        self.assertTrue(
            out[1].startswith("PCIE WARN 02:00.0 (tray 1) ASIC 2") and "chip to host under 50% of" in out[1]
        )
        self.assertTrue(out[2].startswith("PCIE FAIL 04:00.0") and "RuntimeError: x" in out[2])
        self.assertEqual(out[3][:44], "PCIe per chip (4 chips, 32 MB each way): hos")
        self.assertIn("2 FAIL, 1 WARN", out[3])

    def test_classify_compares_the_wrong_values_not_only_their_places(self):
        import numeric_capture as nc

        first = [((0, 1), 2.5), ((3, 4), "nan")]
        self.assertEqual(nc.classify([first, list(first), list(first)]), "written-wrong")
        self.assertEqual(nc.classify([first, [((0, 1), 2.5), ((3, 4), 7.0)], list(first)]), "transient")
        self.assertEqual(nc.classify([first, [], []]), "read-path")

    def test_fault_cut_by_tray(self):
        import load_line as ll

        with tempfile.TemporaryDirectory() as d:
            tel, faults = os.path.join(d, "c8.txt"), os.path.join(d, "fault_times.tsv")
            with open(tel, "w") as f:
                for t in (100, 110, 120):
                    for b in ("0000:01:00.0", "0000:41:00.0", "0000:c1:00.0"):
                        f.write(f"{t} {b} 50000 100000000\n")
            with open(faults, "w") as f:
                f.write("115\tTRAY DOWN tray 2 lost\n118\tHANG all_gather\n")
            first, box = ll.fault_cut(tel, faults)
        self.assertEqual(first, {"0000:41:00.0": 115.0, "0000:01:00.0": 118.0, "0000:c1:00.0": 118.0})
        self.assertEqual(box, 115.0)


if __name__ == "__main__":
    unittest.main()
