#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the weekly tier's bus-bar and link coverage: the load tests
in the tier, the HSC input-voltage judgement around them, and the chip recount
after them."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag import hsc  # noqa: E402
from diag.device_tests import PYTESTS  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Phase  # noqa: E402
from diag.snapshot_checks import expected_bdf, pcie_enum_check  # noqa: E402
from tiers import TIERS  # noqa: E402

CLEAN = (0x0000, 0x00)
UV_WARN = (0x0000, 0x20)
OFF = (0x0040, 0x00)


def _status(**per_hsc) -> dict:
    """All 8 HSCs at CLEAN, with the given (tray, addr) overridden."""
    status = {(t, a): CLEAN for t in hsc.HSC_BUS for a in hsc.HSC_ADDRS}
    status.update(per_hsc.get("override", {}))
    return status


def _judge(before, after):
    phase = Phase(name="tests")
    hsc.judge_hsc(before, after, phase)
    (check,) = phase.checks
    return check


def _galaxy_configs(path: Path, case_id: str) -> set[str]:
    """Fabric configs of every ``_galaxy(ttnn.FabricConfig.X, case_id)`` param in ``path``."""
    return {
        call.args[0].attr
        for call in ast.walk(ast.parse(path.read_text()))
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_galaxy" and call.args[1].value == case_id
    }


class TestWeeklyPytests(unittest.TestCase):
    def test_targets_exist_in_tree(self):
        for name in TIERS["weekly"].tests:
            if name in PYTESTS:
                target = PYTESTS[name]["target"].split("::")[0]
                self.assertTrue((SUITE_DIR.parents[3] / target).is_file(), target)

    def test_fabric_step_selects_only_plain_2d_mesh_cases(self):
        # A torus case asserts in fabric init on a galaxy without the torus cables.
        spec = PYTESTS["fabric_all_gather_matmul_galaxy"]
        k = spec["args"][spec["args"].index("-k") + 1]
        self.assertTrue(k.startswith("SP8xTP4-mesh and ("), k)
        self.assertEqual(_galaxy_configs(SUITE_DIR.parents[3] / spec["target"], "SP8xTP4-mesh"), {"FABRIC_2D"})


class TestHscJudgement(unittest.TestCase):
    def test_clean_before_and_after_passes(self):
        self.assertEqual(_judge(_status(), _status()).status, PASS)

    def test_flag_newly_set_fails(self):
        check = _judge(_status(), _status(override={(2, 0x40): UV_WARN}))
        self.assertEqual(check.status, FAIL)
        self.assertIn("tray 2", check.details)
        self.assertIn("VIN_UV_WARN", check.details)

    def test_off_newly_set_fails(self):
        self.assertEqual(_judge(_status(), _status(override={(4, 0x42): OFF})).status, FAIL)

    def test_over_current_trip_names_its_reason(self):
        from diag.hsc import _flags, flag_names

        trip = (0x4852, 0x00)  # IOUT summary, power not good, OFF, IOUT_OC_FAULT, CML
        self.assertEqual(flag_names(_flags(trip)), "OFF, IOUT_OC_FAULT")
        check = _judge(_status(), _status(override={(1, 0x42): trip}))
        self.assertEqual(check.status, FAIL)
        self.assertIn("tray 1 HSC 0x42: OFF, IOUT_OC_FAULT newly set during the tests", check.details)

    def test_flag_already_set_before_warns(self):
        latched = _status(override={(3, 0x40): UV_WARN})
        self.assertEqual(_judge(latched, latched).status, WARN)

    def test_unread_hsc_warns(self):
        self.assertEqual(_judge(_status(), _status(override={(1, 0x40): None})).status, WARN)

    def test_no_answer_at_all_is_skip_not_fail(self):
        none = {k: None for k in _status()}
        self.assertEqual(_judge(none, none).status, SKIP)


class TestHscRead(unittest.TestCase):
    def test_reads_are_ipmitool_raw_reads_only(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            nbytes = int(cmd[-2], 16)
            return mock.Mock(returncode=0, stdout=" ".join(["00"] * nbytes))

        with mock.patch.object(hsc.subprocess, "run", fake_run):
            status = hsc.read_hsc_status()
        self.assertEqual(len(status), 8)
        self.assertTrue(all(v == (0, 0) for v in status.values()))
        self.assertEqual(len(calls), 16)
        for cmd in calls:
            i = cmd.index("raw")
            self.assertEqual(cmd[i + 1 : i + 3], ["0x06", "0x52"])
            self.assertIn(cmd[-1], ("0x79", "0x7c"))  # STATUS_WORD, STATUS_INPUT: no write register

    def test_failed_read_is_none(self):
        with mock.patch.object(hsc.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")):
            self.assertTrue(all(v is None for v in hsc.read_hsc_status().values()))


class TestPcieRecount(unittest.TestCase):
    @staticmethod
    def _devs(n):
        bdfs = [expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9)]
        return [{"board_info": {"bus_id": b}} for b in bdfs[:n]]

    def test_full_count_passes(self):
        self.assertEqual(pcie_enum_check(self._devs(32), "pcie_enum_count_after_tests").status, PASS)

    def test_short_count_fails_and_names_missing(self):
        check = pcie_enum_check(self._devs(31), "pcie_enum_count_after_tests")
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.name, "pcie_enum_count_after_tests")
        self.assertEqual(len(check.data["missing_bdfs"]), 1)

    def test_recount_uses_the_fresh_snapshot(self):
        from diag_runner import recount_after_tests

        check = recount_after_tests(lambda: {"device_info": self._devs(24)})
        self.assertEqual((check.name, check.status), ("pcie_enum_count_after_tests", FAIL))
        self.assertEqual(len(check.data["missing_bdfs"]), 8)

    def test_unreadable_chip_fails_with_one_line_reason(self):
        from diag_runner import recount_after_tests

        err = (
            "tt-smi -f failed (rc=1): \x1b[91m Error in detecting devices!\n"
            "Read 0xffffffff over PCIe ID 24: the board should be reset.\n"
            "Location: /project/device/tt_device/tt_device.cpp:272\n"
            " 1. /opt/tt-smi/.venv/lib/python3.10/site-packages/tt_umd/tt_umd.so(+0x101343) [0x7d38b3301343]\n"
            " 2. tt::umd::TTDevice::init_tt_device()\n\x1b[0m"
        )

        def snapshot():
            raise RuntimeError(err)

        check = recount_after_tests(snapshot)
        self.assertEqual((check.name, check.status), ("pcie_enum_count_after_tests", FAIL))
        self.assertEqual(
            check.details,
            "no snapshot after the tests: tt-smi -f failed (rc=1):  Error in detecting devices!; "
            "Read 0xffffffff over PCIe ID 24: the board should be reset.",
        )

    def test_hung_tt_smi_times_out_into_one_line_fail(self):
        import subprocess

        from diag_runner import recount_after_tests
        from diag.snapshot import SNAPSHOT_TIMEOUT_S, take_snapshot

        def run(cmd, **kwargs):
            # A hung chip never returns: only a bounded call gets out.
            if kwargs.get("timeout") is None:
                raise AssertionError("tt-smi -f called without a timeout")
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with mock.patch("diag.snapshot.subprocess.run", side_effect=run):
            check = recount_after_tests(lambda: take_snapshot("tt-smi", Path("/nonexistent/after.json")))
        self.assertEqual((check.name, check.status), ("pcie_enum_count_after_tests", FAIL))
        self.assertEqual(
            check.details, f"no snapshot after the tests: tt-smi -f timed out after {SNAPSHOT_TIMEOUT_S} s"
        )

    def test_missing_tt_smi_fails_instead_of_raising(self):
        from diag_runner import recount_after_tests

        def snapshot():
            raise FileNotFoundError("tt-smi not found")

        self.assertEqual(recount_after_tests(snapshot).status, FAIL)


if __name__ == "__main__":
    unittest.main()
