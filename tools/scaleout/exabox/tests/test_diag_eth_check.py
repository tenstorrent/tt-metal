#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The eth check phase against ethchk-upstream-tests.sh's report and exit codes.

The reports below are abridged from the script's own output. The phase runs a
stub script that prints one and exits with a chosen code, and a stub docker on
PATH, so no hardware and no image are involved.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.eth_check import no_verdict_reason, parse_eth_check_report, run_eth_check  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Phase  # noqa: E402

PROBLEM_REPORT = """\

================ TT ETHERNET LINK REPORT ================
  host bh-glx-110-a07u02   utc 2026-11-16T16:27:27Z   arch blackhole   2 reads 30 s apart
  machine Blackhole Galaxy (card type from the driver: galaxy-blackhole x32)
  image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.83.0-dev20261108
  ethchk version 3.1, 2026-11-15

  VERDICT: PROBLEM - links broken or corrupting data (2 found)
    - T1 TRACE P3: 1 of 2 built-in links not linked (on-board trace, nothing to reseat)
    - dev 4 core 1-1 ch0: link down
    also worth a look:
    - QSFP_DD port 7: linked on T1, T2, not on T3, T4 - trays are normally cabled alike, so a cable is probably dead or missing
  LINKS:   446 on-host links: 207 of 208 built-in + 239 on cables. QSFP 1-6 linked on all 4 trays.
           That is not a cabling this script has a name for. It has names for:
           mesh 208, Y torus 224.

  CHECK                        RESULT                 EXPECTED             STATUS
  ----------------------------------------------------------------------------------
  chips on PCIe                32 of 32 ok            32                   OK
  links down / cores stopped   1                      0                    ** PROBLEM **

  NOT CHECKED HERE
    - links to other hosts: check_eth_status lists on-host links only

  raw triage output: /data/hc/logs/ethchk.bh-glx-110-a07u02.20261116T162717Z

SUMMARY host=bh-glx-110-a07u02 verdict=PROBLEM pcie_ok=32/32 chips=32 kind=bh-galaxy card=galaxy-blackhole onhost_links=446 builtin=207/208 cabling=irregular down=1 uncorr_max_per_s=0.0
=========================================================
"""

HEALTHY_REPORT = """\

================ TT ETHERNET LINK REPORT ================
  host bh-glx-110-a07u02   utc 2026-11-16T16:27:09Z   arch blackhole   2 reads 30 s apart
  image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.83.0-dev20261108
  ethchk version 3.1, 2026-11-15

  VERDICT: HEALTHY - no fault found, no errors rising. The cabling is described below, not judged.
  LINKS:   448 on-host links: 208 built-in + 240 on cables. QSFP 1-6 linked on all 4 trays, as in an XY torus.

  raw triage output: /data/hc/logs/ethchk.bh-glx-110-a07u02.20261116T162658Z

SUMMARY host=bh-glx-110-a07u02 verdict=HEALTHY pcie_ok=32/32 chips=32 kind=bh-galaxy onhost_links=448 builtin=208/208 cabling=xy-torus down=0
=========================================================
"""

WARNING_REPORT = """\
  VERDICT: WARNING - nothing proven broken, worth a look (1 found)
    - 2 link(s) that were up in the last run are not linked now: dev 3 core 1-1 ch0, dev 3 core 2-1 ch1 - a fault, unless the host was recabled
  LINKS:   446 on-host links: 208 built-in + 238 on cables.

SUMMARY host=bh-glx-110-a07u02 verdict=WARNING gone_since_last=2
"""

PCIE_STOP_REPORT = """\

================ TT ETHERNET LINK REPORT ================
  host bh-glx-110-a07u02   utc 2026-11-16T16:28:07Z

  VERDICT: PROBLEM - a chip dropped off PCIe. Link check NOT run, so the chips are left as they are.
    - dev 3 0000:43:00.0 (T2 ASIC3): dropped off PCIe (Command=0x0000, memory decoding off)

  Do NOT reset yet: report the device above first (a reset wipes the evidence).
  To run the link check anyway: ETHCHK_FORCE=1 bash ethchk-upstream-tests.sh

"""

TRIAGE_FAILED_STDERR = """\

  ERROR: the link check printed no link table. Last lines of /data/hc/logs/ethchk.h.t/sample1.csv.log:
    TimeoutDeviceRegisterError: timed out reading 0x7cc00 on device 3

  Triage itself failed - the lines above say why (a hung chip shows up as a timeout).
"""

DOCKER_UNUSABLE_STDOUT = """\
  ERROR: Docker isn't usable as user: permission denied while trying to connect to the Docker daemon socket
  If that says permission denied, run it as:  sudo bash ethchk-upstream-tests.sh
"""


class TestParseEthCheckReport(unittest.TestCase):
    def test_verdict_and_findings_in_order(self):
        r = parse_eth_check_report(PROBLEM_REPORT)
        self.assertEqual(r["verdict"], "PROBLEM - links broken or corrupting data (2 found)")
        self.assertEqual(
            r["findings"],
            [
                "- T1 TRACE P3: 1 of 2 built-in links not linked (on-board trace, nothing to reseat)",
                "- dev 4 core 1-1 ch0: link down",
                "also worth a look:",
                "- QSFP_DD port 7: linked on T1, T2, not on T3, T4 - trays are normally cabled alike, "
                "so a cable is probably dead or missing",
            ],
        )

    def test_not_checked_bullets_are_not_findings(self):
        # Same indent as the findings, but under a later heading.
        r = parse_eth_check_report(PROBLEM_REPORT)
        self.assertFalse(any("links to other hosts" in f for f in r["findings"]))

    def test_links_line_joins_its_continuation(self):
        r = parse_eth_check_report(PROBLEM_REPORT)
        self.assertTrue(r["links"].startswith("446 on-host links: 207 of 208 built-in"))
        self.assertTrue(r["links"].endswith("mesh 208, Y torus 224."))

    def test_summary_fields_and_provenance(self):
        r = parse_eth_check_report(PROBLEM_REPORT)
        self.assertEqual(r["summary"]["verdict"], "PROBLEM")
        self.assertEqual(r["summary"]["builtin"], "207/208")
        self.assertEqual(r["image"], "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.83.0-dev20261108")
        self.assertEqual(r["version"], "version 3.1, 2026-11-15")
        self.assertEqual(r["raw_dir"], "/data/hc/logs/ethchk.bh-glx-110-a07u02.20261116T162717Z")

    def test_healthy_has_no_findings(self):
        r = parse_eth_check_report(HEALTHY_REPORT)
        self.assertTrue(r["verdict"].startswith("HEALTHY"))
        self.assertEqual(r["findings"], [])

    def test_pcie_stop_report(self):
        r = parse_eth_check_report(PCIE_STOP_REPORT)
        self.assertTrue(r["verdict"].startswith("PROBLEM - a chip dropped off PCIe"))
        self.assertEqual(len(r["findings"]), 1)
        self.assertIn("0000:43:00.0", r["findings"][0])
        self.assertEqual(r["summary"], {})

    def test_no_report(self):
        r = parse_eth_check_report("")
        self.assertEqual((r["verdict"], r["findings"], r["summary"]), ("", [], {}))


class TestNoVerdictReason(unittest.TestCase):
    def test_triage_failure_keeps_the_quoted_cause(self):
        reason = no_verdict_reason("", TRIAGE_FAILED_STDERR)
        self.assertIn("ERROR: the link check printed no link table", reason)
        self.assertIn("TimeoutDeviceRegisterError", reason)
        self.assertIn("Triage itself failed", reason)

    def test_without_an_error_line_the_last_line(self):
        self.assertEqual(no_verdict_reason("", "mkdir: /x: Permission denied\n"), "mkdir: /x: Permission denied")


class TestRunEthCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.logs = self.tmp / "logs"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        docker = self.bin / "docker"
        docker.write_text("#!/bin/sh\nexit 0\n")
        docker.chmod(0o755)
        self.path = f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"

    def tearDown(self):
        self._tmp.cleanup()

    def script(self, stdout: str = "", rc: int = 0, stderr: str = "") -> Path:
        (self.tmp / "out.txt").write_text(stdout)
        (self.tmp / "err.txt").write_text(stderr)
        script = self.tmp / "ethchk.sh"
        script.write_text(
            "#!/bin/bash\n"
            f'cat "{self.tmp}/out.txt"\n'
            'echo "ETHCHK_DIR=$ETHCHK_DIR"\n'
            f'cat "{self.tmp}/err.txt" >&2\n'
            f"exit {rc}\n"
        )
        return script

    def run_phase(self, script: Path, *, path: str | None = None, dry_run: bool = False, env=None) -> Phase:
        phase = Phase(name="eth_check")
        environ = {k: v for k, v in os.environ.items() if k != "ETHCHK_DIR"}
        environ.update(PATH=path if path is not None else self.path, **(env or {}))
        quiet = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
        with mock.patch.dict(os.environ, environ, clear=True), quiet[0], quiet[1]:
            run_eth_check(phase, dry_run, self.logs, script_override=str(script))
        return phase

    def statuses(self, phase: Phase) -> dict[str, str]:
        return {c.name: c.status for c in phase.checks}

    def test_exit_codes_are_the_verdict(self):
        for rc, report, expected in (
            (0, HEALTHY_REPORT, PASS),
            (2, PROBLEM_REPORT, FAIL),
            (3, WARNING_REPORT, WARN),
            (4, PCIE_STOP_REPORT, FAIL),
        ):
            with self.subTest(rc=rc):
                phase = self.run_phase(self.script(report, rc))
                self.assertEqual(self.statuses(phase), {"ethchk_run": PASS, "ethchk_verdict": expected})

    def test_verdict_details_carry_the_findings_and_links(self):
        phase = self.run_phase(self.script(PROBLEM_REPORT, 2))
        verdict = next(c for c in phase.checks if c.name == "ethchk_verdict")
        lines = verdict.details.split("\n")
        self.assertEqual(lines[0], "PROBLEM - links broken or corrupting data (2 found)")
        self.assertIn("- dev 4 core 1-1 ch0: link down", lines)
        self.assertTrue(lines[-1].startswith("LINKS: 446 on-host links"))
        self.assertEqual(verdict.ip, "eth")
        self.assertEqual(verdict.data["summary"]["cabling"], "irregular")

    def test_run_records_log_and_provenance(self):
        phase = self.run_phase(self.script(PROBLEM_REPORT, 2))
        run = next(c for c in phase.checks if c.name == "ethchk_run")
        self.assertEqual(run.data["rc"], 2)
        self.assertTrue(run.data["image"].startswith("ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:"))
        log = self.logs / "ethchk.log"
        self.assertTrue(log.is_file())
        self.assertIn("VERDICT: PROBLEM", log.read_text())

    def test_runs_land_in_logs_by_default(self):
        phase = self.run_phase(self.script(HEALTHY_REPORT, 0))
        run = next(c for c in phase.checks if c.name == "ethchk_run")
        self.assertIn(f"ETHCHK_DIR={self.logs}", run.data["stdout_tail"])

    def test_callers_ethchk_dir_wins(self):
        mine = str(self.tmp / "ethchk-runs")
        phase = self.run_phase(self.script(HEALTHY_REPORT, 0), env={"ETHCHK_DIR": mine})
        run = next(c for c in phase.checks if c.name == "ethchk_run")
        self.assertIn(f"ETHCHK_DIR={mine}", run.data["stdout_tail"])

    def test_docker_unusable_is_skip_with_the_reason(self):
        phase = self.run_phase(self.script(DOCKER_UNUSABLE_STDOUT, 5))
        self.assertEqual(self.statuses(phase), {"ethchk_run": SKIP})
        self.assertIn("permission denied", phase.checks[0].details)
        self.assertIn("sudo bash ethchk-upstream-tests.sh", phase.checks[0].details)

    def test_triage_failure_is_warn_with_the_reason(self):
        phase = self.run_phase(self.script("", 1, TRIAGE_FAILED_STDERR))
        self.assertEqual(self.statuses(phase), {"ethchk_run": WARN})
        self.assertIn("TimeoutDeviceRegisterError", phase.checks[0].details)

    def test_verdict_code_without_a_report_is_warn(self):
        # A clean exit with no report is not a clean reading.
        phase = self.run_phase(self.script("", 0))
        self.assertEqual(self.statuses(phase), {"ethchk_run": WARN})

    def test_no_docker_is_skip_without_running(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        phase = self.run_phase(self.script(HEALTHY_REPORT, 0), path=str(empty))
        self.assertEqual(self.statuses(phase), {"ethchk_run": SKIP})
        self.assertIn("docker not on PATH", phase.checks[0].details)
        self.assertFalse((self.logs / "ethchk.log").exists())

    def test_missing_script_is_skip(self):
        phase = self.run_phase(self.tmp / "nope.sh")
        self.assertEqual(self.statuses(phase), {"ethchk_run": SKIP})
        self.assertIn("--eth-check-script=", phase.checks[0].details)

    def test_dry_run_runs_nothing(self):
        phase = self.run_phase(self.script(HEALTHY_REPORT, 0), dry_run=True)
        self.assertEqual(self.statuses(phase), {"ethchk_run": SKIP})
        self.assertTrue(phase.checks[0].details.startswith("(dry) ETHCHK_DIR="))
        self.assertFalse((self.logs / "ethchk.log").exists())


if __name__ == "__main__":
    unittest.main()
