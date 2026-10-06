#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for diag_runner.py's sys-triage phase, against a stand-in
sys-triage script: the SKIP reasons, how the BMC credentials reach the tool
(environment only, never argv or the report), and how its warnings and missing
index are reported."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag_runner import (  # noqa: E402
    PASS,
    SKIP,
    WARN,
    Phase,
    resolve_bmc_env,
    run_sys_triage,
)

CREDS = {"bmc_ip": "10.0.0.7", "bmc_user": "ttop", "bmc_password": "s3cr3t-pw"}


def fake_sys_triage(directory: Path, body: str) -> Path:
    """An executable that records its argv and BMC environment, then runs body."""
    path = directory / "sys-triage"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out.parent / 'seen.json').write_text(json.dumps({'argv': sys.argv[1:], "
        "'env': {k: os.environ.get(k) for k in ('BMC_IP', 'BMC_USER', 'BMC_PASSWORD')}}))\n" + textwrap.dedent(body)
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class SysTriageCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.logs = self.tmp / "logs"
        env = {k: v for k, v in os.environ.items() if k not in ("BMC_IP", "BMC_USER", "BMC_PASSWORD")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_phase(self, binary: Path | None, tier: str = "pre_reboot", **kw) -> Phase:
        phase = Phase(name="sys_triage", gates=False)
        run_sys_triage(
            tier,
            phase,
            False,
            self.logs,
            binary_override=str(binary) if binary else None,
            **kw,
        )
        return phase


class TestSkips(SysTriageCase):
    def test_only_pre_reboot_runs_it(self) -> None:
        binary = fake_sys_triage(self.tmp, "(out / 'triage.txt').write_text('x')\n")
        for tier in ("light", "medium", "deploy"):
            with self.subTest(tier=tier):
                phase = self.run_phase(binary, tier=tier, **CREDS)
                self.assertEqual([c.status for c in phase.checks], [SKIP])
                self.assertIn(f"tier '{tier}'", phase.checks[0].details)
        self.assertFalse((self.logs / "seen.json").exists(), "tool must not run outside pre_reboot")

    def test_tool_not_on_path_skips(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp)}):
            phase = self.run_phase(None, **CREDS)
        self.assertEqual(phase.checks[0].status, SKIP)
        self.assertIn("sys-triage not on PATH", phase.checks[0].details)

    def test_bad_override_skips_with_reason(self) -> None:
        phase = self.run_phase(self.tmp / "nope", **CREDS)
        self.assertEqual(phase.checks[0].status, SKIP)
        self.assertIn("--sys-triage-path", phase.checks[0].details)

    def test_missing_credentials_skip_and_name_what_is_missing(self) -> None:
        binary = fake_sys_triage(self.tmp, "(out / 'triage.txt').write_text('x')\n")
        phase = self.run_phase(binary, bmc_ip="10.0.0.7")
        self.assertEqual(phase.checks[0].status, SKIP)
        self.assertIn("--bmc-user/$BMC_USER", phase.checks[0].details)
        self.assertIn("--bmc-password/$BMC_PASSWORD", phase.checks[0].details)
        self.assertFalse((self.logs / "seen.json").exists(), "tool must not run without credentials")


class TestCredentials(SysTriageCase):
    def test_flags_win_over_environment(self) -> None:
        os.environ.update({"BMC_IP": "1.1.1.1", "BMC_USER": "envuser", "BMC_PASSWORD": "envpw"})
        env, missing = resolve_bmc_env("10.0.0.7", None, None)
        self.assertEqual(missing, [])
        self.assertEqual(env, {"BMC_IP": "10.0.0.7", "BMC_USER": "envuser", "BMC_PASSWORD": "envpw"})

    def test_password_reaches_tool_by_environment_only(self) -> None:
        binary = fake_sys_triage(
            self.tmp,
            """
            (out / 'triage.txt').write_text('ok')
            print('cpld-dump failed for s3cr3t-pw', file=sys.stderr)
            """,
        )
        phase = self.run_phase(binary, **CREDS)
        seen = json.loads((self.logs / "seen.json").read_text())
        self.assertEqual(seen["argv"], [str(self.logs / "sys_triage")])
        self.assertEqual(seen["env"], {"BMC_IP": "10.0.0.7", "BMC_USER": "ttop", "BMC_PASSWORD": "s3cr3t-pw"})
        stored = json.dumps([c.__dict__ for c in phase.checks]) + (self.logs / "sys_triage.log").read_text()
        self.assertNotIn("s3cr3t-pw", stored)


class TestOutcome(SysTriageCase):
    def test_clean_bundle_passes(self) -> None:
        binary = fake_sys_triage(
            self.tmp,
            """
            (out / 'triage.txt').write_text('ok')
            (out / 'ubb0-T38.txt').write_text('ok')
            """,
        )
        check = self.run_phase(binary, **CREDS).checks[0]
        self.assertEqual(check.status, PASS)
        self.assertEqual(check.data["files"], ["triage.txt", "ubb0-T38.txt"])

    def test_warnings_make_it_warn_and_are_kept(self) -> None:
        binary = fake_sys_triage(
            self.tmp,
            """
            (out / 'triage.txt').write_text('ok')
            print("warning: fw-version failed: [Errno 2] No such file or directory: 'modinfo'", file=sys.stderr)
            """,
        )
        check = self.run_phase(binary, **CREDS).checks[0]
        self.assertEqual(check.status, WARN)
        self.assertEqual(check.data["warnings"], ["fw-version failed: [Errno 2] No such file or directory: 'modinfo'"])
        self.assertIn("modinfo", check.details)

    def test_no_index_skips(self) -> None:
        binary = fake_sys_triage(self.tmp, "print('error: tt-smi: not found', file=sys.stderr)\nsys.exit(1)\n")
        check = self.run_phase(binary, **CREDS).checks[0]
        self.assertEqual(check.status, SKIP)
        self.assertIn("no triage.txt", check.details)
        self.assertIn("tt-smi: not found", check.details)

    def test_stale_bundle_is_cleared(self) -> None:
        stale = self.logs / "sys_triage"
        stale.mkdir(parents=True)
        (stale / "triage.txt").write_text("old run")
        binary = fake_sys_triage(self.tmp, "sys.exit(1)\n")
        check = self.run_phase(binary, **CREDS).checks[0]
        self.assertEqual(check.status, SKIP)
        self.assertFalse((stale / "triage.txt").exists())


if __name__ == "__main__":
    unittest.main()
