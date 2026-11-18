#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tiers.py against the places that repeat what it says.

The shell wrappers validate the tier name themselves, before any Python runs,
so a tier added to tiers.py and not to them is a tier nobody can launch. The
docker wrapper also hardcodes which tiers want the sys-triage mounts and the
syseng package, and the fleet harness keeps its own --tier choices. These tests
fail until they agree.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from tiers import TIERS  # noqa: E402

# A case-statement arm listing the tiers, e.g. `flash|light|medium|deploy|pre_reboot)`.
TIER_ARM_RE = re.compile(r"^\s*((?:[a-z_]+\|)+[a-z_]+)\)", re.MULTILINE)


def tier_arms(script: str) -> list[set[str]]:
    text = (SUITE_DIR / script).read_text()
    return [set(m.group(1).split("|")) for m in TIER_ARM_RE.finditer(text) if "light" in m.group(1)]


class TestShellWrappersAgree(unittest.TestCase):
    def test_run_diag_accepts_every_tier(self):
        arms = tier_arms("run_diag.sh")
        self.assertEqual(len(arms), 1, arms)
        self.assertEqual(arms[0], set(TIERS))

    def test_run_diag_docker_accepts_every_tier(self):
        arms = tier_arms("run_diag_docker.sh")
        # The positional form and the case under -t/--tier.
        self.assertEqual(len(arms), 2, arms)
        for arm in arms:
            self.assertEqual(arm, set(TIERS))

    def test_docker_sys_triage_tiers(self):
        # run_diag_docker.sh adds the journal mounts and the BMC credentials
        # only for `pre_reboot`.
        self.assertEqual({name for name, tier in TIERS.items() if tier.sys_triage}, {"pre_reboot"})

    def test_docker_qsfp_tiers(self):
        # run_diag_docker.sh sets WANT_QSFP, which installs the syseng package and
        # warns about missing BMC API credentials, on every tier but these.
        self.assertEqual({name for name, tier in TIERS.items() if not tier.qsfp_tests}, {"flash", "light"})
        text = (SUITE_DIR / "run_diag_docker.sh").read_text()
        self.assertIn("[[ ${TIER} != flash && ${TIER} != light ]]", text)

    def test_run_health_check_accepts_every_tier(self):
        text = (SUITE_DIR / "test_infrastructure" / "run_health_check.py").read_text()
        m = re.search(r'"--tier",.*?choices=\[([^\]]*)\]', text, re.DOTALL)
        self.assertIsNotNone(m)
        self.assertEqual(set(re.findall(r'"([a-z_]+)"', m.group(1))), set(TIERS))

    def test_run_health_check_non_resolving_tiers(self):
        # A clean flash run must not auto-close a ticket its device tests never looked at.
        text = (SUITE_DIR / "test_infrastructure" / "run_health_check.py").read_text()
        m = re.search(r"^NON_RESOLVING_TIERS = frozenset\(\{([^}]*)\}\)", text, re.MULTILINE)
        self.assertIsNotNone(m)
        names = set(re.findall(r'"([a-z_]+)"', m.group(1)))
        self.assertIn("flash", names)
        self.assertLessEqual(names, set(TIERS))


class TestDiagRunnerCli(unittest.TestCase):
    def test_tier_choices_follow_tiers(self):
        from diag_runner import build_diag_arg_parser

        action = next(a for a in build_diag_arg_parser()._actions if "--tier" in a.option_strings)
        self.assertEqual(list(action.choices), list(TIERS))

    def test_docker_wrapper_feature_probe_still_matches(self):
        # run_diag_docker.sh greps diag_runner.py for this flag to tell an image
        # that predates the sys-triage phase, and drops the phase's flags if it
        # is missing. The parser has to stay in that file for the probe to work.
        self.assertIn("--skip-sys-triage", (SUITE_DIR / "diag_runner.py").read_text())


if __name__ == "__main__":
    unittest.main()
