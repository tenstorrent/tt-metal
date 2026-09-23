#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recover.sh's retrain rounds and targeted glx_reset escalation.

recover.sh only reaches hardware through mpirun, so a stub mpirun on PATH is enough to
drive the loop without a cluster: it classifies each launch as validation or reset, fakes
run_cluster_validation's unretrainable-channel report, and records which hosts each reset
targeted. The point of the escalation is that a reset hits only the hosts the unretrainable
cable lands on, which is what these tests pin down.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
EXABOX_DIR = TESTS_DIR.parent
REPO_ROOT = EXABOX_DIR.parents[2]
RECOVER_SH = EXABOX_DIR / "recover.sh"

ALL_HOSTS = ["node-a", "node-b", "node-c", "node-d"]
FAULTY_HOSTS = ["node-b", "node-d"]

STUB_MPIRUN = '''#!/usr/bin/env python3
"""Stub mpirun for recover.sh tests: fakes validation, records reset targets."""
import os
import re
import sys

argv = sys.argv[1:]
payload = " ".join(argv)

trace = os.environ["STUB_TRACE"]
state = os.environ["STUB_STATE"]
faulty = [h for h in os.environ.get("STUB_FAULTY_HOSTS", "").split(",") if h]
pass_after_resets = int(os.environ.get("STUB_PASS_AFTER_RESETS", "2"))


def hosts():
    return argv[argv.index("--host") + 1] if "--host" in argv else ""


def counter(name, bump):
    path = state + "." + name
    value = int(open(path).read()) if os.path.exists(path) else 0
    if bump:
        value += 1
        with open(path, "w") as handle:
            handle.write(str(value))
    return value


def record(line):
    with open(trace, "a") as handle:
        handle.write(line + "\\n")


if "glx_reset" in payload:
    counter("reset", True)
    record("RESET " + hosts())
    sys.exit(0)

if "run_cluster_validation" in payload:
    match = re.search(r"max-retrains\\s+(\\d+)", payload)
    record("VALIDATION budget=" + (match.group(1) if match else "?"))
    if os.environ.get("STUB_EMIT_RETRAIN_SUMMARY"):
        print("Link Retraining Summary: 8 link endpoint(s) retrained over 2 retrain iteration(s)")
    # The targeted reset is what "fixes" the links, so pass once it has happened.
    if counter("reset", False) >= pass_after_resets:
        sys.exit(0)
    match = re.search(r"output-path\\s+(\\S+)", payload)
    out_dir = match.group(1) if match else "."
    with open(os.path.join(out_dir, "unretrainable_channels.yaml"), "w") as handle:
        handle.write("unretrainable_channels:\\n")
        for index, host in enumerate(faulty):
            handle.write("  - host: %s\\n" % host)
            handle.write("    tray_id: %d\\n" % (index + 1))
            handle.write("    asic_location: 1\\n")
            handle.write("    channel: 2\\n")
    sys.exit(1)

# Everything else (interface probe, host checks) just needs to succeed.
sys.exit(0)
'''


def first_non_loopback_interface():
    """recover.sh rejects an --mpi-if that ip(8) does not know about."""
    if shutil.which("ip") is None:
        return None
    result = subprocess.run(["ip", "-o", "link", "show"], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        fields = line.split(":")
        if len(fields) < 2:
            continue
        name = fields[1].strip().split("@")[0]
        if name and name != "lo":
            return name
    return None


MPI_INTERFACE = first_non_loopback_interface()


@unittest.skipIf(MPI_INTERFACE is None, "needs a non-loopback interface for --mpi-if")
class RecoverRetrainRoundsTest(unittest.TestCase):
    def run_recover(self, *extra_args, faulty=FAULTY_HOSTS, pass_after_resets=2, retrain_summary=False):
        """Runs recover.sh against the stub; returns (exit code, trace lines)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            stub = bin_dir / "mpirun"
            stub.write_text(STUB_MPIRUN)
            stub.chmod(0o755)

            out_dir = tmp_path / "out"
            out_dir.mkdir()
            trace = tmp_path / "trace.txt"

            env = dict(os.environ)
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            env["STUB_TRACE"] = str(trace)
            env["STUB_STATE"] = str(tmp_path / "state")
            env["STUB_FAULTY_HOSTS"] = ",".join(faulty)
            env["STUB_PASS_AFTER_RESETS"] = str(pass_after_resets)
            if retrain_summary:
                env["STUB_EMIT_RETRAIN_SUMMARY"] = "1"

            command = [
                str(RECOVER_SH),
                "--hosts",
                ",".join(ALL_HOSTS),
                "--config",
                "4x32",
                "--mpi-if",
                MPI_INTERFACE,
                "--skip-version-check",
                "--skip-mpi-stress-test",
                "--no-regenerate-on-failure",
                "--sleep-duration",
                "0",
                "--output",
                str(out_dir),
                *extra_args,
            ]
            result = subprocess.run(
                command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False, timeout=300
            )
            lines = trace.read_text().splitlines() if trace.exists() else []
            return result.returncode, lines

    def test_escalation_resets_only_the_unretrainable_hosts(self):
        code, trace = self.run_recover("--max-retrains", "4", "--reset-every", "2")

        self.assertEqual(
            trace,
            [
                f"RESET {','.join(ALL_HOSTS)}",
                "VALIDATION budget=2",
                f"RESET {','.join(FAULTY_HOSTS)}",
                "VALIDATION budget=2",
            ],
        )
        self.assertEqual(code, 0)

    def test_targeted_reset_spares_healthy_hosts(self):
        _, trace = self.run_recover("--max-retrains", "4", "--reset-every", "2")

        targeted = [line for line in trace if line.startswith("RESET")][1]
        for host in FAULTY_HOSTS:
            self.assertIn(host, targeted)
        for host in set(ALL_HOSTS) - set(FAULTY_HOSTS):
            self.assertNotIn(host, targeted)

    def test_reset_every_at_max_retrains_keeps_the_original_single_round(self):
        code, trace = self.run_recover("--max-retrains", "4", "--reset-every", "4")

        self.assertEqual(trace, [f"RESET {','.join(ALL_HOSTS)}", "VALIDATION budget=4"])
        self.assertNotEqual(code, 0)

    def test_no_targeted_reset_when_no_host_is_named(self):
        # Validation failed for some reason other than a cable, which a reset will not fix.
        code, trace = self.run_recover("--max-retrains", "4", "--reset-every", "2", faulty=[])

        self.assertEqual(trace, [f"RESET {','.join(ALL_HOSTS)}", "VALIDATION budget=2"])
        self.assertNotEqual(code, 0)

    def test_skip_reset_issues_no_reset_at_all(self):
        code, trace = self.run_recover("--max-retrains", "4", "--reset-every", "2", "--skip-reset")

        self.assertEqual(trace, ["VALIDATION budget=4"])
        self.assertNotEqual(code, 0)

    def test_rerun_on_retrain_revalidates_after_a_pass(self):
        code, trace = self.run_recover(
            "--max-retrains",
            "4",
            "--reset-every",
            "2",
            "--rerun-on-retrain",
            pass_after_resets=0,
            retrain_summary=True,
        )

        self.assertEqual(trace, [f"RESET {','.join(ALL_HOSTS)}", "VALIDATION budget=2", "VALIDATION budget=2"])
        self.assertEqual(code, 0)

    def test_rerun_on_retrain_leaves_a_failed_round_to_the_targeted_reset(self):
        # Retrains from a failed round are already spent, so the round escalates instead of
        # rerunning; a rerun here would also spend retrains the round budget never accounted for.
        code, trace = self.run_recover(
            "--max-retrains", "4", "--reset-every", "2", "--rerun-on-retrain", retrain_summary=True
        )

        # The failed first round escalates with no rerun in between; the rerun only follows the
        # round that passes.
        self.assertEqual(
            trace,
            [
                f"RESET {','.join(ALL_HOSTS)}",
                "VALIDATION budget=2",
                f"RESET {','.join(FAULTY_HOSTS)}",
                "VALIDATION budget=2",
                "VALIDATION budget=2",
            ],
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
