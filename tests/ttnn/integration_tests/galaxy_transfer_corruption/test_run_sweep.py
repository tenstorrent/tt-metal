# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of sweep coverage and result qualification.

Run with python3 -m unittest discover -s <this directory> -p test_run_sweep.py.
Only the hardware launcher subprocess is replaced; reports and aggregation use
real files so missing/partial captures cannot silently turn into a clean sweep.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("run_sweep.py")
spec = importlib.util.spec_from_file_location("run_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name)
        self.output = self.checkout / "results"

    def launch(self, command, *, cwd):
        device = int(command[command.index("--device") + 1])
        output = Path(command[command.index("--output") + 1])
        if "--dry-run" not in command:
            output.mkdir(parents=True)
            (output / "report.json").write_text(
                json.dumps(
                    dict(
                        stage="complete",
                        device=device,
                        iterations=100000,
                        requested_iterations=100000,
                        source_unchanged_end=True,
                        faults=[],
                    )
                )
            )
            (output / "launcher.json").write_text(json.dumps(dict(result_qualified=True, child_exit_code=0)))
        return subprocess.CompletedProcess(command, 0)

    def run_with(self, launcher=None, dry_run=False):
        with patch.object(sweep, "subprocess") as process:
            process.run.side_effect = launcher or self.launch
            code = sweep.run_sweep(self.checkout, self.output, 100000, dry_run=dry_run)
        return code, json.loads((self.output / "sweep.json").read_text())

    def test_all_32_devices_have_separate_qualified_results(self):
        code, result = self.run_with()
        self.assertEqual(code, 0)
        self.assertEqual([r["device"] for r in result["results"]], list(range(32)))
        self.assertEqual(len(list(self.output.glob("device-*/report.json"))), 32)
        self.assertEqual(result["total_iterations"], 3200000)
        self.assertEqual(result["clean_devices"], list(range(32)))
        self.assertEqual(result["stage"], "complete")

    def test_first_device_fault_does_not_skip_later_devices(self):
        def launcher(command, *, cwd):
            child = self.launch(command, cwd=cwd)
            output = Path(command[command.index("--output") + 1])
            if int(command[command.index("--device") + 1]) == 0:
                report = json.loads((output / "report.json").read_text())
                report.update(iterations=560, faults=[dict(iteration=559, changed_bytes=1)])
                (output / "report.json").write_text(json.dumps(report))
                (output / "launcher.json").write_text(json.dumps(dict(result_qualified=True, child_exit_code=1)))
                child.returncode = 1
            return child

        code, result = self.run_with(launcher)
        self.assertEqual(code, 1)
        self.assertEqual(result["fault_devices"], [0])
        self.assertEqual(result["clean_devices"], list(range(1, 32)))
        self.assertEqual(result["results"][0]["iterations"], 560)
        self.assertTrue((self.output / "device-00" / "report.json").is_file())

    def test_setup_failure_takes_precedence_over_fault_and_sweep_continues(self):
        def launcher(command, *, cwd):
            device = int(command[command.index("--device") + 1])
            if device == 11:
                return subprocess.CompletedProcess(command, 2)
            child = self.launch(command, cwd=cwd)
            if device == 12:
                output = Path(command[command.index("--output") + 1])
                report = json.loads((output / "report.json").read_text())
                report["faults"] = [dict(iteration=99999, changed_bytes=1)]
                (output / "report.json").write_text(json.dumps(report))
                (output / "launcher.json").write_text(json.dumps(dict(result_qualified=True, child_exit_code=1)))
                child.returncode = 1
            return child

        code, result = self.run_with(launcher)
        self.assertEqual(code, 2)
        self.assertEqual(result["error_devices"], [11])
        self.assertEqual(result["fault_devices"], [12])
        self.assertEqual(len(result["results"]), 32)
        self.assertIn(31, result["clean_devices"])

    def test_zero_exit_with_incomplete_or_wrong_device_report_is_an_error(self):
        for mutation in [dict(iterations=99999), dict(device=31), dict(source_unchanged_end=False)]:
            with self.subTest(mutation=mutation):
                self.output = self.checkout / str(len(list(self.checkout.iterdir())))

                def launcher(command, *, cwd):
                    child = self.launch(command, cwd=cwd)
                    if int(command[command.index("--device") + 1]) == 0:
                        output = Path(command[command.index("--output") + 1])
                        report = json.loads((output / "report.json").read_text())
                        report.update(mutation)
                        (output / "report.json").write_text(json.dumps(report))
                    return child

                code, result = self.run_with(launcher)
                self.assertEqual(code, 2)
                self.assertEqual(result["error_devices"], [0])
                self.assertIn(31, result["clean_devices"])

    def test_zero_exit_without_evidence_is_an_error(self):
        code, result = self.run_with(lambda command, *, cwd: subprocess.CompletedProcess(command, 0))
        self.assertEqual(code, 2)
        self.assertEqual(result["error_devices"], list(range(32)))
        self.assertEqual(result["clean_devices"], [])

    def test_dry_run_plans_all_devices_without_claiming_clean_results(self):
        code, result = self.run_with(dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(result["stage"], "dry-run")
        self.assertEqual([r["device"] for r in result["results"]], list(range(32)))
        self.assertEqual(result["clean_devices"], [])
        self.assertEqual(result["total_iterations"], 0)
        self.assertEqual(list(self.output.glob("device-*/report.json")), [])


if __name__ == "__main__":
    unittest.main()
