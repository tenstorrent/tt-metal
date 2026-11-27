#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ERISC debug dump: how its ELF is found, how its result is graded, and how
the phase runs it, against a stand-in for the python it runs under."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(TESTS_DIR))

from diag.erisc_debug import (  # noqa: E402
    DEFAULT_ERISC_ELF_DIR,
    find_elf,
    format_fw_version,
    run_erisc_debug,
    summarize_erisc_debug,
)
from diag.model import PASS, SKIP, WARN, Phase  # noqa: E402
from test_diag_eth_status import fake_sysfs  # noqa: E402

ELF_DIR = str(DEFAULT_ERISC_ELF_DIR)


def core(device: int, i: int, failed=(), error=None) -> dict:
    location = f"{i}-1 (e0,{i})"
    if error:
        return {"device": device, "location": location, "error": error}
    return {
        "device": device,
        "location": location,
        "error": None,
        "failed_sections": list(failed),
        "pc": "0x77924",
        "callstack": ["service_eth_msg (eth_runtime.cpp:567)"],
        "postcodes": {
            "eth": {"value": "0xc0dea005", "name": "POSTCODE_ETH_INIT_PASS"},
            "serdes": {"value": "0xc0dea015", "name": "POSTCODE_SERDES_INIT_PASS"},
            "macpcs": {"value": "0xc0de1234", "name": "UNKNOWN"},
        },
    }


def dump_result(versions: dict[str, list[int]], found: set[str]) -> dict:
    """What the dump writes for chips running ``versions`` when only the ELFs of
    ``found`` are in the ELF directory: 14 ETH cores a chip."""
    elfs, cores = [], []
    for tag, ids in versions.items():
        path = f"{ELF_DIR}/eth_init-{tag}.elf" if tag in found else None
        reason = None if path else f"not found in {ELF_DIR}"
        elfs.append({"eth_fw": tag, "devices": ids, "needed": f"eth_init-{tag}.elf", "path": path, "reason": reason})
        cores += [core(d, i) for d in ids for i in range(14)] if path else []
    return {"elf_dir": ELF_DIR, "elfs": elfs, "cores": cores, "error": None}


GALAXY = {"v1.14.0": list(range(32))}


class TestElf(unittest.TestCase):
    def test_found_by_either_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(find_elf(d, "v1.14.0"))
            (d / "eth_init_v1.14.0.elf").write_bytes(b"")
            self.assertEqual(find_elf(d, "v1.14.0"), d / "eth_init_v1.14.0.elf")
            # The release's own name wins.
            (d / "eth_init-v1.14.0.elf").write_bytes(b"")
            self.assertEqual(find_elf(d, "v1.14.0"), d / "eth_init-v1.14.0.elf")
            self.assertIsNone(find_elf(d, "v1.13.0"))

    def test_version_format(self):
        self.assertEqual(format_fw_version(1 << 16 | 14 << 8 | 2), "v1.14.2")
        self.assertEqual(format_fw_version(None), "unknown")

    def test_docker_wrapper_mounts_the_default(self):
        self.assertIn(f'DEFAULT_ERISC_ELF_DIR="{ELF_DIR}"', (SUITE_DIR / "run_diag_docker.sh").read_text())


class TestSummary(unittest.TestCase):
    def test_every_core_read(self):
        status, details, data = summarize_erisc_debug(dump_result(GALAXY, {"v1.14.0"}))
        self.assertEqual(status, PASS, details)
        self.assertIn("448 ETH cores read on 32 device(s) with eth_init-v1.14.0.elf", details)
        self.assertIn("0 with FW sections in L1 that differ from the ELF", details)
        self.assertEqual(data["postcodes"]["eth"], {"POSTCODE_ETH_INIT_PASS": 448})

    def test_missing_elf_names_what_is_needed(self):
        status, details, _ = summarize_erisc_debug(dump_result(GALAXY, set()))
        self.assertEqual(status, SKIP)
        self.assertIn(f"ETH FW v1.14.0 on 32 device(s) needs eth_init-v1.14.0.elf: not found in {ELF_DIR}", details)

    def test_some_chips_without_their_elf(self):
        status, details, _ = summarize_erisc_debug(
            dump_result({"v1.14.0": list(range(30)), "v1.13.0": [30, 31]}, {"v1.14.0"})
        )
        self.assertEqual(status, WARN)
        self.assertIn("420 ETH cores read on 30 device(s)", details)
        self.assertIn("not read: ETH FW v1.13.0 on 2 device(s) needs eth_init-v1.13.0.elf", details)

    def test_unreadable_cores_and_differing_sections(self):
        result = dump_result(GALAXY, {"v1.14.0"})
        result["cores"][0] = core(0, 0, error="RuntimeError: NOC read timed out")
        result["cores"][1] = core(0, 1, failed=[".text"])
        status, details, data = summarize_erisc_debug(result)
        self.assertEqual(status, WARN)
        self.assertIn("447 ETH cores read", details)
        self.assertIn("1 with FW sections in L1 that differ", details)
        self.assertIn("1 core(s) could not be read", details)
        self.assertEqual(data["sections_differ"], ["dev 0 1-1 (e0,1)"])


# Stands in for the triage venv's python: leaves its arguments in "ran", copies
# result.json beside it to --json, and exits with the code in "rc".
FAKE_PYTHON = textwrap.dedent(
    """\
    #!/bin/sh
    here="$(dirname "$0")"
    echo "$@" > "$here/ran"
    echo "visible=${TT_METAL_VISIBLE_DEVICES:-unset}"
    [ -f "$here/stderr" ] && cat "$here/stderr" >&2
    for a in "$@"; do case "$a" in --json=*) json="${a#--json=}" ;; esac; done
    [ -f "$here/result.json" ] && cp "$here/result.json" "$json"
    exit "$(cat "$here/rc" 2>/dev/null || echo 0)"
    """
)


class TestRunEriscDebug(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.tt_metal, self.sys_root, self.logs = root / "tt-metal", root / "sys", root / "out" / "logs"
        self.bin = self.tt_metal / "python_env" / "bin"
        self.bin.mkdir(parents=True)
        (self.bin / "python").write_text(FAKE_PYTHON)
        (self.bin / "python").chmod(0o755)

    def given(self, result: dict | None = None, rc: int = 0, stderr: str = "") -> None:
        if result is not None:
            (self.bin / "result.json").write_text(json.dumps(result))
        (self.bin / "rc").write_text(str(rc))
        if stderr:
            (self.bin / "stderr").write_text(stderr)

    def run_phase(self, dry_run: bool = False):
        phase = Phase(name="erisc_debug", gates=False)
        with mock.patch.dict("os.environ", {"TT_METAL_VISIBLE_DEVICES": "0"}):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_erisc_debug(self.tt_metal, phase, dry_run, self.logs, sys_root=self.sys_root)
        self.assertEqual(len(phase.checks), 1, phase.checks)
        return phase.checks[0]

    def test_dump_kept_beside_the_report(self):
        fake_sysfs(self.sys_root)
        self.given(dump_result(GALAXY, {"v1.14.0"}))
        check = self.run_phase()
        self.assertEqual(check.status, PASS, check.details)
        self.assertEqual(
            (self.bin / "ran").read_text().split(),
            ["-u", "-m", "diag.erisc_debug", f"--elf-dir={ELF_DIR}", f"--json={self.logs / 'erisc_debug.json'}"],
        )
        # Every chip is read, whatever the caller left visible to metal.
        self.assertIn("visible=unset", (self.logs / "erisc_debug.txt").read_text())
        self.assertTrue((self.logs / "erisc_debug.log").is_file())

    def test_missing_elf_skips_naming_it(self):
        fake_sysfs(self.sys_root)
        self.given(dump_result(GALAXY, set()))
        check = self.run_phase()
        self.assertEqual(check.status, SKIP)
        self.assertIn(f"needs eth_init-v1.14.0.elf: not found in {ELF_DIR}", check.details)

    def test_no_result_skips_with_the_reason(self):
        fake_sysfs(self.sys_root)
        self.given(rc=1, stderr="ERROR: No module named 'ttexalens'; install tools/triage/requirements.txt")
        check = self.run_phase()
        self.assertEqual(check.status, SKIP)
        self.assertIn("rc=1 but no result written: ERROR: No module named 'ttexalens'", check.details)

    def test_dropped_chip_opens_none(self):
        fake_sysfs(self.sys_root, {"0000:46:00.0": (0xFFFF, 0xFFFF)})
        self.given(dump_result(GALAXY, {"v1.14.0"}))
        check = self.run_phase()
        self.assertEqual(check.status, SKIP)
        self.assertIn("chips not opened: ", check.details)
        self.assertFalse((self.bin / "ran").exists())

    def test_dry_run_runs_nothing(self):
        check = self.run_phase(dry_run=True)
        self.assertEqual(check.status, SKIP)
        self.assertTrue(check.details.startswith("(dry) "))
        self.assertFalse((self.bin / "ran").exists())


class TestDumpProcess(unittest.TestCase):
    def test_without_ttexalens_it_says_so_and_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "erisc_debug.json"
            blocked = (
                "import runpy, sys; sys.modules['ttexalens'] = None; "
                f"sys.argv = ['erisc_debug', '--json={out}']; "
                "runpy.run_module('diag.erisc_debug', run_name='__main__', alter_sys=True)"
            )
            cp = subprocess.run(
                [sys.executable, "-c", blocked], cwd=SUITE_DIR, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(cp.returncode, 1, cp.stderr)
            self.assertTrue(cp.stderr.startswith("ERROR: "), cp.stderr)
            self.assertIn("tools/triage/requirements.txt", cp.stderr)
            self.assertFalse(out.exists())


class TestInTheRun(unittest.TestCase):
    def test_last_phase_and_does_not_gate(self):
        from diag_runner import run_diag

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _rc, report = run_diag(
                    "flash",
                    input_snapshot=d / "missing.json",
                    output=d / "report.json",
                    snapshot_out=d / "snapshot.json",
                    dry_run=True,
                )
        self.assertEqual(list(report["phases"])[-1], "erisc_debug")
        self.assertFalse(report["phases"]["erisc_debug"]["gates"])


if __name__ == "__main__":
    unittest.main()
