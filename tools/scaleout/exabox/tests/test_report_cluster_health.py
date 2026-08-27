#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for report_cluster_health.py and report_adapters.py."""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

TESTS_DIR = Path(__file__).resolve().parent
EXABOX_DIR = TESTS_DIR.parent
for _path in (EXABOX_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fixtures  # noqa: E402
from cluster_health_schema import loads_and_validate, validate_record  # noqa: E402
from report_adapters import reason_for, status_for  # noqa: E402
from analyze_host_health_results import main as analyze_main  # noqa: E402
from report_backfill import Leftover  # noqa: E402
from report_cluster_health import (  # noqa: E402
    RecordRequest,
    _read_optional,
    build_record,
    leftover_namespace,
    main,
    parse_gsd_hostnames,
    parse_rank_bindings_yaml,
    parse_rankfile,
)

REPORT_SOURCE = EXABOX_DIR / "report_cluster_health.py"
ADAPTER_SOURCE = EXABOX_DIR / "report_adapters.py"
BACKFILL_SOURCE = EXABOX_DIR / "report_backfill.py"
ANALYZE_HOST_SOURCE = EXABOX_DIR / "analyze_host_health_results.py"
SCHEMA_SOURCE = EXABOX_DIR / "cluster_health_schema.py"
README_SOURCE = EXABOX_DIR / "README.md"

HOSTS = "bh-glx-110-c01u02,bh-glx-110-c01u08"
TS = "2026-08-19T03:12:00Z"


def _assert_shared_dir_mode(testcase: unittest.TestCase, path: Path) -> None:
    """Group-writable, sticky, not world-writable. Setgid is Linux-only."""
    mode = path.stat().st_mode
    testcase.assertEqual(mode & stat.S_IRWXU, stat.S_IRWXU)
    testcase.assertEqual(mode & stat.S_IRWXG, stat.S_IRWXG)
    testcase.assertEqual(mode & stat.S_IRWXO, 0)
    testcase.assertTrue(mode & stat.S_ISVTX)


def _run(argv: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        if env is None:
            rc = main(argv)
        else:
            with patch.dict("os.environ", env, clear=False):
                rc = main(argv)
    return rc, stdout.getvalue(), stderr.getvalue()


def _base_argv(*extra: str) -> list[str]:
    return [
        "--test-type",
        "physical",
        "--hosts",
        HOSTS,
        "--analyzer-code",
        "1",
        "--artifact-dir",
        fixtures.ARTIFACT_DIR,
        "--ts",
        TS,
        "--dry-run",
        *extra,
    ]


class TestAdapters(unittest.TestCase):
    def test_physical_codes(self):
        self.assertEqual(status_for("physical", 0), "passed")
        self.assertEqual(status_for("physical", 1), "failed")
        self.assertEqual(status_for("physical", 50), "failed")
        self.assertEqual(status_for("physical", 66), "failed")

    def test_fabric_scheme(self):
        self.assertEqual(status_for("fabric", 0), "passed")
        self.assertEqual(status_for("fabric", 3), "failed")
        self.assertEqual(status_for("fabric", 50), "degraded")
        self.assertEqual(status_for("fabric", 66), "skipped")

    def test_dispatch_codes(self):
        self.assertEqual(status_for("dispatch", 0), "passed")
        self.assertEqual(status_for("dispatch", 1), "failed")

    def test_recover_codes(self):
        self.assertEqual(status_for("recover", None), "passed")
        self.assertEqual(status_for("recover", 0), "passed")
        self.assertEqual(status_for("recover", 1), "failed")

    def test_host_codes(self):
        self.assertEqual(status_for("host", 0), "passed")
        self.assertEqual(status_for("host", 1), "failed")
        self.assertEqual(status_for("host", 2), "degraded")
        self.assertEqual(status_for("host", 99), "failed")
        with self.assertRaises(ValueError):
            status_for("host", None)

    def test_unknown_test_type(self):
        with self.assertRaises(ValueError):
            status_for("ccl", 0)
        with self.assertRaises(ValueError):
            reason_for("ccl", 1)

    def test_physical_reasons(self):
        expected = {
            1: "Unhealthy links (repeated - same links failing)",
            2: "Unhealthy links (scattered - failures across different links)",
            3: "DRAM training failures",
            4: "Missing connections",
            5: "Extra connections",
            6: "Missing global connection",
            7: "FSD configuration error",
            8: "MGD topology mismatch",
            9: "Workload timeout",
            10: "ARC timeout",
            11: "AICLK timeout",
            12: "Network error",
            13: "Device init error (missing devices)",
            50: "Inconclusive - unrecognized errors",
            66: "No log files found to analyze",
        }
        for code, reason in expected.items():
            with self.subTest(code=code):
                self.assertEqual(reason_for("physical", code), reason)
        self.assertEqual(reason_for("physical", 99), "Unknown physical analysis error (rc=99)")

    def test_fabric_reasons(self):
        expected = {
            1: "MGD error (topology mismatch)",
            2: "Firmware initialization failed",
            3: "Fabric router sync timeout",
            4: "Test hanging (incomplete log)",
            5: "NOC address conflict",
            6: "Ethernet core timeout",
            50: "Inconclusive (manual review required)",
            66: "Input error (log file not found)",
        }
        for code, reason in expected.items():
            with self.subTest(code=code):
                self.assertEqual(reason_for("fabric", code), reason)
        self.assertEqual(reason_for("fabric", 99), "Unknown fabric analysis error (rc=99)")

    def test_dispatch_recover_and_host_reasons(self):
        self.assertEqual(reason_for("dispatch", 1), "One or more dispatch tests failed")
        self.assertEqual(reason_for("dispatch", 66), "No dispatch test log file found")
        self.assertEqual(reason_for("recover", 7), "Recover failed (rc=7)")
        self.assertEqual(reason_for("host", 1), "Diagnostic failed")
        self.assertEqual(reason_for("host", 2), "Diagnostic warning")
        self.assertEqual(reason_for("host", 99), "Diagnostic failed (code=99)")

    def test_pass_has_no_reason(self):
        for test_type in ("physical", "fabric", "dispatch", "recover", "host"):
            with self.subTest(test_type=test_type):
                self.assertEqual(reason_for(test_type, 0), "")
        self.assertEqual(reason_for("recover", None), "")


class TestDryRunCli(unittest.TestCase):
    def test_one_stdout_line_validates(self):
        rc, out, err = _run(_base_argv())
        self.assertEqual(rc, 0, err)
        lines = [ln for ln in out.splitlines() if ln]
        self.assertEqual(len(lines), 1)
        record = loads_and_validate(lines[0], file_written=False)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["analyzer_code"], 1)
        self.assertEqual(record["labels"]["failure_reason"], "Unhealthy links (repeated - same links failing)")
        self.assertNotIn("record_id", record)
        self.assertNotIn("topology", record)

    def test_explicit_failure_reason_wins(self):
        rc, out, err = _run(_base_argv("--label", "failure_reason=Operator diagnosis"))
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["labels"]["failure_reason"], "Operator diagnosis")

    def test_labels_only_under_labels(self):
        rc, out, err = _run(_base_argv("--label", "quad=110-C-Q1", "--label", "superpod=SC36_3"))
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["labels"]["quad"], "110-C-Q1")
        self.assertEqual(record["labels"]["superpod"], "SC36_3")
        self.assertNotIn("quad", record.get("topology", {}))
        validate_record(record, file_written=False)

    def test_recover_omits_analyzer_code(self):
        rc, out, err = _run(
            [
                "--test-type",
                "recover",
                "--hosts",
                "bh-glx-110-c01u02",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertNotIn("analyzer_code", record)
        self.assertEqual(record["status"], "passed")
        self.assertNotIn("failure_reason", record.get("labels", {}))
        validate_record(record, file_written=False)

    def test_recover_failure_has_reason_but_omits_analyzer_code(self):
        rc, out, err = _run(
            [
                "--test-type",
                "recover",
                "--hosts",
                "bh-glx-110-c01u02",
                "--analyzer-code",
                "7",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertNotIn("analyzer_code", record)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["labels"]["failure_reason"], "Recover failed (rc=7)")
        validate_record(record, file_written=False)

    def test_missing_analyzer_code_for_physical(self):
        argv = [
            "--test-type",
            "physical",
            "--hosts",
            HOSTS,
            "--artifact-dir",
            fixtures.ARTIFACT_DIR,
            "--dry-run",
        ]
        rc, _out, err = _run(argv)
        self.assertEqual(rc, 1)
        self.assertIn("analyzer_code", err)

    def test_rejects_topology_flag(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(_base_argv("--topology", "ignored.yaml"))
        self.assertIn("--topology is not accepted", stderr.getvalue())


class TestPortableTopology(unittest.TestCase):
    def _descriptor(self, name: str, text: str) -> str:
        return str(fixtures.temp_file(self, name, text))

    def test_cabling_and_deployment(self):
        rc, out, err = _run(
            _base_argv(
                "--cabling",
                self._descriptor("cabling.textproto", fixtures.CABLING_TWO_HOST),
                "--deployment",
                self._descriptor("deployment.textproto", fixtures.DEPLOYMENT_TWO_HOST),
            )
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        topo = record["topology"]
        self.assertIn("bh_galaxy_sp_0/node_0", topo["instance_paths"])
        self.assertIn("bh_galaxy_sp_0/node_1", topo["instance_paths"])
        hosts = {item["hostname"] for item in topo["physical"]}
        self.assertEqual(hosts, {"bh-glx-110-c01u02", "bh-glx-110-c01u08"})
        self.assertEqual(topo["physical"][0]["aisle"], "C")
        validate_record(record, file_written=False)

    def test_gsd_host_keys(self):
        rc, out, err = _run(_base_argv("--gsd", self._descriptor("gsd.yaml", fixtures.GSD_TWO_HOST)))
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        hosts = {item["hostname"] for item in record["topology"]["physical"]}
        self.assertEqual(hosts, {"bh-glx-110-c01u02", "bh-glx-110-c01u08"})
        self.assertEqual(
            parse_gsd_hostnames(fixtures.GSD_TWO_HOST),
            ["bh-glx-110-c01u02", "bh-glx-110-c01u08"],
        )

    def test_rankfile_and_bindings(self):
        rc, out, err = _run(
            _base_argv(
                "--rankfile",
                self._descriptor("rankfile.txt", fixtures.RANKFILE_TWO_HOST),
                "--rank-bindings",
                self._descriptor("rank_bindings.yaml", fixtures.RANK_BINDINGS_TWO_HOST),
            )
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        bindings = record["topology"]["rank_bindings"]
        self.assertEqual(len(bindings), 2)
        self.assertEqual(bindings[0]["host"], "bh-glx-110-c01u02")
        self.assertEqual(bindings[1]["mesh_id"], 0)
        self.assertEqual(parse_rankfile(fixtures.RANKFILE_TWO_HOST)[0], "bh-glx-110-c01u02")
        parsed = parse_rank_bindings_yaml(fixtures.RANK_BINDINGS_TWO_HOST)
        self.assertEqual(parsed[1]["mesh_host_rank"], 1)

    def test_bindings_without_mesh_host_rank(self):
        rc, out, err = _run(
            _base_argv(
                "--rankfile",
                self._descriptor("rankfile.txt", fixtures.RANKFILE_TWO_HOST),
                "--rank-bindings",
                self._descriptor("rank_bindings.yaml", fixtures.RANK_BINDINGS_NO_HOST_RANK),
            )
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        bindings = record["topology"]["rank_bindings"]
        self.assertEqual(len(bindings), 2)
        self.assertNotIn("mesh_host_rank", bindings[0])
        self.assertEqual(bindings[1]["mesh_id"], 1)
        self.assertEqual(bindings[1]["host"], "bh-glx-110-c01u08")
        validate_record(record, file_written=False)

    def test_nested_block_skipped(self):
        yaml = """\
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    extra_nested:
      deep_key: ignored
      nested:
        deeper: also_ignored
    host: h1
  - rank: 1
    mesh_id: 1
    extra_nested:
      deep_key: ignored
    host: h2
"""
        result = parse_rank_bindings_yaml(yaml)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0],
            {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0, "host": "h1"},
        )
        self.assertEqual(result[1], {"rank": 1, "mesh_id": 1, "host": "h2"})
        self.assertNotIn("extra_nested", result[0])
        self.assertNotIn("deep_key", result[0])

    def test_fsd_physical(self):
        rc, out, err = _run(_base_argv("--fsd", self._descriptor("fsd.textproto", fixtures.FSD_TWO_HOST)))
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["topology"]["physical"][1]["shelf_u"], 8)

    def test_hosts_only_omits_topology(self):
        rc, out, err = _run(_base_argv())
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertNotIn("topology", record)


class TestStoreWrite(unittest.TestCase):
    def test_writes_record_and_stdout_has_uri(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--test-type",
                "physical",
                "--hosts",
                HOSTS,
                "--analyzer-code",
                "0",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--store-root",
                tmp,
            ]
            rc, out, err = _run(argv)
            self.assertEqual(rc, 0, err)
            record = loads_and_validate(out.strip(), file_written=True)
            dest = Path(record["record_uri"])
            self.assertTrue(dest.is_file())
            self.assertTrue(str(dest).startswith("/"))
            self.assertEqual(dest.parent.name, "2026-08-19")
            on_disk = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["record_id"], record["record_id"])
            _assert_shared_dir_mode(self, dest.parent)

    def test_date_dir_chmod_repairs_umask_masked_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            date_dir = Path(tmp) / "2026-08-19"
            date_dir.mkdir()
            os.chmod(date_dir, 0o755)
            argv = [
                "--test-type",
                "physical",
                "--hosts",
                HOSTS,
                "--analyzer-code",
                "0",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--store-root",
                tmp,
            ]
            rc, out, err = _run(argv)
            self.assertEqual(rc, 0, err)
            _assert_shared_dir_mode(self, date_dir)

    def test_creates_missing_store_parents_group_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "nested" / "store"
            argv = [
                "--test-type",
                "physical",
                "--hosts",
                HOSTS,
                "--analyzer-code",
                "0",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--store-root",
                str(store),
            ]
            rc, out, err = _run(argv)
            self.assertEqual(rc, 0, err)
            date_dir = store / "2026-08-19"
            self.assertTrue(date_dir.is_dir())
            _assert_shared_dir_mode(self, store)
            _assert_shared_dir_mode(self, Path(tmp) / "nested")
            _assert_shared_dir_mode(self, date_dir)
            self.assertEqual(date_dir.stat().st_gid, store.stat().st_gid)

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err = _run(_base_argv("--store-root", tmp))
            self.assertEqual(rc, 0, err)
            record = json.loads(out.strip())
            self.assertNotIn("record_id", record)
            self.assertEqual(list(Path(tmp).rglob("*.json")), [])

    def test_identical_retry_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--test-type",
                "physical",
                "--hosts",
                HOSTS,
                "--analyzer-code",
                "0",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--store-root",
                tmp,
            ]
            rc1, out1, err1 = _run(argv)
            self.assertEqual(rc1, 0, err1)
            dest = Path(json.loads(out1.strip())["record_uri"])
            mtime = dest.stat().st_mtime_ns
            rc2, out2, err2 = _run(argv)
            self.assertEqual(rc2, 0, err2)
            self.assertNotIn("refusing to overwrite", err2)
            self.assertEqual(dest.stat().st_mtime_ns, mtime)
            self.assertEqual(json.loads(out2.strip())["record_id"], json.loads(out1.strip())["record_id"])

    def test_different_content_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--test-type",
                "physical",
                "--hosts",
                HOSTS,
                "--analyzer-code",
                "0",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
                "--store-root",
                tmp,
                "--label",
                "quad=first",
            ]
            rc1, out1, err1 = _run(argv)
            self.assertEqual(rc1, 0, err1)
            dest = Path(json.loads(out1.strip())["record_uri"])
            original = dest.read_text(encoding="utf-8")
            argv[-1] = "quad=second"
            rc2, out2, err2 = _run(argv)
            self.assertEqual(rc2, 0, err2)
            self.assertIn("refusing to overwrite", err2)
            self.assertEqual(dest.read_text(encoding="utf-8"), original)
            second = json.loads(out2.strip())
            self.assertNotIn("record_id", second)
            validate_record(second, file_written=False)

    def test_env_store_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--test-type",
                "physical",
                "--hosts",
                HOSTS,
                "--analyzer-code",
                "0",
                "--artifact-dir",
                fixtures.ARTIFACT_DIR,
                "--ts",
                TS,
            ]
            rc, out, err = _run(argv, env={"CLUSTER_HEALTH_STORE_ROOT": tmp})
            self.assertEqual(rc, 0, err)
            loads_and_validate(out.strip(), file_written=True)


class TestFromDiagReport(unittest.TestCase):
    def _diag(self, text: str) -> str:
        return str(fixtures.temp_file(self, "diag_report.json", text))

    def test_pass_fills_clocks_and_labels(self):
        rc, out, err = _run(
            [
                "--from-diag-report",
                self._diag(fixtures.DIAG_REPORT_PASS),
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0, err)
        record = loads_and_validate(out.strip(), file_written=False)
        self.assertEqual(record["test_type"], "host")
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["analyzer_code"], 0)
        self.assertEqual(record["hosts"], ["bh-glx-110-c01u02"])
        self.assertEqual(record["ts"], "2026-08-19T03:12:36Z")
        self.assertEqual(record["duration_s"], 36.5)
        self.assertEqual(record["labels"]["tier"], "light")
        self.assertEqual(record["labels"]["board_rev"], "RevC")
        self.assertTrue(record["artifact_uri"])

    def test_warn_is_degraded(self):
        rc, out, err = _run(["--from-diag-report", self._diag(fixtures.DIAG_REPORT_WARN), "--dry-run"])
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["status"], "degraded")
        self.assertEqual(record["analyzer_code"], 2)

    def test_fail_is_failed(self):
        rc, out, err = _run(["--from-diag-report", self._diag(fixtures.DIAG_REPORT_FAIL), "--dry-run"])
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["analyzer_code"], 1)
        self.assertEqual(record["hosts"], ["bh-glx-110-c01u08"])
        self.assertEqual(record["labels"]["tier"], "medium")
        self.assertNotIn("board_rev", record["labels"])

    def test_refuses_diag_dry_run(self):
        rc, _out, err = _run(["--from-diag-report", self._diag(fixtures.DIAG_REPORT_DRY_RUN), "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertIn("dry-run", err)

    def test_cannot_combine_with_test_type(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(
                [
                    "--from-diag-report",
                    self._diag(fixtures.DIAG_REPORT_PASS),
                    "--test-type",
                    "host",
                ]
            )
        self.assertIn("--from-diag-report cannot be combined", stderr.getvalue())


class TestAnalyzeHostHealthCli(unittest.TestCase):
    def _diag(self, text: str) -> str:
        return str(fixtures.temp_file(self, "diag_report.json", text))

    def test_prints_analysis_exit_code(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = analyze_main(["--json", self._diag(fixtures.DIAG_REPORT_WARN)])
        self.assertEqual(rc, 2)
        self.assertIn("Analysis exit code: 2", stdout.getvalue())
        self.assertIn("ts=2026-08-19T03:12:10Z", stdout.getvalue())
        self.assertIn("duration_s=10.0", stdout.getvalue())

    def test_refuses_dry_run_report(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = analyze_main(["--json", self._diag(fixtures.DIAG_REPORT_DRY_RUN)])
        self.assertEqual(rc, 1)
        self.assertIn("dry-run", stderr.getvalue())


class TestRecordRequest(unittest.TestCase):
    def test_leftover_namespace_builds_record_request(self):
        leftover = Leftover(
            test_type="physical",
            hosts=HOSTS,
            analyzer_code=1,
            artifact_dir=fixtures.ARTIFACT_DIR,
            ts=TS,
            mtime=datetime.now(timezone.utc),
            source=Path("unused.log"),
        )
        args = argparse.Namespace(
            cabling=None,
            deployment=None,
            fsd=None,
            gsd=None,
            rankfile=None,
            rank_bindings=None,
            source=None,
            triggered_by="operator",
            trigger_kind=None,
            orchestrator_id=None,
            cluster="exabox",
            label=["quad=110-C-Q1"],
        )
        request = leftover_namespace(leftover, args)
        self.assertIsInstance(request, RecordRequest)
        self.assertEqual(request.source, "backfill")
        self.assertEqual(request.trigger_kind, "backfill")
        request.label = ["superpod=SC36_3"]
        record = build_record(request)
        self.assertEqual(record["cluster"], "exabox")
        self.assertEqual(record["labels"]["superpod"], "SC36_3")
        self.assertNotIn("quad", record["labels"])


class TestReadOptional(unittest.TestCase):
    def test_missing_path_returns_none(self):
        self.assertIsNone(_read_optional(None))
        self.assertIsNone(_read_optional(""))

    def test_warns_on_unexpected_exception(self):
        stderr = io.StringIO()
        with patch(
            "report_cluster_health.read_descriptor_text",
            side_effect=RuntimeError("boom"),
        ), redirect_stderr(stderr):
            self.assertIsNone(_read_optional("/tmp/descriptor.textproto"))
        self.assertIn("could not read /tmp/descriptor.textproto: boom", stderr.getvalue())


class TestNoSiteCoupling(unittest.TestCase):
    def test_report_modules_omit_forbidden_strings(self):
        for path in (
            REPORT_SOURCE,
            ADAPTER_SOURCE,
            BACKFILL_SOURCE,
            ANALYZE_HOST_SOURCE,
            SCHEMA_SOURCE,
            README_SOURCE,
        ):
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            self.assertNotIn("topology.yaml", text)
            for needle in ("stackstorm", "kubernetes", "k8s"):
                self.assertNotIn(needle, lowered)


if __name__ == "__main__":
    unittest.main()
