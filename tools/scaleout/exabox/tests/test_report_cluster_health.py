#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for report_cluster_health.py and report_adapters.py."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

EXABOX_DIR = Path(__file__).resolve().parents[1]
if str(EXABOX_DIR) not in sys.path:
    sys.path.insert(0, str(EXABOX_DIR))

from cluster_health_schema import loads_and_validate, validate_record  # noqa: E402
from report_adapters import status_for  # noqa: E402
from analyze_host_health_results import main as analyze_main  # noqa: E402
from report_cluster_health import (  # noqa: E402
    main,
    parse_gsd_hostnames,
    parse_rank_bindings_yaml,
    parse_rankfile,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPORT_SOURCE = EXABOX_DIR / "report_cluster_health.py"
ADAPTER_SOURCE = EXABOX_DIR / "report_adapters.py"
BACKFILL_SOURCE = EXABOX_DIR / "report_backfill.py"
ANALYZE_HOST_SOURCE = EXABOX_DIR / "analyze_host_health_results.py"

HOSTS = "bh-glx-110-c01u02,bh-glx-110-c01u08"
TS = "2026-08-19T03:12:00Z"


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
        str(FIXTURES),
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


class TestDryRunCli(unittest.TestCase):
    def test_one_stdout_line_validates(self):
        rc, out, err = _run(_base_argv())
        self.assertEqual(rc, 0, err)
        lines = [ln for ln in out.splitlines() if ln]
        self.assertEqual(len(lines), 1)
        record = loads_and_validate(lines[0], file_written=False)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["analyzer_code"], 1)
        self.assertNotIn("record_id", record)
        self.assertNotIn("topology", record)

    def test_labels_only_under_labels(self):
        rc, out, err = _run(
            _base_argv("--label", "quad=110-C-Q1", "--label", "superpod=SC36_3")
        )
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
                str(FIXTURES),
                "--ts",
                TS,
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertNotIn("analyzer_code", record)
        self.assertEqual(record["status"], "passed")
        validate_record(record, file_written=False)

    def test_missing_analyzer_code_for_physical(self):
        argv = [
            "--test-type",
            "physical",
            "--hosts",
            HOSTS,
            "--artifact-dir",
            str(FIXTURES),
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
    def test_cabling_and_deployment(self):
        rc, out, err = _run(
            _base_argv(
                "--cabling",
                str(FIXTURES / "cabling_two_host.textproto"),
                "--deployment",
                str(FIXTURES / "deployment_two_host.textproto"),
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
        rc, out, err = _run(_base_argv("--gsd", str(FIXTURES / "gsd_two_host.yaml")))
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        hosts = {item["hostname"] for item in record["topology"]["physical"]}
        self.assertEqual(hosts, {"bh-glx-110-c01u02", "bh-glx-110-c01u08"})
        self.assertEqual(
            parse_gsd_hostnames((FIXTURES / "gsd_two_host.yaml").read_text()),
            ["bh-glx-110-c01u02", "bh-glx-110-c01u08"],
        )

    def test_rankfile_and_bindings(self):
        rc, out, err = _run(
            _base_argv(
                "--rankfile",
                str(FIXTURES / "rankfile_two_host.txt"),
                "--rank-bindings",
                str(FIXTURES / "rank_bindings_two_host.yaml"),
            )
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        bindings = record["topology"]["rank_bindings"]
        self.assertEqual(len(bindings), 2)
        self.assertEqual(bindings[0]["host"], "bh-glx-110-c01u02")
        self.assertEqual(bindings[1]["mesh_id"], 0)
        self.assertEqual(parse_rankfile((FIXTURES / "rankfile_two_host.txt").read_text())[0], "bh-glx-110-c01u02")
        parsed = parse_rank_bindings_yaml((FIXTURES / "rank_bindings_two_host.yaml").read_text())
        self.assertEqual(parsed[1]["mesh_host_rank"], 1)

    def test_fsd_physical(self):
        rc, out, err = _run(_base_argv("--fsd", str(FIXTURES / "fsd_two_host.textproto")))
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
                str(FIXTURES),
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
                str(FIXTURES),
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
                str(FIXTURES),
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
                str(FIXTURES),
                "--ts",
                TS,
            ]
            rc, out, err = _run(argv, env={"CLUSTER_HEALTH_STORE_ROOT": tmp})
            self.assertEqual(rc, 0, err)
            loads_and_validate(out.strip(), file_written=True)


class TestFromDiagReport(unittest.TestCase):
    def test_pass_fills_clocks_and_labels(self):
        rc, out, err = _run(
            [
                "--from-diag-report",
                str(FIXTURES / "diag_report_pass.json"),
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
        rc, out, err = _run(
            ["--from-diag-report", str(FIXTURES / "diag_report_warn.json"), "--dry-run"]
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["status"], "degraded")
        self.assertEqual(record["analyzer_code"], 2)

    def test_fail_is_failed(self):
        rc, out, err = _run(
            ["--from-diag-report", str(FIXTURES / "diag_report_fail.json"), "--dry-run"]
        )
        self.assertEqual(rc, 0, err)
        record = json.loads(out.strip())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["analyzer_code"], 1)
        self.assertEqual(record["hosts"], ["bh-glx-110-c01u08"])
        self.assertEqual(record["labels"]["tier"], "medium")
        self.assertNotIn("board_rev", record["labels"])

    def test_refuses_diag_dry_run(self):
        rc, _out, err = _run(
            ["--from-diag-report", str(FIXTURES / "diag_report_dry_run.json"), "--dry-run"]
        )
        self.assertEqual(rc, 1)
        self.assertIn("dry-run", err)

    def test_cannot_combine_with_test_type(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(
                [
                    "--from-diag-report",
                    str(FIXTURES / "diag_report_pass.json"),
                    "--test-type",
                    "host",
                ]
            )
        self.assertIn("--from-diag-report cannot be combined", stderr.getvalue())


class TestAnalyzeHostHealthCli(unittest.TestCase):
    def test_prints_analysis_exit_code(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = analyze_main(["--json", str(FIXTURES / "diag_report_warn.json")])
        self.assertEqual(rc, 2)
        self.assertIn("Analysis exit code: 2", stdout.getvalue())
        self.assertIn("ts=2026-08-19T03:12:10Z", stdout.getvalue())
        self.assertIn("duration_s=10.0", stdout.getvalue())

    def test_refuses_dry_run_report(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = analyze_main(["--json", str(FIXTURES / "diag_report_dry_run.json")])
        self.assertEqual(rc, 1)
        self.assertIn("dry-run", stderr.getvalue())


class TestNoSiteCoupling(unittest.TestCase):
    def test_report_modules_omit_forbidden_strings(self):
        for path in (REPORT_SOURCE, ADAPTER_SOURCE, BACKFILL_SOURCE, ANALYZE_HOST_SOURCE):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("topology.yaml", text)
            self.assertNotIn("/data/stackstorm/cluster-health", text)


if __name__ == "__main__":
    unittest.main()
