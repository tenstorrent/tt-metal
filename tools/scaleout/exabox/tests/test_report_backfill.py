#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Backfill leftover discovery and concurrent store publish tests."""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

EXABOX_DIR = Path(__file__).resolve().parents[1]
if str(EXABOX_DIR) not in sys.path:
    sys.path.insert(0, str(EXABOX_DIR))

from cluster_health_schema import SCHEMA_ID, loads_and_validate, validate_record  # noqa: E402
from report_backfill import (  # noqa: E402
    discover_leftovers,
    leftover_from_log,
    parse_recover_code_from_text,
)
from report_cluster_health import main  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TREE = FIXTURES / "artifact_tree"


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(argv)
    return rc, stdout.getvalue(), stderr.getvalue()


def _backfill_argv(*extra: str) -> list[str]:
    return [
        "--from-artifact-dir",
        str(TREE),
        "--triggered-by",
        "operator",
        "--dry-run",
        *extra,
    ]


def _publish_worker(exabox_dir: str, store_root: str, hosts: str, ts: str, artifact: str) -> str:
    if exabox_dir not in sys.path:
        sys.path.insert(0, exabox_dir)
    from report_cluster_health import publish_record as publish

    record = {
        "schema": SCHEMA_ID,
        "ts": ts,
        "test_type": "physical",
        "status": "passed",
        "hosts": [hosts],
        "analyzer_code": 0,
        "artifact_uri": artifact,
    }
    published = publish(record, store_root)
    return published["record_uri"]


class TestBackfillDiscover(unittest.TestCase):
    def test_json_sidecar_wins_over_footer(self):
        log_path = TREE / "logs" / "physical_validation-20260819T031200Z.log"
        leftover = leftover_from_log("physical", log_path, TREE)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 0)

    def test_skips_log_without_hosts(self):
        log_path = TREE / "logs" / "physical_validation-20260819T040000Z.log"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            leftover = leftover_from_log("physical", log_path, TREE)
        self.assertIsNone(leftover)
        self.assertIn("no hosts", stderr.getvalue())

    def test_discover_four_complete_leftovers(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            leftovers = discover_leftovers(TREE)
        kinds = {item.test_type for item in leftovers}
        self.assertEqual(kinds, {"physical", "fabric", "dispatch", "recover"})
        self.assertEqual(len(leftovers), 4)
        self.assertIn("no hosts", stderr.getvalue())
        self.assertNotIn("no wrapper log", stderr.getvalue())


class TestBackfillCli(unittest.TestCase):
    def test_dry_run_stdout_lines_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err = _run(_backfill_argv("--store-root", tmp))
            self.assertEqual(rc, 0, err)
            lines = [ln for ln in out.splitlines() if ln]
            self.assertEqual(len(lines), 4)
            records = [loads_and_validate(ln, file_written=False) for ln in lines]
            by_type = {rec["test_type"]: rec for rec in records}
            self.assertEqual(by_type["physical"]["analyzer_code"], 0)
            self.assertEqual(by_type["physical"]["status"], "passed")
            self.assertEqual(by_type["fabric"]["analyzer_code"], 3)
            self.assertEqual(by_type["fabric"]["status"], "failed")
            self.assertEqual(by_type["dispatch"]["status"], "failed")
            self.assertNotIn("analyzer_code", by_type["recover"])
            self.assertEqual(by_type["recover"]["status"], "passed")
            self.assertEqual(by_type["physical"]["source"], "backfill")
            self.assertEqual(by_type["physical"]["trigger_kind"], "backfill")
            self.assertEqual(by_type["physical"]["triggered_by"], "operator")
            self.assertEqual(list(Path(tmp).rglob("*.json")), [])

    def test_write_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--from-artifact-dir",
                str(TREE),
                "--triggered-by",
                "operator",
                "--store-root",
                tmp,
            ]
            rc, out, err = _run(argv)
            self.assertEqual(rc, 0, err)
            lines = [ln for ln in out.splitlines() if ln]
            self.assertEqual(len(lines), 4)
            uris = []
            for line in lines:
                record = loads_and_validate(line, file_written=True)
                dest = Path(record["record_uri"])
                self.assertTrue(dest.is_file())
                uris.append(dest)
            self.assertEqual(len(set(uris)), 4)

    def test_mtime_window_drops_old(self):
        recover = TREE / "logs" / "recover-20260819T031000Z.log"
        others = [
            TREE / "logs" / "physical_validation-20260819T031200Z.log",
            TREE / "logs" / "fabric_tests-20260819T031300Z.log",
            TREE / "logs" / "dispatch_tests-20260819T031400Z.log",
        ]
        old = datetime(2026, 8, 1, 12, tzinfo=timezone.utc).timestamp()
        recent = datetime(2026, 8, 19, 12, tzinfo=timezone.utc).timestamp()
        os.utime(recover, (old, old))
        for path in others:
            os.utime(path, (recent, recent))
        rc, out, err = _run(_backfill_argv("--from", "2026-08-10", "--to", "2026-08-20"))
        self.assertEqual(rc, 0, err)
        types = {json.loads(ln)["test_type"] for ln in out.splitlines() if ln}
        self.assertEqual(types, {"physical", "fabric", "dispatch"})
        self.assertNotIn("recover", types)

    def test_store_root_required_without_dry_run(self):
        rc, _out, err = _run(
            [
                "--from-artifact-dir",
                str(TREE),
                "--triggered-by",
                "operator",
            ]
        )
        self.assertEqual(rc, 1)
        self.assertIn("store-root", err)

    def test_triggered_by_required(self):
        rc, _out, err = _run(
            [
                "--from-artifact-dir",
                str(TREE),
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 1)
        self.assertIn("triggered-by", err)


class TestConcurrentStore(unittest.TestCase):
    def test_two_writers_different_record_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context("spawn")
            jobs = [
                (str(EXABOX_DIR), tmp, "bh-glx-110-c01u02", "2026-08-19T03:12:00Z", "/tmp/a"),
                (str(EXABOX_DIR), tmp, "bh-glx-110-c01u08", "2026-08-19T03:13:00Z", "/tmp/b"),
            ]
            with ctx.Pool(2) as pool:
                uris = pool.starmap(_publish_worker, jobs)
            paths = [Path(uri) for uri in uris]
            self.assertEqual(len(paths), 2)
            self.assertNotEqual(paths[0].name, paths[1].name)
            for path in paths:
                self.assertTrue(path.is_file())
                on_disk = json.loads(path.read_text(encoding="utf-8"))
                validate_record(on_disk, file_written=True)
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])


EDGE = FIXTURES / "edge"


class TestRecoverRealFooters(unittest.TestCase):
    def test_failed_even_when_recovery_completed_follows(self):
        leftover = leftover_from_log("recover", EDGE / "recover-failed-real.log", EDGE)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 1)
        self.assertFalse(leftover.incomplete)

    def test_last_success_wins_after_failed_attempt(self):
        leftover = leftover_from_log("recover", EDGE / "recover-succeeded-real.log", EDGE)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 0)
        self.assertFalse(leftover.incomplete)

    def test_recovery_completed_alone_is_not_success(self):
        text = (
            "=== Recover - x ===\nHOSTS=h1\n"
            "Recovery completed at Wed Aug 19 05:04:23 UTC 2026\n"
        )
        self.assertIsNone(parse_recover_code_from_text(text))

    def test_incomplete_recover_is_degraded(self):
        leftover = leftover_from_log("recover", EDGE / "recover-incomplete.log", EDGE)
        self.assertIsNotNone(leftover)
        self.assertTrue(leftover.incomplete)
        self.assertIsNone(leftover.analyzer_code)


class TestIncompleteAndLargeLogs(unittest.TestCase):
    def test_truncated_wrapper_emits_degraded(self):
        leftover = leftover_from_log(
            "physical", EDGE / "physical_validation-truncated.log", EDGE
        )
        self.assertIsNotNone(leftover)
        self.assertTrue(leftover.incomplete)
        rc, out, err = _run(
            [
                "--from-artifact-dir",
                str(self._tree_with(EDGE / "physical_validation-truncated.log")),
                "--triggered-by",
                "operator",
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0, err)
        rec = loads_and_validate(out.splitlines()[0], file_written=False)
        self.assertEqual(rec["status"], "degraded")
        self.assertNotIn("analyzer_code", rec)
        self.assertEqual(rec["labels"]["incomplete"], "true")
        self.assertEqual(rec["labels"]["incomplete_reason"], "missing_terminal_outcome")
        self.assertIn("degraded=1", err)

    def test_large_log_footer_beyond_64k(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            dest = logs / "physical_validation-20260819T060000Z.log"
            header = (
                "=== Physical Validation - 20260819T060000Z ===\n"
                "HOSTS=bh-glx-110-c01u02\n"
                "OUTPUT_DIR=/tmp/huge\n"
            )
            padding = "x" * 100_000
            dest.write_text(header + padding + "\nAnalysis exit code: 13\n", encoding="utf-8")
            leftover = leftover_from_log("physical", dest, Path(tmp))
            self.assertIsNotNone(leftover)
            self.assertEqual(leftover.analyzer_code, 13)
            self.assertFalse(leftover.incomplete)

    def test_analysis_exit_with_host_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            dest = logs / "physical_validation-20260819T060200Z.log"
            dest.write_text(
                "=== Physical Validation - 20260819T060200Z ===\n"
                "HOSTS=bh-glx-110-c01u02\n"
                "[bh-glx-110-c01u02][06:02:00] Analysis exit code: 7\n",
                encoding="utf-8",
            )
            leftover = leftover_from_log("physical", dest, Path(tmp))
            self.assertIsNotNone(leftover)
            self.assertEqual(leftover.analyzer_code, 7)
            self.assertFalse(leftover.incomplete)

    def test_backfill_cli_keeps_caller_labels(self):
        rc, out, err = _run(_backfill_argv("--label", "superpod=SC16_1", "--label", "ring=SC16"))
        self.assertEqual(rc, 0, err)
        records = [json.loads(ln) for ln in out.splitlines() if ln]
        self.assertTrue(records)
        for rec in records:
            self.assertEqual(rec["labels"]["superpod"], "SC16_1")
            self.assertEqual(rec["labels"]["ring"], "SC16")

    def test_footer_across_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            dest = logs / "physical_validation-20260819T060100Z.log"
            header = (
                "=== Physical Validation - 20260819T060100Z ===\n"
                "HOSTS=bh-glx-110-c01u02\n"
            )
            encoded_header = header.encode("utf-8")
            footer = b"\nAnalysis exit code: 4\n"
            # Split "Analysis" across the last 64KiB reverse-chunk start.
            prefix = 65536 - 8
            dest.write_bytes(encoded_header + (b"y" * prefix) + footer)
            leftover = leftover_from_log("physical", dest, Path(tmp))
            self.assertIsNotNone(leftover)
            self.assertEqual(leftover.analyzer_code, 4)

    def _tree_with(self, log_path: Path) -> str:
        tmp = tempfile.mkdtemp()
        logs = Path(tmp) / "logs"
        logs.mkdir()
        (logs / log_path.name).write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        return tmp


class TestRecursiveDedup(unittest.TestCase):
    def test_recursive_discovers_nested_logs_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "executions" / "one" / "logs"
            b = Path(tmp) / "nightly" / "SC16_1" / "logs"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            src = TREE / "logs" / "dispatch_tests-20260819T031400Z.log"
            (a / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            fabric = TREE / "logs" / "fabric_tests-20260819T031300Z.log"
            (b / fabric.name).write_text(fabric.read_text(encoding="utf-8"), encoding="utf-8")
            leftovers = discover_leftovers(Path(tmp), recursive=True)
            kinds = {item.test_type for item in leftovers}
            self.assertEqual(kinds, {"dispatch", "fabric"})
            rc, out, err = _run(
                [
                    "--from-artifact-dir",
                    tmp,
                    "--recursive",
                    "--triggered-by",
                    "operator",
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0, err)
            self.assertEqual(len([ln for ln in out.splitlines() if ln]), 2)
            self.assertIn("discovered=2", err)
            self.assertIn("emitted=2", err)

    def test_duplicate_leftovers_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            src = TREE / "logs" / "dispatch_tests-20260819T031400Z.log"
            text = src.read_text(encoding="utf-8")
            (logs / src.name).write_text(text, encoding="utf-8")
            leftovers = discover_leftovers(Path(tmp))
            leftovers = leftovers + leftovers
            from report_backfill import leftover_key

            keys = [leftover_key(item) for item in leftovers]
            self.assertEqual(len(keys), 2)
            self.assertEqual(len(set(keys)), 1)


class TestHostDiagLeftover(unittest.TestCase):
    def test_discover_diag_report_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "diag_report.json"
            dest.write_text((FIXTURES / "diag_report_pass.json").read_text(encoding="utf-8"), encoding="utf-8")
            leftovers = discover_leftovers(Path(tmp))
            self.assertEqual(len(leftovers), 1)
            leftover = leftovers[0]
            self.assertEqual(leftover.test_type, "host")
            self.assertEqual(leftover.hosts, "bh-glx-110-c01u02")
            self.assertEqual(leftover.analyzer_code, 0)
            self.assertEqual(leftover.ts, "2026-08-19T03:12:36Z")
            self.assertEqual(leftover.duration_s, 36.5)
            self.assertEqual(leftover.labels["tier"], "light")
            self.assertEqual(leftover.labels["board_rev"], "RevC")
            rc, out, err = _run(
                [
                    "--from-artifact-dir",
                    tmp,
                    "--triggered-by",
                    "operator",
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0, err)
            record = loads_and_validate(out.splitlines()[0], file_written=False)
            self.assertEqual(record["test_type"], "host")
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["duration_s"], 36.5)
            self.assertEqual(record["labels"]["tier"], "light")

    def test_skips_dry_run_diag_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "diag_report.json"
            dest.write_text((FIXTURES / "diag_report_dry_run.json").read_text(encoding="utf-8"), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                leftovers = discover_leftovers(Path(tmp))
            self.assertEqual(leftovers, [])
            self.assertIn("dry-run", stderr.getvalue())

    def test_cli_duration_s_does_not_stamp_physical(self):
        rc, out, err = _run(_backfill_argv("--duration-s", "99"))
        self.assertEqual(rc, 0, err)
        for line in out.splitlines():
            if not line:
                continue
            rec = json.loads(line)
            self.assertNotIn("duration_s", rec)

    def test_recursive_nested_diag_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "node-a" / "results"
            nested.mkdir(parents=True)
            (nested / "diag_report.json").write_text(
                (FIXTURES / "diag_report_warn.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            leftovers = discover_leftovers(Path(tmp), recursive=True)
            self.assertEqual(len(leftovers), 1)
            self.assertEqual(leftovers[0].analyzer_code, 2)


if __name__ == "__main__":
    unittest.main()
