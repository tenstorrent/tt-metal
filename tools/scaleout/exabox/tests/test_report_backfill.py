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

TESTS_DIR = Path(__file__).resolve().parent
EXABOX_DIR = TESTS_DIR.parent
for _path in (EXABOX_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fixtures  # noqa: E402
from cluster_health_schema import SCHEMA_ID, loads_and_validate, validate_record  # noqa: E402
from report_backfill import (  # noqa: E402
    discover_leftovers,
    extract_trailing_json,
    leftover_from_log,
    parse_compact_ts,
    parse_recover_code_from_text,
)
from report_cluster_health import main  # noqa: E402


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(argv)
    return rc, stdout.getvalue(), stderr.getvalue()


def _backfill_argv(tree: Path, *extra: str) -> list[str]:
    return [
        "--from-artifact-dir",
        str(tree),
        "--triggered-by",
        "operator",
        "--dry-run",
        *extra,
    ]


def _base_publish_record(hosts: str, ts: str, artifact: str) -> dict:
    return {
        "schema": SCHEMA_ID,
        "ts": ts,
        "test_type": "physical",
        "status": "passed",
        "hosts": [hosts],
        "analyzer_code": 0,
        "artifact_uri": artifact,
    }


def _publish_worker(exabox_dir: str, store_root: str, hosts: str, ts: str, artifact: str) -> str:
    if exabox_dir not in sys.path:
        sys.path.insert(0, exabox_dir)
    from report_cluster_health import publish_record as publish

    published = publish(_base_publish_record(hosts, ts, artifact), store_root)
    return published["record_uri"]


def _publish_labeled_worker(
    exabox_dir: str,
    store_root: str,
    hosts: str,
    ts: str,
    artifact: str,
    label: str,
) -> dict:
    if exabox_dir not in sys.path:
        sys.path.insert(0, exabox_dir)
    from report_cluster_health import publish_record as publish

    record = _base_publish_record(hosts, ts, artifact)
    record["labels"] = {"quad": label}
    published = publish(record, store_root)
    return {
        "has_record_id": "record_id" in published,
        "quad": published.get("labels", {}).get("quad"),
        "record_uri": published.get("record_uri"),
    }


class TestExtractTrailingJson(unittest.TestCase):
    def test_nested_object_at_end(self):
        text = 'noise {"outer": {"inner": 1}} trailing'
        obj = extract_trailing_json(text)
        self.assertIsNone(obj)

    def test_valid_sentinel_key(self):
        text = 'ignored {"analysis_exit_code": 0, "output_dir": "/tmp"}'
        obj = extract_trailing_json(text)
        self.assertEqual(obj["analysis_exit_code"], 0)

    def test_unbalanced_braces_returns_none(self):
        self.assertIsNone(extract_trailing_json('{"foo": 1'))

    def test_json_ending_with_string_returns_none(self):
        self.assertIsNone(extract_trailing_json('{"analysis_exit_code": 0, "output_dir": "/tmp"'))

    def test_nested_object_with_sentinel_parses_outer(self):
        text = 'log {"analysis_exit_code": 2, "nested": {"inner": 1}}'
        obj = extract_trailing_json(text)
        self.assertEqual(obj["analysis_exit_code"], 2)
        self.assertEqual(obj["nested"], {"inner": 1})

    def test_braces_inside_string_values(self):
        text = 'prefix {"analysis_exit_code": 0, "output_dir": "/tmp/}weird{"}'
        obj = extract_trailing_json(text)
        self.assertEqual(obj["output_dir"], "/tmp/}weird{")


class TestParseCompactTs(unittest.TestCase):
    def test_compact_yyyymmdd(self):
        self.assertEqual(parse_compact_ts("20240115T143000Z"), "2024-01-15T14:30:00Z")

    def test_rfc3339_passthrough(self):
        self.assertEqual(parse_compact_ts("2024-01-15T14:30:00Z"), "2024-01-15T14:30:00Z")

    def test_no_ts_token_returns_none(self):
        self.assertIsNone(parse_compact_ts("not-a-date"))


class TestBackfillDiscover(unittest.TestCase):
    def test_json_sidecar_wins_over_footer(self):
        tree = fixtures.artifact_tree(self)
        log_path = tree / "logs" / fixtures.PHYSICAL_LOG
        leftover = leftover_from_log("physical", log_path, tree)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 0)

    def test_skips_log_without_hosts(self):
        tree = fixtures.artifact_tree(self)
        log_path = tree / "logs" / fixtures.PHYSICAL_NO_HOSTS_LOG
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            leftover = leftover_from_log("physical", log_path, tree)
        self.assertIsNone(leftover)
        self.assertIn("no hosts", stderr.getvalue())

    def test_discover_four_complete_leftovers(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            leftovers = discover_leftovers(fixtures.artifact_tree(self))
        kinds = {item.test_type for item in leftovers}
        self.assertEqual(kinds, {"physical", "fabric", "dispatch", "recover"})
        self.assertEqual(len(leftovers), 4)
        self.assertIn("no hosts", stderr.getvalue())
        self.assertNotIn("no wrapper log", stderr.getvalue())


class TestBackfillCli(unittest.TestCase):
    def test_dry_run_stdout_lines_no_files(self):
        tree = fixtures.artifact_tree(self)
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err = _run(_backfill_argv(tree, "--store-root", tmp))
            self.assertEqual(rc, 0, err)
            lines = [ln for ln in out.splitlines() if ln]
            self.assertEqual(len(lines), 4)
            records = [loads_and_validate(ln, file_written=False) for ln in lines]
            by_type = {rec["test_type"]: rec for rec in records}
            self.assertEqual(by_type["physical"]["analyzer_code"], 0)
            self.assertEqual(by_type["physical"]["status"], "passed")
            self.assertNotIn("failure_reason", by_type["physical"].get("labels", {}))
            self.assertEqual(by_type["fabric"]["analyzer_code"], 3)
            self.assertEqual(by_type["fabric"]["status"], "failed")
            self.assertEqual(by_type["fabric"]["labels"]["failure_reason"], "Fabric router sync timeout")
            self.assertEqual(by_type["dispatch"]["status"], "failed")
            self.assertEqual(by_type["dispatch"]["labels"]["failure_reason"], "One or more dispatch tests failed")
            self.assertNotIn("analyzer_code", by_type["recover"])
            self.assertEqual(by_type["recover"]["status"], "passed")
            self.assertEqual(by_type["physical"]["source"], "backfill")
            self.assertEqual(by_type["physical"]["trigger_kind"], "backfill")
            self.assertEqual(by_type["physical"]["triggered_by"], "operator")
            self.assertEqual(list(Path(tmp).rglob("*.json")), [])

    def test_write_store(self):
        tree = fixtures.artifact_tree(self)
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--from-artifact-dir",
                str(tree),
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
        tree = fixtures.artifact_tree(self)
        recover = tree / "logs" / fixtures.RECOVER_LOG
        others = [
            tree / "logs" / fixtures.PHYSICAL_LOG,
            tree / "logs" / fixtures.FABRIC_LOG,
            tree / "logs" / fixtures.DISPATCH_LOG,
        ]
        old = datetime(2026, 8, 1, 12, tzinfo=timezone.utc).timestamp()
        recent = datetime(2026, 8, 19, 12, tzinfo=timezone.utc).timestamp()
        os.utime(recover, (old, old))
        for path in others:
            os.utime(path, (recent, recent))
        rc, out, err = _run(_backfill_argv(tree, "--from", "2026-08-10", "--to", "2026-08-20"))
        self.assertEqual(rc, 0, err)
        types = {json.loads(ln)["test_type"] for ln in out.splitlines() if ln}
        self.assertEqual(types, {"physical", "fabric", "dispatch"})
        self.assertNotIn("recover", types)

    def test_store_root_required_without_dry_run(self):
        rc, _out, err = _run(
            [
                "--from-artifact-dir",
                str(fixtures.artifact_tree(self)),
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
                str(fixtures.artifact_tree(self)),
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

    def test_same_record_id_different_content_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context("spawn")
            shared = (str(EXABOX_DIR), tmp, "bh-glx-110-c01u02", "2026-08-19T03:12:00Z", "/tmp/a")
            jobs = [(*shared, "first"), (*shared, "second")]
            with ctx.Pool(2) as pool:
                results = pool.starmap(_publish_labeled_worker, jobs)
            winners = [item for item in results if item["has_record_id"]]
            losers = [item for item in results if not item["has_record_id"]]
            self.assertEqual(len(winners), 1, results)
            self.assertEqual(len(losers), 1, results)
            files = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(files), 1)
            on_disk = json.loads(files[0].read_text(encoding="utf-8"))
            validate_record(on_disk, file_written=True)
            self.assertEqual(on_disk["labels"]["quad"], winners[0]["quad"])
            self.assertEqual(Path(winners[0]["record_uri"]).resolve(), files[0].resolve())
            self.assertNotEqual(winners[0]["quad"], losers[0]["quad"])
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])


class TestRecoverRealFooters(unittest.TestCase):
    def _recover(self, text: str):
        log = fixtures.temp_file(self, "recover-20260819T051000Z.log", text)
        return leftover_from_log("recover", log, log.parent)

    def test_failed_even_when_recovery_completed_follows(self):
        leftover = self._recover(fixtures.RECOVER_FAILED_LOG)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 1)
        self.assertFalse(leftover.incomplete)

    def test_last_success_wins_after_failed_attempt(self):
        leftover = self._recover(fixtures.RECOVER_SUCCEEDED_LOG)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 0)
        self.assertFalse(leftover.incomplete)

    def test_recovery_completed_alone_is_not_success(self):
        text = "=== Recover - x ===\nHOSTS=h1\n" "Recovery completed at Wed Aug 19 05:04:23 UTC 2026\n"
        self.assertIsNone(parse_recover_code_from_text(text))

    def test_incomplete_recover_is_degraded(self):
        leftover = self._recover(fixtures.RECOVER_INCOMPLETE_LOG)
        self.assertIsNotNone(leftover)
        self.assertTrue(leftover.incomplete)
        self.assertIsNone(leftover.analyzer_code)


class TestIncompleteAndLargeLogs(unittest.TestCase):
    def test_truncated_wrapper_emits_degraded(self):
        name = "physical_validation-truncated.log"
        tree = fixtures.log_tree(self, name, fixtures.PHYSICAL_TRUNCATED_LOG)
        leftover = leftover_from_log("physical", tree / "logs" / name, tree)
        self.assertIsNotNone(leftover)
        self.assertTrue(leftover.incomplete)
        rc, out, err = _run(_backfill_argv(tree))
        self.assertEqual(rc, 0, err)
        rec = loads_and_validate(out.splitlines()[0], file_written=False)
        self.assertEqual(rec["status"], "degraded")
        self.assertNotIn("analyzer_code", rec)
        self.assertEqual(rec["labels"]["incomplete"], "true")
        self.assertEqual(rec["labels"]["incomplete_reason"], "missing_terminal_outcome")
        self.assertEqual(rec["labels"]["failure_reason"], "Incomplete run (missing terminal outcome)")
        self.assertIn("degraded=1", err)

    def test_large_log_footer_beyond_64k(self):
        name = "physical_validation-20260819T060000Z.log"
        header = "=== Physical Validation - 20260819T060000Z ===\n" "HOSTS=bh-glx-110-c01u02\n" "OUTPUT_DIR=/tmp/huge\n"
        padding = "x" * 100_000
        tree = fixtures.log_tree(self, name, header + padding + "\nAnalysis exit code: 13\n")
        leftover = leftover_from_log("physical", tree / "logs" / name, tree)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 13)
        self.assertFalse(leftover.incomplete)

    def test_analysis_exit_with_host_prefix(self):
        name = "physical_validation-20260819T060200Z.log"
        tree = fixtures.log_tree(
            self,
            name,
            "=== Physical Validation - 20260819T060200Z ===\n"
            "HOSTS=bh-glx-110-c01u02\n"
            "[bh-glx-110-c01u02][06:02:00] Analysis exit code: 7\n",
        )
        leftover = leftover_from_log("physical", tree / "logs" / name, tree)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 7)
        self.assertFalse(leftover.incomplete)

    def test_backfill_cli_keeps_caller_labels(self):
        rc, out, err = _run(
            _backfill_argv(fixtures.artifact_tree(self), "--label", "superpod=SC16_1", "--label", "ring=SC16")
        )
        self.assertEqual(rc, 0, err)
        records = [json.loads(ln) for ln in out.splitlines() if ln]
        self.assertTrue(records)
        for rec in records:
            self.assertEqual(rec["labels"]["superpod"], "SC16_1")
            self.assertEqual(rec["labels"]["ring"], "SC16")

    def test_footer_across_chunk_boundary(self):
        name = "physical_validation-20260819T060100Z.log"
        tree = fixtures.log_tree(self, name, "")
        dest = tree / "logs" / name
        header = ("=== Physical Validation - 20260819T060100Z ===\n" "HOSTS=bh-glx-110-c01u02\n").encode("utf-8")
        # Split "Analysis" across the last 64KiB reverse-chunk start.
        prefix = 65536 - 8
        dest.write_bytes(header + (b"y" * prefix) + b"\nAnalysis exit code: 4\n")
        leftover = leftover_from_log("physical", dest, tree)
        self.assertIsNotNone(leftover)
        self.assertEqual(leftover.analyzer_code, 4)


class TestRecursiveDedup(unittest.TestCase):
    def test_recursive_discovers_nested_logs_dirs(self):
        root = fixtures.temp_dir(self)
        fixtures.write(
            root / "executions" / "one" / "logs",
            fixtures.DISPATCH_LOG,
            fixtures.ARTIFACT_TREE_LOGS[fixtures.DISPATCH_LOG],
        )
        fixtures.write(
            root / "nightly" / "SC16_1" / "logs",
            fixtures.FABRIC_LOG,
            fixtures.ARTIFACT_TREE_LOGS[fixtures.FABRIC_LOG],
        )
        leftovers = discover_leftovers(root, recursive=True)
        kinds = {item.test_type for item in leftovers}
        self.assertEqual(kinds, {"dispatch", "fabric"})
        rc, out, err = _run(
            [
                "--from-artifact-dir",
                str(root),
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
        tree = fixtures.log_tree(self, fixtures.DISPATCH_LOG, fixtures.ARTIFACT_TREE_LOGS[fixtures.DISPATCH_LOG])
        leftovers = discover_leftovers(tree)
        leftovers = leftovers + leftovers
        from report_backfill import leftover_key

        keys = [leftover_key(item) for item in leftovers]
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(set(keys)), 1)


class TestHostDiagLeftover(unittest.TestCase):
    def test_discover_diag_report_json(self):
        root = fixtures.temp_dir(self)
        fixtures.write(root, "diag_report.json", fixtures.DIAG_REPORT_PASS)
        leftovers = discover_leftovers(root)
        self.assertEqual(len(leftovers), 1)
        leftover = leftovers[0]
        self.assertEqual(leftover.test_type, "host")
        self.assertEqual(leftover.hosts, "bh-glx-110-c01u02")
        self.assertEqual(leftover.analyzer_code, 0)
        self.assertEqual(leftover.ts, "2026-08-19T03:12:36Z")
        self.assertEqual(leftover.duration_s, 36.5)
        self.assertEqual(leftover.labels["tier"], "light")
        self.assertEqual(leftover.labels["board_rev"], "RevC")
        rc, out, err = _run(_backfill_argv(root))
        self.assertEqual(rc, 0, err)
        record = loads_and_validate(out.splitlines()[0], file_written=False)
        self.assertEqual(record["test_type"], "host")
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["duration_s"], 36.5)
        self.assertEqual(record["labels"]["tier"], "light")

    def test_skips_dry_run_diag_report(self):
        root = fixtures.temp_dir(self)
        fixtures.write(root, "diag_report.json", fixtures.DIAG_REPORT_DRY_RUN)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            leftovers = discover_leftovers(root)
        self.assertEqual(leftovers, [])
        self.assertIn("dry-run", stderr.getvalue())

    def test_cli_duration_s_does_not_stamp_physical(self):
        rc, out, err = _run(_backfill_argv(fixtures.artifact_tree(self), "--duration-s", "99"))
        self.assertEqual(rc, 0, err)
        for line in out.splitlines():
            if not line:
                continue
            rec = json.loads(line)
            self.assertNotIn("duration_s", rec)

    def test_recursive_nested_diag_report(self):
        root = fixtures.temp_dir(self)
        fixtures.write(root / "node-a" / "results", "diag_report.json", fixtures.DIAG_REPORT_WARN)
        leftovers = discover_leftovers(root, recursive=True)
        self.assertEqual(len(leftovers), 1)
        self.assertEqual(leftovers[0].analyzer_code, 2)
        rc, out, err = _run(_backfill_argv(root, "--recursive"))
        self.assertEqual(rc, 0, err)
        record = loads_and_validate(out.splitlines()[0], file_written=False)
        self.assertEqual(record["labels"]["failure_reason"], "Diagnostic warning")


if __name__ == "__main__":
    unittest.main()
