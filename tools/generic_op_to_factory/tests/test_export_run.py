# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic DB tests for deterministic export; no remote DB or device required."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest

from tools.generic_op_to_factory import export_run


class ReadCursor:
    """Execute the exporter's SELECTs against an isolated, read-only SQLite DB."""

    def __init__(self, connection):
        self.cursor = connection.cursor()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cursor.close()

    def execute(self, sql, params):
        assert sql.startswith("SELECT ")
        return self.cursor.execute(sql.replace("%s", "?"), params)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchmany(self, size):
        return self.cursor.fetchmany(size)


class ReadConnection:
    def __init__(self, connection):
        self.connection = connection

    def cursor(self):
        return ReadCursor(self.connection)


@pytest.fixture
def snapshot():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(Path(__file__).with_name("eval_schema.sql").read_text())
    run_id = connection.execute(
        """INSERT INTO runs
           (timestamp, prompt_name, run_number, starting_branch, starting_commit,
            created_branch, status, golden_name, eval_commit, op_metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "2026-01-01",
            "sample_op",
            1,
            "source",
            "a" * 40,
            "candidate",
            "complete",
            "sample_suite",
            "b" * 40,
            '{"tile_count":2}',
        ),
    ).lastrowid
    connection.execute(
        "INSERT INTO host_code (run_id, filename, source_code) VALUES (?, ?, ?)",
        (run_id, "planner.py", "# π\r\nx = 1"),
    )
    connection.execute(
        "INSERT INTO kernels (run_id, filename, source_code) VALUES (?, ?, ?)",
        (run_id, "compute.cpp", "// unchanged\nvoid kernel() {}\n"),
    )
    connection.execute(
        "INSERT INTO artifacts (run_id, name, content) VALUES (?, ?, ?)",
        (run_id, "tests/test_sample.py", "# archived generated test\n"),
    )
    for phase in (None, "initial", "refined"):
        connection.execute(
            "INSERT INTO test_results (run_id, test_name, status, phase) VALUES (?, ?, ?, ?)",
            (run_id, "test_sample[case]", "passed", phase),
        )
    connection.commit()
    yield connection, run_id
    connection.close()


def export(snapshot, destination):
    connection, run_id = snapshot
    connection.execute("PRAGMA query_only = ON")
    try:
        return export_run.export_snapshot(
            ReadConnection(connection),
            run_id,
            destination,
            database={"host": "synthetic", "port": "0", "dbname": "fixture"},
        )
    finally:
        connection.execute("PRAGMA query_only = OFF")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_export_preserves_stored_text_provenance_and_all_result_phases(snapshot, tmp_path):
    destination = tmp_path / "package"
    manifest = export(snapshot, destination)
    assert (destination / "source/planner.py").read_bytes() == "# π\r\nx = 1".encode("utf-8")
    assert (destination / "source/kernels/compute.cpp").read_bytes() == b"// unchanged\nvoid kernel() {}\n"
    assert (destination / "artifacts/tests/test_sample.py").read_bytes() == b"# archived generated test\n"
    run = json.loads((destination / "records/run.json").read_bytes())
    assert run["op_metadata_json"] == '{"tile_count":2}'
    assert manifest["provenance"]["eval_commit"] == "b" * 40
    results = [json.loads(line) for line in (destination / "records/test_results.jsonl").read_bytes().splitlines()]
    assert [row["phase"] for row in results] == [None, "initial", "refined"]
    assert manifest["counts"]["test_results"] == 3
    assert set(manifest["counts"]) == {"runs", *export_run.CHILD_TABLES}
    assert manifest["selection"] == "current-recorded-run-snapshot"
    assert manifest["migration_ready"] is False
    assert {entry["kind"] for entry in manifest["unresolved_inputs"]} == {
        "golden_suite_source",
        "external_dependencies",
        "runtime_configuration",
    }
    assert export_run.verify_export(destination) == manifest


def test_unchanged_snapshot_exports_identically_to_different_destinations(snapshot, tmp_path):
    export(snapshot, tmp_path / "first")
    export(snapshot, tmp_path / "second")
    assert tree_bytes(tmp_path / "first") == tree_bytes(tmp_path / "second")


def test_source_change_changes_source_and_snapshot_checksums(snapshot, tmp_path):
    first = export(snapshot, tmp_path / "before")
    snapshot[0].execute("UPDATE host_code SET source_code = ?", ("changed source\n",))
    second = export(snapshot, tmp_path / "after")
    assert first["source_sha256"] != second["source_sha256"]
    assert first["snapshot_sha256"] != second["snapshot_sha256"]


def test_metadata_change_changes_snapshot_but_not_source_checksum(snapshot, tmp_path):
    first = export(snapshot, tmp_path / "before")
    snapshot[0].execute("UPDATE runs SET annotation_notes = ?", ("new annotation",))
    second = export(snapshot, tmp_path / "after")
    assert first["source_sha256"] == second["source_sha256"]
    assert first["snapshot_sha256"] != second["snapshot_sha256"]


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "a/../b",
        "a//b",
        "./file",
        "a/",
        "",
        "a\\b",
        "C:drive",
        "a\nb",
    ],
)
def test_unsafe_database_paths_are_rejected_before_publication(snapshot, tmp_path, name):
    snapshot[0].execute("UPDATE artifacts SET name = ?", (name,))
    with pytest.raises(export_run.ExportError, match="path"):  # allow-pytest.raises: host-only workflow validation
        export(snapshot, tmp_path / "package")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["planner.py", "planner.py/child.py", "kernels/compute.cpp"])
def test_duplicate_and_file_directory_collisions_are_rejected(snapshot, tmp_path, filename):
    snapshot[0].execute(
        "INSERT INTO host_code (run_id, filename, source_code) VALUES (?, ?, ?)",
        (snapshot[1], filename, "other source"),
    )
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        export_run.ExportError, match="Duplicate|collision"
    ):
        export(snapshot, tmp_path / "package")
    assert list(tmp_path.iterdir()) == []


def test_duplicate_artifact_names_are_not_silently_selected(snapshot, tmp_path):
    snapshot[0].execute(
        "INSERT INTO artifacts (run_id, name, content) VALUES (?, ?, ?)",
        (snapshot[1], "tests/test_sample.py", "ambiguous artifact"),
    )
    with pytest.raises(export_run.ExportError, match="Duplicate"):  # allow-pytest.raises: host-only workflow validation
        export(snapshot, tmp_path / "package")


@pytest.mark.parametrize("status", ["queued", "running", "testing"])
def test_active_run_is_not_presented_as_a_final_candidate(snapshot, tmp_path, status):
    snapshot[0].execute("UPDATE runs SET status = ?", (status,))
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        export_run.ExportError, match="still active"
    ):
        export(snapshot, tmp_path / "package")


def test_failed_run_remains_exportable_without_claiming_correctness(snapshot, tmp_path):
    snapshot[0].execute("UPDATE runs SET status = 'failed'")
    manifest = export(snapshot, tmp_path / "package")
    assert manifest["provenance"]["status"] == "failed"
    assert manifest["migration_ready"] is False


def test_missing_run_fails_without_output(snapshot, tmp_path):
    snapshot[0].execute("DELETE FROM runs")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        export_run.ExportError, match="does not exist"
    ):
        export(snapshot, tmp_path / "package")
    assert list(tmp_path.iterdir()) == []


def test_missing_host_source_fails(snapshot, tmp_path):
    snapshot[0].execute("DELETE FROM host_code")
    with pytest.raises(export_run.ExportError, match="source"):  # allow-pytest.raises: host-only workflow validation
        export(snapshot, tmp_path / "package")


def test_empty_source_files_are_preserved_without_interpreting_them(snapshot, tmp_path):
    snapshot[0].execute(
        "INSERT INTO host_code (run_id, filename, source_code) VALUES (?, ?, ?)",
        (snapshot[1], "__init__.py", ""),
    )
    export(snapshot, tmp_path / "package")
    assert (tmp_path / "package/source/__init__.py").read_bytes() == b""
    export_run.verify_export(tmp_path / "package")


def test_results_larger_than_one_fetch_batch_are_all_exported(snapshot, tmp_path):
    snapshot[0].executemany(
        "INSERT INTO test_results (run_id, test_name, status) VALUES (?, ?, ?)",
        [(snapshot[1], f"test_sample[{index}]", "passed") for index in range(2048)],
    )
    manifest = export(snapshot, tmp_path / "package")
    assert manifest["counts"]["test_results"] == 2051
    results = (tmp_path / "package/records/test_results.jsonl").read_bytes().splitlines()
    assert len(results) == 2051
    assert json.loads(results[-1])["test_name"] == "test_sample[2047]"


def test_no_kernel_rows_are_reported_without_inventing_files(snapshot, tmp_path):
    snapshot[0].execute("DELETE FROM kernels")
    manifest = export(snapshot, tmp_path / "package")
    assert manifest["counts"]["kernels"] == 0
    assert "inline" in manifest["notes"][0]
    assert not (tmp_path / "package/source/kernels").exists()


def test_existing_output_is_never_overwritten(snapshot, tmp_path):
    destination = tmp_path / "package"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"user data")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        export_run.ExportError, match="already exists"
    ):
        export(snapshot, destination)
    assert sentinel.read_bytes() == b"user data"


def test_export_failure_does_not_publish_partial_package(snapshot, tmp_path):
    snapshot[0].execute("DROP TABLE device_timings")
    with pytest.raises(sqlite3.OperationalError):  # allow-pytest.raises: host-only workflow validation
        export(snapshot, tmp_path / "package")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["corrupt", "missing", "extra", "symlink", "manifest"])
def test_offline_verification_detects_package_changes(snapshot, tmp_path, change):
    destination = tmp_path / "package"
    export(snapshot, destination)
    source = destination / "source/planner.py"
    if change == "corrupt":
        source.write_bytes(b"changed")
    elif change == "missing":
        source.unlink()
    elif change == "extra":
        (destination / "extra").write_bytes(b"extra")
    elif change == "symlink":
        source.unlink()
        source.symlink_to(tmp_path / "outside")
    else:
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["migration_ready"] = True
        manifest_path.write_bytes(export_run.json_bytes(manifest))
    with pytest.raises(export_run.ExportError):  # allow-pytest.raises: host-only workflow validation
        export_run.verify_export(destination)


def test_verifier_rejects_manifest_path_traversal_even_with_recomputed_checksum(snapshot, tmp_path):
    destination = tmp_path / "package"
    export(snapshot, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0]["path"] = "../outside"
    del manifest["snapshot_sha256"]
    manifest["snapshot_sha256"] = hashlib.sha256(export_run.json_bytes(manifest)).hexdigest()
    manifest_path.write_bytes(export_run.json_bytes(manifest))
    with pytest.raises(export_run.ExportError, match="relative"):  # allow-pytest.raises: host-only workflow validation
        export_run.verify_export(destination)


def test_json_encoding_preserves_nonfinite_and_database_specific_values():
    encoded = export_run.json_bytes(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "decimal": Decimal("1.234567890123456789"),
            "when": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
    )
    assert json.loads(encoded) == {
        "nan": {"$float": "nan"},
        "inf": {"$float": "inf"},
        "decimal": {"$decimal": "1.234567890123456789"},
        "when": {"$datetime": "2026-01-01T00:00:00+00:00"},
    }


def test_remote_connection_is_read_only_consistent_and_does_not_record_credentials(
    monkeypatch,
):
    psycopg2 = pytest.importorskip("psycopg2")
    connection = mock.Mock()
    connection.get_dsn_parameters.return_value = {
        "host": "database",
        "port": "5432",
        "dbname": "eval",
        "password": "private",
        "user": "reader",
    }
    connect = mock.Mock(return_value=connection)
    monkeypatch.setattr(psycopg2, "connect", connect)
    with export_run.remote_snapshot("postgresql://reader:private@database/eval") as (
        opened,
        identity,
    ):
        assert opened is connection
        assert identity == {"host": "database", "port": "5432", "dbname": "eval"}
    connection.set_session.assert_called_once_with(
        readonly=True,
        isolation_level="REPEATABLE READ",
        autocommit=False,
    )
    assert "default_transaction_read_only=on" in connect.call_args.kwargs["options"]
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()
    connection.cursor.assert_not_called()  # No initialization, DDL, or schema migration.


def test_remote_failure_has_no_local_fallback_or_credential_leak(monkeypatch, capsys, tmp_path):
    psycopg2 = pytest.importorskip("psycopg2")
    monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql://reader:private@database/eval")
    monkeypatch.setattr(psycopg2, "connect", mock.Mock(side_effect=psycopg2.OperationalError("private")))
    assert export_run.main(["--run-id", "1", "--output", str(tmp_path / "package")]) == 1
    assert "private" not in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_connection_closes_and_sanitizes_error_when_read_and_rollback_fail(monkeypatch):
    psycopg2 = pytest.importorskip("psycopg2")
    connection = mock.Mock()
    connection.get_dsn_parameters.return_value = {
        "host": "database",
        "port": "5432",
        "dbname": "eval",
    }
    connection.rollback.side_effect = psycopg2.OperationalError("private rollback error")
    monkeypatch.setattr(psycopg2, "connect", mock.Mock(return_value=connection))
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        export_run.ExportError, match="Remote DB read failed"
    ):
        with export_run.remote_snapshot("postgresql://reader@database/eval"):
            raise psycopg2.OperationalError("private read error")
    connection.close.assert_called_once()


@pytest.mark.parametrize("url", [None, "", "sqlite:///local.db"])
def test_cli_requires_an_explicit_remote_database(url):
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        export_run.ExportError, match="explicit PostgreSQL"
    ):
        with export_run.remote_snapshot(url):
            pytest.fail("Invalid database URL was accepted")


def test_offline_verification_does_not_connect_to_database(snapshot, tmp_path, monkeypatch):
    destination = tmp_path / "package"
    export(snapshot, destination)
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    assert export_run.main(["--verify", str(destination)]) == 0
