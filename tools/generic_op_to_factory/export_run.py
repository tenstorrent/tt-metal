# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministically export the current DB snapshot of one explicitly selected run.

No schema initialization, SQLite fallback, source execution, or LLM is involved.
The CLI requires EVAL_DATABASE_URL and an existing psycopg2 installation.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath

FORMAT_VERSION = 1
# The run-owned tables in eval/db.py. Keep raw results from ALL phases, without
# interpreting dashboard canonical-phase rules or selecting a "best" candidate.
CHILD_TABLES = (
    "host_code",
    "kernels",
    "artifacts",
    "kw_breadcrumbs",
    "test_results",
    "score_criteria",
    "refinement_snapshots",
    "phases",
    "device_timings",
    "device_phases",
)
SOURCE_TABLES = {
    "host_code": ("filename", "source_code", "source"),
    "kernels": ("filename", "source_code", "source/kernels"),
    "artifacts": ("name", "content", "artifacts"),
}


class ExportError(ValueError):
    """The requested snapshot cannot be exported or verified unambiguously."""


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, (date, datetime)):
        return {"$datetime" if isinstance(value, datetime) else "$date": value.isoformat()}
    return value


def json_bytes(value):
    """Stable strict JSON, with tagged encodings for non-JSON database values."""
    return (
        json.dumps(
            _json_value(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _safe_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ExportError(f"Invalid artifact path: {name!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ExportError(f"Control character in artifact path: {name!r}")
    if name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/")):
        raise ExportError(f"Artifact path must be canonical and relative: {name!r}")
    return name


def _validate_paths(paths):
    seen = set()
    for path in paths:
        _safe_path(path)
        if path in seen:
            raise ExportError(f"Duplicate artifact path: {path}")
        seen.add(path)
    for path in seen:
        if any(parent.as_posix() in seen for parent in PurePosixPath(path).parents):
            raise ExportError(f"Artifact file/directory collision: {path}")


def _hash_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _file_entry(root, relative_path, **metadata):
    sha256, size = _hash_file(root / relative_path)
    return {"path": relative_path, "sha256": sha256, "size_bytes": size, **metadata}


def _write_bytes(root, relative_path, content):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def _rows(cursor, table, run_id):
    # table comes only from CHILD_TABLES; identifiers never come from CLI/DB data.
    cursor.execute(f"SELECT * FROM {table} WHERE run_id = %s ORDER BY id", (run_id,))
    while batch := cursor.fetchmany(1000):
        for row in batch:
            yield dict(row)


def _source_identity(files):
    content = [
        {key: entry[key] for key in ("path", "sha256", "size_bytes")}
        for entry in files
        if entry["table"] in ("host_code", "kernels") and entry.get("materialized")
    ]
    return hashlib.sha256(json_bytes(content)).hexdigest()


def export_snapshot(connection, run_id, output, *, database):
    """Export an already-open, consistent read-only transaction to a new directory.

    This boundary also allows unit tests to supply an isolated synthetic DB.
    Caller owns the connection and transaction. No file is imported or executed.
    """
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise ExportError("run_id must be a positive integer")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ExportError("Output already exists; choose a new export directory")

    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM runs WHERE id = %s", (run_id,))
        row = cursor.fetchone()
        if row is None:
            raise ExportError(f"Run {run_id} does not exist in the selected database")
        run = dict(row)
        if run.get("status") not in ("complete", "failed"):
            raise ExportError("Run is still active; export a terminal run snapshot")

        source_rows = {table: list(_rows(cursor, table, run_id)) for table in SOURCE_TABLES}
        if not source_rows["host_code"]:
            raise ExportError("Run has no recorded host source")
        materialized = []
        for table, (name_key, content_key, prefix) in SOURCE_TABLES.items():
            for source_row in source_rows[table]:
                name = _safe_path(source_row[name_key])
                content = source_row[content_key]
                if not isinstance(content, str):
                    raise ExportError(f"Non-text content in {table} row {source_row['id']}")
                materialized.append(
                    (
                        f"{prefix}/{name}",
                        content,
                        {
                            "table": table,
                            "row_id": source_row["id"],
                            "database_name": name,
                            "materialized": True,
                        },
                    )
                )
        _validate_paths([item[0] for item in materialized])

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".eval-export-", dir=output.parent) as temporary:
            package = Path(temporary) / "package"
            package.mkdir()
            files = []
            _write_bytes(package, "records/run.json", json_bytes(run))
            files.append(_file_entry(package, "records/run.json", table="runs"))
            counts = {"runs": 1}
            for table in CHILD_TABLES:
                relative_path = f"records/{table}.jsonl"
                count = 0
                with (package / relative_path).open("xb") as stream:
                    rows = source_rows[table] if table in source_rows else _rows(cursor, table, run_id)
                    for child_row in rows:
                        stream.write(json_bytes(child_row))
                        count += 1
                counts[table] = count
                files.append(_file_entry(package, relative_path, table=table))
            for relative_path, content, metadata in materialized:
                _write_bytes(package, relative_path, content.encode("utf-8"))
                files.append(_file_entry(package, relative_path, **metadata))
            files.sort(key=lambda item: item["path"])

            manifest = {
                "format_version": FORMAT_VERSION,
                "database": database,
                "run_id": run_id,
                "selection": "current-recorded-run-snapshot",
                "source_history": "Source tables are replaced by re-ingestion; no phase-specific source selection",
                "provenance": {
                    key: run.get(key)
                    for key in (
                        "prompt_name",
                        "run_number",
                        "status",
                        "created_branch",
                        "starting_branch",
                        "starting_commit",
                        "eval_branch",
                        "eval_commit",
                        "golden_name",
                        "arch",
                    )
                },
                "encoding": "Stored text is UTF-8 encoded without newline changes; original pre-ingest bytes are unavailable",
                "json_encoding": "Sorted-key JSON; non-finite floats, Decimal, date and datetime use single-key $type objects",
                "counts": counts,
                "source_sha256": _source_identity(files),
                "files": files,
                "migration_ready": False,
                "unresolved_inputs": [
                    {
                        "kind": "golden_suite_source",
                        "eval_commit": run.get("eval_commit"),
                        "golden_name": run.get("golden_name"),
                        "reason": "DB results and generated tests do not establish the matching golden-suite source",
                    },
                    {
                        "kind": "external_dependencies",
                        "starting_commit": run.get("starting_commit"),
                        "reason": "Canonical helpers and other unrecorded dependencies must be resolved separately",
                    },
                    {
                        "kind": "runtime_configuration",
                        "reason": "Run metadata/artifacts are preserved; a complete environment and invocation are not guaranteed",
                    },
                ],
            }
            if not source_rows["kernels"]:
                manifest["notes"] = ["No kernel rows recorded; kernels may be inline. No kernel files were invented."]
            manifest["snapshot_sha256"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
            _write_bytes(package, "manifest.json", json_bytes(manifest))

            # Reserve the destination exclusively, then publish the complete directory.
            # Never replace an existing user export, including an empty directory.
            try:
                output.mkdir()
            except FileExistsError as error:
                raise ExportError("Output already exists; choose a new export directory") from error
            try:
                package.rename(output)
            except OSError:
                output.rmdir()  # Only our empty reservation; never recursively remove output.
                raise
    return manifest


def verify_export(output):
    """Verify a saved package offline, including unexpected files and symlinks."""
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ExportError("Export must be a real directory")
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ExportError("Export contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
    try:
        manifest = json.loads((output / "manifest.json").read_bytes())
        if manifest["format_version"] != FORMAT_VERSION:
            raise ExportError("Unsupported export format version")
        payload = {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
        if hashlib.sha256(json_bytes(payload)).hexdigest() != manifest["snapshot_sha256"]:
            raise ExportError("Manifest checksum mismatch")
        files = manifest["files"]
        _validate_paths([entry["path"] for entry in files])
        expected = {entry["path"] for entry in files} | {"manifest.json"}
        if actual != expected:
            raise ExportError("Export has missing or unexpected files")
        for entry in files:
            digest, size = _hash_file(output / entry["path"])
            if digest != entry["sha256"] or size != entry["size_bytes"]:
                raise ExportError(f"File checksum mismatch: {entry['path']}")
        if _source_identity(files) != manifest["source_sha256"]:
            raise ExportError("Source checksum mismatch")
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise ExportError("Invalid or unreadable export package") from error
    return manifest


@contextmanager
def remote_snapshot(database_url):
    """Connect directly: no eval.db.connect() initialization or local fallback."""
    if not database_url or not database_url.startswith(("postgresql://", "postgres://")):
        raise ExportError("Set EVAL_DATABASE_URL to an explicit PostgreSQL URL (prefer the eval_ro role)")
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as error:
        raise ExportError("Use an eval Python environment with psycopg2 installed") from error
    connection = None
    try:
        connection = psycopg2.connect(
            database_url,
            connect_timeout=10,
            cursor_factory=RealDictCursor,
            options="-c default_transaction_read_only=on -c statement_timeout=60000",
        )
        connection.set_session(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
        parameters = connection.get_dsn_parameters()
        # Allowlist non-secret provenance only; never save a DSN or connection error text.
        database = {key: parameters.get(key) for key in ("host", "port", "dbname")}
        yield connection, database
    except psycopg2.Error as error:
        raise ExportError(
            "Remote DB read failed; check connectivity, SELECT access, and schema compatibility"
        ) from error
    finally:
        if connection is not None:
            try:
                connection.rollback()
            except psycopg2.Error:
                # A disconnected socket may also fail rollback. Preserve the sanitized
                # read error rather than exposing a driver exception during cleanup.
                pass
            finally:
                connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run-id",
        type=int,
        help="Explicit runs.id; exports its current recorded snapshot",
    )
    selection.add_argument("--verify", type=Path, help="Verify an existing package offline")
    parser.add_argument("--output", type=Path, help="New directory for the exported package")
    args = parser.parse_args(argv)
    if args.verify is not None and args.output is not None:
        parser.error("--output cannot be combined with --verify")
    if args.run_id is not None and (args.run_id <= 0 or args.output is None):
        parser.error("--run-id must be positive and requires --output")
    try:
        if args.verify is not None:
            manifest = verify_export(args.verify)
        else:
            with remote_snapshot(os.environ.get("EVAL_DATABASE_URL")) as (
                connection,
                database,
            ):
                manifest = export_snapshot(connection, args.run_id, args.output, database=database)
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in (
                        "run_id",
                        "counts",
                        "source_sha256",
                        "snapshot_sha256",
                        "migration_ready",
                    )
                },
                sort_keys=True,
            )
        )
        return 0
    except (ExportError, OSError) as error:
        print(f"Export error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
