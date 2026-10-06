# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Idempotent observation storage; each variant is a replaceable snapshot."""

import json
import sqlite3
from hashlib import sha256
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, timeout=120)
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 3):
            self.connection.close()
            raise ValueError(f"Unsupported coverage database version {version}")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                arch TEXT, trisc TEXT, test TEXT, variant TEXT,
                object TEXT, source TEXT, symbol TEXT, build_axes TEXT, capture TEXT, payload TEXT NOT NULL,
                PRIMARY KEY (arch, trisc, test, variant, object, source, symbol, build_axes, capture)
            )
        """
        )
        # One library scan serves every variant built in the same context, so documents are shared by digest.
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                arch TEXT, trisc TEXT, test TEXT, variant TEXT, document TEXT NOT NULL,
                PRIMARY KEY (arch, trisc, test, variant)
            )"""
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS scan_documents (digest TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.execute("PRAGMA user_version=3")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def replace_variant(
        self,
        arch: str,
        test: str,
        variant: str,
        records: list[dict],
        scans: list[dict] = (),
    ):
        if any(
            (r["arch"], r["test"], r["variant"]) != (arch, test, variant)
            for r in [*records, *scans]
        ):
            raise ValueError("Observation does not belong to this variant")
        with self.connection:
            self.connection.execute(
                "DELETE FROM observations WHERE arch=? AND test=? AND variant=?",
                (arch, test, variant),
            )
            self._insert(records)
            self.connection.execute(
                "DELETE FROM scans WHERE arch=? AND test=? AND variant=?",
                (arch, test, variant),
            )
            self._insert_scans(scans)

    def _insert_scans(self, scans):
        for scan in scans:
            document = {k: v for k, v in scan.items() if k not in ("test", "variant")}
            payload = json.dumps(document)
            digest = sha256(payload.encode()).hexdigest()
            self.connection.execute(
                "INSERT OR IGNORE INTO scan_documents VALUES (?, ?)", (digest, payload)
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO scans VALUES (?, ?, ?, ?, ?)",
                (scan["arch"], scan["trisc"], scan["test"], scan["variant"], digest),
            )
        self.connection.execute(
            "DELETE FROM scan_documents WHERE digest NOT IN (SELECT document FROM scans)"
        )

    def scans(self) -> list[dict]:
        return [
            {**json.loads(payload), "test": test, "variant": variant}
            for test, variant, payload in self.connection.execute(
                "SELECT scans.test, scans.variant, scan_documents.payload FROM scans "
                "JOIN scan_documents ON scans.document = scan_documents.digest "
                "ORDER BY scans.arch, scans.trisc, scans.test, scans.variant"
            )
        ]

    def _insert(self, records):
        self.connection.executemany(
            "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["arch"],
                    r["trisc"],
                    r["test"],
                    r["variant"],
                    r["object"],
                    r["source"],
                    r["symbol"],
                    json.dumps(r.get("build_axes", {}), sort_keys=True),
                    sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
                    json.dumps(r),
                )
                for r in records
            ],
        )

    def records(self) -> list[dict]:
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT payload FROM observations ORDER BY arch, trisc, source, symbol, test, variant, object"
            )
        ]

    def merge(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Store(path) as other, self.connection:
            self._insert(other.records())
            self._insert_scans(other.scans())
