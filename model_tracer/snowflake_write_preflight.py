#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Check that the configured Snowflake credentials can run a trace load, without writing.

Connects exactly as ``load_ttnn_ops_data_v2.py load`` does (same env vars, same role) and
compiles every DML statement shape the loader issues with EXPLAIN. Snowflake checks
privileges at compile time, so EXPLAIN fails on a missing privilege but never executes.
The loader also creates its id sequences on first use, so CREATE SEQUENCE on the schema
is required unless every sequence already exists.

Exit status is non-zero if any check fails.
"""

import argparse
import os
import sys

from tests.sweep_framework.load_ttnn_ops_data_v2 import (
    DEFAULT_SCHEMA,
    _ID_COLUMN_TABLE,
    _connect,
    _qualified_schema,
)

# (verb, table, statement template) for every DML shape in load_data and its helpers.
_DML_CHECKS = [
    ("INSERT", "ttnn_operation", "INSERT INTO {S}.ttnn_operation SELECT * FROM {S}.ttnn_operation WHERE FALSE"),
    ("INSERT", "ttnn_model", "INSERT INTO {S}.ttnn_model SELECT * FROM {S}.ttnn_model WHERE FALSE"),
    ("UPDATE", "ttnn_model", "UPDATE {S}.ttnn_model SET update_ts = update_ts WHERE FALSE"),
    ("INSERT", "ttnn_hardware", "INSERT INTO {S}.ttnn_hardware SELECT * FROM {S}.ttnn_hardware WHERE FALSE"),
    ("INSERT", "ttnn_mesh_config", "INSERT INTO {S}.ttnn_mesh_config SELECT * FROM {S}.ttnn_mesh_config WHERE FALSE"),
    (
        "INSERT",
        "ttnn_configuration",
        "INSERT INTO {S}.ttnn_configuration SELECT * FROM {S}.ttnn_configuration WHERE FALSE",
    ),
    (
        "INSERT",
        "ttnn_configuration_model",
        "INSERT INTO {S}.ttnn_configuration_model SELECT * FROM {S}.ttnn_configuration_model WHERE FALSE",
    ),
    ("DELETE", "ttnn_configuration_model", "DELETE FROM {S}.ttnn_configuration_model WHERE FALSE"),
    ("INSERT", "trace_run", "INSERT INTO {S}.trace_run SELECT * FROM {S}.trace_run WHERE FALSE"),
    ("UPDATE", "trace_run", "UPDATE {S}.trace_run SET notes = notes WHERE FALSE"),
    (
        "MERGE",
        "trace_run_configuration_model",
        "MERGE INTO {S}.trace_run_configuration_model t "
        "USING (SELECT * FROM {S}.trace_run_configuration_model WHERE FALSE) s "
        "ON t.trace_run_id = s.trace_run_id AND t.configuration_id = s.configuration_id AND t.model_id = s.model_id "
        "WHEN MATCHED THEN UPDATE SET execution_count = s.execution_count "
        "WHEN NOT MATCHED THEN INSERT (trace_run_id, configuration_id, model_id, execution_count) "
        "VALUES (s.trace_run_id, s.configuration_id, s.model_id, s.execution_count)",
    ),
    ("INSERT", "trace_run_config", "INSERT INTO {S}.trace_run_config SELECT * FROM {S}.trace_run_config WHERE FALSE"),
    ("DELETE", "trace_run_config", "DELETE FROM {S}.trace_run_config WHERE FALSE"),
    ("INSERT", "trace_run_model", "INSERT INTO {S}.trace_run_model SELECT * FROM {S}.trace_run_model WHERE FALSE"),
    ("DELETE", "trace_run_model", "DELETE FROM {S}.trace_run_model WHERE FALSE"),
]


def _first_line(exc):
    return str(exc).strip().splitlines()[-1][:200]


def run(schema):
    S = _qualified_schema(schema)
    rows = []
    conn = _connect(autocommit=False)
    try:
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
        user, role, warehouse = cur.fetchone()
        rows.append(("connect", f"user={user} role={role} warehouse={warehouse}", True, ""))

        for verb, table, template in _DML_CHECKS:
            try:
                cur.execute("EXPLAIN " + template.format(S=S))
                rows.append((verb, table, True, ""))
            except Exception as e:
                rows.append((verb, table, False, _first_line(e)))

        cur.execute(f"SHOW SEQUENCES IN SCHEMA {S}")
        existing = {r[0].upper() for r in cur.fetchall()}
        missing = sorted(f"SEQ_{t.upper()}" for t in _ID_COLUMN_TABLE.values() if f"SEQ_{t.upper()}" not in existing)
        if not missing:
            rows.append(("SEQUENCE", "all id sequences exist", True, ""))
        else:
            cur.execute(f"SHOW GRANTS TO ROLE {role}")
            cols = [c[0].lower() for c in cur.description]
            grants = [dict(zip(cols, r)) for r in cur.fetchall()]
            can_create = any(
                g.get("granted_on") == "SCHEMA"
                and str(g.get("name", "")).upper() == S.upper()
                and g.get("privilege") in ("CREATE SEQUENCE", "OWNERSHIP")
                for g in grants
            )
            rows.append(
                (
                    "CREATE SEQUENCE",
                    f"{S} (missing: {', '.join(missing)})",
                    can_create,
                    "" if can_create else f"role {role} has no CREATE SEQUENCE or OWNERSHIP on {S}",
                )
            )
    finally:
        conn.rollback()
        conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    args = parser.parse_args()

    try:
        rows = run(args.schema)
    except Exception as e:
        rows = [("connect", os.environ.get("SNOWFLAKE_USER", "?"), False, _first_line(e))]

    ok = all(r[2] for r in rows)
    lines = [
        f"## Snowflake write preflight: {'PASS' if ok else 'FAIL'}",
        "",
        "| Check | Target | Result | Error |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c} | `{t}` | {'ok' if passed else 'FAIL'} | {err} |" for c, t, passed, err in rows]
    text = "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
