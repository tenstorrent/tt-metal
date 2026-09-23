# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Write one warehouse run back out as a perf CSV, for the gate to compare."""

from __future__ import annotations

import argparse
import csv
import json
import os

VIEW = "TTDATASF.LLK_PERF.LLK_PERF_V"


DENSE = {
    "FORMAT_INPUT_A": "formats.input_A",
    "FORMAT_INPUT_B": "formats.input_B",
    "FORMAT_OUTPUT": "formats.output",
    "FORMAT_REGISTER_A": "formats.register_A",
    "FORMAT_REGISTER_B": "formats.register_B",
    "DEST_ACC": "dest_acc",
    "MATH_FIDELITY": "math_fidelity",
    "TILE_CNT": "tile_cnt",
    "NUM_BLOCKS": "num_blocks",
    "LOOP_FACTOR": "loop_factor",
    "SPEED_OF_LIGHT": "speed_of_light",
    "UNPACK_TO_DEST": "unpack_to_dest",
}


PROVENANCE = ("RUN_ID", "PIPELINE", "COMMIT_SHA", "PR_NUMBER", "ARCH", "RUN_TS",
              "EXECUTION_ID", "TEST_NAME")

SELECT = list(DENSE) + ["PARAMS", "MARKER", "METRIC", "MEAN"]


def connect():
    """Open a read connection with the service keypair."""
    pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "")
    if not pem.strip():
        raise SystemExit("::error::SNOWFLAKE_PRIVATE_KEY is empty")

    from cryptography.hazmat.primitives import serialization
    import snowflake.connector

    passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    pkey = serialization.load_pem_private_key(
        pem.encode(), password=passphrase.encode() if passphrase else None
    ).private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return snowflake.connector.connect(
        account=os.environ.get("SNOWFLAKE_ACCOUNT", "TLUIIGS-MN66866"),
        user=os.environ.get("SNOWFLAKE_USER", "SVC_TTOPS_SWEEPS"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "PUBLIC"),
        private_key=pkey,
    )


def pick_run(cursor, view, arch, pipeline, speed_of_light):
    """Newest run of this pipeline, arch and mode. SPEED_OF_LIGHT is a row"""
    cursor.execute(
        f"SELECT RUN_ID, MAX(COMMIT_SHA), MAX(RUN_TS) FROM {view} "
        f"WHERE ARCH = %s AND PIPELINE = %s AND SPEED_OF_LIGHT = %s "
        f"GROUP BY RUN_ID ORDER BY MAX(RUN_TS) DESC LIMIT 1",
        (arch, pipeline, speed_of_light),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {"run_id": row[0], "commit_sha": row[1], "run_ts": str(row[2])}


def fetch(cursor, view, run_id, speed_of_light):
    """Every measurement row of one run, in one speed-of-light mode."""
    cursor.execute(
        f"SELECT {', '.join(SELECT)} FROM {view} "
        f"WHERE RUN_ID = %s AND SPEED_OF_LIGHT = %s",
        (run_id, speed_of_light),
    )
    return cursor.fetchall()


def pivot(rows):
    """Long -> wide: the view's six METRIC rows per (config, marker) become one"""
    n_dense = len(DENSE)
    out = {}
    columns = set(DENSE.values()) | {"marker"}
    for row in rows:
        params_raw, marker, metric, mean = row[n_dense:n_dense + 4]
        record = {DENSE[name]: value for name, value in zip(DENSE, row[:n_dense])}
        record.update(json.loads(params_raw) if params_raw else {})
        record["marker"] = marker
        columns.update(record)
        key = tuple(sorted((k, repr(v)) for k, v in record.items()))
        out.setdefault(key, record)[f"mean({metric})"] = mean
        columns.add(f"mean({metric})")
    return list(out.values()), sorted(columns)


def write_csv(records, columns, path):
    """Write the pivoted records where the gate expects a baseline CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    return len(records)


def _write_meta(path, run, have):
    """Record which run was used, and whether there was one at all."""
    with open(path, "w") as fh:
        json.dump({**run, "source": VIEW}, fh)
    with open("have_baseline.txt", "w") as fh:
        fh.write("true" if have else "false")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view",
                    default=os.environ.get("LLK_PERF_WAREHOUSE_TABLE") or VIEW)
    ap.add_argument("--arch", required=True,
                    choices=("blackhole", "wormhole", "quasar"))
    ap.add_argument("--pipeline", default="baseline",
                    help="baseline (pinned) or nightly (the previous night)")
    ap.add_argument("--speed-of-light", required=True, choices=("true", "false"))
    ap.add_argument("--out", default="baseline_perf/baseline.csv")
    ap.add_argument("--metadata", default="baseline_metadata.json")
    a = ap.parse_args(argv)

    sol = a.speed_of_light == "true"
    conn = connect()
    try:
        cur = conn.cursor()
        run = pick_run(cur, a.view, a.arch, a.pipeline, sol)
        if not run:
            print(f"no {a.pipeline} run for arch={a.arch} speed_of_light={sol}")
            _write_meta(a.metadata, {}, have=False)
            return 0
        rows = fetch(cur, a.view, run["run_id"], sol)
    finally:
        conn.close()

    records, columns = pivot(rows)
    if not records:
        print(f"run {run['run_id']} holds no rows in this mode")
        _write_meta(a.metadata, run, have=False)
        return 0
    n = write_csv(records, columns, a.out)
    print(f"baseline {run['run_id']} ({run['commit_sha']}, {run['run_ts']}): "
          f"{len(rows)} rows -> {n} CSV rows, {len(columns)} columns")
    _write_meta(a.metadata, run, have=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
