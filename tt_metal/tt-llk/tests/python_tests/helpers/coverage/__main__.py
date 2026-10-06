# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PYTHONPATH=tests/python_tests python -m helpers.coverage {report,merge}."""

import argparse
from pathlib import Path

from .report import write_report
from .store import Store


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Render a saved coverage database")
    report.add_argument("--db", required=True, type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--test", help="Filter by pytest node ID or test driver name")
    report.add_argument(
        "--arch", action="append", choices=["wormhole", "blackhole", "quasar"]
    )
    report.add_argument("--root", type=Path, default=root, help="LLK source checkout")
    merge = commands.add_parser("merge", help="Combine CI shard databases")
    merge.add_argument("--db", required=True, type=Path)
    merge.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "merge":
        with Store(args.db) as store:
            for path in args.inputs:
                store.merge(path)
        return
    if not args.db.is_file():
        parser.error(f"Database does not exist: {args.db}")
    with Store(args.db) as store:
        records = store.records()
        scans = store.scans()
    architectures = args.arch or sorted(
        {record["arch"] for record in [*records, *scans]}
    )
    if args.test:
        records = [
            record
            for record in records
            if args.test in record.get("run", {}).get("test", "")
            or args.test in record["test"]
        ]
    output = args.output or args.db.parent / "instantiation_coverage"
    write_report(
        records, output, root=args.root, architectures=architectures, scans=scans
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
