# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import time
from datetime import datetime, timezone

from . import collect, hosts, jira, triage


def _as_of(text: str | None) -> float | None:
    if not text:
        return None
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def cmd_triage(args: argparse.Namespace) -> None:
    as_of = _as_of(args.as_of) or time.time()
    rarity = triage.load_rarity()
    for host in args.host:
        host = hosts.normalize_host(host) or host
        data = collect.collect(host, as_of, args.days)
        for error in data.errors:
            print(f"skipped unreadable {error}", file=sys.stderr)
        report = triage.triage(host, data, jira.host_tickets(host), as_of, args.days, rarity)
        print(triage.report_json(report) if args.json else triage.render(report, verbose=args.verbose))
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="device_triage", description="Device status and likely issue from HC logs and Jira, read live."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("triage", help="status and likely issue for one or more hosts")
    p.add_argument("host", nargs="+")
    p.add_argument("--days", type=int, default=30, help="evidence window (default 30)")
    p.add_argument("--as-of", help="evaluate as of this UTC time, e.g. 2026-10-01T12:00 (default now)")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="more evidence, with file paths")
    p.set_defaults(func=cmd_triage)

    args = parser.parse_args(argv)
    args.func(args)
