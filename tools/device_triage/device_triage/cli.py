# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone

from pathlib import Path

from . import canvas, index, jira_sync, report, tickets, triage, validate
from .config import CANVAS_PATH, DCAMP_HC_ROOT, PROJECT_DIR


def cmd_index(args: argparse.Namespace) -> None:
    index.build(sources=args.source, workers=args.workers, limit=args.limit)
    conn = index.connect()
    print(f"tickets: {tickets.build(conn)} loaded")


def cmd_tickets(args: argparse.Namespace) -> None:
    conn = index.connect()
    print(f"tickets: {tickets.build(conn)} loaded")


def cmd_jira_sync(args: argparse.Namespace) -> None:
    conn = index.connect()
    known_hosts = on_disk_jobs = None
    if not args.skip_hc_runs:
        known_hosts = {h for (h,) in conn.execute("SELECT DISTINCT host FROM run_hosts")}
        on_disk_jobs = {
            Path(p).name.split("_")[0]
            for (p,) in conn.execute(
                "SELECT path FROM runs WHERE source = 'machine_hc' AND path NOT LIKE ?", (f"{DCAMP_HC_ROOT}%",)
            )
        }
    try:
        stats = jira_sync.sync(
            args.project or ["DIIM", "DCAMP"],
            args.attachments_for or ["DIIM"],
            full=args.full,
            max_mb=args.max_mb,
            known_hosts=known_hosts,
            on_disk_jobs=on_disk_jobs,
        )
    except jira_sync.JiraError as exc:
        raise SystemExit(f"jira sync failed: {exc}") from None
    print(
        f"issues: {stats.issues}; comments: {stats.comments}; attachments downloaded: {stats.downloaded} "
        f"({stats.downloaded_bytes / 1e6:.1f} MB), skipped: {stats.skipped}; machine HC runs rebuilt from "
        f"DCAMP: {stats.hc_runs} (diag reports: {stats.diag}); JQL categories: {stats.jql}; "
        f"{stats.requests} GET requests"
    )
    cmd_tickets(args)


def _as_of(text: str | None) -> float | None:
    if not text:
        return None
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def cmd_triage(args: argparse.Namespace) -> None:
    conn = index.connect()
    stats = triage.fleet_stats(conn)
    for host in args.host:
        report = triage.triage(conn, host, as_of=_as_of(args.as_of), window_days=args.days, stats=stats)
        print(triage.report_json(report) if args.json else triage.render(report, verbose=args.verbose))
        print()


def cmd_fleet(args: argparse.Namespace) -> None:
    conn = index.connect()
    rows = report.fleet(conn, days=args.days)
    (PROJECT_DIR / args.out).write_text(report.fleet_markdown(rows, args.days))
    for r in rows:
        if r.status in ("FAILING", "FLAKY", "DEGRADED") or args.all:
            scores = ", ".join(f"{c}={s:.1f}" for c, s in r.scores)
            print(
                f"{r.host:<20} {r.status:<8} {r.likely:<7} {r.confidence:<6} {r.where or '-':<18} {scores:<34} "
                f"{','.join(r.open_tickets) or '-'}"
            )
    print(f"{len(rows)} hosts; full table in {PROJECT_DIR / args.out}")


def cmd_categories(args: argparse.Namespace) -> None:
    conn = index.connect()
    text = report.categories(report.categories_data(conn, window_days=args.days), window_days=args.days)
    (PROJECT_DIR / args.out).write_text(text)
    print("\n".join(line for line in text.splitlines() if line.startswith(("## ", "- "))))
    print(f"full report in {PROJECT_DIR / args.out}")


def cmd_validate(args: argparse.Namespace) -> None:
    conn = index.connect()
    text = validate.summarize(validate.cases(conn, window_days=args.days))
    out = PROJECT_DIR / args.out
    out.write_text(text)
    print("\n".join(text.splitlines()[:30]))
    print(f"... full table in {out}")


def cmd_canvas(args: argparse.Namespace) -> None:
    conn = index.connect()
    data = canvas.write(conn, Path(args.out), days=args.days)
    print(
        f"{len(data['fleet'])} hosts, {len(data['untracked'])} untracked failing/flaky, "
        f"{sum(c['corroborated'] for c in data['categories'])} corroborated ticket-host pairs -> {args.out}"
    )


def cmd_publish(args: argparse.Namespace) -> None:
    dest = Path(args.to)
    target = index.publish(index.connect(), dest)
    docs = dest / "docs"
    docs.mkdir(exist_ok=True)
    for name in ("fleet.md", "categories.md", "validation.md", "findings.md"):
        src = PROJECT_DIR / "docs" / name
        if src.exists():
            shutil.copyfile(src, docs / name)
    print(
        f"snapshot {target} ({target.stat().st_size / 1e6:.0f} MB) and reports in {docs}; readers: "
        f"DEVICE_TRIAGE_CACHE={dest} python3 -m device_triage triage HOST"
    )


def cmd_stats(args: argparse.Namespace) -> None:
    conn = index.connect()
    q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731
    print(
        "units by source:",
        q("SELECT source, COUNT(*), SUM(n_events), SUM(error IS NOT NULL) FROM units GROUP BY source"),
    )
    print("runs by source/status:", q("SELECT source, status, COUNT(*) FROM runs GROUP BY 1, 2 ORDER BY 1, 2"))
    print("events by category/severity:")
    for row in q("SELECT category, severity, COUNT(*), COUNT(DISTINCT host) FROM events GROUP BY 1, 2 ORDER BY 1, 2"):
        print("  ", row)
    print("top signals (fail/warn):")
    for row in q(
        "SELECT category, signal, severity, COUNT(*), COUNT(DISTINCT host) FROM events WHERE severity != 'info'"
        " GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 60"
    ):
        print("  ", row)
    print(
        "errors:",
        q("SELECT source, error, COUNT(*) FROM units WHERE error IS NOT NULL GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10"),
    )


def cmd_sql(args: argparse.Namespace) -> None:
    conn = index.connect()
    for statement in args.query:
        cur = conn.execute(statement)
        if cur.description:
            print(" | ".join(d[0] for d in cur.description))
        for row in cur.fetchall():
            print(" | ".join("" if v is None else str(v)[: args.width] for v in row))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="device_triage", description="Device status and likely issue from HC logs + Jira."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="incrementally (re)index the HC logs into cache/index.sqlite")
    p.add_argument("--source", action="append", choices=list(index.SOURCES), help="only these sources (repeatable)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, help="at most N units per source (for testing)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("triage", help="status and likely issue for one or more hosts")
    p.add_argument("host", nargs="+")
    p.add_argument("--days", type=int, default=30, help="evidence window (default 30)")
    p.add_argument("--as-of", help="evaluate as of this UTC time, e.g. 2026-10-01T12:00 (default now)")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="more evidence, with file paths")
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("fleet", help="status and likely issue for every host with recent machine HC runs")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--all", action="store_true", help="print healthy hosts too")
    p.add_argument("--out", default="docs/fleet.md")
    p.set_defaults(func=cmd_fleet)

    p = sub.add_parser("categories", help="per category: DIIM tickets and the log evidence for each")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default="docs/categories.md")
    p.set_defaults(func=cmd_categories)

    p = sub.add_parser("validate", help="replay triage at ticket creation and compare with ticket categories")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default="docs/validation.md", help="report path, relative to the project dir")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("canvas", help="write the team overview canvas (fleet, categories, validation)")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default=str(CANVAS_PATH))
    p.set_defaults(func=cmd_canvas)

    p = sub.add_parser("jira-sync", help="read-only Jira REST sync (issues, comments, attachments) into cache/jira")
    p.add_argument("--project", action="append", help="project key (repeatable; default DIIM and DCAMP)")
    p.add_argument("--attachments-for", action="append", help="download attachments for these projects (default DIIM)")
    p.add_argument("--full", action="store_true", help="re-fetch everything instead of issues updated since last sync")
    p.add_argument("--max-mb", type=float, default=25.0, help="skip attachments larger than this")
    p.add_argument(
        "--skip-hc-runs", action="store_true", help="don't rebuild machine HC runs from DCAMP bot-ticket attachments"
    )
    p.set_defaults(func=cmd_jira_sync)

    p = sub.add_parser("tickets", help="(re)load Jira tickets from cache/jira into the index")
    p.set_defaults(func=cmd_tickets)

    p = sub.add_parser("publish", help="write a read-only snapshot + reports for shared readers")
    p.add_argument("--to", required=True, help="shared directory, e.g. /data/device_triage")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("stats", help="what the index holds")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("sql", help="run read-only SQL against the index (no sqlite3 CLI on the box)")
    p.add_argument("query", nargs="+")
    p.add_argument("--width", type=int, default=240, help="truncate each cell to N chars")
    p.set_defaults(func=cmd_sql)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except sqlite3.OperationalError as exc:
        parser.exit(1, f"index error: {exc}\n")
