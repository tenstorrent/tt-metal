# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The dashboard's published links (F71): where the overseer publishes, and when it last did.

Only an interactive Claude session can publish a page to claude.ai (the Artifact tool; a `claude -p` run started by
the framework does not get it), so the orchestrator refreshes the local files after every gate and the overseer
publishes them. `<bringup_dir>/dashboard/links.yaml` keeps the links (so every session publishes to the same pages)
and the last publish time; the watch wakes the overseer with DASHBOARD STALE when the local files are newer than the
last publish by more than `stale_after` (default 15 minutes).

    python -m models.demos.common.bringup.dashboard.links --spec S set --style standard --url URL
    python -m models.demos.common.bringup.dashboard.links --spec S published      # after publishing every link
    python -m models.demos.common.bringup.dashboard.links --spec S show
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

FILE = "links.yaml"
STALE_AFTER = 15 * 60


def path(bringup_dir: Path) -> Path:
    return Path(bringup_dir) / "dashboard" / FILE


def load(bringup_dir: Path) -> dict:
    p = path(bringup_dir)
    return (yaml.safe_load(p.read_text()) if p.exists() else None) or {}


def save(bringup_dir: Path, data: dict) -> None:
    p = path(bringup_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))


PUBLISHED = "dashboard_published.json"  # in the run dir, outside git (F74): a commit's pre-commit stash cannot hide it


def published_at(run_dir: Path | None) -> float:
    p = Path(run_dir) / PUBLISHED if run_dir else None
    return float(json.loads(p.read_text()).get("at", 0)) if p and p.exists() else 0.0


def mark_published(run_dir: Path) -> dict:
    rec = {"at": time.time(), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (Path(run_dir) / PUBLISHED).write_text(json.dumps(rec) + "\n")
    return rec


def stale(bringup_dir: Path, stale_after: float = STALE_AFTER, now: float | None = None, run_dir=None) -> str | None:
    """The DASHBOARD STALE line when the published links lag the local files by more than stale_after, else None."""
    d = load(bringup_dir)
    links = d.get("links") or {}
    if not links:
        return None
    files = [Path(bringup_dir) / "dashboard" / f for f in ("index.html", "teletext.html")]
    newest = max((f.stat().st_mtime for f in files if f.exists()), default=0)
    published = published_at(run_dir) if run_dir else float(d.get("published_at") or 0)
    now = now or time.time()
    if newest > published and now - max(published, 0) > stale_after:
        mins = (now - published) / 60 if published else float("inf")
        return (
            f"DASHBOARD STALE: the local dashboard is newer than its last publish ({mins:.0f} min ago); publish "
            + ", ".join(f"{k} -> {v}" for k, v in links.items())
            + ", then `dashboard.links published`"
        )
    return None


def main(argv=None) -> int:
    from models.demos.common.bringup.core.spec import Spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("--style", required=True, help="standard | teletext")
    s.add_argument("--url", required=True)
    sub.add_parser("published")
    sub.add_parser("show")
    a = ap.parse_args(argv)
    from models.demos.common.bringup.core.gate import run_name
    from models.demos.common.bringup.core.ledger import Ledger

    spec = Spec.load(a.spec)
    bdir = spec.bringup_dir
    run_dir = spec.run_dir(run_name(Ledger(bdir)))
    d = load(bdir)
    if a.cmd == "set":
        d.setdefault("links", {})[a.style] = a.url
        save(bdir, d)
    elif a.cmd == "published":
        print("published", mark_published(run_dir)["t"])
    print(yaml.safe_dump(load(bdir), sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
