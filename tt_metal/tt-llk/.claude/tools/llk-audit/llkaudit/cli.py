# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
CLI: load a merged fact base, run the selected checks, emit one advisory JSON
envelope.

  python -m llkaudit.cli --arch wormhole --facts merged.jsonl [--checks a,b]

The fact-base input is the concatenated per-file JSON the extractor prints
(run.sh produces it). Output envelope:

  { tool, arch, authority: "advisory", parse_errors,
    checks: { <name>: { description, blind_spots, count, findings: [...] } } }

Exit code is ALWAYS 0 (augmentor, never a gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import registry
from .checks import ALL
from .factbase import FactBase


def _find_metal_root(start: str) -> str | None:
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "tt_metal", "hw", "inc")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def scope_to_changed(findings: list, changed_files: list) -> list:
    """Diff-scoped filter: keep only findings that TOUCH a changed file — either
    the anchor file, or any evidence line (so a cross-thread shared word whose
    partner writer is in a changed file surfaces even when anchored elsewhere).
    The fact base is still built over the whole tree; only OUTPUT is scoped.
    Empty `changed_files` = no scoping (return all)."""
    if not changed_files:
        return findings
    base = {os.path.basename(c) for c in changed_files}

    def touches(fd: dict) -> bool:
        if os.path.basename(fd.get("file", "")) in base:
            return True
        return any(cb in ev for ev in fd.get("evidence", []) for cb in base)

    return [f for f in findings if touches(f)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llkaudit")
    ap.add_argument("--arch", required=True)
    ap.add_argument(
        "--facts",
        required=True,
        help="file of concatenated per-file extractor JSON (- for stdin)",
    )
    ap.add_argument(
        "--checks",
        default="all",
        help="comma-separated check names, or 'all' (default)",
    )
    ap.add_argument(
        "--metal-root",
        default=None,
        help="repo root for cfg_defines.h (auto-detected if omitted)",
    )
    ap.add_argument(
        "--changed-files",
        default="",
        help="comma-separated file paths (diff-scoped mode). When set, only "
        "findings that touch one of these files (anchor OR any evidence line) "
        "are reported. run.sh --changed populates this from git.",
    )
    ap.add_argument("--list", action="store_true", help="list checks and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, cls in ALL.items():
            print(f"{name:22} {cls.description}")
        return 0

    text = sys.stdin.read() if args.facts == "-" else open(args.facts).read()
    fb = FactBase.from_concatenated_json(args.arch, text)

    # Resolve cfg_defines.h ADDR32 words (for cfg-word-overlap). Best-effort.
    root = args.metal_root or _find_metal_root(os.getcwd())
    rel = registry.CFG_DEFINES_REL.get(args.arch)
    if root and rel:
        fb.addr32 = registry.load_addr32(os.path.join(root, rel))

    if args.checks == "all":
        selected = list(ALL.keys())
    else:
        selected = [c.strip() for c in args.checks.split(",") if c.strip()]
        for c in selected:
            if c not in ALL:
                ap.error(f"unknown check '{c}'. Available: {', '.join(ALL)}")

    changed = [c.strip() for c in args.changed_files.split(",") if c.strip()]

    out = {
        "tool": "llk-audit",
        "arch": args.arch,
        "authority": "advisory",
        "parse_errors": fb.parse_errors,
        "scoped_to_changed": bool(changed),
        "checks": {},
    }
    if changed:
        out["changed_files"] = sorted({os.path.basename(c) for c in changed})
    for name in selected:
        chk = ALL[name]()
        findings = scope_to_changed([f.to_dict() for f in chk.run(fb)], changed)
        out["checks"][name] = {
            "description": chk.description,
            "blind_spots": chk.blind_spots,
            "count": len(findings),
            "findings": findings,
        }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
