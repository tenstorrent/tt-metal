# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
CLI: load a merged fact base, run the selected checks, emit one advisory JSON
envelope.

  python -m llkaudit.cli --arch wormhole --facts merged.jsonl [--checks a,b]

The fact-base input is the concatenated per-file JSON the extractor prints
(run.sh produces it). Output envelope:

  { tool, arch, authority: "advisory", parse_errors, scoped_to_changed,
    changed_files? (diff-scoped only),
    checks: { <name>: { description, blind_spots, count, findings: [...] } } }

A successful run exits 0 (augmentor, never a gate — it does not fail on findings).
Argument errors (unknown check, unreadable --facts, bad --arch) exit 2 via
argparse, as usual.
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
        # Every evidence line starts with "basename:line…", so anchor to that
        # PREFIX (startswith), not a bare substring — otherwise scoping to
        # `params.h` would leak a finding whose evidence is `..._sfpu_params.h:12`
        # (its basename ends with, but isn't, `params.h`).
        return any(
            ev.startswith(f"{cb}:") for ev in fd.get("evidence", []) for cb in base
        )

    return [f for f in findings if touches(f)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llkaudit")
    # --arch/--facts are needed for an audit but NOT for --list, so they are not
    # argparse-`required` (that would make `--list` exit 2 before it runs); they
    # are validated manually below once --list is handled.
    ap.add_argument("--arch", choices=("wormhole", "blackhole", "quasar"))
    ap.add_argument(
        "--facts",
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

    # Not listing -> an audit; now --arch and --facts are required.
    if not args.arch:
        ap.error("--arch is required (choices: wormhole, blackhole, quasar)")
    if not args.facts:
        ap.error("--facts is required (- for stdin)")

    if args.facts == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.facts) as fh:
                text = fh.read()
        except OSError as e:
            ap.error(f"cannot read --facts {args.facts!r}: {e}")
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
        # An empty/whitespace/comma-only --checks must be a hard error, never a
        # silent 0-check "clean" pass (the worst outcome for a recall augmentor).
        if not selected:
            ap.error("no checks selected (--checks was empty). Use 'all' or names.")

    changed = [c.strip() for c in args.changed_files.split(",") if c.strip()]

    # Degradation guard: cfg-word-overlap silently loses its CROSS_THREAD_SHARED_WORD
    # detection (emits only UNRESOLVED, exit 0) when the cfg_defines header can't be
    # read (addr32 empty) — a false all-clear for a class the arch DOES support. The
    # arch has a CFG_DEFINES_REL entry, so an empty addr32 means the header/root was
    # not found, not that the arch lacks defines. Warn loudly AND mark the envelope;
    # this is the one input-degradation path that was not surfaced like the others.
    cfgword_degraded = "cfg-word-overlap" in selected and bool(rel) and not fb.addr32
    if cfgword_degraded:
        print(
            f"llk-audit: WARNING cfg-word-overlap DEGRADED — could not load cfg "
            f"defines ({rel}); field masks unresolved, so CROSS_THREAD_SHARED_WORD "
            f"detection is OFF (only UNRESOLVED emitted). This is NOT a clean "
            f"'no shared words' result — check --metal-root / the header path.",
            file=sys.stderr,
        )

    out = {
        "tool": "llk-audit",
        "arch": args.arch,
        "authority": "advisory",
        "parse_errors": fb.parse_errors,
        "scoped_to_changed": bool(changed),
        "checks": {},
    }
    if cfgword_degraded:
        out["degraded"] = [
            f"cfg-word-overlap: cfg_defines ({rel}) unreadable — field masks "
            "unresolved, CROSS_THREAD_SHARED_WORD detection OFF"
        ]
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
