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
    degraded? (list of input-degradation notes — empty fact base / unreadable
               cfg_defines; present ONLY when the run is degraded, so a degraded
               result can never read as a clean audit),
    checks: { <name>: { description, blind_spots, count,
      findings: [ { file, line, function, kind, hint, detail, evidence,
                    safety? (a sub-annotation like SAFE_BY_MASKING /
                    UNRESOLVED_COWRITER / LOW_CONFIDENCE — never a pass/fail) } ] } } }

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

    # Input-degradation guards — surface each as a stderr WARNING and an envelope
    # `degraded` entry so a degraded run can never read as a clean audit (the
    # false-all-clear the augmentor must refuse). run.sh/bootstrap.sh pre-guard the
    # empty case before invoking the CLI, but the CLI is a documented standalone
    # entry point (`python -m llkaudit.cli --facts ...`), so it carries the guard too.
    degraded: list = []
    # (1) Empty fact base — 0 facts parsed from --facts (empty/failed extraction):
    # every check would report count 0 with parse_errors 0, indistinguishable from a
    # genuinely clean audit.
    if not fb.facts:
        print(
            "llk-audit: WARNING empty fact base — 0 facts parsed from --facts. Every "
            "check will report 0; this is NOT a clean audit (nothing was analyzed).",
            file=sys.stderr,
        )
        degraded.append(
            "empty fact base: 0 facts parsed — nothing analyzed, not a clean audit"
        )
    # (2) cfg-word-overlap loses its CROSS_THREAD_SHARED_WORD detection (only
    # UNRESOLVED, exit 0) when the cfg_defines header can't be read (addr32 empty)
    # though the arch HAS a CFG_DEFINES_REL entry (header/root not found).
    cfgword_degraded = "cfg-word-overlap" in selected and bool(rel) and not fb.addr32
    if cfgword_degraded:
        print(
            f"llk-audit: WARNING cfg-word-overlap DEGRADED — could not load cfg "
            f"defines ({rel}); field masks unresolved, so CROSS_THREAD_SHARED_WORD "
            f"detection is OFF (only UNRESOLVED emitted). This is NOT a clean "
            f"'no shared words' result — check --metal-root / the header path.",
            file=sys.stderr,
        )
        degraded.append(
            f"cfg-word-overlap: cfg_defines ({rel}) unreadable — field masks "
            "unresolved, CROSS_THREAD_SHARED_WORD detection OFF"
        )
    # (3) Diff-scoped run whose changed file(s) contributed NO facts. The fact base
    # is built over the whole tree and only OUTPUT is scoped, so a clean 0-finding
    # result can mean the changed file was never analyzed (a .cpp/test/build file, or
    # a header the extractor emitted nothing for) — NOT that it is race-free.
    if changed:
        fact_files = {os.path.basename(f.get("file", "")) for f in fb.facts}
        missing = sorted({os.path.basename(c) for c in changed} - fact_files)
        if missing:
            print(
                "llk-audit: WARNING changed file(s) contributed NO facts: "
                f"{', '.join(missing)} — either not parsed or with no modeled "
                "construct; a 0-finding result for them is NOT 'race-free'.",
                file=sys.stderr,
            )
            degraded.append(
                "diff-scoped: changed file(s) with 0 facts (verify they parsed): "
                + ", ".join(missing)
            )

    out = {
        "tool": "llk-audit",
        "arch": args.arch,
        "authority": "advisory",
        "parse_errors": fb.parse_errors,
        "scoped_to_changed": bool(changed),
        "checks": {},
    }
    if degraded:
        out["degraded"] = degraded
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
