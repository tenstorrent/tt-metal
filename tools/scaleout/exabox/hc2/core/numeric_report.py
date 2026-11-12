#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""numeric_report.py: the end-of-run numeric lines from the tries' --result JSONs (ub0_repro.py's
result['numeric'], numeric_check.py). Standard library only. Never changes a run's result: exit 0 always.

  numeric_report.py <file or dir> ...     dirs are searched for *.json with a 'numeric' section (run.sh: try<i>.json,
                                          load_test.sh: run<NN>_q<Q>_t<T>/ub0_<host>_<utc>/result.json)

Lines (one per mismatched chip, then one per clean tray; chips summed over every try):
  NUMERIC FAIL <BDF> (tray N) chip N: x/y mismatched, first at iter K in <try>, max abs err E, NaN count M
  NUMERIC PASS tray N (8 chips, C checks)
  NUMERIC NOT CHECKED tray N (...)           a tray with no checked chip (e.g. its chips dropped before a burst ended)
  NUMERIC WARN tray N (<BDFs>): ...         a burst's checked outputs left unread (budget spent or checks off,
                                          ub0_repro.py result['burst']['numeric_skipped'])
  NUMERIC ERROR <try>: <why>                a check that could not run (the chip is not blamed)
  NUMERIC: not checked (...)                no try wrote a numeric section (all-gather load, TRAY_NUMERIC=0, ...)
"""
import glob
import json
import os
import re
import sys

UBBS = (0, 1, 2, 3)


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def find(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        elif os.path.isfile(p):
            out.append(p)
    return sorted(set(out), key=natural)


def label(path, roots):
    for r in roots:
        if os.path.isdir(r) and os.path.abspath(path).startswith(os.path.abspath(r) + os.sep):
            rel = os.path.relpath(path, r)
            return rel.split(os.sep)[0] if os.sep in rel else os.path.splitext(rel)[0]
    return os.path.splitext(os.path.basename(path))[0]


def bdf(pci):
    return pci[5:] if pci and re.match(r"^[0-9a-f]{4}:", pci) else (pci or "?")


def collect(paths):
    tries, chips, errors = [], {}, []
    for f in find(paths):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        n = r.get("numeric") if isinstance(r, dict) else None
        if not isinstance(n, dict) or not n.get("enabled"):
            continue
        t = label(f, paths)
        tries.append((t, n.get("checks", 0)))
        if n.get("error"):
            errors.append(f"NUMERIC ERROR {t}: {n['error']}")
        for c in (n.get("chips") or {}).values():
            key = c.get("pci") or f"metal{c.get('metal')}"
            a = chips.setdefault(key, dict(pci=c.get("pci"), ubb=c.get("ubb"), dev=c.get("dev"), checks=0,
                                           mismatched=0, first=None, max_abs_err=0.0, nan=0, inf=0))
            a["checks"] += c.get("checks", 0)
            a["mismatched"] += c.get("mismatched", 0)
            a["nan"] += c.get("nan", 0)
            a["inf"] += c.get("inf", 0)
            e = c.get("max_abs_err")
            if c.get("mismatched"):
                a["max_abs_err"] = float("inf") if e is None else max(a["max_abs_err"], e)
            if c.get("mismatched") and a["first"] is None and c.get("first_iter") is not None:
                a["first"] = (c["first_iter"], t)
    return tries, chips, errors


def skip_warns(paths, chips=None):
    """One NUMERIC WARN per tray per try whose burst left checked outputs unread, naming the tray's BDFs."""
    out = []
    for f in find(paths):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        b = r.get("burst") if isinstance(r, dict) else None
        if not isinstance(b, dict) or not b.get("numeric_skipped"):
            continue
        n = r.get("numeric") if isinstance(r.get("numeric"), dict) else {}
        trays = {}
        for c in (n.get("chips") or {}).values():
            trays.setdefault(c.get("ubb"), []).append(bdf(c.get("pci")))
        if not trays:   # no chips in this try's numeric section: take the trays and BDFs other tries read
            for c in (chips or {}).values():
                trays.setdefault(c.get("ubb"), []).append(bdf(c.get("pci")))
        why = (f"{b['numeric_skipped']} of {b.get('numeric_pending', '?')} burst checked outputs not read back in "
               f"{label(f, paths)} (--numeric-budget-s spent or checks off; ops 2..n-1 unchecked, only op 1 and the last were read back; a mismatch in between goes unseen)")
        for u in sorted(trays, key=lambda u: (u is None, u)) or [None]:
            bd = sorted(set(trays.get(u, [])))
            out.append(f"NUMERIC WARN tray {u + 1 if u is not None else '?'} ({len(bd)} chips: {' '.join(bd) or 'BDF unknown, no chip in any try'}): {why}")
    return out


def lines(paths):
    tries, chips, errors = collect(paths)
    if not tries:
        return ["NUMERIC: not checked (no try wrote a numeric result: all-gather load, TRAY_NUMERIC=0, or no burst ended)"]
    out = []
    fails = sorted((c for c in chips.values() if c["mismatched"]), key=lambda c: (c["ubb"] is None, c["ubb"], c["pci"] or ""))
    for c in fails:
        k, t = c["first"] or ("?", "?")
        e = "inf (non-finite)" if c["max_abs_err"] == float("inf") else f"{c['max_abs_err']:.4g}"
        out.append(f"NUMERIC FAIL {bdf(c['pci'])} (tray {c['ubb'] + 1 if c['ubb'] is not None else '?'}) chip {c['dev']}: "
                   f"{c['mismatched']}/{c['checks']} mismatched, first at iter {k} in {t}, max abs err {e}, "
                   f"NaN count {c['nan']}" + (f", Inf count {c['inf']}" if c["inf"] else ""))
    for u in UBBS:
        mine = [c for c in chips.values() if c["ubb"] == u]
        if any(c["mismatched"] for c in mine):
            continue
        done = [c for c in mine if c["checks"]]
        if not done:
            out.append(f"NUMERIC NOT CHECKED tray {u + 1} (no chip of it in any try's numeric result)")
            continue
        n = min(c["checks"] for c in done)
        m = max(c["checks"] for c in done)
        out.append(f"NUMERIC PASS tray {u + 1} ({len(done)} chips, {n if n == m else f'{n}-{m}'} checks)")
    unk = [c for c in chips.values() if c["ubb"] is None and not c["mismatched"] and c["checks"]]
    if unk:
        out.append(f"NUMERIC PASS tray ? ({len(unk)} chips with no PCI address, {min(c['checks'] for c in unk)} checks)")
    return out + skip_warns(paths, chips) + errors


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    try:
        print("\n".join(lines(argv)), flush=True)
    except Exception as e:      # the report must never fail a run
        print(f"NUMERIC: report error {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
