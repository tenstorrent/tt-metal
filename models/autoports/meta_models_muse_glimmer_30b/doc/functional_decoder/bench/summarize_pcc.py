# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``logs/pcc_summary.txt`` from the committed suite log.

Every PCC number quoted in ``README.md``, ``work_log.md`` and the
``functional_decoder`` block of ``doc/context_contract.json`` traces back to this
file, which is a pure transcription of the ``assert_pcc`` lines in
``logs/full_test_run.log``.  Generating it with a committed script (rather than
an ad-hoc snippet) is what makes those numbers re-derivable.

Usage::

    python summarize_pcc.py                    # rewrite logs/pcc_summary.txt
    python summarize_pcc.py --check            # exit 1 if it would change
    python summarize_pcc.py --compare <log>    # diff another suite log's PCCs
                                               # against the committed run
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/functional_decoder/
SUITE_LOG = ROOT / "logs/full_test_run.log"
SUMMARY = ROOT / "logs/pcc_summary.txt"

#: the acceptance bar asserted by ``tests/test_functional_decoder.py``
PCC_THRESHOLD = 0.995


def read_pccs(log: pathlib.Path) -> list[tuple[float, str]]:
    """``[(value, label)]`` for every ``assert_pcc`` line in the suite log."""
    rows: list[tuple[float, str]] = []
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"assert_pcc:\d+ - (.+?): ([0-9.]+)$", line.strip())
        if m:
            rows.append((float(m.group(2)), m.group(1)))
    return rows


def render(rows: list[tuple[float, str]]) -> str:
    ordered = sorted(rows)
    out = [
        "# Functional decoder PCC summary",
        f"# {len(ordered)} asserted HF-vs-TTNN checks, worst {ordered[0][0]:.6f} "
        f"(bar {PCC_THRESHOLD}), best {ordered[-1][0]:.6f}",
        "# ascending; generated from logs/full_test_run.log by bench/summarize_pcc.py",
        "",
    ]
    out += [f"{pcc:.6f}  {label}" for pcc, label in ordered]
    return "\n".join(out) + "\n"


def compare(other: pathlib.Path) -> int:
    """Compare another suite log's PCC checks against the committed run.

    Used to re-verify the committed evidence on live hardware without replacing
    it: the layer is deterministic, so a rerun of the same code must reproduce
    every value.
    """
    base = dict((label, pcc) for pcc, label in read_pccs(SUITE_LOG))
    new = dict((label, pcc) for pcc, label in read_pccs(other))
    missing = sorted(set(base) - set(new))
    extra = sorted(set(new) - set(base))
    deltas = {label: new[label] - base[label] for label in set(base) & set(new) if new[label] != base[label]}

    print(f"committed {SUITE_LOG.name}: {len(base)} checks")
    print(f"rerun     {other.name}: {len(new)} checks")
    print(f"labels only in committed run: {len(missing)}")
    print(f"labels only in rerun:         {len(extra)}")
    print(f"values that differ:           {len(deltas)}")
    for label in missing[:10]:
        print(f"  missing: {label}")
    for label in extra[:10]:
        print(f"  extra:   {label}")
    for label, delta in sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"  delta {delta:+.3e}  {label} ({base[label]:.9f} -> {new[label]:.9f})")
    if missing or extra or deltas:
        print("RERUN_DIFFERS")
        return 1
    print(f"RERUN_IDENTICAL {len(base)} checks bit-identical to the committed run")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the summary is stale")
    ap.add_argument("--compare", metavar="LOG", help="compare another suite log against the committed one")
    args = ap.parse_args()

    if args.compare:
        return compare(pathlib.Path(args.compare))

    rows = read_pccs(SUITE_LOG)
    if not rows:
        print(f"no assert_pcc lines in {SUITE_LOG}", file=sys.stderr)
        return 2
    text = render(rows)

    if args.check:
        if not SUMMARY.is_file() or SUMMARY.read_text() != text:
            print(f"{SUMMARY} is stale against {SUITE_LOG}", file=sys.stderr)
            return 1
        print(f"{SUMMARY} matches {SUITE_LOG}")
        return 0

    SUMMARY.write_text(text)
    worst = min(rows)
    print(f"wrote {SUMMARY} - {len(rows)} checks, worst {worst[0]:.6f} ({worst[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
