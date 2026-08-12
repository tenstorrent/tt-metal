# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn the committed suite log into ``logs/pcc_summary.txt``.

Every PCC number quoted in the README and the context contract comes from an
``assert_pcc`` line in ``logs/full_test_run.log``, so this renders them all in one
sorted table and splits them the way the acceptance bars split: **real
checkpoint** weights against ``PCC_THRESHOLD`` (0.995) and **synthetic**
i.i.d.-Gaussian weights against the looser documented bar.  Keeping the split
explicit is the point -- the shipped BFP4 MLP policy is selected on the real-weight
column, and a summary that pooled them would hide that.

    python summarize_pcc.py [--check]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/optimized_decoder/
LOG = ROOT / "logs/full_test_run.log"
OUT = ROOT / "logs/pcc_summary.txt"

LINE = re.compile(r"assert_pcc:\d+ - (.+?): ([0-9.]+)\s*$")


def read() -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in LOG.read_text(errors="ignore").splitlines():
        match = LINE.search(raw.strip())
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def render(pccs: dict[str, float]) -> str:
    real = {k: v for k, v in pccs.items() if "real-weight" in k}
    synthetic = {k: v for k, v in pccs.items() if "real-weight" not in k}
    lines = [
        "Muse-Glimmer-30B optimized decoder - every asserted PCC check",
        f"source: {LOG.relative_to(ROOT.parent.parent)}",
        "",
        f"{len(pccs)} asserted HF-vs-TTNN checks",
        f"  {len(real):4d} on the released bf16 checkpoint, bar 0.995, worst "
        f"{min(real.values(), default=float('nan')):.6f}",
        f"  {len(synthetic):4d} on i.i.d.-Gaussian synthetic weights, worst "
        f"{min(synthetic.values(), default=float('nan')):.6f}",
        "",
        "-- released checkpoint (the bar the precision policy is selected on) --",
    ]
    for label, value in sorted(real.items(), key=lambda kv: kv[1]):
        lines.append(f"{value:.6f}  {label}")
    lines += ["", "-- synthetic weights --"]
    for label, value in sorted(synthetic.items(), key=lambda kv: kv[1]):
        lines.append(f"{value:.6f}  {label}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the summary is stale")
    args = ap.parse_args()
    pccs = read()
    if not pccs:
        print(f"no assert_pcc lines in {LOG}", file=sys.stderr)
        return 1
    text = render(pccs)
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is stale against {LOG}", file=sys.stderr)
            return 1
        print(f"{OUT} matches {LOG}")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(pccs)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
