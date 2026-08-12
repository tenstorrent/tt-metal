# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``logs/pcc_summary.txt`` from the committed suite log.

Every PCC number quoted in the README and work log traces back to this file,
which is a pure transcription of the ``assert_pcc`` and accuracy-control lines
in ``logs/full_test_run.log``.  Generating it with a committed script (rather
than an ad-hoc snippet) is what makes those numbers re-derivable.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/fused_decoder/
SUITE_LOG = ROOT / "logs/full_test_run.log"
SUMMARY = ROOT / "logs/pcc_summary.txt"


def main() -> int:
    rows, controls = [], []
    for line in SUITE_LOG.read_text(errors="ignore").splitlines():
        m = re.search(r"assert_pcc:\d+ - (.+?): ([0-9.]+)$", line.strip())
        if m:
            rows.append((float(m.group(2)), m.group(1)))
        m = re.search(r"accuracy vs HF (.+?): unfused=([0-9.]+) fused=([0-9.]+) delta=([-+0-9.]+)", line)
        if m:
            controls.append(m.groups())
    rows.sort()
    hf = [r for r in rows if "vs unfused" not in r[1]]
    eq = [r for r in rows if "vs unfused" in r[1]]

    with SUMMARY.open("w") as f:
        f.write("# Fused decoder PCC summary\n")
        f.write(f"# {len(rows)} asserted checks: {len(hf)} HF-vs-TTNN (worst {hf[0][0]:.6f}, bar 0.995),\n")
        f.write(f"#   {len(eq)} fused-vs-unfused equivalence (worst {eq[0][0]:.6f}, bar 0.995)\n")
        f.write(f"# plus {len(controls)} accuracy controls (fused vs unfused, both against the same HF reference)\n")
        f.write("# generated from logs/full_test_run.log by bench/summarize_pcc.py\n\n")
        for pcc, label in rows:
            f.write(f"{pcc:.6f}  {label}\n")
        f.write("\n# accuracy controls: prefill must strictly improve; decode has a 5e-4 BF16 band\n")
        for window, unfused, fused, delta in controls:
            f.write(f"  {window:44s} unfused={unfused} fused={fused} delta={delta}\n")

    print(
        f"wrote {SUMMARY} - {len(rows)} asserted checks "
        f"({len(hf)} HF worst {hf[0][0]:.6f}, {len(eq)} equivalence worst {eq[0][0]:.6f}), "
        f"{len(controls)} controls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
