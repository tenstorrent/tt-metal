# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn the committed suite log into ``logs/pcc_summary.txt``.

Every PCC number quoted in the README and the context contract comes from an
``assert_pcc`` line in ``logs/full_test_run.log``, so this renders them all in one
sorted table, split the way the bars split: the released **checkpoint** against
``PCC_THRESHOLD`` (0.995), the i.i.d.-Gaussian **synthetic** harness against the
documented looser bar, and — new to this stage — the **multichip-vs-single-chip**
comparison against 0.999, which is the only population that sees the
parallelisation rather than the precision policy.

    python summarize_pcc.py [--check]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/multichip_decoder/
#: Two invocations, two logs; see ``refresh_context_contract.py``.
LOGS = (ROOT / "logs/full_test_run.log", ROOT / "logs/vs_single_chip_run.log")
LOG = LOGS[0]
OUT = ROOT / "logs/pcc_summary.txt"

LINE = re.compile(r"(?:assert_pcc|test_multichip_matches_single_chip):\d+ - (.+?): ([0-9.]+)\s*$")


def read() -> dict[str, float]:
    out: dict[str, float] = {}
    for log in LOGS:
        for raw in log.read_text(errors="ignore").splitlines():
            match = LINE.search(raw.strip())
            if match:
                out[match.group(1)] = float(match.group(2))
    return out


def split(pccs: dict[str, float]):
    versus = {k: v for k, v in pccs.items() if "vs single-chip" in k}
    # The two-layer chain composes two layers' precision error, so it has its own
    # bar; pooling it with the single-layer population would report its value as
    # the single-layer worst case, which is the misreading this split prevents.
    stacked = {k: v for k, v in pccs.items() if "two-layer stack" in k}
    rest = {k: v for k, v in pccs.items() if k not in versus and k not in stacked}
    real = {k: v for k, v in rest.items() if "real-weight" in k}
    synthetic = {k: v for k, v in rest.items() if "real-weight" not in k}
    return versus, real, synthetic, stacked


def render(pccs: dict[str, float]) -> str:
    versus, real, synthetic, stacked = split(pccs)

    def worst(group):
        return min(group.values(), default=float("nan"))

    lines = [
        "Muse-Glimmer-30B multichip decoder - every asserted PCC check",
        "source: " + ", ".join(str(log.relative_to(ROOT.parent.parent)) for log in LOGS),
        "",
        f"{len(pccs)} asserted checks",
        f"  {len(versus):4d} multichip vs single-chip TTNN, bar 0.999, worst {worst(versus):.6f}",
        f"  {len(real):4d} vs HF on the released bf16 checkpoint, bar 0.995, worst {worst(real):.6f}",
        f"  {len(synthetic):4d} vs HF on i.i.d.-Gaussian synthetic weights, single layer, bar 0.99, "
        f"worst {worst(synthetic):.6f}",
        f"  {len(stacked):4d} vs HF chained through two layers, bar 0.96, worst {worst(stacked):.6f}",
        "",
        "-- multichip vs single-chip TTNN (the only population that sees the fracture) --",
    ]
    for label, value in sorted(versus.items(), key=lambda kv: kv[1]):
        lines.append(f"{value:.6f}  {label}")
    lines += ["", "-- released checkpoint (the acceptance bar) --"]
    for label, value in sorted(real.items(), key=lambda kv: kv[1]):
        lines.append(f"{value:.6f}  {label}")
    lines += ["", "-- two chained layers (bar 0.96: two layers compose two layers' error) --"]
    for label, value in sorted(stacked.items(), key=lambda kv: kv[1]):
        lines.append(f"{value:.6f}  {label}")
    lines += ["", "-- synthetic weights, single layer --"]
    for label, value in sorted(synthetic.items(), key=lambda kv: kv[1]):
        lines.append(f"{value:.6f}  {label}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 if the summary is stale")
    args = parser.parse_args()
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
