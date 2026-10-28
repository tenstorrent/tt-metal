# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""The v1.14 / v1.31 variance table of a precision change, from device dumps (CPU, D5 item 1).

    python models/experimental/flowstate_forecast/tests/perf/variance_table.py r0=<dir> cand=<dir>[,<dir>] ...
        [--ceiling suite:single_long/accuracy=0.017] [--out table.md]

Each ``label=dir[,dir]`` names one set of dumps (``<shape>-<precision>.pt`` and ``suite_<case>-<precision>.pt``,
``combined_normalized`` of the head forward). Per case and precision, for each set: abs_err_max, p99 |err|, the
signed mean error and the per-quantile mean error (normalized head, against the golden), and with ``--ceiling``
the count of elements over that case's ceiling (v1.31: "single point" is a count).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS, TEST_CASES
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tests.perf.broken_outputs import BENCH


def find(dirs, key, p):
    name = f"suite_{key[6:]}-{p}.pt" if key.startswith("suite:") else f"{key}-{p}.pt"
    for d in dirs:
        if (d / name).is_file():
            return d / name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="+", help="label=dir[,dir]")
    ap.add_argument("--ceiling", action="append", default=[], help="<case>/<precision>=<abs_err_max ceiling>")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sets = [(s.split("=", 1)[0], [Path(d) for d in s.split("=", 1)[1].split(",")]) for s in args.sets]
    ceil = {c.split("=")[0]: float(c.split("=")[1]) for c in args.ceiling}
    lines = [
        "| case | precision | set | abs_err_max | p99 abs err | signed mean | over ceiling | mean err q0..q8 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in list(BENCH) + [f"suite:{c}" for c in TEST_CASES]:
        g = torch.load(
            GOLDENS / (f"{key[6:]}.pt" if key.startswith("suite:") else f"bench_{key}.pt"), weights_only=True
        )
        gh = M.head(g["combined_normalized"]).double()
        for p in ("accuracy", "performance"):
            c = ceil.get(f"{key}/{p}")
            for label, dirs in sets:
                f = find(dirs, key, p)
                if f is None:
                    lines.append(f"| {key} | {p} | {label} | no dump | | | | |")
                    continue
                d = M.head(torch.load(f, weights_only=True)["combined_normalized"]).double() - gh
                a = d.abs().flatten()
                over = f"{int((a > c).sum())} of {a.numel()}" if c is not None else ""
                qm = " ".join(f"{float(v):+.4f}" for v in d.mean((0, 2)))
                lines.append(
                    f"| {key} | {p} | {label} | {float(a.max()):.4f} | {float(torch.quantile(a.float(), 0.99)):.4f} "
                    f"| {float(d.mean()):+.5f} | {over} | {qm} |"
                )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
