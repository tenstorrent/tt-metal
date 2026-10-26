# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""D5 (b): score the v1.9 broken outputs on every suite case under the suite's row-count rule, both precisions.

    python models/experimental/tirex2_forecast/tests/perf/broken_suite.py [--out out.json]     (CPU only)

The suite's own floors (``test_model.floors`` + ``applicable``) decide; each broken output must fail at least
one applicable floor, else it is listed as a miss. ``ragged`` is left out: its oracle is a CPU forward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden
from models.experimental.tirex2_forecast.tests.pcc import metrics as M
from models.experimental.tirex2_forecast.tests.pcc.test_model import applicable, floors
from models.experimental.tirex2_forecast.tests.perf.broken_outputs import broken
from models.experimental.tirex2_forecast.tests.perf.check_gates import passes


def cases():
    for name in ("model16", "single", "multivar4"):
        g = golden(name)
        tok = df.readout_tokens(g["horizon"])
        n = g["scale"].shape[0]
        tm = torch.ones(n, dtype=torch.bool)
        if name == "multivar4":
            groups, per, _ = g["context"].shape
            tm = (torch.arange(groups * per) % per) == 0
        yield name, g["head_out"][:, tok], g["q_one"], g["loc"], g["scale"], g["horizon"], tm, name == "single"
    g = golden("bench_long_context")
    yield "long_context", g["head_out"], g["q_one"], g["loc"], g["scale"], g["horizon"], torch.ones(
        32, dtype=torch.bool
    ), False
    for row in (0, 7):
        r = slice(row, row + 1)
        yield f"long_context[row {row}]", g["head_out"][r], g["q_one"][r], g["loc"][r], g["scale"][r], g[
            "horizon"
        ], torch.ones(1, dtype=torch.bool), True
    g = golden("bench_paper")
    r = slice(0, 64)
    yield "paper64", g["head_out"][r], g["q_one"][r], g["loc"][r], g["scale"][r], g["horizon"], torch.ones(
        64, dtype=torch.bool
    ), False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    results, miss = [], []
    for name, g_head, g_q, loc, scale, horizon, tm, single in cases():
        for bname, (head, what) in broken(g_head).items():
            q = M.forecast_from_head(head, loc, scale, horizon, tm)
            m = M.gate_metrics(head, g_head, q, g_q, scale[tm])
            for precision in ("accuracy", "performance"):
                f = applicable(floors(precision, single), m["active_rows"], f"{name}/{precision}/{bname}")
                fails = [k for k, v in f.items() if not passes(k, m[k], v)]
                results.append({"case": name, "broken": bname, "rows": what, "precision": precision, "fails": fails})
                print(f"{name:22s} {precision:11s} {bname:19s} ({what}): fails {len(fails)} {fails}")
                if not fails:
                    miss.append(f"{name}/{precision}/{bname}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"{len(miss)} miss(es)" + (": " + ", ".join(miss) if miss else ""))
    sys.exit(1 if miss else 0)


if __name__ == "__main__":
    main()
