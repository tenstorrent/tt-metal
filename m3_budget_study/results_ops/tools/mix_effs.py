#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blend CAL.effs of one mesh from several stage kinds into one "mixed" set: per op, eff = 1 / sum_i(w_i / eff_i).

That is the eff of the weighted mean op time (same roofline), i.e. a pipeline whose layers are split across stage
kinds in proportion w_i and rebalanced so every stage finishes together. The sim takes one effs set per mesh, not
per stage, so this is how a galaxy with 2 torus and 2 non-torus [4,2] stages is approximated.

  mix_effs.py --mesh 4x2 --in A.json:0.5,B.json:0.5 --out MIX.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="4x2")
    ap.add_argument("--in", dest="inputs", required=True, help="file:weight[,file:weight..]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in a.inputs.split(",")]
    tot = sum(w for _, w in parts)
    effs = [(json.load(open(f))[a.mesh], w / tot) for f, w in parts]
    out = {}
    for kind in ("moe", "dense"):
        out[kind] = {op: round(1.0 / sum(w / e[kind][op] for e, w in effs), 6) for op in effs[0][0][kind]}
    about = [
        f"MIXED CAL.effs['{a.mesh}']: per op 1 / sum(w_i / eff_i) over {', '.join(f'{f} (w={w:g})' for f, w in parts)}.",
        "= eff of the weighted mean op time: layers split across the stage kinds and rebalanced (tools/mix_effs.py).",
    ]
    json.dump({"about": about, a.mesh: out}, open(a.out, "w"), indent=1)
    open(a.out, "a").write("\n")


if __name__ == "__main__":
    main()
