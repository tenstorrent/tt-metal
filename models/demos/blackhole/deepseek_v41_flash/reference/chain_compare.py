# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side comparison of device chain variants (tests/test_chain_steps_device.py outputs) for ONE metric per layer.
    python -m ...reference.chain_compare --ref DIR --metric relx|pcc|flip|cos  a.pt b.pt ...   (columns in the order given)
metric: pcc (mean over steps), relx (rel. L2 error excluding the 8 massive channels, mean over steps), flip (step-0 expert-set flip rate vs the
reference routing), cos (mean per-token cosine)."""
import argparse
import os

import torch

from models.demos.blackhole.deepseek_v41_flash.reference.chain_analyze import pcc, rel_excl, set_flip, tok_cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", default="relx")
    ap.add_argument("--layers", default=None)
    ap.add_argument("devs", nargs="+")
    a = ap.parse_args()
    D = [torch.load(p) for p in a.devs]
    names = [os.path.basename(p).replace("h45a_chain_", "").replace(".pt", "") for p in a.devs]
    layers = sorted(set.intersection(*[{k[0] for k in d["x"]} for d in D]))
    if a.layers:
        lo, hi = a.layers.split("-")
        layers = [l for l in layers if int(lo) <= l <= int(hi)]
    print("layer " + " ".join(f"{n[:14]:>14s}" for n in names))
    for L in layers:
        ref = torch.load(os.path.join(a.ref, f"layer_{L}.pt"), mmap=True)
        row = []
        for d in D:
            steps = sorted({k[1] for k in d["x"] if k[0] == L})
            vals = []
            for s in steps:
                xo = d["x"][(L, s)].float()
                ro = ref["dec_out_steps"][s][0].float().reshape(-1, 4, 5120)
                if a.metric == "pcc":
                    vals.append(pcc(xo, ro))
                elif a.metric == "relx":
                    vals.append(rel_excl(xo, ro, 8))
                elif a.metric == "cos":
                    vals.append(float(tok_cos(xo, ro).mean()))
                elif a.metric == "flip" and s == 0 and (L, 0) in d["idx"]:
                    vals.append(float(set_flip(d["idx"][(L, 0)], ref["routing_idx"]).float().mean()))
            row.append(sum(vals) / len(vals) if vals else float("nan"))
        print(f"{L:5d} " + " ".join(f"{v:14.4f}" for v in row))


if __name__ == "__main__":
    main()
