# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP speculative decode (mtp.SpecDecoder) after the pipeline-parallel prefill: the hub-fed spec tokens
(generate_prefilled, feed kinds 1 and 2) against the main model's own greedy tokens (greedy, host-written token
states) for corpus prompts; writes both token lists and the spec stats.

usage: python -m qwen36_resident.tests.mtp_check --corpus ids.pt --prompts 2048,8192 --out mtp.json
"""
import argparse
import faulthandler
import json
import sys

import torch

from qwen36_resident import weights as Q
from qwen36_resident.mtp import SpecDecoder
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--prompts", default="2048,8192")
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    corpus = torch.load(a.corpus)["ids"]
    mesh = open_mesh()
    out = []
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        spec = SpecDecoder(mesh, ck, ck.config["num_hidden_layers"], a.max_len, prefill=True)
        for T in [int(t) for t in a.prompts.split(",")]:
            prompt = corpus[:T].tolist()
            fed, stats = spec.generate_prefilled(prompt, a.new)
            greedy = spec.greedy(prompt, a.new, prefill=True)
            row = dict(T=T, spec=fed[T:], greedy=greedy[T:], identical=fed == greedy, stats=stats)
            out.append(row)
            log(f"T={T}: spec == greedy {row['identical']}, {json.dumps(stats)}")
    finally:
        json.dump(out, open(a.out, "w"), indent=1)
        close_mesh(mesh)


if __name__ == "__main__":
    main()
