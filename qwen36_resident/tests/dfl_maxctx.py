# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ResidentDFlash at a long max_pos: the DRAM it takes per chip (after the build, and after each prompt) and, per
corpus prompt length, the prefill path's TTFT, the spec decode's tok/s and acceptance, and whether its tokens equal the
verify model's greedy after the same prefill (else the first difference and the greedy logit margin there).

usage: python -m qwen36_resident.tests.dfl_maxctx [--max-pos 262144] [--T 32000,65536,131072,250000] [--new 128]
"""
import argparse
import faulthandler
import json
import sys
import time

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.tests.long_ctx import close_mesh, dram, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--max-pos", type=int, default=262144)
ap.add_argument("--T", default="32000,65536,131072,250000")
ap.add_argument("--new", type=int, default=128)
ap.add_argument("--out", default="")
ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
a = ap.parse_args()
faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)
mesh = open_mesh()
rows = []
try:
    t0 = time.time()
    spec = ResidentDFlash(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos=a.max_pos, prefill=True)
    used, total = dram(mesh)
    log(
        f"max_pos {a.max_pos} (built as {spec.max_pos}): built in {time.time() - t0:.0f} s, DRAM {used:.2f} / {total:.2f} GB "
        f"per chip"
    )
    corpus = torch.load(a.corpus)["ids"]
    for T in [int(x) for x in a.T.split(",")]:
        ids = corpus[:T].tolist()
        assert len(ids) == T, f"corpus has {len(ids)} tokens"
        time.sleep(2)
        gen, st = spec.generate(ids, a.new)
        used, _ = dram(mesh)
        # the lossless check on the same prompt path: the verify model's greedy after the same prefill
        ref, gaps = spec.greedy(ids, a.new, other=gen, prefill=True)
        first = next((k for k in range(a.new) if gen[k] != ref[k]), None)
        st["greedy_first_diff"] = first
        st["greedy_margin"] = gaps[first] if first is not None else None
        st = {k: v for k, v in st.items() if k != "accepts"}
        rows.append(dict(T=T, **st, dram_gb=used))
        log(
            f"T={T}: TTFT {st['first_token_s']:.2f} s, {st['tok_s']:.1f} tok/s, {st['mean_accept']:.2f} accepted/iter, "
            f"{st['ms_per_iter']:.1f} ms/iter; DRAM {used:.2f} GB; == greedy after the same prefill: {first is None}"
            + ("" if first is None else f" (first diff {first}, margin {st['greedy_margin']:.3f})")
        )
finally:
    close_mesh(mesh)
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
