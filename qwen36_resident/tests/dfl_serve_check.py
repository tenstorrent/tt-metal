# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ResidentDFlash's serving API (start / next_block / stop, what the vLLM adapter drives) against its generate():
per prompt the first token (start's logits' argmax) and the blocks of n tokens must reproduce generate()'s tokens
exactly; also a request stopped mid-way and the next one started (no state carried across requests), and the
position limit (next_block returns fewer tokens once a verify step would pass it).

usage: python -m qwen36_resident.tests.dfl_serve_check [--n 8] [--new 128] [--long 3000]
"""
import argparse
import faulthandler
import sys
import time

import torch
from transformers import AutoTokenizer

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.serve.prompts import PROMPTS, chat_ids
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh


def blocks(spec, ids, total, n, limit):
    lg = spec.start(ids)
    assert int(lg.argmax()) == spec.first_token, (int(lg.argmax()), spec.first_token)
    out, sizes = [spec.first_token], []
    while len(out) < total:
        b = spec.next_block(n, limit)
        sizes.append(len(b))
        if not b:
            break
        out += b
    return out[:total], sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--long", default="3000")
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    longs = [int(x) for x in a.long.split(",") if x]
    mesh = open_mesh()
    fails = 0
    try:
        t0 = time.time()
        spec = ResidentDFlash(mesh, ck, max_pos=max(longs + [2048]) + a.new + 256, prefill=True)
        log(f"built in {time.time() - t0:.0f} s")
        corpus = torch.load(a.corpus)["ids"]
        cases = [(p["name"], chat_ids(tok, p)) for p in PROMPTS]
        cases += [(f"corpus{T}", corpus[:T].tolist()) for T in longs]
        for name, ids in cases:
            ref, _ = spec.generate(ids, a.new)
            got, sizes = blocks(spec, ids, a.new, a.n, spec.max_pos)
            same = got == ref
            fails += not same
            log(f"{name}: P={len(ids)} start + next_block({a.n}) == generate: {same} (blocks {sizes[:6]}...)")
        # a request stopped after one block, then the first case again: the same tokens as before
        name, ids = cases[0]
        spec.start(cases[-1][1])
        spec.next_block(a.n, spec.max_pos)
        spec.stop()
        got, _ = blocks(spec, ids, a.new, a.n, spec.max_pos)
        ref, _ = spec.generate(ids, a.new)
        fails += got != ref
        log(f"after a stopped request: {name} == generate: {got == ref}")
        # the position limit: blocks stop short once a verify step would pass it
        lim = len(ids) + 40
        got, sizes = blocks(spec, ids, a.new, a.n, lim)
        ok = len(got) <= lim - len(ids) + spec.R and got == ref[: len(got)]
        fails += not ok
        log(f"limit {lim}: {len(got)} tokens, a prefix of generate's: {ok} (blocks {sizes})")
    finally:
        close_mesh(mesh)
    log(f"serve check: {'PASS' if not fails else f'FAIL ({fails})'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
