# SPDX-License-Identifier: Apache-2.0
"""B31 acceptance harness: tokens/round of the greedy K=3 spec stream on the b24 22-prompt corpus (gsm 8, code 8, story 2, chat 4; B24_PROMPTS=/b24/b24_prompts.pt), batches of up to 4 prompts of one class,
NR rounds each, NL from HY4_SERVE_LAYERS. Prints RES lines per class and ALL (E = mean tokens per round per live user, rounds after an EOS excluded)."""
import json
import os
import time

import torch

from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

K = int(os.environ.get("K", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "32768"))
NR = int(os.environ.get("NR", "50"))
R = 4
EOS = 120025


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    d = torch.load(os.environ.get("B24_PROMPTS", "/b31/../b24/b24_prompts.pt"))
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    try:
        t0 = time.time()
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        eng.warm()
        res(event="warm", sec=round(time.time() - t0, 1))
        allh = [0] * (K + 1)
        for cls in ("gsm", "code", "story", "chat"):
            ps = d[cls]
            ch = [0] * (K + 1)
            for b0 in range(0, len(ps), R):
                batch = ps[b0 : b0 + R]
                eng.drain()
                for r in range(R):
                    eng.release(r)
                first = eng.prefill({r: batch[r] for r in range(len(batch))})
                done = {r: first[r] == EOS for r in range(len(batch))}
                blk = None
                for i in range(NR):
                    out = eng.round([K] * R, False, plugin_block=blk)
                    blk = {r: (out[r]["tok"], out[r]["pos"]) for r in range(R) if eng.live[r]}
                    for r in range(len(batch)):
                        if done[r]:
                            continue
                        ch[out[r]["m"]] += 1
                        if EOS in out[r]["a"][: out[r]["m"] + 1]:
                            done[r] = True
            n = sum(ch)
            res(event="cls", cls=cls, hist=ch, E=round(sum((j + 1) * c for j, c in enumerate(ch)) / max(1, n), 4))
            allh = [a + b for a, b in zip(allh, ch)]
        n = sum(allh)
        res(event="all", hist=allh, E=round(sum((j + 1) * c for j, c in enumerate(allh)) / max(1, n), 4))
        res(event="finished")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
