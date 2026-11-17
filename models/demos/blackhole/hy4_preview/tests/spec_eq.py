# SPDX-License-Identifier: Apache-2.0
"""verify-round track P1 gate at NL=78 (engine level, no HTTP): greedy K=3 spec stream vs cap-0 stream (same verify-shape path) on REAL prompts (gsm50 chat prompts, ~750 tokens), 2 batches of 4 = 8 prompts.
Env: NPROMPTS (8), NR (rounds, 60), MAXLEN, K, OUT, GSMP. Prints RES lines; eq = exact stream equality over the common length."""
import json
import os
import time

import torch

from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

K = int(os.environ.get("K", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "32768"))
NR = int(os.environ.get("NR", "60"))
NP = int(os.environ.get("NPROMPTS", "8"))
R = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    g = torch.load(os.environ.get("GSMP", "/q/gsm50.pt"))
    prompts = [x["ids"] for x in g[:NP]]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    try:
        t0 = time.time()
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        res(event="built", sec=round(time.time() - t0, 1), pages=eng.pages)
        eng.warm()
        res(event="warm", sec=round(time.time() - t0, 1))

        def run(batch, cap):
            eng.drain()
            for r in range(R):
                eng.release(r)
            first = eng.prefill({r: batch[r] for r in range(len(batch))})
            st = {r: [first[r]] for r in range(len(batch))}
            blk = None
            hist = [0] * (K + 1)
            for i in range(NR):
                out = eng.round([cap] * R, False, plugin_block=blk)
                blk = {r: (out[r]["tok"], out[r]["pos"]) for r in range(R) if eng.live[r]}
                for r in range(len(batch)):
                    em = out[r]["a"][: out[r]["m"] + 1]
                    st[r] += em
                    hist[out[r]["m"]] += 1
            return st, hist

        allok = True
        for b0 in range(0, NP, R):
            batch = prompts[b0 : b0 + R]
            s0, h0 = run(batch, 0)
            s3, h3 = run(batch, K)
            s3b, _ = run(batch, K)
            for r in range(len(batch)):
                n = min(len(s0[r]), len(s3[r]))
                fd = next((i for i in range(n) if s0[r][i] != s3[r][i]), -1)
                eq = fd < 0
                allok &= eq
                res(
                    event="eq",
                    prompt=b0 + r,
                    n=n,
                    spec_eq_cap0=eq,
                    first_diff=fd,
                    spec_repeat_eq=s3[r][:n] == s3b[r][:n],
                    n0=len(s0[r]),
                    n3=len(s3[r]),
                    hist3=h3,
                )
        # timing: wall ms per sync round (device round + ~3 ms host), 4 live rows, 30 rounds after 5 warm
        batch = prompts[:R]
        eng.drain()
        for r in range(R):
            eng.release(r)
        first = eng.prefill({r: batch[r] for r in range(R)})
        blk = None
        for i in range(35):
            if i == 5:
                t0 = time.perf_counter()
            out = eng.round([K] * R, False, plugin_block=blk)
            blk = {r: (out[r]["tok"], out[r]["pos"]) for r in range(R) if eng.live[r]}
        res(event="round_ms", ms=round((time.perf_counter() - t0) / 30 * 1e3, 2))
        if os.environ.get("HY4_B36_DBG", "0") == "1":
            from models.demos.blackhole.hy4_preview.tt import hy4_pool as _HP

            res(event="sel3x_dbg", stats=_HP.dbg_dump(md))
        res(event="finished", all_equal=allok)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
