# SPDX-License-Identifier: Apache-2.0
"""host-overhead track harness (no HTTP server): Hy4Engine build -> warm -> scenarios comparing the synchronous round loop (HY4_SPEC_PIPE=0 behaviour) with the pipelined one (speculative enqueue of round N+1).
Emulated plugin work between rounds: HW_MS (sleep). Env: K, MAXLEN, NROUNDS, HW_MS (comma list), SHORT_LEN, LONG_LEN (0 = skip), OUT.
Scenarios: lossless (cap0 vs spec sync vs spec pipe streams, mid-run prefill + release), timing (ms/round, tok/s/user) at short and long contexts. Prints RES json lines."""
import json
import os
import time

import torch

from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

K = int(os.environ.get("K", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "1048576"))
NR = int(os.environ.get("NROUNDS", "40"))
HWS = [float(x) for x in os.environ.get("HW_MS", "0,22").split(",")]
SHORT = int(os.environ.get("SHORT_LEN", "300"))
LONG = int(os.environ.get("LONG_LEN", "0"))
R = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def prompts_for(seed, n):
    g = torch.Generator().manual_seed(seed)
    return {r: torch.randint(1000, 100000, (n + 37 * r,), generator=g).tolist() for r in range(R)}


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    try:
        t0 = time.time()
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        res(event="built", sec=round(time.time() - t0, 1), pages=eng.pages)
        t0 = time.time()
        eng.warm()
        res(event="warm", sec=round(time.time() - t0, 1))
        eng.tl = []

        def start(P):
            eng.drain()
            for r in range(R):
                eng.release(r)
            first = eng.prefill(P)
            return {r: [first[r]] for r in range(R)}

        def loop(st, n, pipe, hw, cap=None, live=None, mid=None, tag=""):
            """n rounds from the current state; returns (wall ms/round, tokens/round/user, hist)."""
            eng.drain()
            eng.pipe = pipe
            caps = [K if cap is None else cap] * R
            blk = None
            hist = [0] * (K + 1)
            ntok = 0
            eng.tl.clear()
            t0 = time.perf_counter()
            for i in range(n):
                if mid is not None and i == 3:
                    row, toks = mid
                    f = eng.prefill({row: toks})
                    st[row] = [f[row]]
                    blk = None
                out = eng.round(list(caps), False, plugin_block=blk)
                blk = {r: (out[r]["tok"], out[r]["pos"]) for r in range(R) if eng.live[r]}
                for r in range(R):
                    if not eng.live[r] or (mid is not None and r == mid[0] and i < 3 and False):
                        continue
                    em = out[r]["a"][: out[r]["m"] + 1]
                    st[r] += em
                    hist[out[r]["m"]] += 1
                    ntok += len(em)
                if hw:
                    time.sleep(hw / 1e3)
            wall = time.perf_counter() - t0
            eng.drain()
            nl = max(sum(eng.live), 1)
            tl = eng.tl
            agg = {
                k: round(sum(x[k] for x in tl) / max(len(tl), 1) * 1e3, 2)
                for k in ("prep", "upload_submit", "read_enq_spec_submit", "wait_read")
            }
            r_ = dict(
                tag=tag,
                pipe=pipe,
                hw_ms=hw,
                rounds=n,
                ms_per_round=round(wall / n * 1e3, 2),
                tok_per_round_user=round(ntok / (nl * n), 3),
                tok_s_user=round(ntok / (nl * wall), 2),
                hist=hist,
                hits=eng.pipe_hits,
                discards=eng.pipe_discards,
                tl=agg,
            )
            res(event="loop", **r_)
            return r_

        # ---------------- lossless
        P = prompts_for(1, SHORT)
        s_c0 = start(P)
        loop(s_c0, NR, False, 0, cap=0, tag="cap0")
        s_sync = start(P)
        loop(s_sync, NR, False, 0, tag="spec_sync")
        for hw in HWS:
            s_p = start(P)
            loop(s_p, NR, True, hw, tag="spec_pipe")
            for r in range(R):
                n = min(len(s_p[r]), len(s_sync[r]), len(s_c0[r]))
                res(
                    event="eq",
                    hw=hw,
                    row=r,
                    n=n,
                    pipe_eq_sync=s_p[r][:n] == s_sync[r][:n],
                    pipe_eq_cap0=s_p[r][:n] == s_c0[r][:n],
                )
        # mid-run admission (prefill of row 2 at round 3) + sync vs pipe
        other = prompts_for(7, 500)[2]
        a = start(P)
        loop(a, NR, False, 0, mid=(2, other), tag="mid_sync")
        b = start(P)
        loop(b, NR, True, HWS[-1], mid=(2, other), tag="mid_pipe")
        for r in range(R):
            n = min(len(a[r]), len(b[r]))
            res(event="eq_mid", row=r, n=n, equal=a[r][:n] == b[r][:n])

        # two live rows only
        def two(pipe, hw, tag):
            c = start(P)
            eng.release(1)
            eng.release(3)
            loop(c, NR, pipe, hw, tag=tag)
            return c

        c1 = two(False, 0, "two_sync")
        c2 = two(False, 0, "two_sync_repeat")
        d = two(True, HWS[-1], "two_pipe")
        for r in (0, 2):

            def first_diff(x, y):
                n = min(len(x), len(y))
                return next((i for i in range(n) if x[i] != y[i]), -1)

            res(
                event="eq_two",
                row=r,
                sync_vs_repeat=first_diff(c1[r], c2[r]),
                sync_vs_pipe=first_diff(c1[r], d[r]),
                n=min(len(c1[r]), len(d[r])),
            )
        # ---------------- timing at short context
        st = start(P)
        for hw in HWS:
            loop(st, NR, False, hw, tag="short_sync")
            loop(st, NR, True, hw, tag="short_pipe")
        if LONG:
            P2 = prompts_for(3, LONG)
            st = start(P2)
            for hw in HWS:
                loop(st, NR, False, hw, tag="long_sync")
                loop(st, NR, True, hw, tag="long_pipe")
        res(event="finished")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
