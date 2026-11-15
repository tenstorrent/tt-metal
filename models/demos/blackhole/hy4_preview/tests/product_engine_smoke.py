# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""product-integ engine smoke (WHOLE MESH, no vLLM): Hy4Engine build -> warm (synthetic prefill + MTP + traces) -> real prefill of 4 prompts -> rounds.
Env: NL (layers, default 78; small for plumbing checks), K (0 = plain device loop, 1..3 spec), MAXLEN (8192), NROUNDS (12), LENS ("100,700,2600,5000"), SCEN (comma list: basic,masked,sampled,cap0), OUT.
Scenarios: basic = prefill 4 prompts, NROUNDS greedy rounds (streams recorded); masked = same, but prefill row 2 anew after 3 rounds while rows 0,1,3 are live: their streams must equal the basic ones;
sampled = rounds on the sampled trace (T=0.8) for NROUNDS; cap0 = the basic scenario with caps 0 (every round emits one token): lossless equality vs basic.
Prints RES json lines. Failure = exception / mismatch."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

K = int(os.environ.get("K", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
NR = int(os.environ.get("NROUNDS", "12"))
LENS = [int(x) for x in os.environ.get("LENS", "100,700,2600,5000").split(",")]
SCEN = os.environ.get("SCEN", "basic,masked").split(",")
OUT = os.environ.get("OUT", "/out")
R = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def prompts_for(seed, lens):
    g = torch.Generator().manual_seed(seed)
    return {r: torch.randint(1000, 100000, (lens[r],), generator=g).tolist() for r in range(len(lens))}


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    try:
        t0 = time.time()
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        res(event="built", sec=round(time.time() - t0, 1), pages=eng.pages, K=K)
        t0 = time.time()
        eng.warm()
        res(event="warm", sec=round(time.time() - t0, 1))
        P = prompts_for(1, LENS)
        streams = {}

        def run(tag, caps_val=None, sampled=False, mid_prefill=None):
            for r in range(R):
                eng.release(r)
            first = eng.prefill(P)
            st = {r: [first[r]] for r in range(R)}
            hist = [0] * (eng.K + 1)
            for n in range(NR):
                if mid_prefill is not None and n == 3:
                    row, toks = mid_prefill
                    f2 = eng.prefill({row: toks})
                    res(event="mid_prefill", row=row, first=f2[row])
                    st[row] = [f2[row]]
                caps = [eng.K if caps_val is None else caps_val] * R
                t1 = time.perf_counter()
                out = eng.round(
                    caps, sampled, temps=[0.8] * R if sampled else None, seeds=[11, 22, 33, 44] if sampled else None
                )
                dt = time.perf_counter() - t1
                for r in range(R):
                    if mid_prefill is not None and r == mid_prefill[0] and n < 3:
                        continue
                    em = out[r]["a"][: out[r]["m"] + 1]
                    st[r] += em
                    hist[out[r]["m"]] += 1
                res(event="round", tag=tag, n=n, ms=round(dt * 1e3, 1), m=[out[r]["m"] for r in range(R)])
            res(event="done", tag=tag, m_hist=hist, streams={r: st[r][:24] for r in range(R)})
            return st

        if K == 0:
            # plain device loop (no MTP): prefill via the engine, then load_inputs / submit / read; the masked scenario prefills row 2 while the others decode
            dl = eng.dl

            def plain(tag, mid=None):
                first = eng.prefill(P)
                tok = [first[r] for r in range(R)]
                pos = [len(P[r]) for r in range(R)]
                st = {r: [first[r]] for r in range(R)}
                dl.load_inputs(tok, pos, [1] * R)
                v = dl.set_sampling([0.0] * R, [1] * R, [1.0] * R, [1, 2, 3, 4], active=[1] * R)
                for n in range(NR):
                    if mid is not None and n == 3:
                        row, toks = mid
                        f2 = eng.prefill({row: toks})
                        ttnn.synchronize_device(md)
                        # the plugin's reload contract after a prefill: tokens / positions of ALL rows re-uploaded (the live rows continue where they are)
                        tok = [st[r][-1] for r in range(R)]
                        pos = [len(P[r]) + len(st[r]) - 1 for r in range(R)]
                        st[row] = [f2[row]]
                        tok[row], pos[row] = f2[row], len(toks)
                        dl.load_inputs(tok, pos, [1] * R)
                    t1 = time.perf_counter()
                    dl.submit(v)
                    h = dl.read_async()
                    out = dl.tokens(h)
                    dt = time.perf_counter() - t1
                    for r in range(R):
                        if mid is not None and r == mid[0] and n < 3:
                            continue
                        st[r].append(out[r])
                    res(event="plain_step", tag=tag, n=n, ms=round(dt * 1e3, 1))
                res(event="done", tag=tag, streams={r: st[r][:24] for r in range(R)})
                return st

            b = plain("plain")
            sm = plain("plain_masked", mid=(2, prompts_for(7, [900] * R)[2]))
            for r in (0, 1, 3):
                n = min(len(b[r]), len(sm[r]))
                res(event="masked_vs_basic", row=r, n=n, equal=b[r][:n] == sm[r][:n])
            res(event="finished")
            return
        if "basic" in SCEN:
            streams["basic"] = run("basic")
        if "cap0" in SCEN:
            streams["cap0"] = run("cap0", caps_val=0)
            b = streams.get("basic")
            if b:
                for r in range(R):
                    n = min(len(b[r]), len(streams["cap0"][r]))
                    res(event="cap0_vs_basic", row=r, n=n, equal=b[r][:n] == streams["cap0"][r][:n])
        if "masked" in SCEN and streams.get("basic"):
            other = prompts_for(7, [900] * R)[2]
            sm = run("masked", mid_prefill=(2, other))
            b = streams["basic"]
            for r in (0, 1, 3):
                n = min(len(b[r]), len(sm[r]))
                res(event="masked_vs_basic", row=r, n=n, equal=b[r][:n] == sm[r][:n])
        if "sampled" in SCEN and eng.sampled:
            run("sampled", sampled=True)
        res(event="finished")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
