# SPDX-License-Identifier: Apache-2.0
"""B37: why is spec != cap0 at the 1M pool? Engine level (like spec_eq.py), greedy K=3 spec streams vs cap-0 streams, plus
  - cap0 vs cap0 and spec vs spec repeat determinism,
  - per emitted token the fp32 top-2 logit margin (device-side, appended to the verify greedy; HY4_B37_MARGIN=1),
  - at the first divergent token of a prompt: the final indexer ids (idx8) of every indexer layer for cap0 (slot 0 of round i0) and spec (slot j of round i1), first layer whose ids differ.
Env: MODE=gsm|long, NPROMPTS (8), NR (60), MAXLEN, LONGPT (.pt {ids:[...]}), NDUMP (max prompts dumped per batch, 2), OUT, GSMP, B37_TAG.
Prints RES json lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, rep

K = int(os.environ.get("K", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "32768"))
NR = int(os.environ.get("NR", "60"))
NP = int(os.environ.get("NPROMPTS", "8"))
MODE = os.environ.get("MODE", "gsm")
NDUMP = int(os.environ.get("NDUMP", "2"))
R = 4
V8 = 8


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


class Margin:
    """device-side top-2 of the fp32 verify logits per slot; persistent output buffer (created before the first capture)."""

    def __init__(self, eng):
        self.eng = eng
        S = eng.S
        self.buf = rep(eng.md, torch.zeros(1, 1, S, 3 * V8), ttnn.TILE_LAYOUT, ttnn.float32)
        orig = eng.runner.greedy
        ch = eng.ch

        def wrapped(lg):
            tok = orig(lg)
            lgf = lg if lg.dtype == ttnn.float32 else ttnn.typecast(lg, ttnn.float32)
            if lgf.layout != ttnn.TILE_LAYOUT:
                lgf = ttnn.to_layout(lgf, ttnn.TILE_LAYOUT)
            mx1 = ttnn.max(lgf, dim=-1, keepdim=True)
            e = ttnn.typecast(ttnn.eq(lgf, mx1), ttnn.float32)
            cnt = ttnn.sum(e, dim=-1, keepdim=True)
            mx2 = ttnn.max(ttnn.subtract(lgf, ttnn.multiply(e, 1e9)), dim=-1, keepdim=True)
            g = ch.mc.allgather(ttnn.concat([mx1, mx2, cnt], dim=-1), ch.ccl, axis=1, dim=3)
            ttnn.assign(g, self.buf)
            return tok

        eng.runner.greedy = wrapped

    def read(self, row):
        """-> list over slots of (top1, top2) fp32 of the verify logits of mesh row ``row``."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.buf)[row * V8]).float().reshape(-1, 3 * V8)
        out = []
        for j in range(t.shape[0]):
            c = []
            for col in range(V8):
                m1, m2, n1 = float(t[j, 3 * col]), float(t[j, 3 * col + 1]), float(t[j, 3 * col + 2])
                c += [m1] * (2 if n1 > 1.5 else 1) + [m2]
            c.sort(reverse=True)
            out.append((c[0], c[1]))
        return out


def read_idx(eng, row):
    """final idx8 of every indexer layer for mesh row ``row``: {layer: uint32 [8 dev, NC, S, K]}"""
    out = {}
    S = eng.S
    for li, ly in enumerate(eng.ch.layers[:78]):
        a = ly.attn
        if getattr(a, "has_indexer", False) and getattr(a, "idx8_buf", None) is not None:
            ts = ttnn.get_device_tensors(a.idx8_buf)
            Kk = int(a.idx8_buf.shape[-1])
            out[li] = torch.stack(
                [ttnn.to_torch(ts[row * V8 + c]).reshape(8, S, Kk).to(torch.int64).to(torch.int32) for c in range(V8)]
            )
    return out


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    outdir = os.environ.get("OUT", "/v/out")
    tag = os.environ.get("B37_TAG", "b37")
    if MODE == "gsm":
        g = torch.load(os.environ.get("GSMP", "/q/gsm50.pt"))
        prompts = [x["ids"] for x in g[:NP]]
    else:
        g = torch.load(os.environ["LONGPT"])
        prompts = g["ids"]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    try:
        t0 = time.time()
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        res(event="built", sec=round(time.time() - t0, 1), pages=eng.pages)
        mg = Margin(eng) if os.environ.get("HY4_B37_MARGIN", "1") == "1" else None
        eng.warm()
        eng.pipe = False  # reads between rounds must not race a speculative round
        eng.drain()
        res(event="warm", sec=round(time.time() - t0, 1), margin=mg is not None)

        snap = {}

        def stream(batch, cap, nr, dump=None, reuse=False):
            """batch {row: ids}; -> (st {row: tokens}, mgs {row: [(t1,t2)|None]}, where {row: [(round, slot)|None]}, dumps {(round,row): idx})"""
            rows = sorted(batch)
            if reuse and snap:
                eng.drain()
                for r in range(R):
                    eng.release(r)
                for r in rows:
                    eng.pend[r], eng.pos[r], eng.drafts[r], eng.live[r] = snap["s"][r]
                eng.dirty, eng.q_valid, eng.params_dirty = True, False, True
                first = snap["first"]
            else:
                eng.drain()
                for r in range(R):
                    eng.release(r)
                first = eng.prefill(batch)
                snap["first"] = first
                snap["s"] = {r: (eng.pend[r], eng.pos[r], list(eng.drafts[r]), eng.live[r]) for r in rows}
            st = {r: [first[r]] for r in rows}
            ms = {r: [None] for r in rows}
            wh = {r: [None] for r in rows}
            dumps = {}
            blk = None
            for i in range(nr):
                out = eng.round([cap] * R, False, plugin_block=blk)
                blk = {r: (out[r]["tok"], out[r]["pos"]) for r in range(R) if eng.live[r]}
                for r in rows:
                    mm = mg.read(r) if mg else None
                    em = out[r]["a"][: out[r]["m"] + 1]
                    for j, t in enumerate(em):
                        st[r].append(t)
                        ms[r].append(mm[j] if mm else None)
                        wh[r].append((i, j))
                    if dump and (i, r) in dump:
                        dumps[(i, r)] = read_idx(eng, r)
            return st, ms, wh, dumps

        longmode = MODE == "long"
        batches = [(0, prompts[:1])] if longmode else [(b0, prompts[b0 : b0 + R]) for b0 in range(0, NP, R)]
        allm, divm = [], []
        for b0, pb in batches:
            batch = {r: pb[r] for r in range(len(pb))}
            t1 = time.time()
            c0a, m0a, w0a, _ = stream(batch, 0, NR)
            res(event="stream", which="cap0a", sec=round(time.time() - t1, 1), prefill_len=[len(x) for x in pb])
            c0b, _, _, _ = stream(batch, 0, NR, reuse=longmode)
            s3a, m3a, w3a, _ = stream(batch, K, NR, reuse=longmode)
            s3b, _, _, _ = stream(batch, K, NR, reuse=longmode)
            tgt0, tgt3 = {}, {}
            for r in range(len(pb)):
                n = min(len(c0a[r]), len(s3a[r]))
                fd = next((i for i in range(n) if c0a[r][i] != s3a[r][i]), -1)
                if os.environ.get("B37_TEST_DUMP") == "1" and r == 0 and fd < 0:
                    fd = 5  # pipeline test only
                fdc = next((i for i in range(min(len(c0a[r]), len(c0b[r]))) if c0a[r][i] != c0b[r][i]), -1)
                fds = next((i for i in range(min(len(s3a[r]), len(s3b[r]))) if s3a[r][i] != s3b[r][i]), -1)
                mc = m0a[r][fd] if fd >= 0 else None
                ms_ = m3a[r][fd] if fd >= 0 else None
                res(
                    event="eq",
                    prompt=b0 + r,
                    n=n,
                    spec_eq_cap0=fd < 0,
                    first_diff=fd,
                    cap0_repeat_fd=fdc,
                    spec_repeat_fd=fds,
                    cap0_margin_at_fd=mc,
                    spec_margin_at_fd=ms_,
                    where_cap0=w0a[r][fd] if fd >= 0 else None,
                    where_spec=w3a[r][fd] if fd >= 0 else None,
                    tok_cap0=c0a[r][fd] if fd >= 0 else None,
                    tok_spec=s3a[r][fd] if fd >= 0 else None,
                )
                lim = fd if fd >= 0 else n
                for i in range(1, lim):
                    allm.append((m0a[r][i][0] - m0a[r][i][1]) if m0a[r][i] else None)
                if fd >= 0:
                    divm.append((m0a[r][fd][0] - m0a[r][fd][1], m3a[r][fd][0] - m3a[r][fd][1]))
                    if len(tgt0) < NDUMP:
                        tgt0[(w0a[r][fd][0], r)] = (r, fd)
                        tgt3[(w3a[r][fd][0], r)] = (r, fd, w3a[r][fd][1])
            if tgt0 and os.environ.get("B37_DUMP", "1") == "1":
                _, _, _, d0 = stream(batch, 0, NR, dump=set(tgt0), reuse=longmode)
                _, _, _, d3 = stream(batch, K, NR, dump=set(tgt3), reuse=longmode)
                for (i0, r), (_, fd) in tgt0.items():
                    (i3, _r), (_, _, j3) = next((k, v) for k, v in tgt3.items() if k[1] == r)
                    a, b = d0[(i0, r)], d3[(i3, r)]
                    per = []
                    first_set = first_raw = None
                    for li in sorted(a):
                        x, y = a[li][:, :, 0, :], b[li][:, :, j3, :]  # [8 dev, NC, K]
                        raw = bool((x == y).all())
                        sx, _ = torch.sort(x.reshape(8, -1), dim=-1)
                        sy, _ = torch.sort(y.reshape(8, -1), dim=-1)
                        sete = bool((sx == sy).all())
                        nd = int((x != y).sum())
                        per.append((li, raw, sete, nd))
                        if not raw and first_raw is None:
                            first_raw = li
                        if not sete and first_set is None:
                            first_set = li
                    res(
                        event="idx_cmp",
                        prompt=b0 + r,
                        fd=fd,
                        cap0_round=i0,
                        spec_round=i3,
                        spec_slot=j3,
                        first_raw_diff_layer=first_raw,
                        first_set_diff_layer=first_set,
                        n_layers=len(per),
                        layers_raw_diff=[p[0] for p in per if not p[1]],
                        layers_set_diff=[p[0] for p in per if not p[2]],
                        ndiff=[p[3] for p in per if p[3]],
                    )
                    torch.save(
                        dict(c0=a, spec=b, slot=j3, fd=fd), f"{outdir}/{tag}_idx_p{b0 + r}.pt"
                    )  # for offline analysis
        am = [x for x in allm if x is not None]
        if am:
            t = torch.tensor(am)
            res(
                event="margin_stats",
                n=len(am),
                frac_lt=[[x, float((t < x).float().mean())] for x in (0.001, 0.01, 0.03, 0.1, 0.3, 1.0)],
                q50=float(t.median()),
            )
        res(event="div_margins_cap0_spec", v=divm)
        res(event="finished")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
