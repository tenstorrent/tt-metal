# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill2 single-chip pipeline test: K1b (gather + gate/up + score) -> K2b x6 (down + NoC combine to token owners) vs torch.
Env: TT=2048 NC=110 NREP=2 NCHK=48 SKEW=1 SCALE=1 STAGE=full|k1 TRACE=1 OUT=json"""
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import bank_tiles_tensor
from models.demos.blackhole.hy4_preview.tt.hy4_tp32_prefill import HID, LIMIT, PlanTensors, Tp32PrefillExperts
from models.demos.blackhole.hy4_preview.tt.hy4_tp32pf_plan import build_plan

TT = int(os.environ.get("TT", "2048"))
NC = int(os.environ.get("NC", "110"))
NE = 256
NREP = int(os.environ.get("NREP", "2"))
NCHK = int(os.environ.get("NCHK", "48"))
SKEW = float(os.environ.get("SKEW", "1.0"))
SCALE = int(os.environ.get("SCALE", "1"))
STAGE = os.environ.get("STAGE", "full")
TRACE = int(os.environ.get("TRACE", "1"))


def traced_us(md, fn, n=6, rep_=2):
    def body():
        for _ in range(rep_):
            fn()

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=16384, trace_region_size=100_000_000)
    torch.manual_seed(0)
    w01_tiles = (torch.randn(1, 8, NE * 96, 32, 32) * 0.05).bfloat16()
    w2_tiles = (torch.randn(1, 8, 6 * NE * 8, 32, 32) * 0.05).bfloat16()
    w01 = bank_tiles_tensor(md, w01_tiles)
    w2 = bank_tiles_tensor(md, w2_tiles)
    # read back the bfp4-rounded tiles for the reference
    w01r = (
        ttnn.to_torch(w01)
        .reshape(1, NE * 96, 32, 1, 8, 32)
        .permute(0, 3, 4, 1, 2, 5)
        .reshape(8, NE * 96, 32, 32)
        .float()
    )
    w2r = (
        ttnn.to_torch(w2)
        .reshape(1, 6 * NE * 8, 32, 1, 8, 32)
        .permute(0, 3, 4, 1, 2, 5)
        .reshape(8, 6 * NE * 8, 32, 32)
        .float()
    )

    def wg_wu(e):
        w = w01r[:, e * 96 : (e + 1) * 96].reshape(8, 24, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(6144, 128)
        return w[:, :64], w[:, 64:]

    def w2e(e):
        w = torch.zeros(64, 6144)
        for b in range(8):
            for s_ in range(6):
                t = w2r[b, (s_ * NE + e) * 8 : (s_ * NE + e) * 8 + 8].reshape(2, 4, 32, 32)
                cc = (24 * b + 4 * s_) * 32
                w[:, cc : cc + 128] = t.permute(0, 2, 1, 3).reshape(64, 128)
        return w

    ex = Tp32PrefillExperts(md, w01, w2, nc=NC, scale=SCALE)
    for rep in range(NREP):
        pop = torch.exp(torch.randn(NE) * 0.7 * SKEW)
        ids = torch.stack([torch.multinomial(pop, 8, replacement=False) for _ in range(TT)])
        sc = torch.rand(TT, 8) + 0.1
        sc = sc / sc.sum(1, keepdim=True)
        pl = build_plan(ids, sc, nc=NC, xl1=bool(int(os.environ.get("TP32PF_XL1", "1"))))
        p("rep", rep, "mtiles", pl.mt_used, "maxload", pl.max_load, "maxexp", max(pl.mts))
        X = torch.randn(TT, HID).bfloat16()
        x = ttnn.from_torch(
            X.reshape(1, 1, TT, HID),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt = PlanTensors(md, pl)
        act, out = ex.alloc(TT)
        if STAGE == "k2":
            # K2 alone: random act rows (already scaled), reference = sum over the 8 slots of act_row @ W2 slice
            cap_ = pl.dest.shape[0] * 2
            Arand = (torch.randn(cap_ * 32, 64) * 0.5).bfloat16()
            at = Arand.reshape(cap_, 32, 2, 32).permute(0, 2, 1, 3).reshape(1, 1, cap_ * 2 * 32, 32)
            act = ttnn.from_torch(
                at, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            flat_ = ids.reshape(-1)
            order_ = torch.argsort(flat_ * (TT * 8) + torch.arange(TT * 8), stable=True)
            cnt_ = torch.bincount(flat_, minlength=NE)
            rp = {}
            pos_ = 0
            for e in range(NE):
                for r_, pp in enumerate(order_[pos_ : pos_ + int(cnt_[e])].tolist()):
                    rp[pp] = pl.start[e] * 32 + r_
                pos_ += int(cnt_[e])
            if int(os.environ.get("TP32PF_DBG", "0")) == 1:
                outd = ttnn.from_torch(
                    torch.zeros(1, 1, cap_ * 32, 1024).bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                S = 0
                ex.k2(S, pt, act, outd, TT)
                ttnn.synchronize_device(md)
                Od = ttnn.to_torch(outd).reshape(cap_ * 32, 1024).float()
                # row r of expert e: act row @ W2 slice (columns of pass S: (24 b + 4 S + n) tiles, vc = 4 b + n)
                bad = []
                for t in torch.randperm(TT)[:24].tolist():
                    for j in range(8):
                        e = int(ids[t, j])
                        w = w2e(e)
                        cols = torch.cat(
                            [torch.arange((24 * b + 4 * S) * 32, (24 * b + 4 * S) * 32 + 128) for b in range(8)]
                        )
                        ref = Arand[rp[t * 8 + j]].float() @ w[:, cols]
                        bad.append(pcc(ref, Od[rp[t * 8 + j]]))
                p("DBG1 rows pcc min", min(bad), "median", sorted(bad)[len(bad) // 2], "n", len(bad))
                break
            if int(os.environ.get("TP32PF_DBG", "0")) == 2:
                outd = ttnn.from_torch(
                    torch.zeros(1, 1, TT * 8, 1024).bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                S = 0
                ex.k2(S, pt, act, outd, TT)
                ttnn.synchronize_device(md)
                Od = ttnn.to_torch(outd).reshape(TT * 8, 1024).float()
                res = []
                for t in torch.randperm(TT)[:48].tolist():
                    for j in range(8):
                        e = int(ids[t, j])
                        w = w2e(e)
                        cols = torch.cat(
                            [torch.arange((24 * b + 4 * S) * 32, (24 * b + 4 * S) * 32 + 128) for b in range(8)]
                        )
                        ref = Arand[rp[t * 8 + j]].float() @ w[:, cols]
                        res.append((pcc(ref, Od[t * 8 + j]), t, j))
                res.sort()
                p(
                    "DBG2 slot rows pcc worst",
                    res[:6],
                    "median",
                    res[len(res) // 2][0],
                    "n",
                    len(res),
                    "nonfinite rows",
                    int((~torch.isfinite(Od).all(1)).sum()),
                )
                break
            if int(os.environ.get("TP32PF_DBG", "0")) == 3:
                outd = ttnn.from_torch(
                    torch.zeros(1, 1, TT, 1024).bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                S = 0
                ex.k2(S, pt, act, outd, TT)
                ttnn.synchronize_device(md)
                Od = ttnn.to_torch(outd).reshape(TT, 1024).float()
                res = []
                for t in torch.randperm(TT)[:48].tolist():
                    acc = torch.zeros(1024)
                    for j in range(8):
                        e = int(ids[t, j])
                        w = w2e(e)
                        cols = torch.cat(
                            [torch.arange((24 * b + 4 * S) * 32, (24 * b + 4 * S) * 32 + 128) for b in range(8)]
                        )
                        acc += Arand[rp[t * 8 + j]].float() @ w[:, cols]
                    res.append((pcc(acc, Od[t]), t))
                res.sort()
                p(
                    "DBG3 summed rows pcc worst",
                    res[:6],
                    "median",
                    res[len(res) // 2][0],
                    "nonfinite rows",
                    int((~torch.isfinite(Od).all(1)).sum()),
                )
                break
            for S in range(int(os.environ.get("NPASS", "6"))):
                ex.k2(S, pt, act, out, TT)
            ttnn.synchronize_device(md)
            O = ttnn.to_torch(out).reshape(TT, HID).float()
            toks = torch.randperm(TT, generator=torch.Generator().manual_seed(rep))[:NCHK].tolist()
            refs, gots = [], []
            for t in toks:
                acc = torch.zeros(HID)
                for j in range(8):
                    acc += Arand[rp[t * 8 + j]].float() @ w2e(int(ids[t, j]))
                refs.append(acc)
                gots.append(O[t])
            Rr, Gg = torch.stack(refs), torch.stack(gots)
            npass = int(os.environ.get("NPASS", "6"))
            # compare only the computed passes (columns 24b+4S+n tiles)
            colmask = torch.zeros(HID, dtype=torch.bool)
            for S in range(npass):
                for b in range(8):
                    c0 = (24 * b + 4 * S) * 32
                    colmask[c0 : c0 + 128] = True
            p(
                "rep",
                rep,
                "K2 only partial pcc",
                round(pcc(Rr[:, colmask], Gg[:, colmask]), 5),
                "per-token",
                [round(pcc(a[colmask], b[colmask]), 4) for a, b in list(zip(refs, gots))[:6]],
                "max|ref|",
                float(Rr[:, colmask].abs().max()),
                "max|got|",
                float(Gg[:, colmask].abs().max()),
            )
            if TRACE:
                us2 = traced_us(md, lambda: [ex.k2(S, pt, act, out, TT) for S in range(6)])
                p("rep", rep, "K2 x6 us", round(us2, 1))
            continue
        ex.k1(x, pt, act)
        ttnn.synchronize_device(md)
        if 3 <= int(os.environ.get("TP32PF_DBG", "0")) < 20:
            nb = int(os.environ.get("TP32PF_DBG", "0")) - 2  # blocks summed (j = 0 .. nb - 1)
            T = ttnn.to_torch(act).reshape(-1, 32, 32).float()
            gi = pl.gidx.reshape(-1)
            res = []
            for c in range(NC):
                nu = int(pl.plan[c, 0])
                for u in range(min(nu, 2)):
                    e, mt0, mtc = [int(v) for v in pl.plan[c, 1 + 3 * u : 4 + 3 * u]]
                    mt = mt0
                    b = c & 7
                    wg, wu = wg_wu(e)
                    for m in range(2):
                        rows = gi[(mt + m) * 32 : (mt + m) * 32 + 32].long()
                        ref = 0
                        for jj in range(nb):
                            bb = (jj + c) & 7
                            ref = ref + X[rows][:, bb * 768 : (bb + 1) * 768].float() @ wg[bb * 768 : (bb + 1) * 768]
                        for k in range(2):
                            got = T[(mt + m) * 2 + k]
                            res.append((pcc(ref[:, k * 32 : k * 32 + 32], got), c, mt + m, k))
            res.sort()
            p("DBG3 block-0 matmul pcc worst", res[:4], "n", len(res), "median", res[len(res) // 2][0])
            break
        if int(os.environ.get("TP32PF_DBG", "0")) == 1:
            T = ttnn.to_torch(act).reshape(-1, 2, 2, 32, 32).float()  # [mt-pair, m*2+c.. ] approx
            T = ttnn.to_torch(act).reshape(-1, 32, 32).float()  # tile index = mt*2 + c
            gi = pl.gidx.reshape(-1)
            bad, tot = 0, 0
            for c in range(NC):
                nu = int(pl.plan[c, 0])
                for u in range(nu):
                    e, mt0, mtc = [int(v) for v in pl.plan[c, 1 + 3 * u : 4 + 3 * u]]
                    for pr in range(mtc // 2):
                        mt = mt0 + 2 * pr
                        b = c & 7
                        for m in range(2):
                            rows = gi[(mt + m) * 32 : (mt + m) * 32 + 32].long()
                            for k in range(2):
                                ref = X[rows][:, b * 768 + k * 32 : b * 768 + k * 32 + 32].float()
                                got = T[(mt + m) * 2 + k]
                                tot += 1
                                if not torch.equal(ref, got):
                                    bad += 1
                                    if bad <= 3:
                                        p(
                                            "DBG1 mismatch core",
                                            c,
                                            "mt",
                                            mt + m,
                                            "k",
                                            k,
                                            "ref",
                                            ref[0, :4].tolist(),
                                            "got",
                                            got[0, :4].tolist(),
                                            "pcc",
                                            pcc(ref, got),
                                        )
            p("DBG1 tiles checked", tot, "bad", bad)
            break
        # ---- k1 check: act rows (scaled by score) of some pairs
        A = ttnn.to_torch(act).reshape(-1, 2, 32, 32).permute(0, 2, 1, 3).reshape(-1, 64).float()
        worst1 = 1.0
        g = torch.Generator().manual_seed(rep)
        toks = torch.randperm(TT, generator=g)[:NCHK].tolist()
        # row of (token t, slot j): position within expert e
        ids_l = ids.tolist()
        rowpos = {}
        cntr = {}
        flat = ids.reshape(-1)
        order = torch.argsort(flat * (TT * 8) + torch.arange(TT * 8), stable=True)
        pos = 0
        cnt = torch.bincount(flat, minlength=NE)
        for e in range(NE):
            for r_, pp in enumerate(order[pos : pos + int(cnt[e])].tolist()):
                rowpos[pp] = pl.start[e] * 32 + r_
            pos += int(cnt[e])
        pcs1 = []
        for t in toks:
            for j in range(8):
                e = ids_l[t][j]
                wg, wu = wg_wu(e)
                xr = X[t].float()
                gg, uu = xr @ wg, xr @ wu
                ref = (
                    torch.nn.functional.silu(gg.clamp(max=LIMIT))
                    * uu.clamp(-LIMIT, LIMIT)
                    * (sc[t, j] if SCALE else 1.0)
                )
                pcs1.append((ref, A[rowpos[t * 8 + j]]))
        fin = [bool(torch.isfinite(b).all()) for _, b in pcs1]
        p(
            "finite rows",
            sum(fin),
            "/",
            len(fin),
            "first rows pcc",
            [round(pcc(a, b), 3) if torch.isfinite(b).all() else None for a, b in pcs1[:8]],
        )
        p("row0 ref", pcs1[0][0][:6].tolist(), "got", pcs1[0][1][:6].tolist())
        R = torch.stack([a for a, _ in pcs1])
        G = torch.stack([b for _, b in pcs1])
        worst1 = pcc(R, G)
        p(
            "rep",
            rep,
            "K1 act pcc",
            round(worst1, 5),
            "rows",
            R.shape[0],
            "max|ref|",
            float(R.abs().max()),
            "max|got|",
            float(G.abs().max()),
        )
        if STAGE == "full":
            for S in range(6):
                ex.k2(S, pt, act, out, TT)
            ttnn.synchronize_device(md)
            O = ttnn.to_torch(out).reshape(TT, HID).float()
            refs, gots = [], []
            for t in toks:
                acc = torch.zeros(HID)
                for j in range(8):
                    e = ids_l[t][j]
                    wg, wu = wg_wu(e)
                    xr = X[t].float()
                    a = torch.nn.functional.silu((xr @ wg).clamp(max=LIMIT)) * (xr @ wu).clamp(-LIMIT, LIMIT)
                    acc += (a * (sc[t, j] if SCALE else 1.0)) @ w2e(e)
                refs.append(acc)
                gots.append(O[t])
            Rr, Gg = torch.stack(refs), torch.stack(gots)
            pp_ = pcc(Rr, Gg)
            per_tok = [round(pcc(a, b), 4) for a, b in list(zip(refs, gots))[:8]]
            p(
                "rep",
                rep,
                "FULL partial pcc",
                round(pp_, 5),
                "per-token",
                per_tok,
                "max|ref|",
                float(Rr.abs().max()),
                "max|got|",
                float(Gg.abs().max()),
            )
        if TRACE:
            us1 = traced_us(md, lambda: ex.k1(x, pt, act))
            p("rep", rep, "K1 us", round(us1, 1))
            if STAGE == "full":
                us2 = traced_us(md, lambda: [ex.k2(S, pt, act, out, TT) for S in range(6)])
                p("rep", rep, "K2 x6 us", round(us2, 1))
                p("rep", rep, "K1 + K2 us", round(us1 + us2, 1))
    ttnn.close_mesh_device(md)


main()
