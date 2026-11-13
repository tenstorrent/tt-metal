"""S>=12 PCC debug: per-row / per-head PCC of the Hy4 split kernel."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from dk_dev import *

import ttnn

mod = load_split()
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
cases = [c.split(":") for c in os.environ.get("CASES", "576:12:4:256").split(",")]
try:
    for D, S, G, kc in cases:
        D, S, G, kc = int(D), int(S), int(G), int(kc)
        T = int(os.environ.get("T", "65536"))
        H, Hr, topk = 32, 8, 2048
        torch.manual_seed(0)
        kv = torch.randn(T, D).to(torch.bfloat16)
        if os.environ.get("ONEHOT"):
            kv = torch.zeros(T, D, dtype=torch.bfloat16)
            kv[torch.arange(T), torch.arange(T) % 512] = 1
        per = T // S
        regs = [(s * per, (s + 1) * per) for s in range(S)] if os.environ.get("REGION", "1") == "1" else [(0, T)] * S
        idx = make_idx(S, topk, regs, int(os.environ.get("VALID", topk)))
        q = torch.randn(Hr, S, D).to(torch.bfloat16)
        if os.environ.get("QZERO"):
            q = torch.zeros_like(q)
        sl = torch.linspace(-2, 2, Hr)
        qp = torch.zeros(H, S, D, dtype=torch.bfloat16)
        qp[:Hr] = q
        qdev = (
            qp.reshape(1, H, S, D)
            if os.environ.get("SS_QLAY", "0") == "0"
            else qp.permute(1, 0, 2).contiguous().reshape(1, S, H, D)
        )
        sp = torch.zeros(H)
        sp[:Hr] = sl
        _q, _kv, _ix = (
            to_dev(md, qdev, ttnn.bfloat16),
            to_dev(md, kv.reshape(1, 1, T, D), ttnn.bfloat16),
            idx_dev(md, idx),
        )
        out = mod.sparse_sdpa_split(
            _q,
            _kv,
            _ix,
            512,
            SCALE,
            kc,
            G,
            ckc_of(md),
            to_dev(md, (sp / SCALE).reshape(1, 1, 1, H).to(torch.bfloat16), ttnn.bfloat16),
        )
        ttnn.synchronize_device(md)
        if os.environ.get("REP"):
            out = mod.sparse_sdpa_split(
                to_dev(md, qdev, ttnn.bfloat16),
                to_dev(md, kv.reshape(1, 1, T, D), ttnn.bfloat16),
                idx_dev(md, idx),
                512,
                SCALE,
                kc,
                G,
                ckc_of(md),
                to_dev(md, (sp / SCALE).reshape(1, 1, 1, H).to(torch.bfloat16), ttnn.bfloat16),
            )
        o = ttnn.to_torch(out)[0].float()  # [H,S,512]
        ref = ref_sparse(q, kv, idx, sl)
        res = {
            "addr": [hex(_q.buffer_address()), hex(_kv.buffer_address()), hex(_ix.buffer_address())],
            "D": D,
            "S": S,
            "G": G,
            "kc": kc,
            "pcc_real": round(pcc(o[:Hr], ref), 5),
        }
        res["pcc_per_row"] = [round(pcc(o[:Hr, s], ref[:, s]), 4) for s in range(S)]

        if os.environ.get("MAT"):
            res["bad_matrix"] = [
                "".join("X" if pcc(o[h, s], ref[h, s]) < 0.99 else "." for s in range(S)) for h in range(Hr)
            ]
        if os.environ.get("DIAG"):
            dg = []
            for s_ in range(S):
                for h_ in range(Hr):
                    c = pcc(o[h_, s_], ref[h_, s_])
                    if c < 0.99:
                        best = max(
                            ((pcc(o[h_, s_], ref[h2, s2]), h2, s2) for h2 in range(Hr) for s2 in range(S)),
                            key=lambda x: x[0],
                        )
                        dg.append(
                            (
                                h_,
                                s_,
                                round(c, 3),
                                "zeros%.2f" % float((o[h_, s_] == 0).float().mean()),
                                "ratio%.3f" % float(o[h_, s_].norm() / ref[h_, s_].norm()),
                                "best",
                                round(best[0], 3),
                                best[1],
                                best[2],
                            )
                        )
            res["diag"] = dg[:24]
        if os.environ.get("MATCH"):
            mv = torch.stack([kv[idx[s_][idx[s_] >= 0]].float()[:, :512].mean(0) for s_ in range(S)])  # mean V per row
            mt = []
            for s_ in range(S):
                cs = [pcc(o[1, s_], mv[s2]) for s2 in range(S)]
                b = max(range(S), key=lambda i: cs[i])
                mt.append((s_, round(cs[s_], 3), b, round(cs[b], 3)))
            res["match"] = mt
        if os.environ.get("INTACT"):
            kb = ttnn.to_torch(_kv).reshape(T, D)
            qb = ttnn.to_torch(_q)
            ib = ttnn.to_torch(_ix)
            res["kv_bad_rows"] = int((kb != kv).any(dim=1).sum())
            res["q_equal"] = bool((qb.reshape(qdev.shape) == qdev).all())
            res["idx_equal"] = bool((ib.reshape(S, topk).to(torch.int64) & 0xFFFFFFFF == (idx & 0xFFFFFFFF)).all())
        if os.environ.get("ONEHOT"):
            nv = int(os.environ.get("VALID", topk))
            an = []
            for s_ in range(S):
                ids = idx[s_][idx[s_] >= 0]
                cnt_ref = torch.bincount(ids % 512, minlength=512)
                cnt_dev = (o[0, s_] * (nv + float(torch.exp(sl[0])))).round().long().clamp(min=0)
                if (cnt_ref != cnt_dev).any():
                    miss = (cnt_ref - cnt_dev).clamp(min=0)
                    extra = (cnt_dev - cnt_ref).clamp(min=0)
                    pos_missing = [int(p_) for p_ in range(ids.numel()) if miss[int(ids[p_]) % 512] > 0]
                    an.append(
                        (s_, "missing_pos", pos_missing[:40], "nmiss", int(miss.sum()), "nextra", int(extra.sum()))
                    )
            res["onehot"] = an
        res["pad_head_absmax"] = float(o[Hr:].abs().max())
        emit(res)
finally:
    ttnn.close_mesh_device(md)
