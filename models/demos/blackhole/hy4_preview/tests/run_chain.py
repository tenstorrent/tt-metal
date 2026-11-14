# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chain S1/S2 device runner (inside the image, whole 4x8 mesh). Env: NL (layers, default 4), TEST (short|long|both), KVFMT, HY4_CKPT, DUMPS, OUT.
Stage A: layers 0..NL-1 teacher-free chain from embeddings vs the oracle dump (dump_v1: layers 0-3 carry raw streams for the 7 short prompts and the last 32 positions of the long one).
Prints 'RES {json}' lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain, dram_view
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
DUMPS = os.environ.get("DUMPS", "/dumps")
OUT = os.environ.get("OUT", "/out")
NL = int(os.environ.get("NL", "4"))
TEST = os.environ.get("TEST", "both")
KVFMT = os.environ.get("KVFMT", "bf16")
PAD = 120002
ORACLE_LAYERS = os.environ.get("ORACLE_LAYERS", "")


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def to_rows(t, md):
    """per mesh row device-0-of-the-row tensor -> list of 4 torch."""
    c = int(md.shape[1])
    devs = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(devs[r * c]) for r in range(int(md.shape[0]))]


def cpu_head(ch, x4):
    """oracle hc_head + norm + lm_head on [S,4,D] fp32 (formulas of reference/oracle.py)."""
    g = lambda k: ch.rd.get(k).float()
    xf = x4.flatten(1)
    eps = ch.cfg.rms_norm_eps
    m = torch.nn.functional.linear(
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps), g("model.hc_head.hc_head_fn")
    )
    w = torch.sigmoid(m * g("model.hc_head.hc_head_scale")[0] + g("model.hc_head.hc_head_base")) + ch.cfg.hc_eps
    h = (w.unsqueeze(-1) * x4).sum(1)
    hn = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * g("model.norm.weight")
    return hn, hn @ g("lm_head.weight").T


def compare_layers(taps, md, ref_of, rows_real, tag, nl):
    """taps[L] device x4 [1,4,n,D]; ref_of(L, row) -> (oracle out [S,4,D], first device position of the compared slice)."""
    out = []
    for L in range(nl):
        devs = to_rows(taps[L], md)
        rec = {"layer": L}
        pc_all, mins = [], []
        for r in range(4):
            ref, p0 = ref_of(L, r)
            S = ref.shape[0]
            dev = devs[r].reshape(4, -1, 6144)[:, p0 : p0 + S].permute(1, 0, 2).float()
            pc_all.append(pcc(dev, ref))
            mins.append(min(pcc(dev[t], ref[t]) for t in range(S)))
        rec.update(pcc_rows=[round(x, 5) for x in pc_all], pcc_min=round(min(pc_all), 5), tok_min=round(min(mins), 4))
        res(test=tag, **rec)
        out.append(rec)
    return out


def main():
    ids_all = (
        torch.load(f"{OUT}/chain/prompt_ids.pt")
        if os.path.exists(f"{OUT}/chain/prompt_ids.pt")
        else torch.load(f"{OUT}/prompt_ids.pt")
    )
    md = open_mesh(trace=500_000_000)
    try:
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        res(event="dram_open", **dram_view(md))
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT, kv_fmt=KVFMT)
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: print(s, flush=True))
        if os.environ.get("MTP", "0") == "1":
            ch.add_layers(78, 79, log=lambda s: print(s, flush=True))
        ch.build_head()
        json.dump(ch.load_log, open(f"{OUT}/load_log_{os.environ.get('TAGNAME', 'x')}.json", "w"))
        res(
            event="built",
            sec=round(time.time() - t0, 1),
            layers=NL,
            mtp=os.environ.get("MTP", "0"),
            rd_bytes=ch.rd.bytes_read,
        )
        res(event="dram_built", **dram_view(md))
        if TEST in ("short", "both"):
            ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
            lens = []
            for r in range(4):
                ids[r, : len(ids_all[r])] = torch.tensor(ids_all[r])
                lens.append(len(ids_all[r]))
            taps = {} if (NL <= 4 or ORACLE_LAYERS) else None
            t0 = time.time()
            x4, _ = ch.prefill(ids, taps=taps)
            ttnn.synchronize_device(md)
            res(event="short_prefill", sec=round(time.time() - t0, 2))
            ref = {}

            def ref_of(L, r):
                k = (L, r)
                if k not in ref:
                    ref[k] = torch.load(f"{ORACLE_LAYERS or DUMPS}/layer{L:02d}_p{r}.pt")["out"][0]
                return ref[k], 0

            if ORACLE_LAYERS:
                compare_layers(taps, md, ref_of, lens, "short", NL)
            if NL <= 4:
                compare_layers(taps, md, ref_of, lens, "short", NL) if not ORACLE_LAYERS else None
                hn_ref_rows = [cpu_head(ch, ref_of(NL - 1, r)[0]) for r in range(4)]
                hn, logits = ch.head(x4)
                hn_rows = to_rows(hn, md)
                lg = ttnn.to_torch(
                    logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                )
                lg = lg.reshape(4, CHUNK, -1).float()
                for r in range(4):
                    S = lens[r]
                    hr, lr = hn_ref_rows[r]
                    top_dev, top_ref = lg[r, :S].argmax(-1), lr.argmax(-1)
                    res(
                        test="short_head",
                        row=r,
                        hidden_pcc=round(pcc(hn_rows[r].reshape(CHUNK, -1)[:S].float(), hr), 5),
                        logits_pcc=round(pcc(lg[r, :S], lr), 5),
                        top1_last=bool(top_dev[-1] == top_ref[-1]),
                        top1_all=round(float((top_dev == top_ref).float().mean()), 3),
                    )
            if NL == 78:
                fin = json.load(open(f"{DUMPS}/final.json"))
                hn, logits = ch.head(x4)
                hn_rows = to_rows(hn, md)
                lg = ttnn.to_torch(
                    logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                )
                lg = lg.reshape(4, CHUNK, -1).float()
                for r in range(4):
                    S = lens[r]
                    fh = torch.load(f"{DUMPS}/final_hidden_p{r}.pt")["final_normed"][0]
                    top5 = lg[r, S - 1].topk(5)[1].tolist()
                    res(
                        test="full_head",
                        row=r,
                        hidden_pcc=round(pcc(hn_rows[r].reshape(CHUNK, -1)[:S].float(), fh), 5),
                        hidden_pcc_last=round(pcc(hn_rows[r].reshape(CHUNK, -1)[S - 1].float(), fh[-1]), 5),
                        top1_match=top5[0] == fin[r]["last_top5"][0],
                        top5_dev=top5,
                        top5_ref=fin[r]["last_top5"],
                        top5_overlap=len(set(top5) & set(fin[r]["last_top5"])),
                    )
            for t in (taps or {}).values():
                ttnn.deallocate(t)
            ch.reset()
        if TEST in ("long", "both"):
            ids_l = ids_all[7]
            S = len(ids_l)
            nch = -(-S // CHUNK)
            full = torch.full((nch * CHUNK,), PAD, dtype=torch.long)
            full[:S] = torch.tensor(ids_l)
            ids_by = None
            t0 = time.time()
            for c in range(nch):
                chunk = full[c * CHUNK : (c + 1) * CHUNK].unsqueeze(0).repeat(4, 1)
                taps = {} if (c == nch - 1 and NL <= 4) else None
                x4, ids_by = ch.prefill(chunk, start=c * CHUNK, ids_by_layer=ids_by, taps=taps)
                ttnn.synchronize_device(md)
                res(event="long_chunk", c=c, sec=round(time.time() - t0, 2))
                if c < nch - 1:
                    ttnn.deallocate(x4)
                    for v in ids_by.values():
                        ttnn.deallocate(v)
                    ids_by = None
            last0 = (nch - 1) * CHUNK

            def ref_l(L, r):
                return torch.load(f"{DUMPS}/layer{L:02d}_p7.pt")["out"][0], S - 32 - last0

            if NL <= 4:
                compare_layers(taps, md, ref_l, None, "long", NL)
                hn, logits = ch.head(x4)
                hr, lr = cpu_head(ch, ref_l(NL - 1, 0)[0])
                lg = ttnn.to_torch(
                    logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                )
                lg = lg.reshape(4, CHUNK, -1).float()[0, S - 32 - last0 : S - last0]
                res(
                    test="long_head",
                    logits_pcc=round(pcc(lg, lr), 5),
                    top1_all=round(float((lg.argmax(-1) == lr.argmax(-1)).float().mean()), 3),
                    top1_last=bool(lg[-1].argmax() == lr[-1].argmax()),
                )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
