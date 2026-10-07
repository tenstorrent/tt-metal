# SPDX-License-Identifier: Apache-2.0
"""Traced chunked prefill (tt/prefill.py) vs the track-0 CPU dumps, then prefill -> decode hand-off.
Env: PF_DUMP (mimo-prefill-s128b16 | s512b4 | s2048b1), PF_CH (chunk, default 2048), PF_LAYERS (6), PF_TRACE (1), PF_DECODE (1), PF_OUT (jsonl), MIMO_EXPERT_OP.
Dump batch b -> mesh row b // UPR, slot b % UPR with UPR = ceil(B / 4). Checks:
 (1) hidden after the last built layer at EVERY prompt position vs the dump h_out of that layer (PCC per user, min over chunks),
 (2) KV pool contents of every built layer vs the dump state (GA: K and V[:128] for all positions; SWA: last 128 positions),
 (3) hand-off: per layer, teacher-forced decode steps (dump dec_in) reading the pool written by the chunked prefill vs dump dec_out,
 (4) first-token logits of the truncated model vs CPU final norm + lm_head of the dump h_out."""
import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.attn_ref import pcc
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedPrefill

E = os.environ.get
_d = E("PF_DUMP", "s128b16")
DUMP = _d if _d.startswith("/") else f"/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-{_d}"
NL = int(E("PF_LAYERS", "6"))
CH = int(E("PF_CH", "2048"))
OUT = E("PF_OUT", "/tmp/pf.jsonl")
H = 4096


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("PF_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 1024 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_prefill_handoff(mesh_device):
    md = mesh_device
    toks = torch.load(f"{DUMP}/tokens.pt")["prefill_tokens"]  # [B,S]
    B, S = toks.shape
    UPR = -(-B // 4)
    where = {b: (b // UPR, b % UPR) for b in range(B)}
    prompts = [[toks[b].tolist() for b in range(B) if where[b][0] == r] for r in range(4)]
    maxpos = -(-(S + 8) // CH) * CH  # room for the decode steps after the prompt
    lay = KVLayout(users=UPR, max_pos=maxpos, chunk=CH)
    model = MiMoModel(md, n_layers=NL, layout=lay, log=lambda s: print(s, flush=True))
    pf = ChunkedPrefill(model, trace=E("PF_TRACE", "1") == "1")
    res = pf.prefill(prompts, collect_full=True)
    # second run: replay of the captured trace on the same data must be bit-identical in KV and hidden
    res2 = pf.prefill(prompts, collect_full=True)
    rec(
        "timing",
        dump=os.path.basename(DUMP),
        B=B,
        S=S,
        chunk=CH,
        layers=NL,
        steps=res["steps"],
        step_ms=[round(t, 1) for t in pf.times],
    )
    rec("replay_identical", full=all(bool(torch.equal(res["full"][k], res2["full"][k])) for k in res["full"]))

    d = [torch.load(f"{DUMP}/layer_{L}.pt", mmap=True) for L in range(NL)]
    ref_h = d[NL - 1]["prefill"]["h_out"]
    per = {}
    for b in range(B):
        hv = res["full"][where[b]].float()
        per[b] = pcc(hv, ref_h[b].float())
    rec(
        "hidden_vs_dump",
        layer=NL - 1,
        pcc_min=round(min(per.values()), 5),
        per_user={b: round(v, 5) for b, v in per.items()},
    )

    # ---- first token logits
    ft = pf.first_tokens(res["last"])
    w = model.ck.get("model.norm.weight").float()
    lm = model.ck.get("lm_head.weight").float()
    pcs, agree = [], []
    for b in range(B):
        x = ref_h[b, -1].float()
        hn = (x * torch.rsqrt(x.pow(2).mean() + 1e-6) * w).bfloat16().float()
        ref = hn @ lm.T
        pcs.append(pcc(ft[where[b]][1], ref))
        agree.append(int(ft[where[b]][0] == int(ref.argmax())))
    rec("first_token_vs_cpu", pcc_min=round(min(pcs), 5), top1_agree=sum(agree), of=B)

    # ---- KV pool vs dump
    tabs = lay.tables()
    for L in range(NL):
        lyr = model.layers[L]
        cfg, swa = lyr.cfg, lyr.swa
        tab = tabs["swa" if swa else "ga"]
        kd, vd = ttnn.get_device_tensors(lyr.attn.k_cache), ttnn.get_device_tensors(lyr.attn.v_cache)
        kp, vp, pad = [], [], []
        st = d[L]["state"]
        for b in range(B):
            r, u = where[b]
            for c in range(8):
                kh = c * cfg.nkv // 8
                kc, vc = ttnn.to_torch(kd[r * 8 + c]).float(), ttnn.to_torch(vd[r * 8 + c]).float()
                if swa:
                    ps = list(range(max(0, S - 128), S))
                    ref_k = torch.stack([st["k_ring"][b, kh, p % 128] for p in ps]).float()
                    ref_v = torch.stack([st["v_ring"][b, kh, p % 128] for p in ps]).float()
                else:
                    ps = list(range(S))
                    ref_k, ref_v = st["k"][b, kh].float(), st["v"][b, kh].float()
                blk = [int(tab[u][(p // 64) % (lay.ring_blocks if swa else 10**9)]) for p in ps]
                kdev = torch.stack([kc[bl, 0, p % 64] for bl, p in zip(blk, ps)])
                vdev = torch.stack([vc[bl, 0, p % 64] for bl, p in zip(blk, ps)])
                kp.append(pcc(kdev, ref_k))
                vp.append(pcc(vdev[:, :128], ref_v))
                pad.append(float(vdev[:, 128:].abs().max()))
        rec(
            "kv_vs_dump",
            layer=L,
            kind="swa" if swa else "ga",
            k_pcc_min=round(min(kp), 5),
            v_pcc_min=round(min(vp), 5),
            v_pad_absmax=max(pad),
        )

    # ---- hand-off: decode of each layer reading the pool, teacher forced with dump dec_in
    if E("PF_DECODE", "1") == "1":
        for L in range(NL):
            lyr = model.layers[L]
            dio = lyr.attn.decode_io()
            steps = []
            for s_, di in enumerate(d[L]["decode"]):
                x = torch.zeros(4, 32, H)
                for b in range(B):
                    r, u = where[b]
                    x[r, u] = di["dec_in"][b, 0].float()
                pos = [0] * 32
                for u in range(UPR):
                    pos[u] = S + s_
                lyr.attn.decode_set(dio, pos)
                dt = model.upload_rows(x)
                o = lyr.decode(dt, dio["cur"], dio["idx"])
                oh = model.rows_to_host(o)
                pc = [pcc(oh[where[b][0], where[b][1]], di["dec_out"][b, 0].float()) for b in range(B)]
                steps.append(dict(step=s_, pcc_min=round(min(pc), 5)))
                ttnn.deallocate(o)
                ttnn.deallocate(dt)
            rec("handoff_decode", layer=L, steps=steps)
    pf.release()
