# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter on device vs the CPU oracle (reference/dflash_ref.py), teacher forced on the device captures of the 48-layer prefill (mimo-device-s128b16).
Anchors p in DF_ANCHORS (all users share p in one round); ctx = captured hidden states of layers [0,11,23,35,47] (outputs, L47 pre-norm) at positions < p; anchor token = gold token at p.
Checks per round: PCC of the draft hidden rows (7 per user, after the final norm) vs the oracle (bf16 CPU), top-1 draft agreement per position vs the oracle (same bf16 lm_head).
Env: DF_DEV (capture dir), DF_WDTYPE (bf8|bf16), DF_USERS (users per row: 4 uses all 16 sequences, 1 uses sequences 0,4,8,12), DF_ANCHORS, DF_OUT."""
import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference import dflash_ref as D
from models.demos.blackhole.mimo_v26.reference.loader import Ckpt
from models.demos.blackhole.mimo_v26.tests.attn_ref import pcc
from models.demos.blackhole.mimo_v26.tt.dflash import DFlashDrafter

E = os.environ.get
DEV = E("DF_DEV", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16")
OUT = E("DF_OUT", "/tmp/dflash.jsonl")
WD = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[E("DF_WDTYPE", "bf8")]
UPR = int(E("DF_USERS", "4"))
ANCH = [int(v) for v in E("DF_ANCHORS", "16,64,120").split(",")]


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DF_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_dflash_device(mesh_device):
    md = mesh_device
    ck = Ckpt()
    tok = torch.load(f"{DEV}/logits.pt")["prefill_tokens"]  # [16,128]
    feats = torch.cat(
        [torch.load(f"{DEV}/layer_{L}.pt")["prefill"]["h_out"] for L in D.TARGET_LAYERS], -1
    )  # [16,128,20480] bf16
    sd = D.load_dflash()
    Etab = ck.get("model.embed_tokens.weight").bfloat16()
    head = ck.get("lm_head.weight").bfloat16()
    hn = D.ctx_features(feats, sd)
    sd32 = {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}
    hn32 = D.ctx_features(feats.float(), sd32)
    E32 = Etab.float()
    sd32 = {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}
    hn32 = D.ctx_features(feats.float(), sd32)
    E32 = Etab.float()
    rep = ttnn.ReplicateTensorToMesh(md)
    embed = ttnn.from_torch(
        Etab,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    lm = ttnn.from_torch(
        head.T.contiguous().reshape(1, 1, 4096, -1),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, md.shape, dims=(None, 3)),
    )
    dr = DFlashDrafter(md, embed, lm, wdtype=WD, sd=sd)
    users = [r * 4 + u for r in range(4) for u in range(UPR)]  # sequence index of (row, slot)
    for p in ANCH:
        f = torch.stack([feats[b, :p] for b in users]).reshape(4, UPR, p, -1)
        dr.set_context(f)
        a_tok = torch.stack([tok[b, p] for b in users]).reshape(4, UPR)
        y = dr.draft(a_tok, torch.full((4, UPR), p))
        hid = dr.hidden_host(y)[:, :, 1:]  # [rows, U, 7, 4096]
        lg = dr.logits(y)[:, :, 1:] if False else None
        ref = (
            torch.stack([D.draft_blocks(hn[b], tok[b, p : p + 1], torch.tensor([p]), sd, Etab)[0] for b in users])
            .reshape(4, UPR, 7, 4096)
            .float()
        )
        pcs = [pcc(hid[r, u], ref[r, u]) for r in range(4) for u in range(UPR)]
        # top-1 agreement: both through the same bf16 lm_head on CPU
        dt = (hid.reshape(-1, 4096).bfloat16() @ head.T).argmax(-1).reshape(4, UPR, 7)
        rt = (ref.reshape(-1, 4096).bfloat16() @ head.T).argmax(-1).reshape(4, UPR, 7)
        agree = (dt == rt).float().mean((0, 1)).tolist()
        # device lm_head path too
        dl = dr.logits(y).reshape(4, UPR, 8, -1)[:, :, 1:].argmax(-1)
        agree_dev = (dl == rt).float().mean((0, 1)).tolist()
        ref32 = torch.stack(
            [D.draft_blocks(hn32[b], tok[b, p : p + 1], torch.tensor([p]), sd32, E32)[0] for b in users]
        ).reshape(4, UPR, 7, 4096)
        r32t = (ref32.reshape(-1, 4096).bfloat16() @ head.T).argmax(-1).reshape(4, UPR, 7)
        pc32 = [pcc(hid[r, u], ref32[r, u]) for r in range(4) for u in range(UPR)]
        floor = [pcc(ref[r, u], ref32[r, u]) for r in range(4) for u in range(UPR)]
        rec(
            "round32",
            anchor=p,
            dev_vs_fp32_pcc_min=round(min(pc32), 5),
            bf16oracle_vs_fp32_pcc_min=round(min(floor), 5),
            dev_vs_fp32_top1=[round(a, 4) for a in (dt == r32t).float().mean((0, 1)).tolist()],
            bf16oracle_vs_fp32_top1=[round(a, 4) for a in (rt == r32t).float().mean((0, 1)).tolist()],
        )
        rec(
            "round",
            wdtype=E("DF_WDTYPE", "bf8"),
            U=UPR,
            anchor=p,
            hidden_pcc_min=round(min(pcs), 5),
            hidden_pcc_mean=round(sum(pcs) / len(pcs), 5),
            top1_agree_per_pos=[round(a, 4) for a in agree],
            top1_agree_devhead=[round(a, 4) for a in agree_dev],
        )
        ttnn.deallocate(y)
