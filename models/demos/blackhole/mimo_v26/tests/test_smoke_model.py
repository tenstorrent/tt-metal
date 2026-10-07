# SPDX-License-Identifier: Apache-2.0
"""First-token + 1 decode step smoke of the first SMOKE_LAYERS layers (48 = full model) on the 4x8 mesh, two ISLs (S=128 and S=256) in ONE process.

Phase 1 (SMOKE_LAYERS <= 6): dump tokens (S=128, 16 users) -> device logits of the last token vs a CPU final-norm + lm_head applied to the track-0 dump h_out of
  the last built layer (bf16 oracle for a truncated model): logits PCC and top-1 agreement.
Phase 2: real GSM8K prompts (chat template, gsm8k_512.json), each mesh row's slot u gets a prompt of the SAME token length (decode positions are shared by all
  rows), right-padded with the pad id to S; first token = argmax of the last real position, then one decode step at position len. ISL A = S 128, ISL B = S 256.
Env: SMOKE_LAYERS (4), SMOKE_OUT (jsonl), MIMO_CKPT.
"""
import json
import os
from collections import defaultdict

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.attn_ref import pcc
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel

E = os.environ.get
NL = int(E("SMOKE_LAYERS", "4"))
OUT = E("SMOKE_OUT", "/tmp/smoke.jsonl")
DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
GSM = "/mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_512.json"
PAD = 151643
ROWS, UPR = 4, 4


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("SMOKE_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def pick(items, lo, hi, skip=0):
    """4 lengths with >= ROWS prompts each inside (lo, hi]: returns [slot] -> list of ROWS items of equal token length"""
    by = defaultdict(list)
    for it in items:
        n = len(it["prompt_ids"])
        if lo < n <= hi:
            by[n].append(it)
    good = sorted([n for n, v in by.items() if len(v) >= ROWS], reverse=True)[skip : skip + UPR]
    assert len(good) == UPR, (lo, hi, {n: len(v) for n, v in by.items()})
    return [(n, by[n][:ROWS]) for n in good]


def run_isl(model, tok, items, S, lo, skip=0):
    slots = pick(items, lo, 128, skip)
    tokens = torch.full((ROWS, UPR * S), PAD, dtype=torch.int64)
    lens = torch.zeros(ROWS, UPR, dtype=torch.int64)
    for u, (n, its) in enumerate(slots):
        for r, it in enumerate(its):
            tokens[r, u * S : u * S + n] = torch.tensor(it["prompt_ids"])
            lens[r, u] = n
    ids, lg = model.prefill_first_token(tokens, lens, S, UPR)
    # decode 1 step: feed the first token at position len (same per row by construction)
    dtok = torch.full((ROWS, 32), PAD, dtype=torch.int64)
    dtok[:, :UPR] = ids
    pos = [0] * 32
    for u, (n, _) in enumerate(slots):
        pos[u] = n
    dl = model.decode_step(dtok, pos)
    ids2 = dl[:, :UPR].argmax(-1)
    rows = []
    for r in range(ROWS):
        for u, (n, its) in enumerate(slots):
            rows.append(
                dict(
                    row=r,
                    slot=u,
                    idx=its[r]["idx"],
                    len=n,
                    tok1=int(ids[r, u]),
                    txt1=tok.decode([int(ids[r, u])]),
                    tok2=int(ids2[r, u]),
                    txt2=tok.decode([int(ids2[r, u])]),
                    finite=bool(torch.isfinite(lg[r, u]).all() and torch.isfinite(dl[r, u]).all()),
                )
            )
    rec("gsm8k_first_tokens", S=S, layers=NL, lens=[n for n, _ in slots], rows=rows)
    return rows


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 64 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_smoke_model(mesh_device):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL"), trust_remote_code=True)
    items = json.load(open(GSM))["items"]
    model = MiMoModel(
        mesh_device, n_layers=NL, max_pos=512, users=32, ring_blocks=256 // 64 + 8, log=lambda s: print(s, flush=True)
    )
    rec("env", layers=NL, expert_op=E("MIMO_EXPERT_OP", "mm"), cache=model.wc.stats)

    if NL <= 6:
        toks = torch.load(f"{DUMP}/tokens.pt")["prefill_tokens"].reshape(ROWS, UPR * 128)
        lens = torch.full((ROWS, UPR), 128, dtype=torch.int64)
        ids, lg = model.prefill_first_token(toks, lens, 128, UPR)
        d = torch.load(f"{DUMP}/layer_{NL - 1}.pt")["prefill"]["h_out"][:, -1].float()  # [16,4096]
        w = model.ck.get("model.norm.weight").float()
        hn = (d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + 1e-6) * w).bfloat16().float()
        ref = hn @ model.ck.get("lm_head.weight").float().T  # [16, V]
        dev = lg.reshape(16, -1)
        pc = [pcc(dev[b], ref[b]) for b in range(16)]
        agree = float((dev.argmax(-1) == ref.argmax(-1)).float().mean())
        rec(
            "truncated_logits_vs_cpu",
            layers=NL,
            pcc_min=round(min(pc), 5),
            pcc_mean=round(sum(pc) / 16, 5),
            top1_agree=agree,
        )

    rA = run_isl(model, tok, items, 128, 0)
    rB = run_isl(model, tok, items, 256, 0, skip=4) if os.environ.get("SMOKE_ISLB", "1") == "1" else []
    assert all(r["finite"] for r in rA + rB)
