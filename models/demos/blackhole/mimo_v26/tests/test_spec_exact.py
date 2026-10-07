# SPDX-License-Identifier: Apache-2.0
"""Spec-decode exactness + speed vs plain greedy decode (same prompts, same prefill path), on the 4x8 mesh.
Env: SPEC_LAYERS (48; <48 = skeleton run: capture layers are remapped, acceptance meaningless), SPEC_U (users per mesh row, <=4), SPEC_NEW (max new tokens), SPEC_START (first GSM8K row),
SPEC_OUT (jsonl), SPEC_TRACE (1). Prompts: GSM8K test rows, chat template with enable_thinking=False. n = 4*U prompts (one per (row,slot)).
Reports: token identity per user vs the plain stream (first divergence index), accepted tokens per round (mean over live users), plain decode ms/token, spec ms/round and tokens/s/user."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer
from models.demos.blackhole.mimo_v26.tests.decode_common import gsm_prompts
from models.demos.blackhole.mimo_v26.tt.dflash import DFlashDrafter
from models.demos.blackhole.mimo_v26.tt.generator import load_eos
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedGenerator, ChunkedPrefill
from models.demos.blackhole.mimo_v26.tt.spec import CAP, SpecEngine

E = os.environ.get
NL = int(E("SPEC_LAYERS", "48"))
U = int(E("SPEC_U", "4"))
NEW = int(E("SPEC_NEW", "256"))
START = int(E("SPEC_START", "0"))
TRACE = E("SPEC_TRACE", "1") == "1"
OUT = E("SPEC_OUT", "/tmp/spec.jsonl")
CH = 256


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("SPEC_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return dict(
        alloc_mb_per_bank=round(v.total_bytes_allocated_per_bank / 2**20, 1),
        free_mb_per_bank=round(v.total_bytes_free_per_bank / 2**20, 1),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 4 * 1024 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(21600)
@torch.no_grad()
def test_spec_exact(mesh_device):
    md = mesh_device
    tok = tokenizer()
    items = gsm_prompts(4 * U, tok, START)
    prompts = [it["prompt_ids"] for it in items]
    maxlen = max(len(p) for p in prompts)
    max_pos = -(-(maxlen + NEW + 64) // CH) * CH
    lay = KVLayout(users=U, max_pos=max_pos, chunk=CH)
    model = MiMoModel(md, n_layers=NL, layout=lay, log=lambda s: None)
    cap = CAP if NL == 48 else tuple(min(i, NL - 1) for i in (0, 1, 2, 3, 3))
    pf = ChunkedPrefill(model, trace=True, cap_layers=cap)
    dr = DFlashDrafter(md, model.embed, model.lm_head, U)
    rec("env", layers=NL, U=U, new=NEW, cap=cap, max_pos=max_pos, dram=dram(md), prompt_lens=[len(p) for p in prompts])
    eos = load_eos()
    # ---- plain greedy
    gen = ChunkedGenerator(model, max_pos, pf, trace=TRACE, keep_logits=False, U=U, eos=eos)
    # ONE traced prefill shared by the plain and the spec run (a second traced-prefill replay after decode ops hangs the device, see LOG): the KV it wrote below the
    # prompt length is never modified by plain decode (writes start at position n), spec rounds overwrite positions >= n before reading them.
    rows = 4
    per_row = [[] for _ in range(rows)]
    for i, p in enumerate(prompts):
        per_row[i % rows].append(list(p))
    pre = pf.prefill(per_row, capture_feats=True)
    ft = pf.first_tokens(pre["last"])
    ids = torch.zeros(rows, U, dtype=torch.long)
    for (r_, u_), (t_, _) in ft.items():
        ids[r_, u_] = t_
    gen.prefill = lambda prompts_, S=None: (ids, None)
    r = gen.generate(prompts, NEW, S=maxlen)
    plain = r["tokens"]
    steady = sorted(r["times"][5:])
    rec(
        "plain",
        ms_per_token_median=round(steady[len(steady) // 2] * 1e3, 2) if steady else None,
        steps=len(r["times"]),
        lens=[len(t) for t in plain],
        dram=dram(md),
    )
    gen.dec.release()
    # ---- plain repeat (baseline: decode is not bit-reproducible run to run, LOG determinism entries)
    gen2 = ChunkedGenerator(model, max_pos, pf, trace=TRACE, keep_logits=False, U=U, eos=eos)
    gen2.prefill = lambda prompts_, S=None: (ids, None)
    plain2 = gen2.generate(prompts, NEW, S=maxlen)["tokens"]
    gen2.dec.release()
    pp = []
    for a, b in zip(plain, plain2):
        n_ = min(len(a), len(b))
        pp.append(next((k for k in range(n_) if a[k] != b[k]), None))
    rec(
        "plain_vs_plain",
        identical=sum(1 for d_, a, b in zip(pp, plain, plain2) if d_ is None and len(a) == len(b)),
        first_div=pp,
    )
    # ---- spec
    eng = SpecEngine(model, pf, dr, U, max_pos, eos, cap=cap, trace=TRACE)
    eng.start(prompts, NEW, res=pre)
    t0 = time.perf_counter()
    out = eng.run()
    wall = time.perf_counter() - t0
    ident, first_div = [], []
    for i, (a, b) in enumerate(zip(plain, out)):
        n = min(len(a), len(b))
        d = next((k for k in range(n) if a[k] != b[k]), None)
        first_div.append(d)
        ident.append(d is None and len(a) == len(b))
    acc = [x for rr in eng.stats["rounds"] for x in rr]
    nr = len(eng.stats["rounds"])
    toks = sum(len(t) for t in out)
    rec(
        "spec",
        n=len(prompts),
        identical=sum(ident),
        first_div=first_div,
        out_lens=[len(t) for t in out],
        plain_lens=[len(t) for t in plain],
        rounds=nr,
        accepted_mean=round(sum(acc) / max(1, len(acc)), 3),
        accepted_hist=[acc.count(k) for k in range(1, 9)],
        draft_ms_mean=round(1e3 * sum(eng.stats["t_draft"]) / max(1, nr), 2),
        verify_ms_mean_fallback_8_steps=round(1e3 * sum(eng.stats["t_verify"]) / max(1, nr), 2),
        append_ms_mean=round(1e3 * sum(eng.stats["t_append"]) / max(1, nr), 2),
        wall_s=round(wall, 2),
        tok_s_user=round(toks / len(prompts) / wall, 2),
        dram=dram(md),
    )
    json.dump(dict(plain=plain, plain2=plain2, spec=out), open(OUT.replace(".jsonl", "_tokens.json"), "w"))
    rec("texts", sample=[tok.decode(out[i], skip_special_tokens=True)[:300] for i in range(min(4, len(out)))])
    eng.dec.release()
    dr.release()
    pf.release()
