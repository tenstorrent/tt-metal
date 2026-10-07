# SPDX-License-Identifier: Apache-2.0
"""Segment timing (traced replays, wall, pipelined) of the spec-decode pieces on the 4x8 mesh: verify (block 8) vs plain decode per layer / per stage (attention, MoE/FFN), and the drafter
(cumulative layers, attention half, lm_head, append). Every segment is captured as its OWN trace on the real hidden state (layers before it are run eagerly), replayed PROF_REPS times
back to back with one synchronize, so the number is device time + per-trace dispatch (sub-0.1 ms segments are noise).
Env: PROF_U (1), PROF_NL (12), PROF_REPS (10), PROF_OUT (jsonl), MIMO_VERIFY_WRITE (skip|seq|batch), MIMO_EXPERT_OP etc."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer
from models.demos.blackhole.mimo_v26.tests.decode_common import gsm_prompts
from models.demos.blackhole.mimo_v26.tt.decode import HIDDEN, Decoder
from models.demos.blackhole.mimo_v26.tt.dflash import DFlashDrafter
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedPrefill

E = os.environ.get
U = int(E("PROF_U", "1"))
NL = int(E("PROF_NL", "12"))
REPS = int(E("PROF_REPS", "10"))
OUT = E("PROF_OUT", "/tmp/spec_prof.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("PROF_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 2 * 1024 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_spec_profile(mesh_device):
    md = mesh_device
    tok = tokenizer()
    items = gsm_prompts(4 * U, tok, 0)
    prompts = [it["prompt_ids"] for it in items]
    CH = 256
    max_pos = 512
    lay = KVLayout(users=U, max_pos=max_pos, chunk=CH)
    model = MiMoModel(md, n_layers=NL, layout=lay, log=lambda s: None)
    pf = ChunkedPrefill(model, trace=True)
    per_row = [[] for _ in range(4)]
    for i, p in enumerate(prompts):
        per_row[i % 4].append(list(p))
    res = pf.prefill(per_row)
    ft = pf.first_tokens(res["last"])
    first = torch.zeros(4, U, dtype=torch.long)
    pos = torch.zeros(4, U, dtype=torch.long)
    for i, p in enumerate(prompts):
        r, u = i % 4, i // 4
        first[r, u] = ft[(r, u)][0]
        pos[r, u] = len(p)
    os.environ["MIMO_CHECK_CAP"] = "0"
    segs = {}

    def seg(name, fn, reps=REPS):
        o = fn()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        o2 = fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(md, tid)
        segs[name] = round(ms, 3)
        rec("seg", name=name, ms=round(ms, 3))
        return o2

    dead = torch.zeros(4, U, dtype=torch.bool)
    block = torch.cat([first[:, :, None], first[:, :, None].expand(4, U, 7)], 2)
    dv = Decoder(model, trace=False, U=U, block=8)
    dv.set_inputs(block, pos, dead)
    dp = Decoder(model, trace=False, U=U, block=1)
    dp.set_inputs(first, pos, dead)

    def embed(d, B):
        e = ttnn.embedding(d.tok, model.embed, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(e, (1, 1, B, HIDDEN))

    for name, d, blk in (("verify", dv, 8), ("plain", dp, 1)):
        B = U * blk
        h = embed(d, B)
        for L, lyr in enumerate(model.layers):
            kind = ("dense" if L == 0 else "moe") + "_" + ("swa" if lyr.swa else "global")
            hin = ttnn.clone(h)
            live = d.live.t
            upd = d.upd if blk > 1 else None
            seg(
                f"{name}_L{L}_{kind}",
                lambda: lyr.decode(hin, d.pos["cur"], d.pos["idx"], live=live, block=blk, upd=upd),
            )
            if L in (1, 5):

                def attn_part():
                    xn = lyr.rms(hin, lyr.w_in)
                    a = lyr.attn.decode(xn, d.pos["cur"], d.pos["idx"], blk, upd)
                    return ttnn.add(hin, a, dtype=ttnn.bfloat16)

                h1 = attn_part()
                seg(f"{name}_L{L}_attn_part", attn_part)
                fn_ = ttnn.clone(lyr.rms(h1, lyr.w_post))
                seg(f"{name}_L{L}_moe_part", lambda: lyr.moe_decode(fn_, live))
            h = lyr.decode(h, d.pos["cur"], d.pos["idx"], live=live, block=blk, upd=upd)

        # lm_head + norm
        def head():
            from models.demos.blackhole.mimo_v26.tt.layer import EPS, _k_norm

            x = ttnn.rms_norm(h, epsilon=EPS, weight=model.w_norm, compute_kernel_config=_k_norm())
            lg = ttnn.linear(x, model.lm_head, dtype=ttnn.bfloat16)
            return ttnn.max(lg, dim=3, keepdim=True), ttnn.argmax(
                ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True
            )

        seg(f"{name}_head", head)

    # ---- drafter
    dr = DFlashDrafter(md, model.embed, model.lm_head, U)
    dr.set_round(first, pos)
    t0 = time.perf_counter()
    for _ in range(5):
        dr.set_round(first, pos)
    rec("host", set_round_ms=round((time.perf_counter() - t0) / 5 * 1e3, 2))
    seg("draft_embed_only", lambda: dr._draft_forward(nl=0, head=False))
    for k in range(1, 6):
        seg(f"draft_{k}layers_nohead", lambda: dr._draft_forward(nl=k, head=False))
    seg("draft_1layer_attn_only", lambda: dr._draft_forward(nl=1, attn_only_last=True, head=False))
    seg("draft_full", lambda: dr._draft_forward())
    # append: eager-path fc (K-sharded in_feats + all_reduce) and the device-resident path
    dr.set_append(
        torch.zeros(4, U, 8, 20480, dtype=torch.bfloat16),
        pos[:, :, None] + torch.arange(8),
        torch.ones(4, U, 8, dtype=torch.bool),
    )
    seg("append_hostfeats", dr._append_forward)
    dr.feat_src = ttnn.from_torch(
        torch.zeros(1, 1, U * 8, 20480).bfloat16(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    seg("append_devfeats", dr._append_forward)
    rec("summary", U=U, NL=NL, segs=segs)
