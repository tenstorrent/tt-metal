# SPDX-License-Identifier: Apache-2.0
"""Spec-decode VERIFY step (virtual users, block=8) vs 8 sequential plain decode steps, token exact, plus the 5-layer-style hidden capture.
For U in DEC_US (1,2,4) users per mesh row on a DEC_LAYERS-layer model (one build, Generator.reconfigure between U), ragged real GSM8K prompts:
  1. prefill, then ONE verify step over blocks [t0, d1..d7] (random token ids, so no draft is 'right') at positions len..len+7 of every user. Done FIRST on a freshly prefilled cache whose
     positions >= len hold pad junk: a clobbered or invisible same-step KV write would change the result (stale correct data cannot hide it).
  2. re-prefill, then 8 sequential decode steps (block=1 Decoder) feeding the same tokens at the same positions.
  Gate: argmax of virtual user t == argmax of sequential step t for every row, user, t (token exact); captured hidden states (all layers) of verify == sequential: bit-identical or PCC >= 0.9999.
Env: DEC_LAYERS(4) DEC_US(1,2,4) DEC_BLOCK(8) DEC_TRACE(1)."""
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import pcc  # noqa: E402
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import Decoder  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/verify_decode.jsonl")
ROWS = 4


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def feat_host(mesh, t):
    dts = ttnn.get_device_tensors(t)
    cols = mesh.shape[1]
    return torch.cat([ttnn.to_torch(dts[r * cols]).float() for r in range(ROWS)], 0)[:, 0]  # [rows, B, F]


def kv_rows(model, mesh, pos, U, blk, layer=0, col=0):
    """K and V rows of positions pos..pos+blk-1 of every (row, user) for `layer`, device column `col`: [rows, U, blk, 192] each"""
    a = model.layers[layer].attn
    cols = mesh.shape[1]
    kd, vd = ttnn.get_device_tensors(a.k_cache), ttnn.get_device_tensors(a.v_cache)
    ks, vs = torch.zeros(ROWS, U, blk, 192), torch.zeros(ROWS, U, blk, 192)
    for r in range(ROWS):
        kc, vc = ttnn.to_torch(kd[r * cols + col]).float(), ttnn.to_torch(vd[r * cols + col]).float()
        for u in range(U):
            for t in range(blk):
                p = int(pos[r, u]) + t
                bl = int(a.page_table[u][(p // 64) % a.ring_blocks]) if a.ring_blocks else int(a.page_table[u][p // 64])
                ks[r, u, t], vs[r, u, t] = kc[bl, 0, p % 64], vc[bl, 0, p % 64]
    return ks, vs


def main():
    NL = int(E("DEC_LAYERS", "4"))
    Us = [int(v) for v in E("DEC_US", "1,2,4").split(",")]
    blk = int(E("DEC_BLOCK", "8"))
    trace = E("DEC_TRACE", "1") == "1"
    tok = tokenizer()
    items = gsm_prompts(4 * max(Us), tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=trace, keep_logits=False, U=Us[0])
        g = torch.Generator().manual_seed(5)
        for k, U in enumerate(Us):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            n = ROWS * U
            prompts = [it["prompt_ids"] for it in items[:n]]
            place = gen.place(n)
            pos = torch.zeros(ROWS, U, dtype=torch.long)
            for i, (r, s) in enumerate(place):
                pos[r, s] = len(prompts[i])
            toks = torch.randint(1000, 100000, (ROWS, U, blk), generator=g)
            dead = torch.zeros(ROWS, U, dtype=torch.bool)
            cap = tuple(range(NL))
            # 1. verify on a fresh prefill
            gen.prefill(prompts, S)
            vd = Decoder(model, trace=trace, U=U, block=blk, capture=cap)
            av, bv = vd.verify(toks, pos, dead)
            kv_ver = kv_rows(model, mesh, pos, U, blk)
            fv = feat_host(mesh, vd.feat)  # [rows, U*blk, NL*4096]
            av2, _ = vd.verify(toks, pos, dead)
            fv2 = feat_host(mesh, vd.feat)
            verify_repeat_equal = bool(torch.equal(fv, fv2))
            vd.close()
            # 2. sequential on a fresh prefill
            gen.prefill(prompts, S)
            sd = Decoder(model, trace=trace, U=U, block=1, capture=cap)
            aseq = torch.zeros(ROWS, U, blk, dtype=torch.long)
            fseq = torch.zeros(ROWS, U, blk, NL * 4096)
            for t in range(blk):
                sd.set_inputs(toks[:, :, t], pos + t, dead)
                a, b = sd.run()
                aseq[:, :, t] = a
                fseq[:, :, t] = feat_host(mesh, sd.feat)
            sd.close()
            kv_seq = kv_rows(model, mesh, pos, U, blk)
            kdiff = (kv_ver[0] != kv_seq[0]).any(-1)  # [rows,U,blk]
            vdiff = (kv_ver[1] != kv_seq[1]).any(-1)
            rec(
                "kv_rows",
                U=U,
                k_rows_differ_per_t=kdiff.float().mean((0, 1)).tolist(),
                v_rows_differ_per_t=vdiff.float().mean((0, 1)).tolist(),
            )
            match = av == aseq
            fvr = fv.reshape(ROWS, U, blk, -1)
            ident = bool(torch.equal(fvr, fseq))
            pcs = [pcc(fvr[r, u, t], fseq[r, u, t]) for r in range(ROWS) for u in range(U) for t in range(blk)]
            sl = lambda l, t: slice(l * 4096, (l + 1) * 4096)
            per_layer = [
                round(
                    min(
                        pcc(fvr[r, u, t, sl(l, t)], fseq[r, u, t, sl(l, t)])
                        for r in range(ROWS)
                        for u in range(U)
                        for t in range(blk)
                    ),
                    5,
                )
                for l in range(NL)
            ]
            per_t0 = [
                round(
                    min(pcc(fvr[r, u, 0, sl(l, 0)], fseq[r, u, 0, sl(l, 0)]) for r in range(ROWS) for u in range(U)), 5
                )
                for l in range(NL)
            ]
            rec(
                "verify_vs_sequential",
                verify_repeat_bit_identical=verify_repeat_equal,
                per_layer_pcc_min=per_layer,
                per_layer_pcc_min_t0=per_t0,
                U=U,
                B=n,
                virtual_rows=U * blk,
                lens=[len(p) for p in prompts][:8],
                argmax_equal_frac=float(match.float().mean()),
                argmax_all_equal=bool(match.all()),
                mismatches=[(int(r), int(u), int(t)) for r, u, t in match.logical_not().nonzero()][:8],
                feat_bit_identical=ident,
                feat_pcc_min=round(min(pcs), 6),
                dram=dram(mesh),
            )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
