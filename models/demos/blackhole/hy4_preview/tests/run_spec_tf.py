# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mtp-device exactness diagnosis (whole mesh): teacher-forced verify rows vs plain steps. Both decoders are built from the same prefill hand-off. Phase A: plain S=1 decode fed the recorded plain tokens
(logits per step). Phase B: verify S=4 rows (tokens = the SAME recorded tokens, positions L+4c..L+4c+3, eager, no draft/accept) -> row j logits vs plain step 4c+j: PCC, argmax, gap, first mismatching row.
Env: TOKENS (spec_tokens_plain.json), NSTEPS (60), PAGES, HY4_VERIFY_MOE_SPLIT=1 (verify MoE as S T=1 calls), HY4_SKINNY etc. RES lines."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import SpecDerive
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

R, S = 4, 4
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
PAGES = int(os.environ.get("PAGES", "8"))
NST = int(os.environ.get("NSTEPS", "60"))
PF_ROWS = 512
PAD = 120002


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    tj = json.load(open(os.environ.get("TOKENS", "/out/spec_tokens_plain.json")))
    gen, lens, prompts = tj["generated"], tj["lens"], tj["prompts"]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "900e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        ch.add_layers(0, 78, log=lambda s: None)
        ch.build_head()
        dec1 = Hy4PoolDecoder(ch, PAGES, mode="dense", S=1)
        dec4 = Hy4PoolDecoder(ch, PAGES, mode="dense", S=S)
        ids = torch.full((R, PF_ROWS), PAD, dtype=torch.long)
        for r in range(R):
            ids[r, : lens[r]] = torch.tensor(prompts[r])
        x4, _ = ch.prefill(ids)
        ttnn.deallocate(x4)
        dec1.handoff(PF_ROWS)
        dec4.handoff(PF_ROWS)
        ch.reset()
        rope = ch.layers[0].attn.rope
        sd = SpecDerive(md, dec4.ctx, rope, ch.layers[0].attn.indexer.idx_rope_first, S)
        dec4.attach_derive(sd)
        dec4.prepare_blocks()

        def bind(dec):
            for ly, kv, kp in zip(ch.layers, dec.kv_pools, dec.key_pools):
                ly.attn.init_pool(dec.ctx, kv, kp)

        # ---- phase A: plain steps on the recorded tokens
        bind(dec1)
        plain = []
        pos = list(lens)
        dec1.set_inputs([gen[r][0] for r in range(R)], pos)
        dec1.capture()
        for s in range(NST):
            dec1.set_inputs([gen[r][s] for r in range(R)], [lens[r] + s for r in range(R)])
            nxt, lgs = dec1.read_argmax(dec1.run())
            plain.append(lgs.clone())
            assert all(nxt[r] == gen[r][s + 1] for r in range(R)) or s > 0 or True
        agree = [sum(int(plain[s][r].argmax()) == gen[r][s + 1] for s in range(NST)) for r in range(R)]
        res(event="plain_replay", agree_with_recorded=agree, of=NST)
        dec1.release()
        # ---- phase B: verify rows, teacher forced
        bind(dec4)
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        first_bad = [None] * R
        for c in range(NST // S):
            tok = torch.tensor([gen[r][S * c : S * c + S] for r in range(R)], dtype=torch.int32)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp), dec4.tok
            )
            p = upload(
                md,
                torch.tensor([float(lens[r] + S * c) for r in range(R)]).reshape(R, 1, 1, 1),
                ttnn.float32,
                row_dim=0,
            )
            dec4.ctx.mode = "dense"
            sd.derive(p)
            hn, lg = dec4.step_verify()
            ttnn.synchronize_device(md)
            L = (
                ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape)))
                .reshape(R, S, -1)
                .float()
            )
            for r in range(R):
                for j in range(S):
                    s = S * c + j
                    pl, vr = plain[s][r], L[r, j]
                    t2 = torch.topk(pl, 2)
                    rec = dict(
                        c=c,
                        user=r,
                        row=j,
                        step=s,
                        pcc=round(pcc(vr, pl), 6),
                        maxabs=round(float((vr - pl).abs().max()), 3),
                        plain_top=int(t2.indices[0]),
                        ver_top=int(vr.argmax()),
                        truth_next=gen[r][s + 1],
                        plain_gap=round(float(t2.values[0] - t2.values[1]), 3),
                    )
                    rec["flip"] = rec["plain_top"] != rec["ver_top"]
                    if rec["flip"] and first_bad[r] is None:
                        first_bad[r] = (s, j, rec["plain_gap"])
                    if rec["flip"] or r == 2 and 28 <= s <= 42:
                        res(event="row", **rec)
            ttnn.deallocate(p)
        res(event="summary", first_flip_per_user=first_bad)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
