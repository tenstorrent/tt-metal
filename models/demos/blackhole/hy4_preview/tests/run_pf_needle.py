# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pool-prefill S2: WHOLE-MESH needle test with the DIRECT-TO-POOL chunked prefill: 4 users (one per mesh row), pass-key needles at depths 10/40/70/90 %, chain prefill chunk by chunk (512 rows) written straight
into the sequence-sharded pool (HY4_POOL=1 HY4_POOL_PF=1: no replicated cache, NO hand-off) -> traced pool decode, greedy NSTEPS tokens. Needs all 4 slots of a host (EP32 MoE).
Env: NEEDLE (prompts .pt from gen_needle_prompts.py), NSTEPS (12), TAGNAME, MAXCHUNKS (debug), HY4_* as the integ recommended set + HY4_POOL_PF=1."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = os.environ.get("OUT", "/out")
PF = 512
NSTEPS = int(os.environ.get("NSTEPS", "12"))
TAG = os.environ.get("TAGNAME", "pf")
PAD = 120002


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    assert os.environ.get("HY4_POOL") == "1" and os.environ.get("HY4_POOL_PF") == "1"
    d = torch.load(os.environ["NEEDLE"])
    ids_all, L = d["ids"], d["L"]
    nch = -(-L // PF)
    maxch = int(os.environ.get("MAXCHUNKS", "0")) or nch
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        import models.demos.blackhole.hy4_preview.tt.hy4_pf_attention as _pa

        _pa.LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=os.environ.get("HY4_CKPT", "/ckpt"))
        ch.add_layers(0, 78, log=lambda s: None)
        ch.add_layers(78, 79, log=lambda s: None)
        ch.build_head()
        pages = -(-L // 512) + 2
        dec = Hy4PoolDecoder(ch, pages, mode="topk")
        dec.ctx.set_bucket(min(pages * 512 // 256 * 256, (-(-L // 4096)) * 4096), "topk")
        res(event="built", pages=pages, bucket=dec.ctx.bucket_tokens, **dram_view(md))
        first = None
        t0 = time.time()
        ids_by_layer = {}
        for c in range(maxch):
            ids = torch.full((4, PF), PAD, dtype=torch.long)
            for r in range(4):
                seg = ids_all[r][c * PF : (c + 1) * PF]
                ids[r, : len(seg)] = torch.tensor(seg)
            tc = time.time()
            x4, ids_by_layer = ch.prefill(ids, start=c * PF, ids_by_layer=ids_by_layer)
            if c == nch - 1:
                hn, logits = ch.head(x4)
                lg = (
                    ttnn.to_torch(
                        logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                    )
                    .reshape(4, PF, -1)
                    .float()
                )
                last = (L - 1) - c * PF
                first = [int(lg[r, last].argmax()) for r in range(4)]
            ttnn.synchronize_device(md)
            ttnn.deallocate(x4)
            if c % 8 == 0 or c == nch - 1 or c < 6:
                res(
                    event="prefill_chunk",
                    c=c,
                    of=nch,
                    chunk_sec=round(time.time() - tc, 2),
                    sec=round(time.time() - t0, 1),
                    **{k: v for k, v in dram_view(md).items() if k in ("alloc_per_bank", "free_per_bank")},
                )
        res(event="prefill_done", sec=round(time.time() - t0, 1), first=first)
        if maxch < nch:
            return
        ch.reset()
        toks = [[t] for t in first]
        pos = [L] * 4
        dec.set_inputs([t[-1] for t in toks], pos)
        dec.capture()
        for s in range(NSTEPS):
            dec.set_inputs([t[-1] for t in toks], pos)
            nxt, _ = dec.read_argmax(dec.run())
            for r in range(4):
                toks[r].append(nxt[r])
                pos[r] += 1
        res(event="generated", toks=toks, keys=d["keys"], depths=d["depths"])
        json.dump(dict(toks=toks, keys=d["keys"], depths=d["depths"], L=L), open(f"{OUT}/needle_out_{TAG}.json", "w"))
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
