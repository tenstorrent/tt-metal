# SPDX-License-Identifier: Apache-2.0
"""Device-resident spec VERIFY rounds (Decoder(block=8, vdev=True)) vs the host-driven flow (Decoder.verify + accept_prefix), token exact.
Per U in DEC_US (1,2,4) on a DEC_LAYERS-layer model, ragged real GSM8K prompts, R=DEC_ROUNDS rounds:
  reference: host verify rounds; drafts are built so that every user accepts a different number m of drafts (7 refinement passes: draft j := target argmax j-1 for j <= m, random after).
  device: set_verify_state once, per round only write_drafts (device op from a staged device tensor) + non-blocking trace replay; the epilogue does accept_prefix / bonus / positions / alive on
  device; ONE bulk read of the round ring at the end. Gate: post (target argmax of all 8 rows) and committed count equal the reference for every round/row/user, positions/anchor equal.
  eos pass: one token of the reference output becomes eos: alive/commit flags must follow the host spec driver semantics (user dead after the round that commits the eos).
Env: DEC_LAYERS(4) DEC_US(1,2,4) DEC_ROUNDS(3)."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import Decoder  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/verify_device.jsonl")
ROWS, BLK = 4, 8


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("VDEV_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def main():
    NL = int(E("DEC_LAYERS", "4"))
    Us = [int(v) for v in E("DEC_US", "1,2,4").split(",")]
    R = int(E("DEC_ROUNDS", "3"))
    tok = tokenizer()
    items = gsm_prompts(4 * max(Us), tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=Us[0])
        g = torch.Generator().manual_seed(7)
        for k_, U in enumerate(Us):
            if k_ > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            n = ROWS * U
            prompts = [it["prompt_ids"] for it in items[:n]]
            place = gen.place(n)
            pos0 = torch.zeros(ROWS, U, dtype=torch.long)
            for i, (r, s) in enumerate(place):
                pos0[r, s] = len(prompts[i])
            first, _ = gen.prefill(prompts, S)
            anchor0 = first.clone()
            dead = torch.zeros(ROWS, U, dtype=torch.bool)
            # ---- reference (host driven)
            hd = Decoder(model, trace=True, U=U, block=BLK)
            anchor, pos = anchor0.clone(), pos0.clone()
            ref = []
            for rd in range(R):
                m = torch.randint(0, BLK, (ROWS, U), generator=g)
                d = torch.randint(1000, 100000, (ROWS, U, BLK - 1), generator=g)
                for it in range(BLK - 1):
                    post, _ = hd.verify(torch.cat([anchor[:, :, None], d], 2), pos, dead)
                    d[:, :, it] = torch.where(it < m, post[:, :, it], d[:, :, it])
                block = torch.cat([anchor[:, :, None], d], 2)
                post, _ = hd.verify(block, pos, dead)
                kacc = Decoder.accept_prefix(post, block)
                ref.append(dict(d=d.clone(), post=post.clone(), k=kacc.clone(), anchor=anchor.clone(), pos=pos.clone()))
                anchor = post.gather(2, kacc[:, :, None])[:, :, 0]
                pos = pos + kacc + 1
            hd.close()
            rec("reference", U=U, k=[r_["k"].flatten().tolist() for r_ in ref])
            # ---- device-resident
            for variant in ("plain", "eos"):
                gen.prefill(prompts, S)
                eos = ()
                if variant == "eos":
                    eos = (int(ref[0]["post"][0, 0, 0]),)
                dd = Decoder(model, trace=True, U=U, block=BLK, vdev=True, eos=eos, max_pos=max_pos, chunk=R)
                dd.set_verify_state(anchor0, pos0, ~dead)
                t0 = time.perf_counter()
                enq = []
                for rd in range(R):
                    h = ttnn.from_torch(
                        ref[rd]["d"].float().reshape(ROWS, 1, U, BLK - 1),
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=dd.rowmap,
                    )
                    dev = ttnn.to_device(h, mesh)
                    dd.write_drafts(dev)
                    t1 = time.perf_counter()
                    dd.verify_enqueue(1)
                    enq.append(time.perf_counter() - t1)
                post, com = dd.read_rounds(R)
                wall = time.perf_counter() - t0
                # expected
                alive = torch.ones(ROWS, U, dtype=torch.bool)
                ok = True
                bad = []
                for rd in range(R):
                    rp, rk = ref[rd]["post"], ref[rd]["k"]
                    exp_com = torch.where(alive, rk + 1, torch.zeros_like(rk))
                    exp_post = torch.where(alive[:, :, None], rp, torch.zeros_like(rp))
                    if not (torch.equal(com[rd], exp_com) and torch.equal(post[rd], exp_post)):
                        ok = False
                        bad.append(rd)
                    for r in range(ROWS):
                        for u in range(U):
                            if alive[r, u] and any(int(rp[r, u, j]) in eos for j in range(int(rk[r, u]) + 1)):
                                alive[r, u] = False
                dpos = ttnn.to_torch(ttnn.get_device_tensors(dd.vpos)[0]).flatten()[:U]
                rec(
                    "device_vs_reference",
                    U=U,
                    variant=variant,
                    equal=ok,
                    bad_rounds=bad,
                    commits=com.flatten().tolist(),
                    alive_end=alive.flatten().tolist(),
                    row0_pos_end=dpos.tolist(),
                    enqueue_ms=[round(e_ * 1e3, 3) for e_ in enq],
                    wall_ms=round(wall * 1e3, 1),
                )
                dd.close()
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
