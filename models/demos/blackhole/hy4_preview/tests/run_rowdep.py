# SPDX-License-Identifier: Apache-2.0
"""verify-round track P1: row-dependence of verify logits (TP32 stack). Verify rounds S=4 per user, position advances by 1 per round (rows 1..3 are drafts that get overwritten next round),
row 0 of every user = the true teacher-forced token. Variants change ONLY rows 1..3 (and, for 'users', all rows of users 1..3): row-0 logits must be bit-equal across variants (causal attention).
Per-layer dumps (h_in, attn, h2, mlp) at DUMP_ROUNDS find the first layer/op where row 0 differs. Env: NL, NR (rounds), PAGES, TFP, OUT, DUMP_ROUNDS (comma), VARIANTS (comma)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import SpecDerive
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

R, S = 4, 4
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
NL = int(os.environ.get("NL", "4"))
PAGES = int(os.environ.get("PAGES", "8"))
NR = int(os.environ.get("NR", "12"))
DUMPR = [int(x) for x in os.environ.get("DUMP_ROUNDS", "0,6").split(",") if x]
VARS = os.environ.get("VARIANTS", "true,junk,zero,rep,users,junk2").split(",")
TAG = os.environ.get("QTAG", "rd")
OUT = os.environ.get("OUT", "/out")


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def make_tokens(tf, variant):
    """[R, NR + S] token matrix per (user, position index); row j of round c = column c + j."""
    n = NR + S
    base = torch.tensor([p[:n] for p in tf], dtype=torch.int64)  # real continuation
    g = torch.Generator().manual_seed({"junk": 11, "junk2": 23, "users": 5}.get(variant, 0))
    m = base.clone()
    if variant in ("junk", "junk2"):
        # rows 1..3 of each round: junk drafts. column c+j for j>=1 is a different token than the real one; row 0 of round c+1 = real token again, so build per-round below
        pass
    return base, g


def round_tokens(base, variant, c, g):
    """[R, S] tokens of round c."""
    t = base[:, c : c + S].clone()
    if variant == "true":
        return t
    if variant in ("junk", "junk2"):
        t[:, 1:] = torch.randint(1000, 100000, (R, S - 1), generator=g)
    elif variant == "zero":
        t[:, 1:] = 0
    elif variant == "rep":
        t[:, 1:] = t[:, :1]
    elif variant == "users":
        t[1:, :] = torch.randint(1000, 100000, (R - 1, S), generator=g)
    return t


def main():
    tf = torch.load(os.environ["TFP"])
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        t0 = time.time()
        ch.add_layers(0, NL, log=lambda s: None)
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), nl=NL, specs=[str(ly.spec.mlp) for ly in ch.layers])
        dec4 = Hy4PoolDecoder(ch, PAGES, mode="dense", S=S)
        rope = ch.layers[0].attn.rope
        sd = SpecDerive(md, dec4.ctx, rope, ch.layers[0].attn.indexer.idx_rope_first, S)
        dec4.attach_derive(sd)
        dec4.prepare_blocks()
        for ly, kv, kp in zip(ch.layers, dec4.kv_pools, dec4.key_pools):
            ly.attn.init_pool(dec4.ctx, kv, kp)
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))

        def run(variant):
            base, _ = make_tokens(tf, variant)
            g = torch.Generator().manual_seed({"junk": 11, "junk2": 23, "users": 5}.get(variant, 0))
            lg_out, dumps = [], {}
            for c in range(NR):
                t = round_tokens(base, variant, c, g)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        t.to(torch.int32).contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp
                    ),
                    dec4.tok,
                )
                p = upload(md, torch.tensor([float(c)] * R).reshape(R, 1, 1, 1), ttnn.float32, row_dim=0)
                dec4.ctx.mode = "dense"
                sd.derive(p)
                dec4.dump = [] if c in DUMPR else None
                hn, lg = dec4.step_verify()
                ttnn.synchronize_device(md)
                if dec4.dump is not None:
                    dumps[c] = dec4.dump
                    dec4.dump = None
                lg_out.append(
                    ttnn.to_torch(
                        lg, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                    )
                    .reshape(R, S, -1)
                    .float()
                )
                ttnn.deallocate(p)
            return lg_out, dumps

        results = {}
        for v in VARS:
            results[v] = run(v)
            res(event="ran", variant=v)
        ref_l, ref_d = results[VARS[0]]
        for v in VARS[1:]:
            l, d = results[v]
            # logits row 0 per round: users differing, max abs diff
            per_round = []
            for c in range(NR):
                diff = (l[c][:, 0] - ref_l[c][:, 0]).abs().amax(dim=-1)  # [R]
                per_round.append([round(float(x), 4) for x in diff])
            res(event="logit_row0", variant=v, vs=VARS[0], per_round_user_maxabs=per_round)
            for c in sorted(d):
                first = None
                tab = []
                for (nm, i, tv), (nm2, i2, tr) in zip(d[c], ref_d[c]):
                    # tensors [32 dev, ..., S rows, D]; row j index = dim -2
                    df = (tv - tr).abs()
                    rows = df.reshape(df.shape[0], -1, df.shape[-2], df.shape[-1]).amax(dim=(1, 3))  # [32, S]
                    r0 = float(rows[:, 0].max())
                    r0u = [float(rows[8 * u : 8 * u + 8, 0].max()) for u in range(R)]
                    tab.append((nm, i, r0, r0u, float(rows[:, 1:].max()) if rows.shape[1] > 1 else 0.0))
                    if first is None and r0 > 0:
                        first = (nm, i, r0)
                res(
                    event="first_rowdep",
                    variant=v,
                    round=c,
                    first=first,
                    table=[(a, b, round(x, 5), [round(y, 5) for y in yy], round(z, 4)) for a, b, x, yy, z in tab],
                )
        res(event="finished")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
