# SPDX-License-Identifier: Apache-2.0
"""Mesh-level (4x8) chunked prefill + decode test of MiMo attention, SWA and global layer, vs torch fp32 reference with random weights of real shapes.
Run: ATTN_KINDS=swa,ga python test_attention_mesh.py   (results: JSON lines 'ATTN_RESULT ...'; PCC gate 0.999)
Layout: 4 mesh rows = 4 independent user groups (different data per row), 8 columns = TP. U users per row are prefilled (one per call), decode runs 32 slots/row."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import pcc, ref_attention  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.attention import (  # noqa: E402
    BLOCK,
    DQK,
    DV,
    GA,
    HIDDEN,
    NQ,
    SWA,
    MiMoAttention,
    convert_weights,
)

CH = int(os.environ.get("ATTN_CH", "1024"))
NCH = int(os.environ.get("ATTN_NCH", "3"))
U = int(os.environ.get("ATTN_U", "2"))
ND = int(os.environ.get("ATTN_ND", "3"))
SINKM = float(os.environ.get("ATTN_SINKM", "3"))
P = CH * NCH
RAGGED = os.environ.get("ATTN_RAGGED", "0") == "1"  # user u gets NCH-u chunks
WD = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[os.environ.get("ATTN_WDTYPE", "bf8")]
LENS = [CH * max(1, NCH - u) if RAGGED else P for u in range(U)]
MAXPOS = ((P + ND + 255) // 256) * 256 + 256
OUT = os.environ.get("ATTN_OUT", "/tmp/attn_mesh.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("ATTN_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def main():
    kinds = os.environ.get("ATTN_KINDS", "swa,ga").split(",")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
    try:
        rows = 4
        g = torch.Generator().manual_seed(0)
        for kind in kinds:
            cfg = SWA if kind == "swa" else GA
            q_w = torch.randn(NQ * DQK, HIDDEN, generator=g) / 64
            k_w = torch.randn(cfg.nkv * DQK, HIDDEN, generator=g) / 64
            v_w = torch.randn(cfg.nkv * DV, HIDDEN, generator=g) / 64
            o_w = torch.randn(HIDDEN, NQ * DV, generator=g) / 90
            sink = ((torch.rand(NQ, generator=g) * 0.58 + 0.56) * SINKM).bfloat16() if cfg.sink else None
            hw = convert_weights(cfg, q_w, k_w, v_w, o_w, sink)
            ring = CH // BLOCK + 8
            nb = ring if cfg.window else MAXPOS // BLOCK
            perm = torch.randperm(32 * nb, generator=g).reshape(32, nb).to(torch.int32)
            t0 = time.time()
            m = MiMoAttention(mesh, cfg, hw, perm, MAXPOS, ring_blocks=ring, wdtype=WD)
            rec("build", kind=kind, secs=round(time.time() - t0, 1), ring=ring if cfg.window else None)
            # data: x[row, user, T, 4096]
            T = P + ND
            x = (torch.randn(rows, U, T, HIDDEN, generator=g)).bfloat16()
            t0 = time.time()
            ref = torch.zeros(rows, U, T, HIDDEN)
            for r in range(rows):
                for u in range(U):
                    ref[r, u, : LENS[u] + ND] = ref_attention(
                        x[r, u, : LENS[u] + ND], q_w, k_w, v_w, o_w, sink, cfg.nkv, cfg.theta, cfg.window
                    )
            rec("ref_done", kind=kind, secs=round(time.time() - t0, 1))
            to_mesh = lambda t, d=ttnn.bfloat16: ttnn.from_torch(
                t,
                device=mesh,
                dtype=d,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None)),
            )
            gather = lambda t: ttnn.to_torch(
                t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
            ).float()
            # ---- chunked prefill, user by user
            pre = torch.zeros(rows, U, P, HIDDEN)
            colspread = 0.0
            t0 = time.time()
            for u in range(U):
                for c in range(LENS[u] // CH):
                    xin = x[:, u, c * CH : (c + 1) * CH].reshape(rows, 1, CH, HIDDEN)
                    o = m.prefill_chunk(to_mesh(xin), c * CH, u)
                    h = gather(o)  # [rows, 8, CH, 4096]
                    colspread = max(colspread, float((h - h[:, :1]).abs().max()))
                    pre[:, u, c * CH : (c + 1) * CH] = h[:, 0]
                    ttnn.deallocate(o)
            rec("prefill_time", kind=kind, secs=round(time.time() - t0, 1))
            res = {
                f"r{r}u{u}": round(pcc(pre[r, u, : LENS[u]], ref[r, u, : LENS[u]]), 5)
                for r in range(rows)
                for u in range(U)
            }
            per_chunk = {
                f"c{c}": round(
                    min(
                        pcc(pre[r, u, c * CH : (c + 1) * CH], ref[r, u, c * CH : (c + 1) * CH])
                        for r in range(rows)
                        for u in range(U)
                        if (c + 1) * CH <= LENS[u]
                    ),
                    5,
                )
                for c in range(NCH)
            }
            rec(
                "prefill_pcc",
                kind=kind,
                min=min(res.values()),
                per_user=res,
                min_per_chunk=per_chunk,
                col_spread_absmax=colspread,
            )
            # ---- decode: slots 0..U-1 continue the prefilled users, the other slots hold zero caches
            xd_all = (torch.randn(rows, 32, ND, HIDDEN, generator=g)).bfloat16()
            for u in range(U):
                xd_all[:, u] = x[:, u, LENS[u] : LENS[u] + ND]
            dres = []
            for step in range(ND):
                pos = [(LENS[u] if u < U else P) + step for u in range(32)]
                cur, idx = m.decode_inputs(pos)
                xin = xd_all[:, :, step].reshape(rows, 1, 32, HIDDEN)
                o = m.decode(to_mesh(xin), cur, idx)
                h = gather(o)  # [rows, 8, 32, 4096]
                cs = float((h - h[:, :1]).abs().max())
                dd = {
                    f"r{r}u{u}": round(pcc(h[r, 0, u], ref[r, u, LENS[u] + step]), 5)
                    for r in range(rows)
                    for u in range(U)
                }
                if (
                    step == ND - 1
                ):  # determinism: repeat the step 3x (cache write is idempotent), output must be bit-identical
                    same = True
                    for _ in range(3):
                        o2 = m.decode(to_mesh(xin), cur, idx)
                        same &= bool(torch.equal(gather(o2), h))
                        ttnn.deallocate(o2)
                    rec("decode_determinism_3x", kind=kind, bit_identical=same)
                dres.append(dict(step=step, min=min(dd.values()), col_spread=cs, per_user=dd))
                ttnn.deallocate(o)
            rec("decode_pcc", kind=kind, min=min(d["min"] for d in dres), steps=dres)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
