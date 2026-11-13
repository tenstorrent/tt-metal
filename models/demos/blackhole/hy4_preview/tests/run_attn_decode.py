# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S1 device test, DECODE of the Hy4 attention layer on the 4x8 mesh (users on the mesh rows) vs the oracle goldens. Run via run_attn.sh (SCRIPT=run_attn_decode.py).
Env: LAYER, KVFMT (bf16|scaled_fp8), USERS (1: one user replicated on the 4 rows | 4: user u on mesh row u, goldens golden, golden_v1..v3), POSITIONS (e.g. 512,1024,2048,2560), V1 (1: DSV4 key-split kernel), WDT.
The context state (KV rows + indexer keys of positions < P) is built by the device PREFILL path (512-token chunks), then one decode step is run for position P and compared with the golden row P
(h = golden attention input of token P, out = golden attention output of token P). Prints RES lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import idx_setmatch, load_sd, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
L = int(os.environ.get("LAYER", "1"))
KVFMT = os.environ.get("KVFMT", "bf16")
USERS = int(os.environ.get("USERS", "1"))
V1 = os.environ.get("V1", "0") == "1"
POS = [int(x) for x in os.environ.get("POSITIONS", "512,1024,2048,2560").split(",")]
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("WDT", "bf16")]
CHUNK = 512


def up_rows(md, tensors, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """per-user torch tensors (same shape) -> mesh tensor, user u on mesh row u (USERS == 1: replicated)."""
    if USERS == 1:
        return upload(md, tensors[0], dtype, layout)
    return upload(md, torch.cat(tensors, dim=0), dtype, layout, row_dim=0)


def snap(a):
    return (a.cache.rows, a.cache.length, a.indexer.keys if a.indexer else None, a.indexer.length if a.indexer else 0)


def restore(a, s):
    a.cache.rows, a.cache.length = s[0], s[1]
    if a.indexer:
        a.indexer.keys, a.indexer.length = s[2], s[3]


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    golds = []
    for u in range(USERS):
        d = GOLD if u == 0 else f"{GOLD}_v{u}"
        golds.append(torch.load(f"{d}/L{L}.pt"))
    conv = golds[0]["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=800_000_000)
    try:
        mc = mesh_4x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        attn = Hy4Attention(
            md,
            mc,
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt=KVFMT,
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=WDT,
        )
        shared = not attn.has_indexer
        built = 0
        res = []
        for P in sorted(POS):
            while built < P:  # prefill chunks up to P
                p0 = built
                h = up_rows(md, [gg["h"][p0 : p0 + CHUNK].reshape(1, 1, CHUNK, -1).to(torch.bfloat16) for gg in golds])
                ids_in = None
                if shared:
                    ids_in = up_rows(
                        md,
                        [gg["idx"][p0 : p0 + CHUNK].to(torch.int32).reshape(1, 1, CHUNK, -1) for gg in golds],
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                attn.prefill_chunk(h, p0, ids_in)
                built += CHUNK
            ttnn.synchronize_device(md)
            h = up_rows(md, [gg["h"][P].reshape(1, 1, 1, -1).to(torch.bfloat16) for gg in golds])
            ids_in = None
            if shared:
                ids_in = up_rows(
                    md,
                    [gg["idx"][P].to(torch.int32).reshape(1, 1, 1, -1) for gg in golds],
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
            s0 = snap(attn)
            attn.taps = {}
            t = time.perf_counter()
            out, ids = attn.decode_step(h, P, ids_in, v1=V1)
            ttnn.synchronize_device(md)
            first = time.perf_counter() - t
            for r in range(USERS):
                o = to_host(out, md, row=r).reshape(-1).float()
                ref = golds[r]["out"][P]
                rec = {
                    "layer": L,
                    "mode": "decode",
                    "kv": KVFMT,
                    "v1": V1,
                    "users": USERS,
                    "row": r,
                    "pos": P,
                    "pcc": round(pcc(o, ref), 6),
                    "max_abs": round((o - ref).abs().max().item(), 5),
                    "ref_absmax": round(ref.abs().max().item(), 4),
                }
                lat = to_host(attn.taps["lat"], md, row=r).reshape(-1).float()
                rec["pcc_lat"] = round(pcc(lat, golds[r]["lat"][P]), 6)
                if not shared and P + 1 > cfg.index_topk:
                    di = to_host(ids, md, row=r).reshape(1, -1).to(torch.int64)
                    di = torch.where(di >= 2**31, torch.full_like(di, -1), di)
                    ov, same = idx_setmatch(di, golds[r]["idx"][P : P + 1])
                    rec["idx_overlap"] = round(ov, 5)
                res.append(rec)
                print("RES " + json.dumps(rec), flush=True)
            if os.environ.get("TIMING", "0") == "1":
                from models.demos.blackhole.hy4_preview.tests.run_attn_device import trace_time

                attn.taps = None
                tabs0 = attn.rope_tabs(P, 1)
                dids = attn.dense_ids(P) if (attn.has_indexer and P + 1 <= cfg.index_topk) else None
                tt_ = trace_time(
                    md,
                    lambda: attn.decode_step(h, P, ids_in, v1=V1, append=False, tabs=tabs0, dense_ids=dids),
                    lambda: tabs0,
                )
                print(
                    "RES "
                    + json.dumps(
                        {"layer": L, "mode": "decode", "kv": KVFMT, "v1": V1, "users": USERS, "pos": P, **tt_}
                    ),
                    flush=True,
                )
            restore(attn, s0)
        print("SUMMARY " + json.dumps({"layer": L, "kv": KVFMT, "min_pcc": min(x["pcc"] for x in res)}), flush=True)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
