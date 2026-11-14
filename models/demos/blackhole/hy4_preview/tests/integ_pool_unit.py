"""integ unit test (1x8 slot): hy4_pool kernels row_write / slice_copy (rows and key tiles) / id_filter against host references."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import ttnn
from dm_block import close_row, open_row  # noqa: E402

from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP  # noqa: E402


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def rep(md, t, dtype, layout):
    return ttnn.from_torch(
        t, device=md, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(md), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def shard0(md, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t, device=md, dtype=dtype, layout=layout, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def host_u8(md, t):
    """per-device bytes [8, rows, W] of an fp8 mesh tensor"""
    x = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
    return x.contiguous().view(torch.uint8)


def main():
    full, md = open_row()
    dp = ttnn.experimental.deepseek_prefill
    try:
        torch.manual_seed(1)
        P, NT = 4, 2048
        colid = shard0(md, torch.cat([torch.arange(8).reshape(8, 1, 1, 1).to(torch.int32), torch.zeros(8, 1, 1, 15, dtype=torch.int32)], dim=3), ttnn.uint32)
        # ---- natural packed rows [1,1,NT,656]
        lat = (torch.randn(1, 1, NT, 512) * 2).bfloat16()
        rope = torch.randn(1, 1, NT, 64).bfloat16()
        latd = rep(md, lat, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        roped = rep(md, rope, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f8, sc = dp.per_token_cast_to_fp8(latd, round_scale_to_power_of_two=True)
        nat = dp.pack_scaled_fp8_kv_cache(f8, sc, roped)
        nat_b = host_u8(md, nat).reshape(8, NT, -1)[0]  # device 0's natural rows (replicated)
        res(event="nat", shape=list(nat.shape), host=list(nat_b.shape))
        pool = ttnn.empty([1, 1, P * 64, 656], dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        HP.slice_copy(nat, pool, colid, units=64, page=HP.ROWB, npages=NT // 512)
        ttnn.synchronize_device(md)
        pb = host_u8(md, pool).reshape(8, P * 64, -1)
        ok = True
        for c in range(8):
            for p in range(NT // 512):
                a = pb[c, p * 64 : p * 64 + 64, :656]
                b = nat_b[p * 512 + 64 * c : p * 512 + 64 * c + 64, :656]
                ok &= bool((a == b).all())
        res(event="slice_copy_rows", ok=ok)
        # ---- decode append (row_write): token t appended
        pos = 1000
        _, col, loc = HP.locate(pos)
        ids = torch.full((8, 1, 1, 16), -1, dtype=torch.int32)
        ids[int(col), 0, 0, 0] = int(loc)
        idsd = shard0(md, ids, ttnn.int32)
        # src: one new packed row (take natural row 7 of a different block as 'new' data): build via quantiser on 32 padded rows
        lat2 = torch.zeros(1, 1, 32, 512).bfloat16()
        lat2[0, 0, 0] = (torch.randn(512) * 3).bfloat16()
        rope2 = torch.zeros(1, 1, 32, 64).bfloat16()
        rope2[0, 0, 0] = torch.randn(64).bfloat16()
        l2 = rep(md, lat2, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        r2 = rep(md, rope2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f2, s2 = dp.per_token_cast_to_fp8(l2, round_scale_to_power_of_two=True)
        new = dp.pack_scaled_fp8_kv_cache(f2, s2, r2)
        new_b = host_u8(md, new).reshape(8, 32, -1)[0, 0, :656]
        HP.row_write(new, idsd, pool, n=1)
        ttnn.synchronize_device(md)
        pb2 = host_u8(md, pool).reshape(8, P * 64, -1)
        okw = bool((pb2[int(col), int(loc), :656] == new_b).all())
        unchanged = True
        for c in range(8):
            for r in range(P * 64):
                if (c, r) != (int(col), int(loc)):
                    unchanged &= bool((pb2[c, r, :656] == pb[c, r, :656]).all())
        res(event="row_write", written=okw, others_unchanged=unchanged, col=int(col), loc=int(loc))
        # ---- key tiles (bf16 natural -> bfp8 -> column slices)
        keys = torch.randn(1, 1, NT, 128).bfloat16()
        kd = rep(md, keys, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kb = ttnn.typecast(kd, ttnn.bfloat8_b)
        kpool = ttnn.empty([1, 1, P * 64, 128], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        HP.slice_copy(kb, kpool, colid, units=8, page=HP.KEYTILE, npages=NT // 512)
        ttnn.synchronize_device(md)
        kn = ttnn.to_torch(ttnn.get_device_tensors(kb)[0]).float().reshape(NT, 128)
        kp = ttnn.to_torch(kpool, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().reshape(8, P * 64, 128)
        okk = True
        for c in range(8):
            for p in range(NT // 512):
                okk &= bool(torch.equal(kp[c, p * 64 : p * 64 + 64], kn[p * 512 + 64 * c : p * 512 + 64 * c + 64]))
        res(event="slice_copy_keys", ok=okk)
        # ---- id_filter
        S, K = 4, 2048
        g = torch.randint(0, 1 << 20, (S, 1, 1, K), dtype=torch.int64)
        g[0, 0, 0, 1500:] = HP.SENT
        g[1, 0, 0, 100:] = HP.SENT
        g32 = g.to(torch.int64).clamp(max=HP.SENT).to(torch.uint32 if hasattr(torch, "uint32") else torch.int64)
        gi = g.clone()
        gi[gi == HP.SENT] = -1
        gd = rep(md, gi.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ci = torch.zeros(8, 1, 1, 16, dtype=torch.int32)
        ci[:, 0, 0, 0] = torch.arange(8)
        ci[:, 0, 0, 1] = 1 << 19
        cid2 = shard0(md, ci, ttnn.uint32)
        out = HP.id_filter(gd, cid2, S)
        ttnn.synchronize_device(md)
        o = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(8, 8 * S, K).to(torch.int64) & 0xFFFFFFFF
        okf = True
        for c in range(8):
            for s in range(S):
                ids_ = g[s, 0, 0].numpy()
                v = ids_[(ids_ != HP.SENT) & (ids_ < (1 << 19))]
                _, cc, loc_ = HP.locate(v)
                ref = loc_[cc == c]
                for j in range(8):
                    row = o[c, j * S + s].numpy()
                    n = len(ref)
                    okf &= bool((row[:n] == ref).all() and (row[n:] == HP.SENT).all())
        res(event="id_filter", ok=okf)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
