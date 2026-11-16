"""B20: fused_kv_append vs the stock chain (per_token_cast_to_fp8 + to_layout + pack_scaled_fp8_kv_cache + row_write): byte equality of the pool over varying inputs/ids + traced timing."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP

NCALL = int(os.environ.get("NCALL", "6"))
S = int(os.environ.get("S", "1"))
NR = 256


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g = torch.Generator().manual_seed(7)
        rep = ttnn.ReplicateTensorToMesh(md)
        shd = lambda d: ttnn.ShardTensorToMesh(md, dim=d)
        T = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(
            t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        init = torch.randint(0, 256, (8, 1, NR, 704), generator=g, dtype=torch.int32).to(torch.uint8)
        mkpool = lambda: ttnn.from_torch(
            init,
            device=md,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=shd(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pa, pb = mkpool(), mkpool()
        dp = ttnn.experimental.deepseek_prefill
        allok = True
        for it in range(NCALL):
            mag = 10.0 ** float(torch.randint(-2, 3, (1,), generator=g))
            lat = T(
                (
                    torch.randn(1, 1, S, 512, generator=g)
                    * mag
                    * (1 + 3 * (torch.rand(1, 1, S, 512, generator=g) > 0.99))
                )
            )
            kpe = T(torch.randn(1, 1, S, 64, generator=g))
            idv = torch.randint(0, NR, (8, 1, 1, 16), generator=g, dtype=torch.int32)
            idv[torch.rand(8, 1, 1, 16, generator=g) < 0.3] = -1  # skip (0xFFFFFFFF)
            ids = ttnn.from_torch(
                idv,
                device=md,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shd(0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
            rope_rm = ttnn.to_layout(kpe, ttnn.ROW_MAJOR_LAYOUT)
            packed = dp.pack_scaled_fp8_kv_cache(f8, sc, rope_rm)
            HP.row_write(packed, ids, pa, n=S)
            FA.fused_kv_append(lat, kpe, ids, pb, S)
            ttnn.synchronize_device(md)
            a = ttnn.to_torch(pa, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
            b = ttnn.to_torch(pb, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
            a, b = a[..., :656], b[..., :656]  # bytes 656..703 are padding (the stock row_write copies garbage there)
            eq = bool(torch.equal(a, b))
            nd = int((a != b).sum())
            allok &= eq
            R(event="call", it=it, S=S, mag=mag, equal=eq, ndiff=nd)
            if not eq:
                d = (a != b).nonzero()
                R(event="first_diffs", idx=d[:8].tolist(), cols=sorted(set(int(x[-1]) for x in d[:200])))
        R(event="summary", all_equal=allok)
        if os.environ.get("TIME", "0") == "1":
            import time

            def stock():
                f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
                rr = ttnn.to_layout(kpe, ttnn.ROW_MAJOR_LAYOUT)
                pk = dp.pack_scaled_fp8_kv_cache(f8, sc, rr)
                HP.row_write(pk, ids, pa, n=S)
                for t in (f8, sc, rr, pk):
                    ttnn.deallocate(t)

            NI = 40
            for nm, fn in (("stock", stock), ("fused", lambda: FA.fused_kv_append(lat, kpe, ids, pb, S))):
                for _ in range(2):
                    fn()
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(NI):
                    fn()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ts = []
                for _ in range(6):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / NI * 1e6)
                R(event="time", which=nm, us_per_call=min(ts), all=ts)
        if os.environ.get("PROF", "0") == "1":

            def stock():
                f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
                rr = ttnn.to_layout(kpe, ttnn.ROW_MAJOR_LAYOUT)
                pk = dp.pack_scaled_fp8_kv_cache(f8, sc, rr)
                HP.row_write(pk, ids, pa, n=S)
                for t in (f8, sc, rr, pk):
                    ttnn.deallocate(t)

            for nm, fn in (("stock", stock), ("fused", lambda: FA.fused_kv_append(lat, kpe, ids, pb, S))):
                for _ in range(2):
                    fn()
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(3):
                    fn()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
                R(event="prof", which=nm)
    finally:
        ttnn.close_mesh_device(md)


main()
