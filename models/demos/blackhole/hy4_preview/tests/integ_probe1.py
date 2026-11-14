"""integ probe 1 (1x8 slot): (A) stock T=1 SCALED_FP8 row quantise+pack ops, (B) paged_update_cache on a bfp8 TILE key pool with skip (-1) on non-owner columns."""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import ttnn
from dk_dev import pcc  # noqa: E402
from dm_block import close_row, open_row  # noqa: E402

from models.demos.blackhole.hy4_preview.tt.hy4_kv_cache import pack_scaled_fp8_host  # noqa: E402


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def to_mesh(md, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, shard=True):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0) if shard else ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    full, md = open_row()
    dp = ttnn.experimental.deepseek_prefill
    try:
        torch.manual_seed(0)
        # ---- A: quantise + pack, 32-row padded input (row 0 real)
        lat = torch.randn(1, 1, 32, 512).bfloat16() * 3
        rope = torch.randn(1, 1, 32, 64).bfloat16()
        latd = to_mesh(md, lat.repeat(8, 1, 1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        roped = to_mesh(md, rope.repeat(8, 1, 1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)

        def quant():
            f8, sc = dp.per_token_cast_to_fp8(latd, round_scale_to_power_of_two=True)
            rr = ttnn.to_layout(roped, ttnn.ROW_MAJOR_LAYOUT)
            return dp.pack_scaled_fp8_kv_cache(f8, sc, rr)

        packed = quant()
        ttnn.synchronize_device(md)
        p = ttnn.to_torch(packed, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0].reshape(32, -1)
        res(event="A_shape", shape=list(packed.shape), dtype=str(packed.dtype), host_shape=list(p.shape))
        ref_p, ref_deq = pack_scaled_fp8_host(torch.cat([lat.reshape(32, 512), rope.reshape(32, 64)], dim=1))
        pu = p.contiguous().view(torch.uint8).reshape(32, -1); print("pu", pu.shape, flush=True)
        res(
            event="A_cmp",
            scales_equal=bool((pu[:, 512:528] == ref_p[:, 512:528]).all()),
            rope_equal=bool((pu[:, 528:] == ref_p[:, 528:]).all()),
            lat_bytes_equal_frac=float((pu[:, :512] == ref_p[:, :512]).float().mean()),
        )
        from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us  # noqa: F401

        n, tid = 10, None
        quant()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(n):
            quant()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        res(event="A_us_per_call_traced", us=round((time.perf_counter() - t0) / 5 / n * 1e6, 1))

        # ---- B: paged_update_cache bfp8
        P = 8
        for dt in (ttnn.bfloat8_b, ttnn.bfloat16):
            try:
                cache = ttnn.from_torch(
                    torch.zeros(8, P, 1, 64, 128).reshape(8 * P, 1, 64, 128),
                    device=md,
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                key = torch.randn(1, 1, 32, 128).bfloat16()
                key[:, :, 1:] = 0
                kd = ttnn.from_torch(
                    key.repeat(8, 1, 1, 1),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                kcfg = ttnn.create_sharded_memory_config(
                    shape=(32, 128),
                    core_grid=ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(8, 8), row_wise=True),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                kds = ttnn.to_memory_config(kd, kcfg)
                idx = torch.full((8, 1), -1, dtype=torch.int32)
                idx[3, 0] = 70  # column 3 owns: local row 70 = page 1, slot 6
                idxd = ttnn.from_torch(
                    idx,
                    device=md,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                pt = ttnn.from_torch(
                    torch.arange(P, dtype=torch.int32).reshape(1, P).repeat(8, 1),
                    device=md,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.experimental.paged_update_cache(cache, kds, update_idxs_tensor=idxd, page_table=pt)
                ttnn.synchronize_device(md)
                out = [ttnn.to_torch(t).reshape(P * 64, 128).float() for t in ttnn.get_device_tensors(cache)]
                nz = [int((o.abs().sum(1) > 0).sum()) for o in out]
                row70 = out[3][70]
                res(event="B_" + str(dt), nonzero_rows_per_col=nz, row70_pcc=pcc(row70, key[0, 0, 0].float()), nz_in_col3=[int(i) for i in (out[3].abs().sum(1) > 0).nonzero().flatten()][:4])
            except Exception as e:  # noqa: BLE001
                res(event="B_" + str(dt), error=repr(e)[:600])
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
