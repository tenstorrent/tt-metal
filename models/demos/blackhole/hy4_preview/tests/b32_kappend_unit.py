# SPDX-License-Identifier: Apache-2.0
"""B32 unit: key_append (one program) vs S stock paged_update_cache calls: bit-identical dequantised pool, varying calls x N, positions straddling tile / page boundaries, skipped rows (-1), different per-device positions."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_key_append as KA

PAGES = 8
S = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    md = open_mesh(trace=int(100e6))
    try:
        rep = ttnn.ReplicateTensorToMesh(md)
        g = torch.Generator().manual_seed(5)
        kcfg = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        mk = lambda t, dt, lay: ttnn.from_torch(
            t, dtype=dt, layout=lay, device=md, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        pt = mk(torch.arange(PAGES, dtype=torch.int32).reshape(1, PAGES), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        base = torch.randn(PAGES, 1, 64, 128, generator=g) * 3
        poolA = mk(base, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        poolB = mk(base, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        cases = [
            [29, 30, 31, 32],
            [60, 61, 62, 63],
            [62, 63, 64, 65],
            [5, 6, 7, 8],
            [0, 1, 2, 3],
            [28, -1, 30, 31],
            [-1, -1, -1, -1],
            [100, 101, 102, 103],
            [127, 128, 129, 130],
            [250, 251, 252, 253],
            [33, 34, 35, 36],
        ]
        ok = True
        for it, pos in enumerate(cases * int(os.environ.get("REPS", "3"))):
            k = torch.randn(1, 1, 32, 128, generator=g) * (1 + it % 4)
            kt = mk(k, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            idx = [mk(torch.full((1, 1), p, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for p in pos]
            for j in range(S):
                kj = ttnn.slice(kt, [0, 0, j, 0], [1, 1, j + 1, 128])
                kd = ttnn.to_memory_config(kj, kcfg)
                ttnn.experimental.paged_update_cache(poolA, kd, update_idxs_tensor=idx[j], page_table=pt)
            KA.key_append(kt, poolB, idx, pt, S)
            ttnn.synchronize_device(md)
            a = ttnn.to_torch(ttnn.get_device_tensors(poolA)[0]).float()
            b = ttnn.to_torch(ttnn.get_device_tensors(poolB)[0]).float()
            eq = bool(torch.equal(a, b))
            ok &= eq
            res(event="case", it=it, pos=pos, equal=eq, maxdiff=float((a - b).abs().max()))
        # timing
        kt = mk(torch.randn(1, 1, 32, 128, generator=g), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        idx = [mk(torch.full((1, 1), 100 + j, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for j in range(S)]

        def stock():
            for j in range(S):
                kj = ttnn.slice(kt, [0, 0, j, 0], [1, 1, j + 1, 128])
                kd = ttnn.to_memory_config(kj, kcfg)
                ttnn.experimental.paged_update_cache(poolA, kd, update_idxs_tensor=idx[j], page_table=pt)

        def new():
            KA.key_append(kt, poolB, idx, pt, S)

        for name, fn in (("stock", stock), ("new", new)):
            fn()
            ttnn.synchronize_device(md)
            v = []
            for _ in range(5):
                t = time.perf_counter()
                for _ in range(50):
                    fn()
                ttnn.synchronize_device(md)
                v.append((time.perf_counter() - t) / 50 * 1e6)
            v.sort()
            res(event="time_us_eager", name=name, us=round(v[2], 1))
        res(event="finished", all_equal=ok)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
