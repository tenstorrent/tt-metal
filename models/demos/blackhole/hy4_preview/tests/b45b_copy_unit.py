# SPDX-License-Identifier: Apache-2.0
"""B45b device unit test (1 slot = 1 mesh row of 8 chips): row-local page-list copy active <-> shadow pages (tt/hy4_blocks_copy.py), bit-exact against the CPU reference on every device,
non-contiguous page lists, copy-out then copy-in round trip, other pages untouched, DRAM/L1 allocator flat over repeats, timing per copy.
env: BC_ACT (active pages, 640), BC_SH (shadow pages, 128), BC_NL (8 kv), BC_NK (2 key), BC_REPS (20)."""

import json
import os
import random

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_blocks_copy as BCP


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def mem(md):
    d, l = ttnn.get_memory_view(md, ttnn.BufferType.DRAM), ttnn.get_memory_view(md, ttnn.BufferType.L1)
    return dict(
        dram_alloc=d.total_bytes_allocated_per_bank,
        dram_free=d.total_bytes_free_per_bank,
        l1_alloc=l.total_bytes_allocated_per_bank,
    )


def main():
    cols = 8
    A, S = int(os.environ.get("BC_ACT", "640")), int(os.environ.get("BC_SH", "128"))
    P = A + S
    NL, NK, reps = (
        int(os.environ.get("BC_NL", "8")),
        int(os.environ.get("BC_NK", "2")),
        int(os.environ.get("BC_REPS", "20")),
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, cols), l1_small_size=16384)
    ok_all = True
    try:
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, 2), mesh_shape=(1, cols))
        mc = ttnn.ConcatMesh2dToTensor(md, dims=(0, 2), mesh_shape=(1, cols))
        g = torch.Generator().manual_seed(11)
        tensors = []
        for i in range(NL):
            x = (torch.randn(1, 1, cols * P * 64, 656, generator=g) * 4).to(torch.float8_e4m3fn).float()
            t = ttnn.from_torch(
                x,
                dtype=ttnn.fp8_e4m3,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mp,
            )
            tensors.append((f"kv{i}", t, "kv"))
        for i in range(NK):
            x = torch.randn(P, 1, cols * 64, 128, generator=g).bfloat16()
            t = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mp,
            )
            tensors.append((f"key{i}", t, "key"))
        m0 = mem(md)
        bc = BCP.Hy4PageCopy(md, tensors)
        bc._ctl_row(0)
        res(stage="built", mem_before=m0, mem_after=mem(md))

        def rb(name, t):  # -> [cols, pages*units, ...] uint8 view
            a = ttnn.to_torch(t, mesh_composer=mc)
            if name.startswith("kv"):
                a = a.reshape(1, 1, cols, a.shape[2] // cols, a.shape[-1]).squeeze(1).squeeze(0)
            else:
                a = a.reshape(P, 1, cols, 64, 128).permute(2, 0, 1, 3, 4).reshape(cols, P * 64, 128)
            return a.contiguous().view(torch.uint8)

        def units(name):
            return 64  # kv: 64 rows per pool page; key: 64 tile-rows per page

        rnd = random.Random(3)
        for trial, n in enumerate((1, 7, 64, 200)):
            src = sorted(rnd.sample(range(A), n))
            dst = rnd.sample(range(A, P), n)
            pairs = list(zip(src, dst))  # copy-out (active -> shadow), shuffled shadow targets = non-contiguous runs
            before = {nm: rb(nm, t) for nm, t, _ in tensors}
            dt = bc.copy(0, pairs)
            bad = []
            for nm, t, _ in tensors:
                after = rb(nm, t)
                exp = before[nm].clone()
                u = units(nm)
                for s, d in pairs:
                    exp[:, d * u : (d + 1) * u] = before[nm][:, s * u : (s + 1) * u]
                if not torch.equal(after, exp):
                    bad.append(nm)
            # copy-in round trip: shadow -> NEW active pages (the ones just vacated), compare to the original bytes
            back = [(d, s) for s, d in pairs]
            bc.copy(0, back)
            for nm, t, _ in tensors:
                after = rb(nm, t)
                u = units(nm)
                for s, d in pairs:
                    if not torch.equal(after[:, s * u : (s + 1) * u], before[nm][:, s * u : (s + 1) * u]):
                        bad.append(nm + "_rt")
                        break
            ok_all &= not bad
            res(
                stage="copy",
                pages=n,
                runs=len(BCP.runs_of(pairs)),
                ms=round(dt * 1e3, 2),
                tensors=len(tensors),
                bad=bad,
            )
        m1 = mem(md)
        pairs = [(i, A + i) for i in range(min(S, 100))]
        ts = [bc.copy(0, pairs) for _ in range(reps)]
        m2 = mem(md)
        flat = m1 == m2
        ok_all &= flat
        res(
            stage="reps",
            pages=len(pairs),
            ms_min=round(min(ts) * 1e3, 2),
            ms_max=round(max(ts) * 1e3, 2),
            flat=flat,
            mem_before=m1,
            mem_after=m2,
        )
        res(stage="done", ok=bool(ok_all), stats=bc.stats, mem_final=mem(md))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
