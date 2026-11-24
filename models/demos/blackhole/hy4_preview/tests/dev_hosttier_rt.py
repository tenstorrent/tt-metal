# SPDX-License-Identifier: Apache-2.0
"""hy4fix device round trip of the host tier backend (tt/hy4_chain_backend.DeviceBackend.to_host / from_host): random uint32 pools on a 4x8 mesh, spill pages of row A to host, restore into row B at another page offset,
compare bit-exact (dest pages == source pages; every other page / row untouched); reports GB/s per direction (row aggregate) and the largest single call.
env: HT_PAGES (160 pool pages), HT_NL (6 kv), HT_NK (2 key), HT_N (140 pages moved), HT_SRC (2), HT_DST (1), HT_P0S (3), HT_P0D (7)"""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_xrow as X
from models.demos.blackhole.hy4_preview.tt.hy4_chain_backend import DeviceBackend


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    rows, cols = 4, 8
    P, NL, NK = (
        int(os.environ.get("HT_PAGES", "160")),
        int(os.environ.get("HT_NL", "6")),
        int(os.environ.get("HT_NK", "2")),
    )
    n, src, dst = (
        int(os.environ.get("HT_N", "140")),
        int(os.environ.get("HT_SRC", "2")),
        int(os.environ.get("HT_DST", "1")),
    )
    p0s, p0d = int(os.environ.get("HT_P0S", "3")), int(os.environ.get("HT_P0D", "7"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=16384)
    try:
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, 2), mesh_shape=(rows, cols))
        mc = ttnn.ConcatMesh2dToTensor(md, dims=(0, 2), mesh_shape=(rows, cols))
        g = torch.Generator().manual_seed(11)
        tensors = []
        for i in range(NL):
            x = torch.randint(-(2**31), 2**31 - 1, (rows, 1, cols * P * 64, 176), generator=g, dtype=torch.int32)
            tensors.append(
                (
                    f"kv{i}",
                    ttnn.from_torch(
                        x,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=md,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mp,
                    ),
                    "kv",
                )
            )
        for i in range(NK):
            x = torch.randn(rows * P, 1, cols * 64, 128, generator=g).bfloat16()
            tensors.append(
                (
                    f"key{i}",
                    ttnn.from_torch(
                        x,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=md,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mp,
                    ),
                    "key",
                )
            )
        xr = X.Hy4Xrow(md, tensors)
        xr.warm()
        be = DeviceBackend(md, tensors, xrow=xr)

        def snap():
            return {nm: ttnn.to_torch(t, mesh_composer=mc) for nm, t, _ in tensors}

        print("HEADROOM", be.can_copy(2, 1, 140), getattr(be, "last_headroom", None), flush=True)
        before = snap()
        t0 = time.perf_counter()
        G = be.tensor_groups(n, float(os.environ.get("HT_SLICE_MB", "64")))
        payload, sl_ms = [], []
        for g in range(max(G, 1)):
            t1 = time.perf_counter()
            payload += be.to_host(src, p0s, n, (g, G)) if G else be.to_host(src, p0s, n)
            sl_ms.append(round((time.perf_counter() - t1) * 1e3))
        d2h = time.perf_counter() - t0
        print("SLICES", G, "ms per slice max", max(sl_ms), "median", sorted(sl_ms)[len(sl_ms) // 2], flush=True)
        t0 = time.perf_counter()
        be.from_host(payload, dst, p0d, n)
        h2d = time.perf_counter() - t0
        after = snap()
        nbytes = n * 3_750_912 * NL / 79  # rough: only NL kv layers here
        bad = []
        for nm, t, kind in tensors:
            b, a = before[nm], after[nm]
            if kind == "kv":  # [rows,1,cols*P*64,176]; per device slice of P*64 rows; page = 64 rows
                b = b.reshape(rows, 1, cols, P * 64, 176).clone()
                a = a.reshape(rows, 1, cols, P * 64, 176)
                exp = b.clone()
                exp[dst, :, :, p0d * 64 : (p0d + n) * 64] = b[src, :, :, p0s * 64 : (p0s + n) * 64]
            else:  # [rows*P,1,cols*64,128]
                b = b.reshape(rows, P, 1, cols * 64, 128).clone()
                a = a.reshape(rows, P, 1, cols * 64, 128)
                exp = b.clone()
                exp[dst, p0d : p0d + n] = b[src, p0s : p0s + n]
            if not torch.equal(exp, a):
                bad.append(nm)
                if kind == "kv" and nm == "kv0":
                    diff = (exp != a).reshape(rows, cols, P, 64 * (176 if kind == "kv" else 1)).any(-1)
                    for r_ in range(rows):
                        for c_ in range(cols):
                            pg = diff[r_, c_].nonzero().flatten().tolist()
                            if pg:
                                print("DIFF row", r_, "col", c_, "pages", pg[:6], "n", len(pg), flush=True)
                    # does dst region equal ANY other (row,col) source?
                    reg = a[dst, :, :, p0d * 64 : (p0d + 1) * 64]
                    for r_ in range(rows):
                        for c_ in range(cols):
                            if torch.equal(reg[:, c_], b[r_, :, c_, p0s * 64 : (p0s + 1) * 64]):
                                print("DST col", c_, "page0 == src row", r_, "col", c_, flush=True)
        host_bytes = sum(blk.numel() * 4 for _, _, _, blk in payload)
        res(
            ok=not bad,
            bad=bad,
            d2h_s=round(d2h, 3),
            h2d_s=round(h2d, 3),
            host_gb=round(host_bytes / 1e9, 3),
            d2h_gbps=round(host_bytes / 1e9 / d2h, 3),
            h2d_gbps=round(host_bytes / 1e9 / h2d, 3),
            tensors=len(tensors),
            n_pages=n,
        )
        if os.environ.get("HT_OOM"):
            keep = []
            mv = lambda: ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
            for mb in (256, 64, 16, 4, 1):
                while mv().largest_contiguous_bytes_free_per_bank > 13_000_000 + mb * 0.52e6:
                    keep.append(
                        ttnn.zeros([1, 1, mb * 1000, 1024], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
                    )
            v = mv()
            print(
                "OOM_STATE free",
                v.total_bytes_free_per_bank,
                "largest",
                v.largest_contiguous_bytes_free_per_bank,
                "alloc",
                v.total_bytes_allocated_per_bank,
                flush=True,
            )
            print("OOM_CAN_COPY", be.can_copy(2, 1, 140), getattr(be, "last_headroom", None), flush=True)
            try:
                be.copy_pages(2, 3, 1, 7, 140)
                print("OOM_COPY_RESULT ok (101 tensors x 2 chunks with ~12 MB/bank free)", flush=True)
            except Exception as e_:
                print("OOM_COPY_RESULT FAILED", repr(e_)[:120], flush=True)
        print("HT_RT_OK" if not bad else "HT_RT_FAIL", flush=True)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
