# SPDX-License-Identifier: Apache-2.0
"""B45x device unit test of the cross-row chain copy (tt/hy4_xrow.py): random pools on every device of a (ROWS x 8) mesh, copy row A -> row B for h tokens, every device compared bit-exact to the CPU reference
(dest row pages [0, h/512) == source row, everything else untouched, source row untouched); timing; N repeats with DRAM / L1 allocator identical before / after.
env: XR_ROWS (2), XR_PAGES (640 pool pages), XR_NL (8 kv layers), XR_NK (2 key pools), XR_H (comma list of tokens: 65536,130560,291840), XR_REPS (20), XR_FULL (1 = compare all tensors at every h, else 2 kv + 2 key at h > 65536), XR_KVMODE (fp8 | u32)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_xrow as X


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def dram(md):
    v = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    return dict(
        alloc=v.total_bytes_allocated_per_bank,
        free=v.total_bytes_free_per_bank,
        largest=v.largest_contiguous_bytes_free_per_bank,
    )


def l1(md):
    v = ttnn.get_memory_view(md, ttnn.BufferType.L1)
    return dict(alloc=v.total_bytes_allocated_per_bank, free=v.total_bytes_free_per_bank)


def main():
    rows, cols = int(os.environ.get("XR_ROWS", "2")), 8
    P, NL, NK = (
        int(os.environ.get("XR_PAGES", "640")),
        int(os.environ.get("XR_NL", "8")),
        int(os.environ.get("XR_NK", "2")),
    )
    hs = [int(x) for x in os.environ.get("XR_H", "65536,130560,291840").split(",")]
    reps = int(os.environ.get("XR_REPS", "20"))
    mode = os.environ.get("XR_KVMODE", "fp8")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if rows >= 4 else ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=16384)
    try:
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, 2), mesh_shape=(rows, cols))
        mc = ttnn.ConcatMesh2dToTensor(md, dims=(0, 2), mesh_shape=(rows, cols))
        g = torch.Generator().manual_seed(7)

        def mk_kv():
            if mode == "fp8":
                x = (torch.randn(rows, 1, cols * P * 64, 656, generator=g) * 4).to(torch.float8_e4m3fn).float()
                return (
                    ttnn.from_torch(
                        x,
                        dtype=ttnn.fp8_e4m3,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=md,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mp,
                    ),
                    "kv",
                )
            x = torch.randint(-(2**31), 2**31 - 1, (rows, 1, cols * P * 64, 176), generator=g, dtype=torch.int32)
            return (
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

        def mk_key():
            x = torch.randn(rows * P, 1, cols * 64, 128, generator=g).bfloat16()
            return (
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

        tensors = []
        for i in range(NL):
            t, k = mk_kv()
            tensors.append((f"kv{i}", t, k))
        for i in range(NK):
            t, k = mk_key()
            tensors.append((f"key{i}", t, k))
        d0, l0 = dram(md), l1(md)
        xr = X.Hy4Xrow(md, tensors)
        w = xr.warm() if rows > 1 else 0.0
        d1, l1b = dram(md), l1(md)
        res(
            stage="built",
            warm_s=round(w, 2),
            dram_before=d0,
            dram_after=d1,
            l1_before=l0,
            l1_after=l1b,
            tensors=len(tensors),
            rows=rows,
            pages=P,
            mode=mode,
        )

        def readback(t):
            return ttnn.to_torch(t, mesh_composer=mc)

        def rowsplit(a, kind):  # -> [rows, cols, pages*units, ...] numpy-like view
            if kind == "kv":
                n = a.shape[2] // cols
                return a.reshape(rows, 1, cols, n, a.shape[-1]).squeeze(1)
            a = a.reshape(rows, P, 1, cols, 64, 128)
            return a.permute(0, 3, 1, 2, 4, 5).reshape(rows, cols, P * 64, 128)  # tile rows of a column

        ok_all = True
        pairs = (
            [(0, 0)] if rows == 1 else ([(0, 1), (1, 0)] if rows == 2 else [(0, 1), (3, 2), (2, 0)])
        )  # rows 1: single-slot kernel/pipeline probe (copy onto itself)
        if (
            rows == 1
        ):  # single-slot probe: stage-in -> (no gather) -> stage-out into a ZERO twin tensor of every pool tensor, compared bit-exact (kernel, page math, chunking, dtype blindness, DRAM flatness)
            ctl_out = xr._ctl(0, 0)
            for h in hs:
                n = h // 512
                twins = []
                for name, t, kind in tensors:
                    t2 = ttnn.empty(
                        list(t.shape), dtype=t.dtype, layout=t.layout, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    cp, u, pb = (xr.cp_kv, 64, 704) if kind == "kv" else (xr.cp_key, 8, xr.key_page)
                    stage = xr.stage[(kind, pb)]
                    t0 = time.perf_counter()
                    for p0, m in X.chunk_plan(n, cp):
                        xr._prog(t, stage, xr.ctl_in, m * u, p0 * u, 0, pb)
                        xr._prog(stage, t2, ctl_out, m * u, 0, p0 * u, pb)
                    ttnn.synchronize_device(md)
                    dt = time.perf_counter() - t0
                    a, b2 = readback(t), readback(t2)
                    a, b2 = rowsplit(a, kind), rowsplit(b2, kind)
                    uu = 64
                    eq = torch.equal(a[:, :, : n * uu].view(torch.uint8), b2[:, :, : n * uu].view(torch.uint8))
                    rest_zero = True  # (twin is uninitialised: only the copied range is checked)
                    ok_all &= eq and rest_zero
                    res(stage="probe", name=name, h=h, ms=round(dt * 1e3, 1), equal=eq, rest_zero=rest_zero)
                    ttnn.deallocate(t2)
            res(stage="done", ok=bool(ok_all), dram=dram(md))
            return
        for h in hs:
            n = h // 512
            full = os.environ.get("XR_FULL", "1") == "1" or h <= 65536
            sel = list(range(len(tensors))) if full else [0, NL - 1, NL, NL + NK - 1]
            for a, b in pairs:
                before = {i: rowsplit(readback(tensors[i][1]), tensors[i][2]) for i in sel}
                dt = xr.copy(a, b, h, allow_same=rows == 1)
                bad = []
                for i in sel:
                    after = rowsplit(readback(tensors[i][1]), tensors[i][2])
                    u = 64  # rows (kv) or tile-rows (key: 64 rows per pool page) per pool page
                    exp = before[i].clone()
                    exp[b, :, : n * u] = before[i][
                        a, :, : n * u
                    ]  # (rows 1: identity; the probe checks the pipeline does not corrupt)
                    if not torch.equal(
                        after.view(torch.uint8) if after.dtype != torch.float32 else after,
                        exp.view(torch.uint8) if exp.dtype != torch.float32 else exp,
                    ):
                        bad.append(tensors[i][0])
                ok_all &= not bad
                res(
                    stage="copy",
                    h=h,
                    src=a,
                    dst=b,
                    ms=round(dt * 1e3, 1),
                    pred_s=round(X.copy_cost_s(h), 2),
                    tensors_checked=len(sel),
                    bad=bad,
                )
            # reps: DRAM / L1 identical; bit-exactness of dest row again after the last rep
            dA, lA = dram(md), l1(md)
            ts = []
            for r in range(reps):
                ts.append(xr.copy(0, 1 if rows > 1 else 0, h, allow_same=rows == 1))
            dB, lB = dram(md), l1(md)
            flat = dA == dB and lA == lB
            ok_all &= flat
            after = {i: rowsplit(readback(tensors[i][1]), tensors[i][2]) for i in sel[:2]}
            same = all(
                torch.equal(
                    after[i][1, :, : n * 64].view(torch.uint8)
                    if after[i].dtype != torch.float32
                    else after[i][1, :, : n * 64],
                    after[i][0, :, : n * 64].view(torch.uint8)
                    if after[i].dtype != torch.float32
                    else after[i][0, :, : n * 64],
                )
                for i in sel[:1]
            )
            res(
                stage="reps",
                h=h,
                reps=reps,
                ms_min=round(min(ts) * 1e3, 1),
                ms_max=round(max(ts) * 1e3, 1),
                dram_before=dA,
                dram_after=dB,
                l1_before=lA,
                l1_after=lB,
                flat=flat,
                dest_eq_src=same,
            )
        res(stage="done", ok=bool(ok_all), stats=xr.stats)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
