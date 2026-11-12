# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-row copy with O(layers) host ops instead of O(layers x blocks) (DSV41_XROW_OPS=1, default off).

``slot_copy`` (paged_kernels/slot_copy.cpp) copies N blocks between two bf16 TILE tensors in ONE launch from a slot table; ``read_key_blocks`` gathers the key store slots of the donor row into
one contiguous tensor per layer (one from_device per column and layer), ``write_key_blocks`` uploads one stage tensor per layer (N blocks) and scatters it into the receiving row's store slots.
"""
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.apc_paged_store import IDIM, SKIP, TILE_BYTES, _acc, _one


def _tab(md, per_row_src, per_row_dst, nt):
    """host list per mesh row -> device uint32 RM [rows, 2 * nt] (src ids, dst ids), sharded over the rows"""
    rows, cols = tuple(md.shape)
    t = torch.full((rows, 2 * nt), SKIP, dtype=torch.int64)
    for r in range(rows):
        s, d = per_row_src.get(r), per_row_dst.get(r)
        if s is not None:
            t[r, : len(s)] = torch.as_tensor(s, dtype=torch.int64)
        if d is not None:
            t[r, nt : nt + len(d)] = torch.as_tensor(d, dtype=torch.int64)
    t = torch.where(t < 0, torch.full_like(t, SKIP), t)
    return ttnn.from_torch(
        t.to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )


def slot_copy(tab, src, dst, n, nt, ncores=16):
    """dst[tab[nt + i]] = src[tab[i]] for i < n (blocks of NP = R / 32 * 4 tile pages), in place on dst"""
    R, Dd = int(src.shape[2]), int(src.shape[3])
    assert int(dst.shape[2]) == R and int(dst.shape[3]) == Dd and R % 32 == 0 and Dd % 32 == 0
    NP = R // 32 * (Dd // 32)  # tile pages per block (keys: R/32*4; halos [K*NL,1,128,512]: 64 per (slot, layer) block)
    ncores = min(ncores, n)
    ct = [NP, TILE_BYTES, 0, nt, n, int(src.shape[0]), int(dst.shape[0]), ncores] + _acc(tab) + _acc(src) + _acc(dst)
    cores = [ttnn.CoreCoord(c % 8, c // 8) for c in range(ncores)]
    _one(
        cores,
        2 * nt * 4 + NP * TILE_BYTES,
        ct,
        [[c] for c in range(ncores)],
        "slot_copy.cpp",
        [tab.buffer_address(), src.buffer_address(), dst.buffer_address()],
        [tab, src, dst],
        0x9C7,
    )


def _pad32(n):
    return (n + 31) // 32 * 32


def read_key_blocks(md, store, row, slots, group=256):
    """store {layer: [K,1,R,128] tile mesh tensor}; the donor row ``row``'s slots ``slots`` (list of int) -> {layer: torch [N, cols, R, 128]} (column c's block data)"""
    rows, cols = tuple(md.shape)
    N = len(slots)
    out = {lid: [] for lid in store}
    for g0 in range(0, N, group):
        sl = slots[g0 : g0 + group]
        n = len(sl)
        nt = _pad32(n)
        tab = _tab(md, {row: sl}, {row: list(range(n))}, nt)
        pend = []
        for lid, st in store.items():
            R = int(st.shape[2])
            o = ttnn.zeros([n, 1, R, IDIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
            slot_copy(tab, st, o, n, nt)
            devs = ttnn.get_device_tensors(o)
            pend.append((lid, o, [ttnn.from_device(devs[row * cols + c], blocking=False) for c in range(cols)]))
        ttnn.synchronize_device(md)
        for lid, o, hs in pend:
            out[lid].append(torch.stack([ttnn.to_torch(h).reshape(n, -1, IDIM) for h in hs], dim=1))
            ttnn.deallocate(o)
        ttnn.deallocate(tab)
    return {lid: (v[0] if len(v) == 1 else torch.cat(v)) for lid, v in out.items()}


def write_key_blocks(md, store, row, slots, keys, group=256):
    """keys {layer: torch [N, cols, R, 128]} -> store slots ``slots`` (int, < 0 = skip) of mesh row ``row`` (all columns), in place; synchronised on return"""
    rows, cols = tuple(md.shape)
    N = len(slots)
    mapper = ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(rows, cols))
    for g0 in range(0, N, group):
        sl = slots[g0 : g0 + group]
        n = len(sl)
        nt = _pad32(n)
        tab = _tab(md, {r: list(range(n)) for r in range(rows)}, {row: sl}, nt)
        keep = [tab]
        for lid, st in store.items():
            stage = ttnn.from_torch(
                keys[lid][g0 : g0 + n].contiguous(),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            slot_copy(tab, stage, st, n, nt)
            keep.append(stage)
        ttnn.synchronize_device(md)
        for x in keep:
            ttnn.deallocate(x)
