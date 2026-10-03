# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of the two custom paged-KV ops (tt/paged_ops.py) against a host model, single-op, fast:
* paged_kv_step: ring row write, compressed latent write through the page table, sparse index rows (ring rows, all / selected entries, sentinel tail)
* paged_scatter_rows: pool[ids[i]] = src[i]"""

import random

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

DP = {"l1_small_size": 16384}


def dev(md, t, dtype, layout, mapper=None):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper if mapper is not None else ttnn.ReplicateTensorToMesh(md),
    )


def dev0(md, t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@pytest.mark.parametrize("ratio,src,nq", [(0, 2, 1), (2, 2, 1), (1, 20, 1), (2, 14, 1), (1, 20, 3)])
@pytest.mark.parametrize("positions", [[0, 5, 127, 128], [300, 1000, 1500, 2047]])
@pytest.mark.parametrize("pdt", ["bf16", "fp8"])
@torch.no_grad()
def test_paged_kv_step(mesh_device, ratio, src, nq, positions, pdt):
    md = mesh_device
    T = 4
    U = T  # users per mesh row
    ROWS = U * nq
    ring = 128 if nq == 1 else 160
    pages = 20
    pool_t = P.PagedKVPool(md, U, num_pages=U * pages, n_ring_layers=2, max_ctx=pages * 128, ring_rows=ring)
    for a in pool_t.allocs:
        random.Random(3).shuffle(a._free)
    for b in range(pool_t.B):
        pool_t.admit(b, 2048 + 32)
    pool_t.sync_page_table()
    # pool content: random everywhere (so a wrong write / wrong index would be seen)
    R_ = pool_t.total_rows
    base_pool = torch.randn(1, 1, R_, 512).to(torch.bfloat16)
    fp8 = pdt == "fp8"
    q8 = lambda t: t.to(torch.float8_e4m3fn).float() if fp8 else t.float()  # host model of the pool's storage format
    pool = dev(
        md, base_pool.float() if fp8 else base_pool, ttnn.fp8_e4m3 if fp8 else ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    )
    kv_in = (
        torch.randn(1, ROWS if nq > 1 else U, 32, 512).to(torch.bfloat16)
        if nq == 1
        else torch.randn(1, 1, ROWS, 512).to(torch.bfloat16)
    )
    lat_in = torch.randn(1, 1, ROWS, 512).to(torch.bfloat16)
    kv_mode = 0 if nq == 1 else 1
    kv = dev(md, kv_in, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    lat = dev(md, lat_in, ttnn.bfloat16, ttnn.TILE_LAYOUT) if ratio else None
    pos_h = torch.tensor([positions[u] + j for u in range(U) for j in range(nq)], dtype=torch.int32)
    pos = dev(md, pos_h, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    # table of this device (mesh row 0: users 0..U-1 of that row)
    pt_dev = pool_t.page_table
    pt_h = pool_t.table_host()[:U].long()
    topk = 128 + 512 if ratio else 128
    ids_t = None
    if ratio:
        rows_ids = []
        for p in pos_h.tolist():
            N = (p + 1) // ratio
            sel = torch.randperm(N)[:512].sort().values if N > 512 else torch.arange(0)
            row = torch.full((512,), 0xFFFFFFFF, dtype=torch.long)
            row[: len(sel)] = sel
            rows_ids.append(row)
        ids_h = torch.stack(rows_ids)
        ids_t = dev(
            md, ids_h.to(torch.int64).to(torch.int32).reshape(ROWS, 1, 1, 512), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
    ring_base = pool_t.ring_base(1)
    idx = P.paged_kv_step(
        pool,
        kv,
        lat,
        pos,
        pt_dev,
        ids_t,
        ring_base=ring_base,
        layer_key=7,
        ratio=ratio,
        src_off=P.SRC_OFF[src],
        topk_out=topk,
        ring_rows=ring,
        kv_mode=kv_mode,
        nq=nq,
    )
    got_idx = dev0(md, idx).reshape(ROWS, topk).long()
    got_pool = (
        dev0(md, ttnn.typecast(pool, ttnn.bfloat16) if fp8 else pool).reshape(R_, 512).float()
    )  # fp8 cannot be read back directly
    exp_pool = q8(base_pool.reshape(R_, 512))
    from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageLayout

    lay = PageLayout()
    for i in range(ROWS):
        u, p = i // nq, int(pos_h[i])
        # kv row of query i
        kv_row = kv_in[0, u, 0] if nq == 1 else kv_in[0, 0, i]
        exp_pool[ring_base + u * ring + p % ring] = q8(kv_row)
        if ratio:
            j = p // ratio
            r = lay.phys_rows(pt_h[u], P.SOURCES.index(src), torch.tensor([j]))[0]
            exp_pool[r] = q8(lat_in[0, 0, i])
    assert torch.equal(got_pool, exp_pool), "pool rows differ from the host model"
    for i in range(ROWS):
        u, p = i // nq, int(pos_h[i])
        nr = min(p + 1, 128)
        exp = [ring_base + u * ring + (q % ring) for q in range(p - nr + 1, p + 1)]
        if ratio:
            N = (p + 1) // ratio
            ent = torch.arange(N) if N <= 512 else ids_h[i][ids_h[i] < N]
            exp += lay.phys_rows(pt_h[u], P.SOURCES.index(src), ent).tolist()
        exp += [0xFFFFFFFF] * (topk - len(exp))
        assert got_idx[i].tolist() == exp, f"row {i} pos {p}: indices differ"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@torch.no_grad()
def test_scatter_rows(mesh_device):
    md = mesh_device
    R_, N = 5000, 300
    base = torch.randn(1, 1, R_, 512).to(torch.bfloat16)
    pool = dev(md, base, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    src = torch.randn(N, 512).to(torch.bfloat16)
    ids = torch.randperm(R_ - 64)[:N]  # leaves room for the base offset
    ids_skip = ids.clone()
    ids_skip[::7] = 0xFFFFFFFF
    P.paged_scatter_rows(
        pool,
        dev(md, src, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        dev(md, ids_skip.to(torch.int32).reshape(1, N), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        base_offset=7,
    )
    got = dev0(md, pool).reshape(R_, 512)
    exp = base.reshape(R_, 512).clone()
    keep = ids_skip != 0xFFFFFFFF
    exp[ids_skip[keep] + 7] = src[keep]
    assert torch.equal(got, exp)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@pytest.mark.parametrize("backend", ["matmul", "fused"])
@torch.no_grad()
def test_indexer_ragged_valid_lengths(mesh_device, backend):
    """Decode indexer with PER-USER valid entry counts (ragged batch): every selected id is < the user's own N and the selected score sum equals the
    top-512 of that user's valid scores (computed from the device scores)."""
    from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer

    md = mesh_device
    rows = md.shape[0]
    U, n_alloc = 4, 2080
    B = rows * U
    gen = torch.Generator().manual_seed(3)
    w = {
        "wq_b": torch.randn(4096, 1280, generator=gen).to(torch.bfloat16) * 0.05,
        "weights_proj": torch.randn(32, 5120, generator=gen).to(torch.bfloat16) * 0.05,
    }
    idx = DSV41DecodeIndexer(
        md,
        w,
        torch.polar(torch.ones(4096, 32), torch.zeros(4096, 32)),
        users_per_row=U,
        n_alloc=n_alloc,
        ratio=1,
        backend=backend,
    )
    idx.load_keys(torch.randn(B, 2048, 128, generator=gen) * 0.7)
    pos = torch.tensor([700, 1200, 1999, 2047])
    n_valid = pos + 1  # ratio 1
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(
        t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    st = {
        "C": up(torch.ones(U, 1, 1, 128).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "S": up(torch.zeros(U, 1, 1, 128).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "valid": up(
            torch.full((1, 1, 1, 1), int(n_valid.max()), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        ),
        "nvalid": up(n_valid.float().reshape(U, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT),
    }
    x = up((torch.randn(1, 1, U, 5120, generator=gen) * 0.5).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    qr = up((torch.randn(1, 1, U, 1280, generator=gen)).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    ids, scores = idx.forward(x, qr, st, return_scores=True)
    got = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(U, -1).long()
    if backend == "fused":
        sc = torch.stack([ttnn.to_torch(ttnn.get_device_tensors(s_)[0]).reshape(32, -1)[0] for s_ in scores]).float()
    else:
        sc = ttnn.to_torch(ttnn.get_device_tensors(scores)[0]).reshape(U, -1).float()
    for u in range(U):
        n = int(n_valid[u])
        sel = got[u][got[u] != 0xFFFFFFFF]
        assert len(sel) == 512 and int(sel.max()) < n, f"user {u}: ids must be < {n} (max {int(sel.max())})"
        gold = sc[u, :n].topk(512).values.sum()
        assert (
            abs(float(sc[u, sel].sum() - gold)) <= 1e-2 * abs(float(gold)) + 1e-3
        ), f"user {u}: selection is not the top-512 of the valid scores"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@pytest.mark.parametrize("slab_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@torch.no_grad()
def test_write_keys(mesh_device, slab_dtype):
    """DSV41DecodeIndexer.write_keys: prefill write of index keys into the TILE slab (entry 0 via fill_cache, entry offset via slice_write)."""
    from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer

    md = mesh_device
    rows = md.shape[0]
    U, n_alloc = 4, 256
    B = rows * U
    gen = torch.Generator().manual_seed(5)
    w = {"wq_b": torch.zeros(4096, 1280).to(torch.bfloat16), "weights_proj": torch.zeros(32, 5120).to(torch.bfloat16)}
    idx = DSV41DecodeIndexer(
        md,
        w,
        torch.polar(torch.ones(300, 32), torch.zeros(300, 32)),
        users_per_row=U,
        n_alloc=n_alloc,
        ratio=2,
        key_dtype=slab_dtype,
    )
    idx.load_keys(torch.zeros(B, 0, 128))
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, md.shape[1]))
    mk = lambda t: ttnn.from_torch(
        t,
        device=md,
        dtype=slab_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    k0 = (torch.randn(rows, 1, 64, 128, generator=gen)).to(torch.bfloat16)  # user 1 of every mesh row: entries 0..63
    k1 = (torch.randn(rows, 1, 96, 128, generator=gen)).to(torch.bfloat16)  # user 1: entries 64..159 (a second chunk)
    idx.write_keys(1, mk(k0), 0)
    idx.write_keys(1, mk(k1), 64)
    ttnn.synchronize_device(md)
    got = ttnn.to_torch(ttnn.get_device_tensors(idx.k_cache)[0]).float().reshape(U, n_alloc, 128)  # mesh row 0
    tol = 0.0 if slab_dtype == ttnn.bfloat16 else 0.1
    exp = torch.cat([k0[0, 0], k1[0, 0]]).float()
    err = (got[1, :160] - exp).abs().max()
    assert float(err) <= tol + (0.0 if tol == 0 else 0.0), f"max abs error {float(err)}"
    assert (
        float(got[0].abs().max()) == 0 and float(got[1, 160:].abs().max()) == 0 and float(got[2].abs().max()) == 0
    ), "write must not touch other rows / users"
