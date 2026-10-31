# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device unit tests of the key store (key_save / key_load) and the halo store (halo_copy), tt/apc_paged_store.py, on the (4, 8) mesh: every device (mesh row, column) holds DIFFERENT random content, so a
wrong row / column / user / slot / layer shows up. Bit-exact comparisons (bf16 viewed as int16). Includes a captured-trace test (the tables are read at replay time)."""

import random

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_store as S

DP = {"l1_small_size": 16384, "trace_region_size": 1 << 24}
U = 2


def mesh_dims(md):
    return tuple(md.shape)


def to_dev(md, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    rows, cols = mesh_dims(md)
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def from_dev(md, t):
    """-> int16 view [rows * d0, cols * d1, ...] (device (r, c) block at [r * d0 : , c * d1 : ])"""
    rows, cols = mesh_dims(md)
    h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(rows, cols)))
    return h.to(torch.bfloat16).view(torch.int16)


def rnd(g, *shape):
    return (torch.randn(*shape, generator=g) * 3).to(torch.bfloat16)


def i16(t):
    return t.to(torch.bfloat16).view(torch.int16)


def make_slab(md, g, ratio, max_ctx=16384, chunk=512):
    rows, cols = mesh_dims(md)
    geo = S.KeyGeom(chunk, ratio, cols)
    T = geo.slab_rows(max_ctx)
    host = rnd(g, rows * U, cols, T, 128)
    return geo, host, to_dev(md, host)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="trace")], indirect=True)
@pytest.mark.parametrize("ratios", [(2,), (2, 1)])
@torch.no_grad()
def test_key_roundtrip(mesh_device, ratios):
    md = mesh_device
    rows, cols = mesh_dims(md)
    g = torch.Generator().manual_seed(11)
    K = 64
    owners = {f"o{i}": r for i, r in enumerate(ratios)}
    ks = S.KeyStore(md, U, owners, K=K, nbmax=64)
    slabs, hosts = {}, {}
    for n, r in owners.items():
        geo, hosts[n], slabs[n] = make_slab(md, g, r)
    zero_ref = {n: torch.zeros(rows * K, cols, ks.geom[n].nrb * 32, 128, dtype=torch.int16) for n in owners}
    rng = random.Random(5)
    # (round, which entries saved): distinct slots per (row, user); some SKIP; slots reused across rounds (overwrite)
    expect = {n: zero_ref[n].clone() for n in owners}
    for rnd_i in range(3):
        vals = []
        for r_ in range(rows):
            slots = rng.sample(range(K), U)
            vals += [None if rng.random() < 0.3 else s for s in slots]
        ks.set_save(vals)
        ks.save(slabs)
        for n in owners:
            nrb32 = ks.geom[n].nrb * 32
            for i, v in enumerate(vals):
                if v is None:
                    continue
                r_, u = divmod(i, U)
                expect[n][r_ * K + v] = i16(hosts[n][r_ * U + u, :, :nrb32, :])
        for n in owners:
            got = from_dev(md, ks.store[n])
            assert torch.equal(got, expect[n]), (rnd_i, n, "store")
            assert torch.equal(from_dev(md, slabs[n]), i16(hosts[n])), (rnd_i, n, "save must not modify the slab")
        # refresh the slab scratch content for the next round (in place, same address)
        for n, r in owners.items():
            a0 = slabs[n].buffer_address()
            hosts[n] = rnd(g, *hosts[n].shape)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    hosts[n],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
                ),
                slabs[n],
            )
            assert slabs[n].buffer_address() == a0
    # load into FRESH slabs (other content): history blocks of the lists, nothing else
    dst, dhost = {}, {}
    for n, r in owners.items():
        _, dhost[n], dst[n] = make_slab(md, g, r)
    lists = []
    for r_ in range(rows):
        for u in range(U):
            lists.append(None if (r_ + u) % 4 == 3 else rng.sample(range(K), rng.choice([1, 2, 5, 16])))
    ks.set_load(lists)
    ks.load(dst)
    for n in owners:
        geo = ks.geom[n]
        want = i16(dhost[n]).clone()
        for i, lst in enumerate(lists):
            if lst is None:
                continue
            r_, u = divmod(i, U)
            for b, slot in enumerate(lst):
                want[r_ * U + u, :, geo.hist_row_block(b) * 32 : (geo.hist_row_block(b) + geo.nrb) * 32, :] = expect[n][
                    r_ * K + slot
                ]
        assert torch.equal(from_dev(md, dst[n]), want), (n, "load")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="trace")], indirect=True)
@torch.no_grad()
def test_key_load_clamped_and_empty(mesh_device):
    """an all-SKIP table loads nothing; a list longer than the slab history is clamped (no write past the slab)"""
    md = mesh_device
    rows, cols = mesh_dims(md)
    g = torch.Generator().manual_seed(2)
    ks = S.KeyStore(md, U, {"a": 2}, K=16, nbmax=64)
    geo, host, slab = make_slab(md, g, 2, max_ctx=4096)  # 4096 / 512 = 8 history blocks
    ks.set_load([None] * (rows * U))
    ks.load({"a": slab})
    assert torch.equal(from_dev(md, slab), i16(host))
    ks.set_load([list(range(16))] * (rows * U))  # 16 listed blocks; the slab has room for 9 (8 + the pad row block)
    ks.load({"a": slab})
    got = from_dev(md, slab)
    nb = geo.max_blocks(int(host.shape[2]))
    assert nb >= 8
    # rows beyond the history of nb blocks (the pad rows) untouched
    end = (geo.hist_row_block(nb - 1) + geo.nrb) * 32
    assert torch.equal(got[:, :, end:], i16(host)[:, :, end:])
    assert torch.equal(got[:, :, : geo.Cc8], i16(host)[:, :, : geo.Cc8])  # scratch untouched


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="trace")], indirect=True)
@pytest.mark.parametrize("part", ["A", "B"])
@torch.no_grad()
def test_halo_roundtrip(mesh_device, part):
    md = mesh_device
    rows, cols = mesh_dims(md)
    g = torch.Generator().manual_seed(7)
    KA, KB = 6, 4
    hs = S.HaloStore(md, U, halo_a=KA, halo_b=KB)
    K = hs.K[part]
    layers = hs.layers[part]
    NL = len(layers)
    hhost = {l: rnd(g, rows * U, cols, 128, 512) for l in range(40)}
    halos = {l: to_dev(md, hhost[l]) for l in range(40)}
    addr = {l: halos[l].buffer_address() for l in halos}
    rng = random.Random(3)
    expect = torch.zeros(rows * K * NL, cols, 128, 512, dtype=torch.int16)  # device block (r, c): rows [r * K * NL ...)
    for rnd_i in range(3):
        vals = []
        for r_ in range(rows):
            vals += [None if rng.random() < 0.25 else s for s in rng.sample(range(K), U)]
        hs.set_save(part, vals)
        hs.save(part, halos)
        for i, v in enumerate(vals):
            if v is None:
                continue
            r_, u = divmod(i, U)
            for j, l in enumerate(layers):
                expect[r_ * K * NL + v * NL + j] = i16(hhost[l][r_ * U + u])
        assert torch.equal(from_dev(md, hs.store[part]), expect), (rnd_i, "store")
        for l in range(40):
            assert torch.equal(from_dev(md, halos[l]), i16(hhost[l])), "save must not modify the halos"
        # new halo content in place
        for l in range(40):
            hhost[l] = rnd(g, *hhost[l].shape)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    hhost[l],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
                ),
                halos[l],
            )
    # restore: users with a slot get the stored halo of the part's layers; others and the other part's layers keep theirs
    vals = []
    for r_ in range(rows):
        vals += [None if (r_ == 1 and u == 0) else rng.randrange(K) for u in range(U)]
    hs.set_restore(part, vals)
    hs.restore(part, halos)
    for l in range(40):
        want = i16(hhost[l]).clone()
        if l in layers:
            j = layers.index(l)
            for i, v in enumerate(vals):
                if v is None:
                    continue
                r_, u = divmod(i, U)
                want[r_ * U + u] = expect[r_ * K * NL + v * NL + j]
        assert torch.equal(from_dev(md, halos[l]), want), (l, "restore")
        assert halos[l].buffer_address() == addr[l], "restore must write in place"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="trace")], indirect=True)
@torch.no_grad()
def test_captured_trace(mesh_device):
    """key_save + halo save (part A, 3 layers' worth is not allowed: the full 21) captured in a trace: the slot tables updated with h2d between replays steer the replay"""
    md = mesh_device
    rows, cols = mesh_dims(md)
    g = torch.Generator().manual_seed(21)
    K = 8
    ks = S.KeyStore(md, U, {"a": 2}, K=K, nbmax=64)
    hs = S.HaloStore(md, U, halo_a=K, halo_b=0)
    geo, host, slab = make_slab(md, g, 2, max_ctx=4096)
    hhost = {l: rnd(g, rows * U, cols, 128, 512) for l in range(40)}
    halos = {l: to_dev(md, hhost[l]) for l in range(40)}
    # compile / warm outside the trace
    ks.set_save([None] * (rows * U))
    hs.set_save("A", [None] * (rows * U))
    ks.save({"a": slab})
    hs.save("A", halos)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    ks.save({"a": slab})
    hs.save("A", halos)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    rng = random.Random(9)
    exp_k = torch.zeros(rows * K, cols, geo.nrb * 32, 128, dtype=torch.int16)
    exp_h = torch.zeros(rows * K * 21, cols, 128, 512, dtype=torch.int16)
    for it in range(3):
        vals = []
        for r_ in range(rows):
            vals += [None if rng.random() < 0.3 else s for s in rng.sample(range(K), U)]
        ks.set_save(vals)
        hs.set_save("A", vals)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        for i, v in enumerate(vals):
            if v is None:
                continue
            r_, u = divmod(i, U)
            exp_k[r_ * K + v] = i16(host[r_ * U + u, :, : geo.nrb * 32, :])
            for j in range(21):
                exp_h[r_ * K * 21 + v * 21 + j] = i16(hhost[j][r_ * U + u])
        assert torch.equal(from_dev(md, ks.store["a"]), exp_k), it
        assert torch.equal(from_dev(md, hs.store["A"]), exp_h), it
    ttnn.release_trace(md, tid)
