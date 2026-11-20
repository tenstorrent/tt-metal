# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""kv_frames::locate on the device, with a reader of this piece (kernels/kv_probe_reader.cpp; no kernel of the
sparse-MLA or scan pieces runs here): one core reads items from a cache tensor at the addresses a frame table in L1
gives, and the host compares every byte with what it put there.

    pool, scrambled table     a frame-major pool filled from the host; the conversations' frames are out of order
                              and interleaved
    cache, identity table     today's slot-major layout addressed through the table
    real formats              an fp8_e4m3 latent pool and a bfloat8_b index-key pool as the prefill stack allocates
                              them (one case each)
    cost                      traced: 16,384 address computations through the table, through today's contiguous
                              arithmetic (same addresses: checksums equal), and neither

One chip, no fabric. Mock Galaxy (HY4_MOCK=1): the programs are compiled and their buffers checked, nothing runs."""

import os
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.kv_store import device as dv
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store.manager import KvStore

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
one_chip = pytest.mark.parametrize(
    "device_params, mesh_device", [({"trace_region_size": 4 << 20}, (1, 1))], indirect=True, ids=["1x1"]
)
ROWS = 4  # decode rows of the test tables
CORE = (0, 0)


def scrambled_store(frames: int = 6):
    """Three conversations whose frames are out of order and interleaved (a = [4, 0], b = [5, 2, 3], c = [1]), on
    decode rows 0, 1 and 3; made with the manager's own calls."""
    assert frames == 6
    F = fr.FRAME_TOKENS
    store = KvStore(num_frames=frames, rows=ROWS)
    store.reserve("t", F)  # [0]
    store.reserve("y", 3 * F)  # [1, 2, 3]
    store.reserve("a", F)  # [4]
    store.reserve("b", F)  # [5]
    store.evict("t")
    store.reserve("a", 2 * F)
    store.evict("y")
    store.reserve("c", F)
    store.reserve("b", 3 * F)
    assert [store.conversation(k).frames for k in "abc"] == [[4, 0], [5, 2, 3], [1]]
    for row, key in ((0, "a"), (1, "b"), (3, "c")):
        store.bind(row, key)
    store.check()
    return store


def requests_for(store: KvStore, kind: fr.Kind, n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        row = rng.choice([r for r, key in enumerate(store.binding) if key is not None])
        frames = store.conversation(store.binding[row]).frames
        out.append((row, rng.randrange(kind.layers), rng.randrange(len(frames) * fr.FRAME_ROWS)))
    return out


def expected_item(geometry, host, frame, layer, stripe_row):
    view = dv.frame_view(geometry, host, frame, layer)
    q = stripe_row % fr.FRAME_ROWS
    if geometry.kind.row_stride:
        return view[q]
    block = q // fr.SHARD_ROWS  # the bank's shard of the frame: 4 pages of the raw twin, 32 rows of a real tensor
    per = view.shape[0] // fr.BANKS
    return view[block * per : (block + 1) * per].reshape(-1)


@one_chip
@pytest.mark.parametrize("kind", [fr.LATENT, fr.INDEX], ids=["latent", "index"])
def test_rows_through_the_table(mesh_device, kind):
    """Raw twins, chosen bytes: a pool with a scrambled table, then a slot-major cache with identity tables."""
    frames, n = 6, 512
    tables = dv.FrameTables(mesh_device, [CORE], ROWS, tables=2)
    gen = torch.Generator().manual_seed(1)

    # --- frame-major pool, scrambled table
    pool = fr.Geometry.pool(kind)
    host = torch.randint(0, 256, dv.raw_shape(pool, frames), dtype=torch.uint8, generator=gen)
    tensor = dv.raw_to_device(mesh_device, pool, host)
    dv.check_layout(tensor, pool, frames)
    store = scrambled_store(frames)
    tables.write(store.table(fr.Geometry.pool(fr.LATENT)), store.table(fr.Geometry.pool(fr.INDEX)))
    probe = dv.Probe(mesh_device, pool, tensor, tables, table=0 if kind is fr.LATENT else 1, max_requests=n)
    reqs = requests_for(store, kind, n, seed=5)
    probe.set_requests(reqs)
    if MOCK:
        probe.compile()
    else:
        back = tables.read()
        assert back[0, 0].reshape(-1).tolist() == list(store.table(fr.Geometry.pool(fr.LATENT)))
        assert back[0, 1].reshape(-1).tolist() == list(store.table(fr.Geometry.pool(fr.INDEX)))
        assert int(back[0, 0, 2, 0]) == fr.UNMAPPED  # decode row 2 serves nobody
        assert torch.equal(ttnn.to_torch(tensor), host), "the pool on the device is not what the host wrote"
        probe()
        got = probe.read().reshape(n, -1)
        banks, used = set(), set()
        for i, (row, layer, r) in enumerate(reqs):
            frame = store.conversation(store.binding[row]).frames[r >> 8]
            want = expected_item(pool, host, frame, layer, r)
            assert torch.equal(
                got[i, : want.numel()], want
            ), f"request {i}: decode row {row} layer {layer} stripe row {r} (frame {frame})"
            banks.add((r >> 5) & 7)
            used.add(frame)
        ids = {row: store.frame_ids(row)[:3] for row in (0, 1, 3)}
        logger.info(
            f"HY4_MK10 frames_device {kind.name} pool of {frames} frames, scrambled table {ids}: {n} items byte-equal "
            f"to the host ({got.shape[1]} B each), frames {sorted(used)}, banks {sorted(banks)}"
        )

    # --- today's slot-major layout through identity tables (2 users x 3 frames per slot)
    cache = fr.Geometry.contiguous(kind, 3)
    host2 = torch.randint(0, 256, dv.raw_shape(cache, 2), dtype=torch.uint8, generator=gen)
    tensor2 = dv.raw_to_device(mesh_device, cache, host2)
    dv.check_layout(tensor2, cache, 2)
    ident = [cache.identity_table(u, 3) if u < 2 else [fr.UNMAPPED] * fr.TABLE_FRAMES for u in range(ROWS)]
    flat = [x for t in ident for x in t]
    tables.write(flat, flat)
    probe2 = dv.Probe(mesh_device, cache, tensor2, tables, table=0, max_requests=n)
    rng = random.Random(6)
    reqs2 = [(rng.randrange(2), rng.randrange(kind.layers), rng.randrange(3 * fr.FRAME_ROWS)) for _ in range(n)]
    probe2.set_requests(reqs2)
    if MOCK:
        probe2.compile()
        probe2.compile(mode=dv.MODE_CONTIGUOUS)
        return
    probe2()
    got = probe2.read().reshape(n, -1)
    for i, (u, layer, r) in enumerate(reqs2):
        want = expected_item(cache, host2, cache.contiguous_frame(u, r >> 8), layer, r)
        assert torch.equal(got[i, : want.numel()], want), f"identity table, request {i}: user {u} layer {layer} row {r}"
    also = ""
    if kind is fr.LATENT:  # today's row arithmetic (sm::kv_row_*, latent rows only) reads the same bytes
        table_bytes = got.clone()
        probe2(mode=dv.MODE_CONTIGUOUS)
        assert torch.equal(probe2.read().reshape(n, -1), table_bytes)
        also = " and to the read through sm::kv_row_bank / kv_row_offset"
    logger.info(
        f"HY4_MK10 frames_device {kind.name} slot-major cache (2 users x 3 frames), identity table: {n} items "
        f"byte-equal to the host{also}"
    )


@one_chip
def test_real_formats(mesh_device):
    """The prefill stack's own dtypes: an fp8_e4m3 ROW_MAJOR latent pool and a bfloat8_b TILE index-key pool."""
    frames, n = 2, 256
    tables = dv.FrameTables(mesh_device, [CORE], ROWS, tables=2)
    store = KvStore(num_frames=frames, rows=ROWS)
    store.reserve("t", fr.FRAME_TOKENS)
    store.reserve("a", fr.FRAME_TOKENS)
    store.evict("t")
    store.reserve("a", 2 * fr.FRAME_TOKENS)
    store.bind(1, "a")
    assert store.conversation("a").frames == [1, 0]
    tables.write(store.table(fr.Geometry.pool(fr.LATENT)), store.table(fr.Geometry.pool(fr.INDEX)))
    gen = torch.Generator().manual_seed(2)

    # index keys: BFP8 tiles written by the host, decoded by the host on the way back
    pool = fr.Geometry.pool(fr.INDEX)
    shape = pool.tensor_shape(frames)
    keys = torch.randn(shape, generator=gen)
    tensor = ttnn.from_torch(
        keys,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dv.nd_memory_config(mesh_device, fr.SHARD_ROWS, fr.INDEX.width),
    )
    dv.check_layout(tensor, pool, frames)
    probe = dv.Probe(mesh_device, pool, tensor, tables, table=1, max_requests=n, out_like="bfp8")
    reqs = requests_for(store, fr.INDEX, n, seed=8)
    probe.set_requests(reqs)

    # latent rows: packed by the prefill stack's own ops (the sparse-MLA piece's test fixture), compared as bytes
    from models.experimental.hy4_preview.fused.sparse_mla.layout import SparseMlaConfig
    from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs

    lpool = fr.Geometry.pool(fr.LATENT)
    lshape = lpool.tensor_shape(frames)
    cfg = SparseMlaConfig(rows_per_slot=fr.FRAME_ROWS, num_layers=fr.LATENT.layers)
    latent, rope = hs.random_latents(lshape[0], fr.FRAME_ROWS, cfg, seed=3)
    ltensor, _ = hs.make_storage(mesh_device, cfg, lshape[0], latent, rope)
    assert tuple(ltensor.shape) == lshape and ltensor.dtype == ttnn.fp8_e4m3
    dv.check_layout(ltensor, lpool, frames)
    lprobe = dv.Probe(mesh_device, lpool, ltensor, tables, table=0, max_requests=n, out_like="fp8")
    lreqs = requests_for(store, fr.LATENT, n, seed=9)
    lprobe.set_requests(lreqs)
    if MOCK:
        probe.compile()
        lprobe.compile()
        return
    host = ttnn.to_torch(tensor)
    probe()
    got = ttnn.to_torch(probe.out).reshape(n, 32, fr.INDEX.width)
    for i, (row, rank, r) in enumerate(reqs):
        frame = store.conversation("a").frames[r >> 8]
        want = expected_item(pool, host, frame, rank, r).reshape(32, fr.INDEX.width)
        assert torch.equal(got[i], want), f"bfloat8_b request {i}: rank {rank} row {r} frame {frame}"
    logger.info(f"HY4_MK10 frames_device bfloat8_b index-key pool: {n} blocks equal to the host's decode")
    whole = ttnn.ConcatMeshToTensor(mesh_device, dim=0)  # fp8_e4m3 reads back through a mesh composer only
    lhost = ttnn.to_torch(ltensor, mesh_composer=whole).contiguous().view(torch.uint8).reshape(lshape)
    lprobe()
    lgot = ttnn.to_torch(lprobe.out, mesh_composer=whole).contiguous().view(torch.uint8).reshape(n, fr.LATENT.width)
    for i, (row, layer, r) in enumerate(lreqs):
        frame = store.conversation("a").frames[r >> 8]
        want = expected_item(lpool, lhost, frame, layer, r)
        assert torch.equal(lgot[i], want), f"fp8_e4m3 request {i}: layer {layer} row {r} frame {frame}"
    assert lhost.unique().numel() > 100
    logger.info(
        f"HY4_MK10 frames_device fp8_e4m3 latent pool (packed by the prefill ops): {n} rows byte-equal to the host's read-back"
    )


@one_chip
def test_cost_per_row_access(mesh_device):
    """What the table costs against today's arithmetic: 64 requests evaluated 255 times each without DRAM access,
    and 2,048 row reads, on a slot-major cache where both give the same addresses."""
    kind, n = fr.LATENT, dv.REQ_PER_PAGE
    cache = fr.Geometry.contiguous(kind, 3)
    tensor = dv.allocate(mesh_device, cache, 2, raw=True)
    tables = dv.FrameTables(mesh_device, [CORE], ROWS, tables=1)
    ident = [cache.identity_table(u, 3) if u < 2 else [fr.UNMAPPED] * fr.TABLE_FRAMES for u in range(ROWS)]
    tables.write([x for t in ident for x in t])
    rng = random.Random(12)
    reqs = [(rng.randrange(2), rng.randrange(kind.layers), rng.randrange(3 * fr.FRAME_ROWS)) for _ in range(2048)]
    repeat = 255
    probe = dv.Probe(mesh_device, cache, tensor, tables, table=0, max_requests=n)
    probe.set_requests(reqs[:n])
    reader = dv.Probe(mesh_device, cache, tensor, tables, table=0, max_requests=2048)
    reader.set_requests(reqs)
    modes = {"table": dv.MODE_NO_READS, "contiguous": dv.MODE_NO_READS | dv.MODE_CONTIGUOUS, "floor": dv.MODE_FLOOR}
    if MOCK:
        for mode in modes.values():
            probe.compile(mode=mode, repeat=repeat)
        return
    sums, times = {}, {}
    for name, mode in modes.items():
        program = probe.program(mode=mode, repeat=repeat)
        run = lambda: ttnn.generic_op(probe.io(), program)
        times[name] = tracing.traced_ms(mesh_device, run)["replay_ms"] * 1e3
        if name != "floor":
            words = ttnn.to_torch(probe.out).reshape(-1)[:8].to(torch.int64)
            sums[name] = sum(int(words[i]) << (8 * i) for i in range(4)), sum(
                int(words[4 + i]) << (8 * i) for i in range(4)
            )
    assert sums["table"] == sums["contiguous"] and sums["table"][1] == n * repeat, sums
    host = 0
    for _ in range(repeat):
        for u, layer, r in reqs[:n]:
            bank, off = fr.locate(ident[u], cache.slot_bytes(layer), r, kind.row_stride)
            host = ((((host << 1) | (host >> 31)) & 0xFFFFFFFF) + off + bank) & 0xFFFFFFFF
    assert sums["table"][0] == host != 0, (sums, host)
    calls = n * repeat
    per = {k: (times[k] - times["floor"]) * 1e3 / calls for k in ("table", "contiguous")}
    read_times = {}
    for name, mode in (("table", 0), ("contiguous", dv.MODE_CONTIGUOUS)):
        program = reader.program(mode=mode, batch=32)
        run = lambda: ttnn.generic_op(reader.io(), program)
        read_times[name] = tracing.traced_ms(mesh_device, run)["replay_ms"] * 1e3
    logger.info(
        f"HY4_MK10 frames_cost {calls} address computations on one core (traced): table {times['table']:.1f} us, "
        f"contiguous {times['contiguous']:.1f} us, floor {times['floor']:.1f} us -> {per['table']:.1f} ns per row "
        f"through the table, {per['contiguous']:.1f} ns through sm::kv_row_*; checksums equal ({sums['table'][0]:#x}); "
        f"2048 row reads + writes: table {read_times['table']:.1f} us, contiguous {read_times['contiguous']:.1f} us"
    )
