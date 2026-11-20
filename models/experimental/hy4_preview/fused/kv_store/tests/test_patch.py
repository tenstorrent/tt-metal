# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The frame-table patch of PATCH.md, tried on COPIES of the owners' kernels (``patch.py`` writes them under
``kv_store/patched/``; the owners' files and Python are used as they are, the patched kernel is swapped into the
program descriptor their code builds):

    sm_reader    read_rows on today's slot-major storage with identity tables: the output tensor equals the
                 unpatched reader's bit for bit, traced latency next to it; then on a frame-major pool that
                 holds the same rows in scattered frames: the same output again
    sm_append    twin storages, the unpatched and the patched append: equal afterwards (incl. a position beyond
                 the frames)
    scan_reader  the tree scan on today's key cache with identity tables and on a pool with scattered frames:
                 the same fused words as the unpatched scan
    table write  what rewriting the L1 table tensor costs the host

One chip. Mock Galaxy: the patched kernels are compiled in the owners' programs, nothing runs."""

import os
import random
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.kv_store import device as dv
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store import patch

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
one_chip = pytest.mark.parametrize(
    "device_params, mesh_device", [({"trace_region_size": 8 << 20}, (1, 1))], indirect=True, ids=["1x1"]
)
FUSED = patch.FUSED
INCLUDES = [
    str(FUSED / "kv_store" / "kernels"),
    str(FUSED / "sparse_mla" / "kernels"),
    str(FUSED / "indexer_scan" / "kernels"),
]
L = 2  # layer slots per user in the test storages
FRAMES_PER_USER = 4
LATENT2 = fr.Kind("latent", layers=L, width=656, shard_bytes=32 * 704, row_stride=704)
INDEX2 = fr.Kind("index", layers=L, width=128, shard_bytes=4352, row_stride=0)


def test_patch_applies():
    """Every replacement finds its text exactly once in the owners' current files."""
    for relative in patch.PATCHES:
        patch.apply(relative)
    logger.info(f"HY4_MK10 patch size (lines removed, added): {patch.size()}")


def swap(desc, name, named=(), cbs=()):
    """Put the patched copy of kernel ``name`` into a program descriptor the owner's code built."""
    sources = patch.patched_sources()
    kernels, hit = list(desc.kernels), 0
    for k in kernels:
        if str(k.kernel_source).endswith("/" + name):
            k.kernel_source = sources[name]
            if named:
                k.named_compile_time_args = list(k.named_compile_time_args) + list(named)
            k.compiler_include_paths = list(k.compiler_include_paths) + INCLUDES
            hit += 1
    assert hit == 1, (name, hit)
    desc.kernels = kernels
    desc.cbs = list(desc.cbs) + list(cbs)
    assert (
        sum(str(k.kernel_source) == sources[name] for k in desc.kernels) == 1
    ), "the descriptor did not take the patched kernel"
    assert not named or all(
        dict(k.named_compile_time_args).get(n) == v
        for k in desc.kernels
        if str(k.kernel_source) == sources[name]
        for n, v in named
    )
    return desc


def scattered(users, seed):
    """frames[u][k]: the pool frame of the k-th 8192 tokens of user u: a permutation of all frames."""
    order = list(range(users * FRAMES_PER_USER))
    random.Random(seed).shuffle(order)
    return [order[u * FRAMES_PER_USER : (u + 1) * FRAMES_PER_USER] for u in range(users)]


def to_pool(cache: torch.Tensor, frames, rows_per_frame):
    """[users * L, R, w] slot-major host data -> [frames * L, rows per frame, w] frame-major."""
    users = len(frames)
    pool = torch.zeros(users * FRAMES_PER_USER * L, rows_per_frame, cache.shape[-1], dtype=cache.dtype)
    for u in range(users):
        for k, f in enumerate(frames[u]):
            for layer in range(L):
                pool[f * L + layer] = cache[u * L + layer, k * rows_per_frame : (k + 1) * rows_per_frame]
    return pool


def table_rows(geometry, frames, layer_bytes=0):
    out = []
    for user_frames in frames:
        out += [geometry.frame_base(f) + layer_bytes for f in user_frames] + [fr.UNMAPPED] * (
            fr.TABLE_FRAMES - len(user_frames)
        )
    return out


@one_chip
def test_sm_reader_and_append(mesh_device):
    from models.experimental.hy4_preview.fused.sparse_mla import op as smop
    from models.experimental.hy4_preview.fused.sparse_mla.layout import SparseMlaConfig
    from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs

    device = mesh_device
    R = FRAMES_PER_USER * fr.FRAME_ROWS
    cfg = SparseMlaConfig(rows_per_slot=R, num_layers=L)
    pool_cfg = SparseMlaConfig(rows_per_slot=fr.FRAME_ROWS, num_layers=L)
    users, layer = cfg.users, 1
    cache_g, pool_g = fr.Geometry.contiguous(LATENT2, FRAMES_PER_USER), fr.Geometry.pool(LATENT2)
    latent, rope = hs.random_latents(users * L, R, cfg, seed=3)
    kv, _ = hs.make_storage(device, cfg, users * L, latent, rope)
    frames = scattered(users, seed=1)
    pool, _ = hs.make_storage(
        device,
        pool_cfg,
        users * FRAMES_PER_USER * L,
        to_pool(latent, frames, fr.FRAME_ROWS),
        to_pool(rope, frames, fr.FRAME_ROWS),
    )
    counts = [1, 37, 512, 1024, 0, 64, 129, 300]
    lists = [hs.spread_ids(cfg, n, seed=10 + u) for u, n in enumerate(counts)]
    sel = hs.make_selection(device, cfg, lists)
    make_out = lambda: hs.to_dev(
        torch.zeros(1, 1, users * cfg.topk, cfg.width), device, ttnn.bfloat16, ttnn.TILE_LAYOUT
    )
    out_ref, out_ident, out_pool = make_out(), make_out(), make_out()

    op, pool_op = smop.SparseMla(cfg, device), smop.SparseMla(pool_cfg, device)
    reader_cores = [(x0, y) for x0 in (smop.X0_LEFT, smop.X0_RIGHT) for y in range(cfg.groups)]
    append_core = (5, 0)
    tables = dv.FrameTables(device, reader_cores, rows=users)
    append_tables = dv.FrameTables(device, [append_core], rows=users)
    CB_FRAMES, APPEND_CB_FRAMES = 9, 2
    ident = [x for u in range(users) for x in cache_g.identity_table(u)]
    # Host check before anything runs. The patch changes WHERE a row is read, not what the reader pushes: the
    # chunk loop, the control pages and the slab pushes are the owner's lines, driven by the same selection
    # table, so the converter (a compute kernel) is handed exactly what the unpatched reader hands it. What the
    # host can get wrong is the table: every selected row must be in a mapped frame and inside the tensor.
    for geometry, flat, n in ((cache_g, ident, users), (pool_g, table_rows(pool_g, frames), users * FRAMES_PER_USER)):
        limit = geometry.bytes_per_bank(n)
        for u, ids in enumerate(lists):
            row_table = flat[u * fr.TABLE_FRAMES : (u + 1) * fr.TABLE_FRAMES]
            for r in ids.tolist():
                bank, offset = fr.locate(row_table, geometry.slot_bytes(layer), r, 704)
                assert 0 <= bank < 8 and offset + 704 <= limit, (u, r, offset, limit)

    def reader(op_, storage, out, geometry):
        desc = op_.read_rows_program(storage, sel, out, layer)
        swap(
            desc,
            "sm_reader.cpp",
            named=[("cb_frames", CB_FRAMES), ("slot_bytes", geometry.slot_bytes(1))],
            cbs=[tables.cb(CB_FRAMES)],
        )
        return [storage, sel, out, tables.tensor], desc

    programs = {
        "unpatched": ([kv, sel, out_ref], op.read_rows_program(kv, sel, out_ref, layer)),
        "identity": reader(op, kv, out_ident, cache_g),
        "pool": reader(pool_op, pool, out_pool, pool_g),
    }
    # the append: twin storages
    kv2, _ = hs.make_storage(device, cfg, users * L, latent, rope)
    chip = 5
    own = [chip * 128, 3 * 4096 + chip * 128 + 77, 7 * 4096 + chip * 128 + 127, 5 * 4096 + chip * 128 + 1, 640 + 5]
    positions = [own[0], 6 * 128 + 3, own[1], own[2], 4096 + 9, 8 * 4096 + chip * 128, own[3], own[4]]  # user 5: beyond
    gen = torch.Generator().manual_seed(11)
    new_rows = hs.pack_rows(
        device,
        torch.randn(users, 1, 1, cfg.latent, generator=gen) * 3.0,
        torch.randn(users, 1, 1, cfg.rope, generator=gen),
    )
    pos = hs.to_dev(torch.tensor(positions, dtype=torch.int32).reshape(1, 1, 1, -1), device, ttnn.uint32)
    append_ref = op.append_program(kv, new_rows, pos, layer, chip, core=append_core)
    append_new = swap(
        op.append_program(kv2, new_rows, pos, layer, chip, core=append_core),
        "sm_append.cpp",
        named=[("cb_frames", APPEND_CB_FRAMES), ("slot_bytes", cache_g.slot_bytes(1))],
        cbs=[append_tables.cb(APPEND_CB_FRAMES)],
    )
    if MOCK:
        for io, desc in programs.values():
            ttnn.experimental.prepare_generic_op(io, desc)
        ttnn.experimental.prepare_generic_op([new_rows, pos, kv], append_ref)
        ttnn.experimental.prepare_generic_op([new_rows, pos, kv2, append_tables.tensor], append_new)
        return

    # --- the reader
    tables.write(ident)
    times = {}
    for name in ("unpatched", "identity"):
        io, desc = programs[name]
        times[name] = tracing.traced_ms(device, lambda: ttnn.generic_op(io, desc))["replay_ms"] * 1e3
    ref = ttnn.to_torch(out_ref)[0, 0].to(torch.bfloat16).view(torch.int16)
    got = ttnn.to_torch(out_ident)[0, 0].to(torch.bfloat16).view(torch.int16)
    assert ref.abs().sum() > 0 and torch.equal(
        got, ref
    ), "patched reader, identity table: output differs from the unpatched reader"
    tables.write(table_rows(pool_g, frames))
    io, desc = programs["pool"]
    times["pool"] = tracing.traced_ms(device, lambda: ttnn.generic_op(io, desc))["replay_ms"] * 1e3
    got = ttnn.to_torch(out_pool)[0, 0].to(torch.bfloat16).view(torch.int16)
    assert torch.equal(got, ref), "patched reader on the pool with scattered frames: output differs"
    logger.info(
        f"HY4_MK10 patch sm_reader: read_rows of {sum(counts)} rows (counts {counts}) bit-equal to the unpatched reader "
        f"with identity tables on today's storage and on a frame-major pool with frames {frames[:2]}...; traced "
        f"{times['unpatched']:.1f} us unpatched, {times['identity']:.1f} us identity, {times['pool']:.1f} us pool"
    )

    # --- the append
    append_tables.write(ident)
    ttnn.generic_op([new_rows, pos, kv], append_ref)
    ttnn.generic_op([new_rows, pos, kv2, append_tables.tensor], append_new)
    whole = ttnn.ConcatMeshToTensor(device, dim=0)
    a = ttnn.to_torch(kv, mesh_composer=whole).contiguous().view(torch.uint8)
    b = ttnn.to_torch(kv2, mesh_composer=whole).contiguous().view(torch.uint8)
    fresh = (
        ttnn.to_torch(hs.make_storage(device, cfg, users * L, latent, rope)[0], mesh_composer=whole)
        .contiguous()
        .view(torch.uint8)
    )
    changed = (a.reshape(users * L, R, -1) != fresh.reshape(users * L, R, -1)).any(-1).sum().item()
    assert torch.equal(a, b), "patched append: the storage differs from the unpatched append's"
    assert changed == 5, changed
    logger.info(
        f"HY4_MK10 patch sm_append: twin storages equal after the unpatched and the patched append, {changed} rows written (3 skipped: 2 on other chips, 1 beyond the frames)"
    )

    # --- what a table rewrite costs the host
    for cores, what in (
        (reader_cores, "16 cores (the row readers)"),
        ([(x, y) for x in range(8) for y in range(8)], "64 cores (the scan cores)"),
    ):
        t = dv.FrameTables(device, cores, rows=users)
        ttnn.synchronize_device(device)
        best = None
        for _ in range(5):
            start = time.perf_counter()
            t.write(ident)
            ttnn.synchronize_device(device)
            ms = 1e3 * (time.perf_counter() - start)
            best = ms if best is None else min(best, ms)
        logger.info(f"HY4_MK10 table write, one chip, 8 decode rows on {what}: {best:.2f} ms on the host")


# 4 frames per user with ragged contexts, and a shape near the AgentX one: its contexts capped at the 262,144
# tokens of 32 frames per user (1,408 to 8,192 visible stripe rows on the chip)
SCAN_CASES = {
    "small": (4, [32768, 20000, 9000, 3 * 4096 + 700, 600, 32768, 12345, 27000]),
    "agentx": (32, [45609, 71284, 92682, 115666, 144011, 175480, 215110, 262144]),
}


@one_chip
@pytest.mark.parametrize("case", list(SCAN_CASES))
def test_scan_reader(mesh_device, case, monkeypatch):
    from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as isc

    per_user, contexts = SCAN_CASES[case]
    monkeypatch.setattr(sys.modules[__name__], "FRAMES_PER_USER", per_user)
    users, R, rank, chip = 8, FRAMES_PER_USER * fr.FRAME_ROWS, 1, 5
    cache_g, pool_g = fr.Geometry.contiguous(INDEX2, FRAMES_PER_USER), fr.Geometry.pool(INDEX2)
    g = torch.Generator().manual_seed(4)
    keys = torch.randn(users * L, R, isc.KEY_DIM, generator=g).bfloat16().float()
    q = torch.randn(users, isc.HEADS, isc.KEY_DIM, generator=g).bfloat16().float()
    w = torch.randn(users, isc.HEADS, generator=g)
    frames = scattered(users, seed=2)
    cache = isc.keys_to_device(mesh_device, keys)
    pool = isc.keys_to_device(mesh_device, to_pool(keys, frames, fr.FRAME_ROWS))
    cfg = lambda **kw: isc.ScanConfig(users=users, stripe_rows=R, mode="tree", **kw)
    scans = {
        "unpatched": isc.Scan(
            mesh_device, cfg(), cache, [u * L + rank for u in range(users)], chip_index=lambda c: chip
        ),
        # the user's part of the slot offset is in the table now: every user passes the layer's slot step alone
        "identity": isc.Scan(mesh_device, cfg(), cache, [rank] * users, chip_index=lambda c: chip),
        "pool": isc.Scan(mesh_device, cfg(), cache, [0] * users, chip_index=lambda c: chip),
        "pool8": isc.Scan(
            mesh_device, cfg(stream_depth=8, blocks_per_read=1), cache, [0] * users, chip_index=lambda c: chip
        ),
    }
    patched = ("identity", "pool", "pool8")
    for name in ("pool", "pool8"):
        scans[name].keys = pool  # (the owner's constructor checks the slot-major shape)
    tables = {name: dv.FrameTables(mesh_device, scans[name].cores, rows=users) for name in patched}
    CB_FRAMES = 7
    for name in patched:
        scan = scans[name]
        mesh_desc = ttnn.MeshProgramDescriptor()
        desc = scan.append_to(ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[]), (0, 0))
        swap(desc, "scan_reader.cpp", cbs=[tables[name].cb(CB_FRAMES)])
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = desc
        scan._descriptor = mesh_desc
        scan._io = (
            lambda s, t: (
                lambda: [s.keys, s.qt, s.sel, s.step] + ([s.sel_l1] if s.sel_l1 is not None else []) + [t.tensor, s.out]
            )
        )(scan, tables[name])
    for scan in scans.values():
        scan.set_query(*isc.device_query(q, w))
        scan.set_contexts(contexts)
    # Host check before anything runs: the compute kernel of a scan core waits for `padded` key blocks, 8 at a
    # time. With either table the patched reader must push exactly that many, never run a packet over the end of
    # the key buffer, and read only inside the tensor (patch.scan_packets mirrors the patched lines).
    flat = {
        "identity": [x for u in range(users) for x in cache_g.identity_table(u)],
        "pool": table_rows(pool_g, frames, layer_bytes=pool_g.slot_bytes(rank)),
    }
    flat["pool8"] = flat["pool"]
    limits = {"identity": cache_g.bytes_per_bank(users), "pool": pool_g.bytes_per_bank(users * FRAMES_PER_USER)}
    limits["pool8"] = limits["pool"]
    sizes = {name: set() for name in patched}
    for name in patched:
        c = scans[name].cfg
        base = scans[name].slots[0] * c.bank_blocks * isc.KEY_BLOCK_BYTES
        for u in range(users):
            rows = isc.visible_rows(contexts[u], chip, 32, 128, c.stripe_rows)
            for bank in range(8):
                whole_blocks, tail = rows // 32, rows % 32
                num = (
                    whole_blocks // 8
                    + (1 if whole_blocks % 8 > bank else 0)
                    + (1 if whole_blocks % 8 == bank and tail else 0)
                )
                padded = -(-num // c.blocks_per_pass) * c.blocks_per_pass
                packets = patch.scan_packets(
                    flat[name][u * fr.TABLE_FRAMES : (u + 1) * fr.TABLE_FRAMES],
                    num,
                    padded,
                    c.blocks_per_read,
                    c.k_blocks,
                )
                assert sum(p[1] for p in packets) == padded and sum(p[2] for p in packets) == num
                for first, blocks, read, offset in packets:
                    assert blocks <= c.blocks_per_read and (
                        read == 0 or base + offset + read * isc.KEY_BLOCK_BYTES <= limits[name]
                    )
                    sizes[name].add(read)
    assert 3 in sizes["identity"] and sizes["pool"] <= {0, 1} and sizes["pool8"] <= {0, 1}, sizes
    if MOCK:
        scans["unpatched"].compile()
        for name in patched:
            ttnn.experimental.prepare_generic_op(scans[name]._io(), scans[name]._descriptor)
        return
    for name in patched:
        tables[name].write(flat[name])
    words, times = {}, {}
    for name, scan in scans.items():
        times[name] = tracing.traced_ms(mesh_device, scan)["replay_ms"] * 1e3
        res = scan.read()
        words[name] = res["word"][res["valid"]]
        assert int(res["valid"].sum()) > 2048
    assert torch.equal(
        words["identity"], words["unpatched"]
    ), "patched scan, identity tables: other words than the unpatched scan"
    assert torch.equal(words["pool"], words["unpatched"]), "patched scan on a pool with scattered frames: other words"
    assert torch.equal(words["pool8"], words["unpatched"]), "patched scan on a pool, 8 packets in flight: other words"
    logger.info(
        f"HY4_MK10 patch scan_reader {case}: tree scan of 8 users x {R} stripe rows (contexts {contexts}): {words['unpatched'].numel()} "
        f"valid fused words equal to the unpatched scan with identity tables and on a frame-major pool; traced "
        f"{times['unpatched']:.1f} us unpatched, {times['identity']:.1f} us identity (packets of 3 blocks), {times['pool']:.1f} us pool "
        f"(packets of 1 block, 4 in flight), {times['pool8']:.1f} us pool (packets of 1 block, 8 in flight)"
    )
