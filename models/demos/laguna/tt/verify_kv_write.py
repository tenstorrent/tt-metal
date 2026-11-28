# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paged KV-cache write for the DFlash verify rows as one generic_op (Laguna; kernels/verify_kv/*.cpp).

The verify writes B rows (B <= 32 per call: one 32-row chunk of the batched verify, or the K + 1 candidates of one
user) whose positions are consecutive for each user, so several rows land in one 32-row cache tile. Main's
paged_update_cache / paged_fused_update_cache give every row its own core, which reads the tile, patches its row and
writes the tile back; two rows of one tile on two cores race (the last writer wins and drops the other's row).

Here one core per row still untilizes its row, but the rows of one cache tile meet on one core: the lowest such row
("leader") reads the tile once, copies its own row and every other row of that tile ("followers", read over the NOC
from their cores' L1) into it, re-tilizes and writes it once; the followers read and write no cache tile. K rows and V
rows run in the same program on disjoint cores (the decoder's K / V shards: K on core rows 0-3, V on rows 4-7).

The positions (cur_pos) and the page table are read on device, so one trace replays for new positions. The untilize /
tilize calls are the ones paged_fused_update_cache uses (kernels/verify_kv/compute.cpp), so the cache bits match its
group mode (jerrywangTT/laguna-s-merge-main 8437ef46690) and the one-row-at-a-time writes of main's
paged_update_cache.

Inputs: caches [blocks, nkv, block_size, head_dim] bf8 / bf16 TILE, DRAM interleaved (K and V the same spec); k_sh /
v_sh [1, B, nkv, head_dim] TILE, height-sharded row-major in L1 one row per core (the decoder's _shard_kv layout,
nkv <= 32); cur_pos int32 [B] row-major, interleaved (-1 skips a row); page_table int32 [>= B, blocks] row-major,
interleaved (row b is row b's page table)."""

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels" / "verify_kv"
TILE = 32

# circular buffer indices (the paged_fused_update_cache program's, 78608b96a50^ factory lines 184-192)
CB_CACHE, CB_IN_K, CB_IN_V, CB_INDEX, CB_META = 0, 1, 2, 3, 5
CB_OUT, CB_UNTILIZED_CACHE, CB_UNTILIZED_CACHE2, CB_UNTILIZED_INPUT = 16, 24, 25, 26
SEM_READY, SEM_DONE = 0, 1

_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def _round_up(x, m):
    return -(-x // m) * m


def _check_pair(cache, kv_sh, name):
    if cache.dtype not in _TILE_BYTES or cache.layout != ttnn.TILE_LAYOUT or cache.is_sharded():
        raise ValueError(f"verify_kv_write: {name} cache must be bf8 / bf16 TILE interleaved, got {cache.dtype}")
    if cache.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"verify_kv_write: {name} cache must be in DRAM")
    mc = kv_sh.memory_config()
    if not kv_sh.is_sharded() or mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        raise ValueError(f"verify_kv_write: {name} rows must be height-sharded (one row per core), got {mc}")
    spec = mc.shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"verify_kv_write: {name} rows must be L1 row-major shards")
    if kv_sh.dtype != ttnn.bfloat16 or kv_sh.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"verify_kv_write: {name} rows must be bf16 TILE, got {kv_sh.dtype}")
    ps, cps = list(kv_sh.padded_shape), list(cache.padded_shape)
    B = int(ps[1])
    if ps[0] != 1 or ps[2] != TILE or ps[3] != cps[3] or list(spec.shape) != [TILE, cps[3]]:
        raise ValueError(f"verify_kv_write: {name} rows {ps} (shard {list(spec.shape)}) do not fit cache {cps}")
    if spec.grid.num_cores() != B:
        raise ValueError(f"verify_kv_write: {name} rows: {B} rows on {spec.grid.num_cores()} cores")
    return B


def verify_kv_write(k_cache, k_sh, v_cache, v_sh, cur_pos, page_table):
    """Write the B verify rows k_sh (v_sh) into k_cache (v_cache) at positions cur_pos through page_table, in place.
    v_cache / v_sh may be None (one cache). K and V rows on overlapping cores run as two programs."""
    pairs = [(k_cache, k_sh)] + ([(v_cache, v_sh)] if v_cache is not None else [])
    Bs = {_check_pair(c, s, n) for (c, s), n in zip(pairs, ("K", "V"))}
    if len(Bs) != 1:
        raise ValueError(f"verify_kv_write: K and V row counts differ: {Bs}")
    B = Bs.pop()
    if len(pairs) == 2:
        k_grid = k_sh.memory_config().shard_spec.grid
        v_cores = ttnn.corerange_to_cores(v_sh.memory_config().shard_spec.grid, B, True)
        if any(k_grid.contains(c) for c in v_cores):
            _program([pairs[0]], B, cur_pos, page_table)
            _program([pairs[1]], B, cur_pos, page_table)
            return
        kc, vc = k_cache, v_cache
        if kc.dtype != vc.dtype or list(kc.padded_shape) != list(vc.padded_shape):
            raise ValueError("verify_kv_write: K and V caches must have the same dtype and shape")
    _program(pairs, B, cur_pos, page_table)


def _program(pairs, B, cur_pos, page_table):
    cache0 = pairs[0][0]
    device = cache0.device()
    for t, name in ((cur_pos, "cur_pos"), (page_table, "page_table")):
        if t.dtype != ttnn.int32 or t.layout != ttnn.ROW_MAJOR_LAYOUT or t.is_sharded():
            raise ValueError(f"verify_kv_write: {name} must be int32 row-major interleaved, got {t.dtype} {t.layout}")
    if int(cur_pos.shape[-1]) != B or cur_pos.logical_volume() != B:
        raise ValueError(f"verify_kv_write: cur_pos {list(cur_pos.shape)} must hold the {B} row positions in one row")
    if len(page_table.shape) != 2 or int(page_table.shape[0]) < B:
        raise ValueError(f"verify_kv_write: page_table {list(page_table.shape)} needs one row per verify row ({B})")

    cps = list(cache0.padded_shape)
    num_heads, block_size, head_dim = int(cps[1]), int(cps[2]), int(cps[3])
    if block_size % TILE or head_dim % TILE or num_heads > TILE:
        raise ValueError(f"verify_kv_write: unsupported cache shape {cps}")
    Wt = head_dim // TILE
    St = block_size // TILE
    Wbytes = head_dim * 2  # untilized rows are bf16 (fp32 dest accumulation off, as paged_fused_update_cache default)
    cache_tile = _TILE_BYTES[cache0.dtype]
    meta_bytes = _round_up((5 + 2 * B) * 4, 64)

    grids = [s.memory_config().shard_spec.grid for _, s in pairs]
    all_cores = grids[0] if len(grids) == 1 else grids[0].merge(grids[1])
    cores = [ttnn.corerange_to_cores(g, B, True) for g in grids]
    noc_xy = []
    for set_cores in cores:
        for c in set_cores:
            p = device.worker_core_from_logical_core(c)
            noc_xy += [p.x, p.y]
    if len(pairs) == 1:  # the V block of the NOC table is never read; keep the layout fixed
        noc_xy += [0, 0] * B

    def cb(index, fmt, page, pages, grid=all_cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    untilized_cache = ttnn.CBDescriptor(  # cb_untilized_cache and cb_untilized_cache2 share one region
        total_size=2 * Wt * 2048,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_UNTILIZED_CACHE, data_format=ttnn.bfloat16, page_size=2048),
            ttnn.CBFormatDescriptor(buffer_index=CB_UNTILIZED_CACHE2, data_format=ttnn.bfloat16, page_size=2048),
        ],
    )
    cbs = [
        cb(CB_CACHE, cache0.dtype, cache_tile, 2 * Wt),
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN_K, pairs[0][1]),
        cb(CB_INDEX, ttnn.int32, _round_up(B * 4, 64), 1),
        cb(CB_META, ttnn.uint32, meta_bytes + 64 * B, 1),
        cb(CB_OUT, cache0.dtype, cache_tile, min(num_heads, 2) * Wt),
        untilized_cache,
        cb(CB_UNTILIZED_INPUT, ttnn.bfloat16, 2048, 2 * Wt),
    ]
    if len(pairs) == 2:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_IN_V, pairs[1][1]))

    v_cache = pairs[1][0] if len(pairs) == 2 else cache0
    in_v = CB_IN_V if len(pairs) == 2 else CB_IN_K  # one cache: the kernels never touch an input-V buffer
    cache_args = list(ttnn.TensorAccessorArgs(cache0).get_compile_time_args())
    block_size_t = block_size // TILE  # tiles per block (St: tiles per cache head; the same for a paged cache)
    reader_ct = [CB_IN_K, in_v, CB_CACHE, CB_INDEX, CB_META, Wt, num_heads, block_size, block_size_t, St, B]
    reader_ct += [meta_bytes]
    reader_ct += cache_args
    reader_ct += list(ttnn.TensorAccessorArgs(cur_pos).get_compile_time_args())
    reader_ct += list(ttnn.TensorAccessorArgs(page_table).get_compile_time_args())
    writer_ct = [CB_OUT, CB_UNTILIZED_CACHE, CB_UNTILIZED_CACHE2, CB_UNTILIZED_INPUT, CB_META, Wt, Wbytes, num_heads]
    writer_ct += [St, B, SEM_READY, SEM_DONE] + cache_args
    compute_ct = [CB_IN_K, in_v, CB_CACHE, CB_UNTILIZED_CACHE, CB_UNTILIZED_CACHE2, CB_UNTILIZED_INPUT, CB_OUT]
    compute_ct += [Wt, num_heads]

    rt = [((c.x, c.y), [s, row]) for s, set_cores in enumerate(cores) for row, c in enumerate(set_cores)]
    addrs = [cache0.buffer_address(), v_cache.buffer_address()]

    def kernel(name, ct, common, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=ct,
            runtime_args=rt,
            common_runtime_args=common,
            config=config,
        )

    kernels = [
        kernel("reader.cpp", reader_ct, addrs + [cur_pos.buffer_address(), page_table.buffer_address()],
               ttnn.ReaderConfigDescriptor()),
        kernel("writer.cpp", writer_ct, addrs + noc_xy, ttnn.WriterConfigDescriptor()),
        # paged_fused_update_cache's compute config: fp32 dest accumulation off, every other knob at its default
        kernel("compute.cpp", compute_ct, [], ttnn.ComputeConfigDescriptor()),
    ]  # fmt: skip
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=0) for i in (SEM_READY, SEM_DONE)]
    io = [cur_pos, page_table] + [s for _, s in pairs] + [c for c, _ in pairs]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
