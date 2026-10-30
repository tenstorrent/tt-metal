# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``PagedUpdateRows``: the packed verify's K (or V) rows of every record row written into the paged cache as ONE
program, bit-identical to the per-position ``paged_update_cache`` loop of ``tt/attention/decode.py``
(``packed_kv_write_loop``: per position p slice, reshard, update = 6 programs per position per layer, 36 at K=5;
this op: 1 program per K / V tensor, 2 per layer).

Contract (the loop's inputs): ``src [1, nkv, S_pad, head_dim]`` bf16 TILE interleaved (DRAM or L1), the
prefill-style K or V after the norm and RoPE, row ``b * P + p`` = record row ``b``'s packed position ``p``
(``S_pad >= B * P``, tile padded); ``write_idxs``: the P ``[B]`` int32 ROW_MAJOR tensors of the loop
(``write_idxs[p][b]`` = the cache position of row b's position p; ``-1`` skips it, as the stock does); ``page_table
[B, max_blocks]`` int32 ROW_MAJOR; ``cache [blocks, ., ., head_dim]`` bf16 or bfloat8_b TILE DRAM, addressed
through the view ``(num_kv_heads, block_size)`` the loop passes (``effective_block_size``, the HMA-shared view);
``cache_position_modulo`` (the bounded ring: positions wrap before the page-table lookup, a whole number of
blocks). The cache is updated in place; nothing else is written.

Program: one (row b, KV head h) unit per core (``grid.plan_paged_update_rows``: ``min(B * nkv, grid)`` cores, the
units dealt contiguously). A unit's P consecutive positions touch at most two cache tile rows (P <= 32 and the
ring is a tile multiple), so the unit runs J = min(P, 2) tile-row jobs: per job the reader fetches the row's Wt
cache tiles, the compute untilizes them to bf16 row-major (the stock update_cache compute, same kernel_lib helper
and template arguments: 16-bit dest, fast untilize / tilize where the stock takes them), the writer patches the
job's rows from the untilized source rows (the stock's NoC loopback copy of one row), the compute tilizes the
patched block back in the cache format and the writer stores the tiles; a job no position landed in is a null
job (stale block, no store). Exactly one core touches a tile row (the row's blocks are its own: the loop's
batched op makes the same assumption), so the read-modify-write cannot race, and the P rows of a tile row are
patched in one round trip -- the stock's P sequential round trips are idempotent (bf16 and bfp8 decode exactly
to bf16 and re-encode to the same bytes), so the bytes match.

Trace-safe: every address and the unit list are per-core runtime args (re-applied on a program-cache hit), the
shapes are fixed per record, no host reads; the kernels end with a barrier. The positions are device data
(the record rewrites its ``write_idxs`` before a replay), read by the kernels.
"""

import os
from pathlib import Path

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, plan_paged_update_rows, row_major_cores

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
BF16_TILE = 2048
CACHE_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
ENV = "GEMMA4_PACKED_KV_WRITE_OP"  # "1": the packed verify writes K / V through this op (default: the loop)

# CB indices (kernels/paged_update_rows_common.hpp reads them from the compile-time args)
CB_SRC, CB_UNT_SRC, CB_CACHE, CB_UNT_A, CB_UNT_B, CB_OUT, CB_IDX, CB_PT, CB_JOBS = range(9)
N_SRC = 2  # source tile rows untilized per unit


def enabled() -> bool:
    """``GEMMA4_PACKED_KV_WRITE_OP=1``: the op serves the packed verify's KV write (the served default stays the
    per-position loop until the device A/B passes)."""
    return os.environ.get(ENV, "0").strip() == "1"


def _align(n, a):
    return (n + a - 1) // a * a


class PagedUpdateRows:
    @staticmethod
    def golden(cache, src, write_idxs, page_table, *, num_kv_heads, block_size, cache_position_modulo=None):
        """torch reference on a bf16 cache: ``cache [blocks, nkv, block, D]`` (the view), ``src [1, nkv, S, D]``,
        ``write_idxs`` a ``[P, B]`` int tensor (-1 = skip), ``page_table [B, blocks]``. Returns the updated cache
        (a copy). A bfp8 cache's bytes come from the packer: compare the op with the loop instead."""
        out = cache.clone()
        P, B = write_idxs.shape
        for b in range(B):
            for p in range(P):
                raw = int(write_idxs[p, b])
                if raw == -1:
                    continue
                q = raw % int(cache_position_modulo) if cache_position_modulo else raw
                blk = int(page_table[b, q // block_size])
                out[blk, :, q % block_size, :] = src[0, :, b * P + p, :]
        return out

    @classmethod
    def unsupported_reason(
        cls, cache, write_idxs, page_table, *, batch, packed_p, num_kv_heads, head_dim, block_size, modulo=None
    ):
        """A string naming why the loop must serve this call, ``None`` when the op takes it."""
        if page_table is None:
            return "no page table (the unpaged update)"
        if write_idxs is None or len(write_idxs) != packed_p:
            return f"write_idxs must hold P = {packed_p} position tensors"
        if not 1 <= packed_p <= TILE:
            return f"P = {packed_p} outside 1..{TILE}"
        if not 1 <= batch <= TILE:
            return f"batch {batch} outside 1..{TILE}"
        if cache.dtype not in CACHE_DTYPES or cache.layout != ttnn.TILE_LAYOUT or cache.is_sharded():
            return "cache must be a bf16 / bfp8 TILE interleaved tensor"
        if int(cache.padded_shape[-1]) != head_dim or head_dim % TILE:
            return f"cache width {int(cache.padded_shape[-1])} != head_dim {head_dim} (a tile multiple)"
        if block_size % TILE or int(cache.padded_shape[2]) * int(cache.padded_shape[1]) % (block_size * num_kv_heads):
            return f"block size {block_size} x {num_kv_heads} heads is not a tile view of the cache"
        if modulo is not None and (modulo <= 0 or modulo % block_size):
            return f"cache_position_modulo {modulo} is not a whole number of {block_size}-token blocks"
        if page_table.layout != ttnn.ROW_MAJOR_LAYOUT or page_table.dtype != ttnn.int32 or len(page_table.shape) != 2:
            return "page_table must be a 2-D int32 ROW_MAJOR tensor"
        if int(page_table.shape[0]) < batch:
            return f"page_table rows {int(page_table.shape[0])} < batch {batch}"
        spec = None
        for t in write_idxs:
            if t.layout != ttnn.ROW_MAJOR_LAYOUT or t.dtype != ttnn.int32 or int(t.shape[-1]) < batch:
                return "write_idxs must be int32 ROW_MAJOR [B] tensors"
            key = (tuple(t.shape), t.memory_config(), int(t.buffer_aligned_page_size()))
            if spec is None:
                spec = key
            elif key != spec:
                return "write_idxs must share one shape and memory config (one accessor layout)"
        return None

    @classmethod
    def op(cls, cache, src, write_idxs, page_table, *, batch, num_kv_heads, block_size, cache_position_modulo=None):
        """Write the packed rows of ``src`` into ``cache`` in place (see the module docstring): ``batch`` record
        rows (the first ``batch`` rows of ``page_table`` and entries of every ``write_idxs[p]``). Returns ``cache``."""
        assert src.layout == ttnn.TILE_LAYOUT and src.dtype == ttnn.bfloat16 and not src.is_sharded(), "src: bf16 TILE"
        P = len(write_idxs)
        _, nkv, s_pad, head_dim = (int(v) for v in src.padded_shape)
        B = int(batch)
        assert nkv == num_kv_heads, (nkv, num_kv_heads)
        assert s_pad >= B * P and s_pad % TILE == 0, (s_pad, B, P)
        modulo = int(cache_position_modulo) if cache_position_modulo else 0
        reason = cls.unsupported_reason(
            cache,
            write_idxs,
            page_table,
            batch=B,
            packed_p=P,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            modulo=modulo or None,
        )
        assert reason is None, reason
        device = cache.device()
        Wt = head_dim // TILE
        St = block_size // TILE
        J = min(P, 2)
        cache_tile = BF16_TILE if cache.dtype == ttnn.bfloat16 else 1088
        idx_stick = int(write_idxs[0].buffer_aligned_page_size())
        pt_stick = int(page_table.buffer_aligned_page_size())
        pt_entries = int(page_table.padded_shape[1])
        jobs_page = _align((2 * J + 3 * P) * 4, 64)

        grid = compute_grid(device)
        n_cores, per = plan_paged_update_rows(grid, B, nkv)
        cores = row_major_cores(grid, n_cores)
        core_set = core_range_set(cores)

        def cb(idx, pages, page, fmt, extra=()):
            fds = [ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page) for i in (idx, *extra)]
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=core_set, format_descriptors=fds)

        cbs = [
            cb(CB_SRC, N_SRC * Wt, BF16_TILE, ttnn.bfloat16),
            cb(CB_UNT_SRC, N_SRC * Wt, BF16_TILE, ttnn.bfloat16),
            cb(CB_CACHE, 2 * Wt, cache_tile, cache.dtype),
            # the untilized cache tile row and its patched view: one L1 region under two indices (the stock's
            # untilized_cache / untilized_cache2 alias)
            cb(CB_UNT_A, 2 * Wt, BF16_TILE, ttnn.bfloat16, extra=(CB_UNT_B,)),
            cb(CB_OUT, 2 * Wt, cache_tile, cache.dtype),
            cb(CB_IDX, P, idx_stick, ttnn.int32),
            cb(CB_PT, 1, pt_stick, ttnn.int32),
            cb(CB_JOBS, 2, jobs_page, ttnn.int32),
        ]
        ct = [
            CB_SRC,
            CB_UNT_SRC,
            CB_CACHE,
            CB_UNT_A,
            CB_UNT_B,
            CB_OUT,
            CB_IDX,
            CB_PT,
            CB_JOBS,
            B,
            P,
            Wt,
            St,
            nkv,
            int(block_size),
            modulo,
            s_pad // TILE,
            J,
            idx_stick,
            pt_stick,
            pt_entries,
            cache_tile,
        ]
        for t in (src, write_idxs[0], page_table, cache):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        idx_addrs = [t.buffer_address() for t in write_idxs]
        rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        start = 0
        for i, core in enumerate(cores):
            units = []
            for u in range(start, start + per[i]):
                units += [u // nkv, u % nkv]
            start += per[i]
            rt_r[core.x][core.y] = [src.buffer_address(), page_table.buffer_address(), cache.buffer_address()]
            rt_r[core.x][core.y] += idx_addrs + [per[i]] + units
            rt_c[core.x][core.y] = [per[i]]
            rt_w[core.x][core.y] = [cache.buffer_address(), per[i]]
        # the stock paged_update_cache's compute config at the loop's call (no compute_kernel_config): LoFi /
        # approximate / 16-bit dest / double-buffered dest -- only the dest width and sync mode reach the
        # untilize / tilize paths, and the bits depend on them
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, dst_full_sync_en=False
        )

        def kernel(name, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / name),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        kernels = [
            kernel("reader_paged_update_rows.cpp", rt_r, ttnn.ReaderConfigDescriptor()),
            kernel("compute_paged_update_rows.cpp", rt_c, compute_cfg),
            kernel("writer_paged_update_rows.cpp", rt_w, ttnn.WriterConfigDescriptor()),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([src, *write_idxs, page_table, cache], program)
        return cache
