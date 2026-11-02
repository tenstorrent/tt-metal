# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ternary decode matmul (A3) on deepseek_v3_b1's DRAM streaming skeleton.

y [1, N] = x [1, K] @ W [K, N], M=1. x is a 1x32 tiny-tile row replicated on every
worker; W is width-sharded over the DRAM banks with each bank's tiles reordered so a
K-stick per output column is contiguous; each worker streams its own columns.

Step (a), this file: codes only, with ``workers_per_bank`` in {1, 2}. With 2 workers a
bank's columns are split in halves and the second worker reads from an address
offset, so the reader is unchanged.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import get_max_page_size_and_num_pages
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

KERNEL = "models/demos/blackhole/bonsai2/native/a3/kernels/ternary_mm_kernel.cpp"


# Cores kept free of the A3 workers' persistent L1 buffer: the model's final RMSNorm runs on
# (0, 0) alone with ~1.5 MB of static CBs and clashes with anything resident there.
AVOID = frozenset({(0, 0)})


def worker_cores_rect(device, workers_per_bank: int):
    """Workers as two exact rectangles (logical cols 0-1 and 7-8, rows 1-8; 32 cores at 4 per bank),
    each bank taking the rows nearest its optimal reader. A scattered set is many rectangles for the
    dispatcher; this tests whether that is the ~7 us gap before every A3 op in the traced profile."""
    assert workers_per_bank == 4
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    blocks = {"L": [(x, y) for y in range(1, 9) for x in (0, 1)], "R": [(x, y) for y in range(1, 9) for x in (7, 8)]}
    out = []
    for b, c in sorted(enumerate(primary), key=lambda t: t[1].y):
        free = blocks["L" if c.x < 4 else "R"]
        pick = sorted(free, key=lambda p: (abs(p[1] - c.y), p[0]))[:4]
        for sub, (x, y) in enumerate(pick):
            free.remove((x, y))
            out.append((ttnn.CoreCoord(x, y), b, sub))
    return out


def worker_cores(device, workers_per_bank: int, avoid=AVOID):
    import os

    if os.environ.get("BONSAI_A3_RECT", "1") == "1" and workers_per_bank == 4:  # default; =0 for the scattered set
        return worker_cores_rect(device, workers_per_bank)
    """[(core, bank, sub)] -- the optimal reader per bank, plus a neighbour per extra worker."""
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    used = set(avoid) | {(c.x, c.y) for c in primary}
    out = [(ttnn.CoreCoord(c.x, c.y), b, 0) for b, c in enumerate(primary) if (c.x, c.y) not in avoid]
    subs = [(b, c, 0) for b, c in enumerate(primary) if (c.x, c.y) in avoid]
    subs += [(b, c, s) for s in range(1, workers_per_bank) for b, c in enumerate(primary)]
    for b, c, sub in subs:
        near = sorted(((dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if dx or dy),
                      key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0])))
        for dx, dy in near:
            x, y = c.x + dx, c.y + dy
            if 0 <= x < grid.x and 0 <= y < grid.y and (x, y) not in used:
                used.add((x, y))
                out.append((ttnn.CoreCoord(x, y), b, sub))
                break
        else:
            raise RuntimeError(f"no free neighbour for bank {b} at {c}")
    return out


def core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def shuffle_for_banks(w, banks: int):
    """[K, N] -> same shape, each bank shard's tiles reordered column-major (K-sticks)."""
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    return shuffle_tensor_tiles(w, 32, banks)


def weight_mem(device, k: int, n: int):
    banks = device.dram_grid_size().x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                       [k, n // banks], ttnn.ShardOrientation.ROW_MAJOR))


def act_mem(cores, k: int):
    """x replicated: one [1, K] shard per worker (the tensor is [workers, K])."""
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(core_set(cores), [1, k], ttnn.ShardOrientation.ROW_MAJOR))


def out_mem(cores, n: int):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(core_set(cores), (1, n // len(cores)), ttnn.ShardOrientation.ROW_MAJOR))


_progs = {}


def clear_cache():
    """Call after freeing any tensor a cached program was built on: the descriptor's CBs
    hold that tensor's Buffer, and ttnn's generic_op cache hit re-binds it (segfault)."""
    _progs.clear()


def _sig(t):
    """Cache key part: a freed tensor's address is reused by the next allocation, so the
    dtype and shape go in too (a bfp4 weight landing on a freed bfp2 one segfaulted)."""
    return (t.buffer_address(), str(t.dtype), tuple(t.shape))


def ternary_mm(x, w, y, workers, subblock_k: int):
    key = (_sig(x), _sig(w), _sig(y), subblock_k, len(workers))
    if key not in _progs:
        _progs[key] = _ternary_mm_prog(x, w, y, workers, subblock_k)
    ttnn.generic_op([x, w, y], _progs[key])
    return y


def _ternary_mm_prog(x, w, y, workers, subblock_k: int):
    """x: [len(workers), K] 1x32 tiles, height-sharded (act_mem). w: shuffled, weight_mem.
    y: [1, N] 1x32 tiles, out_mem. ``workers`` from worker_cores()."""
    device = x.device()
    k = x.memory_config().shard_spec.shape[1]
    kt = k // 32
    banks = device.dram_grid_size().x
    wpb = len(workers) // banks
    pcn = w.memory_config().shard_spec.shape[1] // 32 // wpb
    assert kt % subblock_k == 0
    tile_bytes = w.get_tile().get_tile_size(w.dtype)
    page, pages = get_max_page_size_and_num_pages(device, subblock_k, tile_bytes)
    cores = [c for c, _, _ in workers]
    grid = core_set(cores)
    max_dest = 4  # fp32 Dst, half sync
    sbw = max(d for d in range(1, max_dest + 1) if pcn % d == 0)

    cb_in1 = ttnn.CBDescriptor(
        total_size=3 * subblock_k * tile_bytes, core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=w.dtype, page_size=tile_bytes,
                                                    tile=ttnn.TileDescriptor(w.get_tile()))])
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, x), cb_in1, ttnn.cb_descriptor_from_sharded_tensor(4, y)]

    vcs, seen = [], []
    for core, bank, _ in workers:
        vc = bank & 0x3
        if any(c.y == core.y and (b & 0x3) == vc for c, b in seen):
            vc = (vc + 1) & 0x3
        seen.append((core, bank))
        vcs.append((core, vc))
    common = [("tm_cb_in0", 0), ("tm_cb_in1", 1), ("tm_cb_out", 4), ("tm_subblock_k", subblock_k),
              ("tm_per_core_n", pcn), ("tm_num_subblocks_k", kt // subblock_k)]
    uk = UnifiedKernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=grid,
        ncrisc_named_compile_time_args=common + [
            ("tm_in1_page_size", page), ("tm_in1_num_pages", pages),
            ("tm_in1_block_bytes", subblock_k * tile_bytes), ("tm_num_tiles_k", kt)],
        brisc_named_compile_time_args=[],
        trisc_named_compile_time_args=common + [("tm_subblock_w", sbw), ("tm_tile_r_dim", 1)],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
        unified_compile_time_core_descriptors=[UnifiedCompileTimeCoreDescriptor(
            named_compile_time_arg="tm_is_active_core", core_range=grid, value=1, other_value=0)],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor(named_compile_time_arg="tm_bank_id",
                                         core_values=[(c, b) for c, b, _ in workers], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="tm_vc", core_values=vcs, other_value=0),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="tm_in1_addr",
                core_values=[(c, w.buffer_address() + s * pcn * kt * tile_bytes) for c, _, s in workers],
                other_value=0),
        ],
    )
    return ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels, semaphores=[], cbs=cbs)


# ---------------------------------------------------------------------------------------------
# Step (b): per-128-group scales applied inside the kernel.

KERNEL_SCALED = "models/demos/blackhole/bonsai2/native/a3/kernels/ternary_mm_scaled_kernel.cpp"
GROUP = 128  # weights per scale
SUPER = 8  # groups per custom_mm pass (rows of the masked in0 tile)


def mask_x(x):
    """x [K] -> [8, K]: row r keeps the groups g with g % 8 == r (block-masked in0)."""
    import torch

    k = x.shape[-1]
    rows = (torch.arange(k) // GROUP) % SUPER
    return torch.where(rows[None, :] == torch.arange(SUPER)[:, None], x.reshape(1, k), torch.zeros(1, k))


def scale_rows(s):
    """s [K/128, N] -> [s_tiles*32, N] for the kernel's DST layout: super-group G's 8 rows
    sit at rows (G%4)*8 of 32x32 tile G//4; the bank shuffle then makes each n's tiles contiguous."""
    import torch

    groups, n = s.shape
    num_g = groups // SUPER
    s_tiles = (num_g + 3) // 4
    out = torch.zeros(s_tiles * 32, n, dtype=torch.float32)
    for G in range(num_g):
        r0 = (G // 4) * 32 + (G % 4) * 8
        out[r0:r0 + SUPER] = s[G * SUPER:(G + 1) * SUPER]
    return out


def ternary_mm_scaled(x8, w, s, y, workers):
    key = ("s", _sig(x8), _sig(w), _sig(s), _sig(y), len(workers))
    if key not in _progs:
        _progs[key] = _ternary_mm_scaled_prog(x8, w, s, y, workers)
    ttnn.generic_op([x8, w, s, y], _progs[key])
    return y


def _ternary_mm_scaled_prog(x8, w, s, y, workers):
    """x8: [workers*8, K] 8x32 tiles, height-sharded [8, K] per worker (mask_x per shard).
    w: codes, shuffle_for_banks + weight_mem. s: scale_rows(...), shuffled, fp32 TILE, weight_mem-style.
    y: [1, N] 1x32 tiles, out_mem."""
    device = x8.device()
    k = x8.memory_config().shard_spec.shape[1]
    kt = k // 32
    assert kt % 32 == 0, "K must be a multiple of 1024 (8 groups of 128)"
    num_g = kt // 32
    s_tiles = (num_g + 3) // 4
    banks = device.dram_grid_size().x
    wpb = len(workers) // banks
    pcn = w.memory_config().shard_spec.shape[1] // 32 // wpb
    tile_bytes = w.get_tile().get_tile_size(w.dtype)
    page, pages = get_max_page_size_and_num_pages(device, 32, tile_bytes)
    grid = core_set([c for c, _, _ in workers])

    def cb(idx, fmt, nbytes, tile_bytes_, tile):
        return ttnn.CBDescriptor(total_size=nbytes, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=fmt, page_size=tile_bytes_, tile=ttnn.TileDescriptor(tile))])

    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, x8),
           cb(1, w.dtype, 3 * 32 * tile_bytes, tile_bytes, w.get_tile()),
           cb(2, ttnn.float32, 2 * s_tiles * 4096, 4096, ttnn.Tile([32, 32])),
           ttnn.cb_descriptor_from_sharded_tensor(4, y)]

    vcs, seen = [], []
    for core, bank, _ in workers:
        vc = bank & 0x3
        if any(c.y == core.y and (b & 0x3) == vc for c, b in seen):
            vc = (vc + 1) & 0x3
        seen.append((core, bank))
        vcs.append((core, vc))
    common = [("tm_per_core_n", pcn), ("tm_num_subblocks_k", num_g), ("tm_num_tiles_k", kt)]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                       math_approx_mode=False, dst_full_sync_en=True)
    cfg.unpack_to_dest_mode = modes
    per_core = lambda name, vals: PerCoreCompileTimeDescriptor(  # noqa: E731
        named_compile_time_arg=name, core_values=vals, other_value=0)
    uk = UnifiedKernelDescriptor(
        kernel_source=KERNEL_SCALED,
        core_ranges=grid,
        ncrisc_named_compile_time_args=common + [
            ("tm_in1_page_size", page), ("tm_in1_num_pages", pages), ("tm_in1_block_bytes", 32 * tile_bytes)],
        brisc_named_compile_time_args=common,
        trisc_named_compile_time_args=common,
        trisc_compute_config=cfg,
        unified_compile_time_core_descriptors=[UnifiedCompileTimeCoreDescriptor(
            named_compile_time_arg="tm_is_active_core", core_range=grid, value=1, other_value=0)],
        per_core_compile_time_descriptors=[
            per_core("tm_bank_id", [(c, b) for c, b, _ in workers]),
            per_core("tm_vc", vcs),
            per_core("tm_in1_addr", [(c, w.buffer_address() + i * pcn * kt * tile_bytes) for c, _, i in workers]),
            per_core("tm_s_addr", [(c, s.buffer_address() + i * pcn * s_tiles * 4096) for c, _, i in workers]),
        ],
    )
    return ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels, semaphores=[], cbs=cbs)


# ---------------------------------------------------------------------------------------------
# Input prep: x' = H(D x) scattered into the A3 workers' persistent block-masked buffer.

KERNEL_PREP = "models/demos/blackhole/bonsai2/native/a3/kernels/prep_hadamard_kernel.cpp"


def hadamard_blocks(had, width: int):
    """[width, 1024] float32: rows of block b are diag(d_b) H / 32, so x_b @ M_b = rotate(x)_b."""
    import numpy as np
    import torch

    from models.demos.blackhole.bonsai2.native.hadamard import sylvester

    h = sylvester(had.block).astype(np.float32)  # already normalized: entries +-1/32
    d = had.signs[width].astype(np.float32).reshape(-1, had.block, 1)
    return torch.from_numpy((d * h[None]).reshape(width, had.block))


def hadamard_tensor(device, had, width: int):
    """M as bfp2 (+-1/32 is exact), laid out so each prep unit (b, r) is one contiguous run:
    a one-tile-wide [U*128*32, 32] tensor, height-sharded over the DRAM banks, where unit
    u = 8 b + r holds tiles M[32 b + i, j] for j in 4r..4r+3, i in 0..31 (j-major). One 40 KB
    read per unit instead of 128 scattered 320 B pages (which cost 15 of 24 us)."""
    import torch

    mb = hadamard_blocks(had, width)  # [width, 1024]
    nb = width // 1024
    t = mb.reshape(nb, 32, 32, 32, 32).permute(0, 3, 1, 2, 4)  # [b, j, i, row, col]
    t = t.reshape(nb, 8, 4, 32, 32, 32)  # [b, r, jj, i, row, col]
    t = t.reshape(nb * 8 * 128 * 32, 32)
    banks = device.dram_grid_size().x
    units = nb * 8
    assert units % banks == 0
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM,
                            ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                                             ttnn.CoreCoord(banks - 1, 0))}),
                                           [units // banks * 128 * 32, 32], ttnn.ShardOrientation.ROW_MAJOR))
    return ttnn.to_device(ttnn.from_torch(t.contiguous(), dtype=ttnn.DataType.BFLOAT2_B, layout=ttnn.TILE_LAYOUT),
                          device, memory_config=mem)


def masked_input_buffer(device, workers, k: int):
    """The A3 input: [workers*8, K] 8x32 tiles, one [8, K] shard per worker, no fill
    (prep() writes every tile whole). Allocate per call and free once A3 has read it."""
    cores = [c for c, _, _ in workers]
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(core_set(cores), [8, k], ttnn.ShardOrientation.ROW_MAJOR))
    spec = ttnn.TensorSpec(ttnn.Shape([1, 1, 8 * len(workers), k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mem,
                           ttnn.Tile([8, 32]))
    return ttnn.allocate_tensor_on_device(spec, device)


def prep_cores(device, num_units: int, units_per_core: int):
    grid = device.compute_with_storage_grid_size()
    n = -(-num_units // units_per_core)
    assert n <= grid.x * grid.y
    return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]


def prep(x, m, x8, workers, src_tile32: bool = True, units_per_core: int = 0, scatter: bool = True, debug: int = 0):
    """x: [.., 32, K] bf16 32x32 tiles (row 0 live) or [1, K] 1x32 tiles, interleaved.
    m: hadamard_tensor for width K. x8: masked_input_buffer. Writes x8 in place."""
    assert not (src_tile32 and x.memory_config().buffer_type == ttnn.BufferType.DRAM), \
        "32x32-tile x must be in L1: its 32 B row reads from DRAM land with the wrong 64 B phase"
    device = x.device()
    k = x.shape[-1]
    nb = k // 1024
    units = [(b, r) for b in range(nb) for r in range(8)]
    if not units_per_core:  # fewest per core that fits the grid
        g = device.compute_with_storage_grid_size()
        units_per_core = -(-len(units) // (g.x * g.y))
    cores = prep_cores(device, len(units), units_per_core)
    grid = core_set(cores)
    per = [units[i * units_per_core:(i + 1) * units_per_core] for i in range(len(cores))]
    m_page = m.get_tile().get_tile_size(m.dtype)
    upb = len(units) // device.dram_grid_size().x  # units per DRAM bank shard
    wxy = []
    for c, _, _ in workers:
        v = device.worker_core_from_logical_core(c)
        wxy.append((v.x << 16) | v.y)
    flat = lambda us: [v for u in us for v in u]  # noqa: E731

    def cb(idx, fmt, tile, ntiles):
        size = tile.get_tile_size(fmt)
        return ttnn.CBDescriptor(total_size=ntiles * size, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=fmt, page_size=size, tile=ttnn.TileDescriptor(tile))])

    t1 = ttnn.Tile([1, 32])
    cbs = [cb(0, ttnn.bfloat16, t1, 32), cb(1, m.dtype, ttnn.Tile([32, 32]), 64), cb(2, ttnn.bfloat16, t1, 2),
           cb(3, ttnn.bfloat16, ttnn.Tile([8, 32]), 1)]
    x_is_dram = x.memory_config().buffer_type == ttnn.BufferType.DRAM
    uk = UnifiedKernelDescriptor(
        kernel_source=KERNEL_PREP,
        core_ranges=grid,
        ncrisc_named_compile_time_args=[("pp_src_tile32", int(src_tile32)), ("pp_x_is_dram", int(x_is_dram)),
                                        ("pp_m_page", m_page), ("pp_debug", debug), ("pp_units_per_bank", upb)],
        brisc_named_compile_time_args=[],
        trisc_named_compile_time_args=[],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=[(c, [x.buffer_address(), m.buffer_address(), len(us)] + flat(us)) for c, us in zip(cores, per)],
            brisc_args=[(c, [x8.buffer_address(), len(us)] + flat(us) + [len(wxy) if scatter else 0] + wxy) for c, us in zip(cores, per)],
            trisc_args=[(c, [len(us)]) for c, us in zip(cores, per)],
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([x, m, x8], prog)
    return x8


# ---------------------------------------------------------------------------------------------
# v2: runtime addresses (one compile for all layers), fused gate/up, interleaved output.

KERNEL_V2 = "models/demos/blackhole/bonsai2/native/a3/kernels/ternary_mm_v2_kernel.cpp"


def ternary_linear(x8, w, s, out, workers, gu: bool = False, k: int = 0, cache: bool = True, n_real: int = 0):
    """cache=False when x8 is a fresh allocation: a cached descriptor would re-bind a freed buffer."""
    if not cache:
        ttnn.generic_op([x8, _t(w), _t(s), out], _v2_prog(x8, w, s, out, workers, gu, k, n_real))
        return out
    """x8: masked_input_buffer filled by prep(). w/s: from weights_for_device().
    out: L1 interleaved, [.., 32, N] 32x32 tiles (row 0 written, rest zero) or [1, 1, 1, N] 1x32 tiles.
    gu: w/s hold (gate, up) column pairs; out gets silu(gate) * up."""
    key = ("v2", _sig(x8), _sig(w), _sig(s), _sig(out), len(workers), gu, k)
    if key not in _progs:
        if len(_progs) > 256:  # outputs are fresh allocations, so keys churn; descriptors don't bind them
            _progs.clear()
        _progs[key] = _v2_prog(x8, w, s, out, workers, gu, k, n_real)
    ttnn.generic_op([x8, _t(w), _t(s), out], _progs[key])
    return out


def _v2_prog(x8, w, s, out, workers, gu, k, n_real=0):
    """n_real: output tiles past this index are padding columns of w and are not written."""
    device = x8.device()
    k = k or x8.memory_config().shard_spec.shape[1]  # x8 may be wider (one buffer serves every K)
    kt = k // 32
    assert kt % 32 == 0
    num_g = kt // 32
    s_tiles = (num_g + 3) // 4
    cols = 2 if gu else 1
    assert 3 + cols * s_tiles <= 8, "DST full-sync fp32 holds 8 tiles"
    banks = device.dram_grid_size().x
    wpb = len(workers) // banks
    packed = isinstance(w, PackedW)
    cols_per_core = (w.cols_per_bank if packed else w.memory_config().shard_spec.shape[1] // 32) // wpb
    pcn = cols_per_core // cols
    tile_bytes = PackedW.TILE_BYTES if packed else w.get_tile().get_tile_size(w.dtype)
    page, pages = get_max_page_size_and_num_pages(device, 32, tile_bytes)
    grid = core_set([c for c, _, _ in workers])
    out_tile32 = out.get_tile().tile_shape[0] == 32
    assert not out.is_sharded()
    out_is_dram = out.memory_config().buffer_type == ttnn.BufferType.DRAM

    def cb(idx, fmt, ntiles, tile):
        size = tile.get_tile_size(fmt)
        return ttnn.CBDescriptor(total_size=ntiles * size, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=fmt, page_size=size, tile=ttnn.TileDescriptor(tile))])

    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, x8),
           ttnn.CBDescriptor(total_size=3 * 32 * tile_bytes, core_ranges=grid, format_descriptors=[
               ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.DataType.BFLOAT2_B, page_size=tile_bytes,
                                       tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))]),
           cb(2, ttnn.float32, 2 * cols * s_tiles, ttnn.Tile([32, 32])),
           cb(4, ttnn.bfloat16, 2, ttnn.Tile([1, 32])),
           cb(5, ttnn.bfloat16, 1, ttnn.Tile([32, 32])),
           cb(6, ttnn.float32, 1, ttnn.Tile([32, 32]))]
    packed_s = isinstance(s, PackedS)
    tail_t = s.tail_t if packed_s else 1
    s_core_bytes = (cols_per_core // tail_t * s.tiles_per_group() if packed_s else cols_per_core * s_tiles) * 4096
    vcs, seen = [], []
    for core, bank, _ in workers:
        vc = bank & 0x3
        if any(c.y == core.y and (b & 0x3) == vc for c, b in seen):
            vc = (vc + 1) & 0x3
        seen.append((core, bank))
        vcs.append((core, vc))
    common = [("tm_per_core_n", pcn), ("tm_num_g", num_g), ("tm_gu", int(gu)), ("tm_is_active_core", 1),
              ("tm_shared_exp", int(packed)), ("tm_tail_t", tail_t)]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                       math_approx_mode=False, dst_full_sync_en=True)
    cfg.unpack_to_dest_mode = modes
    per_core = lambda name, vals: PerCoreCompileTimeDescriptor(  # noqa: E731
        named_compile_time_arg=name, core_values=vals, other_value=0)
    col0 = lambda b, i: (b * wpb + i)  # noqa: E731  this core's slot among all cores, in column order
    uk = UnifiedKernelDescriptor(
        kernel_source=KERNEL_V2,
        core_ranges=grid,
        ncrisc_named_compile_time_args=common + [
            ("tm_in1_page_size", page), ("tm_in1_num_pages", pages), ("tm_in1_block_bytes", 32 * tile_bytes)],
        brisc_named_compile_time_args=common + [("tm_out_tile32", int(out_tile32)), ("tm_out_is_dram", int(out_is_dram))],
        trisc_named_compile_time_args=common,
        trisc_compute_config=cfg,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=[(c, [w.buffer_address() + i * cols_per_core * kt * tile_bytes, b, vc])
                         for (c, b, i), (_, vc) in zip(workers, vcs)],
            brisc_args=[(c, [s.buffer_address() + i * s_core_bytes, out.buffer_address(),
                             col0(b, i) * pcn, n_real // 32 or (1 << 30), b]) for c, b, i in workers],
        ),
    )
    return ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels, semaphores=[], cbs=cbs)


class PackedW:
    """bfp2 code tiles without the exponent section (A2b): 256 B each, every exponent forced to
    127 (+-1.0) in the unpacker. Raw bytes, laid out per DRAM bank exactly as the bank-shuffled
    Bfp2_b tensor (column-major K-sticks); `t` is a uint32 row-major tensor height-sharded so each
    bank's bytes are contiguous in that bank."""

    TILE_BYTES = 256

    def __init__(self, t, k, n, banks):
        self.t, self.k, self.n = t, k, n
        self.cols_per_bank = n // 32 // banks
        self.dtype = ttnn.DataType.BFLOAT2_B
        self.shape = (k, n)

    def buffer_address(self):
        return self.t.buffer_address()


class PackedS:
    """fp32 scale tiles with shared tails (see the A3 v2 kernel): per column s_full full tiles, and
    one tail tile per tail_t consecutive columns; per DRAM bank, per group: [tail][main c0]...
    Raw bytes, uint32 row-major, height-sharded so each bank's bytes are contiguous."""

    def __init__(self, t, tail_t, s_full, rem, cols_per_bank):
        self.t, self.tail_t, self.s_full, self.rem, self.cols_per_bank = t, tail_t, s_full, rem, cols_per_bank
        self.dtype, self.shape = ttnn.float32, tuple(t.shape)

    def buffer_address(self):
        return self.t.buffer_address()

    def tiles_per_group(self):
        return self.tail_t * self.s_full + (1 if self.rem else 0)


def tail_group(num_g, cols_per_core):
    rem = num_g % 4
    if rem == 0:
        return 1
    return max(t for t in (4, 2, 1) if t * rem <= 4 and cols_per_core % t == 0)


def _fp32_tile_bytes(t):
    """[.., 32, 32] fp32 -> [.., 1024] in face order (0,0),(0,1),(1,0),(1,1), row-major faces."""
    lead = t.shape[:-2]
    return t.reshape(*lead, 2, 16, 2, 16).swapaxes(-3, -2).reshape(*lead, 1024)


def pack_scales(scales_gn, banks, tail_t):
    """scales [K/128, N] -> uint32 [banks * rows, 256] in the PackedS layout."""
    import numpy as np

    groups, n = scales_gn.shape
    num_g = groups // SUPER
    s_full, rem = num_g // 4, num_g % 4
    nt = n // 32
    s = np.asarray(scales_gn, dtype=np.float32).reshape(num_g, SUPER, nt, 32)  # G, row, col tile, col
    full = np.zeros((nt, max(s_full, 1), 32, 32), np.float32)
    for G in range(4 * s_full):
        full[:, G // 4, (G % 4) * 8:(G % 4) * 8 + 8, :] = s[G].transpose(1, 0, 2)
    groups_out = []
    for g0 in range(0, nt, tail_t):
        if rem:
            tail = np.zeros((32, 32), np.float32)
            for ti in range(tail_t):
                for r in range(rem):
                    b = (ti * rem + r) * 8
                    tail[b:b + 8] = s[4 * s_full + r, :, g0 + ti]
            groups_out.append(_fp32_tile_bytes(tail)[None])
        for c in range(g0, g0 + tail_t):
            if s_full:
                groups_out.append(_fp32_tile_bytes(full[c, :s_full]))
    flat = np.concatenate(groups_out).reshape(-1)  # all banks, in column order; banks own contiguous columns
    return flat.view(np.uint32).reshape(banks, -1).reshape(-1, 256)


def _t(w):
    return w.t if isinstance(w, (PackedW, PackedS)) else w


def pack_shared_exp(codes_kn, banks):
    """codes [K, N] in {-1,0,1} -> uint32 [banks * rows, 256] of 256 B tiles, per bank: for each of
    its N-tiles, the K-tiles in order (the DRAM-streaming stick order)."""
    import numpy as np

    from models.demos.blackhole.bonsai2.native.bfp2 import pack_tiles

    k, n = codes_kn.shape
    kt, nt = k // 32, n // 32
    pnb = nt // banks
    c = np.asarray(codes_kn, dtype=np.int8).reshape(kt, 32, banks, pnb, 32)
    c = c.transpose(2, 3, 0, 1, 4).reshape(-1, 32, 32)  # bank, n_local, k: tiles in stream order
    data = pack_tiles(c, shared_exp=True).reshape(banks, -1)  # [banks, pnb * kt * 256] bytes
    return data.view(np.uint32).reshape(-1, 256)


def weights_for_device(device, codes_kn, scales_gn=None, cache=None, shared_exp=None, kn=None, workers_per_bank=4):
    """codes [K, N] in {-1,0,1}, scales [K/128, N] -> (codes, PackedS scales) in DRAM.
    codes_kn may instead be a callable returning (codes, scales), only called on a cache miss (a warm
    load never decodes the GGUF); then kn=(K, N) must be given. cache: path prefix for the host
    .tensorbin files (written once, never rewritten: other worktrees hardlink them).
    Codes are PackedW 256 B tiles (A2b) by default, or a Bfp2_b tensor with BONSAI_A2B=0."""
    import os

    import numpy as np
    import torch

    if shared_exp is None:  # A2b 256 B code tiles by default; BONSAI_A2B=0 for the 320 B Bfp2_b tiles
        shared_exp = os.environ.get("BONSAI_A2B", "1") == "1"
    banks = device.dram_grid_size().x
    k, n = kn if kn is not None else codes_kn.shape
    rows = {}

    def get():
        if not rows:
            c, sc = codes_kn() if callable(codes_kn) else (codes_kn, scales_gn)
            rows["c"], rows["s"] = c, sc
        return rows["c"], rows["s"]

    def host(name, build, dtype, layout=ttnn.TILE_LAYOUT):
        path = f"{cache}.{name}.tensorbin" if cache else None
        if path and os.path.exists(path):
            return ttnn.load_tensor(path)
        t = ttnn.from_torch(build(), dtype=dtype, layout=layout)
        if path:
            ttnn.dump_tensor(path, t)
        return t

    def banked(t):
        """uint32 [banks * r, 256] row-major -> DRAM, each bank's rows contiguous in that bank."""
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM,
                                ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                                                 ttnn.CoreCoord(banks - 1, 0))}),
                                               [t.shape[-2] // banks, 256], ttnn.ShardOrientation.ROW_MAJOR))
        return ttnn.to_device(t, device, memory_config=mem)

    num_g = k // 32 // 32
    tail_t = tail_group(num_g, n // 32 // banks // workers_per_bank)
    if os.environ.get("BONSAI_SCALE_TAILS", "1") == "0":  # one tail tile per column (the pre-tail byte count)
        tail_t = 1
    s2 = host(f"s2t{tail_t}", lambda: torch.from_numpy(
        pack_scales(np.asarray(get()[1], dtype=np.float32), banks, tail_t).view(np.int32)), ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT)
    s_dev = PackedS(banked(s2), tail_t, num_g // 4, num_g % 4, n // 32 // banks)
    if shared_exp:
        p = host("w256", lambda: torch.from_numpy(pack_shared_exp(np.asarray(get()[0]), banks).view(np.int32)),
                 ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        return PackedW(banked(p), k, n, banks), s_dev

    def build_w():
        c = torch.as_tensor(get()[0], dtype=torch.float32)
        return shuffle_for_banks(c.reshape(1, 1, *c.shape), banks)

    w = host("w", build_w, ttnn.DataType.BFLOAT2_B)
    return ttnn.to_device(w, device, memory_config=weight_mem(device, k, n)), s_dev


def interleave_cols(a, b):
    """[R, N] x2 -> [R, 2N] with 32-column tiles alternating a, b (the GU column-pair stream)."""
    import numpy as np

    r, n = a.shape
    return np.stack([a.reshape(r, n // 32, 32), b.reshape(r, n // 32, 32)], axis=2).reshape(r, 2 * n)


# ---------------------------------------------------------------------------------------------
# Fused decode RMSNorm + prep at the 5120 sites (kernels/rmsprep_kernel.cpp).

KERNEL_RMSPREP = "models/demos/blackhole/bonsai2/native/a3/kernels/rmsprep_kernel.cpp"


def norm_gamma(device, w):
    """(1 + w) as fp32 1x32 tiles [1, 1, 1, K] in DRAM (128 B pages) for rmsprep."""
    import torch

    g = (1.0 + torch.as_tensor(w, dtype=torch.float32)).reshape(1, 1, 1, -1)
    return ttnn.from_torch(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([1, 32]), device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def rmsprep(x, g, m, x8, workers, eps):
    """x: residual [.., 32, K] bf16 32x32 tiles (row 0 live), L1 or DRAM; g: norm_gamma; m: hadamard_tensor(K).
    Writes x8 = mask(H(D (rmsnorm(x) * g))) on every A3 worker, like rmsnorm -> prep."""
    import struct

    device = x.device()
    k = x.shape[-1]
    nb = k // 1024
    units = [(b, r) for b in range(nb) for r in range(8)]
    excl = {(c.x, c.y) for c, _, _ in workers}
    grid_sz = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid_sz.x, i // grid_sz.x) for i in range(grid_sz.x * grid_sz.y)]
    cores = [c for c in cores if (c.x, c.y) not in excl][:len(units)]
    assert len(cores) == len(units), "one unit per core"
    grid = core_set(cores)
    m_page = m.get_tile().get_tile_size(m.dtype)
    upb = len(units) // device.dram_grid_size().x
    vxy = lambda c: (lambda v: (v.x << 16) | v.y)(device.worker_core_from_logical_core(c))  # noqa: E731
    wxy = [vxy(c) for c, _, _ in workers]
    cxy = [vxy(c) for c in cores]
    fbits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]  # noqa: E731

    def cb(idx, fmt, tile, ntiles):
        size = tile.get_tile_size(fmt)
        return ttnn.CBDescriptor(total_size=ntiles * size, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=fmt, page_size=size, tile=ttnn.TileDescriptor(tile))])

    t1, t32 = ttnn.Tile([1, 32]), ttnn.Tile([32, 32])
    cbs = [cb(0, ttnn.bfloat16, t1, 32), cb(1, m.dtype, t32, 64), cb(2, ttnn.bfloat16, t1, 2),
           cb(3, ttnn.bfloat16, ttnn.Tile([8, 32]), 1), cb(7, ttnn.float32, t1, 32), cb(8, ttnn.float32, t1, 1),
           cb(9, ttnn.float32, t1, 1), cb(10, ttnn.bfloat16, t1, 32), cb(11, ttnn.float32, t32, 1),
           cb(12, ttnn.float32, t32, 1)]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in (0, 1)]
    x_is_dram = x.memory_config().buffer_type == ttnn.BufferType.DRAM
    # rstd for the MATH thread: a per-call L1 shard at the same address on every prep core
    rb = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(
        ttnn.Shape([len(cores), 32]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                          ttnn.ShardSpec(grid, [1, 32], ttnn.ShardOrientation.ROW_MAJOR))), device)
    brisc = []
    for i, (c, (b, r)) in enumerate(zip(cores, units)):
        a = [x8.buffer_address(), b, r, rb.buffer_address(), cxy[0], int(i == 0), len(cores), nb, fbits(1.0 / k), fbits(eps), len(wxy)] + wxy
        brisc.append((c, a + (cxy if i == 0 else [])))
    uk = UnifiedKernelDescriptor(
        kernel_source=KERNEL_RMSPREP,
        core_ranges=grid,
        ncrisc_named_compile_time_args=[("rp_x_is_dram", int(x_is_dram)), ("rp_m_page", m_page),
                                        ("rp_units_per_bank", upb)],
        brisc_named_compile_time_args=[],
        trisc_named_compile_time_args=[],
        # full-sync DST: the SFPU steps address DST slots absolutely (half-sync alternates halves per acquire)
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False,
            dst_full_sync_en=True),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=[(c, [x.buffer_address(), m.buffer_address(), g.buffer_address(), b, r])
                         for c, (b, r) in zip(cores, units)],
            brisc_args=brisc,
            trisc_args=[(c, [rb.buffer_address()]) for c in cores],
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels, semaphores=sems, cbs=cbs)
    ttnn.generic_op([x, g, m, rb, x8], prog)
    ttnn.deallocate(rb)
    return x8
