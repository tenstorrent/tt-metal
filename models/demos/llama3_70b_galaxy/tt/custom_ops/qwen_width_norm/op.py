# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Width-split local RMSNorm of the Qwen context-parallel residual, one generic_op program per device.

Replaces qwen_cp_prefill.local_norm (2 width slices + rms_norm_pre_all_gather x2 + stats concat +
rms_norm_post_all_gather x2 + output concat: 8 ops) for the layout-B residual [1, 1, R, W] bf16 -> bf8.

Core (cx, cy) of the gx x gy compute grid owns width chunk cx (whole blocks of 4 tiles; the first Wt/4 % gx
chunks take one more block) of the tile rows of row group cy (the first Rt % gy groups take one more row). Per
tile row it computes the partial mean(x^2) over its chunk (fp32), writes it to slot cx of every row-group core's
CB_GATH and bumps their semaphore; each core adds the gx partials in slot order (the same order everywhere),
then rstd = rsqrt(sum + eps), y = x * rstd * gamma, packed to bf8. x is read once.

Numerics: not bit-identical to the stock pair. The stock pre op rounds each half's sum(x^2) to bf16 stats; here
the partials stay fp32 (added on the FPU, TF32 operands) until rsqrt.

Contract (every device runs the same program):
    x       [1, 1, R, W] bf16 or bf8 TILE DRAM interleaved (the first layer's residual is bf8), R % 32 == 0, W % 128 == 0, W / 128 >= gx
    gamma   [1, 1, W / 32, 32] bf16 ROW_MAJOR DRAM interleaved (DistributedNorm.norm.weight)
    returns [1, 1, R, W] bf8 TILE DRAM interleaved (new tensor)
"""

from __future__ import annotations

import struct

import ttnn

KERNEL_ROOT = "models/demos/llama3_70b_galaxy/tt/custom_ops/qwen_width_norm/kernels"
DB = 4
CB_X, CB_SCALER, CB_EPS, CB_GAMMA, CB_XSQ, CB_PART, CB_RSTD, CB_FUSION, CB_GATH, CB_ACC, CB_OUT = (
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    16,
)
SEM_RECV = 0

_CACHE: dict = {}
_CACHE_MAX = 1024


def _tile_bytes(dtype) -> int:
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}[dtype]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _accessor(t) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    """(start, count) of `parts` near-equal pieces of n (the first n % parts take one more)."""
    base, extra = divmod(n, parts)
    out, s = [], 0
    for i in range(parts):
        c = base + (1 if i < extra else 0)
        out.append((s, c))
        s += c
    return out


def width_chunks(wt: int, gx: int) -> list[tuple[int, int]]:
    """(first tile, tiles) of each width chunk: whole blocks of DB tiles."""
    return [(s * DB, c * DB) for s, c in _split(wt // DB, gx)]


def row_groups(rt: int, gy: int) -> list[tuple[int, int]]:
    """(first tile row, tile rows) of each row group."""
    return _split(rt, gy)


_NOC_TABLES: dict = {}


def _noc_tables(device, gx, gy):
    """NOC x of the grid columns and NOC y of the grid rows (translated coords: one table for every device of the
    mesh), checked once per (mesh, grid) to be a product grid."""
    key = (id(device), gx, gy)
    if key not in _NOC_TABLES:
        nx = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(cx, 0)).x) for cx in range(gx)]
        ny = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(0, cy)).y) for cy in range(gy)]
        for cy in range(gy):
            for cx in range(gx):
                p = device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
                if (int(p.x), int(p.y)) != (nx[cx], ny[cy]):
                    raise ValueError(f"worker grid is not a product grid at ({cx}, {cy}): {p}")
        _NOC_TABLES[key] = (nx, ny)
    return _NOC_TABLES[key]


def _build(device, x, gamma, out, eps, grid, fidelity):
    R, W = int(x.shape[-2]), int(x.shape[-1])
    rt, wt = R // 32, W // 32
    gx, gy = grid
    chunks = width_chunks(wt, gx)
    maxw = max(c for _, c in chunks)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})

    def cb(idx, tiles, dtype):
        size = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=tiles * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)],
        )

    bf16, bf8, f32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
    cbs = [
        cb(CB_X, 2 * maxw, x.dtype),
        cb(CB_SCALER, 1, bf16),
        cb(CB_EPS, 1, bf16),
        cb(CB_GAMMA, maxw, bf16),
        cb(CB_XSQ, maxw, f32),
        cb(CB_PART, 2, f32),
        cb(CB_RSTD, 1, f32),
        cb(CB_FUSION, 2 * DB, f32),
        cb(CB_GATH, 2 * gx, f32),
        cb(CB_ACC, 2, f32),
        cb(CB_OUT, 2 * DB, bf8),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_RECV, core_ranges=cores, initial_value=0)]

    noc_x, noc_y = _noc_tables(device, gx, gy)
    # wnorm_common.hpp prefix; the per-core work comes from the logical coords in the kernels (no per-core args)
    common_ct = [wt, gx, gy, rt, maxw, _f32_bits(eps), _f32_bits(1.0 / W)]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=common_ct + _accessor(x) + _accessor(gamma),
            common_runtime_args=[x.buffer_address(), gamma.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=common_ct,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                bfp8_pack_precise=False,
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=common_ct + _accessor(out) + noc_x + noc_y,
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def qwen_width_norm(x, gamma, *, eps: float, grid=None, fidelity=ttnn.MathFidelity.HiFi2):
    """RMSNorm of x over its last dim with gamma, bf16 in, bf8 out (see the module docstring)."""
    s = tuple(x.shape)
    if len(s) != 4 or s[0] != 1 or s[1] != 1 or s[2] % 32 or s[3] % (32 * DB):
        raise ValueError(f"x must be [1, 1, R, W] with R % 32 == 0 and W % {32 * DB} == 0, got {s}")
    if x.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b) or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"x must be bfloat16 or bfloat8_b TILE, got {x.dtype} {x.layout}")
    mc = x.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"x must be DRAM interleaved, got {mc}")
    wt = s[3] // 32
    if tuple(gamma.shape) != (1, 1, wt, 32) or gamma.dtype != ttnn.bfloat16 or gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"gamma must be row-major bf16 [1, 1, {wt}, 32], got {tuple(gamma.shape)} {gamma.dtype}")
    device = x.device()
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = (int(g.x), int(g.y))
    gx, gy = grid
    if gx < 2 or wt // DB < gx:
        raise ValueError(f"need 2 <= gx <= W / {32 * DB}, got gx={gx}, W={s[3]}")
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(s)), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    key = (
        tuple(device.shape),
        str(x.dtype),
        grid,
        s,
        _f32_bits(eps),
        str(fidelity),
        x.buffer_address(),
        gamma.buffer_address(),
        out.buffer_address(),
    )
    mp = _CACHE.get(key)
    if mp is None:
        mp = ttnn.MeshProgramDescriptor()
        rows, cols = list(device.shape)
        mp[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))] = _build(
            device, x, gamma, out, eps, grid, fidelity
        )
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = mp
    return ttnn.generic_op([x, gamma, out], mp)
