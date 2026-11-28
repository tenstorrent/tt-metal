# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN fused device kernels for the linear branch (stage 4), run through `ttnn.generic_op`. The kernel sources are
in `kernels/` and compile at first use like any ttnn kernel; nothing in ttnn is rebuilt.

`delta_solve`: `vdn_factor_apply` (`linear_ops_openvdn`) as one kernel, one 128x128 matrix pair per core.

Precision is that of the composites they replace: fp32 operands split into bf16 hi/lo parts for every matmul,
accumulated in fp32 DST at HiFi4; sums and scalings in fp32 on the SFPU; fp32 tiles enter DST unpacked (exact).
"""

from __future__ import annotations

import math
import os

import torch

import ttnn

from . import linear_ops_openvdn as ops

KERNEL_DIR = "models/tt_dit/models/transformers/openvdn/kernels"
TILE = ttnn.TILE_SIZE
FP32_TILE_BYTES = TILE * TILE * 4
BF16_TILE_BYTES = TILE * TILE * 2
MAX_CBS = 64  # the unpack-to-dest vector covers Blackhole's buffer count


def enabled(name: str) -> bool:
    """`OPENVDN_KERNELS` (comma list, default all) selects which fused kernels replace their composites."""
    selected = os.environ.get("OPENVDN_KERNELS", "solve,scan,conv")
    return name in {s.strip() for s in selected.split(",")}


def _cores(device: ttnn.MeshDevice, count: int) -> tuple[list[ttnn.CoreCoord], ttnn.CoreRangeSet]:
    """The first `count` worker cores, row by row, and their range set."""
    grid = device.compute_with_storage_grid_size()
    assert count <= grid.x * grid.y, f"{count} work units need more than the {grid.x}x{grid.y} grid"
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    ranges = []
    for y in range(math.ceil(count / grid.x)):
        last = min(grid.x, count - y * grid.x) - 1
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(last, y)))
    return cores, ttnn.CoreRangeSet(ranges)


def _split_work(device: ttnn.MeshDevice, units: int) -> list[tuple[ttnn.CoreCoord, int, int]]:
    """`(core, first unit, unit count)` per core, units spread as evenly as the grid allows."""
    grid = device.compute_with_storage_grid_size()
    num_cores = min(units, grid.x * grid.y)
    cores, _ = _cores(device, num_cores)
    base, extra = divmod(units, num_cores)
    out, first = [], 0
    for i, core in enumerate(cores):
        n = base + (1 if i < extra else 0)
        out.append((core, first, n))
        first += n
    return out


def _cb(index: int, dtype: ttnn.DataType, tiles: int, cores: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    page = FP32_TILE_BYTES if dtype == ttnn.float32 else BF16_TILE_BYTES
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _compute_config(unpack_to_dest: list[int]) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * MAX_CBS
    for i in unpack_to_dest:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    return config


def _accessor_args(*tensors: ttnn.Tensor) -> list[int]:
    args = []
    for t in tensors:
        args += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    return args


def _empty(device: ttnn.MeshDevice, shape: list[int]) -> ttnn.Tensor:
    return ttnn.empty(shape, ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------------------------- delta-rule solve


def solve_start(max_eig: float) -> tuple[float, int]:
    """Newton-Schulz start `X0 = c I` and iteration count for SPD matrices with eigenvalues in `[1, max_eig]`.
    `c` is `2 / (1 + max_eig)` rounded down to bf16 (so `X0` splits exactly and `c lambda < 2` still holds); the count
    is `ops.newton_schulz_iterations`' rule for the resulting worst start `1 - c`."""
    c = torch.tensor(2.0 / (1.0 + max_eig), dtype=torch.float32)
    c = (c.view(torch.int32) & ~0xFFFF).view(torch.float32).item()
    r0, tol, extra = 1.0 - c, 1e-7, 3
    return c, max(1, math.ceil(math.log2(math.log(tol) / math.log(r0)))) + extra


def _solve_constants(device: ttnn.MeshDevice, c: float) -> ttnn.Tensor:
    def make():
        eye = torch.eye(TILE)
        return torch.stack([eye, torch.zeros(TILE, TILE), c * eye]).reshape(1, 3, TILE, TILE)

    return ops._constant(device, ("solve_constants", c), make, ttnn.float32)


def delta_solve(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    alpha_rows: ttnn.Tensor | None,
    *,
    max_eig: float,
    iterations: int | None = None,
) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
    """`vdn_factor_apply(alpha, a, b)` fused: `inv = (I + a)^-1`, returns `(Diag(alpha) inv, b inv)`.

    a, b: `[..., 128, 128]` fp32 tiled, interleaved. alpha_rows: `[..., 1, 128]` fp32 tiled (`ops.rows_to_heads`), or
    None (no transition). Eigenvalues of `I + a` must lie in `[1, max_eig]`. `iterations` overrides the
    Newton-Schulz count (tests)."""
    device = a.device()
    shape = list(a.shape)
    assert shape[-2:] == [128, 128] and list(b.shape) == shape, f"delta_solve: a {shape}, b {list(b.shape)}"
    assert a.dtype == ttnn.float32 and b.dtype == ttnn.float32
    units = math.prod(shape[:-2])
    has_alpha = alpha_rows is not None
    if has_alpha:
        assert list(alpha_rows.shape) == shape[:-2] + [1, 128] and alpha_rows.dtype == ttnn.float32

    c, default_iterations = solve_start(max_eig)
    iterations = default_iterations if iterations is None else iterations
    constants = _solve_constants(device, c)
    injection = _empty(device, shape)
    transition = _empty(device, shape) if has_alpha else injection
    alpha = alpha_rows if has_alpha else constants

    work = _split_work(device, units)
    _, cores = _cores(device, len(work))
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, first, n in work:
        reader_rt[core.x][core.y] = [
            a.buffer_address(),
            b.buffer_address(),
            alpha.buffer_address(),
            constants.buffer_address(),
            first,
            n,
        ]
        writer_rt[core.x][core.y] = [transition.buffer_address(), injection.buffer_address(), first, n]
        compute_rt[core.x][core.y] = [n]

    f32, bf16 = ttnn.float32, ttnn.bfloat16
    cbs = [
        _cb(0, f32, 2, cores),  # a
        _cb(1, f32, 2, cores),  # b
        _cb(2, f32, 4, cores),  # alpha, column broadcast
        _cb(3, f32, 3, cores),  # constants
        _cb(4, f32, 1, cores),  # reader scratch
        _cb(6, f32, 16, cores),  # fp32 staging
        _cb(8, bf16, 16, cores),  # M hi
        _cb(9, bf16, 16, cores),  # M lo
        _cb(12, bf16, 16, cores),  # E hi
        _cb(13, bf16, 16, cores),  # E lo
        _cb(14, bf16, 16, cores),  # B hi
        _cb(15, bf16, 16, cores),  # B lo
        _cb(16, f32, 2, cores),  # transition out
        _cb(17, f32, 2, cores),  # injection out
    ]
    for x32, x_hi, x_lo in ((7, 10, 11), (20, 21, 22)):  # X, two sets
        cbs += [_cb(x32, f32, 16, cores), _cb(x_hi, bf16, 16, cores), _cb(x_lo, bf16, 16, cores)]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/delta_solve_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[int(has_alpha)] + _accessor_args(a, b, alpha, constants),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/delta_solve_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[int(has_alpha)] + _accessor_args(transition, injection),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/delta_solve_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[iterations, int(has_alpha)],
            runtime_args=compute_rt,
            config=_compute_config([0, 1, 2, 3, 6, 7, 20]),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    io = [a, b, alpha, constants] + ([transition] if has_alpha else []) + [injection]
    ttnn.generic_op(io, program)
    return (transition if has_alpha else None), injection


# ---------------------------------------------------------------------------------------------- frame scan


def frame_scan(
    transition: ttnn.Tensor, injection: ttnn.Tensor, *, text_state: ttnn.Tensor | None = None
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """`ops.bidirectional_scan` fused: `state <- state @ T[f] + J[f]` forward and in reverse over the frames from
    `text_state` `[1, H, 128, 128]` (or zero). transition, injection: `[F, H, 128, 128]` fp32 tiled. Returns the
    prefix and suffix banks, same shape. Work units are (head, direction, 32-row block of the state), spread over
    the cores."""
    device = transition.device()
    shape = list(transition.shape)
    frames, heads = shape[0], shape[1]
    assert shape[2:] == [128, 128] and list(injection.shape) == shape, f"frame_scan: {shape}, {list(injection.shape)}"
    has_start = text_state is not None
    if has_start:
        assert list(text_state.shape) == [1, heads, 128, 128] and text_state.dtype == ttnn.float32
    blocks = 128 // TILE
    prefix, suffix = _empty(device, shape), _empty(device, shape)
    start = text_state if has_start else injection

    work = _split_work(device, heads * 2 * blocks)  # unit u: head u // 8, reverse (u // 4) % 2, block u % 4
    _, core_set = _cores(device, len(work))
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, first, n in work:
        reader_rt[core.x][core.y] = [
            transition.buffer_address(),
            injection.buffer_address(),
            start.buffer_address(),
            first,
            n,
            frames,
            heads,
        ]
        writer_rt[core.x][core.y] = [prefix.buffer_address(), suffix.buffer_address(), first, n, frames, heads]
        compute_rt[core.x][core.y] = [frames, n]

    f32, bf16 = ttnn.float32, ttnn.bfloat16
    cbs = [
        _cb(0, f32, 2, core_set),  # T
        _cb(1, f32, 2, core_set),  # J rows
        _cb(2, f32, 2, core_set),  # start rows
        _cb(6, f32, 16, core_set),  # T staging
        _cb(8, bf16, 16, core_set),  # T hi
        _cb(9, bf16, 16, core_set),  # T lo
        _cb(16, f32, 2, core_set),  # out
    ]
    for s32, s_hi, s_lo in ((7, 10, 11), (20, 21, 22)):  # state block, two sets
        cbs += [_cb(s32, f32, 4, core_set), _cb(s_hi, bf16, 4, core_set), _cb(s_lo, bf16, 4, core_set)]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/frame_scan_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[int(has_start)] + _accessor_args(transition, injection, start),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/frame_scan_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=_accessor_args(prefix),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/frame_scan_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[int(has_start)],
            runtime_args=compute_rt,
            config=_compute_config([0, 1, 2, 6, 7, 20]),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    io = [transition, injection] + ([text_state] if has_start else []) + [prefix, suffix]
    ttnn.generic_op(io, program)
    return prefix, suffix


# ---------------------------------------------------------------------------------------------- features


def conv_fits(device: ttnn.MeshDevice, channels: int) -> bool:
    """`conv_spatial` gives each 32-channel group its own cores, so the groups must fit the grid."""
    grid = device.compute_with_storage_grid_size()
    return channels % 128 == 0 and channels // TILE <= grid.x * grid.y


def conv_spatial(
    x: ttnn.Tensor,
    weights: ttnn.Tensor,
    *,
    frames: int,
    frame_size: tuple[int, int],
    kernel: int,
    row0: int = 0,
) -> ttnn.Tensor:
    """Depthwise `kernel` x `kernel` conv per frame (zero padding `kernel // 2`), straight into the head layout.

    x: `[1, 1, rows, C]` bf16 tiled, frame f's tokens at rows `row0 + f S ..` (S = gh gw, row-major over the grid).
    weights: `[kernel^2, 1, 32, C]` fp32 tiled, tap `i k + j` (cross-correlation offset (i - k/2, j - k/2)) with every
    tile row the channel weights. Returns `[frames, C / 128, S, 128]` fp32 tiled, pad rows zero. `kernel=1` with unit
    weights is an exact re-layout."""
    device = x.device()
    gh, gw = frame_size
    S, channels = gh * gw, x.shape[-1]
    rt, c_tiles, heads = math.ceil(S / TILE), channels // TILE, channels // 128
    taps = kernel * kernel
    assert channels % 128 == 0 and x.dtype == ttnn.bfloat16 and list(weights.shape) == [taps, 1, TILE, channels]
    out = _empty(device, [frames, heads, S, 128])

    grid = device.compute_with_storage_grid_size()
    per_group = max(1, min(grid.x * grid.y // c_tiles, frames * rt))
    cores, core_set = _cores(device, per_group * c_tiles)
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    base, extra = divmod(frames * rt, per_group)
    for g in range(c_tiles):
        first = 0
        for k in range(per_group):
            core = cores[g * per_group + k]
            n = base + (1 if k < extra else 0)
            reader_rt[core.x][core.y] = [x.buffer_address(), weights.buffer_address(), g, first, n, row0]
            writer_rt[core.x][core.y] = [out.buffer_address(), g, first, n]
            compute_rt[core.x][core.y] = [n]
            first += n

    reach = (kernel // 2) * (gw + 1)
    window = (TILE + 2 * reach + TILE - 1) // TILE + 1
    cbs = [
        _cb(0, ttnn.float32, taps, core_set),  # weights
        _cb(1, ttnn.bfloat16, window, core_set),  # input window
        _cb(2, ttnn.bfloat16, 2 * taps, core_set),  # shifted tiles, one tile's taps ahead
        _cb(16, ttnn.float32, 2, core_set),  # out
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/conv_spatial_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[kernel, gh, gw, c_tiles] + _accessor_args(x, weights),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/conv_spatial_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[heads, rt] + _accessor_args(out),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/conv_spatial_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[taps],
            runtime_args=compute_rt,
            config=_compute_config([0]),
        ),
    ]
    ttnn.generic_op([x, weights, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def _ones_tile(device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ops._constant(device, ("ones_tile",), lambda: torch.ones(1, 1, TILE, TILE), ttnn.bfloat16)


def conv_temporal_features(y: ttnn.Tensor, weights: ttnn.Tensor | None, *, l2: bool) -> ttnn.Tensor:
    """`prepare_linear_features` after the spatial conv, in the head layout: the 5-tap temporal conv over frames
    (zero padded; skipped when `weights` is None), SiLU, the L2 norm over each head's 128 channels when `l2`, bf16.

    y: `[F, H, S, 128]` fp32 tiled (`conv_spatial`). weights: `[5, 1, 32, H 128]` fp32 tiled, rows the channel
    weights. Returns `[F, H, S, 128]` bf16 tiled."""
    device = y.device()
    frames, heads, S, _ = y.shape
    rt, c_tiles = math.ceil(S / TILE), heads * 128 // TILE
    temporal = weights is not None
    if temporal:
        assert list(weights.shape) == [5, 1, TILE, heads * 128]
    ones = _ones_tile(device)
    w = weights if temporal else ones
    out = ttnn.empty([frames, heads, S, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    work = _split_work(device, frames * heads * rt)
    _, core_set = _cores(device, len(work))
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, first, n in work:
        reader_rt[core.x][core.y] = [
            y.buffer_address(),
            w.buffer_address(),
            ones.buffer_address(),
            first,
            n,
            frames,
            heads,
            rt,
        ]
        writer_rt[core.x][core.y] = [out.buffer_address(), first, n]
        compute_rt[core.x][core.y] = [n, frames, first, heads, rt]

    f32, bf16 = ttnn.float32, ttnn.bfloat16
    cbs = [
        _cb(0, f32, 4, core_set),  # y
        _cb(1, f32, 5 * c_tiles if temporal else 1, core_set),  # weights
        _cb(2, bf16, 1, core_set),  # ones
        _cb(3, f32, 1, core_set),  # 1 / norm
        _cb(6, f32, 4, core_set),  # silu output
        _cb(7, f32, 1, core_set),  # row partial sums
        _cb(8, bf16, 1, core_set),  # sums hi
        _cb(9, bf16, 1, core_set),  # sums lo
        _cb(16, bf16, 2, core_set),  # out
    ]
    flags = [int(temporal), int(l2), c_tiles]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/conv_temporal_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=flags + _accessor_args(y, w, ones),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/conv_temporal_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=_accessor_args(out),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/conv_temporal_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=flags,
            runtime_args=compute_rt,
            config=_compute_config([0, 1, 3, 6, 7]),
        ),
    ]
    ttnn.generic_op([y, w, ones, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
