# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Row-major selection kernels of the DeepSeek-V4.1 indexer (bead F10; ``tt/v41/kernels``, launched through
``ttnn.generic_op``). Both work row by row on a DRAM-interleaved row-major bf16 tensor (one page per row), such as
the indexer's score, so every chip's rows stay local (no collectives; replicated or sharded mesh tensors work).

* :func:`segment_max`: maxima of consecutive runs of 8 or 32 elements of every row in one read of the row, with each
  row's newest element's run pinned to +inf (the candidate source's pin).
* :func:`gather_runs`: per row, the runs of 8 or 32 elements named by a row of run ids, in id order (-inf for an id
  past the row, e.g. the 0xFFFFFFFF sentinel).
"""

import ttnn

_KERNEL_DIR = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE_BYTES = 32 * 32 * 2  # bf16 tile
SEGMAX_BLOCK = 4  # tiles per segment-max chunk: 4096 consecutive row elements
GATHER_SEGMENT_BYTES = 64 << 10  # gather_runs streams each source row through L1 in segments of this size
GATHER_MAX_SEGMENTS = 64  # gather_runs.cpp MAX_SEGS


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _cores(device, n: int):
    """The first ``n`` cores of the compute grid, row-major -> ([(x, y)], CoreRangeSet)."""
    grid = device.compute_with_storage_grid_size()
    n = min(n, grid.x * grid.y)
    last = ttnn.CoreCoord((n - 1) % grid.x, (n - 1) // grid.x)
    ranges = []
    if last.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last.y - 1)))
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, last.y), last))
    return [(i % grid.x, i // grid.x) for i in range(n)], ttnn.CoreRangeSet(ranges)


def _split(total: int, parts: int) -> list[tuple[int, int]]:
    """``total`` items in ``parts`` contiguous (start, count) ranges, the first ``total % parts`` one longer."""
    base, extra = divmod(total, parts)
    out, start = [], 0
    for i in range(parts):
        count = base + (i < extra)
        out.append((start, count))
        start += count
    return out


def _cb(index: int, total: int, cores, dtype=ttnn.bfloat16, page: int = _TILE_BYTES) -> ttnn.CBDescriptor:
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=total, core_ranges=cores, format_descriptors=[fmt])


def _check_row_major(t, dtype, what: str):
    memory = t.memory_config()
    assert t.layout == ttnn.ROW_MAJOR_LAYOUT and t.dtype == dtype, f"{what} must be row-major {dtype}"
    assert memory.buffer_type == ttnn.BufferType.DRAM and not memory.is_sharded(), f"{what} must be DRAM-interleaved"


def segment_max(x, seg: int, pin, pin_base: int, pin_mask: int):
    """Row-major bf16 ``x`` [1, 1, R, W] (DRAM-interleaved, W a multiple of 32) -> row-major bf16
    [1, 1, R, round_up(W / seg, 32)]: output j of row i is ``max(x[i, seg * j : seg * (j + 1)])`` (-inf past W / seg),
    except the row's pinned output: +inf at ``e // seg`` with ``e = (pin_base + q_i) & pin_mask`` when ``e < W``,
    ``q_i`` the chunk query index of the chip's row i (``pin``: ``V41ChunkTables.query_first()``). ``seg`` 8 or 32.
    One fused op (kernels/segmax_*): per 4096-element chunk a tilize and a MAX row reduce (for 8, after adding four
    column masks)."""
    assert seg in (8, 32)
    _check_row_major(x, ttnn.bfloat16, "segment_max input")
    rows, width = x.shape[2], x.shape[3]
    assert width % 32 == 0, f"row width {width} is not a multiple of 32"
    out_width = _round_up(width // seg, 32)
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, out_width]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    block = SEGMAX_BLOCK
    per_tile = 4 if seg == 8 else 1
    results, out_per_chunk = block * per_tile, block * 32 * per_tile
    chunks = -(-width // (block * 1024))
    coords, cores = _cores(device, rows)
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), (first, count) in zip(coords, _split(rows, len(coords))):
        reader_args[cx][cy] = [x.buffer_address(), first, count, width * 2]
        writer_args[cx][cy] = [out.buffer_address(), first, count, chunks, pin.buffer_address(), pin_base, pin_mask]
        compute_args[cx][cy] = [count * chunks]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/segmax_reader.cpp",
            core_ranges=cores,
            compile_time_args=[0, 1, 2, seg, block] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/segmax_writer.cpp",
            core_ranges=cores,
            compile_time_args=[16, 5, seg, block, out_width, width // seg]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pin).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/segmax_compute.cpp",
            core_ranges=cores,
            compile_time_args=[seg, block],
            runtime_args=compute_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
            ),
        ),
    ]
    scratch = 64 + 2 * out_per_chunk * 2
    cbs = [
        _cb(0, 2 * block * _TILE_BYTES, cores),
        _cb(1, _TILE_BYTES, cores),
        _cb(3, 2 * block * _TILE_BYTES, cores),
        _cb(16, 2 * results * _TILE_BYTES, cores),
        _cb(5, scratch, cores, page=scratch),
        _cb(2, 4 * _TILE_BYTES, cores),  # column masks (8 only; the compute kernel binds every CB)
        _cb(4, 4 * block * _TILE_BYTES, cores),
    ]
    return ttnn.generic_op([x, pin, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


def gather_runs(x, ids, run: int):
    """Row-major bf16 ``x`` [1, 1, R, W] and uint32 ``ids`` [1, 1, R, K] (both DRAM-interleaved) -> row-major bf16
    [1, 1, R, K * run]: run r of row i is ``x[i, run * c : run * (c + 1)]`` for ``c = ids[i, r]``, -inf where
    ``c >= W / run`` (the sentinel included). ``run`` 8 or 32 (16- or 64-byte runs). One data-movement op
    (kernels/gather_runs.cpp) on both RISC-Vs of every core: each source row is read once, sequentially."""
    assert run in (8, 32)
    _check_row_major(x, ttnn.bfloat16, "gather_runs input")
    _check_row_major(ids, ttnn.uint32, "gather_runs ids")
    rows, width, k = x.shape[2], x.shape[3], ids.shape[3]
    assert ids.shape[2] == rows and width % 32 == 0
    run_bytes = run * 2
    assert -(-width * 2 // GATHER_SEGMENT_BYTES) <= GATHER_MAX_SEGMENTS, f"row width {width} is too wide"
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, k * run]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    coords, cores = _cores(device, -(-rows // 2))
    units = _split(rows, 2 * len(coords))  # unit 2c on core c's reader RISC-V, 2c + 1 on its writer
    accessors = (
        ttnn.TensorAccessorArgs(x).get_compile_time_args()
        + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
    )
    scratch = 64 + GATHER_SEGMENT_BYTES + k * run_bytes + 2 * k * 4
    kernels = []
    for risc, config in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
        args = ttnn.RuntimeArgs()
        for c, (cx, cy) in enumerate(coords):
            first, count = units[2 * c + risc]
            args[cx][cy] = [
                x.buffer_address(),
                ids.buffer_address(),
                out.buffer_address(),
                first,
                count,
                width // run,
                width * 2,
            ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNEL_DIR}/gather_runs.cpp",
                core_ranges=cores,
                compile_time_args=[risc, run_bytes, k, GATHER_SEGMENT_BYTES] + accessors,
                runtime_args=args,
                config=config,
            )
        )
    cbs = [_cb(risc, scratch, cores, page=scratch) for risc in (0, 1)]
    return ttnn.generic_op([x, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
