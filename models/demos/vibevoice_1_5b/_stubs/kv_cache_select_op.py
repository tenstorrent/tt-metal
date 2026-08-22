# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fused C++ Metalium kernel (via ttnn.generic_op) for the traceable decode-time
KV-cache write in `qwen2_decoder_layer.py`:

    cache_new = cache + onehot * (new_val - cache)          (== cache*(1-onehot) + new_val*onehot)

Replaces the previous 3 chained `BinaryNg` ops (mul, mul, add — each round-tripping its
result through DRAM) with ONE kernel launch that reads `cache` / `onehot` / `new_val`
tiles from DRAM, computes entirely in L1/dst-registers (broadcast onehot's single valid
column across columns, new_val's single valid row across rows, then sub/mul/add), and
writes the result straight back into `cache`'s own DRAM buffer (true in-place update, no
intermediate tensor).

Shapes (confirmed live from the model, decode-time / write_onehot branch):
    cache    ([1, KV, C, HD])   float32, TILE_LAYOUT, DRAM interleaved
    onehot   ([1, 1,  C, 1 ])   float32, TILE_LAYOUT, DRAM interleaved (one-hot along C)
    new_val  ([1, KV, 1, HD])   float32, TILE_LAYOUT, DRAM interleaved

C and HD are always tile-aligned (32-multiples) in this model, so no partial-tile padding
handling is required beyond what TILE_LAYOUT already guarantees (onehot's padded columns
1..31 and new_val's padded rows 1..31 are zero, which is exactly what the COL/ROW hardware
broadcast primitives require of their source tile)."""

import ttnn

_KERNEL_DIR = "models/demos/vibevoice_1_5b/_stubs/kv_cache_select_kernels"

_CB_CACHE = 0
_CB_ONEHOT = 1
_CB_NEW = 2
_CB_ONEHOT_B = 3
_CB_NEW_B = 4
_CB_DIFF = 5
_CB_SCALED = 6
_CB_OUT = 7


def _single_core_grid():
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def kv_cache_select(cache: "ttnn.Tensor", onehot: "ttnn.Tensor", new_val: "ttnn.Tensor") -> "ttnn.Tensor":
    """In-place: cache <- cache + onehot * (new_val - cache). Returns `cache`.

    `cache`: [1, KV, C, HD] float32 TILE DRAM-interleaved (written in place).
    `onehot`: [1, 1, C, 1] float32 TILE DRAM-interleaved (one-hot along C).
    `new_val`: [1, KV, 1, HD] float32 TILE DRAM-interleaved.
    """
    assert cache.dtype == ttnn.float32 and onehot.dtype == ttnn.float32 and new_val.dtype == ttnn.float32
    assert cache.layout == ttnn.TILE_LAYOUT and onehot.layout == ttnn.TILE_LAYOUT and new_val.layout == ttnn.TILE_LAYOUT

    _, KV, C, HD = cache.shape
    assert tuple(onehot.shape) == (1, 1, C, 1)
    assert tuple(new_val.shape) == (1, KV, 1, HD)
    assert C % 32 == 0 and HD % 32 == 0

    RT = C // 32
    CT = HD // 32
    n_tiles = KV * RT * CT

    core_grid = _single_core_grid()
    tile = ttnn.Tile([32, 32])
    tile_desc = ttnn.TileDescriptor(tile)
    page_size = 32 * 32 * 4  # float32

    def _cb(index):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=index,
            data_format=ttnn.float32,
            page_size=page_size,
            tile=tile_desc,
        )
        return ttnn.CBDescriptor(total_size=2 * page_size, core_ranges=core_grid, format_descriptors=[fmt])

    cbs = [_cb(i) for i in (_CB_CACHE, _CB_ONEHOT, _CB_NEW, _CB_ONEHOT_B, _CB_NEW_B, _CB_DIFF, _CB_SCALED, _CB_OUT)]

    dims_ct_args = [KV, RT, CT]

    reader_ct_args = list(dims_ct_args)
    reader_ct_args += ttnn.TensorAccessorArgs(cache).get_compile_time_args()
    reader_ct_args += ttnn.TensorAccessorArgs(onehot).get_compile_time_args()
    reader_ct_args += ttnn.TensorAccessorArgs(new_val).get_compile_time_args()

    writer_ct_args = list(dims_ct_args)
    writer_ct_args += ttnn.TensorAccessorArgs(cache).get_compile_time_args()

    core = ttnn.CoreCoord(0, 0)
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [cache.buffer_address(), onehot.buffer_address(), new_val.buffer_address()]

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [cache.buffer_address()]

    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = [n_tiles]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/reader_kv_select.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/writer_kv_select.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/compute_kv_select.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[],
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        ),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )

    # cache is passed as BOTH an input (read old values) and the pre-allocated output
    # (written in place) — the writer targets the exact same DRAM buffer it was read
    # from, tile-for-tile, so this is a true in-place update.
    io_tensors = [cache, onehot, new_val, cache]
    ttnn.generic_op(io_tensors, program_descriptor)
    return cache
