# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""C++ Metalium matmul for gemma3's QKV projections -- the cpp rung on both
MinimalMatmulDeviceOperation 1024 x 3840 x 8192 (long prefill) and
MatmulDeviceOperation 32 x 3840 x 8192 (decode).

Both are measured no-gains, and this kernel is CORRECT on both: at the decode shape it scores PCC
0.999496 against ttnn and 0.992677 against torch (ttnn's own is 0.993091) and still runs 340.4 us
against the real DRAM-sharded op's 63.5 us -- 52 GB/s of weight bandwidth against 279. See the
_GEMMA3_QKV_DECODE_CPP_MM block in model_config.py for the full table and why M=1 tile does not
rescue the partitioning.

The tt-lang (ttl) rung on this op is toolchain-blocked; this rung is not, because
a hand kernel calls ``matmul_block_init``/``matmul_tiles``, which this tree declares.

DESIGN -- deliberately the SIMPLE partitioning, not the 2D-mcast one. The output tile space
(Mt x Nt) is split across the full grid with ``split_work_to_cores``; each core walks its own output
tiles and, for each, streams the Kt A-tiles of its row and the Kt B-tiles of its column, reducing in
DST. The three kernels are the stock ``matmul_multi_core`` example bodies used UNMODIFIED, so the
arithmetic is correct by construction.

That choice is on purpose. The sophisticated port -- ``matmul_multicore_reuse_mcast``, which reuses
each in0/in1 tile across a core's block -- was already built for this model's sibling FF1/FF3 shape
and produced garbage PCC in four separate configurations (0.054 / -0.10 / -0.008 / -0.054) at 2.8x
the stock op's per-call time, with the unresolved suspect being how a DRAM-width-sharded weight is
addressed through TensorAccessor page ids. wqkv is DRAM-sharded AND bfloat4_b, i.e. strictly the
harder case. So the question this rung answers is "can a hand kernel beat ttnn here", and the
cheapest honest way to answer it is a kernel whose correctness is not in doubt.

The expected answer is no, and the arithmetic says so before the device does: this partitioning
re-reads every A row once per output COLUMN and every B column once per output ROW, so at
Mt=32 / Nt=256 / Kt=120 it moves ~2.5 GB per call against the ~50 MB ttnn's multicast version
needs. It is recorded as a measured no-gain rather than skipped, because that is what clears the
rung -- an argument that a kernel cannot win is not the same as a measurement.
"""
from __future__ import annotations

import ttnn

TILE = 32

_EX = "tt_metal/programming_examples/matmul/matmul_multi_core/kernels"
_READER = f"{_EX}/dataflow/reader_mm_output_tiles_partitioned.cpp"
_WRITER = f"{_EX}/dataflow/writer_unary_interleaved_start_id.cpp"
_COMPUTE = f"{_EX}/compute/mm.cpp"

# Bytes per 32x32 tile, by dtype. bfp formats carry a per-tile exponent section.
_TILE_BYTES = {
    ttnn.bfloat16: 32 * 32 * 2,
    ttnn.float32: 32 * 32 * 4,
    ttnn.bfloat8_b: 32 * 32 + 64,
    ttnn.bfloat4_b: 32 * 32 // 2 + 64,
}


def can_run(a, b):
    """Only the 2D, tile-aligned, non-batched case the example kernels are written for."""
    if a.dtype not in _TILE_BYTES or b.dtype not in _TILE_BYTES:
        return False
    if len(b.shape) < 2 or len(a.shape) < 2:
        return False
    if any(int(d) != 1 for d in list(a.shape)[:-2]) or any(int(d) != 1 for d in list(b.shape)[:-2]):
        return False
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    return k == int(b.shape[-2]) and m % TILE == 0 and k % TILE == 0 and n % TILE == 0


def matmul(a, b, out_dtype=None, memory_config=None):
    """Drop-in for a 2D ttnn matmul over interleaved DRAM tensors."""
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    out_dtype = out_dtype or a.dtype
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, m, n]), out_dtype, a.layout, a.device(), memory_config)

    grid = a.device().compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    (_, core_grid, group_1, group_2, work_1, work_2) = ttnn.split_work_to_cores(all_cores, mt * nt)

    a_bytes, b_bytes, o_bytes = _TILE_BYTES[a.dtype], _TILE_BYTES[b.dtype], _TILE_BYTES[out_dtype]
    # Double-buffer each stream: the reader fills one tile while compute consumes the other.
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * a_bytes,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=a.dtype, page_size=a_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * b_bytes,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=b.dtype, page_size=b_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * o_bytes,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out_dtype, page_size=o_bytes)],
        ),
    ]

    # The reader builds two TensorAccessors back to back, so a's args come first and b's follow at
    # a_args.next_compile_time_args_offset(); the writer builds one for the output.
    reader_ct = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    writer_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tile = 0
    for group, per_core in ((group_1, work_1), (group_2, work_2)):
        if per_core == 0:
            continue
        for core_range in group.ranges():
            for cx in range(core_range.start.x, core_range.end.x + 1):
                for cy in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt[cx][cy] = [a.buffer_address(), b.buffer_address(), mt, kt, nt, tile, per_core]
                    writer_rt[cx][cy] = [out.buffer_address(), per_core, tile]
                    compute_rt[cx][cy] = [per_core, kt]
                    tile += per_core

    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_READER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=[],
                runtime_args=compute_rt,
                # LoFi to match what the op already runs at; fp32_dest_acc_en=False keeps the
                # subblock cap at 8 and matches compute_kernel_config_lofi.
                config=ttnn.ComputeConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    return ttnn.generic_op([a, b, out], program)
