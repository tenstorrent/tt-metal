# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""C++/Metalium kernel for the MLP projections -- MEASURED, NOT WIRED IN.

``mm_generic_op`` is an output-tile-partitioned matmul driven through
``ttnn.generic_op``, assembled from the repo's own Metalium kernels
(``tt_metal/programming_examples/matmul/matmul_multi_core``) rather than hand-written
kernel bodies.

RESULT: it works and it is correct, and it is slower than the stock op.

  * Isolated, [128, 4096] x [4096, 3584]: PCC 0.99892 against ttnn.linear with bf16
    weights, 0.99938 with bf4_b weights. So the mixed-dtype matmul that tt-lang refuses
    to lower (see ttl_kernels.py) is fine at the Metalium layer -- which is what the
    stock op does underneath.
  * Wired in for the prefill ff1 projection: full-model PCC held at 0.9339, but
    device_ms went 2.5495 -> 3.3624 (+32%) and prefill 19.09 -> 31.71 ms.

The reason is structural, not a tuning miss. Partitioning by OUTPUT TILE means each
core re-reads a full K-strip of BOTH operands for every output tile it owns: for this
shape that is 448 output tiles x 2 x 128 tiles = ~115k tile reads, against the ~15k a
blocked/multicast matmul needs (4x128 for the activation plus 128x112 for the weights,
each read once). ~7.7x the DRAM traffic on an op the roofline already calls
memory-bound. Beating ttnn here would mean reimplementing the reuse and multicast that
``matmul_multicore_reuse_mcast`` already has -- i.e. rewriting the production kernel,
not adapting an example.

Kept as the evidence artifact for the C++ rung, and as a working generic_op assembly
(CB descriptors, TensorAccessor compile-time args, per-core runtime args for a reader /
compute / writer triple) for anyone who does want to start from a correct baseline.
"""

import ttnn

TILE = 32
_EXAMPLES = "matmul/matmul_multi_core/kernels"
_READER = f"tt_metal/programming_examples/{_EXAMPLES}/dataflow/reader_mm_output_tiles_partitioned.cpp"
_WRITER = f"tt_metal/programming_examples/{_EXAMPLES}/dataflow/writer_unary_interleaved_start_id.cpp"
_COMPUTE = f"tt_metal/programming_examples/{_EXAMPLES}/compute/mm.cpp"

# Tile page size in bytes per data format. Block floats carry one shared exponent byte
# per 16 data, on top of the mantissa bytes.
_TILE_BYTES = {
    ttnn.bfloat16: TILE * TILE * 2,
    ttnn.bfloat8_b: TILE * TILE + (TILE * TILE) // 16,
    ttnn.bfloat4_b: (TILE * TILE) // 2 + (TILE * TILE) // 16,
}


def _cb(index, dtype, core_ranges, num_tiles=2):
    page = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=num_tiles * page,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def mm_generic_op(mesh_device, a, b, out, math_fidelity=None):
    """``out = a @ b`` over output tiles partitioned across the compute grid.

    ``a`` is ``[1, 1, M, K]``, ``b`` is ``[1, 1, K, N]`` and ``out`` is a pre-allocated
    ``[1, 1, M, N]``. Layouts and dtypes are taken from the tensors themselves -- the
    TensorAccessor compile-time args carry the buffer type and any shard spec -- so the
    model's DRAM-width-sharded block-float weights need no copy.
    """
    m_tiles = a.shape[-2] // TILE
    k_tiles = a.shape[-1] // TILE
    n_tiles = out.shape[-1] // TILE

    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, core_grid, group1, group2, work1, work2 = ttnn.split_work_to_cores(all_cores, m_tiles * n_tiles)

    reader_ct = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    writer_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    offset = 0
    for ranges, work in ((group1, work1), (group2, work2)):
        for core_range in ranges.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt[x][y] = [
                        a.buffer_address(),
                        b.buffer_address(),
                        m_tiles,
                        k_tiles,
                        n_tiles,
                        offset,
                        work,
                    ]
                    writer_rt[x][y] = [out.buffer_address(), work, offset]
                    compute_rt[x][y] = [work, k_tiles]
                    offset += work

    compute_cfg = ttnn.ComputeConfigDescriptor()
    if math_fidelity is not None:
        compute_cfg.math_fidelity = math_fidelity

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
                config=compute_cfg,
            ),
        ],
        semaphores=[],
        cbs=[
            _cb(0, a.dtype, core_grid),
            _cb(1, b.dtype, core_grid),
            _cb(16, out.dtype, core_grid),
        ],
    )
    return ttnn.generic_op([a, b, out], program)
