# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental direct tiled model-input preparation for single-token GDN.

The caller supplies exp(log_decay), retaining the current TTNN exp operation.
This packs values/gates and normalizes Q/K with the existing FP32 arithmetic.
Compact convolution input is experimental; all outputs are caller-owned.
"""

from models.demos.qwen38_27b_qb2.tt.gdn_step.op import HERE, kernel_source, work_items


def prepare(q, k, v, decay, beta, normalized_q, normalized_k, values, gates, *, compact_qkv=False):
    import ttnn

    tensors = [q, k, v, decay, beta, normalized_q, normalized_k, values, gates]
    mesh = q.device()
    if "BLACKHOLE" not in str(mesh.arch()).upper():
        raise ValueError("Direct GDN preparation currently targets Blackhole only")
    if type(compact_qkv) is not bool:
        raise ValueError("compact_qkv must be a Boolean")
    if len(q.shape) != 3 or q.shape[2] != 512:
        raise ValueError("Q/K require three-dimensional 512-channel inputs")
    batch = q.shape[1] if compact_qkv else q.shape[0]
    if batch < 1 or (compact_qkv and batch > 32):
        raise ValueError("Unsupported direct preparation batch")
    for index, (tensor, width, dtype) in enumerate(
        zip(
            tensors[:5],
            (512, 512, 1536, 12, 12),
            (ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, ttnn.bfloat16),
        )
    ):
        compact = compact_qkv and index < 3
        shapes = ((1, batch, width),) if compact else ((batch, 1, width), (batch, 32, width))
        padded = (1 if compact else batch, 32, (width + 31) // 32 * 32)
        if (
            tuple(tensor.shape) not in shapes
            or tuple(tensor.padded_shape) != padded
            or tensor.dtype != dtype
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
            or tensor.device() != mesh
        ):
            raise ValueError("Require tiled interleaved model inputs with exactly one padded token tile")
    for tensor, shape in zip(tensors[5:], ((batch * 4, 128), (batch * 4, 128), (batch * 12, 128), (batch * 12, 8))):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != ttnn.float32
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tensor.device() != mesh
        ):
            raise ValueError("Prepared outputs require declared FP32 row-major DRAM geometry")
    if len({(str(t.memory_config().buffer_type), t.buffer_address()) for t in tensors}) != len(tensors):
        raise ValueError("Direct GDN preparation operands must not alias")
    grid = mesh.compute_with_storage_grid_size()
    assignments = work_items(batch * 4, grid.x, grid.y)
    cores = ttnn.num_cores_to_corerangeset(len(assignments), grid, row_wise=True)
    read, write, compute = (ttnn.RuntimeArgs() for _ in range(3))
    for x, y, first, stride, count in assignments:
        read[x][y] = [*(t.buffer_address() for t in (*tensors[:5], values, gates)), first, stride, count]
        write[x][y] = [normalized_q.buffer_address(), normalized_k.buffer_address(), first, stride, count]
        compute[x][y] = [count]

    def accessors(items):
        return [arg for tensor in items for arg in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]

    config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (0, 1, 11, 12, 13):
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ctargs,
            runtime_args=args,
            config=cfg,
        )
        for source, args, ctargs, cfg in (
            (
                (HERE / "flat_prepare_reader.cpp").read_text(),
                read,
                [int(compact_qkv), *accessors([*tensors[:5], values, gates])],
                ttnn.ReaderConfigDescriptor(),
            ),
            (kernel_source("prepare_compute.cpp"), compute, [], config),
            (kernel_source("prepare_writer.cpp"), write, accessors(tensors[5:7]), ttnn.WriterConfigDescriptor()),
        )
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.float32, page_size=4096)],
        )
        for cb, pages in {0: 8, 1: 8, 9: 1, 10: 1, 11: 4, 12: 4, 13: 1}.items()
    ]
    return ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
