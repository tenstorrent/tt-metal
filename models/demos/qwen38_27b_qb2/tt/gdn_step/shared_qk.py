# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental FP32 Q/K preparation, with persistent caller-owned outputs."""

from models.demos.qwen38_27b_qb2.tt.gdn_step.op import kernel_source, work_items


def prepare(q, k, normalized_q, normalized_k):
    """Normalize compact [shared_heads,128] once; keep the recurrence math unchanged.

    Both output tensors must be allocated before capture and retained by the
    caller for all replays. This function never allocates device tensors.
    """
    import ttnn

    tensors = [q, k, normalized_q, normalized_k]
    mesh = q.device()
    shape = tuple(q.shape)
    if len(shape) != 2 or shape[0] < 1 or shape[1] != 128:
        raise ValueError("Shared Q/K must have shape [heads,128]")
    for tensor in tensors:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != ttnn.float32
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tensor.device() != mesh
        ):
            raise ValueError("Shared Q/K inputs and outputs require matching FP32 row-major DRAM tensors")
    if len({t.buffer_address() for t in tensors}) != 4:
        raise ValueError("Shared Q/K inputs and outputs must not alias")
    if "BLACKHOLE" not in str(mesh.arch()).upper():
        raise ValueError("The shared FP32 Q/K experiment currently targets Blackhole only")
    grid = mesh.compute_with_storage_grid_size()
    assignments = work_items(shape[0], grid.x, grid.y)
    cores = ttnn.num_cores_to_corerangeset(len(assignments), grid, row_wise=True)
    read_args, write_args, compute_args = (ttnn.RuntimeArgs() for _ in range(3))
    for x, y, first, stride, count in assignments:
        read_args[x][y] = [q.buffer_address(), k.buffer_address(), first, stride, count]
        write_args[x][y] = [normalized_q.buffer_address(), normalized_k.buffer_address(), first, stride, count]
        compute_args[x][y] = [count]

    def accessors(values):
        return [arg for tensor in values for arg in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]

    config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (0, 1, 11, 12, 13):
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kernel_source(filename),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ctargs,
            runtime_args=args,
            config=cfg,
        )
        for filename, args, ctargs, cfg in (
            ("prepare_reader.cpp", read_args, accessors(tensors[:2]), ttnn.ReaderConfigDescriptor()),
            ("prepare_writer.cpp", write_args, accessors(tensors[2:]), ttnn.WriterConfigDescriptor()),
            ("prepare_compute.cpp", compute_args, [], config),
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
