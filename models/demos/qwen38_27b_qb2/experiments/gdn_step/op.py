# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental DRAM-backed, in-place FP32 single-token GDN recurrence.

Q/K must already be L2 normalized and Q scaled by 128**-0.5. Gates hold
exp(log_decay), beta in columns 0/1 (six alignment columns are ignored).
This is the recurrence only: normalization, convolution, and output gating
are not included in its latency. No qualified model imports this module.
"""

from pathlib import Path

HERE = Path(__file__).parent


def work_items(heads, grid_x, grid_y):
    """Disjoint head assignments, including uneven final waves."""
    if min(heads, grid_x, grid_y) <= 0:
        raise ValueError("Head count and grid dimensions must be positive")
    cores = min(heads, grid_x * grid_y)
    return [(i % grid_x, i // grid_x, i, cores, (heads - 1 - i) // cores + 1) for i in range(cores)]


def step(q, k, v, gates, state, output):
    """Mutate state[heads,128,128]; write output[heads,128], all FP32 DRAM.

    The caller preallocates output and retains all tensors through trace
    replay. Every invocation must reconstruct its descriptor so replacement
    buffers get new runtime addresses. generic_op's descriptor adapter copies
    the complete runtime arguments on cache hits; its regression is exercised
    with two independently allocated tensor sets in the hardware test.
    """
    import ttnn

    mesh = state.device()
    heads = state.shape[0]
    shapes = [(heads, 128)] * 3 + [(heads, 8), (heads, 128, 128), (heads, 128)]
    tensors = [q, k, v, gates, state, output]
    for index, (tensor, shape) in enumerate(zip(tensors, shapes)):
        expected_layout = ttnn.TILE_LAYOUT if index == 4 else ttnn.ROW_MAJOR_LAYOUT
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != ttnn.float32
            or tensor.layout != expected_layout
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tensor.device() != mesh
        ):
            raise ValueError(f"Tensor {index} must be FP32 interleaved DRAM {shape} with layout {expected_layout}")
    if len({tensor.buffer_address() for tensor in tensors}) != len(tensors):
        raise ValueError("Inputs/output must not alias; state is the only in-place tensor")
    if "BLACKHOLE" not in str(mesh.arch()).upper():
        raise ValueError("The experimental FP32 kernel currently targets Blackhole only")
    grid = mesh.compute_with_storage_grid_size()
    assignments = work_items(heads, grid.x, grid.y)
    cores = ttnn.num_cores_to_corerangeset(len(assignments), grid, row_wise=True)
    read_args, write_args, compute_args = (ttnn.RuntimeArgs() for _ in range(3))
    for x, y, first, stride, count in assignments:
        read_args[x][y] = [*(t.buffer_address() for t in tensors[:5]), first, stride, count]
        write_args[x][y] = [state.buffer_address(), output.buffer_address(), first, stride, count]
        compute_args[x][y] = [count]

    def accessors(values):
        return [arg for tensor in values for arg in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]

    config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in range(9):
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = []
    for filename, args, ctargs, cfg in [
        ("reader.cpp", read_args, accessors(tensors[:5]), ttnn.ReaderConfigDescriptor()),
        ("writer.cpp", write_args, accessors([state, output]), ttnn.WriterConfigDescriptor()),
        ("compute.cpp", compute_args, [], config),
    ]:
        kernels.append(
            ttnn.KernelDescriptor(
                # Source text enters generic_op's cache hash, so editing a kernel
                # cannot silently reuse its previous descriptor in one process.
                kernel_source=(HERE / filename).read_text(),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=ctargs,
                runtime_args=args,
                config=cfg,
            )
        )
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.float32, page_size=4096)],
        )
        for cb, pages in enumerate([4, 4, 4, 1, 1, 16, 4, 16, 4, 1, 1])
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    return ttnn.generic_op(tensors, descriptor)
