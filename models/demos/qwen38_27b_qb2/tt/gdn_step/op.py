# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in DRAM-backed, in-place FP32 single-token GDN recurrence.

By default Q/K are L2 normalized and Q scaled by 128**-0.5; normalize_qk=True
accepts raw Q/K and performs that preparation in FP32 inside the kernel. Gates hold
exp(log_decay), beta in columns 0/1 (six alignment columns are ignored).
Convolution and output gating remain external. Full-model qualification is
still required; report whether optional normalization is included in timing.
"""

from pathlib import Path

HERE = Path(__file__).parent


def kernel_source(filename):
    # Inline shared math so generic_op's source hash covers its exact contents.
    return (HERE / filename).read_text().replace('#include "compute_math.hpp"', (HERE / "compute_math.hpp").read_text())


def shared_head_count(value_heads, qk_head_repeat):
    if type(qk_head_repeat) is not int or qk_head_repeat < 1 or value_heads % qk_head_repeat:
        raise ValueError("Q/K head repetition must divide the value-head count")
    return value_heads // qk_head_repeat


def work_items(heads, grid_x, grid_y, value_splits=1):
    """Disjoint (head, value-column partition) assignments, including waves."""
    if min(heads, grid_x, grid_y) <= 0:
        raise ValueError("Head count and grid dimensions must be positive")
    if type(value_splits) is not int or value_splits not in (1, 2, 4):
        raise ValueError("Value splits must be 1, 2, or 4")
    items = heads * value_splits
    cores = min(items, grid_x * grid_y)
    return [(i % grid_x, i // grid_x, i, cores, (items - 1 - i) // cores + 1) for i in range(cores)]


def circular_buffer_pages(value_splits, input_buffer_items, *, normalize_qk=False):
    """Bound input lookahead while retaining one writer-owned output window."""
    if type(value_splits) is not int or value_splits not in (1, 2, 4):
        raise ValueError("Value splits must be 1, 2, or 4")
    if type(input_buffer_items) is not int or input_buffer_items not in (1, 2):
        raise ValueError("Input buffer items must be 1 or 2")
    if type(normalize_qk) is not bool:
        raise ValueError("normalize_qk must be a Boolean")
    columns = 4 // value_splits
    inputs = [4, 4, columns, 1, 1, 4 * columns]
    return (
        [pages * input_buffer_items for pages in inputs]
        + [columns, 4 * columns, columns, 1, 1]
        + ([4, 4, 1] if normalize_qk else [])
    )


def step(q, k, v, gates, state, output, *, value_splits=1, input_buffer_items=1, normalize_qk=False, qk_head_repeat=1):
    """Mutate state[heads,128,128]; write output[heads,128], all FP32 DRAM.

    The caller preallocates output and retains all tensors through trace
    replay. Every invocation must reconstruct its descriptor so replacement
    buffers get new runtime addresses. generic_op's descriptor adapter copies
    the complete runtime arguments on cache hits; its regression is exercised
    with two independently allocated tensor sets in the hardware test.
    """
    pages_per_cb = circular_buffer_pages(value_splits, input_buffer_items, normalize_qk=normalize_qk)
    import ttnn

    mesh = state.device()
    heads = state.shape[0]
    qk_heads = shared_head_count(heads, qk_head_repeat)
    shapes = [(qk_heads, 128)] * 2 + [(heads, 128), (heads, 8), (heads, 128, 128), (heads, 128)]
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
    assignments = work_items(heads, grid.x, grid.y, value_splits)
    value_columns = 4 // value_splits
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
    if normalize_qk:
        for cb in (11, 12, 13):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = []
    for filename, args, ctargs, cfg in [
        (
            "reader.cpp",
            read_args,
            [value_columns, qk_head_repeat, *accessors(tensors[:5])],
            ttnn.ReaderConfigDescriptor(),
        ),
        ("writer.cpp", write_args, [value_columns, *accessors([state, output])], ttnn.WriterConfigDescriptor()),
        ("compute.cpp", compute_args, [value_columns, int(normalize_qk)], config),
    ]:
        kernels.append(
            ttnn.KernelDescriptor(
                # Source text enters generic_op's cache hash, so editing a kernel
                # cannot silently reuse its previous descriptor in one process.
                kernel_source=kernel_source(filename),
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
        for cb, pages in enumerate(pages_per_cb)
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    return ttnn.generic_op(tensors, descriptor)
