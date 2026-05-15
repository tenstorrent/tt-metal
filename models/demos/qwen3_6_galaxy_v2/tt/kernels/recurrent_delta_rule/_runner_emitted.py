# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Auto-generated runner for ttlang_kernel."""

import ttnn

GRID_COLS = 1
GRID_ROWS = 1
NUM_TENSORS = 8

KERNEL_PATHS = [
    ("/tmp/tt-admin/ttlang_kernel_recurrent_compute_6032056d.cpp", "compute"),
    ("/tmp/tt-admin/ttlang_kernel_recurrent_read_2513cc96.cpp", "noc"),
    ("/tmp/tt-admin/ttlang_kernel_recurrent_write_e4e045f2.cpp", "noc"),
]

KERNEL_TENSOR_INDICES = [
    [],  # compute
    [5, 4, 2, 0, 3],  # noc
    [7, 6],  # noc
]

CB_CONFIGS = [
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 0
    None,  # CB 1
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 2
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 3
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 4
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 5
    ((1, 1), 4, ttnn.float32, 4096, 16384),  # CB 6
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 7
    ((1, 1), 2, ttnn.float32, 4096, 8192),  # CB 8
]


def run(tensors, device=None):
    """Run the ttlang_kernel on device."""
    assert len(tensors) == 8, f"Expected 8 tensors, got {len(tensors)}"

    if device is None:
        device = tensors[0].device()

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_COLS - 1, GRID_ROWS - 1))]
    )

    tensor_accessor_args = []
    for tensor in tensors:
        tensor_accessor_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

    cb_descriptors = []
    for i, (shape, block_count, dtype, page_size, total_size) in enumerate(CB_CONFIGS):
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=dtype,
            page_size=page_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        cb_descriptors.append(cb_desc)

    cb_indices = list(range(9))
    kernel_descriptors = []
    noc_idx = 0

    for kernel_idx, (kernel_path, thread_type) in enumerate(KERNEL_PATHS):
        tensor_indices = KERNEL_TENSOR_INDICES[kernel_idx]
        common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]

        if thread_type == "compute":
            compile_time_args = cb_indices
            config = ttnn.ComputeConfigDescriptor()
        else:
            compile_time_args = cb_indices + tensor_accessor_args
            if noc_idx == 0:
                config = ttnn.ReaderConfigDescriptor()
            else:
                config = ttnn.WriterConfigDescriptor()
            noc_idx += 1

        kernel_desc = ttnn.KernelDescriptor(
            kernel_source=kernel_path,
            core_ranges=core_ranges,
            compile_time_args=compile_time_args,
            common_runtime_args=common_runtime_args,
            config=config,
        )
        kernel_descriptors.append(kernel_desc)

    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
    )

    return ttnn.generic_op(list(tensors), program)


if __name__ == "__main__":
    print("Runner generated. See run() function for usage.")
