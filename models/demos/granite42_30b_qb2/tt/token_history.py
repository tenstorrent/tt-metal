# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Append sampled UINT32 tokens to preallocated history without host work."""

from pathlib import Path

import ttnn


def append_tokens(tokens, history, cursor):
    tensors = [tokens, history, cursor]
    if any(t.dtype != ttnn.uint32 or t.layout != ttnn.ROW_MAJOR_LAYOUT for t in tensors):
        raise ValueError("Token history requires row-major UINT32 tensors")
    batch = tokens.shape[-1]
    if tuple(tokens.shape) != (1, 1, 1, batch) or tuple(history.shape)[1:] != (1, 1, batch):
        raise ValueError("History rows must match the physical token bucket")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    args = [batch, history.shape[0]]
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/token_history.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((0, 0), [t.buffer_address() for t in tensors])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    scratch = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=256)],
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch], semaphores=[]))
