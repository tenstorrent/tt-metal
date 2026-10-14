# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record sampled uint32 rows without a host read or a whole-history copy."""

from pathlib import Path

import ttnn


def history_program(tokens, index, history):
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    compile_args = [history.shape[2]]
    for tensor in (tokens, index, history):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[0][0] = [t.buffer_address() for t in (tokens, index, history)]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "record_tokens.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=compile_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    scratch = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=256)],
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[scratch])
