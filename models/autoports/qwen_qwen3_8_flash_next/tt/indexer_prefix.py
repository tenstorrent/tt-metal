# SPDX-License-Identifier: Apache-2.0
"""Device-only conservative key-prefix bound for qualified B32 serving owners."""

from pathlib import Path

import ttnn


def prepare_indexer_prefix(mesh, positions, output, *, seq, lane, blocks):
    """Write max(512, ceil32(floor((position[lane]+seq)/4))) into output.

    Output is one persistent UINT32 scalar in interleaved DRAM. Callers may
    reuse it between lanes only when this producer and every indexer consumer
    execute serially on CQ0. Neither this function nor the kernel reads a value
    back to the host. The exact score mask must still discard incomplete groups.
    """
    if len(positions.shape) != 2 or positions.shape[1] != 32 or not 1 <= positions.shape[0] <= 32:
        raise ValueError("Expected packed [B,32] positions for B in [1,32]")
    if tuple(output.shape) != (1,):
        raise ValueError("Indexer prefix output must have logical shape [1]")
    for tensor in (positions, output):
        if tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Indexer prefix tensors must be UINT32 ROW_MAJOR")
        if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Indexer prefix tensors must use interleaved DRAM")
    if not isinstance(blocks, int) or not 512 <= blocks <= 0xFFFFFFFF // 4 or blocks % 32:
        raise ValueError("Key capacity must be >=512, tile aligned, and fit uint32 token positions")
    if not isinstance(seq, int) or not 1 <= seq <= blocks * 4:
        raise ValueError("Sequence length must be positive and fit full context")
    if not isinstance(lane, int) or not 0 <= lane < positions.shape[0]:
        raise ValueError("Lane lies outside packed positions")
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tensors = [positions, output]
    args = [positions.shape[0], blocks]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=256)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "indexer_prefix.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((0, 0), [positions.buffer_address(), output.buffer_address(), lane, seq])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
