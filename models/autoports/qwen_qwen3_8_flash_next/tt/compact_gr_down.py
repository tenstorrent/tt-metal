# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental exact Blackhole GR-down compaction after the unchanged gather."""

from pathlib import Path

import ttnn


def compact_gr_down(value, *, output=None):
    """Copy four 80-valid/96-physical rank segments into 320 columns, Q1..32.

    Reproduce the selected Blackhole1x4 slice/concat conversion semantics:
    exponent-zero BF16 values become positive zero, and exponent-all-ones
    values become signed infinity. Other storage bits are unchanged. This
    integer-only mapping matches the exhaustive65536-pattern baseline audit
    in doc/optimized_full_model/gr_compaction_canonicalization_v1.json.
    Physical padding rows become zero. No reduction, collective, floating-point
    arithmetic or host conversion is performed. Production dispatch remains
    unchanged: the integrated mixed-prefill check failed despite isolated tests.
    """
    if not isinstance(value, ttnn.Tensor) or len(value.shape) != 4:
        raise ValueError("Expected a device tensor [1,1,Q,384]")
    count = value.shape[-2]
    if not 1 <= count <= 32:
        raise ValueError("Expected logical Q in1..32")
    device = value.device()
    if device.arch() != ttnn.device.Arch.BLACKHOLE or tuple(device.shape) != (1, 4):
        raise ValueError("Legacy compaction canonicalization is validated only on a Blackhole1x4 mesh")

    def validate(tensor, width):
        tile = tensor.get_tile()
        if (
            not ttnn.is_tensor_storage_on_device(tensor)
            or not tensor.is_allocated()
            or tensor.device() != device
            or tuple(tensor.shape) != (1, 1, count, width)
            or tuple(tensor.padded_shape) != (1, 1, 32, width)
            or tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tuple(tile.tile_shape) != (32, 32)
            or tile.transpose_within_face
            or tile.transpose_of_faces
        ):
            raise ValueError("Expected allocated BF16 standard TILE interleaved DRAM on the same device")

    validate(value, 384)
    if output is None:
        output = ttnn.empty(
            (1, 1, count, 320),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if not isinstance(output, ttnn.Tensor):
        raise ValueError("Expected a device output tensor")
    validate(output, 320)
    if output.buffer_address() == value.buffer_address():
        raise ValueError("Output must not alias input")
    grid = device.compute_with_storage_grid_size()
    used = min(10, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(used)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    args = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        args[core.x][core.y] = [value.buffer_address(), output.buffer_address(), i, used, count]
    compile_args = []
    for tensor in (value, output):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).parent / "kernels/compact_gr_down_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=ranges,
        compile_time_args=compile_args,
        runtime_args=args,
        config=ttnn.WriterConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=2048,
        core_ranges=ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
    )
    ttnn.generic_op([value, output], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))
    return output
