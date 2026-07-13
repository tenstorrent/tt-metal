# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe bounded-memory uniform generation from a device seed tensor.

Metal Trace bakes ordinary ``ttnn.rand(seed=<python int>)`` runtime arguments into
the captured command stream. This DG-local generic op reads the base seed from a
persistent device tensor instead, so the host can refresh one scalar outside the
trace and replay the same chunked-Gumbel graph with a different block seed.
"""

from __future__ import annotations

import math
import struct

import ttnn

_READER_KERNEL = "models/experimental/diffusion_gemma/tt/kernels/read_trace_seed.cpp"
_COMPUTE_KERNEL = "models/experimental/diffusion_gemma/tt/kernels/compute_trace_seeded_uniform.cpp"
_WRITER_KERNEL = "ttnn/cpp/ttnn/operations/uniform/device/kernels/writer_uniform.cpp"
_PLAN_CACHE = {}


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _active_core_plan(device, num_tiles: int):
    key = (id(device), num_tiles)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    grid = device.compute_with_storage_grid_size()
    all_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(int(grid.x) - 1, int(grid.y) - 1))]
    )
    cores = ttnn.corerange_to_cores(all_grid, row_wise=True)
    num_cores = min(len(cores), num_tiles)
    base, extra = divmod(num_tiles, num_cores)
    active = []
    start = 0
    for core_index, core in enumerate(cores[:num_cores]):
        count = base + (1 if core_index < extra else 0)
        active.append((core, core_index, start, count))
        start += count
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _, _, _ in active])
    plan = (active, core_set)
    _PLAN_CACHE[key] = plan
    return plan


def allocate_uniform_buffer(device, shape, *, dtype=ttnn.float32):
    if dtype != ttnn.float32:
        raise ValueError("trace-seeded uniform currently supports float32 output only")
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError("uniform buffer dimensions must be positive")
    if int(shape[-1]) % ttnn.TILE_SIZE != 0 or int(shape[-2]) % ttnn.TILE_SIZE != 0:
        raise ValueError("trace-seeded uniform buffer requires tile-aligned trailing dimensions")
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def _program(seed_tensor, output_tensor, *, seed_offset: int):
    tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    seed_page = tile.get_tile_size(ttnn.uint32)
    output_page = tile.get_tile_size(ttnn.float32)
    num_tiles = math.prod(int(dim) for dim in output_tensor.shape) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
    active, core_set = _active_core_plan(output_tensor.device(), num_tiles)

    seed_cb = 0
    output_cb = 24
    cbs = [
        ttnn.CBDescriptor(
            total_size=seed_page,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=seed_cb, data_format=ttnn.uint32, page_size=seed_page)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * output_page,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=output_cb, data_format=ttnn.float32, page_size=output_page)
            ],
        ),
    ]

    reader_ct = [seed_cb]
    reader_ct.extend(ttnn.TensorAccessorArgs(seed_tensor).get_compile_time_args())
    reader_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    from_bits = _float_bits(0.0)
    to_bits = _float_bits(1.0 - 1.0e-6)
    for core, core_index, start, count in active:
        reader_rt[core.x][core.y] = [seed_tensor.buffer_address()]
        compute_rt[core.x][core.y] = [int(seed_offset), core_index, count, from_bits, to_bits]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, count]

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[seed_cb, output_cb],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=True,
        ),
    )
    writer_ct = [output_cb, output_cb]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        defines=[("OUTPUT_DTYPE_FLOAT32", "1")],
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def trace_seeded_uniform(seed_tensor, output_tensor, *, seed_offset: int):
    """Fill ``output_tensor`` with U[0,1) using ``seed_tensor[0] + seed_offset``."""
    if seed_tensor.get_dtype() != ttnn.uint32:
        raise ValueError("trace seed tensor must be uint32")
    if output_tensor.get_dtype() != ttnn.float32:
        raise ValueError("trace-seeded uniform output must be float32")
    if seed_tensor.device() is not output_tensor.device():
        raise ValueError("seed and output must be on the same device")
    if not 0 <= int(seed_offset) <= 0xFFFFFFFF:
        raise ValueError("seed_offset must fit uint32")
    device = output_tensor.device()
    if hasattr(device, "shape") and device.get_num_devices() > 1:
        seeds = ttnn.get_device_tensors(seed_tensor)
        outputs = ttnn.get_device_tensors(output_tensor)
        if len(seeds) != len(outputs) or len(outputs) != device.get_num_devices():
            raise ValueError("trace-seeded uniform mesh tensors must cover every device")
        mesh_program = ttnn.MeshProgramDescriptor()
        rows, cols = int(device.shape[0]), int(device.shape[1])
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                coord = ttnn.MeshCoordinate(row, col)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = _program(
                    seeds[index],
                    outputs[index],
                    seed_offset=int(seed_offset),
                )
        program = mesh_program
    else:
        program = _program(seed_tensor, output_tensor, seed_offset=int(seed_offset))
    ttnn.generic_op([seed_tensor, output_tensor], program)
    return output_tensor
