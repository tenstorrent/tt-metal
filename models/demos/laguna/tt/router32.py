# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact fp32 top-K routing for one 32-token decode tile as one generic_op (Laguna). See kernels/router32_reader.cpp."""

import struct
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def route32(sel, scores, top_k, routed_scaling, norm_topk_prob, memory_config=ttnn.L1_MEMORY_CONFIG):
    """sel, scores: [1, 1, 32, E] fp32 TILE interleaved. Returns the dense fp32 routing matrix [1, 1, 32, E]."""
    device = sel.device()
    E = sel.shape[-1]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, E]), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config)
    grid_size = device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(32, grid_size, True)
    buf = ttnn.CBDescriptor(
        total_size=4 * E * 4,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4 * E * 4)],
    )
    args = []
    for t in (sel, scores, out):
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    scale_bits = struct.unpack("<I", struct.pack("<f", float(routed_scaling)))[0]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "router32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[E, int(top_k), grid_size.x, int(bool(norm_topk_prob)), scale_bits, 4096] + args,
        common_runtime_args=[sel.buffer_address(), scores.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[buf])
    ttnn.generic_op([sel, scores, out], program)
    return out
