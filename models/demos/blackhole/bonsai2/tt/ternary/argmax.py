# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy decode token in one op (kernels/argmax_row0.cpp): the argmax of row 0 of the bf16 TILE logits over the real
vocab, straight from the LM head's output, on the decode cores (core_ranges' layout). Replaces slice + reshape +
typecast + to_layout + argmax, which the weight stream (bonsai2/tt/stream) needs: those stock ops would run on its
sender cores too."""

import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import available, core_xy, row_major_crs

KERNELS = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"
_OUT = {}


def prealloc(device):
    """The result buffer, allocated before any trace is captured (a buffer allocated after a capture can take
    addresses the trace's own buffers use)."""
    if device not in _OUT:
        _OUT[device] = ttnn.from_torch(
            torch.zeros(1, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def row0_argmax(logits: ttnn.Tensor, vocab: int) -> ttnn.Tensor:
    """logits [.., 32-row tile row, Npad] bf16 TILE -> a [1, 16] uint32 tensor whose element 0 is the index."""
    device = logits.device()
    assert logits.dtype == ttnn.bfloat16 and logits.layout == ttnn.TILE_LAYOUT, (logits.dtype, logits.layout)
    assert device in _OUT, "argmax.prealloc(device) before the decode trace is captured"
    out = _OUT[device]
    grid = device.compute_with_storage_grid_size()
    Nt = logits.padded_shape[-1] // 32
    ncores = min(available(grid.x * grid.y), Nt)
    crs = row_major_crs(ncores, grid.x)
    hub = device.worker_core_from_logical_core(core_xy(0, grid.x))

    def cb(index, size):
        return ttnn.CBDescriptor(
            total_size=size,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)],
        )

    ct = ttnn.TensorAccessorArgs(logits).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "argmax_row0.cpp",
                core_ranges=crs,
                compile_time_args=ct,
                common_runtime_args=[
                    logits.buffer_address(),
                    out.buffer_address(),
                    Nt,
                    ncores,
                    grid.x,
                    vocab,
                    hub.x,
                    hub.y,
                ],
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)],
        cbs=[cb(0, 16 * 128), cb(1, 16 * ncores)],
    )
    return ttnn.generic_op([logits, out], program)
