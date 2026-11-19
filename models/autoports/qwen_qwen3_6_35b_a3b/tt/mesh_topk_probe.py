# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental exact FP32 256-to-8 selector for replicated B1 mesh logits.

All persistent tensors and descriptors are setup-owned. Each instance is a
serial workspace: its two output allocations outlive every trace using it, and
the caller must consume outputs before another call overwrites them. No alias
is freed or reassigned while a trace is live. Padding is initialized once and
never changed; each invocation overwrites all eight logical values and IDs.
"""

import torch

import ttnn

SELECTOR_CPP = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// IEEE bitwise numeric order, preserving every FP32 mantissa bit. NaNs sort
// above numbers, signed zeros compare equal, and equal keys keep the lower ID.
inline bool greater_fp32(uint32_t a, uint32_t b) {
    const bool an = (a & 0x7fffffffU) > 0x7f800000U;
    const bool bn = (b & 0x7fffffffU) > 0x7f800000U;
    if (an || bn) return an && !bn;
    if ((a & 0x7fffffffU) == 0) a = 0;
    if ((b & 0x7fffffffU) == 0) b = 0;
    const uint32_t ak = (a & 0x80000000U) ? ~a : (a ^ 0x80000000U);
    const uint32_t bk = (b & 0x80000000U) ? ~b : (b ^ 0x80000000U);
    return ak > bk;
}

void kernel_main() {
    constexpr uint32_t vo = get_compile_time_arg_val(0);
    constexpr uint32_t io = get_compile_time_arg_val(1);
    const auto input = TensorAccessor(TensorAccessorArgs<2>(), get_arg_val<uint32_t>(0));
    const auto values = TensorAccessor(TensorAccessorArgs<vo>(), get_arg_val<uint32_t>(1));
    const auto indices = TensorAccessor(TensorAccessorArgs<io>(), get_arg_val<uint32_t>(2));
    Noc noc;
    DataflowBuffer scratch(0);
    scratch.reserve_back(1);
    const uint32_t bias = (64 - (get_write_ptr(0) & 63)) & 63;
    auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0) + bias);
    // Row zero occupies the first 16 words of the top-left/top-right faces.
    for (uint32_t tile = 0; tile < 8; ++tile) {
        noc.async_read(input, scratch, 64,
            {.page_id=tile, .offset_bytes=0}, {.offset_bytes=bias+tile*128});
        noc.async_read(input, scratch, 64,
            {.page_id=tile, .offset_bytes=1024}, {.offset_bytes=bias+tile*128+64});
    }
    noc.async_read_barrier();
    uint32_t best[8], ids[8];
    uint32_t count = 0;
    for (uint32_t i = 0; i < 256; ++i) {
        const uint32_t value = data[i];
        if (count == 8 && !greater_fp32(value, best[7])) continue;
        uint32_t pos = count < 8 ? count++ : 7;
        while (pos > 0 && greater_fp32(value, best[pos-1])) {
            best[pos] = best[pos-1]; ids[pos] = ids[pos-1]; --pos;
        }
        best[pos] = value; ids[pos] = i;
    }
    for (uint32_t i = 0; i < 8; ++i) {
        data[512+i] = best[i];
        data[528+i] = ids[i];
    }
    scratch.push_back(1);
    scratch.wait_front(1);
    noc.async_write(scratch, values, 32, {.offset_bytes=bias+2048}, {.page_id=0});
    noc.async_write(scratch, indices, 32, {.offset_bytes=bias+2112}, {.page_id=0});
    noc.async_write_barrier();
    scratch.pop_front(1);
}
"""


class MeshFP32TopK:
    def __init__(self, mesh):
        self.mesh = mesh
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value, dtype):
            return ttnn.from_torch(
                value,
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        self.input_template = upload(torch.zeros(1, 1, 1, 256), ttnn.float32)
        self.values = upload(torch.zeros(1, 1, 1, 8), ttnn.float32)
        self.ids = upload(torch.zeros(1, 1, 1, 8, dtype=torch.int32), ttnn.uint32)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        accessors = [ttnn.TensorAccessorArgs(t) for t in (self.input_template, self.values, self.ids)]
        assert all(not a.get_common_runtime_args() for a in accessors)
        xa, va, ia = [a.get_compile_time_args() for a in accessors]
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = [self.input_template.buffer_address(), self.values.buffer_address(), self.ids.buffer_address()]
        self.program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=SELECTOR_CPP,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=[2 + len(xa), 2 + len(xa) + len(va)] + xa + va + ia,
                    runtime_args=runtime,
                    config=ttnn.ReaderConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=4096,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 4096)],
                )
            ],
        )

    def __call__(self, logits):
        if tuple(logits.shape) != (1, 1, 1, 256) or logits.dtype != ttnn.float32:
            raise ValueError("MeshFP32TopK requires logical [1,1,1,256] FP32 logits")
        if logits.layout != ttnn.TILE_LAYOUT or logits.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("MeshFP32TopK requires DRAM-interleaved tiled logits")
        self.program.kernels[0].runtime_args[0][0] = [
            logits.buffer_address(),
            self.values.buffer_address(),
            self.ids.buffer_address(),
        ]
        ttnn.generic_op([logits, self.values, self.ids], self.program)
        return self.values, self.ids
