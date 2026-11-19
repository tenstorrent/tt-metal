# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qualified narrow activation reader; baseline FP32 FPU/cumsum is unchanged.

The reader source equals the accepted read32 probe. Its two 32-byte reads cover
both valid BF16 half rows; destination bias preserves the Blackhole DRAM NoC
alignment. No SFPU dead-store experiment is enabled here. The helper remains
replaceable at the decoder's direct_coefficient_scan boundary.
"""

import math

from .fused_decoder import (
    STREAMING_SCAN_COMPUTE,
    STREAMING_SCAN_PRODUCT_WRITER,
    STREAMING_SCAN_WRITER,
    DirectCoefficientStreamingScan,
    terminal_projection_schedule,
)

NARROW_COEFFICIENT_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t activation_offset=get_compile_time_arg_val(0);
 constexpr uint32_t output_offset=get_compile_time_arg_val(1);
 constexpr bool coefficients_only=get_compile_time_arg_val(2)!=0;
 const auto weight=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));
 const auto activation=TensorAccessor(TensorAccessorArgs<activation_offset>(),get_arg_val<uint32_t>(1));
 const auto coefficient_output=TensorAccessor(TensorAccessorArgs<output_offset>(),get_arg_val<uint32_t>(6));
 const uint32_t start=get_arg_val<uint32_t>(2),runs=get_arg_val<uint32_t>(3);
 const uint32_t ht=get_arg_val<uint32_t>(4),wt=get_arg_val<uint32_t>(5);
 Noc noc;DataflowBuffer a(0),b(1),raw(2);
 // Initialize both contiguous coefficient CB pages before publishing either.
 // Only column0 changes afterward; padded columns stay zero on every reuse.
 b.reserve_back(2);
 auto* initial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 for(uint32_t i=0;i<1024;++i) initial[i]=0;
 for(uint32_t run=start;run<start+runs;++run) {
  if(coefficients_only && run%wt!=0) continue;
  for(uint32_t h=0;h<ht;++h) {
   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;
   const uint32_t column=(run/wt)*ht+h;
   raw.reserve_back(1);b.reserve_back(1);
   if constexpr(!coefficients_only) {
    a.reserve_back(1);
    noc.async_read(weight,a,2048,{.page_id=tile},{.offset_bytes=0});
   }
   // BH requires source/destination low six bits to agree.
   const uint32_t raw_bias=(64-(get_write_ptr(2)&63))&63;
   noc.async_read(activation,raw,32,{.page_id=column,.offset_bytes=0},{.offset_bytes=raw_bias});
   noc.async_read(activation,raw,32,{.page_id=column,.offset_bytes=512},{.offset_bytes=raw_bias+512});
   noc.async_read_barrier();raw.push_back(1);raw.wait_front(1);
   auto* source=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(2)+raw_bias);
   auto* target=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(1));
   // Original activation TILE row0, pair-flipped inside each 32-lane tile.
   // Coefficient TILE column0: face0 rows0..15, face2 rows16..31.
   for(uint32_t r=0;r<32;++r) {
    const uint32_t paired=r^1;
    const uint32_t src=(paired/16)*256+paired%16;
    const uint32_t dst=(r/16)*512+(r%16)*16;
    target[dst]=source[src];
   }
   raw.pop_front(1);b.push_back(1);
   if constexpr(coefficients_only) {
    b.wait_front(1);
    noc.async_write(b,coefficient_output,2048,{}, {.page_id=column});
    noc.async_write_barrier();b.pop_front(1);
   } else { a.push_back(1); }
  }
 }
}
"""


class NarrowCoefficientStreamingScan:
    """Same streaming FPU/cumsum, with raw BF16 original-input coefficient construction."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = (ttnn, mesh)
        self.grid = mesh.compute_with_storage_grid_size()
        self.reader_source = NARROW_COEFFICIENT_READER
        self.compute_source = STREAMING_SCAN_COMPUTE
        self.baseline = DirectCoefficientStreamingScan(ttnn, mesh)

    def supports(self, weight, activation):
        t = self.ttnn
        shape = tuple(weight.shape)
        # Exact real input/output/router contracts qualified in stage7. Split
        # projection experiments and other shapes keep their original helper.
        if shape not in (
            (4, 1, 512, 3136),
            (2, 1, 512, 2048),
            (4, 1, 512, 256),
            (2, 1, 1024, 3136),
            (2, 1, 1024, 256),
        ):
            return False
        total_k = shape[0] * shape[2]
        memory = activation.memory_config()
        source_memory = memory in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG) or (
            memory.buffer_type == t.BufferType.L1 and memory.memory_layout == t.TensorMemoryLayout.WIDTH_SHARDED
        )
        return (
            tuple(activation.shape) == (1, 1, total_k)
            and tuple(activation.padded_shape) == (1, 32, total_k)
            and weight.dtype == activation.dtype == t.bfloat16
            and weight.layout == activation.layout == t.TILE_LAYOUT
            and weight.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
            and source_memory
        )

    def __call__(self, weight, activation, *, products_only=False, coefficients_only=False):
        if not self.supports(weight, activation):
            return self.baseline(weight, activation, products_only=products_only, coefficients_only=coefficients_only)
        t = self.ttnn
        shape = tuple(weight.shape)
        assert shape[-2] in (512, 1024) and shape[-1] % 32 == 0 and (len(shape) == 4) and (shape[1] == 1)
        total_k = math.prod(shape[:-2]) * shape[-2]
        assert tuple(activation.shape) == (1, 1, total_k)
        assert not (products_only and coefficients_only)
        assert weight.dtype == activation.dtype == t.bfloat16 and weight.layout == activation.layout == t.TILE_LAYOUT
        assert weight.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        ht, wt = (shape[-2] // 32, shape[-1] // 32)
        partition = terminal_projection_schedule(math.prod(shape[:-2]) * wt, self.grid.x, self.grid.y)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partition])
        output_shape = (
            shape[:-1] + (1,) if coefficients_only else shape if products_only else shape[:-2] + (1, shape[-1])
        )
        output = t.empty(
            output_shape,
            device=self.mesh,
            dtype=t.bfloat16 if coefficients_only else t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG if products_only or coefficients_only else t.L1_MEMORY_CONFIG,
        )
        rd, wr, cr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partition:
            rd[x][y] = [
                weight.buffer_address(),
                activation.buffer_address(),
                start,
                count,
                ht,
                wt,
                output.buffer_address(),
            ]
            wr[x][y] = [output.buffer_address(), start, count, ht, wt]
            cr[x][y] = [count]
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )

        def kernel(source, conf, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=args,
                runtime_args=runtime,
            )

        accessors = [t.TensorAccessorArgs(value) for value in (weight, activation, output)]
        assert all((not a.get_common_runtime_args() for a in accessors))
        wa, xa, oa = [a.get_compile_time_args() for a in accessors]
        kernels = [
            kernel(
                self.reader_source,
                t.ReaderConfigDescriptor(),
                [3 + len(wa), 3 + len(wa) + len(xa), int(coefficients_only)] + wa + xa + oa,
                rd,
            )
        ]
        if not coefficients_only:
            kernels += [
                kernel(self.compute_source, config, [ht, int(products_only)], cr),
                kernel(
                    STREAMING_SCAN_PRODUCT_WRITER if products_only else STREAMING_SCAN_WRITER,
                    t.WriterConfigDescriptor(),
                    oa,
                    wr,
                ),
            ]
        program = t.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                t.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, page)],
                )
                for index, pages, page, dtype in [
                    (0, 2, 2048, t.bfloat16),
                    (1, 2, 2048, t.bfloat16),
                    (2, 1, 2048, t.bfloat16),
                    (16, 1, 4096, t.float32),
                    (17, 1, 4096, t.float32),
                ]
            ],
        )
        t.generic_op([weight, activation, output], program)
        return output
