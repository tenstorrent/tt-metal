# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated per-worker coefficient cache; unchanged run-major exact arithmetic.

CB1 retains one K block until its last local output-column run consumes it.
No production implementation or caller is changed by importing this module.
"""

import math

from .fused_decoder import STREAMING_SCAN_PRODUCT_WRITER, STREAMING_SCAN_WRITER, terminal_projection_schedule
from .scan_terminal_probe import compute_source

CACHE_READER = r"""
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
 // Full-depth reserve pins one contiguous cache. Compute pops exactly ht
 // only after the last local output-column run belonging to this K block.
 b.reserve_back(ht);
 auto* initial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 for(uint32_t i=0;i<ht*512;++i) initial[i]=0;
 uint32_t cached_block=0xffffffff;
 for(uint32_t run=start;run<start+runs;++run) {
  if(coefficients_only && run%wt!=0) continue;
  const uint32_t block=run/wt;
  if(block!=cached_block) {
   b.reserve_back(ht);
   const uint32_t cache_base=get_write_ptr(1);
   for(uint32_t h=0;h<ht;++h) {
    const uint32_t column=block*ht+h;
    raw.reserve_back(1);
    STAGE7_ACTIVATION_READ
    noc.async_read_barrier();raw.push_back(1);raw.wait_front(1);
    auto* source=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(2)+raw_bias);
    auto* target=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cache_base+h*2048);
    for(uint32_t r=0;r<32;++r) {
     const uint32_t paired=r^1;
     const uint32_t src=(paired/16)*256+paired%16;
     const uint32_t dst=(r/16)*512+(r%16)*16;
     target[dst]=source[src];
    }
    raw.pop_front(1);
   }
   b.push_back(ht);cached_block=block;
  }
  if constexpr(coefficients_only) {
   b.wait_front(ht);
   for(uint32_t h=0;h<ht;++h)
    noc.async_write(b,coefficient_output,2048,{.offset_bytes=h*2048},{.page_id=block*ht+h});
   noc.async_write_barrier();b.pop_front(ht);
  } else {
   for(uint32_t h=0;h<ht;++h) {
    a.reserve_back(1);
    const uint32_t tile=block*ht*wt+h*wt+run%wt;
    noc.async_read(weight,a,2048,{.page_id=tile},{.offset_bytes=0});
    noc.async_read_barrier();a.push_back(1);
   }
  }
 }
}
"""


def cache_reader_source(read_bytes, padding="zero"):
    if read_bytes == 2048:
        read = (
            "const uint32_t raw_bias=0;\n    noc.async_read(activation,raw,2048,{.page_id=column},{.offset_bytes=0});"
        )
    elif read_bytes in (32, 64):
        read = f"""const uint32_t raw_bias=(64-(get_write_ptr(2)&63))&63;
    noc.async_read(activation,raw,{read_bytes},{{.page_id=column,.offset_bytes=0}},{{.offset_bytes=raw_bias}});
    noc.async_read(activation,raw,{read_bytes},{{.page_id=column,.offset_bytes=512}},{{.offset_bytes=raw_bias+512}});"""
    else:
        raise ValueError("read_bytes must be 2048,32,64")
    source = CACHE_READER.replace("STAGE7_ACTIVATION_READ", read)
    clear = " auto* initial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n for(uint32_t i=0;i<ht*512;++i) initial[i]=0;"
    assert source.count(clear) == 1
    if padding == "none":
        source = source.replace(
            clear,
            " // Only column0 is consumed by SRCB_BCAST_COL. Every valid lane is filled below.\n // Other physical coefficient padding is unspecified, never a logical output.",
        )
    elif padding == "poison":
        source = source.replace(clear, clear.replace("initial[i]=0;", "initial[i]=0x7fc17fc1;"))
    elif padding != "zero":
        raise ValueError("padding must be zero, none, poison")
    return source


def cache_compute_source(stores):
    source = compute_source(stores)
    replacements = {
        " const uint32_t runs=get_arg_val<uint32_t>(0);": " const uint32_t runs=get_arg_val<uint32_t>(0),start=get_arg_val<uint32_t>(1),wt=get_arg_val<uint32_t>(2);",
        " for(uint32_t run=0;run<runs;++run) {": " for(uint32_t run=0;run<runs;++run) {\n  b.wait_front(ht);",
        "a.wait_front(1);b.wait_front(1);tile_regs_acquire();": "a.wait_front(1);tile_regs_acquire();",
        "mul_tiles_bcast_cols(0,1,0,0,0);": "mul_tiles_bcast_cols(0,1,0,h,0);",
        "tile_regs_commit();a.pop_front(1);b.pop_front(1);tile_regs_wait();": "tile_regs_commit();a.pop_front(1);tile_regs_wait();",
        "   tile_regs_release();\n  }\n }": "   tile_regs_release();\n  }\n  if(run+1==runs || (start+run+1)/wt!=(start+run)/wt) b.pop_front(ht);\n }",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


class CachedCoefficientScan:
    def __init__(self, ttnn, mesh, *, read_bytes=2048, stores="baseline", padding="zero"):
        self.ttnn, self.mesh = ttnn, mesh
        self.grid = mesh.compute_with_storage_grid_size()
        self.reader_source = cache_reader_source(read_bytes, padding)
        self.compute_source = cache_compute_source(stores)

    def __call__(self, weight, activation, *, products_only=False, coefficients_only=False):
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
            cr[x][y] = [count, start, wt]
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
                    (1, ht, 2048, t.bfloat16),
                    (2, 1, 2048, t.bfloat16),
                    (16, 1, 4096, t.float32),
                    (17, 1, 4096, t.float32),
                ]
            ],
        )
        t.generic_op([weight, activation, output], program)
        return output
