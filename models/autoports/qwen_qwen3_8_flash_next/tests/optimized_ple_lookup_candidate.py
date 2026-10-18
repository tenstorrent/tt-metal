# SPDX-License-Identifier: Apache-2.0
"""Parallelize independent PLE head lookups without changing BFP4 decoding."""
import ttnn


def lookup_rows(tables,indices,*,output):
    rows=tables[0].shape[-2]
    n=indices.shape[0];mesh=indices.device()
    grid=mesh.compute_with_storage_grid_size()
    _,cores,group1,_,work1,work2=ttnn.split_work_to_cores(grid,n*16,True)
    work=[];start=0
    for i,core in enumerate(ttnn.corerange_to_cores(cores,row_wise=True)):
        count=work1 if i<group1.num_cores() else work2
        work.append(((core.x,core.y),[indices.buffer_address(),output.buffer_address(),start,count,*[t.buffer_address() for t in tables]]))
        start+=count
    args=[rows,len(tables),*ttnn.TensorAccessorArgs(indices).get_compile_time_args(),
          *ttnn.TensorAccessorArgs(output).get_compile_time_args(),
          *ttnn.TensorAccessorArgs(tables[0]).get_compile_time_args()]
    cb=ttnn.CBDescriptor(total_size=1024,core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=1024)])
    kernel=ttnn.KernelDescriptor(kernel_source=SOURCE,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,compile_time_args=args,runtime_args=work,config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([*tables,indices,output],ttnn.ProgramDescriptor(kernels=[kernel],cbs=[cb]))
    return output

SOURCE=r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Decode BFP4_b exactly as blockfloat_common.cpp, directly into BF16 bits.
static uint16_t decode(uint8_t nibble, uint8_t exponent) {
    uint32_t mantissa = nibble & 7;
    if (!mantissa) {
        return 0;
    }
    uint32_t shifts = 0;
    while (!(mantissa & 4)) {
        mantissa <<= 1;
        ++shifts;
    }
    mantissa = (mantissa << 1) & 7;
    return ((nibble >> 3) << 15) | ((exponent - shifts) << 7) | (mantissa << 4);
}

void kernel_main() {
    Noc noc;
    constexpr uint32_t rows_per_shard = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr auto indices_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto indices = TensorAccessor(indices_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t scratch = cb.get_write_ptr();
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto tile = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch + 64);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 640);
    for (uint32_t work = start; work < start + count; ++work) {
        const uint32_t token=work/16, head=work%16;
        noc.async_read(indices, CoreLocalMem<uint32_t>(scratch), 64, {.page_id = token}, {});
        noc.async_read_barrier();
        {
            const uint32_t index = ids[head];
            const uint32_t shard = index / rows_per_shard;
            ASSERT(shard < num_shards);
            const uint32_t row = index % rows_per_shard;
            const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(4 + shard));
            for (uint32_t col_tile = 0; col_tile < 5; ++col_tile) {
                noc.async_read(
                    table, CoreLocalMem<uint32_t>(scratch + 64), 576, {.page_id = (row / 32) * 5 + col_tile}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t face = (row % 32 / 16) * 2 + c / 16;
                    const uint32_t face_row = row % 16;
                    const uint32_t elem = face * 256 + face_row * 16 + c % 16;
                    const uint8_t packed = tile[64 + elem / 2];
                    const uint8_t nibble = (packed >> (4 * (elem % 2))) & 15;
                    out[col_tile * 32 + c] = decode(nibble, tile[face * 16 + face_row]);
                }
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(scratch + 640), output, 320, {}, {.page_id = token, .offset_bytes = head * 320});
        noc.async_write_barrier();
    }
}

"""
