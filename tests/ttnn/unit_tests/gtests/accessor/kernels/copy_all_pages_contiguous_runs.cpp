// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Copies all pages from the input tensor to the output tensor one contiguous run at a time, using
TensorAccessor::num_contiguous_pages. Works for both sharded and interleaved tensors.
This kernel is expected to be executed on only one core (RISCV_0).
*/

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    const uint32_t cb_id = get_compile_time_arg_val(args_dst.next_compile_time_args_offset());
    const uint32_t volume_arg = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 1);
    const uint32_t cb_num_pages = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 2);

    const uint32_t input_base_address = get_common_arg_val<uint32_t>(0);
    const uint32_t output_base_address = get_common_arg_val<uint32_t>(1);

#ifdef EXPLICIT_PAGE_SIZE
    // An explicit page size that may be unaligned; pages then sit the aligned size apart in each bank.
    const auto tensor_accessor_src = TensorAccessor(args_src, input_base_address, EXPLICIT_PAGE_SIZE);
    const auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address, EXPLICIT_PAGE_SIZE);
#else
    const auto tensor_accessor_src = TensorAccessor(args_src, input_base_address);
    const auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address);
#endif

#if INTERLEAVED_LAYOUT
    const uint32_t tensor_volume = volume_arg;
#else
    // Buffer page count includes shard padding; the dspec volume does not.
    const uint32_t tensor_volume = tensor_accessor_src.dspec().tensor_volume();
#endif

    // The CB is only scratch L1.
    cb_reserve_back(cb_id, cb_num_pages);
    CoreLocalMem<uint32_t> scratch(get_write_ptr(cb_id));
    Noc noc;

    // Runs step page ids by page_stride, so one walk per residue class covers every page once.
    const uint32_t page_stride = tensor_accessor_src.contiguous_page_stride();
    ASSERT(page_stride == tensor_accessor_dst.contiguous_page_stride());

    const uint32_t page_size = tensor_accessor_src.get_aligned_page_size();
    for (uint32_t base = 0; base < page_stride; ++base) {
        for (uint32_t page_id = base; page_id < tensor_volume;) {
            const uint32_t src_pages = tensor_accessor_src.num_contiguous_pages(page_id, tensor_volume);
            const uint32_t dst_pages = tensor_accessor_dst.num_contiguous_pages(page_id, tensor_volume);
            const uint32_t run_pages = src_pages < dst_pages ? src_pages : dst_pages;

            // A run can exceed the CB; copy it in CB-sized chunks.
            for (uint32_t done = 0; done < run_pages;) {
                const uint32_t left = run_pages - done;
                const uint32_t chunk = left < cb_num_pages ? left : cb_num_pages;
                const uint32_t byte_offset = done * page_size;

                noc.async_read(
                    tensor_accessor_src,
                    scratch,
                    chunk * page_size,
                    {.page_id = page_id, .offset_bytes = byte_offset},
                    {.offset_bytes = 0});
                noc.async_read_barrier();

                noc.async_write(
                    scratch,
                    tensor_accessor_dst,
                    chunk * page_size,
                    {.offset_bytes = 0},
                    {.page_id = page_id, .offset_bytes = byte_offset});
                noc.async_write_barrier();

                done += chunk;
            }

            page_id += run_pages * page_stride;
        }
    }
}
