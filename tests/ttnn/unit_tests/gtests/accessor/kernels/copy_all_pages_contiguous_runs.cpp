// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Copies all pages from the input tensor to the output tensor one contiguous run at a time, using
AbstractTensorAccessorWrapper::num_contiguous_pages. Works for both sharded and interleaved tensors.
This kernel is expected to be executed on only one core (RISCV_0).
*/

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    const uint32_t cb_id = get_compile_time_arg_val(args_dst.next_compile_time_args_offset());
    const uint32_t page_size = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 1);
    const uint32_t volume_arg = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 2);
    const uint32_t cb_num_pages = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 3);

    const uint32_t input_base_address = get_common_arg_val<uint32_t>(0);
    const uint32_t output_base_address = get_common_arg_val<uint32_t>(1);

    const auto tensor_accessor_src = TensorAccessor(args_src, input_base_address);
    const auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address);
    const auto tensor_accessors_tuple = std::make_tuple(tensor_accessor_src, tensor_accessor_dst);
    const auto wrappers = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

#if INTERLEAVED_LAYOUT
    const uint32_t tensor_volume = volume_arg;
#else
    // The buffer's page count includes shard padding, so take the volume from the dspec instead.
    const uint32_t tensor_volume = tensor_accessor_src.dspec().tensor_volume();
#endif

    // One reserve is enough: this kernel owns the CB and only uses it as scratch.
    cb_reserve_back(cb_id, cb_num_pages);
    const uint32_t l1_addr = get_write_ptr(cb_id);

    // Runs step page ids by page_stride, so one walk per residue class covers every page once.
    // A run stops at the shard edge, so a class usually takes several runs, not one.
    const uint32_t page_stride = wrappers[0].contiguous_page_stride();
    // Src and dst runs must live in the same residue class for a page-aligned copy.
    ASSERT(page_stride == wrappers[1].contiguous_page_stride());

    for (uint32_t base = 0; base < page_stride; ++base) {
        for (uint32_t page_id = base; page_id < tensor_volume;) {
            const uint32_t src_pages = wrappers[0].num_contiguous_pages(page_id, tensor_volume);
            const uint32_t dst_pages = wrappers[1].num_contiguous_pages(page_id, tensor_volume);
            const uint32_t run_pages = src_pages < dst_pages ? src_pages : dst_pages;
            const uint64_t src_addr = wrappers[0].get_noc_addr(page_id);
            const uint64_t dst_addr = wrappers[1].get_noc_addr(page_id);

            // Chunk by what the CB can hold: a run can be far larger than L1.
            for (uint32_t done = 0; done < run_pages;) {
                const uint32_t left = run_pages - done;
                const uint32_t chunk = left < cb_num_pages ? left : cb_num_pages;
                const uint32_t byte_offset = done * page_size;

                noc_async_read(src_addr + byte_offset, l1_addr, chunk * page_size);
                noc_async_read_barrier();

                noc_async_write(l1_addr, dst_addr + byte_offset, chunk * page_size);
                noc_async_write_barrier();

                done += chunk;
            }

            page_id += run_pages * page_stride;
        }
    }
}
