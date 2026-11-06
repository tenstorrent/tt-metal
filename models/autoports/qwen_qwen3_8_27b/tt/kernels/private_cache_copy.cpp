// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t pairs_addr = get_arg_val<uint32_t>(2);
    const uint32_t records = get_arg_val<uint32_t>(3);
    const uint32_t pages_per_record = get_arg_val<uint32_t>(4);
    const uint32_t worker = get_arg_val<uint32_t>(5);
    const uint32_t workers = get_arg_val<uint32_t>(6);
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto map_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const auto pairs = TensorAccessor(map_args, pairs_addr);
    const uint32_t staging = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t mapping = get_write_ptr(tt::CBIndex::c_1);
    for (uint32_t index = worker; index < records * pages_per_record; index += workers) {
        const uint32_t record = index / pages_per_record;
        const uint32_t within = index % pages_per_record;
        noc_async_read(get_noc_addr(record, pairs), mapping, 8);
        noc_async_read_barrier();
        const auto* entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mapping);
        const uint32_t source_page = entry[0] * pages_per_record + within;
        const uint32_t destination_page = entry[1] * pages_per_record + within;
        noc_async_read_page(source_page, src, staging);
        noc_async_read_barrier();
        noc_async_write_page(destination_page, dst, staging);
        noc_async_write_barrier();
    }
}
