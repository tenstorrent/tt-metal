// SPDX-License-Identifier: Apache-2.0
// Exact byte copies between a live request row and a scheduler-owned snapshot.
// The row IDs are device inputs; changing a prefix block never changes the program.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);
    constexpr uint32_t src_rows = get_compile_time_arg_val(3);
    constexpr uint32_t dst_rows = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const uint32_t src_address = get_arg_val<uint32_t>(0);
    const uint32_t dst_address = get_arg_val<uint32_t>(1);
    const uint32_t ids_address = get_arg_val<uint32_t>(2);
    const uint32_t layer_row = get_arg_val<uint32_t>(3);
    const uint32_t first_page = get_arg_val<uint32_t>(4);
    const auto src = TensorAccessor(src_args, src_address, page_bytes);
    const auto dst = TensorAccessor(dst_args, dst_address, page_bytes);
    const auto ids = TensorAccessor(ids_args, ids_address, 32);
    CircularBuffer staging(0);
    CircularBuffer ids_cb(1);
    Noc noc;
    const CoreLocalMem<volatile uint32_t> row_ids(ids_cb.get_write_ptr());
    noc.async_read(ids, ids_cb, 32, {.page_id = layer_row}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t src_row = row_ids[0];
    const uint32_t dst_row = row_ids[1];
    ASSERT(src_row < src_rows && dst_row < dst_rows);
    for (uint32_t page = first_page; page < pages_per_row; page += num_cores) {
        staging.reserve_back(1);
        noc.async_read(src, staging, page_bytes,
                       {.page_id = src_row * pages_per_row + page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        staging.push_back(1);
        staging.wait_front(1);
        auto staged = use<CircularBuffer::AddrSelector::READ_PTR>(staging);
        noc.async_write(staged, dst, page_bytes, {.offset_bytes = 0},
                        {.page_id = dst_row * pages_per_row + page});
        noc.async_write_barrier();
        staging.pop_front(1);
    }
}
