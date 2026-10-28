// SPDX-License-Identifier: Apache-2.0
// Exact top-face copies for logical B1 BF16 rows, standard32 <-> tiny16 tiles.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t src_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t dst_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);
    constexpr uint32_t workers = get_compile_time_arg_val(3);
    constexpr auto src_a = TensorAccessorArgs<4>();
    constexpr auto dst_a = TensorAccessorArgs<src_a.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_a, get_arg_val<uint32_t>(0), src_bytes);
    const auto dst = TensorAccessor(dst_a, get_arg_val<uint32_t>(1), dst_bytes);
    const uint32_t first = get_arg_val<uint32_t>(2);
    CircularBuffer cb(0);
    Noc noc;
    for (uint32_t tile = first; tile < tiles; tile += workers) {
        cb.reserve_back(1);
        if constexpr (dst_bytes == 2048) {
            noc.async_write_zeros(cb, 2048, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }
        noc.async_read(src, cb, 1024, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
        cb.wait_front(1);
        auto staged = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        noc.async_write(staged, dst, dst_bytes, {.offset_bytes = 0}, {.page_id = tile});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
