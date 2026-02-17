#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_hw = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_in0 = 0;
    const uint32_t page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Load all input tiles at once — compute reads 3x without popping
    for (uint32_t i = 0; i < block_hw; ++i) {
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);
        noc_async_read_page(start_id + i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
