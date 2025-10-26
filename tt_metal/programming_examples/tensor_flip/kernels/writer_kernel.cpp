#include "dataflow_api.h"

void kernel_main() {
    // compile time args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    // runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile = get_arg_val<uint32_t>(1);
    uint32_t end_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const DataFormat data_format = get_dataformat(cb_id);
    const uint32_t tile_size = get_tile_size(cb_id);

    const InterleavedAddrGenFast<dst_is_dram> d0 = {
        .bank_base_address = dst_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t tile_id = start_tile; tile_id < end_tile; ++tile_id) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write_tile(tile_id, d0, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
