#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t out_cb = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    DataFormat fmt = static_cast<DataFormat>(get_compile_time_arg_val(0)); //e.g. DataFormat::Float16_b

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(out_cb);
    const uint32_t ublock_size_tiles = 1;

    //BANKED
    const bool IS_DRAM = static_cast<bool>(get_compile_time_arg_val(1));
    InterleavedAddrGenFast<IS_DRAM> dst_addrgen = {
        .bank_base_address = dst_addr,
        .page_size = ublock_size_bytes,
        .data_format = fmt
    };

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = get_noc_addr(i, dst_addrgen);
        cb_wait_front(out_cb, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(out_cb);
        noc_async_write_tile(i,dst_addrgen, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(out_cb, ublock_size_tiles);
        dst_addr = dst_addr;
    }
}
