#include "dataflow_kernel_api.h"

void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    const dataflow::InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i<end_id; i ++) {
        dataflow::cb_wait_front(cb_id_out0, onetile);
        uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);

        dataflow::noc_async_write_tile(i, s, l1_read_addr);

        dataflow::noc_async_write_barrier();

        dataflow::cb_pop_front(cb_id_out0, onetile);
    }
}
