#include "dataflow_kernel_api.h"

void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary
    uint32_t num_contiguous_tiles = get_arg_val<uint32_t>(4);
    uint32_t batch_offset = get_arg_val<uint32_t>(5);
    uint32_t num_batch_tiles = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    const dataflow::InterleavedPow2AddrGen<true> s = {
        .bank_base_address = dst_addr,


        .log_base_2_of_page_size = 11 // TODO(AP): refactor
    };

    uint32_t start_tile = 0;
    for(uint32_t i = 0; i < num_batch_tiles; i++) {
        start_tile = i * batch_offset + start_id;
        for(uint32_t j = start_tile; j < start_tile + num_contiguous_tiles; j++) {
            uint64_t dst_noc_addr = dataflow::get_noc_addr(j, s);

            dataflow::cb_wait_front(cb_id_out0, onetile);
            uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);

            dataflow::noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);

            dataflow::noc_async_write_barrier();

            dataflow::cb_pop_front(cb_id_out0, onetile);

        }
    }
}
