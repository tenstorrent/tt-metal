#include "dataflow_api.h"

void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t Wt_read = get_arg_val<uint32_t>(5);
    uint32_t Wt_skip = get_arg_val<uint32_t>(6);
    uint32_t NC = get_arg_val<uint32_t>(7);
    uint32_t HtWt = get_arg_val<uint32_t>(8); // HtWt of input tensor

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    const InterleavedPow2AddrGen s = {
        .bank_base_address = dst_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11 // TODO(AP): refactor
    };

    uint32_t tile_id = 0;
    uint32_t i_nc = 0;
    for (uint32_t nc = 0; nc < NC; nc++) {
        tile_id = i_nc + Wt_read;
        for (uint32_t i = 0; i < Ht; i++) {
            for (uint32_t j = 0; j < Wt; j++) {
                uint64_t dst_noc_addr = get_noc_addr(tile_id, s);

                cb_wait_front(cb_id_out0, onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);

                noc_async_write_barrier();

                cb_pop_front(cb_id_out0, onetile);

                tile_id++;
            }
            tile_id += Wt_skip;
        }
        i_nc += HtWt;
    }
}
