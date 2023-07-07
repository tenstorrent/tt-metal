#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t C = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t HtWt = get_arg_val<uint32_t>(5);
    uint32_t CHtWt = get_arg_val<uint32_t>(6);
    uint32_t NCHtWt = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const dataflow::InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,


        .log_base_2_of_page_size = 11
    };

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i = 0;
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t h = 0; h < Ht; h++) {
                for (uint32_t w = 0; w < Wt; w++) {
                    uint64_t src_noc_addr = dataflow::get_noc_addr(i, s);
                    dataflow::cb_reserve_back(cb_id_in0, onetile);
                    uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id_in0);
                    dataflow::noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
                    dataflow::noc_async_read_barrier();
                    dataflow::cb_push_back(cb_id_in0, onetile);
                    i++;
                }
            }
            i -= HtWt;
            i += CHtWt;
        }
        i -= NCHtWt;
        i+= HtWt;
    }
}
