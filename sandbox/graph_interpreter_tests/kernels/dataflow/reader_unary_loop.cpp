#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t num_repetitions = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

    for (uint32_t count = 0; count < num_repetitions; count++) {
        uint32_t src_addr_loop = src_addr;
        // read a ublock of tiles from src to CB, and then push the ublock to unpacker
        for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
            uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr_loop);

            cb_reserve_back(cb_id_in0, ublock_size_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

            noc_async_read_barrier();

            cb_push_back(cb_id_in0, ublock_size_tiles);
            src_addr_loop += ublock_size_bytes;
        }
    }
}
