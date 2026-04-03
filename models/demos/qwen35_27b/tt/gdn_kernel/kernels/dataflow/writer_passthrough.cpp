// Simple writer: writes N tiles from CB 16 to DRAM
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t tile_start = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 2048;

    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> wr = {
        .bank_base_address = addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    cb_wait_front(cb_out, num_tiles);
    uint32_t rp = get_read_ptr(cb_out);
    for (uint32_t i = 0; i < num_tiles; i++) {
        noc_async_write_tile(tile_start + i, wr, rp);
        rp += tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, num_tiles);
}
