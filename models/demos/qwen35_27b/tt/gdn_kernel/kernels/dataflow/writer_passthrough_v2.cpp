// Simple writer using InterleavedAddrGen (not Fast)
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t tile_start = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGen<true> wr = {.bank_base_address = addr, .page_size = tile_bytes};

    cb_wait_front(cb_out, num_tiles);
    uint32_t rp = get_read_ptr(cb_out);
    for (uint32_t i = 0; i < num_tiles; i++) {
        uint64_t noc_addr = get_noc_addr(tile_start + i, wr);
        noc_async_write(noc_addr, rp, tile_bytes);
        rp += tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, num_tiles);
}
