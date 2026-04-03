// Simple reader using InterleavedAddrGen (not Fast) for compatibility
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t tile_start = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGen<true> rd = {.bank_base_address = addr, .page_size = tile_bytes};

    cb_reserve_back(cb_in, num_tiles);
    uint32_t wp = get_write_ptr(cb_in);
    for (uint32_t i = 0; i < num_tiles; i++) {
        uint64_t noc_addr = get_noc_addr(tile_start + i, rd);
        noc_async_read(noc_addr, wp, tile_bytes);
        wp += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, num_tiles);
}
