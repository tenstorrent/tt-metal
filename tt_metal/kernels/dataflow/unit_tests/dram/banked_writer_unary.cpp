#include <cstdint>

#include "dataflow_kernel_api.h"

void kernel_main() {
    constexpr std::uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);
    std::uint32_t dst_addr_base = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr bool IS_DRAM = false;
    const uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = get_tile_size(cb_id);
    dataflow::InterleavedAddrGen<IS_DRAM> dst_addrgen = {
        .bank_base_address = dst_addr_base,
        .page_size = page_size,
    };

    // Write tiles from CB to dram(interleaved)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = dataflow::get_noc_addr(i, dst_addrgen);

        dataflow::cb_wait_front(cb_id, ublock_size_tiles);
        uint32_t l1_read_ptr = dataflow::get_read_ptr(cb_id);
        dataflow::noc_async_write(l1_read_ptr, dst_noc_addr, tile_bytes);

        dataflow::noc_async_write_barrier();

        dataflow::cb_pop_front(cb_id, ublock_size_tiles);
    }
}
