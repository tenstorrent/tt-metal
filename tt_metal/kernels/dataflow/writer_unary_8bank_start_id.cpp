#include "dataflow_api.h"

void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    #define tile_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (tile_size_is_pow2)
    constexpr uint32_t tile_size_pow2_exponent = get_compile_time_arg_val(1);
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = dst_addr,


        .log_base_2_of_page_size = tile_size_pow2_exponent // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s = {
        .bank_base_address = dst_addr,


        .page_size = tile_bytes
    };
    #endif

    for (uint32_t i = start_id; i<start_id + num_tiles; i ++) {
        uint64_t dst_noc_addr = get_noc_addr(i, s);

        cb_wait_front(cb_id_out0, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, onetile);
    }
}
