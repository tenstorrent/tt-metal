#include "dataflow_api.h"

//#include "debug_print.h"

void kernel_main() {
    // same arg indices as in reader_bmm_8bank for reuse
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t Mt         = get_arg_val<uint32_t>(2);
    uint32_t Nt         = get_arg_val<uint32_t>(4);
    uint32_t batch      = get_arg_val<uint32_t>(7);

    constexpr int onetile = 1;
    constexpr uint32_t cb_id_out0 = 16;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);
    uint32_t itileC = 0;

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

    // C is MN so we iterate in tile RM order
    for (uint32_t nb = 0; nb < batch; nb ++)
    for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C)   // output tile of C
    for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) { // output tile index of C
        // bmm will generate C's tiles C=A*B, MN=MK*KN, in row major order, we just read them from CB and write out to DRAM
        uint64_t dst_noc_addr = get_noc_addr(itileC, s); // TODO(AP): refactor
        cb_wait_front(cb_id_out0, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, onetile);
        //DPRINT << 'W' << 'C' << itileC << ' ' << 'a' << dst_addr << ENDL();
        //DPRINT << itileC << ' ' << uint32_t(dst_noc_addr) << ENDL();
        itileC ++;
    }
}
