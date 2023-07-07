#include "dataflow_kernel_api.h"

#include "debug_print.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    #ifdef KERNEL_COMPILE_TIME_ARG_0
    constexpr bool write_to_dram = get_compile_time_arg_val(0);
    #else
    constexpr bool write_to_dram = true;
    #endif

    #ifdef OUTPUT_DRAM
    const dataflow::InterleavedPow2AddrGen<OUTPUT_DRAM> s = { dst_addr, 11 };
    #else
    const dataflow::InterleavedPow2AddrGen<write_to_dram> s = { dst_addr, 11 };
    #endif

    #if GENERATE_BCAST_SCALER
    constexpr uint32_t blk = BLOCK_SIZE; // needed for correctness of softmax/LN kernels
    #else
    constexpr uint32_t blk = 1; // needed for correctness of kernels processing single tiles
    #endif
    #ifdef TILE_OFFSET
    uint32_t tile_offset = TILE_OFFSET;
    #else
    constexpr uint32_t tile_offset = 0;
    #endif

    for (uint32_t i = 0; i<num_tiles; i += blk) {
        dataflow::cb_wait_front(cb_id_out0, blk);

        for (uint32_t j = 0; j<blk; j++) {
            uint64_t dst_noc_addr = dataflow::get_noc_addr(i+j+tile_offset, s);
            uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0) + (j<<11);
            dataflow::noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
        }
        dataflow::noc_async_write_barrier();
        dataflow::cb_pop_front(cb_id_out0, blk);
    }
}
