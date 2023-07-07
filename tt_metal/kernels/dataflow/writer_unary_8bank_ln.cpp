#include "dataflow_kernel_api.h"

#define GENERATE_BCAST_SCALER 1
#define TILE_OFFSET get_arg_val<uint32_t>(4)

#ifndef BLOCK_SIZE // can be already defined via add_define
#error "Block size must be defined"
#endif

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // Index 3 to match with regular writer_unary

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    #define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
    #if (tile_dtype_is_bfloat16)
    DataFormat accessor_data_format = DataFormat::Float16;
    #else
    DataFormat accessor_data_format = DataFormat::Bfp8_b;
    #endif

    const dataflow::InterleavedAddrGenFast<OUTPUT_DRAM> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = accessor_data_format
    };

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

        uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j<blk; j++) {
            dataflow::noc_async_write_tile(i+j+tile_offset, s, l1_read_addr);
            l1_read_addr+=tile_bytes;
        }
        dataflow::noc_async_write_barrier();
        dataflow::cb_pop_front(cb_id_out0, blk);
    }
}
