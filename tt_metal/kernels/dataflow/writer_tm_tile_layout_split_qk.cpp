#include <stdint.h>

#include <array>

#include "dataflow_api.h"

// #define DEBUG

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(0);
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(2);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t out_is_dram = get_compile_time_arg_val(1);
    // WRITER COMPILE TIME ARGS
    constexpr uint32_t out_num_tiles_per_tensor = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    uint32_t single_tile_size_bytes = get_tile_size(out_tensor_tile_id);

    constexpr bool out_is_dram_bool = out_is_dram == 1;
#define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
#if (tile_dtype_is_bfloat16)
    const InterleavedAddrGenFast<out_is_dram_bool> sq = {
        .bank_base_address = q_tensor_addr, .page_size = single_tile_size_bytes, .data_format = DataFormat::Float16};
    const InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr, .page_size = single_tile_size_bytes, .data_format = DataFormat::Float16};
#else
    const InterleavedAddrGenFast<out_is_dram_bool> sq = {
        .bank_base_address = q_tensor_addr, .page_size = single_tile_size_bytes, .data_format = DataFormat::Bfp8_b};
    const InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr, .page_size = single_tile_size_bytes, .data_format = DataFormat::Bfp8_b};
#endif

    std::array<InterleavedAddrGenFast<out_is_dram_bool>, 2> qk_output_banks{sq, sk};
    uint32_t out_split_tensor_tile_id;
    uint32_t out_num_tiles_read = out_num_tiles_per_tensor;

    uint32_t bank_id = 0;
    uint32_t tile_id = 0;
    for (const auto& s : qk_output_banks) {
#ifdef DEBUG
        DPRINT << "Writer CB ID :" << out_tensor_tile_id << ENDL();
        DPRINT << "Writer Start Tensor :" << bank_id << ENDL();
        DPRINT << "Writer Num Tiles: " << out_num_tiles_read << ENDL();
#endif
        for (uint32_t tile_id = 0; tile_id < out_num_tiles_per_tensor; tile_id++) {
            cb_wait_front(cb_id_out0, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            noc_async_write_tile(tile_id + out_tensor_tile_id, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, 1);
#ifdef DEBUG
            DPRINT << "Writer Tile ID: " << tile_id << ENDL();
            DPRINT << "Writer Address: " << l1_read_addr << ENDL();
#endif
        }

        out_num_tiles_read += out_num_tiles_per_tensor;
        bank_id++;
    }

#ifdef DEBUG
    DPRINT << "Writer End " << ENDL();
#endif
}
