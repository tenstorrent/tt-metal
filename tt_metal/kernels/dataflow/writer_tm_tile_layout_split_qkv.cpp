#include <stdint.h>
#include <array>
#include "dataflow_kernel_api.h"

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t q_tensor_addr                       = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr                       = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr                       = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_tile_id                  = get_arg_val<uint32_t>(3);

    // COMPILE TIME ARGS
    // dataflow::Interleaved accessor args
    constexpr uint32_t out_is_dram               = get_compile_time_arg_val(1);
    // WRITER COMPILE TIME ARGS
    constexpr uint32_t out_num_tensors           = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_tiles_per_tensor  = get_compile_time_arg_val(3);


    constexpr uint32_t cb_id_out0 = 0; // same as cb_id_in0
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr bool out_is_dram_bool = out_is_dram == 1;
    #define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
    #if (tile_dtype_is_bfloat16)
    const dataflow::InterleavedAddrGenFast<out_is_dram_bool> sq = {
        .bank_base_address = q_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16
    };
    const dataflow::InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16
    };
    const dataflow::InterleavedAddrGenFast<out_is_dram_bool> sv = {
        .bank_base_address = v_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16
    };
    #else
    const dataflow::InterleavedAddrGenFast<out_is_dram_bool> sq = {
        .bank_base_address = q_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Bfp8_b
    };
    const dataflow::InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Bfp8_b
    };
    const dataflow::InterleavedAddrGenFast<out_is_dram_bool> sv = {
        .bank_base_address = v_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Bfp8_b
    };
    #endif

    std::array<dataflow::InterleavedAddrGenFast<out_is_dram_bool>, out_num_tensors> qkv_output_banks{sq, sk, sv};
    uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);
    uint32_t out_split_tensor_tile_id;
    uint32_t out_num_tiles_read = out_num_tiles_per_tensor;

    for (const auto& s : qkv_output_banks) {
        out_split_tensor_tile_id = out_tensor_tile_id;
        dataflow::cb_wait_front(cb_id_out0, out_num_tiles_read);
        for (uint32_t i = 0; i < out_num_tiles_per_tensor; i++) {
            dataflow::noc_async_write_tile(out_split_tensor_tile_id, s, l1_read_addr);
            l1_read_addr += single_tile_size_bytes;
            out_split_tensor_tile_id++;
        }
        out_num_tiles_read += out_num_tiles_per_tensor;
    }

    dataflow::noc_async_write_barrier();
    dataflow::cb_pop_front(cb_id_out0, out_num_tiles_read);
}
