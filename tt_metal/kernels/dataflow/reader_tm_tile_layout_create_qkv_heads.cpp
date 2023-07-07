#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr                     = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_tile_id                  = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr DataFormat data_format             = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr uint32_t in0_is_dram               = get_compile_time_arg_val(1);
    // READER COMPILE TIME ARGS
    constexpr uint32_t block_size                = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_blocks_per_tensor = get_compile_time_arg_val(3);


    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr bool in0_is_dram_bool = in0_is_dram == 1;
    const dataflow::InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format,
    };

    uint32_t cb_id;
    uint32_t l1_write_addr;
    uint32_t out_num_tensors = 3;
    for (uint32_t out_tensor = 0; out_tensor < out_num_tensors; out_tensor++) {
        // Q or V heads
        if (out_tensor == 0 or out_tensor == 2) {
            cb_id = cb_id_in1;
        }
        // V heads
        else if (out_tensor == 1) {
            cb_id = cb_id_in0;
        }

        l1_write_addr = dataflow::get_write_ptr(cb_id);
        for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
            dataflow::cb_reserve_back(cb_id, block_size);
            for (uint32_t i = 0; i < block_size; i++) {
                dataflow::noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr);
                l1_write_addr += single_tile_size_bytes;
                in0_tensor_tile_id++;
            }
            dataflow::noc_async_read_barrier();
            dataflow::cb_push_back(cb_id, block_size);
        }
    }
}
