#include <cstdint>

#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_momentum_out_idx = tt::CBIndex::c_3;
constexpr auto cb_momentum_to_dram_idx = tt::CBIndex::c_4;

constexpr auto cb_param_out_idx = tt::CBIndex::c_16;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

template <typename AddrGen>
inline void write_cb_block_to_dram(
    uint32_t cb_idx, const AddrGen& addr_gen, uint32_t start_idx, uint32_t block_size, uint32_t tile_size_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_write_addr = get_read_ptr(cb_idx);

    for (uint32_t k = 0; k < block_size; ++k) {
        noc_async_write_tile(start_idx + k, addr_gen, l1_write_addr);
        l1_write_addr += tile_size_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t param_out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t momentum_out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_out_idx);
    constexpr auto param_out_args = TensorAccessorArgs<2U>();
    constexpr auto momentum_output_args = TensorAccessorArgs<param_out_args.next_compile_time_args_offset()>();

    const auto param_out_addr_generator = TensorAccessor(param_out_args, param_out_addr, tile_size_bytes);
    const auto momentum_out_addr_generator = TensorAccessor(momentum_output_args, momentum_out_addr, tile_size_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t start_idx = (r * Wt) + c;

            write_cb_block_to_dram(
                cb_momentum_to_dram_idx, momentum_out_addr_generator, start_idx, block_size, tile_size_bytes);
            write_cb_block_to_dram(cb_param_out_idx, param_out_addr_generator, start_idx, block_size, tile_size_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_momentum_to_dram_idx, block_size);
            cb_pop_front(cb_param_out_idx, block_size);
        }
    }
}
