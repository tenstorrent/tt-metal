// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_7;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_9;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_bcasted_idx = tt::CBIndex::c_11;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

constexpr uint32_t onetile = 1;

template <typename AddrGen>
inline void write_cb_block_to_dram(
    uint32_t cb_idx, const AddrGen& addr_gen, uint32_t start_idx, uint32_t block_size, uint32_t tile_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_read_addr = get_read_ptr(cb_idx);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        noc_async_write_tile(start_idx + block_idx, addr_gen, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t da_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    constexpr auto da_args = TensorAccessorArgs<2>();
    constexpr auto dgamma_args = TensorAccessorArgs<da_args.next_compile_time_args_offset()>();
    const auto da_output_addr_generator = TensorAccessor(da_args, da_output_addr, tile_bytes);
    const auto dgamma_output_addr_generator = TensorAccessor(dgamma_args, dgamma_output_addr, tile_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write da (gradient w.r.t. input) and dgamma_components (partial gradients w.r.t. gamma) in blocks.
        // We interleave the writes to avoid waiting for the entire Wt tiles at once.
        // NOTE: The final dL_dgamma (gradient w.r.t. gamma) requires reduction across batches, which is performed on
        // the host. Here, we output the per-tile components for host-side reduction.
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t start_idx = (r * Wt) + c;

            // Write dL_da block
            write_cb_block_to_dram(cb_dL_da_idx, da_output_addr_generator, start_idx, block_size, tile_bytes);

            // Write dL_dgamma_components block
            write_cb_block_to_dram(
                cb_dL_dgamma_components, dgamma_output_addr_generator, start_idx, block_size, tile_bytes);
            noc_async_write_barrier();

            cb_pop_front(cb_dL_dgamma_components, block_size);
            cb_pop_front(cb_dL_da_idx, block_size);
        }
    }
}
