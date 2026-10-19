// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;
constexpr uint32_t cb_zero_idx = tt::CBIndex::c_7;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_8;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_9;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_11;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t da_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    [[maybe_unused]] uint32_t dgamma_tile_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    constexpr auto da_args = TensorAccessorArgs<2>();
    constexpr auto dgamma_args = TensorAccessorArgs<da_args.next_compile_time_args_offset()>();
    const auto da_output_addr_generator = TensorAccessor(da_args, da_output_addr);
    [[maybe_unused]] const auto dgamma_output_addr_generator = TensorAccessor(dgamma_args, dgamma_output_addr);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
#ifdef LOCAL_SUM
        // One fp32 tile per row holding the row sum in column 0.
        write_tiles_by_row(cb_dL_da_idx, da_output_addr_generator, r, onetile, tile_bytes, onetile);
#else
        // Write da (gradient w.r.t. input) and dgamma_components (partial gradients w.r.t. gamma) in blocks.
        // We interleave the writes to avoid waiting for the entire Wt tiles at once.
        // NOTE: The final dL_dgamma (gradient w.r.t. gamma) requires reduction across batches, which is performed on
        // the host. Here, we output the per-tile components for host-side reduction.
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t start_idx = (r * Wt) + c;
#ifdef DGAMMA_ON_CHIP
            write_tiles_by_row(cb_dL_da_idx, da_output_addr_generator, start_idx, block_size, tile_bytes, block_size);
#else
            // Write dL_da block
            write_tiles_by_row</* UseBarrier = */ false>(
                cb_dL_da_idx, da_output_addr_generator, start_idx, block_size, tile_bytes, block_size);

            // Write dL_dgamma_components block
            write_tiles_by_row(
                cb_dL_dgamma_components, dgamma_output_addr_generator, start_idx, block_size, tile_bytes, block_size);
            // Barrier called by write_tiles_by_row with UseBarrier=true above
            cb_pop_front(cb_dL_da_idx, block_size);
#endif
        }
#endif
    }
#ifdef DGAMMA_ON_CHIP
    // The compute kernel accumulates this core's dgamma partials over all of its rows and pushes them once.
    write_tiles_by_row(
        cb_dL_dgamma_components,
        dgamma_output_addr_generator,
        dgamma_tile_row * Wt,
        Wt,
        get_tile_size(cb_dL_dgamma_components),
        Wt);
#endif
}
