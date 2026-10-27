// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/rmsnorm_bw_kernel_args.hpp"

namespace kernel_args = ttml::metal::ops::rmsnorm_bw::device::kernel_args;

// CBs with output data
constexpr uint32_t cb_dL_da_idx = kernel_args::cb::kDLda;
constexpr uint32_t cb_dL_dgamma_components = kernel_args::cb::kDgammaComponents;
constexpr uint32_t cb_dgamma_row_idx = kernel_args::cb::kDgammaRow;

constexpr uint32_t block_size = get_compile_time_arg_val(kernel_args::writer_ct::kBlockSize);
constexpr uint32_t Wt = get_compile_time_arg_val(kernel_args::writer_ct::kWt);

#ifdef DGAMMA_ON_CHIP
// Writes row 0 of each fp32 tile in cb_dgamma_row into row `dgamma_row` of the [num_cores, C] dgamma partials.
// A tile row spans two 16x16 faces: faces 0 and 1 hold tile rows 0..15, faces 2 and 3 rows 16..31.
template <typename AddrGen>
void write_dgamma_row(const AddrGen& dgamma_addr_generator, const uint32_t dgamma_row) {
    constexpr uint32_t tile_height = 32U;
    constexpr uint32_t face_height = 16U;
    constexpr uint32_t face_row_bytes = face_height * sizeof(float);
    constexpr uint32_t face_bytes = face_height * face_row_bytes;
    const uint32_t row_in_tile = dgamma_row % tile_height;
    const uint32_t dst_offset =
        (row_in_tile / face_height) * 2U * face_bytes + (row_in_tile % face_height) * face_row_bytes;
    const uint32_t first_tile = (dgamma_row / tile_height) * Wt;
    for (uint32_t c = 0; c < Wt; ++c) {
        cb_wait_front(cb_dgamma_row_idx, onetile);
        const uint32_t l1_read_addr = get_read_ptr(cb_dgamma_row_idx);
        for (uint32_t face = 0; face < 2U; ++face) {
            noc_async_write(
                l1_read_addr + face * face_bytes,
                dgamma_addr_generator.get_noc_addr(first_tile + c, dst_offset + face * face_bytes),
                face_row_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_dgamma_row_idx, onetile);
    }
    noc_async_write_barrier();
}
#endif

void kernel_main() {
    const uint32_t da_output_addr = get_arg_val<uint32_t>(kernel_args::writer_rt::kDLdaAddr);
    const uint32_t dgamma_output_addr = get_arg_val<uint32_t>(kernel_args::writer_rt::kDgammaAddr);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(kernel_args::writer_rt::kNumRows);
    const uint32_t start_row = get_arg_val<uint32_t>(kernel_args::writer_rt::kStartRow);
    [[maybe_unused]] const uint32_t dgamma_row = get_arg_val<uint32_t>(kernel_args::writer_rt::kDgammaRow);

    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    constexpr auto da_args = TensorAccessorArgs<kernel_args::writer_ct::kNumArgs>();
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
    // The compute kernel accumulates this core's dgamma over all of its rows and sums each tile's rows into one.
    write_dgamma_row(dgamma_output_addr_generator, dgamma_row);
#endif
}
