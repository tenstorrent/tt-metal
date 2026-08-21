// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"  // compute_kernel_hw_startup, tile_regs_*, pack_tile
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/reduce_custom.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) reduce_custom == SDPA block MAX-row kernel, classic circular buffers. One LLKOperand per input
// (data, scaler) + output; reduce_custom is format-free so only geometry + addresses flow through. Reduces a
// block of num_tiles data tiles (per-row MAX across the block) into DST[0] via a single
// experimental::reduce_block_max_row call, then packs the reduced tile to c_16. Output must be bit-identical to
// reduce_custom_legacy.cpp. Differs from that kernel ONLY in the reduce_block_max_row init/op/uninit (id-free
// LLKOperand vs CB-id); hw_startup + pack_tile stay legacy CB-id in BOTH. The CB must be >= num_tiles deep --
// the TEST_F passes cb_depth_tiles=num_tiles.
void kernel_main() {
    constexpr std::uint32_t block_ct_dim = get_compile_time_arg_val(0);  // == num_tiles

    constexpr auto cb_data_id = tt::CBIndex::c_0;
    constexpr auto cb_scaler_id = tt::CBIndex::c_1;
    constexpr auto cb_out_id = tt::CBIndex::c_16;

    CircularBuffer cb0(cb_data_id);
    CircularBuffer cb1(cb_scaler_id);
    CircularBuffer cb16(cb_out_id);

    constexpr auto data_cb = experimental::Cb<cb_data_id>{};
    constexpr auto scaler_cb = experimental::Cb<cb_scaler_id>{};
    constexpr auto out_cb = experimental::Cb<cb_out_id>{};
    constexpr auto data_desc = experimental::to_llk_mem_descriptor(data_cb);
    constexpr auto scaler_desc = experimental::to_llk_mem_descriptor(scaler_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using DataOp = experimental::LLKOperand<static_cast<DataFormat>(data_desc.format), data_desc.shape>;
    using ScalerOp = experimental::LLKOperand<static_cast<DataFormat>(scaler_desc.format), scaler_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    // hw_startup + pack stay legacy CB-id in BOTH kernels to isolate reduce_block_max_row.
    compute_kernel_hw_startup(cb_data_id, cb_scaler_id, cb_out_id);
    experimental::reduce_block_max_row_init<block_ct_dim>(
        DataOp(data_cb.read_address()), OutOp(out_cb.write_address()));

    cb0.wait_front(block_ct_dim);
    cb1.wait_front(block_ct_dim);
    cb16.reserve_back(1);

    tile_regs_acquire();
    experimental::reduce_block_max_row<block_ct_dim>(
        DataOp(data_cb.read_address()), ScalerOp(scaler_cb.read_address()), /*row_start_index=*/0, /*idst=*/0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out_id);
    tile_regs_release();

    cb0.pop_front(block_ct_dim);
    cb1.pop_front(block_ct_dim);
    cb16.push_back(1);

    experimental::reduce_block_max_row_uninit();
}
