// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, reader: the 16 fp32 logits tiles of one 32-row tile row, the reduce scaler tiles the softmax and the
// top-k sum expect, the 16 pre-transposed uint32 index tiles (a constant tensor: row k of tile w holds w*32+k), and
// the two in-place transforms the composed chain does as device ops: the zero fill of the top-k tile's padding
// (fill_implicit_tile_padding) on the values tile and the column-0 broadcast of the denominator (binary_ng's
// col-bcast reader) on the sums tile, each signalled to the compute kernel through a one-entry token CB.  Runtime
// arg 4 (token_mask) names the rows this core produces (the lane form gives each core eight tokens).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_row = get_arg_val<uint32_t>(2);
    const uint32_t rows_in_tile = get_arg_val<uint32_t>(3);
    const uint32_t token_mask =
        get_arg_val<uint32_t>(4);  // bit r: this core produces row r (all ones: one core per tile)

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_max_scaler = get_named_compile_time_arg_val("cb_max_scaler");
    constexpr uint32_t cb_sum_scaler = get_named_compile_time_arg_val("cb_sum_scaler");
    constexpr uint32_t cb_norm_scaler = get_named_compile_time_arg_val("cb_norm_scaler");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t cb_vals = get_named_compile_time_arg_val("cb_vals");
    constexpr uint32_t cb_vals_ready = get_named_compile_time_arg_val("cb_vals_ready");
    constexpr uint32_t cb_sums = get_named_compile_time_arg_val("cb_sums");
    constexpr uint32_t cb_sums_ready = get_named_compile_time_arg_val("cb_sums_ready");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");

    constexpr auto logits_args = TensorAccessorArgs<0, 0>();
    constexpr auto index_args = TensorAccessorArgs<
        logits_args.next_compile_time_args_offset(),
        logits_args.next_common_runtime_args_offset()>();
    const auto logits = TensorAccessor(logits_args, logits_addr);
    const auto index_template = TensorAccessor(index_args, index_addr);

    Noc noc;
    DataflowBuffer in0(cb_in0);
    DataflowBuffer index(cb_index);

    {
        FUSED_ZONE("fz_rt_r_reads");
        const uint32_t tile_bytes = in0.get_entry_size();
        in0.reserve_back(Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            noc.async_read(logits, in0, tile_bytes, {.page_id = tile_row * Wt + w}, {.offset_bytes = w * tile_bytes});
        }
        noc.async_read_barrier();
        in0.push_back(Wt);

        // softmax reader: MAX and SUM scalers (1.0 in row 0 of every face); reduce reader: the SUM scaler 1.0
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_max_scaler,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_sum_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_norm_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);

        index.reserve_back(Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            noc.async_read(index_template, index, tile_bytes, {.page_id = w}, {.offset_bytes = w * tile_bytes});
        }
        noc.async_read_barrier();
        index.push_back(Wt);
    }

    // fill_implicit_tile_padding(scores, 0) in place: columns >= top_k (faces 1 and 3 whole; columns top_k..15 of
    // faces 0 and 2) and rows >= rows_in_tile of the [token, k] tile (and the rows this core does not produce, like
    // padding); the compute kernel pops the tile after its sum
    {
        FUSED_ZONE("fz_rt_r_pad_fill");
        DataflowBuffer vals(cb_vals);
        DataflowBuffer vals_ready(cb_vals_ready);
        vals.wait_front(1);
        {
            volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vals.get_read_ptr());
            for (uint32_t i = 0; i < 256; ++i) {
                tile[256 + i] = 0u;  // face 1: rows 0..15, columns 16..31
                tile[768 + i] = 0u;  // face 3: rows 16..31, columns 16..31
            }
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t face = (row >> 4) * 2;
                const uint32_t base = face * 256 + (row & 15) * 16;
                const uint32_t first_zero = (row < rows_in_tile && ((token_mask >> row) & 1u)) ? top_k : 0u;
                for (uint32_t col = first_zero; col < 16; ++col) {
                    tile[base + col] = 0u;
                }
            }
        }
        vals_ready.reserve_back(1);
        vals_ready.push_back(1);
    }

    // binary_ng col-bcast reader, in place on the sums tile: columns 1..15 of faces 0 and 2 take column 0 (the
    // quotient's columns >= top_k are never read, so faces 1 and 3 stay as the reduce packed them)
    {
        FUSED_ZONE("fz_rt_r_sum_bcast");
        DataflowBuffer sums(cb_sums);
        DataflowBuffer sums_ready(cb_sums_ready);
        sums.wait_front(1);
        {
            volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sums.get_read_ptr());
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t base = (row >> 4) * 2 * 256 + (row & 15) * 16;
                const uint32_t value = tile[base];
                for (uint32_t col = 1; col < 16; ++col) {
                    tile[base + col] = value;
                }
            }
        }
        sums_ready.reserve_back(1);
        sums_ready.push_back(1);
    }
}
