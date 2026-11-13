// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, norm core reader (a query head or the key): the 1.0 reduce scaler, the eps tile, a zero tile, the gamma
// row and the eight input tiles (pages first .. first + 7 of the projection).
// Compile-time args: 0 eps bits, then TensorAccessorArgs: x (the projection), gamma.
// Runtime args: 0 x, 1 gamma addresses, 2 first tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t projection_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_tile = get_arg_val<uint32_t>(2);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    constexpr auto projection_args = TensorAccessorArgs<1>();
    constexpr auto gamma_args = TensorAccessorArgs<projection_args.next_compile_time_args_offset()>();
    const auto projection = TensorAccessor(projection_args, projection_addr);
    const auto gamma = TensorAccessor(gamma_args, gamma_addr);
    const Noc noc;

    {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            1.0f);
        generate_bcast_col_scalar(CircularBuffer(cb_eps), eps_bits);
        CircularBuffer(cb_zero).reserve_back(1);
        tile_rows::fill_words(CircularBuffer(cb_zero).get_write_ptr(), tile_bytes / 4, 0);
        CircularBuffer(cb_zero).push_back(1);
    }

    {
        CircularBuffer(cb_gamma).reserve_back(head_tiles);
        CircularBuffer(cb_x).reserve_back(head_tiles);
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc.async_read(
                gamma,
                CircularBuffer(cb_gamma),
                gamma.get_aligned_page_size(),
                {.page_id = c},
                {.offset_bytes = c * tile_bytes});
            noc.async_read(
                projection,
                CircularBuffer(cb_x),
                projection.get_aligned_page_size(),
                {.page_id = first_tile + c},
                {.offset_bytes = c * tile_bytes});
        }
        noc.async_read_barrier();
        CircularBuffer(cb_gamma).push_back(head_tiles);
        CircularBuffer(cb_x).push_back(head_tiles);
    }
}
