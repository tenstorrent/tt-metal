// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, norm core reader (a query head or the key): the 1.0 reduce scaler, the eps tile, a zero tile, the gamma
// row and the eight input tiles (pages first .. first + 7 of the projection).
// Compile-time args: 0 eps bits, then TensorAccessorArgs: x (the projection), gamma.
// Runtime args: 0 x, 1 gamma addresses, 2 first tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "main_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, x_addr);
    const auto gamma = TensorAccessor(gamma_args, gamma_addr);

    {
        FUSED_ZONE("fz_qs_mt_rn_consts");
        dataflow_kernel_lib::prepare_reduce_scaler<CB_SCALER, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            1.0f);
        generate_bcast_col_scalar(CircularBuffer(CB_EPS), eps_bits);
        cb_reserve_back(CB_ZERO, 1);
        tile_rows::fill_words(get_write_ptr(CB_ZERO), TILE_BYTES / 4, 0);
        cb_push_back(CB_ZERO, 1);
    }

    {
        FUSED_ZONE("fz_qs_mt_rn_reads");
        cb_reserve_back(CB_GAMMA, HEAD_TILES);
        cb_reserve_back(CB_X, HEAD_TILES);
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc_async_read_page(c, gamma, get_write_ptr(CB_GAMMA) + c * TILE_BYTES);
            noc_async_read_page(first + c, x, get_write_ptr(CB_X) + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_GAMMA, HEAD_TILES);
        cb_push_back(CB_X, HEAD_TILES);
    }
}
