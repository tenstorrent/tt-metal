// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The forward relay's write kernel. The columns carry no outputs, so this
// RISC has two jobs: the constant tiles the compute kernel takes (once), and
// the finished rows -- at a row's last visit the compute kernel hands over
// the normalised output block in bfloat16 and the log-sum-exp tiles, and
// this kernel stores them. Everything else the relay does is the reader's.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_fw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t first_slice = get_arg_val<uint32_t>(arg++);
    const uint32_t output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t interm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_count = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_stride = get_arg_val<uint32_t>(arg++);
    // The pair table, as in the relay reader: chunks, pairs, heads, kv_slices,
    // q_heads, kv_heads, heads_per_group, then (row chunk, column chunk) per pair.
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t pairs = get_arg_val<uint32_t>(arg++);
    const uint32_t heads = get_arg_val<uint32_t>(arg++);
    const uint32_t kv_slices = get_arg_val<uint32_t>(arg++);
    const uint32_t q_heads = get_arg_val<uint32_t>(arg++);
    const uint32_t kv_heads = get_arg_val<uint32_t>(arg++);
    const uint32_t heads_per_group = get_arg_val<uint32_t>(arg++);
    const uint32_t pair_table_arg = arg;

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t Bt = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr auto output_args = TensorAccessorArgs<4>();
    constexpr auto interm_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    constexpr uint32_t cb_zero_tile = tt::CBIndex::c_8;
    constexpr uint32_t cb_ones_column = tt::CBIndex::c_28;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_27;
    constexpr uint32_t cb_ones_row = tt::CBIndex::c_29;
    // The finished row from the compute kernel.
    constexpr uint32_t cb_output = tt::CBIndex::c_21;
    constexpr uint32_t cb_lse = tt::CBIndex::c_22;

    using namespace ttml::metal::ops::cyclic_sdpa_bw;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

    const uint32_t out_bytes = get_tile_size(cb_output);
    const uint32_t interm_bytes = get_tile_size(cb_lse);
    const auto output = TensorAccessor(output_args, output_addr, out_bytes);
    const auto intermediates = TensorAccessor(interm_args, interm_addr, interm_bytes);

    // The constants: the transposed causal mask (a triangle and an all -inf
    // tile), the zero tile the mask is added against (and the compute
    // kernel's transpose fence page), the column of ones that masks the lse
    // to column 0, and the all-ones scaler of the column reductions.
    {
        const uint32_t zero_l1 = get_write_ptr(cb_zero_tile);
        cyclic_dataflow::zero_tile(zero_l1, get_tile_size(cb_zero_tile));
        cyclic_dataflow::generate_causal_mask_tiles(cb_attn_mask);
        cyclic_dataflow::generate_ones_column_tile(cb_ones_column);
        cyclic_dataflow::generate_ones_row_tile(cb_ones_row);
        cb_reserve_back(cb_reduce_scaler, 1);
        cyclic_dataflow::fill_constant_bf16_tile(get_write_ptr(cb_reduce_scaler), cyclic_dataflow::kBf16One);
        cb_push_back(cb_reduce_scaler, 1);
    }

    for (uint32_t s = 0; s < slice_count; ++s) {
        const uint32_t sl = first_slice + s * slice_stride;
        const uint32_t idx = sl % heads;
        const uint32_t pair = sl / heads;
        const uint32_t bg = idx % kv_slices;
        const uint32_t sub = idx / kv_slices;
        const uint32_t bh = (bg / kv_heads) * q_heads + (bg % kv_heads) * heads_per_group + sub;
        const uint32_t row_chunk = get_arg_val<uint32_t>(pair_table_arg + 2u * pair);
        const uint32_t row_base = (bh * chunks + row_chunk) * 2u * kCores * row_tiles;
        const uint32_t stat_base = (bh * chunks + row_chunk) * 2u * kCores * Bt;

        for (uint32_t t = 0; t < kTimesteps; ++t) {
            const auto pair_t = sched.pair(my_core, t);
            const uint32_t i = pair_t.i;
            // The row's last visit: no next consumer and no later streak.
            const bool final = sched.next_consumer(i, t) == kNoCore && !sched.has_later_active(i, t);
            if (!final) {
                continue;
            }
            cb_wait_front(cb_output, row_tiles);
            cb_wait_front(cb_lse, Bt);
            const uint32_t out_l1 = get_read_ptr(cb_output);
            const uint32_t lse_l1 = get_read_ptr(cb_lse);
            for (uint32_t k = 0; k < row_tiles; ++k) {
                noc_async_write_page(row_base + (i - 1u) * row_tiles + k, output, out_l1 + k * out_bytes);
            }
            for (uint32_t k = 0; k < Bt; ++k) {
                noc_async_write_page(stat_base + (i - 1u) * Bt + k, intermediates, lse_l1 + k * interm_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output, row_tiles);
            cb_pop_front(cb_lse, Bt);
        }
    }
}
